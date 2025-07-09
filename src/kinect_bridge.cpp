// kinect_bridge.cpp
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>

namespace py = pybind11;

// template <typename T>
// py::array_t<T> copy_array(const py::array_t<T> &input)
// {
//     // Request a buffer descriptor from Python
//     py::buffer_info buf = input.request();

//     // Create a new array with same shape and data type
//     py::array_t<T> result(buf.shape);

//     // Get pointers to input and output buffers
//     auto result_buf = result.request();
//     T *ptr_in = static_cast<T *>(buf.ptr);
//     T *ptr_out = static_cast<T *>(result_buf.ptr);

//     // Calculate total size
//     size_t total_size = 1;
//     for (size_t i = 0; i < buf.shape.size(); i++)
//     {
//         total_size *= buf.shape[i];
//     }

//     // Copy the data
//     std::memcpy(ptr_out, ptr_in, sizeof(T) * total_size);

//     return result;
// }

class KinectBridge
{
private:
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = nullptr;
    libfreenect2::SyncMultiFrameListener *listener = nullptr;
    // libfreenect2::FrameMap frames;

    std::thread acquisition_thread;
    std::mutex frame_mutex;
    std::atomic<bool> running;

    cv::Mat ir_mat;
    cv::Mat depth_mat;
    cv::Mat bgr_mat;

    // libfreenect2::Frame ir_mat;
    // libfreenect2::Frame depth_mat;
    // libfreenect2::Frame bgr_mat;

    void acquisitionLoop()
    {
        libfreenect2::FrameMap frames;
        while (running)
        {
            if (!listener->waitForNewFrame(frames, 10 * 1000))
            {
                std::cout << "Timeout waiting for frames!" << std::endl;
                continue;
            }

            libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
            libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
            libfreenect2::Frame *bgr = frames[libfreenect2::Frame::Color];

            if (depth && ir && bgr)
            {
                // Lock the mutex to safely access shared data
                std::lock_guard<std::mutex> lock(frame_mutex);
                cv::Mat(ir->height, ir->width, CV_32FC1, ir->data).copyTo(ir_mat);
                cv::Mat(depth->height, depth->width, CV_32FC1, depth->data).copyTo(depth_mat);
                cv::Mat bgra_mat(bgr->height, bgr->width, CV_8UC4, bgr->data);
                cv::cvtColor(bgra_mat, bgr_mat, cv::COLOR_BGRA2BGR);

            }

            listener->release(frames);
        }
        std::cout << "Acquisition loop stopped." << std::endl;
    }


public:
    KinectBridge()
    {
        libfreenect2::setGlobalLogger(NULL);
        try
        {
            if (freenect2.enumerateDevices() == 0)
            {
                throw std::runtime_error("No Kinect devices found!");
            }

            std::cout << "Opening default device..." << std::endl;
            dev = freenect2.openDefaultDevice();
            if (!dev)
            {
                throw std::runtime_error("Failed to open Kinect device!");
            }

            std::cout << "Creating listener..." << std::endl;
            listener = new libfreenect2::SyncMultiFrameListener(
                libfreenect2::Frame::Color |
                libfreenect2::Frame::Depth |
                libfreenect2::Frame::Ir);

            dev->setColorFrameListener(listener);
            dev->setIrAndDepthFrameListener(listener);

            std::cout << "Starting device..." << std::endl;
            if (!dev->start())
            {
                throw std::runtime_error("Failed to start Kinect device!");
            }
            std::cout << "Device started successfully" << std::endl;
        }
        catch (const std::exception &e)
        {
            stop();
            throw std::runtime_error(std::string("KinectBridge initialization failed: ") + e.what());
        }

        ir_mat = cv::Mat(424, 512, CV_32FC1);
        depth_mat = cv::Mat(424, 512, CV_32FC1);
        bgr_mat = cv::Mat();

        running = true;
        acquisition_thread = std::thread(&KinectBridge::acquisitionLoop, this);
    }

    ~KinectBridge()
    {
        stop();
    }

    void stop()
    {
        running = false;
        if (acquisition_thread.joinable())
        {
            acquisition_thread.join();
        }
        
        if (dev)
        {
            dev->stop();
            dev->close();
            dev = nullptr;
        }

        if (listener)
        {
            delete listener;
            listener = nullptr;
        }
    }

    py::dict getFrames()
    {
        // try
        // {
        //     if (!listener)
        //     {
        //         throw std::runtime_error("Device not initialized!");
        //     }

        //     //std::cout << "Waiting for new frame..." << std::endl;
        //     if (!listener->waitForNewFrame(frames, 10 * 1000))
        //     {
        //         throw std::runtime_error("Timeout waiting for frames!");
        //     }

        //     //std::cout << "Got new frame, processing..." << std::endl;
        //     libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        //     libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
        //     libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];

        //     if (!rgb || !depth || !ir)
        //     {
        //         listener->release(frames);
        //         throw std::runtime_error("Failed to get valid frames!");
        //     }

        //     // // Create numpy arrays directly from the frame data
        //     // py::array_t<uint8_t> rgb_array(
        //     //     py::buffer_info(
        //     //         static_cast<uint8_t *>(rgb->data),        // Pointer to data
        //     //         sizeof(uint8_t),                          // Size of one scalar
        //     //         py::format_descriptor<uint8_t>::format(), // Python struct-style format descriptor
        //     //         3,                                        // Number of dimensions
        //     //         {rgb->height, rgb->width, 4},             // Shape
        //     //         {rgb->width * 4,                          // Strides (in elements)
        //     //          4,
        //     //          1}));

        //     py::array_t<float> depth_array({depth->height, depth->width},
        //                                    {depth->width * sizeof(float), sizeof(float)},
        //                                    reinterpret_cast<float*>(depth->data));
            
        //     py::array_t<float> ir_array({ir->height, ir->width}, 
        //                                   {ir->width * sizeof(float), sizeof(float)},
        //                                   reinterpret_cast<float*>(ir->data));

        //     // Create copies of the data since we'll release the frames
        //     // py::array_t<uint8_t> rgb_copy = copy_array(rgb_array);
        //     py::array_t<float> depth_copy = copy_array(depth_array);
        //     py::array_t<float> ir_copy = copy_array(ir_array);

        //     // Release the frames
        //     listener->release(frames);

        //     // Return the copied arrays
        //     py::dict result;
        //     // result["rgb"] = rgb_copy;
        //     result["depth"] = depth_copy;
        //     result["ir"] = ir_copy;
        //     return result;
        // }
        // catch (const std::exception &e)
        // {
        //     std::cerr << "Error in getFrames: " << e.what() << std::endl;
        //     if (frames.size() > 0)
        //     {
        //         listener->release(frames);
        //     }
        //     throw;
        // }

        std::lock_guard<std::mutex> lock(frame_mutex);

        py::array_t<float> depth_array = py::array_t<float>(
            {depth_mat.rows, depth_mat.cols},
            {depth_mat.cols * sizeof(float), sizeof(float)},
            depth_mat.ptr<float>()
        );

        py::array_t<float> ir_array = py::array_t<float>(
            {ir_mat.rows, ir_mat.cols}, 
            {ir_mat.cols * sizeof(float), sizeof(float)},
            ir_mat.ptr<float>()
        );

        py::array_t<uint8_t> bgr_array = py::array_t<uint8_t>(
            {bgr_mat.rows, bgr_mat.cols, 3},
            {bgr_mat.cols * 3 * sizeof(uint8_t), 3 * sizeof(uint8_t), sizeof(uint8_t)},
            bgr_mat.ptr<uint8_t>()
        );

        py::dict result;
        result["depth"] = depth_array.attr("copy")();
        result["ir"] = ir_array.attr("copy")();
        result["bgr"] = bgr_array.attr("copy")();
        return result;

    }
};

PYBIND11_MODULE(kinect_bridge, m)
{
    py::class_<KinectBridge>(m, "KinectBridge")
        .def(py::init<>())
        .def("get_frames", &KinectBridge::getFrames)
        .def("stop", &KinectBridge::stop);
}