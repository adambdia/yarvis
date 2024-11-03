// kinect_bridge.cpp
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>
#include <iostream>

namespace py = pybind11;

template <typename T>
py::array_t<T> copy_array(const py::array_t<T> &input)
{
    // Request a buffer descriptor from Python
    py::buffer_info buf = input.request();

    // Create a new array with same shape and data type
    py::array_t<T> result(buf.shape);

    // Get pointers to input and output buffers
    auto result_buf = result.request();
    T *ptr_in = static_cast<T *>(buf.ptr);
    T *ptr_out = static_cast<T *>(result_buf.ptr);

    // Calculate total size
    size_t total_size = 1;
    for (size_t i = 0; i < buf.shape.size(); i++)
    {
        total_size *= buf.shape[i];
    }

    // Copy the data
    std::memcpy(ptr_out, ptr_in, sizeof(T) * total_size);

    return result;
}

class KinectBridge
{
private:
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = nullptr;
    libfreenect2::SyncMultiFrameListener *listener = nullptr;
    libfreenect2::FrameMap frames;

public:
    KinectBridge()
    {
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
            cleanup();
            throw std::runtime_error(std::string("KinectBridge initialization failed: ") + e.what());
        }
    }

    ~KinectBridge()
    {
        cleanup();
    }

    void cleanup()
    {
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
        try
        {
            if (!listener)
            {
                throw std::runtime_error("Device not initialized!");
            }

            std::cout << "Waiting for new frame..." << std::endl;
            if (!listener->waitForNewFrame(frames, 10 * 1000))
            {
                throw std::runtime_error("Timeout waiting for frames!");
            }

            std::cout << "Got new frame, processing..." << std::endl;
            libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
            libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

            if (!rgb || !depth)
            {
                listener->release(frames);
                throw std::runtime_error("Failed to get valid frames!");
            }

            // Create numpy arrays directly from the frame data
            py::array_t<uint8_t> rgb_array(
                py::buffer_info(
                    static_cast<uint8_t *>(rgb->data),        // Pointer to data
                    sizeof(uint8_t),                          // Size of one scalar
                    py::format_descriptor<uint8_t>::format(), // Python struct-style format descriptor
                    3,                                        // Number of dimensions
                    {rgb->height, rgb->width, 4},             // Shape
                    {rgb->width * 4,                          // Strides (in elements)
                     4,
                     1}));

            py::array_t<float> depth_array({depth->height, depth->width},
                                           {depth->width * sizeof(float), sizeof(float)},
                                           reinterpret_cast<float *>(depth->data));

            // Create copies of the data since we'll release the frames
            py::array_t<uint8_t> rgb_copy = copy_array(rgb_array);
            py::array_t<float> depth_copy = copy_array(depth_array);

            // Release the frames
            listener->release(frames);

            // Return the copied arrays
            py::dict result;
            result["rgb"] = rgb_copy;
            result["depth"] = depth_copy;
            return result;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error in getFrames: " << e.what() << std::endl;
            if (frames.size() > 0)
            {
                listener->release(frames);
            }
            throw;
        }
    }
};

PYBIND11_MODULE(kinect_bridge, m)
{
    py::class_<KinectBridge>(m, "KinectBridge")
        .def(py::init<>())
        .def("get_frames", &KinectBridge::getFrames);
}