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

class KinectBridge
{
private:
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = nullptr;
    libfreenect2::SyncMultiFrameListener *listener = nullptr;
    // libfreenect2::FrameMap frames;

    libfreenect2::Registration* registration = nullptr;
    libfreenect2::Frame* undistorted = nullptr;
    libfreenect2::Frame* registered = nullptr;

    std::thread acquisition_thread;
    std::mutex frame_mutex;
    std::atomic<bool> running;

    cv::Mat ir_mat;
    cv::Mat depth_mat;
    cv::Mat bgr_mat;
    cv::Mat registered_mat;

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
                std::cout << "[DEBUG] Timeout waiting for frames!" << std::endl;
                continue;
            }

            libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
            libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
            libfreenect2::Frame *bgr = frames[libfreenect2::Frame::Color];

            if (depth && ir && bgr)
            {
                registration->apply(bgr, depth, undistorted, registered);
                // Lock the mutex to safely access shared data
                std::lock_guard<std::mutex> lock(frame_mutex);
                cv::Mat(ir->height, ir->width, CV_32FC1, ir->data).copyTo(ir_mat);
                cv::Mat(depth->height, depth->width, CV_32FC1, depth->data).copyTo(depth_mat);
                cv::Mat bgra_mat(bgr->height, bgr->width, CV_8UC4, bgr->data);
                cv::cvtColor(bgra_mat, bgr_mat, cv::COLOR_BGRA2BGR);

                cv::Mat registered_bgra(registered->height, registered->width, CV_8UC4, registered->data);
                cv::cvtColor(registered_bgra, registered_mat, cv::COLOR_BGRA2BGR);

            }

            listener->release(frames);
        }
        std::cout << "[DEBUG] Acquisition loop stopped." << std::endl;
    }


public:
    KinectBridge()
    {
        libfreenect2::setGlobalLogger(NULL);
        try
        {
            if (freenect2.enumerateDevices() == 0)
            {
                throw std::runtime_error("[DEBUG] No Kinect devices found!");
            }

            std::cout << "[DEBUG] Opening default device..." << std::endl;
            dev = freenect2.openDefaultDevice();
            if (!dev)
            {
                throw std::runtime_error("[DEBUG] Failed to open Kinect device!");
            }

            std::cout << "[DEBUG] Creating listener..." << std::endl;
            listener = new libfreenect2::SyncMultiFrameListener(
                libfreenect2::Frame::Color |
                libfreenect2::Frame::Depth |
                libfreenect2::Frame::Ir);

            dev->setColorFrameListener(listener);
            dev->setIrAndDepthFrameListener(listener);

            std::cout << "[DEBUG] Starting device..." << std::endl;
            if (!dev->start())
            {
                throw std::runtime_error("[DEBUG] Failed to start Kinect device!");
            }
            std::cout << "[DEBUG] Device started successfully" << std::endl;
        }
        catch (const std::exception &e)
        {
            stop();
            throw std::runtime_error(std::string("[DEBUG] KinectBridge initialization failed: ") + e.what());
        }

        registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
        undistorted = new libfreenect2::Frame(512, 424, 4);
        registered = new libfreenect2::Frame(512, 424, 4);
        
        ir_mat = cv::Mat(424, 512, CV_32FC1);
        depth_mat = cv::Mat(424, 512, CV_32FC1);
        bgr_mat = cv::Mat();
        registered_mat = cv::Mat();

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

        if (registration)
        {
            delete registration;
            registration = nullptr;
        }
        
        if (undistorted)
        {
            delete undistorted;
            undistorted = nullptr;
        }

        if (registered)
        {
            delete registered;
            registered = nullptr;
        }
    }

    py::tuple getPointXYZRGB(int x, int y)
    {
        std::lock_guard<std::mutex> lock(frame_mutex);

        float xt, yt, zt, rgbt;
        registration->getPointXYZRGB(undistorted, registered, y, x, xt, yt, zt, rgbt);
        const uint8_t *p = reinterpret_cast<uint8_t*>(&rgbt);
        uint8_t b = p[0];
        uint8_t g = p[1];
        uint8_t r = p[2];
        return py::make_tuple(xt, yt, zt, r, g, b);
    }

    py::tuple getPointXYZ(int x, int y)
    {
        std::lock_guard<std::mutex> lock(frame_mutex);

        float xt, yt, zt;
        registration->getPointXYZ(undistorted, y, x, xt, yt, zt);
        return py::make_tuple(xt, yt, zt);
    }

    py::dict getFrames()
    {
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

        py::array_t<uint8_t> registered_array({registered_mat.rows, registered_mat.cols, 3}, registered_mat.ptr<uint8_t>());

        py::dict result;
        result["depth"] = depth_array.attr("copy")();
        result["ir"] = ir_array.attr("copy")();
        result["bgr"] = bgr_array.attr("copy")();
        result["registered"] = registered_array.attr("copy")();
        return result;

    }
};

PYBIND11_MODULE(kinect_bridge, m)
{
    py::class_<KinectBridge>(m, "KinectBridge")
        .def(py::init<>())
        .def("get_frames", &KinectBridge::getFrames)
        .def("stop", &KinectBridge::stop)
        .def("get_point_xyzrgb", &KinectBridge::getPointXYZRGB, py::arg("x"), py::arg("y"))
        .def("get_point_xyz", &KinectBridge::getPointXYZ, py::arg("x"), py::arg("y"));
}