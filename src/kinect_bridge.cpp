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

class FrameData {
public:
    std::vector<uint8_t> rgb_data;
    std::vector<float> depth_data;
    std::vector<uint8_t> bgr_data;
    int width;
    int height;
    
    FrameData(libfreenect2::Frame* rgb, libfreenect2::Frame* depth) {
        if (!rgb || !depth) {
            throw std::runtime_error("FrameData: Null frame pointer received!");
        }
        
        width = rgb->width;
        height = rgb->height;
        
        try {
            // Copy RGB data
            size_t rgb_size = width * height * 4;
            rgb_data.resize(rgb_size);
            std::memcpy(rgb_data.data(), rgb->data, rgb_size);
            
            // Copy depth data
            size_t depth_size = depth->width * depth->height * sizeof(float);
            depth_data.resize(depth->width * depth->height);
            std::memcpy(depth_data.data(), depth->data, depth_size);
            
            // Create BGR data
            cv::Mat rgba(height, width, CV_8UC4, rgb_data.data());
            cv::Mat bgr(height, width, CV_8UC3);
            cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
            bgr_data.resize(height * width * 3);
            std::memcpy(bgr_data.data(), bgr.data, bgr_data.size());
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("FrameData: Error processing frames: ") + e.what());
        }
    }
};

class KinectBridge {
private:
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device* dev = nullptr;
    libfreenect2::SyncMultiFrameListener* listener = nullptr;
    libfreenect2::FrameMap frames;
    
public:
    KinectBridge() {
        try {
            if(freenect2.enumerateDevices() == 0) {
                throw std::runtime_error("No Kinect devices found!");
            }
            
            std::cout << "Opening default device..." << std::endl;
            dev = freenect2.openDefaultDevice();
            if (!dev) {
                throw std::runtime_error("Failed to open Kinect device!");
            }
            
            std::cout << "Creating listener..." << std::endl;
            listener = new libfreenect2::SyncMultiFrameListener(
                libfreenect2::Frame::Color | 
                libfreenect2::Frame::Depth | 
                libfreenect2::Frame::Ir
            );
            
            dev->setColorFrameListener(listener);
            dev->setIrAndDepthFrameListener(listener);
            
            std::cout << "Starting device..." << std::endl;
            if (!dev->start()) {
                throw std::runtime_error("Failed to start Kinect device!");
            }
            std::cout << "Device started successfully" << std::endl;
            
        } catch (const std::exception& e) {
            cleanup();
            throw std::runtime_error(std::string("KinectBridge initialization failed: ") + e.what());
        }
    }
    
    ~KinectBridge() {
        cleanup();
    }
    
    void cleanup() {
        if(dev) {
            dev->stop();
            dev->close();
            dev = nullptr;
        }
        if(listener) {
            delete listener;
            listener = nullptr;
        }
    }
    
    py::dict getFrames() {
        try {
            if (!listener) {
                throw std::runtime_error("Device not initialized!");
            }
            
            std::cout << "Waiting for new frame..." << std::endl;
            if(!listener->waitForNewFrame(frames, 10*1000)) {
                throw std::runtime_error("Timeout waiting for frames!");
            }
            
            std::cout << "Got new frame, processing..." << std::endl;
            libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
            libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
            
            if (!rgb || !depth) {
                listener->release(frames);
                throw std::runtime_error("Failed to get valid frames!");
            }
            
            // Create numpy arrays directly from the frame data
            py::array_t<uint8_t> rgb_array({rgb->height, rgb->width, 4},
                                         {rgb->width*4, 4, 1},
                                         static_cast<uint8_t*>(rgb->data));
            
            py::array_t<float> depth_array({depth->height, depth->width},
                                         {depth->width*sizeof(float), sizeof(float)},
                                         static_cast<float*>(depth->data));
            
            // Convert RGB to BGR
            cv::Mat rgba_mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
            cv::Mat bgr_mat;
            cv::cvtColor(rgba_mat, bgr_mat, cv::COLOR_RGBA2BGR);
            
            // Create BGR array
            py::array_t<uint8_t> bgr_array({rgb->height, rgb->width, 3},
                                         {rgb->width*3, 3, 1},
                                         bgr_mat.data);
            
            // Create copies of the data since we'll release the frames
            py::array_t<uint8_t> rgb_copy = rgb_array.copy();
            py::array_t<float> depth_copy = depth_array.copy();
            py::array_t<uint8_t> bgr_copy = bgr_array.copy();
            
            // Release the frames
            listener->release(frames);
            
            // Return the copied arrays
            py::dict result;
            result["rgb"] = rgb_copy;
            result["depth"] = depth_copy;
            result["bgr"] = bgr_copy;
            return result;
            
        } catch (const std::exception& e) {
            std::cerr << "Error in getFrames: " << e.what() << std::endl;
            if (frames.size() > 0) {
                listener->release(frames);
            }
            throw;
        }
    }
};

PYBIND11_MODULE(kinect_bridge, m) {
    py::class_<KinectBridge>(m, "KinectBridge")
        .def(py::init<>())
        .def("get_frames", &KinectBridge::getFrames);
}