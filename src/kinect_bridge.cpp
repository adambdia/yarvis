// kinect_bridge.cpp
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>

namespace py = pybind11;

// Custom holder for frame data
class FrameData {
public:
    std::vector<uint8_t> rgb_data;
    std::vector<float> depth_data;
    std::vector<uint8_t> bgr_data;
    int width;
    int height;
    
    FrameData(libfreenect2::Frame* rgb, libfreenect2::Frame* depth) {
        width = rgb->width;
        height = rgb->height;
        
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
        cv::Mat bgr;
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
        bgr_data.resize(height * width * 3);
        std::memcpy(bgr_data.data(), bgr.data, bgr_data.size());
    }
};

class KinectBridge {
private:
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = nullptr;
    libfreenect2::PacketPipeline *pipeline = nullptr;
    libfreenect2::SyncMultiFrameListener listener;
    libfreenect2::FrameMap frames;
    std::shared_ptr<FrameData> current_frame;
    
public:
    KinectBridge() : listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth | libfreenect2::Frame::Ir) {
        if(freenect2.enumerateDevices() == 0) {
            throw std::runtime_error("No Kinect devices found!");
        }
        
        pipeline = new libfreenect2::OpenGLPacketPipeline();
        dev = freenect2.openDevice(0, pipeline);
        
        dev->setColorFrameListener(&listener);
        dev->setIrAndDepthFrameListener(&listener);
        dev->start();
    }
    
    ~KinectBridge() {
        if(dev) {
            dev->stop();
            dev->close();
        }
        delete pipeline;
    }
    
    py::dict getFrames() {
        if(!listener.waitForNewFrame(frames, 10*1000)) {
            throw std::runtime_error("Timeout waiting for frames!");
        }
        
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
        
        // Create new frame data
        current_frame = std::make_shared<FrameData>(rgb, depth);
        
        // Now we can release the original frames
        listener.release(frames);
        
        // Create numpy arrays that view our copied data
        py::array_t<uint8_t> rgb_array({current_frame->height, 
                                       current_frame->width, 
                                       4},
                                      {current_frame->width*4, 
                                       4, 
                                       1},
                                      current_frame->rgb_data.data(),
                                      py::cast(current_frame));  // Keep frame data alive
        
        py::array_t<float> depth_array({current_frame->height, 
                                       current_frame->width},
                                      {current_frame->width*sizeof(float), 
                                       sizeof(float)},
                                      current_frame->depth_data.data(),
                                      py::cast(current_frame));  // Keep frame data alive
        
        py::array_t<uint8_t> bgr_array({current_frame->height, 
                                       current_frame->width, 
                                       3},
                                      {current_frame->width*3, 
                                       3, 
                                       1},
                                      current_frame->bgr_data.data(),
                                      py::cast(current_frame));  // Keep frame data alive
        
        py::dict result;
        result["rgb"] = rgb_array;
        result["depth"] = depth_array;
        result["bgr"] = bgr_array;
        return result;
    }
};

PYBIND11_MODULE(kinect_bridge, m) {
    py::class_<KinectBridge>(m, "KinectBridge")
        .def(py::init<>())
        .def("get_frames", &KinectBridge::getFrames);
}