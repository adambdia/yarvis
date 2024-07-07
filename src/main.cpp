#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ostream>
#include <signal.h>


bool protonect_shutdown = false; // Whether the running application should shut down.
int sliderMax = 255;
int sliderVal;

typedef void(*frameCallback)(cv::Mat, void*);

void sigint_handler(int s)
{
  protonect_shutdown = true;
}

void frameHandler(cv::Mat receivedFrame, void* data)
{
    cv::Mat segmented;
    double max = 255.0;
    double threshold = (double)*((int*)data);

    std::cout << "hello" << std::endl;
    cv::normalize(receivedFrame, segmented, 0, 255, cv::NORM_MINMAX);
    segmented.convertTo(segmented, CV_8UC1);
    cv::threshold(segmented, segmented, threshold, max, cv::THRESH_TOZERO);
    cv::imshow("yarvis", segmented);
}

int main()
{
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = 0;
    libfreenect2::PacketPipeline *pipeline = 0;
    std::cout << "hello" << std::endl;
    if (freenect2.enumerateDevices() == 0)
    {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }

    std::string serial = freenect2.getDefaultDeviceSerialNumber();

    dev = freenect2.openDefaultDevice();
    if(dev == NULL)
    {
        std::cout << "failed to open device!" << std::endl;
        return -1;
    }

    signal(SIGINT, sigint_handler);

    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Ir);
    libfreenect2::FrameMap frames;

    dev->setIrAndDepthFrameListener(&listener);
    //dev->setColorFrameListener(&listener);

    if (!dev->start()) return -1;

    cv::Mat colorFrame, irFrame;

    std::vector<frameCallback> frameCallbacks;
    frameCallbacks.push_back(frameHandler);

    cv::namedWindow("yarvis", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("treshold", "yarvis", &sliderVal, sliderMax, NULL, NULL);
 
    while (!protonect_shutdown )
    {
        if (!listener.waitForNewFrame(frames, 10 * 1000)) // 10 seconds
        {
            std::cout << "timeout!" << std::endl;
            return -1;
        }
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        //libfreenect2::Frame *color = frames[libfreenect2::Frame::Color];

        cv::Mat(ir->height, ir->width, CV_32FC1, ir->data).copyTo(irFrame);
        //cv::Mat(color->height, color->width, CV_8UC4, color->data).copyTo(colorFrame);

        //irFrame = irFrame / 32768.0;
        //cv::imshow("ir", irFrame);
        //cv::imshow("color", colorFrame);

        for(frameCallback& callback : frameCallbacks)
        {
            callback(irFrame, &sliderVal);
        }
        cv::waitKey(1);

        listener.release(frames);
    }

    dev->stop();
    dev->close();

    return 0;
}
