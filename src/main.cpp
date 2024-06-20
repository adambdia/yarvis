#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <signal.h>


bool protonect_shutdown = false; // Whether the running application should shut down.
double frameGamma = 4096.0f;
double maxGamma = 32768.0f, minGamma = 4096.0f;
int sliderMax = 100;
int sliderVal;

void sigint_handler(int s)
{
  protonect_shutdown = true;
}

void updateGamma(int, void*)
{
    frameGamma = (double)( maxGamma - (maxGamma-minGamma)/sliderVal); 

}
int main()
{
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = 0;
    libfreenect2::PacketPipeline *pipeline = 0;

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

    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Ir | libfreenect2::Frame::Color);
    libfreenect2::FrameMap frames;

    dev->setIrAndDepthFrameListener(&listener);
    dev->setColorFrameListener(&listener);

    if (!dev->start()) return -1;

    cv::Mat colorFrame, irFrame;
 
    while (!protonect_shutdown )
    {
        if (!listener.waitForNewFrame(frames, 10 * 1000)) // 10 seconds
        {
            std::cout << "timeout!" << std::endl;
            return -1;
        }

        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *color = frames[libfreenect2::Frame::Color];

        cv::Mat(ir->height, ir->width, CV_32FC1, ir->data).copyTo(irFrame);
        cv::Mat(color->height, color->width, CV_8UC4, color->data).copyTo(colorFrame);


        cv::imshow("ir", irFrame / 32768.0);
        cv::imshow("color", colorFrame);
        cv::waitKey(1);

        listener.release(frames);
    }

    dev->stop();
    dev->close();

    return 0;
}