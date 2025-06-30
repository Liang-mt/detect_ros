#ifndef VIDEO_PUBLISHER_NODE_HPP
#define VIDEO_PUBLISHER_NODE_HPP


#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <unistd.h> // 添加access函数支持

class VideoPublisher {
public:
    VideoPublisher();
    ~VideoPublisher();
    void run();

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Publisher image_pub_;
    cv::VideoCapture cap_;
    std::string video_source_;
    bool is_video_file_;
    int frame_rate_;

    bool init_video_source();
};

#endif
