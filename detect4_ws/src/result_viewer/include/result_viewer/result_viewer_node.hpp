#ifndef RESULT_VIEWER_NODE_HPP
#define RESULT_VIEWER_NODE_HPP

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vision_msgs/Detection2DArray.h>
#include <mutex>
#include <fstream>

class ResultViewer {
public:
    ResultViewer();
    ~ResultViewer();
    void run();
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Subscriber detection_sub_;
    
    // 参数
    std::string image_topic_;
    std::string detection_topic_;
    std::string classes_file_;
    std::vector<std::string> class_list_;


    cv::Mat current_frame_;
    vision_msgs::Detection2DArray current_detections_;
    std::mutex data_mutex_;
    bool new_data_available_ = false;

    void load_class_list();

    void image_callback(const sensor_msgs::ImageConstPtr& msg);

    void detection_callback(const vision_msgs::Detection2DArray::ConstPtr& msg);

    void draw_detections(cv::Mat& image);
};

#endif
