#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <sensor_msgs/Image.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>

class ObjectDetector {
public:
    ObjectDetector();
    ~ObjectDetector();
private:
    struct Detection {
        int class_id;
        float confidence;
        cv::Rect box;
    };
    
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Publisher detection_pub_;
    image_transport::Publisher result_image_pub_;
    
    // 参数
    std::string model_path_;
    std::string classes_file_;
    bool use_cuda_;
    float conf_threshold_;
    float nms_threshold_;
    float score_threshold_;
    int input_width_;
    int input_height_;
    
    // 网络和类列表
    cv::dnn::Net net_;
    std::vector<std::string> class_list_;
    
    // 颜色列表
    const std::vector<cv::Scalar> colors_ = {
        cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)
    };

    void load_parameters(ros::NodeHandle& private_nh);

    void load_net();

    void load_class_list();

    cv::Mat format_yolov5(const cv::Mat& source);

    void detect(const cv::Mat& image, std::vector<Detection>& output);

    void draw_boxes(cv::Mat& image, const std::vector<Detection>& output);

    vision_msgs::Detection2DArray to_detection_msg(const std::vector<Detection>& detections, const std_msgs::Header& header);

    void image_callback(const sensor_msgs::ImageConstPtr& msg);
};

#endif

