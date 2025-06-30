#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class VideoPublisher {
public:
    VideoPublisher() : it_(nh_) {
        // 参数初始化
        nh_.param<std::string>("video_source", video_source_, "/home/mc/Desktop/detect_ws/src/video_publisher/config_files/test.mp4");
        nh_.param<bool>("is_video_file", is_video_file_, false);
        nh_.param<int>("frame_rate", frame_rate_, 30);
        
        // 创建发布者
        image_pub_ = it_.advertise("camera/image_raw", 1);
        
        // 初始化视频源
        init_video_source();
        
        ROS_INFO("视频发布节点已启动");
        ROS_INFO("视频源: %s", video_source_.c_str());
        ROS_INFO("帧率: %d Hz", frame_rate_);
    }

    void run() {
        ros::Rate rate(frame_rate_);
        cv::Mat frame;
        
        while (ros::ok()) {
            if (!cap_.read(frame)) {
                if (is_video_file_) {
                    cap_.set(cv::CAP_PROP_POS_FRAMES, 0); // 循环播放视频文件
                    continue;
                } else {
                    ROS_ERROR("无法从视频源读取帧");
                    break;
                }
            }
            
            // 发布图像消息
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(
                std_msgs::Header(), "bgr8", frame).toImageMsg();
            image_pub_.publish(msg);
            
            ros::spinOnce();
            rate.sleep();
        }
    }

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Publisher image_pub_;
    cv::VideoCapture cap_;
    std::string video_source_;
    bool is_video_file_;
    int frame_rate_;

    void init_video_source() {
        if (is_video_file_) {
            cap_.open(video_source_);
            if (!cap_.isOpened()) {
                ROS_ERROR("无法打开视频文件: %s", video_source_.c_str());
                ros::shutdown();
            }
        } else {
            // 尝试将字符串转换为整数（摄像头索引）
            try {
                int camera_index = std::stoi(video_source_);
                cap_.open(camera_index);
            } catch (...) {
                cap_.open(video_source_);
            }
            
            if (!cap_.isOpened()) {
                ROS_ERROR("无法打开视频源: %s", video_source_.c_str());
                ros::shutdown();
            }
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "video_publisher_node");
    VideoPublisher publisher;
    publisher.run();
    return 0;
}
