#include "video_publisher/video_publisher_node.hpp"



VideoPublisher::VideoPublisher() : it_(nh_) {
    // 使用私有节点句柄读取参数
    ros::NodeHandle private_nh("~");
    
    // 参数初始化
    private_nh.param<std::string>("video_source", video_source_, "0");
    private_nh.param<bool>("is_video_file", is_video_file_, false);
    private_nh.param<int>("frame_rate", frame_rate_, 30);
    
    // 打印参数值用于调试
    ROS_INFO("Loaded parameters:");
    ROS_INFO("  video_source: %s", video_source_.c_str());
    ROS_INFO("  is_video_file: %s", is_video_file_ ? "true" : "false");
    ROS_INFO("  frame_rate: %d Hz", frame_rate_);
    
    // 检查视频文件是否存在
    if (is_video_file_) {
        if (access(video_source_.c_str(), F_OK) == -1) {
            ROS_ERROR("Video file does not exist: %s", video_source_.c_str());
            ros::shutdown();
            return;
        }
    }
    
    // 创建发布者
    image_pub_ = it_.advertise("camera/image_raw", 1);
    
    // 初始化视频源
    if (!init_video_source()) {
        ROS_ERROR("Failed to initialize video source");
        ros::shutdown();
        return;
    }
    
    ROS_INFO("Video publisher node started");
}

VideoPublisher::~VideoPublisher() {
    // 释放视频捕获资源
    if (cap_.isOpened()) {
        cap_.release();
        ROS_INFO("Video capture released");
    }
    
    // 关闭图像发布者
    if (image_pub_) {
        image_pub_.shutdown();
        ROS_INFO("Image publisher shutdown");
    }
    
    ROS_INFO("VideoPublisher node stopped");
}

void VideoPublisher::run() {
    ros::Rate rate(frame_rate_);
    cv::Mat frame;
    
    while (ros::ok()) {
        if (!cap_.read(frame)) {
            if (is_video_file_) {
                // 重置视频位置并跳过当前循环
                cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
                continue;
            } else {
                ROS_ERROR("Failed to read frame from video source");
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



bool VideoPublisher::init_video_source() {
    if (is_video_file_) {
        // 打开视频文件
        cap_.open(video_source_);
        if (!cap_.isOpened()) {
            ROS_ERROR("Failed to open video file: %s", video_source_.c_str());
            return false;
        }
        ROS_INFO("Successfully opened video file: %s", video_source_.c_str());
        return true;
    } else {
        // 尝试解析为摄像头索引
        try {
            int camera_index = std::stoi(video_source_);
            if (cap_.open(camera_index, cv::CAP_V4L2)) {
                // 设置摄像头参数
                cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                cap_.set(cv::CAP_PROP_FPS, frame_rate_);
                ROS_INFO("Successfully opened camera device: %d", camera_index);
                return true;
            }
        } catch (...) {
            // 不是数字，继续尝试其他方式
        }
        
        // 尝试直接打开设备路径
        if (cap_.open(video_source_, cv::CAP_V4L2)) {
            // 设置摄像头参数
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            cap_.set(cv::CAP_PROP_FPS, frame_rate_);
            ROS_INFO("Successfully opened video device: %s", video_source_.c_str());
            return true;
        }
        
        // 所有尝试都失败
        ROS_ERROR("Failed to open video source: %s", video_source_.c_str());
        return false;
    }
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "video_publisher");
    VideoPublisher publisher;
    publisher.run();
    return 0;
}
