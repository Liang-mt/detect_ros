#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vision_msgs/Detection2DArray.h>
#include <mutex>

class ResultViewer {
public:
    ResultViewer() : it_(nh_) {
        // 订阅图像和检测结果
        image_sub_ = it_.subscribe("detection_result", 1, 
                                  &ResultViewer::image_callback, this);
        detection_sub_ = nh_.subscribe("detections", 10, 
                                     &ResultViewer::detection_callback, this);
        
        ROS_INFO("结果显示节点已启动");
    }

    // 将 run() 改为 public
    void run() {
        ros::Rate rate(30); // 30Hz刷新率
        while (ros::ok()) {
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                if (new_data_available_ && !current_frame_.empty()) {
                    // 绘制检测结果
                    cv::Mat display_frame = current_frame_.clone();
                    draw_detections(display_frame);
                    
                    // 显示图像
                    cv::imshow("目标检测结果", display_frame);
                    cv::waitKey(1);
                    
                    new_data_available_ = false;
                }
            }
            
            ros::spinOnce();
            rate.sleep();
        }
        cv::destroyAllWindows();
    }

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Subscriber detection_sub_;
    
    cv::Mat current_frame_;
    vision_msgs::Detection2DArray current_detections_;
    std::mutex data_mutex_;
    bool new_data_available_ = false;

    void image_callback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            current_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
            new_data_available_ = true;
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge异常: %s", e.what());
        }
    }

    void detection_callback(const vision_msgs::Detection2DArray::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        current_detections_ = *msg;
    }

    void draw_detections(cv::Mat& image) {
        const std::vector<cv::Scalar> colors = {
            cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0),
            cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)
        };
        
        for (const auto& detection : current_detections_.detections) {
            if (detection.results.empty()) continue;
            
            const cv::Scalar color = colors[detection.results[0].id % colors.size()];
            
            // 提取边界框信息
            const float center_x = detection.bbox.center.x;
            const float center_y = detection.bbox.center.y;
            const float width = detection.bbox.size_x;
            const float height = detection.bbox.size_y;
            
            // 转换为OpenCV矩形
            cv::Rect box(
                center_x - width / 2,
                center_y - height / 2,
                width,
                height
            );
            
            // 绘制边界框
            cv::rectangle(image, box, color, 2);
            
            // 创建标签文本 - 修复格式字符串警告
            // 使用 %ld 替代 %d 来匹配 long int 类型
            std::string label = cv::format("ID: %ld | Score: %.2f", 
                                          detection.results[0].id,
                                          detection.results[0].score);
            
            // 绘制标签背景
            int baseline;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                               0.5, 1, &baseline);
            cv::rectangle(image,
                cv::Point(box.x, box.y - text_size.height - 5),
                cv::Point(box.x + text_size.width, box.y),
                color, cv::FILLED);
            
            // 绘制标签文本
            cv::putText(image, label,
                cv::Point(box.x, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "result_viewer_node");
    ResultViewer viewer;
    viewer.run();  // 现在可以访问 public run() 函数
    return 0;
}
