#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vision_msgs/Detection2DArray.h>
#include <mutex>
#include <fstream>

class ResultViewer {
public:
    ResultViewer() : it_(nh_) {
        // 使用私有节点句柄读取参数
        ros::NodeHandle private_nh("~");
        
        // 获取参数
        private_nh.param<std::string>("image_topic", image_topic_, "detection_result");
        private_nh.param<std::string>("detection_topic", detection_topic_, "detections");
        
        // 打印参数值
        ROS_INFO("Result viewer parameters:");
        ROS_INFO("  image_topic: %s", image_topic_.c_str());
        ROS_INFO("  detection_topic: %s", detection_topic_.c_str());
        
        // 订阅图像和检测结果
        //直接订阅相机的参数
        //image_sub_ = it_.subscribe("camera/image_raw", 1, &ResultViewer::image_callback, this);
        image_sub_ = it_.subscribe(image_topic_, 1, &ResultViewer::image_callback, this);
        detection_sub_ = nh_.subscribe(detection_topic_, 10, &ResultViewer::detection_callback, this);

        //读取txt文件
        private_nh.param<std::string>("classes_file", classes_file_, "");
        if (classes_file_.empty()) {
            ROS_ERROR("classes_file parameter is not set!");
            ros::shutdown();
        }
        load_class_list();
        ROS_INFO("  classes_file: %s", classes_file_.c_str());
        ROS_INFO("Result viewer node started");
    }

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
                    cv::imshow("Object Detection Results", display_frame);
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
    
    // 参数
    std::string image_topic_;
    std::string detection_topic_;
    std::string classes_file_;
    std::vector<std::string> class_list_;


    cv::Mat current_frame_;
    vision_msgs::Detection2DArray current_detections_;
    std::mutex data_mutex_;
    bool new_data_available_ = false;

     void load_class_list() {
        std::ifstream ifs(classes_file_);
        if (!ifs.is_open()) {
            ROS_ERROR("Failed to open classes file: %s", classes_file_.c_str());
            ros::shutdown();
        }
        std::string line;
        while (getline(ifs, line)) {
            class_list_.push_back(line);
        }
        ROS_INFO("Loaded %zu classes", class_list_.size());
    }

    void image_callback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            current_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
            new_data_available_ = true;
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    void detection_callback(const vision_msgs::Detection2DArray::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        current_detections_ = *msg;
    }

    void draw_detections(cv::Mat& image) {
        const std::vector<cv::Scalar> colors = {
            cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0),
            cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0),
            cv::Scalar(255, 0, 255), cv::Scalar(0, 0, 255)
        };
        
        for (const auto& detection : current_detections_.detections) {
            if (detection.results.empty()) continue;
            
            const int class_id = detection.results[0].id;
            const float score = detection.results[0].score;
            const cv::Scalar color = colors[class_id % colors.size()];
            
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
            
            // 确保边界框在图像范围内
            if (box.x < 0) box.x = 0;
            if (box.y < 0) box.y = 0;
            if (box.x + box.width > image.cols) box.width = image.cols - box.x;
            if (box.y + box.height > image.rows) box.height = image.rows - box.y;
            
            // 绘制边界框
            cv::rectangle(image, box, color, 2);
            
            // 创建标签文本
            std::string label = cv::format("%s: %.2f", class_list_[class_id].c_str(), score);
            
            // 绘制标签背景
            int baseline;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,  0.5, 1, &baseline);
            
            // 确保标签不会超出图像顶部
            int text_y = box.y - 5;
            if (text_y < text_size.height + 5) {
                text_y = box.y + box.height + text_size.height + 5;
            }
            
            cv::rectangle(image, cv::Point(box.x, text_y - text_size.height - 5), cv::Point(box.x + text_size.width, text_y), color, cv::FILLED);
            
            // 绘制标签文本
            cv::putText(image, label, cv::Point(box.x, text_y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
};

int main(int argc, char** argv) {
    // 修改节点名称与launch文件一致
    ros::init(argc, argv, "result_viewer");
    ResultViewer viewer;
    viewer.run();
    return 0;
}
