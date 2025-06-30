#include "object_detector/object_detection_node.hpp"

ObjectDetector::ObjectDetector() : it_(nh_) 
{
    // 使用私有节点句柄
    ros::NodeHandle private_nh("~");
    
    // 加载参数
    load_parameters(private_nh);
    
    // 初始化网络
    load_net();
    load_class_list();
    
    // 订阅和发布
    image_sub_ = it_.subscribe("camera/image_raw", 1, &ObjectDetector::image_callback, this);
    detection_pub_ = nh_.advertise<vision_msgs::Detection2DArray>("detections", 10);
    result_image_pub_ = it_.advertise("detection_result", 1);
    
    ROS_INFO("Object detection node started");
    ROS_INFO("Using model: %s", model_path_.c_str());
    ROS_INFO("Input size: %dx%d", input_width_, input_height_);
}

ObjectDetector::~ObjectDetector() {
    // 取消订阅
    if (image_sub_) {
        image_sub_.shutdown();
    }
    
    // 关闭发布器
    if (detection_pub_) {
        detection_pub_.shutdown();
    }
    
    if (result_image_pub_) {
        result_image_pub_.shutdown();
    }
    
    ROS_INFO("ObjectDetector destroyed");
}

void ObjectDetector::load_parameters(ros::NodeHandle& private_nh) {
    // 使用私有节点句柄读取参数
    private_nh.param<std::string>("model_path", model_path_, "");
    private_nh.param<std::string>("classes_file", classes_file_, "");
    private_nh.param<bool>("use_cuda", use_cuda_, false);
    private_nh.param<float>("conf_threshold", conf_threshold_, 0.4f);
    private_nh.param<float>("nms_threshold", nms_threshold_, 0.4f);
    private_nh.param<float>("score_threshold", score_threshold_, 0.2f);
    private_nh.param<int>("input_width", input_width_, 640);
    private_nh.param<int>("input_height", input_height_, 640);
    
    // 检查必需参数
    if (model_path_.empty()) {
        ROS_ERROR("model_path parameter is not set!");
        ros::shutdown();
    }
    
    if (classes_file_.empty()) {
        ROS_ERROR("classes_file parameter is not set!");
        ros::shutdown();
    }
    
    // 打印参数值
    ROS_INFO("Parameters:");
    ROS_INFO("  model_path: %s", model_path_.c_str());
    ROS_INFO("  classes_file: %s", classes_file_.c_str());
    ROS_INFO("  use_cuda: %s", use_cuda_ ? "true" : "false");
    ROS_INFO("  conf_threshold: %.2f", conf_threshold_);
    ROS_INFO("  nms_threshold: %.2f", nms_threshold_);
    ROS_INFO("  score_threshold: %.2f", score_threshold_);
    ROS_INFO("  input_size: %dx%d", input_width_, input_height_);
}

void ObjectDetector::load_net() {
    try {
        net_ = cv::dnn::readNet(model_path_);
        if (use_cuda_) {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
            ROS_INFO("Using CUDA acceleration");
        } else {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            ROS_INFO("Using CPU");
        }
    } catch (...) {
        ROS_ERROR("Failed to load model: %s", model_path_.c_str());
        ros::shutdown();
    }
}

void ObjectDetector::load_class_list() {
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

cv::Mat ObjectDetector::format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void ObjectDetector::detect(const cv::Mat& image, std::vector<Detection>& output) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1./255., 
                            cv::Size(input_width_, input_height_), 
                            cv::Scalar(), true, false);
    net_.setInput(blob);
    
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / static_cast<float>(input_width_);
    float y_factor = input_image.rows / static_cast<float>(input_height_);

    float* data = (float*)outputs[0].data;
    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= conf_threshold_) {
            cv::Mat scores(1, class_list_.size(), CV_32FC1, data + 5);
            cv::Point class_id;
            double max_score;
            cv::minMaxLoc(scores, 0, &max_score, 0, &class_id);
            if (max_score > score_threshold_) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = static_cast<int>((x - 0.5 * w) * x_factor);
                int top = static_cast<int>((y - 0.5 * h) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);

                boxes.emplace_back(left, top, width, height);
            }
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, nms_result);
    for (int idx : nms_result) {
        output.push_back({
            class_ids[idx],
            confidences[idx],
            boxes[idx]
        });
    }
}

void ObjectDetector::draw_boxes(cv::Mat& image, const std::vector<Detection>& output) {
    for (const auto& detection : output) {
        const auto& color = colors_[detection.class_id % colors_.size()];
        const auto& box = detection.box;

        cv::rectangle(image, box, color, 2);
        std::string label = cv::format("%s: %.2f", 
            class_list_[detection.class_id].c_str(), 
            detection.confidence);

        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                            0.5, 1, &baseline);

        cv::rectangle(image,
            cv::Point(box.x, box.y - text_size.height - 5),
            cv::Point(box.x + text_size.width, box.y),
            color, cv::FILLED);

        cv::putText(image, label,
            cv::Point(box.x, box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

vision_msgs::Detection2DArray ObjectDetector::to_detection_msg(
    const std::vector<Detection>& detections, 
    const std_msgs::Header& header) 
{
    vision_msgs::Detection2DArray msg;
    msg.header = header;
    
    for (const auto& det : detections) {
        vision_msgs::Detection2D detection;
        detection.header = header;
        
        // 边界框
        detection.bbox.size_x = det.box.width;
        detection.bbox.size_y = det.box.height;
        detection.bbox.center.x = det.box.x + det.box.width / 2.0;
        detection.bbox.center.y = det.box.y + det.box.height / 2.0;
        
        // 类别和置信度
        vision_msgs::ObjectHypothesisWithPose hypothesis;
        hypothesis.id = det.class_id;
        hypothesis.score = det.confidence;
        detection.results.push_back(hypothesis);
        
        msg.detections.push_back(detection);
    }
    
    return msg;
}

void ObjectDetector::image_callback(const sensor_msgs::ImageConstPtr& msg) {
    
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        cv::Mat frame2 = cv_bridge::toCvCopy(msg, "bgr8")->image;
        
        // 检测目标
        std::vector<Detection> detections;
        detect(frame, detections);
        
        if (!detections.empty()) {
            ROS_INFO("Detected %zu objects", detections.size());
            
            // 发布检测结果消息
            vision_msgs::Detection2DArray detections_msg = to_detection_msg(detections, msg->header);
            detection_pub_.publish(detections_msg);
            
            // 绘制边界框
            //draw_boxes(frame, detections);
        }
        
        // 发布带检测结果的图像
        sensor_msgs::ImagePtr result_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
        result_image_pub_.publish(result_msg);
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "object_detection");
    ObjectDetector detector;
    ros::spin();
    return 0;
}
