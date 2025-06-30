cmake_minimum_required(VERSION 3.0.2)
project(object_detector)

# 查找依赖包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
  vision_msgs
)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 定义包信息
catkin_package(
  CATKIN_DEPENDS 
    roscpp 
    std_msgs 
    sensor_msgs 
    image_transport 
    cv_bridge
    vision_msgs
)

# 包含目录
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# 安装配置文件
install(DIRECTORY config_files/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config_files
)

# 创建可执行文件
add_executable(object_detection_node src/object_detection_node.cpp)

# 链接库
target_link_libraries(object_detection_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# 安装目标
install(TARGETS object_detection_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
