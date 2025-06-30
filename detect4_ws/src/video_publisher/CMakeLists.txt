cmake_minimum_required(VERSION 3.0.2)
project(video_publisher)

set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)  #在find_package前面
# 查找依赖包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
)

# 查找OpenCV
find_package(OpenCV 4.11.0 REQUIRED)

# 定义包信息
catkin_package(
  CATKIN_DEPENDS 
    roscpp 
    std_msgs 
    sensor_msgs 
    image_transport 
    cv_bridge
)

# 包含目录
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include
)

# 创建可执行文件
add_executable(video_publisher_node src/video_publisher_node.cpp)

# 链接库
target_link_libraries(video_publisher_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# 安装目标
install(TARGETS video_publisher_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
