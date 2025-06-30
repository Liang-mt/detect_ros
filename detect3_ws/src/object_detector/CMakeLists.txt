cmake_minimum_required(VERSION 3.0.2)
project(object_detector)

# 设置cv_bridge_DIR（如果需要）
set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)  # 在find_package前面

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
find_package(OpenCV 4.11.0 REQUIRED)

# 定义包信息
catkin_package(
  # 包含目录（如果有自定义头文件）
  INCLUDE_DIRS include
  # 导出的库（如果有）
  LIBRARIES ${PROJECT_NAME}
  # catkin依赖项
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs image_transport cv_bridge vision_msgs
  # 系统依赖项（如果有）
  # DEPENDS system_library
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
