cmake_minimum_required(VERSION 3.16)
project(camera_driver)

## ZED SDK
find_package(ZED 4 REQUIRED)
find_package(CUDA 11.8 REQUIRED)
find_package(OpenCV 4 REQUIRED)

# find dependencies
list(APPEND CMAKE_PREFIX_PATH "/opt/ros/humble")
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# 查找 glog 包
set(GLOG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/glog)
find_package(glog REQUIRED PATHS ${GLOG_DIR} NO_DEFAULT_PATH)

# 查找 yaml-cpp 包
set(YAML_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/yaml-cpp)
find_package(yaml-cpp REQUIRED PATHS ${YAML_CPP_DIR} NO_DEFAULT_PATH)

# 包含需要的头文件
include_directories(
    ${ZED_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/publisher
    ${CMAKE_CURRENT_SOURCE_DIR}/parser
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

file(GLOB_RECURSE ALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${ALL_SRCS})

target_link_libraries(${PROJECT_NAME}
    ${ZED_LIBRARIES}
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    glog::glog
)

ament_target_dependencies(${PROJECT_NAME} rclcpp sensor_msgs)

# 安装目标
install(TARGETS
    ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../lib
)

ament_package()
