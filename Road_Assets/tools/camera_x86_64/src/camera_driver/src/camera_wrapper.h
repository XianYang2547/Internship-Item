#pragma once

#include <memory>
#include <sl/Camera.hpp>
#include <string>

#include "camera_params.h"
#include "compressed_image_publisher.h"
#include "image_publisher.h"
#include "sensorData_publisher.h"

namespace camera_driver {

struct CameraGereralParams {
    sl::RESOLUTION rresolution;
    uint32_t fps;
};

class CameraWrapper : public rclcpp::Node {
public:
    CameraWrapper();
    ~CameraWrapper();

public:
    bool Init(const std::string& data_dir);
    void Run();

private:
    void InitPublisher();
    bool SetCameraParams();
    bool OpenCamera();

    bool PublishVideoData();
    bool PublishSensorData();
    void PublishImage(const sl::Mat& img, const std::shared_ptr<ImagePublisher>& pub, const std::string frame_id,
                      rclcpp::Time t);
    void PublishCompressedImage(const sl::Mat& img, const std::shared_ptr<CompressedImagePublisher>& pub,
                                const std::string frame_id, rclcpp::Time t);

    void PublishCompressedDepth(const sl::Mat& img, const std::shared_ptr<CompressedImagePublisher>& pub,
                                const std::string frame_id, rclcpp::Time t);

    void PublishCompressedCloud(const sl::Mat& img, const std::shared_ptr<CompressedImagePublisher>& pub,
                                const std::string frame_id, rclcpp::Time t);
    void PublishLinearAcceleration(const std_msgs::msg::Float64& linear_acceleration_Y, const std::shared_ptr<linear_accelerationPublisher>& pub,
                                const std::string frame_id, rclcpp::Time t);
    bool PublishImu();

private:
    std::string data_dir_;

    sl::InitParameters zed_init_params_;
    sl::Camera zed_;

    ConfigParams config_params_;

    sl::ERROR_CODE conn_status_;

    unsigned int zed_serial_number_;

    // Last frame time
    rclcpp::Time prev_frame_timestamp_;
    rclcpp::Time frame_timestamp_;
    uint32_t frame_count_ = 0;

    // publisher
    std::shared_ptr<ImagePublisher> left_image_pub_ptr_ = nullptr;
    std::shared_ptr<CompressedImagePublisher> left_image_compressed_pub_ptr_ = nullptr;
    std::shared_ptr<ImagePublisher> left_image_raw_pub_ptr_ = nullptr;
    std::shared_ptr<CompressedImagePublisher> left_image_raw_compressed_pub_ptr_ = nullptr;

    std::shared_ptr<ImagePublisher> right_image_pub_ptr_ = nullptr;
    std::shared_ptr<CompressedImagePublisher> right_image_compressed_pub_ptr_ = nullptr;
    std::shared_ptr<ImagePublisher> right_image_raw_pub_ptr_ = nullptr;
    std::shared_ptr<CompressedImagePublisher> right_image_raw_compressed_pub_ptr_ = nullptr;

    std::shared_ptr<ImagePublisher> left_depth_image_pub_ptr_ = nullptr;
    std::shared_ptr<ImagePublisher> right_depth_image_pub_ptr_ = nullptr;
    std::shared_ptr<CompressedImagePublisher> left_depth_image_compressed_pub_ptr_ = nullptr;
    std::shared_ptr<CompressedImagePublisher> right_depth_image_compressed_pub_ptr_ = nullptr;

    std::shared_ptr<ImagePublisher> left_cloud_pub_ptr_ = nullptr;
    std::shared_ptr<CompressedImagePublisher> left_cloud_compressed_pub_ptr_ = nullptr;
    std::shared_ptr<ImagePublisher> right_cloud_pub_ptr_ = nullptr;
    std::shared_ptr<CompressedImagePublisher> right_cloud_compressed_pub_ptr_ = nullptr;
    
    std::shared_ptr<linear_accelerationPublisher> linear_acceleration_pub_ptr_ = nullptr;
};

}  // namespace camera_driver
