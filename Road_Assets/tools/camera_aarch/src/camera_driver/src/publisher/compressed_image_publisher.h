#pragma once
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"

namespace camera_driver {
class CompressedImagePublisher : public rclcpp::Node {
public:
    CompressedImagePublisher(const std::string& node_name);
    ~CompressedImagePublisher() = default;

public:
    bool Init(const std::string& topic);
    void Publish(const sensor_msgs::msg::CompressedImage& img);
    uint32_t GetNumSubscribers();
    std::string GetTopic() const { return topic_; };

private:
    std::string topic_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher_;
};
}  // namespace camera_driver
