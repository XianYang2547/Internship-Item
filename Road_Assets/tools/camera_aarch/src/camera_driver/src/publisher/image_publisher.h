#pragma once
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <string>

namespace camera_driver {
class ImagePublisher : public rclcpp::Node {
public:
    ImagePublisher(const std::string& node_name);
    ~ImagePublisher() = default;

public:
    bool Init(const std::string& topic);
    void Publish(const sensor_msgs::msg::Image& img);

    uint32_t GetNumSubscribers();
    std::string GetTopic() const {return topic_;};

private:
    std::string topic_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};
}  // namespace camera_driver
