#pragma once
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float64.hpp"
#include <string>

namespace camera_driver {
class linear_accelerationPublisher : public rclcpp::Node {
public:
    // 构造函数，接收节点名作为参数
    linear_accelerationPublisher(const std::string& node_name);
    ~linear_accelerationPublisher() = default;
    // 初始化函数，设置发布的主题
    bool Init(const std::string& topic);
    // 发布传感器数据
    void Publish(const std_msgs::msg::Float64& linear_acceleration_Y);
    // 获取订阅者数量
    uint32_t GetNumSubscribers();
    // 获取发布的主题
    std::string GetTopic() const {return topic_;};
private:
    std::string topic_;  // 存储主题名称
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr publisher_;  // 发布器
};
}  // namespace camera_driver
