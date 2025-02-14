#include "sensorData_publisher.h"

namespace camera_driver {

linear_accelerationPublisher::linear_accelerationPublisher(const std::string& node_name) : Node(node_name) {}

bool linear_accelerationPublisher::Init(const std::string &topic) {
    if (topic.empty()) {
        return false;
    }

    topic_ = topic;
    publisher_ = this->create_publisher<std_msgs::msg::Float64>(topic, 10);
    return true;
}

void linear_accelerationPublisher::Publish(const std_msgs::msg::Float64 &linear_acceleration_Y) { publisher_->publish(linear_acceleration_Y); }

uint32_t linear_accelerationPublisher::GetNumSubscribers() { return publisher_->get_subscription_count(); }

}  // namespace camera_driver