#include "image_publisher.h"

namespace camera_driver {

ImagePublisher::ImagePublisher(const std::string& node_name) : Node(node_name) {}

bool ImagePublisher::Init(const std::string &topic) {
    if (topic.empty()) {
        return false;
    }

    topic_ = topic;
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>(topic, 10);
    return true;
}

void ImagePublisher::Publish(const sensor_msgs::msg::Image &img) { publisher_->publish(img); }

uint32_t ImagePublisher::GetNumSubscribers() { return publisher_->get_subscription_count(); }

}  // namespace camera_driver