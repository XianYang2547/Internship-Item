#include "compressed_image_publisher.h"

namespace camera_driver {

CompressedImagePublisher::CompressedImagePublisher(const std::string& node_name) : Node(node_name) {}

bool CompressedImagePublisher::Init(const std::string& topic) {
    if (topic.empty()) {
        return false;
    }

    topic_ = topic;
    publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(topic, 10);
    return true;
}

void CompressedImagePublisher::Publish(const sensor_msgs::msg::CompressedImage& img) { publisher_->publish(img); }

uint32_t CompressedImagePublisher::GetNumSubscribers() { return publisher_->get_subscription_count(); }

}  // namespace camera_driver