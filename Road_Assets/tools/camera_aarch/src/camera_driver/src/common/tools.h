#pragma once
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>  // OpenCV

namespace camera_driver {

/*! \brief Convert StereoLabs timestamp to ROS timestamp
 *  \param t : Stereolabs timestamp to be converted
 *  \param t : ROS2 clock type
 */
rclcpp::Time slTime2Ros(sl::Timestamp t, rcl_clock_type_t clock_type = RCL_ROS_TIME);

/*! \brief sl::Mat to ros message conversion
 * \param img : the image to publish
 * \param frameId : the id of the reference frame of the image
 * \param t : rclcpp ros::Time to stamp the image
 */
std::unique_ptr<sensor_msgs::msg::Image> ImageToROSmsg(
  const sl::Mat & img, const std::string & frameId, const rclcpp::Time & t);

cv::Mat ZedMatToCvMat(const sl::Mat& zedMat);

bool CreateDirectoryRecursively(const std::string& path);

}  // namespace camera_driver