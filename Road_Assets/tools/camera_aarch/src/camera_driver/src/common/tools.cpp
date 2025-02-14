#include "tools.h"

#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sensor_msgs/image_encodings.hpp>
#include <memory>
#include <opencv2/opencv.hpp>

namespace camera_driver {
rclcpp::Time slTime2Ros(sl::Timestamp t, rcl_clock_type_t clock_type)
{
  uint64_t ts_nsec = t.getNanoseconds();
  uint32_t sec = static_cast<uint32_t>(ts_nsec / 1000000000);
  uint32_t nsec = static_cast<uint32_t>(ts_nsec % 1000000000);
  return rclcpp::Time(sec, nsec, clock_type);
}


std::unique_ptr<sensor_msgs::msg::Image> ImageToROSmsg(
  const sl::Mat & img, const std::string & frameId, const rclcpp::Time & t)
{
  std::unique_ptr<sensor_msgs::msg::Image> imgMessage = std::make_unique<sensor_msgs::msg::Image>();

  imgMessage->header.stamp = t;
  imgMessage->header.frame_id = frameId;
  imgMessage->height = img.getHeight();
  imgMessage->width = img.getWidth();

  int num = 1;  // for endianness detection
  imgMessage->is_bigendian = !(*reinterpret_cast<char *>(&num) == 1);

  imgMessage->step = img.getStepBytes();

  size_t size = imgMessage->step * imgMessage->height;

  uint8_t * data_ptr = nullptr;

  sl::MAT_TYPE dataType = img.getDataType();

  switch (dataType) {
    case sl::MAT_TYPE::F32_C1: /**< float 1 channel.*/
      imgMessage->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
      data_ptr = reinterpret_cast<uint8_t *>(img.getPtr<sl::float1>());
      imgMessage->data = std::vector<uint8_t>(data_ptr, data_ptr + size);
      break;

    case sl::MAT_TYPE::F32_C2: /**< float 2 channels.*/
      imgMessage->encoding = sensor_msgs::image_encodings::TYPE_32FC2;
      data_ptr = reinterpret_cast<uint8_t *>(img.getPtr<sl::float2>());
      imgMessage->data = std::vector<uint8_t>(data_ptr, data_ptr + size);
      break;

    case sl::MAT_TYPE::F32_C3: /**< float 3 channels.*/
      imgMessage->encoding = sensor_msgs::image_encodings::TYPE_32FC3;
      data_ptr = reinterpret_cast<uint8_t *>(img.getPtr<sl::float3>());
      imgMessage->data = std::vector<uint8_t>(data_ptr, data_ptr + size);
      break;

    case sl::MAT_TYPE::F32_C4: /**< float 4 channels.*/
      imgMessage->encoding = sensor_msgs::image_encodings::TYPE_32FC4;
      data_ptr = reinterpret_cast<uint8_t *>(img.getPtr<sl::float4>());
      imgMessage->data = std::vector<uint8_t>(data_ptr, data_ptr + size);
      break;

    case sl::MAT_TYPE::U8_C1: /**< unsigned char 1 channel.*/
      imgMessage->encoding = sensor_msgs::image_encodings::MONO8;
      data_ptr = reinterpret_cast<uint8_t *>(img.getPtr<sl::uchar1>());
      imgMessage->data = std::vector<uint8_t>(data_ptr, data_ptr + size);
      break;

    case sl::MAT_TYPE::U8_C2: /**< unsigned char 2 channels.*/
      imgMessage->encoding = sensor_msgs::image_encodings::TYPE_8UC2;
      data_ptr = reinterpret_cast<uint8_t *>(img.getPtr<sl::uchar2>());
      imgMessage->data = std::vector<uint8_t>(data_ptr, data_ptr + size);
      break;

    case sl::MAT_TYPE::U8_C3: /**< unsigned char 3 channels.*/
      imgMessage->encoding = sensor_msgs::image_encodings::BGR8;
      data_ptr = reinterpret_cast<uint8_t *>(img.getPtr<sl::uchar3>());
      imgMessage->data = std::vector<uint8_t>(data_ptr, data_ptr + size);
      break;

    case sl::MAT_TYPE::U8_C4: /**< unsigned char 4 channels.*/
      imgMessage->encoding = sensor_msgs::image_encodings::BGRA8;
      data_ptr = reinterpret_cast<uint8_t *>(img.getPtr<sl::uchar4>());
      imgMessage->data = std::vector<uint8_t>(data_ptr, data_ptr + size);
      break;
  }

  return imgMessage;
}

cv::Mat ZedMatToCvMat(const sl::Mat& img) {
    const auto height = img.getHeight();
    const auto width = img.getWidth();
    cv::Mat result;
    sl::MAT_TYPE dataType = img.getDataType();
    switch (dataType) {
        case sl::MAT_TYPE::F32_C1: /**< float 1 channel.*/
            result = cv::Mat(height, width, CV_32FC1, img.getPtr<sl::float1>());
            break;

        case sl::MAT_TYPE::F32_C2: /**< float 2 channels.*/
            result = cv::Mat(height, width, CV_32FC2, img.getPtr<sl::float2>());
            break;

        case sl::MAT_TYPE::F32_C3: /**< float 3 channels.*/
            result = cv::Mat(height, width, CV_32FC3, img.getPtr<sl::float3>());
            break;

        case sl::MAT_TYPE::F32_C4: /**< float 4 channels.*/
            result = cv::Mat(height, width, CV_32FC4, img.getPtr<sl::float4>());
            break;

        case sl::MAT_TYPE::U8_C1: /**< unsigned char 1 channel.*/
            result = cv::Mat(height, width, CV_8UC1, img.getPtr<sl::uchar1>());
            break;

        case sl::MAT_TYPE::U8_C2: /**< unsigned char 2 channels.*/
            result = cv::Mat(height, width, CV_8UC2, img.getPtr<sl::uchar2>());
            break;

        case sl::MAT_TYPE::U8_C3: /**< unsigned char 3 channels.*/
            result = cv::Mat(height, width, CV_8UC3, img.getPtr<sl::uchar3>());
            break;

        case sl::MAT_TYPE::U8_C4: /**< unsigned char 4 channels.*/
            result = cv::Mat(height, width, CV_8UC4, img.getPtr<sl::uchar4>());
            break;

        case sl::MAT_TYPE::U16_C1: /**< unsigned short 1 channel.*/
            result = cv::Mat(height, width, CV_16UC1, img.getPtr<sl::ushort1>());
            break;
    }

    return result;
}

bool CreateDirectoryRecursively(const std::string& path) {
    if (mkdir(path.c_str(), 0777) == 0) {
        return true;  // 成功创建
    } else if (errno == EEXIST) {
        return true;  // 目录已经存在
    } else if (errno == ENOENT) {
        // 目录的上级目录不存在，递归创建上级目录
        size_t pos = path.find_last_of('/');
        if (pos != std::string::npos) {
            if (CreateDirectoryRecursively(path.substr(0, pos))) {
                return mkdir(path.c_str(), 0777) == 0;
            }
        }
    }
    return false;  // 其他错误
}

}  // namespace camera_driver