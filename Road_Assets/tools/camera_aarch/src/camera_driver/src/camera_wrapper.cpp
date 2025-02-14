#include "camera_wrapper.h"

// #include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>

#include "config_params_parser.h"
#include "tools.h"

namespace camera_driver {

CameraWrapper::CameraWrapper() : Node("camera_wrapper") {}

CameraWrapper::~CameraWrapper() {
    if (zed_.isOpened()) {
        zed_.close();
    }
}

bool CameraWrapper::Init(const std::string& data_dir) {
    if (data_dir.empty()) {
        return false;
    }

    conn_status_ = sl::ERROR_CODE::CAMERA_NOT_DETECTED;

    // step1: load config params and set camera params
    std::string cfg_file = data_dir;
    cfg_file.append("/cfg/params.yaml");
    ConfigParamsParser parser;
    bool parse_success = parser.Parse(cfg_file, config_params_);
    if (!parse_success) {
        return false;
    }

    InitPublisher();

    SetCameraParams();

    // step2: open the camera
    OpenCamera();

    return true;
}

void CameraWrapper::InitPublisher() {
    left_image_pub_ptr_.reset(new ImagePublisher("left_image_node"));
    left_image_pub_ptr_->Init(config_params_.camera_model + "/left/image");
    left_image_raw_pub_ptr_.reset(new ImagePublisher("left_image_raw_node"));
    left_image_raw_pub_ptr_->Init(config_params_.camera_model + "/left/image_raw");
    left_image_compressed_pub_ptr_.reset(new CompressedImagePublisher("left_image_compressed_node"));
    left_image_compressed_pub_ptr_->Init(config_params_.camera_model + "/left/image_compressed");
    left_image_raw_compressed_pub_ptr_.reset(new CompressedImagePublisher("left_image_raw_compressed_node"));
    left_image_raw_compressed_pub_ptr_->Init(config_params_.camera_model + "/left/image_raw_compressed");
    left_depth_image_pub_ptr_.reset(new ImagePublisher("left_depth_image_node"));
    left_depth_image_pub_ptr_->Init(config_params_.camera_model + "/left/depth_image");
    left_depth_image_compressed_pub_ptr_.reset(new CompressedImagePublisher("left_depth_image_compressed_node"));
    left_depth_image_compressed_pub_ptr_->Init(config_params_.camera_model + "/left/depth_image_compressed");
    left_cloud_pub_ptr_.reset(new ImagePublisher("left_cloud_node"));
    left_cloud_pub_ptr_->Init(config_params_.camera_model + "/left/point_cloud");
    left_cloud_compressed_pub_ptr_.reset(new CompressedImagePublisher("left_cloud_compressed_node"));
    left_cloud_compressed_pub_ptr_->Init(config_params_.camera_model + "/left/point_cloud_compressed");

    right_image_pub_ptr_.reset(new ImagePublisher("right_image_node"));
    right_image_pub_ptr_->Init(config_params_.camera_model + "/right/image");
    right_image_raw_pub_ptr_.reset(new ImagePublisher("right_image_raw_node"));
    right_image_raw_pub_ptr_->Init(config_params_.camera_model + "/right/image_raw");
    right_image_compressed_pub_ptr_.reset(new CompressedImagePublisher("right_image_compressed_node"));
    right_image_compressed_pub_ptr_->Init(config_params_.camera_model + "/right/image_compressed");
    right_image_raw_compressed_pub_ptr_.reset(new CompressedImagePublisher("right_image_raw_compressed_node"));
    right_image_raw_compressed_pub_ptr_->Init(config_params_.camera_model + "/right/image_raw_compressed");
    right_depth_image_pub_ptr_.reset(new ImagePublisher("right_depth_image_node"));
    right_depth_image_pub_ptr_->Init(config_params_.camera_model + "/right/depth_image");
    right_depth_image_compressed_pub_ptr_.reset(new CompressedImagePublisher("right_depth_image_compressed_node"));
    right_depth_image_compressed_pub_ptr_->Init(config_params_.camera_model + "/right/depth_image_compressed");
    right_cloud_pub_ptr_.reset(new ImagePublisher("right_cloud_node"));
    right_cloud_pub_ptr_->Init(config_params_.camera_model + "/right/point_cloud");
    right_cloud_compressed_pub_ptr_.reset(new CompressedImagePublisher("right_cloud_compressed_node"));
    right_cloud_compressed_pub_ptr_->Init(config_params_.camera_model + "/right/point_cloud_compressed");

    linear_acceleration_pub_ptr_.reset(new linear_accelerationPublisher("linear_acceleration_node"));
    linear_acceleration_pub_ptr_->Init(config_params_.camera_model + "/linear_acceleration");
}

bool CameraWrapper::SetCameraParams() {
    // set resolution
    std::string resol = config_params_.resolution;
    if (resol == "HD2K") {
        zed_init_params_.camera_resolution = sl::RESOLUTION::HD2K;
    } else if (resol == "HD1080") {
        zed_init_params_.camera_resolution = sl::RESOLUTION::HD1080;
    } else if (resol == "HD720") {
        zed_init_params_.camera_resolution = sl::RESOLUTION::HD720;
    } else if (resol == "VGA") {
        zed_init_params_.camera_resolution = sl::RESOLUTION::VGA;
    } else {
        LOG(ERROR) << "Not valid 'general.grab_resolution' value: " << resol << ". Using 'AUTO' setting." << std::endl;
        zed_init_params_.camera_resolution = sl::RESOLUTION::HD720;
    }

    // set fps
    zed_init_params_.camera_fps = config_params_.fps;

    zed_init_params_.coordinate_system = sl::COORDINATE_SYSTEM::LEFT_HANDED_Y_UP;
    zed_init_params_.depth_mode = sl::DEPTH_MODE::ULTRA;
    zed_init_params_.coordinate_units = sl::UNIT::METER;
    zed_init_params_.depth_minimum_distance = static_cast<float>(0);
    zed_init_params_.depth_maximum_distance = static_cast<float>(50);
    return true;
}

bool CameraWrapper::OpenCamera() {
    while (conn_status_ != sl::ERROR_CODE::SUCCESS) {
        conn_status_ = zed_.open(zed_init_params_);
        LOG(INFO) << "[ZED connection]: " << sl::toString(conn_status_);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        if (!rclcpp::ok()) {
            LOG(INFO) << "Closing ZED " << zed_serial_number_;
            if (zed_.isOpened()) {
                zed_.close();
            }
            LOG(INFO) << "... ZED " << zed_serial_number_ << " closed.";
            return false;
        }
    }

    zed_serial_number_ = zed_.getCameraInformation().serial_number;
    LOG(INFO) << " * Serial Number: " << zed_serial_number_;
    LOG(INFO) << " ...  " << zed_serial_number_ << " ready";
    return true;
}

void CameraWrapper::Run() {
    frame_timestamp_ = slTime2Ros(zed_.getTimestamp(sl::TIME_REFERENCE::CURRENT));
    prev_frame_timestamp_ = frame_timestamp_;

    sl::RuntimeParameters run_params;
    run_params.enable_depth = true;

    while (rclcpp::ok()) {
        // ZED Grab
        auto grab_status = zed_.grab(run_params);
        if (grab_status != sl::ERROR_CODE::SUCCESS) {
            // Detect if a error occurred (for example: the zed have been
            // disconnected) and re-initialize the ZED
            LOG(ERROR) << "Camera grab error: " << sl::toString(grab_status).c_str();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (grab_status != sl::ERROR_CODE::CAMERA_REBOOTING) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            } else {
                rclcpp::Clock clock;             // 创建时钟对象
                rclcpp::Time now = clock.now();  // 获取当前时间

                if ((now - prev_frame_timestamp_).seconds() > 5) {
                    if (zed_.isOpened()) {
                        zed_.close();
                    }

                    conn_status_ = sl::ERROR_CODE::CAMERA_NOT_DETECTED;

                    while (conn_status_ != sl::ERROR_CODE::SUCCESS) {
                        if (!rclcpp::ok()) {
                            LOG(INFO) << "Closing ZED  " << zed_serial_number_ << "...";
                            if (zed_.isOpened()) {
                                zed_.close();
                            }
                            LOG(INFO) << "... ZED " << zed_serial_number_ << " closed.";
                            return;
                        }

                        zed_init_params_.input.setFromSerialNumber(zed_serial_number_);
                        conn_status_ = zed_.open(zed_init_params_);  // Try to initialize the ZED
                        LOG(INFO) << "Connection status: " << sl::toString(conn_status_);
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    }
                }
            }

            continue;
        }

        frame_count_++;

        frame_timestamp_ = slTime2Ros(zed_.getTimestamp(sl::TIME_REFERENCE::IMAGE));
        prev_frame_timestamp_ = frame_timestamp_;

        // publish data
        PublishVideoData();
    }
}

bool CameraWrapper::PublishVideoData() {
    sl::Mat mat_left_img, mat_left_raw_img;
    sl::Mat mat_right_img, mat_right_raw_img;
    sl::Mat mat_left_depth, mat_right_depth;
    sl::Mat mat_left_cloud, mat_right_cloud;
    sl::SensorsData sensor_data;
    std_msgs::msg::Float64 linear_acceleration_Y;
    sl::Timestamp grab_ts;
    bool retrieved = false;

    uint32_t left_image_sub_num = left_image_pub_ptr_->GetNumSubscribers();
    uint32_t left_image_compressed_sub_num = left_image_compressed_pub_ptr_->GetNumSubscribers();
    uint32_t left_image_raw_sub_num = left_image_raw_pub_ptr_->GetNumSubscribers();
    uint32_t left_image_raw_compressed_sub_num = left_image_raw_compressed_pub_ptr_->GetNumSubscribers();
    uint32_t left_depth_image_sub_num = left_depth_image_pub_ptr_->GetNumSubscribers();
    uint32_t left_depth_image_compressed_sub_num = left_depth_image_compressed_pub_ptr_->GetNumSubscribers();
    uint32_t left_cloud_sub_num = left_cloud_pub_ptr_->GetNumSubscribers();
    uint32_t left_cloud_compressed_sub_num = left_cloud_compressed_pub_ptr_->GetNumSubscribers();
    
    uint32_t right_image_sub_num = right_image_pub_ptr_->GetNumSubscribers();
    uint32_t right_image_compressed_sub_num = right_image_compressed_pub_ptr_->GetNumSubscribers();
    uint32_t right_image_raw_sub_num = right_image_raw_pub_ptr_->GetNumSubscribers();
    uint32_t right_image_raw_compressed_sub_num = right_image_raw_compressed_pub_ptr_->GetNumSubscribers();
    
    uint32_t right_depth_image_sub_num = right_depth_image_pub_ptr_->GetNumSubscribers();
    uint32_t right_depth_image_compressed_sub_num = right_depth_image_compressed_pub_ptr_->GetNumSubscribers();
    uint32_t right_cloud_sub_num = right_cloud_pub_ptr_->GetNumSubscribers();
    uint32_t right_cloud_compressed_sub_num = right_cloud_compressed_pub_ptr_->GetNumSubscribers();
    uint32_t linear_acceleration_sub_num = linear_acceleration_pub_ptr_->GetNumSubscribers();


    // ----> Retrieve all required image data
    if (left_image_sub_num > 0 || left_image_compressed_sub_num > 0) {
        zed_.retrieveImage(mat_left_img, sl::VIEW::LEFT, sl::MEM::CPU);
        grab_ts = mat_left_img.timestamp;
        retrieved = true;
    }

    if (left_image_raw_sub_num || left_image_raw_compressed_sub_num > 0) {
        zed_.retrieveImage(mat_left_raw_img, sl::VIEW::LEFT_UNRECTIFIED, sl::MEM::CPU);
        grab_ts = mat_left_raw_img.timestamp;
        retrieved = true;
    }

    if (left_depth_image_sub_num > 0 || left_depth_image_compressed_sub_num > 0) {
        zed_.retrieveMeasure(mat_left_depth, sl::MEASURE::DEPTH_U16_MM, sl::MEM::CPU);
        grab_ts = mat_left_depth.timestamp;
        retrieved = true;

        sl::Timestamp rgb_ts = mat_left_img.timestamp;
        sl::Timestamp depth_ts = mat_left_depth.timestamp;
        if (rgb_ts.data_ns != 0 && (depth_ts.data_ns != rgb_ts.data_ns)) {
            LOG(WARNING) << "!!!!! LEFT DEPTH/RGB ASYNC !!!!! - Delta: " << 1e-9 * static_cast<double>(depth_ts - rgb_ts) << " sec";
        }
    }

    if (left_cloud_sub_num > 0 || left_cloud_compressed_sub_num > 0) {
        zed_.retrieveMeasure(mat_left_cloud, sl::MEASURE::XYZ, sl::MEM::CPU);
        grab_ts = mat_left_cloud.timestamp;
        retrieved = true;

        sl::Timestamp rgb_ts = mat_left_img.timestamp;
        sl::Timestamp depth_ts = mat_left_cloud.timestamp;
        if (rgb_ts.data_ns != 0 && (depth_ts.data_ns != rgb_ts.data_ns)) {
            LOG(WARNING) << "!!!!! LEFT CLOUD/RGB ASYNC !!!!! - Delta: " << 1e-9 * static_cast<double>(depth_ts - rgb_ts) << " sec";
        }

        // const auto height = mat_left_cloud.getHeight();
        // const auto width = mat_left_cloud.getWidth();

        // for (int h = 0; h < height; ++h) {
        //     const sl::float4* ptr = mat_left_cloud.getPtr<sl::float4>(sl::MEM::CPU);
        //     for (int w = 0; w < width; ++w) {
        //         sl::float4 point_cloud_value = ptr[w];
        //         std::cout << "Point Cloud at (" << w << ", " << h << "): " << point_cloud_value.x << ", "
        //                   << point_cloud_value.y << ", " << point_cloud_value.z << ", " << point_cloud_value.w
        //                   << std::endl;
        //     }
        // }
    }

    if (right_image_sub_num > 0 || right_image_compressed_sub_num > 0) {
        zed_.retrieveImage(mat_right_img, sl::VIEW::RIGHT, sl::MEM::CPU);
        grab_ts = mat_right_img.timestamp;
        retrieved = true;
    }

    if (right_image_raw_sub_num > 0 || right_image_raw_compressed_sub_num > 0) {
        zed_.retrieveImage(mat_right_raw_img, sl::VIEW::RIGHT_UNRECTIFIED, sl::MEM::CPU);
        grab_ts = mat_right_raw_img.timestamp;
        retrieved = true;
    }

    if (right_depth_image_sub_num > 0 || right_depth_image_compressed_sub_num > 0) {
        zed_.retrieveMeasure(mat_right_depth, sl::MEASURE::DEPTH_U16_MM_RIGHT, sl::MEM::CPU);
        grab_ts = mat_right_depth.timestamp;
        retrieved = true;

        sl::Timestamp rgb_ts = mat_left_img.timestamp;
        sl::Timestamp depth_ts = mat_right_depth.timestamp;
        if (rgb_ts.data_ns != 0 && (depth_ts.data_ns != rgb_ts.data_ns)) {
            LOG(WARNING) << "!!!!! RIGHT DEPTH/RGB ASYNC !!!!! - Delta: " << 1e-9 * static_cast<double>(depth_ts - rgb_ts) << " sec";
        }
    }

    if (right_cloud_sub_num > 0 || right_cloud_compressed_sub_num > 0) {
        zed_.retrieveMeasure(mat_right_cloud, sl::MEASURE::XYZ_RIGHT, sl::MEM::CPU);
        grab_ts = mat_right_cloud.timestamp;
        retrieved = true;

        sl::Timestamp rgb_ts = mat_left_img.timestamp;
        sl::Timestamp depth_ts = mat_right_cloud.timestamp;
        if (rgb_ts.data_ns != 0 && (depth_ts.data_ns != rgb_ts.data_ns)) {
            LOG(WARNING) << "!!!!! RIGHT CLOUD/RGB ASYNC !!!!! - Delta: " << 1e-9 * static_cast<double>(depth_ts - rgb_ts) << " sec";
        }
    }

    if (linear_acceleration_sub_num > 0 ) {
        zed_.getSensorsData(sensor_data,sl::TIME_REFERENCE::CURRENT);
        sl::SensorsData::IMUData imu_data = sensor_data.imu;
        grab_ts = imu_data.timestamp;
        double data = imu_data.linear_acceleration[1];
        linear_acceleration_Y.data = data;
        
        retrieved = true;

    }


    if (!retrieved) {
        static int cnt = 0;
        if ((cnt % 100) == 0) {
            LOG(INFO) << "Camera data is not subscribed!";
        }
        cnt++;
        return false;
    }

    // ----> Data ROS timestamp
    rclcpp::Time stamp = slTime2Ros(grab_ts);
    LOG(INFO) << "Publish video data,timestamp: " << stamp.seconds();

    if (left_image_sub_num > 0) {
        PublishImage(mat_left_img, left_image_pub_ptr_, "left_frame", stamp);
    }

    if (left_image_compressed_sub_num > 0) {
        PublishCompressedImage(mat_left_img, left_image_compressed_pub_ptr_, "left_frame", stamp);
    }

    if (left_image_raw_sub_num > 0) {
        PublishImage(mat_left_raw_img, left_image_raw_pub_ptr_, "left_frame", stamp);
    }

    if (left_image_raw_compressed_sub_num > 0) {
        PublishCompressedImage(mat_left_raw_img, left_image_raw_compressed_pub_ptr_, "left_frame", stamp);
    }

    if (left_depth_image_sub_num > 0) {
        PublishImage(mat_left_depth, left_depth_image_pub_ptr_, "left_depth", stamp);
    }

    if (left_depth_image_compressed_sub_num > 0) {
        PublishCompressedDepth(mat_left_depth, left_depth_image_compressed_pub_ptr_, "left_depth", stamp);
    }

    if (left_cloud_sub_num > 0) {
        PublishImage(mat_left_cloud, left_cloud_pub_ptr_, "left_cloud", stamp);
    }

    if (left_cloud_compressed_sub_num > 0) {
        PublishCompressedImage(mat_left_cloud, left_cloud_compressed_pub_ptr_, "left_cloud", stamp);
    }

    if (right_image_sub_num > 0) {
        PublishImage(mat_right_img, right_image_pub_ptr_, "right_frame", stamp);
    }

    if (right_image_compressed_sub_num > 0) {
        PublishCompressedImage(mat_right_img, right_image_compressed_pub_ptr_, "right_frame", stamp);
    }

    if (right_image_raw_sub_num > 0) {
        PublishImage(mat_right_raw_img, right_image_raw_pub_ptr_, "right_frame", stamp);
    }

    if (right_image_raw_compressed_sub_num) {
        PublishCompressedImage(mat_right_raw_img, right_image_raw_compressed_pub_ptr_, "right_frame", stamp);
    }

    if (right_depth_image_sub_num > 0) {
        PublishImage(mat_right_depth, right_depth_image_pub_ptr_, "right_depth", stamp);
    }

    if (right_depth_image_compressed_sub_num > 0) {
        PublishCompressedDepth(mat_right_depth, right_depth_image_compressed_pub_ptr_, "right_depth", stamp);
    }

    if (right_cloud_sub_num > 0) {
        PublishImage(mat_right_cloud, right_cloud_pub_ptr_, "right_cloud", stamp);
    }

    if (right_cloud_compressed_sub_num > 0) {
        PublishCompressedImage(mat_right_cloud, right_cloud_compressed_pub_ptr_, "right_cloud", stamp);
    }

    if (linear_acceleration_sub_num > 0) {
        PublishLinearAcceleration(linear_acceleration_Y, linear_acceleration_pub_ptr_, "linear_acceleration_Y", stamp);
    }

    return true;
}

void CameraWrapper::PublishImage(const sl::Mat& img, const std::shared_ptr<ImagePublisher>& pub, const std::string frame_id, rclcpp::Time t) {
    auto image = ImageToROSmsg(img, frame_id, t);

    pub->Publish(*image);

    LOG(INFO) << "Publish image, topic: " << pub->GetTopic() << ", " << std::fixed << std::setprecision(8) << t.seconds();
}

void CameraWrapper::PublishCompressedImage(const sl::Mat& img, const std::shared_ptr<CompressedImagePublisher>& pub, const std::string frame_id, rclcpp::Time t) {
    // compressed image
    auto cv_mat = ZedMatToCvMat(img);

    std::vector<uchar> buffer;
    std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, config_params_.jpg_quality};
    cv::imencode(".jpg", cv_mat, buffer, compression_params);  // 压缩为 JPEG

    sensor_msgs::msg::CompressedImage msg;
    msg.header.stamp = t;
    msg.format = "jpeg";  // 设置压缩格式
    msg.data = buffer;

    pub->Publish(msg);
}

void CameraWrapper::PublishCompressedDepth(const sl::Mat& img, const std::shared_ptr<CompressedImagePublisher>& pub, const std::string frame_id, rclcpp::Time t) {
    // compressed image
    auto cv_mat = ZedMatToCvMat(img);

    // uint16_t min_value = 65535;
    // uint16_t max_value = 0;
    // for (int v = 0; v < cv_mat.rows; ++v) {
    //     for (int u = 0; u < cv_mat.cols; ++u) {
    //         uint16_t value = cv_mat.at<uint16_t>(v, u);
    //         LOG(INFO) << "depth: " << value;
    //         if (value > max_value) {
    //             max_value = value;
    //         }

    //         if (value < min_value) {
    //             min_value = value;
    //         }
    //     }
    // }

    // LOG(INFO) << "min:" << min_value << ",max:" << max_value;

    // 转换为 CV_8U 类型以便压缩（可选）
    cv::Mat depth_image_8U;
    cv_mat.convertTo(depth_image_8U, CV_8U, 255.0 / 65000.0);  // 归一化

    // 压缩深度图像
    std::vector<uchar> buf;
    std::vector<int> compression_params = {cv::IMWRITE_PNG_COMPRESSION, config_params_.png_quality};
    cv::imencode(".png", depth_image_8U, buf, compression_params);

    // 创建压缩图像消息
    sensor_msgs::msg::CompressedImage msg;

    msg.header.stamp = t;
    msg.format = "png";  // 设置压缩格式
    msg.data = buf;

    pub->Publish(msg);
}

void CameraWrapper::PublishCompressedCloud(const sl::Mat& img, const std::shared_ptr<CompressedImagePublisher>& pub, const std::string frame_id, rclcpp::Time t) {
    PublishCompressedDepth(img, pub, frame_id, t);
}

void CameraWrapper::PublishLinearAcceleration(const std_msgs::msg::Float64& linear_acceleration_Y, const std::shared_ptr<linear_accelerationPublisher>& pub, const std::string frame_id, rclcpp::Time t){
    pub->Publish(linear_acceleration_Y);
    LOG(INFO) << "Publish linear_acceleration_Y, topic: " << pub->GetTopic() << ", " << std::fixed << std::setprecision(8) << t.seconds();
}

bool CameraWrapper::PublishSensorData() { return false; }

bool CameraWrapper::PublishImu() { return false; }

}  // namespace camera_driver
