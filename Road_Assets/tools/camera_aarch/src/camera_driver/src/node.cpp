#include "camera_wrapper.h"
#include "tools.h"
#include <glog/logging.h>

void InitLog(const std::string& data_dir) {
    std::string log_dir = data_dir;
    log_dir.append("/logs/");
    camera_driver::CreateDirectoryRecursively(log_dir);

    // 设置日志输出路径
    google::SetLogDestination(google::GLOG_INFO, log_dir.c_str()); // 指定输出路径
    google::SetLogFilenameExtension(".log"); // 设置文件扩展名
    google::SetStderrLogging(google::GLOG_INFO); // 设定输出级别
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    if (argc < 2) {
        std::cout << "please input data directory!";
        return -1;
    }

    const std::string data_dir = argv[1];

    InitLog(data_dir);

    rclcpp::init(argc, argv);

    auto zed_wrapper = std::make_shared<camera_driver::CameraWrapper>();

    zed_wrapper->Init(data_dir);

    zed_wrapper->Run();

    // rclcpp::spin(zed_wrapper);

    google::ShutdownGoogleLogging();

    
    return 0;
}
