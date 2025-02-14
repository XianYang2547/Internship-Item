#pragma once
#include <string>

namespace camera_driver {

struct ConfigParams {
    std::string camera_model = "ZED2i";
    std::string resolution = "HD720";
    uint32_t fps = 30;

    int jpg_quality;
    int png_quality;
};

}  // namespace camera_driver