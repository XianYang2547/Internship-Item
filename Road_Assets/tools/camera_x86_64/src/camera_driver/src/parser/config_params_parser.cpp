#include "config_params_parser.h"

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace camera_driver {

ConfigParamsParser::ConfigParamsParser(/* args */) {}

bool ConfigParamsParser::Parse(const std::string& cfg_file,
                               ConfigParams& params) {
    if (cfg_file.empty()) {
        return false;
    }
    LOG(INFO) << "config file: " << cfg_file;

    try {
        YAML::Node node = YAML::LoadFile(cfg_file);
        params.camera_model = node["general"]["camera_model"].as<std::string>();
        params.resolution =
            node["general"]["grab_resolution"].as<std::string>();
        params.fps = node["general"]["grab_frame_rate"].as<uint32_t>();

        params.jpg_quality = node["compression"]["jpg_quality"].as<int>();
        params.png_quality = node["compression"]["png_quality"].as<int>();
    } catch (const YAML::BadFile& e) {
        LOG(ERROR) << "Error loading YAML file: " << e.what();
    } catch (const YAML::Exception& e) {
        LOG(ERROR) << "YAML parsing error: " << e.what();
    }

    LOG(INFO) << "CONFIG PARAMS:" << std::endl
              << "\tcamera_model:" << params.camera_model << std::endl
              << "\tresolution:" << params.resolution << std::endl
              << "\tfps:" << params.fps << std::endl
              << "\tjpg_quality:" << params.jpg_quality << std::endl
              << "\tpng_quality:" << params.png_quality << std::endl;
    return true;
}
}  // namespace camera_driver