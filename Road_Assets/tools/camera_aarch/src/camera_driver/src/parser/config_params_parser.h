#pragma once
#include <string>

#include "camera_params.h"

namespace camera_driver {
class ConfigParamsParser {
   public:
    ConfigParamsParser(/* args */);
    ~ConfigParamsParser() = default;

   public:
    bool Parse(const std::string& cfg_file, ConfigParams& params);

   private:
    /* data */
};

}  // namespace camera_driver