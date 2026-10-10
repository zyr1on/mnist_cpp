#pragma once

#include "Matrix.h"
#include <string>
#include <vector>
#include <cstdint>

class MnistLoader{
public:
    static uint32_t swap_endian(uint32_t val);
    static std::vector<Matrix> load_images(const std::string& path);
    static std::vector<uint8_t> load_labels(const std::string& path);
    
};