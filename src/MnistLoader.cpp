#include "MnistLoader.h"
#include <fstream>
#include <stdexcept>

// Convert big-endian to little-endian
uint32_t MnistLoader::swap_endian(uint32_t val) {
    return ((val >> 24) & 0x000000FF) |
           ((val >> 8)  & 0x0000FF00) |
           ((val << 8)  & 0x00FF0000) |
           ((val << 24) & 0xFF000000);
}

std::vector<Matrix> MnistLoader::load_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    uint32_t magic = 0, count = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&count), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    count = swap_endian(count);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    int image_size = rows * cols;
    std::vector<Matrix> images;
    images.reserve(count);

    std::vector<uint8_t> buffer(image_size);
    for (uint32_t i = 0; i < count; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), image_size);
        Matrix img(image_size, 1);
        for (int j = 0; j < image_size; ++j) {
            img.data[j] = (static_cast<float>(buffer[j]) / 127.5f) - 1.0f;
        }
        images.push_back(img);
    }
    return images;
}

std::vector<uint8_t> MnistLoader::load_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    uint32_t magic = 0, count = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&count), 4);
    count = swap_endian(count);

    std::vector<uint8_t> labels(count);
    file.read(reinterpret_cast<char*>(labels.data()), count);
    return labels;
}