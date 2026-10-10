#pragma once

#include "Matrix.h"
#include <string>
#include <vector>
#include <cstdint>

class NeuralNetwork {
public:
    Matrix W1; // 128 x 784
    Matrix B1; // 128 x 1
    Matrix W2; // 10 x 128
    Matrix B2; // 10 x 1

    NeuralNetwork();

    // Model Weight Initialization
    void initialize_weights();

    // High-level Python-like API
    void train(const std::vector<Matrix>& train_images,
               const std::vector<uint8_t>& train_labels,
               int epochs = 3,
               float learning_rate = 0.01f,
               int log_interval = 500);

    float evaluate(const std::vector<Matrix>& test_images,
                   const std::vector<uint8_t>& test_labels);

    int predict(const Matrix& image);
    std::vector<float> predict_proba(const Matrix& image);

    void save(const std::string& filepath);
    void load(const std::string& filepath);

private:
    void forward(const Matrix& X, Matrix& Z1, Matrix& A1, Matrix& Z2, Matrix& Out);
    void print_progress(int current, int total, float loss, float acc, double elapsed_sec);
};