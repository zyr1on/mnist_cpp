#include "NeuralNetwork.h"
#include "MathOps.h"
#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <iomanip>

NeuralNetwork::NeuralNetwork()
    : W1(128, 784), B1(128, 1, 0.0f), W2(10, 128), B2(10, 1, 0.0f) {
    initialize_weights();
}

void NeuralNetwork::initialize_weights() {
    std::default_random_engine generator(42);

    // Xavier / Glorot initialization
    float limit1 = std::sqrt(6.0f / (784 + 128));
    std::uniform_real_distribution<float> dist1(-limit1, limit1);
    for (auto& val : W1.data) val = dist1(generator);

    float limit2 = std::sqrt(6.0f / (128 + 10));
    std::uniform_real_distribution<float> dist2(-limit2, limit2);
    for (auto& val : W2.data) val = dist2(generator);
}

void NeuralNetwork::forward(const Matrix& X, Matrix& Z1, Matrix& A1, Matrix& Z2, Matrix& Out) {
    // Hidden layer linear transformation: Z1 = W1 * X + B1
    Z1 = Matrix::matmul(W1, X);
    for (int i = 0; i < Z1.rows; ++i) Z1(i, 0) += B1(i, 0);

    // Hidden layer activation: A1 = tanh(Z1)
    A1 = MathOps::apply_tanh(Z1);

    // Output layer linear transformation: Z2 = W2 * A1 + B2
    Z2 = Matrix::matmul(W2, A1);
    for (int i = 0; i < Z2.rows; ++i) Z2(i, 0) += B2(i, 0);

    // Output layer activation: Out = softmax(Z2)
    Out = MathOps::apply_softmax(Z2);
}

void NeuralNetwork::print_progress(int current, int total, float loss, float acc, double elapsed_sec) {
    int bar_width = 30;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] "
              << std::setw(3) << static_cast<int>(progress * 100.0f) << "% "
              << "(" << current << "/" << total << ") "
              << "| Loss: " << std::fixed << std::setprecision(4) << loss << " "
              << "| Acc: " << std::fixed << std::setprecision(2) << acc << "% "
              << "| Time: " << std::fixed << std::setprecision(1) << elapsed_sec << "s"
              << std::flush;
}

void NeuralNetwork::train(const std::vector<Matrix>& train_images,
                          const std::vector<uint8_t>& train_labels,
                          int epochs,
                          float learning_rate,
                          int log_interval) {
    int total_samples = static_cast<int>(train_images.size());

    std::cout << "\n>>> Training Started [Epochs: " << epochs
              << " | Learning Rate: " << learning_rate
              << " | Samples: " << total_samples << "]" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        float epoch_loss = 0.0f;
        int epoch_correct = 0;

        float window_loss = 0.0f;
        int window_correct = 0;

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << ":" << std::endl;

        for (int i = 0; i < total_samples; ++i) {
            const auto& X = train_images[i];
            uint8_t target = train_labels[i];

            // 1. Forward Pass
            Matrix Z1, A1, Z2, Out;
            forward(X, Z1, A1, Z2, Out);

            auto max_it = std::max_element(Out.data.begin(), Out.data.end());
            int pred = static_cast<int>(std::distance(Out.data.begin(), max_it));
            if (pred == target) {
                epoch_correct++;
                window_correct++;
            }

            float sample_loss = -std::log(std::max(Out.data[target], 1e-7f));
            epoch_loss += sample_loss;
            window_loss += sample_loss;

            // 2. Output layer error: dZ2 = Out - Y
            Matrix dZ2 = Out;
            dZ2.data[target] -= 1.0f;

            // 3. Output layer gradients: dW2 = dZ2 * A1^T
            Matrix A1_T = Matrix::transpose(A1);
            Matrix dW2 = Matrix::matmul(dZ2, A1_T);

            // 4. Hidden layer error: dZ1 = (W2^T * dZ2) * (1 - A1^2)
            Matrix W2_T = Matrix::transpose(W2);
            Matrix dA1 = Matrix::matmul(W2_T, dZ2);

            Matrix dZ1(128, 1);
            for (int j = 0; j < 128; ++j) {
                dZ1.data[j] = dA1.data[j] * MathOps::tanh_derivative(A1.data[j]);
            }

            // 5. Hidden layer gradients: dW1 = dZ1 * X^T
            Matrix X_T = Matrix::transpose(X);
            Matrix dW1 = Matrix::matmul(dZ1, X_T);

            // 6. Stochastic Gradient Descent (SGD) parameter update
            for (size_t k = 0; k < W2.data.size(); ++k) W2.data[k] -= learning_rate * dW2.data[k];
            for (size_t k = 0; k < B2.data.size(); ++k) B2.data[k] -= learning_rate * dZ2.data[k];
            for (size_t k = 0; k < W1.data.size(); ++k) W1.data[k] -= learning_rate * dW1.data[k];
            for (size_t k = 0; k < B1.data.size(); ++k) B1.data[k] -= learning_rate * dZ1.data[k];

            // Progress bar logging
            if ((i + 1) % log_interval == 0 || (i + 1) == total_samples) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - epoch_start).count();
                float curr_loss = window_loss / log_interval;
                float curr_acc = (window_correct * 100.0f) / log_interval;

                print_progress(i + 1, total_samples, curr_loss, curr_acc, elapsed);

                window_loss = 0.0f;
                window_correct = 0;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double total_epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();

        std::cout << "\nEpoch " << (epoch + 1) << " Completed"
                  << " | Avg Loss: " << std::fixed << std::setprecision(4) << (epoch_loss / total_samples)
                  << " | Train Acc: " << std::fixed << std::setprecision(2) << ((epoch_correct * 100.0f) / total_samples) << "%"
                  << " | Time: " << std::fixed << std::setprecision(1) << total_epoch_sec << "s"
                  << "\n----------------------------------------------------------" << std::endl;
    }
}

float NeuralNetwork::evaluate(const std::vector<Matrix>& test_images,
                              const std::vector<uint8_t>& test_labels) {
    int correct = 0;
    int total = static_cast<int>(test_images.size());

    for (int i = 0; i < total; ++i) {
        if (predict(test_images[i]) == test_labels[i]) {
            correct++;
        }
    }

    float accuracy = (correct * 100.0f) / total;
    std::cout << "Evaluation Result: " << correct << " / " << total
              << " (" << std::fixed << std::setprecision(2) << accuracy << "% Accuracy)" << std::endl;
    return accuracy;
}

int NeuralNetwork::predict(const Matrix& image) {
    Matrix Z1, A1, Z2, Out;
    forward(image, Z1, A1, Z2, Out);

    auto max_it = std::max_element(Out.data.begin(), Out.data.end());
    return static_cast<int>(std::distance(Out.data.begin(), max_it));
}

std::vector<float> NeuralNetwork::predict_proba(const Matrix& image) {
    Matrix Z1, A1, Z2, Out;
    forward(image, Z1, A1, Z2, Out);
    return Out.data; // 10 class probabilities
}

void NeuralNetwork::save(const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Unable to save model to " << filepath << std::endl;
        return;
    }
    out.write(reinterpret_cast<const char*>(W1.data.data()), W1.data.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(B1.data.data()), B1.data.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(W2.data.data()), W2.data.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(B2.data.data()), B2.data.size() * sizeof(float));
    std::cout << "Model successfully saved to: " << filepath << std::endl;
}

void NeuralNetwork::load(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Unable to load model from " << filepath << std::endl;
        return;
    }
    in.read(reinterpret_cast<char*>(W1.data.data()), W1.data.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(B1.data.data()), B1.data.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(W2.data.data()), W2.data.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(B2.data.data()), B2.data.size() * sizeof(float));
    std::cout << "Model successfully loaded from: " << filepath << std::endl;
}
