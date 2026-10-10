#include "MnistLoader.h"
#include "NeuralNetwork.h"
#include <iostream>

int main() {
    try {
        std::cout << "[*] Loading dataset..." << std::endl;
        auto train_images = MnistLoader::load_images("data/train-images.idx3-ubyte");
        auto train_labels = MnistLoader::load_labels("data/train-labels.idx1-ubyte");
        auto test_images  = MnistLoader::load_images("data/t10k-images.idx3-ubyte");
        auto test_labels  = MnistLoader::load_labels("data/t10k-labels.idx1-ubyte");

        // Initialize neural network model
        NeuralNetwork model;

        // Train model: 3 epochs, learning rate 0.01
        model.train(train_images, train_labels, 3, 0.01f);

        // Evaluate model on test dataset
        std::cout << "\n[*] Evaluating model on test set..." << std::endl;
        model.evaluate(test_images, test_labels);

        // Save trained weights to binary file
        model.save("mnist_model.bin");

    } catch (const std::exception& e) {
        std::cerr << "[!] Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
