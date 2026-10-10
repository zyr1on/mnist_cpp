# MNIST C++ Engine

A lightweight, dependency-free MNIST neural network engine written in modern C++ (C++20). It implements matrix operations, forward/backward propagation, Xavier initialization, and Stochastic Gradient Descent (SGD) completely from scratch using only the C++ standard library.

Trained weights are serialized into a compact binary format that can be reloaded in C++ or loaded into Python for visual inference.

---

## Highlights

- Zero external dependencies: Pure C++ standard library.
- Fast execution: Cache-friendly matrix operations with compiler-level vectorization (`-O3`, `-march=native`, `/fp:fast`).
- Cross-platform: Compiles and runs cleanly on Windows, Linux, and macOS.
- Portable binary weights: Fast serialization (`.bin`) consumable across languages.
- Python demo tools: Image extraction and visual inference demo scripts.

---

## Network Architecture

| Layer | Input Size | Output Size | Activation |
|---|---|---|---|
| Input | 784 (28x28 flattened) | 784 | - |
| Hidden | 784 | 128 | tanh |
| Output | 128 | 10 | softmax |

- Loss function: Cross-Entropy Loss
- Optimizer: Stochastic Gradient Descent (SGD)
- Weight initialization: Xavier / Glorot uniform distribution

---

## Quickstart (C++ API)

Training and exporting a model takes only a few lines of code:

```cpp
#include "MnistLoader.h"
#include "NeuralNetwork.h"

int main() {
    auto train_images = MnistLoader::load_images("data/train-images.idx3-ubyte");
    auto train_labels = MnistLoader::load_labels("data/train-labels.idx1-ubyte");

    NeuralNetwork model;
    model.train(train_images, train_labels, /*epochs=*/3, /*learning_rate=*/0.01f);
    model.save("mnist_model.bin");

    return 0;
}
```

---

## Building and Running

### Prerequisites

- CMake 3.16 or higher
- C++20 compliant compiler (GCC 11+, Clang 13+, or MSVC 2019+)

The MNIST IDX dataset files should be placed inside the `data/` directory:
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

### Using Build Scripts

- **Windows**:
  ```cmd
  run.bat
  ```

- **Linux / macOS**:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

### Manual CMake Build

```bash
mkdir build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
./build/mnist_engine
```

Upon execution, the engine trains across epochs with an interactive progress bar, evaluates accuracy against the 10,000-sample test set (~89-90% in 3 epochs), and outputs `mnist_model.bin`.

---

## Python Demo and Inference

Optional Python utilities are provided under `demo/` using `numpy`, `matplotlib`, and `pillow`.

```bash
pip install numpy matplotlib pillow
```

### 1. Extract Test Samples to PNG

To extract individual PNG images from the MNIST IDX test set:

```bash
python demo/export_samples.py --count 15 --output demo/samples
```

Options:
- `--count`: Number of sample images to extract (default: 15).
- `--images`: Path to test images IDX file (default: `data/t10k-images.idx3-ubyte`).
- `--labels`: Path to test labels IDX file (default: `data/t10k-labels.idx1-ubyte`).
- `--output`: Target folder (default: `demo/samples`).

### 2. Predict on an Image

Test the C++ generated model (`mnist_model.bin`) on any sample image:

```bash
python demo/predict.py --image demo/samples/sample_0_label_7.png
```

Headless / CLI-only mode (skips displaying the matplotlib window):

```bash
python demo/predict.py --image demo/samples/sample_0_label_7.png --no-gui
```

Output:
```text
[*] Image      : demo/samples/sample_0_label_7.png
[*] Model      : mnist_model.bin
[+] Prediction : 7 (Confidence: 99.76%)
```

---

## Project Structure

```text
.
├── CMakeLists.txt          # Cross-platform build configuration
├── run.bat                 # Windows one-click build and run script
├── run.sh                  # Linux/macOS build and run script
├── main.cpp                # Training and evaluation entry point
├── include/
│   ├── MathOps.h           # Tanh and Softmax activation functions
│   ├── Matrix.h            # Matrix container and basic linear algebra
│   ├── MnistLoader.h       # Binary IDX file parser with endian swapping
│   └── NeuralNetwork.h     # Network architecture, training, and persistence
├── src/
│   ├── MathOps.cpp
│   ├── Matrix.cpp
│   ├── MnistLoader.cpp
│   └── NeuralNetwork.cpp
├── data/                   # MNIST raw binary dataset files
├── demo/
│   ├── export_samples.py   # Extracts MNIST IDX images into PNG files
│   ├── predict.py          # Python inference using saved binary weights
│   └── samples/            # Exported sample PNG images
└── mnist_model.bin         # Trained model weights (generated on train)
```

---

## License

MIT License.
