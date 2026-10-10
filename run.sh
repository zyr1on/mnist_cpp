#!/usr/bin/env bash
set -e

echo "======================================================"
echo "   MNIST C++ Engine - Build and Run"
echo "======================================================"

# 1. Create build directory if it does not exist
if [ ! -d "build" ]; then
    echo "[+] Creating 'build' directory..."
    mkdir -p build
fi

# 2. Configure with CMake (Release mode)
echo "[+] Configuring CMake (Release)..."
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# 3. Build project with parallel jobs
echo "[+] Building project..."
cmake --build build --config Release --parallel

echo "======================================================"
echo "[+] Build successful! Launching program..."
echo "======================================================"
echo ""

# 4. Execute binary
if [ -f "build/Release/mnist_engine" ]; then
    ./build/Release/mnist_engine
elif [ -f "build/mnist_engine" ]; then
    ./build/mnist_engine
elif [ -f "build/Release/mnist_engine.exe" ]; then
    ./build/Release/mnist_engine.exe
elif [ -f "build/mnist_engine.exe" ]; then
    ./build/mnist_engine.exe
else
    echo "[!] Executable 'mnist_engine' not found!"
    exit 1
fi

echo ""
echo "======================================================"
echo "[+] Program execution finished successfully."
echo "======================================================"
