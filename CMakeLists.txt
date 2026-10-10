cmake_minimum_required(VERSION 3.16)
project(mnist_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source files
set(SOURCES
    src/MathOps.cpp
    src/Matrix.cpp
    src/MnistLoader.cpp
    src/NeuralNetwork.cpp
    main.cpp
)

add_executable(mnist_engine ${SOURCES})

target_include_directories(mnist_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Cross-platform compiler optimizations
if(MSVC)
    target_compile_options(mnist_engine PRIVATE
        /O2        # Maximum speed optimization
        /Oi        # Enable intrinsic functions
        /Ot        # Favor fast code
        /fp:fast   # Fast floating point
        /W3
    )
else()
    # GCC / Clang / AppleClang
    target_compile_options(mnist_engine PRIVATE
        -O3
        -ffast-math
        -Wall
        -Wextra
    )

    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(mnist_engine PRIVATE -march=native)
    endif()
endif()
