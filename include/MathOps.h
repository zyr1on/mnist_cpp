#pragma once

#include "Matrix.h"

namespace MathOps {
    float tanh_act(float x);
    float tanh_derivative(float a); // a = tanh(z), derivative: 1 - a^2
    Matrix apply_tanh(const Matrix& Z);
    Matrix apply_softmax(const Matrix& Z);
}