#include "MathOps.h"
#include <cmath>
#include <algorithm>

namespace MathOps {
    float tanh_act(float x) {
        return std::tanh(x);
    }

    float tanh_derivative(float a) {
        return 1.0f - (a * a);
    }

    Matrix apply_tanh(const Matrix& Z) {
        Matrix A(Z.rows, Z.cols);
        for (size_t i = 0; i < Z.data.size(); ++i) {
            A.data[i] = std::tanh(Z.data[i]);
        }
        return A;
    }

    Matrix apply_softmax(const Matrix& Z) {
        Matrix S(Z.rows, Z.cols);
        float max_val = *std::max_element(Z.data.begin(), Z.data.end());
        float sum = 0.0f;

        for (size_t i = 0; i < Z.data.size(); ++i) {
            S.data[i] = std::exp(Z.data[i] - max_val);
            sum += S.data[i];
        }
        for (size_t i = 0; i < S.data.size(); ++i) {
            S.data[i] /= sum;
        }
        return S;
    }
}