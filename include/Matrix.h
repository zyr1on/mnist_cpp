#pragma once

#include <iostream>
#include <vector>

struct Matrix {
    int rows;
    int cols;
    std::vector<float> data;
    Matrix(int r = 0, int c = 0, float init_val = 0.0f);

    inline float &operator()(int r, int c) { 
        return data[r * cols + c]; 
    }
    
    inline const float& operator()(int r, int c) const {
        return data[r * cols + c];
    }

    static Matrix matmul(const Matrix& A, const Matrix& B);
    static Matrix transpose(const Matrix& a);
    

};
