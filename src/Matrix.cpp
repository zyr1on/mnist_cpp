#include "Matrix.h"
#include <cassert>

Matrix::Matrix(int r, int c, float init_val) 
    : rows(r), cols(c), data(r * c, init_val) {}

Matrix Matrix::matmul(const Matrix& A, const Matrix& B) {
    assert(A.cols == B.rows && "Matrix dimensions incompatible for multiplication!");
    Matrix out(A.rows, B.cols, 0.0f);

    for (int i = 0; i < A.rows; ++i) {
        for (int k = 0; k < A.cols; ++k) {
            float a_ik = A(i, k);
            for (int j = 0; j < B.cols; ++j) {
                out(i, j) += a_ik * B(k, j); 
            }
        }
    }
    return out;
}

Matrix Matrix::transpose(const Matrix& A) {
    Matrix out(A.cols, A.rows);
    for (int r = 0; r < A.rows; ++r) {
        for (int c = 0; c < A.cols; ++c) {
            out(c, r) = A(r, c);
        }
    }
    return out;
}
