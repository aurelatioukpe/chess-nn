#include "../include/Matrix.hpp"

Matrix Matrix::zeros(size_t rows, size_t cols)
{
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols)
{
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::identity(size_t n)
{
    Matrix result(n, n, 0.0);
    for (size_t i = 0; i < n; i++) {
        result.data[i * n + i] = 1.0;
    }
    return result;
}

double Matrix::sum() const
{
    double total = 0.0;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            total += data[i * cols + j];
        }
    }
    return total;
}

double Matrix::mean() const
{
    if (rows == 0 || cols == 0) {
        return 0.0;
    }
    return sum() / (rows * cols);
}

double Matrix::max() const
{
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Cannot find max of empty matrix");
    }
    double maxVal = data[0];
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (data[i * cols + j] > maxVal) {
                maxVal = data[i * cols + j];
            }
        }
    }
    return maxVal;
}

double Matrix::min() const
{
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Cannot find min of empty matrix");
    }
    double minVal = data[0];
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (data[i * cols + j] < minVal) {
                minVal = data[i * cols + j];
            }
        }
    }
    return minVal;
}
