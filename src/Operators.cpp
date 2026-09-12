#include "../include/Matrix.hpp"

Matrix Matrix::operator+(const Matrix &other) const
{
    return add(other);
}

Matrix Matrix::operator-(const Matrix &other) const
{
    return subtract(other);
}

Matrix Matrix::operator*(const Matrix &other) const
{
    return multiply(other);
}

Matrix& Matrix::operator+=(const Matrix &other)
{
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Cannot add matrices with different dimensions");
    }
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            data[i * cols + j] += other.data[i * cols + j];
        }
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix &other)
{
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Cannot subtract matrices with different dimensions");
    }
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            data[i * cols + j] -= other.data[i * cols + j];
        }
    }
    return *this;
}

Matrix Matrix::operator*(double scalar) const
{
    return scalarMultiply(scalar);
}

Matrix operator*(double scalar, const Matrix &m)
{
    return m * scalar;
}

bool Matrix::operator==(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols) {
        return false;
    }
    const double epsilon = 1e-9;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (std::abs(data[i * cols + j] - other.data[i * cols + j]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::operator!=(const Matrix &other) const
{
    return !(*this == other);
}
