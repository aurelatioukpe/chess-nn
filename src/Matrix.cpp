#include "../include/Matrix.hpp"

Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols)
{
    // OPTIMISATION: allocation flat d'un seul coup
    data.resize(rows * cols, 0.0);
}

Matrix::Matrix(size_t rows, size_t cols, double initValue) : rows(rows), cols(cols)
{
    // OPTIMISATION: allocation flat d'un seul coup
    data.resize(rows * cols, initValue);
}

double Matrix::get(size_t i, size_t j) const
{
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    return data[i * cols + j];
}

void Matrix::set(size_t i, size_t j, double value)
{
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    data[i * cols + j] = value;
}

double& Matrix::operator()(size_t i, size_t j)
{
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    return data[i * cols + j];
}

const double& Matrix::operator()(size_t i, size_t j) const
{
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    return data[i * cols + j];
}

void Matrix::print() const
{
    std::cout << "Matrix " << rows << "x" << cols << ":\n";
    for (size_t i = 0; i < rows; i++) {
        std::cout << "[ ";
        for (size_t j = 0; j < cols; j++) {
            std::cout << data[i * cols + j];
            if (j < cols - 1) std::cout << ", ";
        }
        std::cout << " ]\n";
    }
    std::cout << std::endl;
}

Matrix Matrix::add(const Matrix &other) const
{
    Matrix result(rows, cols);

    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument(
            "Cannot add matrices with different dimensions"
        );
    }
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[i * cols + j] = data[i * cols + j] + other.data[i * cols + j];
        }
    }
    return result;
}

Matrix Matrix::subtract(const Matrix &other) const
{
    Matrix result(rows, cols);

    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument(
            "Cannot subtract matrices with different dimensions"
        );
    }
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[i * cols + j] = data[i * cols + j] - other.data[i * cols + j];
        }
    }
    return result;
}

void Matrix::fill(double value)
{
    // OPTIMISATION: remplissage direct du vecteur flat
    std::fill(data.begin(), data.end(), value);
}

Matrix Matrix::transpose() const
{
    Matrix result(cols, rows);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[j * rows + i] = data[i * cols + j];
        }
    }
    return result;
}

Matrix Matrix::scalarMultiply(double scalar) const
{
    Matrix result(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[i * cols + j] = data[i * cols + j] * scalar;
        }
    }
    return result;
}

Matrix::Matrix(const Matrix &other) : rows(other.rows), cols(other.cols)
{
    data = other.data;
}

Matrix::Matrix(Matrix &&other) noexcept : rows(other.rows), cols(other.cols)
{
    data = std::move(other.data);
    other.rows = 0;
    other.cols = 0;
}


Matrix& Matrix::operator=(const Matrix &other)
{
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix &&other) noexcept
{
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = std::move(other.data);
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}
