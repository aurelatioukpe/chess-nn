#include "../include/Matrix.hpp"

Matrix Matrix::multiply(const Matrix& other) const
{
    if (cols != other.rows) {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    Matrix result(rows, other.cols, 0.0);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < other.cols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; k++) {
                sum += data[i * cols + k] * other.data[k * other.cols + j];
            }
            result.data[i * result.cols + j] = sum;
        }
    }
    return result;
}

Matrix Matrix::hadamard(const Matrix& other) const
{
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[i * cols + j] = data[i * cols + j] * other.data[i * cols + j];
        }
    }
    return result;
}

void Matrix::randomize(double min, double max)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            data[i * cols + j] = dis(gen);
        }
    }
}

Matrix Matrix::apply(double (*func)(double)) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[i * cols + j] = func(data[i * cols + j]);
        }
    }
    return result;
}

Matrix Matrix::sigmoid() const
{
    return apply([](double x) -> double {
        return 1.0 / (1.0 + std::exp(-x));
    });
}

Matrix Matrix::relu() const
{
    return apply([](double x) -> double {
        return (x > 0.0) ? x : 0.0;
    });
}

Matrix Matrix::tanh() const
{
    return apply([](double x) -> double {
        return std::tanh(x);
    });
}
