#ifndef MATRIX
#define MATRIX

#include <vector>
#include <iostream>
#include <stdexcept>
#include <random>
#include <cmath>
#include <cstdlib>
#include <ctime>

class Matrix {
private:
    // OPTIMISATION: stockage flat au lieu de vector<vector<double>>
    // Améliore la localité de cache et réduit les allocations
    std::vector<double> data;
    size_t rows;
    size_t cols;

public:
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double initValue);
    Matrix(const Matrix &other);
    Matrix(Matrix &&other) noexcept;

    ~Matrix() = default;

    Matrix& operator=(const Matrix &other);
    Matrix& operator=(Matrix &&other) noexcept;

    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix identity(size_t n);

    Matrix add(const Matrix &other) const;
    Matrix subtract(const Matrix &other) const;
    Matrix transpose() const;
    Matrix scalarMultiply(double scalar) const;
    Matrix multiply(const Matrix &other) const;
    Matrix hadamard(const Matrix &other) const;

    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(const Matrix &other) const;
    Matrix& operator+=(const Matrix &other);
    Matrix& operator-=(const Matrix &other);
    Matrix operator*(double scalar) const;
    friend Matrix operator*(double scalar, const Matrix &m);

    bool operator==(const Matrix &other) const;
    bool operator!=(const Matrix &other) const;

    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;

    size_t getRows() const {
        return rows;
    }

    size_t getCols() const {
        return cols;
    }

    double get(size_t i, size_t j) const;
    void set(size_t i, size_t j, double value);

    void fill(double value);
    void randomize(double min = -1.0, double max = 1.0);

    Matrix apply(double (*func)(double)) const;
    Matrix sigmoid() const;
    Matrix relu() const;
    Matrix tanh() const;

    void print() const;
    double sum() const;
    double mean() const;
    double max() const;
    double min() const;
};

#endif
