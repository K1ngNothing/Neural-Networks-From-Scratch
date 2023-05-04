#include <cmath>

#include "activation_function.h"

using namespace model;

Vector Sigmoid::operator()(const Vector& x) const {
    size_t m = x.size();
    Vector result(m);
    for (size_t i = 0; i < m; i++) {
        result(i) = calc_one_coordinate(x(i));
    }
    return result;
}

Matrix Sigmoid::operator[](const Vector& x) const {
    size_t m = x.size();
    Matrix result = Matrix::Zero(m, m);
    for (size_t i = 0; i < m; i++) {
        result(i, i) = calc_one_derivative(x(i));
    }
    return result;
}

double Sigmoid::calc_one_coordinate(double x) const {
    return 1 / (1 + std::exp(-x));
}

double Sigmoid::calc_one_derivative(double x) const {
    return std::exp(-x) / ((1 + std::exp(-x)) * (1 + std::exp(-x)));
}
