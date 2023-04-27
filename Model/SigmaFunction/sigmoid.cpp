#include <cmath>

#include "sigma_function.h"

using namespace model;
using namespace sigma_functions;

Vector Sigmoid::operator()(Vector x) {
    size_t m = x.size();
    Vector result(m);
    for (size_t i = 0; i < m; i++) {
        result(i) = calc_one_coordinate(x(i));
    }
    return result;
}

Matrix Sigmoid::operator[](Vector x) {
    size_t m = x.size();
    Matrix result = Matrix::Zero(m, m);
    for (size_t i = 0; i < m; i++) {
        result(i, i) = calc_one_derivative(x(i));
    }
    return result;
}

double Sigmoid::calc_one_coordinate(double x) {
    return 1 / (1 + std::exp(-x));
}

double Sigmoid::calc_one_derivative(double x) {
    return std::exp(-x) / ((1 + std::exp(-x)) * (1 + std::exp(-x)));
}