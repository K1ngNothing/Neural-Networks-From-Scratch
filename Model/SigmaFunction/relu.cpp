#include "sigma_function.h"

using namespace model;
using namespace sigma_functions;

Vector ReLU::operator()(Vector x) {
    size_t m = x.size();
    Vector result(m);
    for (size_t i = 0; i < m; i++) {
        result(i) = calc_one_coordinate(x(i));
    }
    return result;
}

Matrix ReLU::operator[](Vector x) {
    size_t m = x.size();
    Matrix result = Matrix::Zero(m, m);
    for (size_t i = 0; i < m; i++) {
        result(i, i) = calc_one_derivative(x(i));
    }
    return result;
}

double ReLU::calc_one_coordinate(double x) {
    return (x > 0 ? x : 0);
}

double ReLU::calc_one_derivative(double x) {
    return (x > 0 ? 1 : 0);
}