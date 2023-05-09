#include "predefined.h"

using namespace model;

Vector model::impl::ApplyCoordinateWise(const Vector& x, std::function<double(double)> function) {
    size_t m = x.size();
    Vector result(m);
    for (size_t i = 0; i < m; i++) {
        result(i) = function(x(i));
    }
    return result;
}

Matrix model::impl::GetJacobianMatrix(const Vector& x, std::function<double(double)> function) {
    size_t m = x.size();
    Matrix result = Matrix::Zero(m, m);
    for (size_t i = 0; i < m; i++) {
        result(i, i) = function(x(i));
    }
    return result;
}

Vector SoftMax::operator()(const Vector& x) const {
    size_t m = x.size();
    double max_coefficient = x.maxCoeff();
    double exp_sum = 0;
    for (size_t i = 0; i < m; i++) {
        exp_sum += exp(x(i) - max_coefficient);
    }
    Vector result(m);
    for (size_t i = 0; i < m; i++) {
        result(i) = exp(x(i) - max_coefficient) / exp_sum;
    }
    return result;
}

Matrix SoftMax::operator[](const Vector& x) const {
    size_t m = x.size();
    double max_coefficient = x.maxCoeff();
    double exp_sum = 0;
    for (size_t i = 0; i < m; i++) {
        exp_sum += exp(x(i) - max_coefficient);
    }
    Matrix result = Matrix::Zero(m, m);
    for (size_t i = 0; i < m; i++) {
        double z_i = x(i) - max_coefficient;
        for (size_t j = 0; j < m; j++) {
            if (i == j) {
                result(i, i) = (exp(z_i) / exp_sum) * (1 - exp(z_i) / exp_sum);
            } else {
                double z_j = x(j) - max_coefficient;
                result(i, j) = (exp(z_i) / exp_sum) * (-exp(z_j) / exp_sum);
            }
        }
    }
    return result;
}
