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
