#include "penalty_function.h"

using namespace model;

double EuklideanSquaredDist::operator()(const Vector& x, const Vector& y) const {
    return (x - y).squaredNorm();
}

Vector EuklideanSquaredDist::GetGradientX(const Vector& x, const Vector& y) const {
    return 2 * x - 2 * y;
}
