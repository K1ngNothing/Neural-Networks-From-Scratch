#include "predefined.h"

using namespace model;

double CrossEntropy::operator()(const Vector& x, const Vector& y) const {
    int answer = static_cast<int>(y(0));
    assert(y.size() == 1 && answer >= 0 && x(answer) > 0 && "CrossEntropy::operator()");
    return -log(x(answer));
}

Vector CrossEntropy::GetGradientX(const Vector& x, const Vector& y) const {
    int answer = static_cast<int>(y(0));
    assert(y.size() == 1 && answer >= 0 && x(answer) > 0 && "CrossEntropy::GetGradientX");

    Vector result = Vector::Zero(x.size());
    result(answer) = -1 / x(answer);
    return result;
}
