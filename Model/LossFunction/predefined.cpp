#include "predefined.h"

namespace model {

namespace {

bool CheckCrossEntropyParameters(const Vector& x, const Vector& y) {
    int answer = static_cast<int>(y(0));
    return (y.size() == 1 && answer == y(0) && answer >= 0 && answer < x.size() && x(answer) > 0);
}

}  // namespace

double CrossEntropy::operator()(const Vector& x, const Vector& y) const {
    int answer = static_cast<int>(y(0));
    assert(CheckCrossEntropyParameters(x, y) && "CrossEntropy::operator()");
    return -log(x(answer));
}

Vector CrossEntropy::GetGradientX(const Vector& x, const Vector& y) const {
    int answer = static_cast<int>(y(0));
    assert(CheckCrossEntropyParameters(x, y) && "CrossEntropy::GetGradientX");

    Vector result = Vector::Zero(x.size());
    result(answer) = -1 / x(answer);
    return result;
}

}  // namespace model
