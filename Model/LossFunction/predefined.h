#pragma once

#include "loss_function.h"

namespace model {

class MSE {
public:
    double operator()(const Vector& x, const Vector& y) const {
        return (x - y).squaredNorm() / x.size();
    }
    Vector GetGradientX(const Vector& x, const Vector& y) const {
        return 2 * (x - y) / x.size();
    }
};

}  // namespace model
