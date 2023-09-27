#pragma once

#include "activation_function.h"

namespace model {
namespace impl {

Vector ApplyCoordinateWise(const Vector& x, double (*function)(double));
Matrix GetJacobianMatrix(const Vector& x, double (*function)(double));

namespace calculate_one_coordinate {

inline double Sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}
inline double ReLU(double x) {
    return (x > 0 ? x : 0);
}

}  // namespace calculate_one_coordinate

namespace calculate_one_derivative {

inline double Sigmoid(double x) {
    return std::exp(-x) / ((1 + std::exp(-x)) * (1 + std::exp(-x)));
}
inline double ReLU(double x) {
    return (x > 0 ? 1 : 0);
}

}  // namespace calculate_one_derivative

}  // namespace impl

class Sigmoid {
public:
    Vector operator()(const Vector& x) const {
        return impl::ApplyCoordinateWise(x, &impl::calculate_one_coordinate::Sigmoid);
    }
    Matrix operator[](const Vector& x) const {
        return impl::GetJacobianMatrix(x, &impl::calculate_one_derivative::Sigmoid);
    }
    AFType GetType() const {
        return type_;
    }

private:
    const static AFType type_ = AFType::Sigmoid;
};

class ReLU {
public:
    Vector operator()(const Vector& x) const {
        return impl::ApplyCoordinateWise(x, &impl::calculate_one_coordinate::ReLU);
    }
    Matrix operator[](const Vector& x) const {
        return impl::GetJacobianMatrix(x, &impl::calculate_one_derivative::ReLU);
    }
    AFType GetType() const {
        return type_;
    }

private:
    const static AFType type_ = AFType::ReLU;
};

class Linear {
public:
    Vector operator()(const Vector& x) const {
        return x;
    }
    Matrix operator[](const Vector& x) const {
        return Matrix::Identity(x.size(), x.size());
    }
    AFType GetType() const {
        return type_;
    }

private:
    const static AFType type_ = AFType::Linear;
};

class SoftMax {
public:
    Vector operator()(const Vector& x) const;
    Matrix operator[](const Vector& x) const;
    AFType GetType() const {
        return type_;
    }

private:
    const static AFType type_ = AFType::SoftMax;
};

}  // namespace model
