#include "layer.h"

#include <rng.h>

#include <EigenRand>

using namespace model;
using namespace impl;

Layer::Layer(size_t input_size, size_t output_size, const ActivationFunction& sigma)
    : A_(Eigen::Rand::normal<Matrix>(output_size, input_size, GetRNG()) * 0.01),
      b_(Vector::Zero(output_size)),
      sigma_(sigma),
      delta_A_(Matrix::Zero(output_size, input_size)),
      delta_b_(Vector::Zero(output_size)),
      delta_count_(0) {
}

Vector Layer::PushVector(const Vector& x) {
    last_input_ = x;
    return sigma_(A_ * x + b_);
}

Vector Layer::PushVector(const Vector& x) const {
    return sigma_(A_ * x + b_);
}

RowVector Layer::PushGradient(const RowVector& u) const {
    return u * sigma_[A_ * last_input_ + b_] * A_;
}

void Layer::UpdateDelta(const RowVector& u, double learning_rate) {
    Matrix d_sigma = sigma_[A_ * last_input_ + b_];

    Matrix grad_A = (last_input_ * u * d_sigma).transpose();
    Vector grad_b = (u * d_sigma).transpose();

    delta_A_ -= grad_A * learning_rate;
    delta_b_ -= grad_b * learning_rate;
    delta_count_++;
}

void Layer::ApplyChanges() {
    if (delta_count_) {
        A_ += delta_A_ / delta_count_;
        b_ += delta_b_ / delta_count_;

        delta_A_ = Matrix::Zero(A_.rows(), A_.cols());
        delta_b_ = Vector::Zero(b_.size());
        delta_count_ = 0;
    }
}
