#include "layer.h"

#include <iostream>

using namespace model;

Layer::Layer(size_t input_size, size_t output_size, const ActivationFunction& sigma)
    : A_(Matrix::Random(output_size, input_size)),
      b_(Vector::Zero(output_size)),
      delta_A_(Matrix::Zero(output_size, input_size)),
      delta_b_(Vector::Zero(output_size)),
      sigma_(Sigmoid()) {  // change to sigma
}

Vector Layer::PushVector(const Vector& x) const {
    return sigma_(A_ * x + b_);
}

Vector Layer::PushVector(const Vector& x) {
    last_input_ = x;
    return sigma_(A_ * x + b_);
}

RowVector Layer::PushGradient(const RowVector& u) const {
    return u * sigma_[A_ * last_input_ + b_] * A_;
}

void Layer::UpdateDelta(const RowVector& u, double modifier) {
    Matrix d_sigma = sigma_[A_ * last_input_ + b_];

    Matrix grad_A = (last_input_ * u * d_sigma).transpose();
    Vector grad_b = (u * d_sigma).transpose();

    delta_A_ -= grad_A * modifier;
    delta_b_ -= grad_b * modifier;
}

void Layer::ApplyChanges() {
    // std::cout << "ApplyChanges: max_da: " << delta_A_.maxCoeff() << " max_db: " <<
    // delta_b_.maxCoeff() << std::endl;
    A_ += delta_A_;
    b_ += delta_b_;

    delta_A_ = Matrix::Zero(A_.rows(), A_.cols());
    delta_b_ = Vector::Zero(b_.size());
}
