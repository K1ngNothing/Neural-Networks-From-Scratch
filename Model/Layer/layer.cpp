#include <model.h>

using namespace model;

Model::Layer::Layer(size_t m, size_t n) {
    A_ = Matrix::Random(m, n);
    b_ = Vector::Random(m);

    delta_A_ = Matrix::Zero(m, n);
    delta_b_ = Vector::Zero(m);

    sigma_ = sigma_functions::Sigmoid();
}

Vector Model::Layer::PushVector(Vector x) {
    last_input_ = x;
    return sigma_(A_ * x + b_);
}

RowVector Model::Layer::PushGradient(RowVector u) {
    return u * sigma_[A_ * last_input_ + b_] * A_;
}

void Model::Layer::UpdateDelta(RowVector u, double modifier) {
    Matrix d_sigma = sigma_[A_ * last_input_ + b_];

    Matrix grad_A = (last_input_ * u * d_sigma).transpose();
    Vector grad_b = (u * d_sigma).transpose();

    delta_A_ -= grad_A * modifier;
    delta_b_ -= grad_b * modifier;
}

void Model::Layer::ApplyChanges() {
    A_ += delta_A_;
    b_ += delta_b_;

    delta_A_ = Matrix::Zero(A_.rows(), A_.cols());
    delta_b_ = Vector::Zero(b_.size());
}