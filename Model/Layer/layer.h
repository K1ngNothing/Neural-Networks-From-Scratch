#pragma once

#include <ActivationFunction/activation_function.h>
#include <definitions.h>
#include <file_reader.h>

namespace model {
namespace impl {

class Layer {
public:
    Layer(size_t input_size, size_t output_size, const ActivationFunction& sigma);
    Layer(FileReader& file_reader);

    Vector PushVector(const Vector& x);
    Vector ApplyToVector(const Vector& x) const;
    RowVector PushGradient(const RowVector& u) const;
    void UpdateDelta(const RowVector& u, double learning_rate);
    void ApplyChanges();
    void Serialize(FileReader& file_reader) const;

private:
    Matrix A_;  // Linear paramethers of Layer
    Vector b_;
    ActivationFunction sigma_;  // Non-lineral paramether of Layer

    Vector last_input_;

    Matrix delta_A_;
    Vector delta_b_;
    size_t delta_count_ = 0;
};

}  // namespace impl
}  // namespace model
