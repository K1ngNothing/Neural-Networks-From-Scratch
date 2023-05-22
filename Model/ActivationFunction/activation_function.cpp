#include "activation_function.h"

#include <stdexcept>

#include "predefined.h"

namespace model {
ActivationFunction AFFabric(AFType type) {
    switch (type) {
        case AFType::Sigmoid:
            return Sigmoid();
        case AFType::ReLU:
            return ReLU();
        case AFType::Linear:
            return Linear();
        case AFType::SoftMax:
            return SoftMax();
        default:
            throw std::runtime_error("AFFabric: incorrect AFType");
    }
}
}  // namespace model
