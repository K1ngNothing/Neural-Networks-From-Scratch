#pragma once

#include <definitions.h>

#include <memory>

namespace model {

class ActivationFunction {
private:
    class InnerBase {
    public:
        virtual Vector operator()(const Vector& x) const = 0;
        virtual Matrix operator[](const Vector& x) const = 0;
        virtual std::unique_ptr<InnerBase> Clone() const = 0;

    public:
        virtual ~InnerBase() {
        }
    };

    template <typename T>
    class Inner : public InnerBase {
    public:
        Inner(const T& activation_function) : activation_function_(activation_function) {
        }
        Inner(T&& activation_function) : activation_function_(std::move(activation_function)) {
        }

    public:
        Vector operator()(const Vector& x) const override {
            return activation_function_(x);
        }
        Matrix operator[](const Vector& x) const override {
            return activation_function_[x];
        }
        std::unique_ptr<InnerBase> Clone() const override {
            return std::make_unique<Inner>(activation_function_);
        }

    private:
        T activation_function_;
    };

public:
    template <typename T>
    ActivationFunction(T activation_function)
        : ptr_(std::make_unique<Inner<T>>(std::move(activation_function))) {
    }
    ActivationFunction(const ActivationFunction& other) : ptr_(other.ptr_->Clone()) {
    }
    ActivationFunction(ActivationFunction&& other) : ptr_(std::move(other.ptr_)) {
    }

    Vector operator()(const Vector& x) const {  // f(x)
        return (*ptr_)(x);
    }
    Matrix operator[](const Vector& x) const {  // f'(x)
        return (*ptr_)[x];
    }

private:
    std::unique_ptr<InnerBase> ptr_;
};

}  // namespace model
