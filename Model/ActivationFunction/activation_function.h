#pragma once

#include <definitions.h>

#include <memory>

namespace model {

enum class AFType {
    Sigmoid,
    ReLU,
    Lineral,
    SoftMax,
};

class ActivationFunction {
private:
    class InnerBase {
    public:
        virtual Vector operator()(const Vector& x) const = 0;
        virtual Matrix operator[](const Vector& x) const = 0;
        virtual AFType GetType() const = 0;
        virtual std::unique_ptr<InnerBase> Clone() const = 0;

        virtual ~InnerBase() = default;
    };

    template <typename T>
    class Inner : public InnerBase {
    public:
        Inner(const T& activation_function) : activation_function_(activation_function) {
        }
        Inner(T&& activation_function) : activation_function_(std::move(activation_function)) {
        }

        Vector operator()(const Vector& x) const override {
            return activation_function_(x);
        }
        Matrix operator[](const Vector& x) const override {
            return activation_function_[x];
        }
        AFType GetType() const override {
            return activation_function_.GetType();
        }
        std::unique_ptr<InnerBase> Clone() const override {
            return std::make_unique<Inner>(activation_function_);
        }

    private:
        T activation_function_;
    };

public:
    ActivationFunction() = default;
    template <typename T>
    ActivationFunction(T activation_function)
        : ptr_(std::make_unique<Inner<T>>(std::move(activation_function))) {
    }
    ActivationFunction(const ActivationFunction& other)
        : ptr_(other.ptr_ == nullptr ? nullptr : other.ptr_->Clone()) {
    }
    ActivationFunction(ActivationFunction&& other) noexcept = default;
    ActivationFunction& operator=(const ActivationFunction& other) {
        return *this = ActivationFunction(other);
    }
    ActivationFunction& operator=(ActivationFunction&& other) noexcept = default;

    Vector operator()(const Vector& x) const {  // f(x)
        return (*ptr_)(x);
    }
    Matrix operator[](const Vector& x) const {  // f'(x)
        return (*ptr_)[x];
    }
    AFType GetType() const {
        return (*ptr_).GetType();
    }

private:
    std::unique_ptr<InnerBase> ptr_;
};

ActivationFunction AFFabric(AFType type);

}  // namespace model
