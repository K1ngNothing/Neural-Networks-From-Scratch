#pragma once

#include <definitions.h>

#include <memory>

namespace model {

class ActivationFunction {
private:
    class InnerBase {
    public:
        virtual ~InnerBase() {
        }
        virtual std::unique_ptr<InnerBase> Clone() const = 0;
        virtual Vector operator()(const Vector& x) const = 0;
        virtual Matrix operator[](const Vector& x) const = 0;
    };

    template <typename T>
    class Inner : public InnerBase {
    public:
        Inner(const T& other) : activation_function_(other) {
        }
        Inner(T&& other) : activation_function_(std::move(other)) {
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

    Vector operator()(const Vector& x) const {  // Sigma(x)
        return (*ptr_)(x);
    }
    Matrix operator[](const Vector& x) const {  // Sigma'(x)
        return (*ptr_)[x];
    }

private:
    std::unique_ptr<InnerBase> ptr_;
};

/// Pre-defined SimaFunctions

class Sigmoid {
public:
    Vector operator()(const Vector& x) const;
    Matrix operator[](const Vector& x) const;

private:
    double calc_one_coordinate(double x) const;
    double calc_one_derivative(double x) const;
};

class ReLU {
public:
    Vector operator()(const Vector& x) const;
    Matrix operator[](const Vector& x) const;

private:
    double calc_one_coordinate(double x) const;
    double calc_one_derivative(double x) const;
};

class Lineral {
public:
    Vector operator()(const Vector& x) const {
        return x;
    }
    Matrix operator[](const Vector& x) const {
        return Vector::Ones(x.size());
    }
};

}  // namespace model
