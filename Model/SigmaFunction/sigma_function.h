#pragma once

#include <memory>

#include <definitions.h>

namespace model {
namespace sigma_functions {

class SigmaFunction {
private:
    class InnerBase {
    public:
        virtual ~InnerBase() {
        }
        virtual Vector operator()(Vector x) = 0;
        virtual Matrix operator[](Vector x) = 0;
    };

    template <typename T>
    class Inner : public InnerBase {
    public:
        Inner(T value) {
            value_ = std::move(value);
        }
        Vector operator()(Vector x) override {
            return value_(x);
        }
        Matrix operator[](Vector x) override {
            return value_[x];
        }

    private:
        T value_;
    };

public:
    template <typename T>
    SigmaFunction(T value) {
        ptr_ = std::make_unique<Inner<T>>(std::move(value));
    }
    template <typename T>
    SigmaFunction& operator=(T value) {
        ptr_ = std::make_unique<Inner<T>>(std::move(value));
        return *this;
    }
    Vector operator()(Vector x) {  // Sigma(x)
        return (*ptr_)(x);
    }
    Matrix operator[](Vector x) {  // Sigma'(x)
        return (*ptr_)[x];
    }

private:
    std::unique_ptr<InnerBase> ptr_;
};

/// Pre-defined SimaFunctions

class Sigmoid {
public:
    Vector operator()(Vector x);  
    Matrix operator[](Vector x);  

private:
    double calc_one_coordinate(double x);
    double calc_one_derivative(double x);
};

class ReLU {
public:
    Vector operator()(Vector x);
    Matrix operator[](Vector x);

private:
    double calc_one_coordinate(double x);
    double calc_one_derivative(double x);
};

}  // namespace sigma_functions
}  // namespace model
