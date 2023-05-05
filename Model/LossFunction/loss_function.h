#pragma once

#include <definitions.h>

#include <memory>

namespace model {

class LossFunction {
private:
    class InnerBase {
    public:
        virtual double operator()(const Vector& x, const Vector& y) const = 0;
        virtual Vector GetGradientX(const Vector& x, const Vector& y) const = 0;
        virtual std::unique_ptr<InnerBase> Clone() const = 0;

    public:
        virtual ~InnerBase() {
        }
    };

    template <typename T>
    class Inner : public InnerBase {
    public:
        Inner(const T& loss_function) : loss_function_(loss_function) {
        }
        Inner(T&& loss_function) : loss_function_(std::move(loss_function)) {
        }

    public:
        double operator()(const Vector& x, const Vector& y) const override {
            return loss_function_(x, y);
        }
        Vector GetGradientX(const Vector& x, const Vector& y) const override {
            return loss_function_.GetGradientX(x, y);
        }
        std::unique_ptr<InnerBase> Clone() const override {
            return std::make_unique<Inner>(loss_function_);
        }

    private:
        T loss_function_;
    };

public:
    template <typename T>
    LossFunction(T loss_function) : ptr_(std::make_unique<Inner<T>>(std::move(loss_function))) {
    }
    LossFunction(const LossFunction& other) : ptr_(other.ptr_->Clone()) {
    }
    LossFunction(LossFunction&& other) : ptr_(std::move(other.ptr_)) {
    }

    double operator()(const Vector& x, const Vector& y) const {
        return (*ptr_)(x, y);
    }
    Vector GetGradientX(const Vector& x, const Vector& y) const {
        return ptr_->GetGradientX(x, y);
    }

private:
    std::unique_ptr<InnerBase> ptr_;
};

}  // namespace model
