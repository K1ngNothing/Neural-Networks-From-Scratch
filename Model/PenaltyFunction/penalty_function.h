#pragma once

#include <definitions.h>

#include <memory>

namespace model {

class PenaltyFunction {
private:
    class InnerBase {
    public:
        virtual ~InnerBase() {
        }
        virtual double operator()(const Vector& x, const Vector& y) const = 0;
        virtual Vector GetGradientX(const Vector& x, const Vector& y) const = 0;
    };

    template <typename T>
    class Inner : public InnerBase {
    public:
        Inner(T&& value) : penalty_function_(std::move(value)) {
        }
        double operator()(const Vector& x, const Vector& y) const override {
            return penalty_function_(x, y);
        }
        Vector GetGradientX(const Vector& x, const Vector& y) const override {
            return penalty_function_.GetGradientX(x, y);
        }

    private:
        T penalty_function_;
    };

public:
    template <typename T>
    PenaltyFunction(T penalty_function)
        : ptr_(std::make_unique<Inner<T>>(std::move(penalty_function))) {
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

/// Pre-defined DistanceFunctions

class EuklideanSquaredDist {
public:
    double operator()(const Vector& x, const Vector& y) const;
    Vector GetGradientX(const Vector& x, const Vector& y) const;
};

}  // namespace model
