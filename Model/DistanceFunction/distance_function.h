#pragma once

#include <memory>

#include <definitions.h>

namespace model {
namespace dist_functions {

class DistanceFunction {
private:
    class InnerBase {
    public:
        virtual ~InnerBase() {
        }
        virtual double operator()(Vector x, Vector y) = 0;
        virtual Vector GetGradientX(Vector x, Vector y) = 0;
    };

    template <typename T>
    class Inner : public InnerBase {
    public:
        Inner(T value) {
            value_ = std::move(value);
        }
        double operator()(Vector x, Vector y) {
            return value_(x, y);
        }
        Vector GetGradientX(Vector x, Vector y) {
            return value_.GetGradientX(x, y);
        }

    private:
        T value_;
    };

public:
    template <typename T>
    DistanceFunction(T value) {
        ptr_ = std::make_unique<Inner<T>>(std::move(value));
    }
    template <typename T>
    DistanceFunction& operator=(T value) {
        ptr_ = std::make_unique<Inner<T>>(std::move(value));
        return *this;
    }
    double operator()(Vector x, Vector y) {
        return (*ptr_)(x, y);
    }
    Vector GetGradientX(Vector x, Vector y) {
        return ptr_->GetGradientX(x, y);
    }

private:
    std::shared_ptr<InnerBase> ptr_;
};

/// Pre-defined DistanceFunctions

class EuklideanSquaredDist {
public:
    double operator()(Vector x, Vector y);
    Vector GetGradientX(Vector x, Vector y);
};

}  // namespace dist_functions
}  // namespace model
