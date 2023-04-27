#pragma once

#include <vector>

#include "definitions.h"
#include "DistanceFunction/distance_function.h"
#include "SigmaFunction/sigma_function.h"

namespace model {

struct TrainingPair {
    Vector input;
    Vector output;

    TrainingPair(Vector x, Vector y) : input(std::move(x)), output(std::move(y)) {
    }
    TrainingPair(Vector&& x, Vector&& y) : input(std::move(x)), output(std::move(y)) {
    }
};

class Model {
private:
    class Layer {
    public:
        Layer() = default;
        Layer(size_t m, size_t n);

    public:
        Vector PushVector(Vector x);
        RowVector PushGradient(RowVector u);
        void UpdateDelta(RowVector u, double modifier);
        void ApplyChanges();

    private:
        Matrix A_;  // Lineral paramethers of Layer
        Vector b_;
        sigma_functions::SigmaFunction sigma_ =
            sigma_functions::Sigmoid();  // Non-lineral paramether of Layer

        Vector last_input_;

        Matrix delta_A_;
        Vector delta_b_;
    };

public:
    Model(size_t input_size, size_t output_size, size_t cnt_layers,
          sigma_functions::SigmaFunction sigma);

public:
    double Train(std::vector<TrainingPair> training_data,
                 dist_functions::DistanceFunction dist_func,
                 std::function<double(int)> modifier_func);

    std::vector<Vector> PredictBatch(std::vector<Vector> data);
    Vector Predict(Vector x);

    double AssessEffectiveness(std::vector<TrainingPair> test_data,
                               dist_functions::DistanceFunction dist);

private:
    double TrainOnePair(TrainingPair training_pair, dist_functions::DistanceFunction dist,
                        double modifier);
    void ApplyDeltas();

private:
    size_t layers_count_;
    std::vector<Layer> layers_;
};

};  // namespace model