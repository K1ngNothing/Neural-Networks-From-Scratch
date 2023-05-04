#pragma once

#include <definitions.h>
#include <Layer/layer.h>
#include <PenaltyFunction/penalty_function.h>

#include <vector>

namespace model {

struct TrainingPair {
    Vector input;
    Vector output;

    TrainingPair(Vector x, Vector y) : input(std::move(x)), output(std::move(y)) {
    }
};

class Model {
public:
    Model(const std::vector<size_t>& layer_sizes,
          const std::vector<ActivationFunction>& layer_activation_functions);

public:
    double Train(const std::vector<TrainingPair>& training_data, size_t epoch_count,
                 size_t batch_size, double stop_threshold, const PenaltyFunction& penalty_function,
                 std::function<double(int)> learning_rate_function);

    Vector Predict(const Vector& x) const;

    double GetAverageLoss(const std::vector<TrainingPair>& test_data,
                          const PenaltyFunction& penalty_function) const;

private:
    double TrainOnePair(const TrainingPair& training_pair, const PenaltyFunction& penalty_function,
                        double learning_rate);
    void ApplyDeltas();

private:
    std::vector<Layer> layers_;
};

};  // namespace model
