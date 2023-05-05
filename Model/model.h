#pragma once

#include <definitions.h>
#include <Layer/layer.h>
#include <LossFunction/loss_function.h>

#include <vector>

namespace model {

struct TrainingPair {
    Vector input;
    Vector output;
};

class Model {
public:
    Model(const std::initializer_list<size_t>& layer_sizes,
          const std::initializer_list<ActivationFunction>& layer_activation_functions);

public:
    double Train(const std::vector<TrainingPair>& training_data, size_t epoch_count,
                 double stop_threshold, size_t batch_size, double starting_learning_rate,
                 double learning_rate_decay, const LossFunction& loss_function);

    Vector Predict(const Vector& x) const;

    double GetAverageLoss(const std::vector<TrainingPair>& test_data,
                          const LossFunction& loss_function) const;

private:
    double TrainOnePair(const TrainingPair& training_pair, const LossFunction& loss_function,
                        double learning_rate);
    void ApplyDeltas();

private:
    std::vector<impl::Layer> layers_;
};

};  // namespace model
