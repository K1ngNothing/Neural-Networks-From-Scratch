#include "model.h"

#include <iostream>
#include <ranges>

using namespace model;

Model::Model(const std::initializer_list<size_t>& layer_sizes,
             const std::initializer_list<ActivationFunction>& layer_activation_functions) {
    assert(layer_sizes.size() == layer_activation_functions.size() + 1);

    size_t layers_count = layer_activation_functions.size();
    layers_.reserve(layers_count);
    for (size_t i = 0; i < layers_count; i++) {
        layers_.emplace_back(*(layer_sizes.begin() + i), *(layer_sizes.begin() + i + 1),
                             *(layer_activation_functions.begin() + i));
    }
}

double Model::Train(const std::vector<TrainingPair>& training_data, size_t epoch_count,
                    double stop_threshold, size_t batch_size, double starting_learning_rate,
                    double learning_rate_decay, const LossFunction& loss_function) {
    double average_loss_on_last_epoch = 0;
    for (size_t epoch = 0; epoch < epoch_count; epoch++) {
        average_loss_on_last_epoch = 0;
        double learning_rate = starting_learning_rate / (1 + learning_rate_decay * epoch);

        for (size_t pair_id = 0; pair_id < training_data.size(); pair_id++) {
            const TrainingPair& training_pair = training_data[pair_id];
            average_loss_on_last_epoch += TrainOnePair(training_pair, loss_function, learning_rate);
            if (pair_id % batch_size == 0) {
                ApplyDeltas();
            }
        }
        ApplyDeltas();
        average_loss_on_last_epoch /= training_data.size();
        if (epoch % 5 == 0) {  // TODO: verbose option
            std::cout << "epoch: " << epoch << ", loss: " << average_loss_on_last_epoch
                      << ", learning rate: " << learning_rate << std::endl;
        }

        if (average_loss_on_last_epoch < stop_threshold) {
            std::cout << "stopped at epoch " << epoch << std::endl;
            break;
        }
    }
    return average_loss_on_last_epoch;
}

Vector Model::Predict(const Vector& x) const {
    Vector result = x;
    for (const auto& layer : layers_) {
        result = layer.PushVector(result);
    }
    return result;
}

double Model::GetAverageLoss(const std::vector<TrainingPair>& test_data,
                             const LossFunction& loss_function) const {
    double total_loss = 0;
    for (const TrainingPair& pair : test_data) {
        Vector prediction = Predict(pair.input);
        total_loss += loss_function(prediction, pair.output);
    }
    return total_loss / test_data.size();
}

double Model::TrainOnePair(const TrainingPair& training_pair, const LossFunction& loss_function,
                           double learning_rate) {
    Vector x = training_pair.input;
    for (auto& layer : layers_) {
        x = layer.PushVector(x);
    }

    const Vector& y0 = training_pair.output;
    double loss = loss_function(x, y0);

    RowVector u = loss_function.GetGradientX(x, y0);

    for (auto& layer : std::ranges::reverse_view(layers_)) {
        layer.UpdateDelta(u, learning_rate);
        u = layer.PushGradient(u);
    }
    return loss;
}

void Model::ApplyDeltas() {
    for (auto& layer : layers_) {
        layer.ApplyChanges();
    }
}
