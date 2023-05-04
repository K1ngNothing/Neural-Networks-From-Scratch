#include "model.h"

#include <iostream>
#include <ranges>

using namespace model;

Model::Model(const std::vector<size_t>& layer_sizes,
             const std::vector<ActivationFunction>& layer_activation_functions) {
    assert(layer_sizes.size() == layer_activation_functions.size() + 1);
    size_t layers_count = layer_activation_functions.size();
    layers_.reserve(layers_count);
    for (size_t i = 0; i < layers_count; i++) {
        layers_.emplace_back(layer_sizes[i], layer_sizes[i + 1], layer_activation_functions[i]);
    }
}

double Model::Train(const std::vector<TrainingPair>& training_data, size_t epoch_count,
                    size_t batch_size, double stop_threshold,
                    const PenaltyFunction& penalty_function,
                    std::function<double(int)> learning_rate_function) {
    double average_error_on_last_epoch = INFINITY;
    for (size_t epoch = 0; epoch < epoch_count; epoch++) {
        double current_average_error = 0;
        for (size_t i = 0; i < training_data.size(); i++) {
            current_average_error +=
                TrainOnePair(training_data[i], penalty_function, learning_rate_function(epoch));
            if (i % batch_size == 0) {
                ApplyDeltas();
            }
        }
        ApplyDeltas();
        current_average_error /= training_data.size();
        if (epoch % 100 == 0) {
            std::cout << "epoch: " << epoch << " error: " << current_average_error
                      << " rate: " << learning_rate_function(epoch) << std::endl;
        }

        if (average_error_on_last_epoch - current_average_error < stop_threshold) {
            break;
        }
        average_error_on_last_epoch = current_average_error;
    }
    return average_error_on_last_epoch;
}

Vector Model::Predict(const Vector& x) const {
    Vector result = x;
    for (const auto& layer : layers_) {
        result = layer.PushVector(result);
    }
    return result;
}

double Model::GetAverageLoss(const std::vector<TrainingPair>& test_data,
                             const PenaltyFunction& penalty_function) const {
    double total_loss = 0;
    for (const TrainingPair& pair : test_data) {
        Vector prediction = Predict(pair.input);
        total_loss += penalty_function(prediction, pair.output);
    }
    return total_loss / test_data.size();
}

double Model::TrainOnePair(const TrainingPair& training_pair,
                           const PenaltyFunction& penalty_function, double learning_rate) {
    Vector x = training_pair.input;
    for (auto& layer : layers_) {
        x = layer.PushVector(x);
    }

    const Vector& y0 = training_pair.output;
    double result_deviation = penalty_function(x, y0);

    RowVector u = penalty_function.GetGradientX(x, y0);

    for (auto& layer : std::ranges::reverse_view(layers_)) {
        layer.UpdateDelta(u, learning_rate);
        u = layer.PushGradient(u);
    }
    return result_deviation;
}

void Model::ApplyDeltas() {
    for (auto& layer : layers_) {
        layer.ApplyChanges();
    }
}
