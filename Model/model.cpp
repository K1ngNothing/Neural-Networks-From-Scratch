#include "model.h"

using namespace model;

Model::Model(size_t input_size, size_t output_size, size_t cnt_layers,
        sigma_functions::SigmaFunction sigma)
    : layers_count_(cnt_layers), layers_(cnt_layers) {
    // The first (input_size - 1) layers use (input_size x input_size) matrix A_,
    // the last layer -- (output_size x input_size)
    assert(cnt_layers > 0 && "Model constructor: count of layers should be at least 1");
    for (size_t i = 0; i < cnt_layers - 1; i++) {
        layers_[i] = std::move(Layer(input_size, input_size));
    }
    layers_.back() = std::move(Layer(output_size, input_size));
}

double Model::TrainOnePair(TrainingPair training_pair, dist_functions::DistanceFunction dist,
                           double modifier) {
    Vector x = training_pair.input;
    for (size_t i = 0; i < layers_count_; i++) {
        x = layers_[i].PushVector(x);
    }

    Vector y0 = training_pair.output;
    double result_deviation = dist(x, y0);

    RowVector u = dist.GetGradientX(x, y0);
    for (ssize_t i = layers_count_ - 1; i >= 0; i--) {
        layers_[i].UpdateDelta(u, modifier);
        Vector new_grad = layers_[i].PushGradient(u);
        u = new_grad;
    }
    return result_deviation;
}

void Model::ApplyDeltas() {
    for (size_t i = 0; i < layers_count_; i++) {
        layers_[i].ApplyChanges();
    }
}

double Model::Train(std::vector<TrainingPair> training_data,
                    dist_functions::DistanceFunction dist_func,
                    std::function<double (int)> modifier_func) {
    double result = 0;  // an error on last training_pair
    size_t i = 0;
    for (const TrainingPair& pair : training_data) {
        result = TrainOnePair(pair, dist_func, modifier_func(i));

        // TODO: implement training with batches
        ApplyDeltas();
        i++;
    }
    return result;
}

Vector Model::Predict(Vector x) {
    for (size_t i = 0; i < layers_count_; i++) {
        x = layers_[i].PushVector(x);
    }
    return x;
}

std::vector<Vector> Model::PredictBatch(std::vector<Vector> data) {
    std::vector<Vector> result;
    result.reserve(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result.emplace_back(std::move(Predict(data[i])));
    }
    return result;
}

double Model::AssessEffectiveness(std::vector<TrainingPair> test_data,
                                  dist_functions::DistanceFunction dist) {
    double total_error = 0;
    for (const TrainingPair& pair : test_data) {
        Vector prediction = Predict(pair.input);
        total_error += dist(prediction, pair.output);
    }
    return total_error / test_data.size();
}