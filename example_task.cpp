#include "example_task.h"

#include <ActivationFunction/predefined.h>
#include <LossFunction/predefined.h>
#include <rng.h>

#include <EigenRand>
#include <iostream>

using namespace model;

namespace {

std::vector<TrainingPair> generate_data_set(size_t vector_size, size_t set_size) {
    // Generates set_size pairs {vec, sum(vec)}
    std::vector<TrainingPair> set;
    set.reserve(set_size);
    for (size_t i = 0; i < set_size; i++) {
        Vector vector = Eigen::Rand::normal<model::Vector>(vector_size, 1, GetRNG());
        Vector answer{{vector.sum()}};
        set.push_back({std::move(vector), std::move(answer)});
    }
    return set;
}

}  // namespace

void example_task() {
    // Task: by given vector return sum of it's coordinates
    size_t input_size = 4;
    size_t output_size = 1;
    size_t hidden_layer_size = 2;
    Model model({input_size, hidden_layer_size, output_size}, {ReLU(), Lineral()});

    // Generate data
    std::vector<TrainingPair> training_set = generate_data_set(input_size, 100);
    std::vector<TrainingPair> testing_set = generate_data_set(input_size, 1000);

    // Train model
    size_t epoch_count = 100;
    double stop_threshold = 1e-6;
    size_t batch_size = 10;
    double starting_learning_rate = 0.1;
    double learning_rate_decay = 0.01;

    double training_set_loss = model.Train(training_set, epoch_count, stop_threshold, batch_size,
                                           starting_learning_rate, learning_rate_decay, MSE());
    std::cout << "Average loss on training set: " << training_set_loss << "\n";

    // Test model
    double testing_set_loss = model.GetAverageLoss(testing_set, MSE());
    std::cout << "Average loss on testing set: " << testing_set_loss << "\n";

    // Examples of predictions
    for (size_t i = 0; i < 3; i++) {
        const Vector& input = testing_set[i].input;
        const Vector& answer = testing_set[i].output;
        Vector prediction = model.Predict(input);

        std::cout << "vector: (" << input.transpose() << "), prediction: " << prediction
                  << ", answer: " << answer << "\n";
    }
}
