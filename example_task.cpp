#include "example_task.h"

#include <ActivationFunction/predefined.h>
#include <LossFunction/predefined.h>
#include <rng.h>

#include <EigenRand>
#include <iostream>

namespace {

std::vector<model::TrainingPair> GenerateDataSet(size_t vector_size, size_t set_size) {
    // Generates set_size pairs {vec, sum(vec)}
    std::vector<model::TrainingPair> set;
    set.reserve(set_size);
    for (size_t i = 0; i < set_size; i++) {
        model::Vector vector = Eigen::Rand::normal<model::Vector>(vector_size, 1, model::GetRNG());
        model::Vector answer{{vector.sum()}};
        set.push_back({std::move(vector), std::move(answer)});
    }
    return set;
}

}  // namespace

void ExampleTask() {
    // Task: by given vector return sum of it's coordinates
    size_t input_size = 4;
    size_t output_size = 1;
    size_t hidden_layer_size = 2;
    model::Model model({input_size, hidden_layer_size, output_size},
                       {model::ReLU(), model::Lineral()});

    // Generate data
    std::vector<model::TrainingPair> training_set = GenerateDataSet(input_size, 100);
    std::vector<model::TrainingPair> testing_set = GenerateDataSet(input_size, 1000);

    // Train model
    size_t epoch_count = 1000;
    double stop_threshold = 1e-12;
    size_t batch_size = 10;
    double starting_learning_rate = 0.1;
    double learning_rate_decay = 0.01;

    double training_set_loss =
        model.Train(training_set, epoch_count, stop_threshold, batch_size, starting_learning_rate,
                    learning_rate_decay, model::MSE());
    std::cout << "Average loss on training set: " << training_set_loss << "\n";

    // Test model
    double testing_set_loss = model.GetAverageLoss(testing_set, model::MSE());
    std::cout << "Average loss on testing set: " << testing_set_loss << "\n";

    // Examples of predictions
    for (size_t i = 0; i < 3; i++) {
        const model::Vector& input = testing_set[i].input;
        const model::Vector& answer = testing_set[i].output;
        model::Vector prediction = model.Predict(input);

        std::cout << "vector: (" << input.transpose() << "), prediction: " << prediction
                  << ", answer: " << answer << "\n";
    }
}
