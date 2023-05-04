#include "example_task.h"

#include <iostream>

using namespace model;

namespace {

std::vector<TrainingPair> generate_data_set(size_t vector_size, size_t set_size) {
    // Generates set_size pairs {vec, sum(vec)}
    std::vector<TrainingPair> set;
    set.reserve(set_size);
    for (size_t i = 0; i < set_size; i++) {
        Vector vector = Vector::Random(vector_size);
        Vector answer{{vector.sum() / vector.size()}};
        set.emplace_back(std::move(vector), std::move(answer));
    }
    return set;
}

}  // namespace

void example_task() {
    // Task: by given vector return sum of it's coordinates
    size_t input_size = 4;
    size_t output_size = 1;
    size_t hidden_layer_size = 4;
    Model model({input_size, hidden_layer_size, output_size}, {ReLU(), Lineral()});

    // Generate data
    std::vector<TrainingPair> training_set = generate_data_set(input_size, 100);
    std::vector<TrainingPair> testing_set = generate_data_set(input_size, 1000);

    // Train model
    size_t epoch_count = 100;
    size_t batch_size = 10;
    auto learning_rate_function = [](size_t epoch) -> double {
        double start_learning_rate = 0.05;
        double decay = 0.005;
        return start_learning_rate / (1 + decay * epoch);
    };
    double threshold = 0;
    double last_epoch_error =
        model.Train(training_set, epoch_count, batch_size, threshold, EuklideanSquaredDist(),
                    std::function(learning_rate_function));
    std::cout << "Average loss on training_set: " << last_epoch_error << "\n";

    // Test model
    double average_error = model.GetAverageLoss(testing_set, EuklideanSquaredDist());
    std::cout << "Average loss on testing_set: " << average_error << "\n";
    for (size_t i = 0; i < 3; i++) {
        const Vector& input = testing_set[i].input;
        const Vector& answer = testing_set[i].output;

        std::cout << "vector: (" << input.transpose() << "), prediction: " << model.Predict(input)
                  << ", answer: " << answer << "\n";
    }
}
