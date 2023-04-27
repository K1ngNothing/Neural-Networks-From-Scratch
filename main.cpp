#include <model.h>
#include <iostream>

using namespace model;
using namespace model::sigma_functions;
using namespace model::dist_functions;

std::vector<TrainingPair> generate_data_set(size_t vec_size, size_t set_size) {
    // Generates set_size pairs {vec, sum(vec)}
    std::vector<TrainingPair> set;
    for (size_t i = 0; i < set_size; i++) {
        Vector vec = Vector::Random(vec_size);
        double sum = 0;
        for (size_t j = 0; j < vec_size; j++) {
            sum += vec(j);
        }
        Vector res{{sum}};
        set.emplace_back(vec, res);
    }
    return set;
}

void example_task() {
    // Task: by given vector return sum of it's coordinates
    size_t input_size = 10;
    size_t output_size = 1;
    size_t cnt_layers = 10;
    Model model(input_size, output_size, cnt_layers, ReLU());

    // Generate data
    size_t total_data_size = 1000;
    size_t training_set_size = total_data_size * 0.8;
    size_t testing_set_size = total_data_size * 0.2;
    std::vector<TrainingPair> training_set = generate_data_set(input_size, training_set_size);
    std::vector<TrainingPair> testing_set = generate_data_set(input_size, testing_set_size);

    // Train model
    auto modifier_func = [](int) {
        static double x = 1;
        x *= 0.995;
        return x;
    };
    double last_train_err =
        model.Train(training_set, EuklideanSquaredDist(), std::function(modifier_func));
    std::cout << "Error in the last trainging step: " << last_train_err << "\n";

    // Check effectiveness
    double avg_error = model.AssessEffectiveness(testing_set, EuklideanSquaredDist());
    std::cout << "Average error on testing_set: " << avg_error << "\n";
}

int main() {
    example_task();
    return 0;
}