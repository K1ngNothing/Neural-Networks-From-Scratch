#include "example_task.h"

#include <ActivationFunction/predefined.h>
#include <LossFunction/predefined.h>
#include <model.h>
#include <rng.h>

#include <EigenRand>
#include <iostream>

namespace {
constexpr size_t INPUT_SIZE = 4;
constexpr size_t OUTPUT_SIZE = 1;
constexpr size_t TRAINING_SET_SIZE = 100;
constexpr size_t TESTING_SET_SIZE = 1000;

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

void TrainModel() {
    size_t hidden_layer_size = 2;
    model::Model model({INPUT_SIZE, hidden_layer_size, OUTPUT_SIZE},
                       {model::ReLU(), model::Lineral()});

    std::vector<model::TrainingPair> training_set = GenerateDataSet(INPUT_SIZE, TRAINING_SET_SIZE);

    size_t epoch_count = 100;
    double stop_threshold = 1e-12;
    size_t batch_size = 10;
    double starting_learning_rate = 0.1;
    double learning_rate_decay = 0.01;

    double training_set_loss =
        model.Train(training_set, epoch_count, stop_threshold, batch_size, starting_learning_rate,
                    learning_rate_decay, model::MSE());
    std::cout << "Average loss on training set: " << training_set_loss << "\n";

    // Save layers in file "layers.txt"
    model.Serialize("example_task_layers.txt");
}

void TestModel() {
    // Read layers from "layers.txt"
    model::Model model("example_task_layers.txt");

    std::vector<model::TrainingPair> testing_set = GenerateDataSet(INPUT_SIZE, TESTING_SET_SIZE);

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

}  // namespace

void ExampleTask() {
    std::cout << "Example task: by vetor return sum of it's coordinates\n";
    TrainModel();
    TestModel();
}
