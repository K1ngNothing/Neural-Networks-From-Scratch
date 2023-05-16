#include "digits_recognition.h"

#include <ActivationFunction/predefined.h>
#include <byteswap.h>
#include <LossFunction/predefined.h>
#include <model.h>

#include <fstream>
#include <iostream>
#include <string>

namespace {

int32_t Read_int32_t(std::ifstream& ifs) {
    int32_t number = 0;
    ifs.read(reinterpret_cast<char*>(&number), sizeof(number));

    // read number is high-endian, so we need to flip bytes
    number = bswap_32(number);
    return number;
}

model::Vector ReadOneImage(std::ifstream& images_ifs, int32_t image_size) {
    model::Vector image(image_size);
    for (size_t i = 0; i < image_size; i++) {
        unsigned char pixel = 0;
        images_ifs.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
        image(i) = static_cast<double>(pixel);
    }
    return image;
}

model::Vector ReadOneLabel(std::ifstream& labels_ifs) {
    unsigned char label = 0;
    labels_ifs.read(reinterpret_cast<char*>(&label), sizeof(label));
    return model::Vector{{static_cast<double>(label)}};
}

void ReadMetaInformation(std::ifstream& images_ifs, std::ifstream& labels_ifs,
                         int32_t& number_of_items, int32_t& image_size) {
    int32_t images_magic_number = Read_int32_t(images_ifs);
    assert(images_magic_number == 2051 && "This is not a file with images");
    int32_t number_of_images = Read_int32_t(images_ifs);
    int32_t rows = Read_int32_t(images_ifs);
    int32_t columns = Read_int32_t(images_ifs);

    int32_t labels_magic_number = Read_int32_t(labels_ifs);
    assert(labels_magic_number == 2049 && "This is not a file with labels");
    int32_t number_of_labels = Read_int32_t(labels_ifs);

    assert(number_of_images == number_of_labels &&
           "Files with images and labels are not consistent with each other");
    number_of_items = number_of_images;
    image_size = rows * columns;
}

std::vector<model::TrainingPair> ReadImagesAndLabels(const std::string& path_to_images,
                                                     const std::string& path_to_labels,
                                                     ssize_t read_count = -1) {  // -1 = read all
    std::ifstream images_ifs(path_to_images, std::ios::binary);
    std::ifstream labels_ifs(path_to_labels, std::ios::binary);

    int32_t number_of_items, image_size;
    ReadMetaInformation(images_ifs, labels_ifs, number_of_items, image_size);
    if (read_count >= 0) {
        number_of_items = read_count;
    }

    std::vector<model::TrainingPair> data_set;
    data_set.reserve(number_of_items);
    for (size_t i = 0; i < number_of_items; i++) {
        data_set.push_back({ReadOneImage(images_ifs, image_size), ReadOneLabel(labels_ifs)});
    }
    return data_set;
}

int GetPrediction(const model::Vector& assurance_in_answer) {
    int prediction = -1;
    assurance_in_answer.maxCoeff(&prediction);
    return prediction;
}

constexpr size_t INPUT_SIZE = 784;
constexpr size_t OUTPUT_SIZE = 10;

void TrainModel() {
    std::vector<model::TrainingPair> training_set =
        ReadImagesAndLabels("../MNISTDatabase/test-images", "../MNISTDatabase/test-labels");

    size_t hidden_layer_size = 200;
    model::Model model({INPUT_SIZE, hidden_layer_size, OUTPUT_SIZE},
                       {model::ReLU(), model::SoftMax()});

    size_t epoch_count = 1000;
    double stop_threshold = 0.1;
    size_t batch_size = 10;
    double starting_learning_rate = 0.001;
    double learning_rate_decay = 0.01;

    double training_set_loss =
        model.Train(training_set, epoch_count, stop_threshold, batch_size, starting_learning_rate,
                    learning_rate_decay, model::CrossEntropy());
    std::cout << "Average loss on training set: " << training_set_loss << "\n";
    std::cout << "Accuracy on training set: " << model.GetAccuracy(training_set) << std::endl;

    model.Serialize("digits_recognition_layers.txt");
}

void TestModel() {
    model::Model model("digits_recognition_layers.txt", {model::ReLU(), model::SoftMax()});

    std::vector<model::TrainingPair> testing_set =
        ReadImagesAndLabels("../MNISTDatabase/train-images", "../MNISTDatabase/train-labels", 1000);

    double testing_set_loss = model.GetAverageLoss(testing_set, model::CrossEntropy());
    std::cout << "Average loss on testing set: " << testing_set_loss << "\n";
    std::cout << "Accuracy on testing set: " << model.GetAccuracy(testing_set) << std::endl;

    // Examples of predictions
    for (size_t i = 0; i < 3; i++) {
        const model::Vector& input = testing_set[i].input;
        const model::Vector& answer = testing_set[i].output;
        model::Vector assurance_in_answer = model.Predict(input);

        int prediction = GetPrediction(assurance_in_answer);

        std::cout << "model's assurance in answer:\n"
                  << assurance_in_answer << "\nprediction: " << prediction << ", answer: " << answer
                  << "\n";
    }
}

}  // namespace

void DigitsRecognition() {
    std::cout << "Digits recognition:\n";
    TrainModel();
    TestModel();
}
