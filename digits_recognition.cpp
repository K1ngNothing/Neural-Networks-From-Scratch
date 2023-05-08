#include "digits_recognition.h"
#include <model.h>

#include <fcntl.h>
#include <iostream>
#include <byteswap.h>

namespace {

int32_t Read_int32_t(int fd) {
    int32_t number = 0;
    read(fd, &number, sizeof(number));

    // read number is high-endian, so we need to flip bytes
    number = bswap_32(number);
    return number;
}

model::Vector ReadOneImage(int fd, int32_t image_size) {
    model::Vector image(image_size);
    for (size_t i = 0; i < image_size; i++) {
        unsigned char pixel = 0;
        read(fd, &pixel, sizeof(pixel));
        image(i) = pixel;
    }
    return image;
}

std::vector<model::Vector> ReadImageFile(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("ReadImageFile open error");
        return {};
    }

    int32_t magic_number = Read_int32_t(fd);
    assert(magic_number == 2051 && "This is not an image file");
    int32_t number_of_images = Read_int32_t(fd);
    int32_t rows = Read_int32_t(fd);
    int32_t columns = Read_int32_t(fd);
    int32_t image_size = rows * columns;


    std::vector<model::Vector> images;
    images.reserve(number_of_images);
    for (size_t i = 0; i < number_of_images; i++) {
        images.push_back(std::move(ReadOneImage(fd, image_size)));
    }
    return images;
}

}

void DigitsRecognition() {
    ReadImageFile("../MNISTDatabase/train-images");
}
