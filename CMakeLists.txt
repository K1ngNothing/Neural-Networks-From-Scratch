cmake_minimum_required(VERSION 3.13)

project(NeuralNetworkFromScratch 
        VERSION 1.0 
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(Eigen)
include_directories(Model)

add_executable(NeuralNetwork 
               main.cpp
               Model/model.cpp
               Model/DistanceFunction/euklidean_squared_distance.cpp
               Model/Layer/layer.cpp
               Model/SigmaFunction/sigmoid.cpp
               Model/SigmaFunction/relu.cpp
               )