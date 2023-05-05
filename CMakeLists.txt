cmake_minimum_required(VERSION 3.13)

project(NeuralNetworkFromScratch 
        VERSION 1.0 
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(Eigen)
include_directories(EigenRand/EigenRand)
include_directories(Model)

add_executable(NeuralNetwork 
               example_task.cpp
               main.cpp
               Model/model.cpp
               Model/ActivationFunction/predefined.cpp
               Model/Layer/layer.cpp
               )
