#include "../include/NeuralNetwork.hpp"
#include "../include/FENEncoder.hpp"
#include <iostream>

int main() {
    std::cout << "Creating a basic neural network for chess analysis...\n";
    
    // Architecture: 
    // Input: FENEncoder::getInputSize() (774 neurons)
    // Hidden1: 256 neurons (ReLU)
    // Hidden2: 128 neurons (ReLU)
    // Hidden3: 64 neurons (ReLU)
    // Output: 3 neurons (Softmax) - Nothing, Check, Checkmate
    
    int inputSize = FENEncoder::getInputSize();
    std::vector<size_t> layers = {(size_t)inputSize, 256, 128, 64, 3};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::SOFTMAX
    };
    
    NeuralNetwork network(layers, activations, 0.001, LossType::CROSS_ENTROPY);
    
    network.printArchitecture();
    
    // Sauvegarder le réseau initial
    std::string filename = "my_torch_network_basic.nn";
    network.save(filename);
    
    std::cout << "Basic neural network saved to " << filename << "\n";
    std::cout << "Input size: " << inputSize << " neurons\n";
    std::cout << "This network can now be trained or used for predictions.\n";
    
    return 0;
}
