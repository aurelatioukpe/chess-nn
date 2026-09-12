#include "../include/NeuralNetwork.hpp"
#include "../include/FENEncoder.hpp"
#include <iostream>

int main() {
    std::cout << "Creating a SMALL neural network for testing...\n\n";
    
    // Architecture beaucoup plus petite
    int inputSize = FENEncoder::getInputSize();
    std::vector<size_t> layers = {(size_t)inputSize, 32, 16, 5};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::SOFTMAX
    };
    
    NeuralNetwork network(layers, activations, 0.01, LossType::CROSS_ENTROPY);
    
    network.printArchitecture();
    
    // Sauvegarder le réseau initial
    std::string filename = "my_torch_network_small.nn";
    network.save(filename);
    
    std::cout << "Small neural network saved to " << filename << "\n";
    
    return 0;
}
