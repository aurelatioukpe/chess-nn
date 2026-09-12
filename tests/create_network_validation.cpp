#include "../include/NeuralNetwork.hpp"
#include "../include/FENEncoder.hpp"
#include <iostream>

int main() {
    std::cout << "Creating a SMALL test network for validation...\n\n";
    
    // Architecture SIMPLE pour valider que l'entraînement fonctionne
    int inputSize = FENEncoder::getInputSize();
    std::vector<size_t> layers = {(size_t)inputSize, 64, 32, 5};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::SOFTMAX
    };
    
    NeuralNetwork network(layers, activations, 0.01, LossType::CROSS_ENTROPY);
    
    network.printArchitecture();
    
    std::string filename = "my_torch_network_validation.nn";
    network.save(filename);
    
    std::cout << "Small validation network saved to " << filename << "\n";
    std::cout << "Architecture: 774 -> 64 -> 32 -> 5\n";
    std::cout << "Learning rate: 0.01 (10x plus élevé pour petit dataset)\n";
    
    return 0;
}
