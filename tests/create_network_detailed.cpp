#include "../include/NeuralNetwork.hpp"
#include "../include/FENEncoder.hpp"
#include <iostream>

int main() {
    std::cout << "Creating a detailed neural network for chess analysis...\n";
    std::cout << "This network can distinguish between:\n";
    std::cout << "  - Nothing\n";
    std::cout << "  - Check White\n";
    std::cout << "  - Check Black\n";
    std::cout << "  - Checkmate White\n";
    std::cout << "  - Checkmate Black\n\n";
    
    // Architecture avec 5 sorties pour le mode détaillé
    int inputSize = FENEncoder::getInputSize();
    std::vector<size_t> layers = {(size_t)inputSize, 512, 256, 128, 5};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::SOFTMAX
    };
    
    NeuralNetwork network(layers, activations, 0.001, LossType::CROSS_ENTROPY);
    
    network.printArchitecture();
    
    // Sauvegarder le réseau initial
    std::string filename = "my_torch_network_detailed.nn";
    network.save(filename);
    
    std::cout << "Detailed neural network saved to " << filename << "\n";
    std::cout << "Input size: " << inputSize << " neurons\n";
    std::cout << "Output size: 5 neurons (detailed mode)\n";
    
    return 0;
}
