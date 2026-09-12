#include "../include/NeuralNetwork.hpp"
#include "../include/FENEncoder.hpp"
#include <iostream>

int main() {
    std::cout << "Testing network predictions...\n\n";
    
    // Charger le réseau
    NeuralNetwork network("my_torch_network_balanced.nn");
    network.printArchitecture();
    
    FENEncoder fenEncoder;
    OutputEncoder outputEncoder(true);
    
    // Tester plusieurs positions
    std::vector<std::pair<std::string, std::string>> testCases = {
        {"8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59", "Check White"},
        {"8/8/8/1p1p2p1/q1p5/8/K1k5/8 w - - 4 70", "Checkmate Black"},
        {"8/5p1p/5K1P/6P1/7k/8/8/6R1 b - - 14 80", "Nothing"}
    };
    
    for (const auto& testCase : testCases) {
        Matrix input = fenEncoder.encode(testCase.first);
        Matrix output = network.predict(input);
        
        std::cout << "\nExpected: " << testCase.second << "\n";
        std::cout << "Output vector: [";
        for (size_t i = 0; i < output.getRows(); i++) {
            std::cout << output(i, 0);
            if (i < output.getRows() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        
        std::string prediction = outputEncoder.decodeToString(output);
        std::cout << "Prediction: " << prediction << "\n";
    }
    
    return 0;
}
