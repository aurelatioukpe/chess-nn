#include "../include/NeuralNetwork.hpp"
#include "../include/FENEncoder.hpp"
#include <iostream>

int main() {
    std::cout << "Testing FEN encoding and output encoding...\n\n";
    
    // Test 1: FEN encoding
    std::string fen = "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59";
    FENEncoder fenEncoder;
    
    std::cout << "Test FEN: " << fen << "\n";
    Matrix input = fenEncoder.encode(fen);
    std::cout << "Encoded input size: " << input.getRows() << "x" << input.getCols() << "\n";
    std::cout << "Non-zero values: ";
    int count = 0;
    for (size_t i = 0; i < input.getRows(); i++) {
        if (input(i, 0) != 0.0) count++;
    }
    std::cout << count << "\n\n";
    
    // Test 2: Output encoding/decoding
    OutputEncoder outputEncoder(true); // Detailed mode
    
    std::string labels[] = {"Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"};
    
    for (const auto& label : labels) {
        Matrix encoded = outputEncoder.encode(label);
        std::cout << "Label: " << label << "\n";
        std::cout << "Encoded: [";
        for (size_t i = 0; i < encoded.getRows(); i++) {
            std::cout << encoded(i, 0);
            if (i < encoded.getRows() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        
        std::string decoded = outputEncoder.decodeToString(encoded);
        std::cout << "Decoded: " << decoded << "\n";
        std::cout << "Match: " << (label == decoded ? "YES" : "NO") << "\n\n";
    }
    
    return 0;
}
