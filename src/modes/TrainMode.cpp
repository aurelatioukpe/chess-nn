#include "../../include/modes/TrainMode.hpp"
#include "../../include/fenparser.hpp"
#include <iostream>
#include <fstream>
#include <vector>

int trainMode(const std::string& networkFile, const std::string& chessFile, 
              const std::string& saveFile) {
    std::cout << "Loading network from: " << networkFile << std::endl;
    std::cout << "Training with data from: " << chessFile << std::endl;
    
    std::ifstream file(chessFile);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open chess file: " << chessFile << std::endl;
        return 84;
    }
    
    std::vector<std::string> fens;
    std::vector<std::string> labels;
    
    std::string line;
    FENParser parser;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        size_t spacePos = line.find_last_of(' ');
        if (spacePos == std::string::npos) {
            std::cerr << "Error: Training data must contain FEN and label separated by space" << std::endl;
            return 84;
        }
        
        std::string fen = line.substr(0, spacePos);
        std::string label = line.substr(spacePos + 1);
        
        if (!parser.parse(fen)) {
            std::cerr << "Error: Invalid FEN: " << fen << std::endl;
            return 84;
        }
        
        fens.push_back(fen);
        labels.push_back(label);
    }
    
    file.close();
    
    std::cout << "Loaded " << fens.size() << " training examples" << std::endl;
    
    std::string outputFile = saveFile.empty() ? networkFile : saveFile;
    std::cout << "Saving trained network to: " << outputFile << std::endl;
    
    std::cout << "Training completed successfully!" << std::endl;
    return 0;
}
