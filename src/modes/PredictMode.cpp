#include "../../include/modes/PredictMode.hpp"
#include "../../include/fenparser.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

int predictMode(const std::string& networkFile, const std::string& chessFile) {
    std::cout << "Loading network from: " << networkFile << std::endl;
    
    std::ifstream file(chessFile);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open chess file: " << chessFile << std::endl;
        return 84;
    }
    
    std::string line;
    FENParser parser;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::string part;
        std::vector<std::string> parts;
        while (iss >> part) {
            parts.push_back(part);
        }
        
        if (parts.size() >= 6) {
            std::string fen = parts[0] + " " + parts[1] + " " + parts[2] + " " + 
                  parts[3] + " " + parts[4] + " " + parts[5];
            
            if (!parser.parse(fen)) {
                std::cerr << "Error: Invalid FEN: " << fen << std::endl;
                return 84;
            }
            
            std::cout << "Nothing" << std::endl;
        } else {
            std::cerr << "Error: Invalid FEN format (expected 6 parts): " << line << std::endl;
            return 84;
        }
    }
    
    file.close();
    return 0;
}
