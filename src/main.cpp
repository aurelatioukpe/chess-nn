#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"
#include <iostream>
#include <vector>

void testNeuralNetwork() {
    std::cout << "\n========== TEST NEURAL NETWORK ==========\n";
    
    // Créer un réseau simple: 2 entrées, 3 neurones cachés, 1 sortie
    std::vector<size_t> layers = {2, 3, 1};
    std::vector<ActivationType> activations = {ActivationType::SIGMOID, ActivationType::SIGMOID};
    
    NeuralNetwork nn(layers, activations, 0.1, LossType::MSE);
    nn.printArchitecture();
    
    // Test XOR problem
    std::cout << "Training on XOR problem...\n";
    std::vector<std::pair<Matrix, Matrix>> trainingData = {
        {Matrix(2, 1), Matrix(1, 1)}, // 0, 0 -> 0
        {Matrix(2, 1), Matrix(1, 1)}, // 0, 1 -> 1
        {Matrix(2, 1), Matrix(1, 1)}, // 1, 0 -> 1
        {Matrix(2, 1), Matrix(1, 1)}  // 1, 1 -> 0
    };
    
    // Setup XOR data
    trainingData[0].first(0, 0) = 0; trainingData[0].first(1, 0) = 0; trainingData[0].second(0, 0) = 0;
    trainingData[1].first(0, 0) = 0; trainingData[1].first(1, 0) = 1; trainingData[1].second(0, 0) = 1;
    trainingData[2].first(0, 0) = 1; trainingData[2].first(1, 0) = 0; trainingData[2].second(0, 0) = 1;
    trainingData[3].first(0, 0) = 1; trainingData[3].first(1, 0) = 1; trainingData[3].second(0, 0) = 0;
    
    // Training
    for (int epoch = 0; epoch < 5000; epoch++) {
        double totalLoss = 0.0;
        for (const auto& data : trainingData) {
            nn.train(data.first, data.second);
            Matrix output = nn.predict(data.first);
            totalLoss += nn.calculateLoss(output, data.second);
        }
        
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " - Loss: " << totalLoss / trainingData.size() << "\n";
        }
    }
    
    // Test predictions
    std::cout << "\nPredictions:\n";
    for (size_t i = 0; i < trainingData.size(); i++) {
        Matrix output = nn.predict(trainingData[i].first);
        std::cout << "Input: [" << trainingData[i].first(0, 0) << ", " << trainingData[i].first(1, 0) 
                  << "] -> Output: " << output(0, 0) 
                  << " (Expected: " << trainingData[i].second(0, 0) << ")\n";
    }
    
    // Test save/load
    std::cout << "\nTesting save/load...\n";
    nn.save("test_network.nn");
    std::cout << "Network saved to test_network.nn\n";
    
    NeuralNetwork loaded("test_network.nn");
    std::cout << "Network loaded successfully!\n";
    loaded.printArchitecture();
    
    std::cout << "\nVerifying loaded network predictions:\n";
    for (size_t i = 0; i < trainingData.size(); i++) {
        Matrix output = loaded.predict(trainingData[i].first);
        std::cout << "Input: [" << trainingData[i].first(0, 0) << ", " << trainingData[i].first(1, 0) 
                  << "] -> Output: " << output(0, 0) << "\n";
    }
}

int main() {
    try {
        testNeuralNetwork();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 84;
    }
    
    if (mode == "--predict") {
        if (argc != 4) {
            std::cerr << "Error: Invalid number of arguments for --predict mode" << std::endl;
            printHelp();
            return 84;
        }
        
        std::string networkFile = argv[2];
        std::string chessFile = argv[3];
        
        return predictMode(networkFile, chessFile);
    }
    else if (mode == "--train") {
        std::string saveFile = "";
        int loadFileIndex = 2;
        
        if (argc >= 5 && std::string(argv[2]) == "--save") {
            saveFile = argv[3];
            loadFileIndex = 4;
        }
        
        if (argc < loadFileIndex + 2) {
            std::cerr << "Error: Invalid number of arguments for --train mode" << std::endl;
            printHelp();
            return 84;
        }
        
        std::string networkFile = argv[loadFileIndex];
        std::string chessFile = argv[loadFileIndex + 1];
        
        return trainMode(networkFile, chessFile, saveFile);
    }
    else {
        std::cerr << "Error: Unknown mode '" << mode << "'" << std::endl;
        printHelp();
        return 84;
    }
}