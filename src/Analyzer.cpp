#include "../include/Analyzer.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/FENEncoder.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <sstream>
#include <cstring>
#include <limits>

Analyzer::Analyzer() {}

void Analyzer::printHelp() const {
    std::cout << "USAGE\n";
    std::cout << "    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE\n\n";
    std::cout << "DESCRIPTION\n";
    std::cout << "    --train     Launch the neural network in training mode. Each chessboard in FILE must\n";
    std::cout << "                contain inputs to send to the neural network in FEN notation and the expected output\n";
    std::cout << "                separated by space. If specified, the newly trained neural network will be saved in\n";
    std::cout << "                SAVEFILE. Otherwise, it will be saved in the original LOADFILE.\n\n";
    std::cout << "    --predict   Launch the neural network in prediction mode. Each chessboard in FILE must\n";
    std::cout << "                contain inputs to send to the neural network in FEN notation, and optionally an expected\n";
    std::cout << "                output.\n\n";
    std::cout << "    --save      Save neural network into SAVEFILE. Only works in train mode.\n\n";
    std::cout << "    LOADFILE    File containing an artificial neural network\n\n";
    std::cout << "    CHESSFILE   File containing chessboards\n";
}

bool Analyzer::parseArguments(int argc, char* argv[]) {
    if (argc < 2) {
        return false;
    }
    
    // Check for --help
    if (std::strcmp(argv[1], "--help") == 0) {
        printHelp();
        return false;
    }
    
    int argIdx = 1;
    
    // Parse mode
    if (std::strcmp(argv[argIdx], "--predict") == 0) {
        config.predictMode = true;
        argIdx++;
    } else if (std::strcmp(argv[argIdx], "--train") == 0) {
        config.trainMode = true;
        argIdx++;
        
        // Check for --save
        if (argIdx < argc && std::strcmp(argv[argIdx], "--save") == 0) {
            config.saveSpecified = true;
            argIdx++;
            
            if (argIdx >= argc) {
                std::cerr << "Error: --save requires SAVEFILE argument\n";
                return false;
            }
            config.saveFile = argv[argIdx];
            argIdx++;
        }
    } else {
        std::cerr << "Error: Must specify --predict or --train mode\n";
        return false;
    }
    
    // Parse LOADFILE and CHESSFILE
    if (argIdx + 2 != argc) {
        std::cerr << "Error: Missing LOADFILE and/or CHESSFILE arguments\n";
        return false;
    }
    
    config.loadFile = argv[argIdx];
    config.chessFile = argv[argIdx + 1];
    
    // If --save not specified in train mode, save to LOADFILE
    if (config.trainMode && !config.saveSpecified) {
        config.saveFile = config.loadFile;
    }
    
    return true;
}

int Analyzer::run(int argc, char* argv[]) {
    if (!parseArguments(argc, argv)) {
        if (argc >= 2 && std::strcmp(argv[1], "--help") != 0) {
            printHelp();
            return 84;
        }
        return 0;
    }
    
    try {
        if (config.predictMode) {
            return runPredictMode();
        } else if (config.trainMode) {
            return runTrainMode();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 84;
    }
    
    return 0;
}

int Analyzer::runPredictMode() {
    // Charger le réseau
    NeuralNetwork network(config.loadFile);
    
    // Détecter le mode basé sur le nombre de sorties
    size_t outputSize = network.getOutputSize();
    bool detailedMode = (outputSize == 5);
    
    if (outputSize != 3 && outputSize != 5) {
        std::cerr << "Error: Network output size must be 3 or 5, got " << outputSize << std::endl;
        return 84;
    }
    
    // Ouvrir le fichier de positions
    std::ifstream chessFile(config.chessFile);
    if (!chessFile.is_open()) {
        std::cerr << "Error: Cannot open chess file: " << config.chessFile << std::endl;
        return 84;
    }
    
    FENEncoder fenEncoder;
    OutputEncoder outputEncoder(detailedMode);
    
    std::string line;
    while (std::getline(chessFile, line)) {
        if (line.empty()) continue;
        
        // Le FEN complet est composé de 6 parties séparées par des espaces
        // On prend tout sauf potentiellement un label à la fin
        std::string fen = line;
        
        // Compter les parties pour savoir si c'est FEN complet + label optionnel
        std::istringstream tempIss(line);
        std::vector<std::string> parts;
        std::string part;
        while (tempIss >> part) {
            parts.push_back(part);
        }
        
        // Le FEN complet a 6 parties, donc si on en a plus, le dernier est probablement le label
        if (parts.size() >= 6) {
            fen = parts[0] + " " + parts[1] + " " + parts[2] + " " + 
                  parts[3] + " " + parts[4] + " " + parts[5];
        }
        
        try {
            // Encoder et prédire
            Matrix input = fenEncoder.encode(fen);
            Matrix output = network.predict(input);
            
            // DEBUG: Afficher le vecteur de sortie (DISABLED for performance)
            // std::cerr << "DEBUG Output: [";
            // for (size_t i = 0; i < output.getRows(); i++) {
            //     std::cerr << output(i, 0);
            //     if (i < output.getRows() - 1) std::cerr << ", ";
            // }
            // std::cerr << "]" << std::endl;
            
            // Décoder et afficher
            std::string prediction = outputEncoder.decodeToString(output);
            std::cout << prediction << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing FEN: " << fen << " - " << e.what() << std::endl;
            return 84;
        }
    }
    
    chessFile.close();
    return 0;
}

int Analyzer::runTrainMode() {
    // Charger le réseau initial
    NeuralNetwork network(config.loadFile);
    
    // Détecter le mode basé sur le nombre de sorties
    size_t outputSize = network.getOutputSize();
    bool detailedMode = (outputSize == 5);
    
    if (outputSize != 3 && outputSize != 5) {
        std::cerr << "Error: Network output size must be 3 or 5, got " << outputSize << std::endl;
        return 84;
    }
    
    // Ouvrir le fichier de données d'entraînement
    std::ifstream chessFile(config.chessFile);
    if (!chessFile.is_open()) {
        std::cerr << "Error: Cannot open chess file: " << config.chessFile << std::endl;
        return 84;
    }
    
    FENEncoder fenEncoder;
    OutputEncoder outputEncoder(detailedMode);
    
    // Charger toutes les données d'entraînement
    std::vector<std::pair<Matrix, Matrix>> trainingData;
    std::string line;
    int lineNum = 0;
    
    while (std::getline(chessFile, line)) {
        lineNum++;
        if (line.empty()) continue;
        
        // Parse: FEN + label séparés par espace
        std::istringstream iss(line);
        std::string fen;
        std::string label;
        
        // Le FEN peut contenir des espaces, donc on doit être prudent
        // Format attendu: les 6 parties du FEN + label
        std::vector<std::string> parts;
        std::string part;
        while (iss >> part) {
            parts.push_back(part);
        }
        
        if (parts.size() < 7) {
            std::cerr << "Warning: Invalid format at line " << lineNum << ", skipping\n";
            continue;
        }
        
        // Reconstruire le FEN (6 premières parties)
        fen = parts[0] + " " + parts[1] + " " + parts[2] + " " + 
              parts[3] + " " + parts[4] + " " + parts[5];
        
        // Le label peut être 1 ou 2 mots (ex: "Nothing" ou "Check White")
        // Tout ce qui vient après le FEN est le label
        if (parts.size() == 7) {
            label = parts[6];  // 1 mot (Nothing)
        } else if (parts.size() >= 8) {
            label = parts[6] + " " + parts[7];  // 2 mots (Check White, etc.)
        } else {
            std::cerr << "Warning: Missing label at line " << lineNum << ", skipping\n";
            continue;
        }
        
        try {
            Matrix input = fenEncoder.encode(fen);
            Matrix target = outputEncoder.encode(label);
            trainingData.push_back({input, target});
        } catch (const std::exception& e) {
            std::cerr << "Warning: Error at line " << lineNum << ": " << e.what() << ", skipping\n";
        }
    }
    
    chessFile.close();
    
    if (trainingData.empty()) {
        std::cerr << "Error: No valid training data loaded\n";
        return 84;
    }
    
    std::cout << "Loaded " << trainingData.size() << " training examples\n";
    std::cout << "Starting training...\n";
    
    // Entraînement - ajuster les époques selon la taille du dataset
    int epochs = 10; // Réduit pour datasets larges
    if (trainingData.size() < 100) {
        epochs = 100;
    } else if (trainingData.size() < 1000) {
        epochs = 50;
    } else if (trainingData.size() < 10000) {
        epochs = 20;
    } else if (trainingData.size() < 50000) {
        epochs = 15;
    } else {
        epochs = 10;  // Large datasets: 10 epochs suffisent
    }
    
    // Taille du mini-batch - plus grand pour gros datasets
    const size_t batchSize = (trainingData.size() > 50000) ? 64 : 32;
    const bool useMiniBatch = trainingData.size() > 50;  // Utiliser mini-batch si >50 exemples
    
    // Early stopping parameters
    const int patience = 3;  // Stop if no improvement for 3 epochs
    int patienceCounter = 0;
    double bestLoss = std::numeric_limits<double>::infinity();
    
    // Learning rate decay parameters  
    const double lrDecayFactor = 0.9;  // Multiply LR by 0.9 every decayEvery epochs
    const int decayEvery = 3;  // Decay every 3 epochs
    double currentLR = network.getLearningRate();
    
    // L2 regularization - stronger for large datasets to prevent overfitting
    double lambda = (trainingData.size() > 50000) ? 0.0005 : 0.0001;
    network.setLambda(lambda);
    
    std::cout << "Training mode: " << (useMiniBatch ? "Mini-Batch SGD" : "Batch GD") << "\n";
    if (useMiniBatch) {
        std::cout << "Batch size: " << batchSize << "\n";
    }
    std::cout << "Epochs: " << epochs << " (with early stopping, patience=" << patience << ")\n";
    std::cout << "L2 Lambda: " << network.getLambda() << "\n";
    std::cout << "LR Decay: " << lrDecayFactor << " every " << decayEvery << " epochs\n\n";
    
    const int printEvery = (epochs > 10) ? epochs / 10 : 1;
    
    // Random engine pour shuffling
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;
        int correct = 0;
        
        // Shuffle les données au début de chaque epoch
        if (useMiniBatch) {
            std::shuffle(trainingData.begin(), trainingData.end(), gen);
        }
        
        if (useMiniBatch) {
            // Mini-Batch SGD
            size_t numBatches = (trainingData.size() + batchSize - 1) / batchSize;
            
            for (size_t b = 0; b < numBatches; b++) {
                size_t start = b * batchSize;
                size_t end = std::min(start + batchSize, trainingData.size());
                
                // Créer le mini-batch
                std::vector<std::pair<Matrix, Matrix>> batch(
                    trainingData.begin() + start,
                    trainingData.begin() + end
                );
                
                // Entraîner sur le batch
                network.trainBatch(batch);
            }
            
            // Calculer loss et accuracy sur tout le dataset
            for (const auto& data : trainingData) {
                Matrix output = network.predict(data.first);
                totalLoss += network.calculateLoss(output, data.second);
                
                if (outputEncoder.decode(output) == outputEncoder.decode(data.second)) {
                    correct++;
                }
            }
        } else {
            // Batch Gradient Descent classique (pour petits datasets)
            for (const auto& data : trainingData) {
                network.train(data.first, data.second);
                Matrix output = network.predict(data.first);
                totalLoss += network.calculateLoss(output, data.second);
                
                if (outputEncoder.decode(output) == outputEncoder.decode(data.second)) {
                    correct++;
                }
            }
        }
        
        double avgLoss = totalLoss / trainingData.size();
        double accuracy = 100.0 * correct / trainingData.size();
        
        // Print progress
        if (epoch % printEvery == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << "/" << epochs 
                      << " - Loss: " << avgLoss 
                      << " - Accuracy: " << accuracy << "%"
                      << " - LR: " << currentLR << "\n";
        }
        
        // Early stopping check
        if (avgLoss < bestLoss) {
            bestLoss = avgLoss;
            patienceCounter = 0;
        } else {
            patienceCounter++;
            if (patienceCounter >= patience) {
                std::cout << "\nEarly stopping at epoch " << epoch 
                          << " (no improvement for " << patience << " epochs)\n";
                break;
            }
        }
        
        // Learning rate decay
        if ((epoch + 1) % decayEvery == 0) {
            currentLR *= lrDecayFactor;
            network.setLearningRate(currentLR);
            std::cout << "  -> LR decayed to " << currentLR << "\n";
        }
    }
    
    // Sauvegarder le réseau entraîné
    std::cout << "Saving trained network to " << config.saveFile << "...\n";
    network.save(config.saveFile);
    std::cout << "Training complete!\n";
    
    return 0;
}

