#include "../include/NeuralNetwork.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <random>
#include <iomanip>

// ============================================================================
// CONSTRUCTEURS
// ============================================================================

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& layers, 
                             const std::vector<ActivationType>& activationTypes,
                             double lr,
                             LossType loss,
                             double l2Lambda)
    : layerSizes(layers), activations(activationTypes), learningRate(lr), lossFunction(loss), lambda(l2Lambda)
{
    if (layers.size() < 2) {
        throw std::invalid_argument("Network must have at least 2 layers (input and output)");
    }
    
    if (activationTypes.size() != layers.size() - 1) {
        throw std::invalid_argument("Number of activation functions must match number of hidden + output layers");
    }
    
    // Initialiser les poids et biais
    for (size_t i = 0; i < layers.size() - 1; i++) {
        weights.push_back(Matrix(layers[i + 1], layers[i]));
        biases.push_back(Matrix(layers[i + 1], 1));
    }
    
    // Initialiser les gradients accumulés pour mini-batch
    for (size_t i = 0; i < layers.size() - 1; i++) {
        accumulatedWeightGradients.push_back(Matrix(layers[i + 1], layers[i]));
        accumulatedBiasGradients.push_back(Matrix(layers[i + 1], 1));
    }
    
    // Initialisation par défaut (Xavier)
    initializeWeights("xavier");
}

NeuralNetwork::NeuralNetwork(const std::string& filename) {
    load(filename);
}

// ============================================================================
// FONCTIONS D'ACTIVATION
// ============================================================================

double NeuralNetwork::sigmoid(double x) {
    // Protection contre overflow
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double NeuralNetwork::relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double NeuralNetwork::reluDerivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

double NeuralNetwork::tanhFunc(double x) {
    return std::tanh(x);
}

double NeuralNetwork::tanhDerivative(double x) {
    double t = std::tanh(x);
    return 1.0 - t * t;
}

Matrix NeuralNetwork::applyActivation(const Matrix& input, ActivationType type) const {
    switch (type) {
        case ActivationType::SIGMOID:
            return input.apply(sigmoid);
        case ActivationType::RELU:
            return input.apply(relu);
        case ActivationType::TANH:
            return input.apply(tanhFunc);
        case ActivationType::SOFTMAX: {
            // Softmax: exp(x_i) / sum(exp(x_j))
            Matrix result = input;
            double maxVal = input.max();
            
            // Soustraire max pour stabilité numérique
            for (size_t i = 0; i < result.getRows(); i++) {
                result(i, 0) = std::exp(result(i, 0) - maxVal);
            }
            
            double sumExp = result.sum();
            for (size_t i = 0; i < result.getRows(); i++) {
                result(i, 0) /= sumExp;
            }
            return result;
        }
        case ActivationType::LINEAR:
        default:
            return input;
    }
}

Matrix NeuralNetwork::applyActivationDerivative(const Matrix& input, ActivationType type) const {
    switch (type) {
        case ActivationType::SIGMOID:
            return input.apply(sigmoidDerivative);
        case ActivationType::RELU:
            return input.apply(reluDerivative);
        case ActivationType::TANH:
            return input.apply(tanhDerivative);
        case ActivationType::SOFTMAX:
        case ActivationType::LINEAR:
        default:
            return Matrix::ones(input.getRows(), input.getCols());
    }
}

// ============================================================================
// FORWARD PROPAGATION
// ============================================================================

Matrix NeuralNetwork::forward(const Matrix& input) {
    if (input.getRows() != layerSizes[0]) {
        throw std::invalid_argument("Input size does not match network input layer");
    }
    
    if (input.getCols() != 1) {
        throw std::invalid_argument("Input must be a column vector");
    }
    
    // Réutiliser les vecteurs pré-alloués au lieu de clear() + push_back()
    // OPTIMISATION: évite les allocations répétées
    if (layerOutputs.size() != weights.size() + 1) {
        layerOutputs.resize(weights.size() + 1);
        layerInputs.resize(weights.size());
    }
    
    layerOutputs[0] = input; // Sauvegarder l'entrée
    Matrix current = input;
    
    // Propagation à travers chaque couche
    for (size_t i = 0; i < weights.size(); i++) {
        // z = W * a + b
        Matrix z = weights[i] * current + biases[i];
        layerInputs[i] = z;
        
        // a = activation(z)
        current = applyActivation(z, activations[i]);
        layerOutputs[i + 1] = current;
    }
    
    return current;
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    return forward(input);
}

// ============================================================================
// BACKWARD PROPAGATION
// ============================================================================

void NeuralNetwork::backward(const Matrix& target) {
    if (layerOutputs.empty()) {
        throw std::runtime_error("Must call forward() before backward()");
    }
    
    std::vector<Matrix> deltas(weights.size());
    
    // Erreur de sortie
    Matrix output = layerOutputs.back();
    Matrix delta = lossDerivative(output, target);
    
    // Pour softmax avec cross-entropy, la dérivée est simplifiée
    if (activations.back() != ActivationType::SOFTMAX) {
        delta = delta.hadamard(applyActivationDerivative(layerInputs.back(), activations.back()));
    }
    
    deltas[deltas.size() - 1] = delta;
    
    // Propagation de l'erreur en arrière
    for (int i = weights.size() - 2; i >= 0; i--) {
        Matrix weightTranspose = weights[i + 1].transpose();
        delta = weightTranspose * delta;
        delta = delta.hadamard(applyActivationDerivative(layerInputs[i], activations[i]));
        deltas[i] = delta;
    }
    
    // Mise à jour des poids et biais avec L2 regularization
    for (size_t i = 0; i < weights.size(); i++) {
        Matrix activationTranspose = layerOutputs[i].transpose();
        Matrix weightGradient = deltas[i] * activationTranspose;
        
        // L2 regularization: w = w - lr * (grad + lambda * w)
        Matrix weightUpdate = weightGradient * learningRate;
        if (lambda > 0.0) {
            weightUpdate = weightUpdate + weights[i] * (lambda * learningRate);
        }
        weights[i] = weights[i] - weightUpdate;
        biases[i] = biases[i] - deltas[i] * learningRate;
    }
}

void NeuralNetwork::updateWeights(double lr) {
    learningRate = lr;
}

void NeuralNetwork::train(const Matrix& input, const Matrix& target) {
    forward(input);
    backward(target);
}

// ============================================================================
// MINI-BATCH TRAINING
// ============================================================================

void NeuralNetwork::accumulateGradients(const Matrix& target) {
    if (layerOutputs.empty()) {
        throw std::runtime_error("Must call forward() before accumulateGradients()");
    }
    
    std::vector<Matrix> deltas(weights.size());
    
    // Erreur de sortie
    Matrix output = layerOutputs.back();
    Matrix delta = lossDerivative(output, target);
    
    // Pour softmax avec cross-entropy, la dérivée est simplifiée
    if (activations.back() != ActivationType::SOFTMAX) {
        delta = delta.hadamard(applyActivationDerivative(layerInputs.back(), activations.back()));
    }
    
    deltas[deltas.size() - 1] = delta;
    
    // Propagation de l'erreur en arrière
    for (int i = weights.size() - 2; i >= 0; i--) {
        Matrix weightTranspose = weights[i + 1].transpose();
        delta = weightTranspose * delta;
        delta = delta.hadamard(applyActivationDerivative(layerInputs[i], activations[i]));
        deltas[i] = delta;
    }
    
    // ACCUMULER les gradients (ne pas mettre à jour les poids encore)
    for (size_t i = 0; i < weights.size(); i++) {
        Matrix activationTranspose = layerOutputs[i].transpose();
        Matrix weightGradient = deltas[i] * activationTranspose;
        
        accumulatedWeightGradients[i] = accumulatedWeightGradients[i] + weightGradient;
        accumulatedBiasGradients[i] = accumulatedBiasGradients[i] + deltas[i];
    }
}

void NeuralNetwork::applyAccumulatedGradients(size_t batchSize) {
    // Appliquer la moyenne des gradients accumulés avec L2 regularization
    for (size_t i = 0; i < weights.size(); i++) {
        // L2 regularization: w = w - lr * (grad/batch + lambda * w)
        Matrix weightUpdate = accumulatedWeightGradients[i] * (learningRate / batchSize);
        if (lambda > 0.0) {
            weightUpdate = weightUpdate + weights[i] * (lambda * learningRate);
        }
        weights[i] = weights[i] - weightUpdate;
        biases[i] = biases[i] - accumulatedBiasGradients[i] * (learningRate / batchSize);
        
        // Réinitialiser les gradients accumulés
        accumulatedWeightGradients[i].fill(0.0);
        accumulatedBiasGradients[i].fill(0.0);
    }
}

void NeuralNetwork::trainBatch(const std::vector<std::pair<Matrix, Matrix>>& batch) {
    if (batch.empty()) {
        return;
    }
    
    // Réinitialiser les gradients accumulés
    for (size_t i = 0; i < accumulatedWeightGradients.size(); i++) {
        accumulatedWeightGradients[i].fill(0.0);
        accumulatedBiasGradients[i].fill(0.0);
    }
    
    // Accumuler les gradients pour chaque exemple du batch
    for (const auto& example : batch) {
        forward(example.first);
        accumulateGradients(example.second);
    }
    
    // Appliquer la moyenne des gradients
    applyAccumulatedGradients(batch.size());
}

// ============================================================================
// FONCTIONS DE COÛT
// ============================================================================

double NeuralNetwork::calculateLoss(const Matrix& predicted, const Matrix& target) const {
    if (predicted.getRows() != target.getRows() || predicted.getCols() != target.getCols()) {
        throw std::invalid_argument("Predicted and target dimensions must match");
    }
    
    switch (lossFunction) {
        case LossType::MSE: {
            // Mean Squared Error: (1/n) * sum((y - y_pred)^2)
            Matrix diff = predicted - target;
            double sum = 0.0;
            for (size_t i = 0; i < diff.getRows(); i++) {
                for (size_t j = 0; j < diff.getCols(); j++) {
                    sum += diff(i, j) * diff(i, j);
                }
            }
            return sum / (diff.getRows() * diff.getCols());
        }
        
        case LossType::CROSS_ENTROPY: {
            // Cross-Entropy: -sum(y * log(y_pred))
            double sum = 0.0;
            const double epsilon = 1e-15; // Pour éviter log(0)
            for (size_t i = 0; i < predicted.getRows(); i++) {
                for (size_t j = 0; j < predicted.getCols(); j++) {
                    double pred = std::max(epsilon, std::min(1.0 - epsilon, predicted(i, j)));
                    sum -= target(i, j) * std::log(pred);
                }
            }
            return sum;
        }
        
        case LossType::BINARY_CROSS_ENTROPY: {
            // Binary Cross-Entropy: -sum(y*log(p) + (1-y)*log(1-p))
            double sum = 0.0;
            const double epsilon = 1e-15;
            for (size_t i = 0; i < predicted.getRows(); i++) {
                for (size_t j = 0; j < predicted.getCols(); j++) {
                    double pred = std::max(epsilon, std::min(1.0 - epsilon, predicted(i, j)));
                    sum -= target(i, j) * std::log(pred) + (1.0 - target(i, j)) * std::log(1.0 - pred);
                }
            }
            return sum / (predicted.getRows() * predicted.getCols());
        }
        
        default:
            return 0.0;
    }
}

Matrix NeuralNetwork::lossDerivative(const Matrix& predicted, const Matrix& target) const {
    switch (lossFunction) {
        case LossType::MSE:
            // d(MSE)/dy = 2 * (y_pred - y) / n
            return (predicted - target) * (2.0 / predicted.getRows());
        
        case LossType::CROSS_ENTROPY:
        case LossType::BINARY_CROSS_ENTROPY:
            // Pour softmax + cross-entropy: y_pred - y
            return predicted - target;
        
        default:
            return predicted - target;
    }
}

// ============================================================================
// INITIALISATION DES POIDS
// ============================================================================

void NeuralNetwork::initializeWeights(const std::string& strategy) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < weights.size(); i++) {
        size_t inputSize = layerSizes[i];
        size_t outputSize = layerSizes[i + 1];
        
        double limit;
        
        if (strategy == "xavier") {
            // Xavier initialization: sqrt(6 / (n_in + n_out))
            limit = std::sqrt(6.0 / (inputSize + outputSize));
        } else if (strategy == "he") {
            // He initialization: sqrt(2 / n_in)
            limit = std::sqrt(2.0 / inputSize);
        } else {
            // Default random
            limit = 1.0;
        }
        
        std::uniform_real_distribution<> dis(-limit, limit);
        
        for (size_t r = 0; r < weights[i].getRows(); r++) {
            for (size_t c = 0; c < weights[i].getCols(); c++) {
                weights[i](r, c) = dis(gen);
            }
        }
        
        // Biais initialisés à zéro
        biases[i].fill(0.0);
    }
}

void NeuralNetwork::randomizeWeights(double min, double max) {
    for (auto& w : weights) {
        w.randomize(min, max);
    }
    for (auto& b : biases) {
        b.fill(0.0);
    }
}

// ============================================================================
// SAUVEGARDE ET CHARGEMENT
// ============================================================================

void NeuralNetwork::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for saving: " + filename);
    }
    
    // Sauvegarder la signature
    const char* signature = "MYTORCH1";
    file.write(signature, 8);
    
    // Sauvegarder hyperparamètres
    file.write(reinterpret_cast<const char*>(&learningRate), sizeof(double));
    int lossType = static_cast<int>(lossFunction);
    file.write(reinterpret_cast<const char*>(&lossType), sizeof(int));
    
    // Sauvegarder l'architecture
    size_t numLayers = layerSizes.size();
    file.write(reinterpret_cast<const char*>(&numLayers), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(layerSizes.data()), numLayers * sizeof(size_t));
    
    // Sauvegarder les activations
    for (const auto& act : activations) {
        int actType = static_cast<int>(act);
        file.write(reinterpret_cast<const char*>(&actType), sizeof(int));
    }
    
    // Sauvegarder poids et biais
    for (size_t i = 0; i < weights.size(); i++) {
        // Poids
        size_t rows = weights[i].getRows();
        size_t cols = weights[i].getCols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
        
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                double val = weights[i](r, c);
                file.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
        }
        
        // Biais
        rows = biases[i].getRows();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
        for (size_t r = 0; r < rows; r++) {
            double val = biases[i](r, 0);
            file.write(reinterpret_cast<const char*>(&val), sizeof(double));
        }
    }
    
    file.close();
}

void NeuralNetwork::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for loading: " + filename);
    }
    
    // Vérifier la signature
    char signature[9] = {0};
    file.read(signature, 8);
    if (std::string(signature) != "MYTORCH1") {
        throw std::runtime_error("Invalid neural network file format");
    }
    
    // Charger hyperparamètres
    file.read(reinterpret_cast<char*>(&learningRate), sizeof(double));
    int lossType;
    file.read(reinterpret_cast<char*>(&lossType), sizeof(int));
    lossFunction = static_cast<LossType>(lossType);
    
    // Charger l'architecture
    size_t numLayers;
    file.read(reinterpret_cast<char*>(&numLayers), sizeof(size_t));
    layerSizes.resize(numLayers);
    file.read(reinterpret_cast<char*>(layerSizes.data()), numLayers * sizeof(size_t));
    
    // Charger les activations
    activations.clear();
    for (size_t i = 0; i < numLayers - 1; i++) {
        int actType;
        file.read(reinterpret_cast<char*>(&actType), sizeof(int));
        activations.push_back(static_cast<ActivationType>(actType));
    }
    
    // Charger poids et biais
    weights.clear();
    biases.clear();
    
    for (size_t i = 0; i < numLayers - 1; i++) {
        // Poids
        size_t rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
        
        Matrix w(rows, cols);
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                double val;
                file.read(reinterpret_cast<char*>(&val), sizeof(double));
                w(r, c) = val;
            }
        }
        weights.push_back(w);
        
        // Biais
        file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
        Matrix b(rows, 1);
        for (size_t r = 0; r < rows; r++) {
            double val;
            file.read(reinterpret_cast<char*>(&val), sizeof(double));
            b(r, 0) = val;
        }
        biases.push_back(b);
    }
    
    // Initialiser les gradients accumulés pour mini-batch
    accumulatedWeightGradients.clear();
    accumulatedBiasGradients.clear();
    for (size_t i = 0; i < numLayers - 1; i++) {
        accumulatedWeightGradients.push_back(Matrix(layerSizes[i + 1], layerSizes[i]));
        accumulatedBiasGradients.push_back(Matrix(layerSizes[i + 1], 1));
        accumulatedWeightGradients[i].fill(0.0);
        accumulatedBiasGradients[i].fill(0.0);
    }
    
    // Lambda defaults to 0 when loading (set via setLambda() if needed)
    lambda = 0.0;
    
    file.close();
}

// ============================================================================
// AFFICHAGE
// ============================================================================

void NeuralNetwork::printArchitecture() const {
    std::cout << "\n========== Neural Network Architecture ==========\n";
    std::cout << "Learning Rate: " << learningRate << "\n";
    std::cout << "L2 Lambda: " << lambda << "\n";
    std::cout << "Loss Function: ";
    switch (lossFunction) {
        case LossType::MSE: std::cout << "MSE\n"; break;
        case LossType::CROSS_ENTROPY: std::cout << "Cross-Entropy\n"; break;
        case LossType::BINARY_CROSS_ENTROPY: std::cout << "Binary Cross-Entropy\n"; break;
    }
    
    std::cout << "\nLayers:\n";
    for (size_t i = 0; i < layerSizes.size(); i++) {
        std::cout << "  Layer " << i << ": " << layerSizes[i] << " neurons";
        if (i > 0) {
            std::cout << " (";
            switch (activations[i - 1]) {
                case ActivationType::SIGMOID: std::cout << "Sigmoid"; break;
                case ActivationType::RELU: std::cout << "ReLU"; break;
                case ActivationType::TANH: std::cout << "Tanh"; break;
                case ActivationType::SOFTMAX: std::cout << "Softmax"; break;
                case ActivationType::LINEAR: std::cout << "Linear"; break;
            }
            std::cout << ")";
        }
        std::cout << "\n";
    }
    std::cout << "==================================================\n\n";
}
