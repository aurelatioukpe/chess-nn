#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "Matrix.hpp"
#include <vector>
#include <string>
#include <functional>
#include <memory>

enum class ActivationType {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX,
    LINEAR
};

enum class LossType {
    MSE,
    CROSS_ENTROPY,
    BINARY_CROSS_ENTROPY
};

class NeuralNetwork {
private:
    std::vector<size_t> layerSizes;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;
    std::vector<ActivationType> activations;
    
    // Pour la backpropagation
    std::vector<Matrix> layerOutputs;
    std::vector<Matrix> layerInputs;
    
    // Gradients accumulés pour mini-batch
    std::vector<Matrix> accumulatedWeightGradients;
    std::vector<Matrix> accumulatedBiasGradients;
    
    // Hyperparamètres
    double learningRate;
    LossType lossFunction;
    double lambda;  // L2 regularization coefficient
    
    // Méthodes privées pour les activations
    Matrix applyActivation(const Matrix& input, ActivationType type) const;
    Matrix applyActivationDerivative(const Matrix& input, ActivationType type) const;
    
    // Fonctions d'activation statiques
    static double sigmoid(double x);
    static double sigmoidDerivative(double x);
    static double relu(double x);
    static double reluDerivative(double x);
    static double tanhFunc(double x);
    static double tanhDerivative(double x);
    
public:
    // Constructeurs
    NeuralNetwork(const std::vector<size_t>& layers, 
                  const std::vector<ActivationType>& activationTypes,
                  double learningRate = 0.01,
                  LossType loss = LossType::MSE,
                  double lambda = 0.0);
    
    NeuralNetwork(const std::string& filename);
    
    ~NeuralNetwork() = default;
    
    // Forward propagation
    Matrix forward(const Matrix& input);
    Matrix predict(const Matrix& input);
    
    // Backward propagation et entraînement
    void backward(const Matrix& target);
    void updateWeights(double learningRate);
    void train(const Matrix& input, const Matrix& target);
    
    // Mini-batch training
    void trainBatch(const std::vector<std::pair<Matrix, Matrix>>& batch);
    void accumulateGradients(const Matrix& target);
    void applyAccumulatedGradients(size_t batchSize);
    
    // Fonctions de coût
    double calculateLoss(const Matrix& predicted, const Matrix& target) const;
    Matrix lossDerivative(const Matrix& predicted, const Matrix& target) const;
    
    // Sauvegarde et chargement
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    
    // Initialisation des poids
    void initializeWeights(const std::string& strategy = "xavier");
    void randomizeWeights(double min = -1.0, double max = 1.0);
    
    // Getters
    const std::vector<size_t>& getLayerSizes() const { return layerSizes; }
    size_t getOutputSize() const { return layerSizes.back(); }
    double getLearningRate() const { return learningRate; }
    void setLearningRate(double lr) { learningRate = lr; }
    double getLambda() const { return lambda; }
    void setLambda(double l) { lambda = l; }
    
    // Affichage
    void printArchitecture() const;
};

#endif
