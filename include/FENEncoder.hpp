#ifndef FENENCODER_HPP
#define FENENCODER_HPP

#include "Matrix.hpp"
#include "fenparser.hpp"
#include <string>
#include <unordered_map>

class FENEncoder {
private:
    static const int BOARD_SIZE = 8;
    static const int NUM_PIECE_TYPES = 12; // 6 types * 2 couleurs
    
    // Mapping des pièces vers indices
    std::unordered_map<char, int> pieceToIndex;
    
    // OPTIMISATION: cache pour réutiliser les allocations
    mutable Matrix cachedInput;
    mutable FENParser cachedParser;
    mutable bool cacheInitialized;
    
    void initializePieceMapping();
    
public:
    FENEncoder();
    
    // Convertit une position FEN en vecteur d'entrée pour le réseau
    Matrix encode(const std::string& fen);
    
    // Retourne la taille du vecteur d'entrée
    static int getInputSize();
    
    // Méthodes auxiliaires
    Matrix encodeBoardPosition(const std::vector<std::vector<char>>& board);
    void encodeAdditionalFeatures(Matrix& input, char activePlayer, 
                                  const std::string& castling, 
                                  const std::string& enPassant);
};

class OutputEncoder {
public:
    enum class GameState {
        NOTHING = 0,
        CHECK = 1,
        CHECKMATE = 2,
        CHECK_WHITE = 3,
        CHECK_BLACK = 4,
        CHECKMATE_WHITE = 5,
        CHECKMATE_BLACK = 6
    };
    
private:
    bool useDetailedOutput; // true pour inclure les couleurs
    
public:
    OutputEncoder(bool detailed = false);
    
    // Encode un état de jeu en vecteur
    Matrix encode(GameState state);
    Matrix encode(const std::string& stateStr);
    
    // Décode un vecteur en état de jeu
    GameState decode(const Matrix& output);
    std::string decodeToString(const Matrix& output);
    
    // Retourne la taille du vecteur de sortie
    int getOutputSize() const;
    
    // Parse une string en GameState
    static GameState parseState(const std::string& str);
    static std::string stateToString(GameState state);
};

#endif
