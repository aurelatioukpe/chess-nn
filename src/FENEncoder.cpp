#include "../include/FENEncoder.hpp"
#include <stdexcept>
#include <algorithm>
#include <cctype>

// ============================================================================
// FENEncoder Implementation
// ============================================================================

FENEncoder::FENEncoder() : cacheInitialized(false) {
    initializePieceMapping();
}

void FENEncoder::initializePieceMapping() {
    // Pièces blanches (majuscules): indices 0-5
    pieceToIndex['P'] = 0;  // Pion blanc
    pieceToIndex['N'] = 1;  // Cavalier blanc
    pieceToIndex['B'] = 2;  // Fou blanc
    pieceToIndex['R'] = 3;  // Tour blanche
    pieceToIndex['Q'] = 4;  // Dame blanche
    pieceToIndex['K'] = 5;  // Roi blanc
    
    // Pièces noires (minuscules): indices 6-11
    pieceToIndex['p'] = 6;  // Pion noir
    pieceToIndex['n'] = 7;  // Cavalier noir
    pieceToIndex['b'] = 8;  // Fou noir
    pieceToIndex['r'] = 9;  // Tour noire
    pieceToIndex['q'] = 10; // Dame noire
    pieceToIndex['k'] = 11; // Roi noir
}

int FENEncoder::getInputSize() {
    // 8x8 cases * 12 types de pièces (one-hot) + features additionnelles
    // Features: activePlayer (1) + castling (4) + enPassant (1) = 6
    return BOARD_SIZE * BOARD_SIZE * NUM_PIECE_TYPES + 6;
}

Matrix FENEncoder::encode(const std::string& fen) {
    // OPTIMISATION: Réutiliser le parser au lieu d'en créer un nouveau
    if (!cachedParser.parse(fen)) {
        throw std::invalid_argument("Invalid FEN string");
    }
    
    // OPTIMISATION: Réutiliser la matrice au lieu d'en créer une nouvelle
    if (!cacheInitialized) {
        cachedInput = Matrix(getInputSize(), 1);
        cacheInitialized = true;
    }
    cachedInput.fill(0.0);
    
    // Encoder la position du plateau
    auto board = cachedParser.getBoard();
    
    for (int row = 0; row < BOARD_SIZE; row++) {
        for (int col = 0; col < BOARD_SIZE; col++) {
            char piece = board[row][col];
            
            if (piece != ' ') {
                auto it = pieceToIndex.find(piece);
                if (it != pieceToIndex.end()) {
                    int pieceType = it->second;
                    int position = (row * BOARD_SIZE + col) * NUM_PIECE_TYPES + pieceType;
                    cachedInput(position, 0) = 1.0;
                }
            }
        }
    }
    
    // Encoder les features additionnelles
    int baseIdx = BOARD_SIZE * BOARD_SIZE * NUM_PIECE_TYPES;
    
    // Active player (blanc=1, noir=0)
    cachedInput(baseIdx, 0) = (cachedParser.getActivePlayer() == 'w') ? 1.0 : 0.0;
    
    // Castling rights ( bits: KQkq)
    std::string castling = cachedParser.getCastling();
    cachedInput(baseIdx + 1, 0) = (castling.find('K') != std::string::npos) ? 1.0 : 0.0;
    cachedInput(baseIdx + 2, 0) = (castling.find('Q') != std::string::npos) ? 1.0 : 0.0;
    cachedInput(baseIdx + 3, 0) = (castling.find('k') != std::string::npos) ? 1.0 : 0.0;
    cachedInput(baseIdx + 4, 0) = (castling.find('q') != std::string::npos) ? 1.0 : 0.0;
    
    // En passant (1 si possible, 0 sinon)
    // Pour simplifier, on met 1 si ce n'est pas "-"
    cachedInput(baseIdx + 5, 0) = (cachedParser.getEnPassant() != "-") ? 1.0 : 0.0;
    
    return cachedInput;
}

Matrix FENEncoder::encodeBoardPosition(const std::vector<std::vector<char>>& board) {
    Matrix result(BOARD_SIZE * BOARD_SIZE * NUM_PIECE_TYPES, 1);
    result.fill(0.0);
    
    for (int row = 0; row < BOARD_SIZE; row++) {
        for (int col = 0; col < BOARD_SIZE; col++) {
            char piece = board[row][col];
            
            if (piece != ' ') {
                auto it = pieceToIndex.find(piece);
                if (it != pieceToIndex.end()) {
                    int pieceType = it->second;
                    int position = (row * BOARD_SIZE + col) * NUM_PIECE_TYPES + pieceType;
                    result(position, 0) = 1.0;
                }
            }
        }
    }
    
    return result;
}

void FENEncoder::encodeAdditionalFeatures(Matrix& input, char activePlayer,
                                          const std::string& castling,
                                          const std::string& enPassant) {
    int baseIdx = BOARD_SIZE * BOARD_SIZE * NUM_PIECE_TYPES;
    
    input(baseIdx, 0) = (activePlayer == 'w') ? 1.0 : 0.0;
    input(baseIdx + 1, 0) = (castling.find('K') != std::string::npos) ? 1.0 : 0.0;
    input(baseIdx + 2, 0) = (castling.find('Q') != std::string::npos) ? 1.0 : 0.0;
    input(baseIdx + 3, 0) = (castling.find('k') != std::string::npos) ? 1.0 : 0.0;
    input(baseIdx + 4, 0) = (castling.find('q') != std::string::npos) ? 1.0 : 0.0;
    input(baseIdx + 5, 0) = (enPassant != "-") ? 1.0 : 0.0;
}

// ============================================================================
// OutputEncoder Implementation
// ============================================================================

OutputEncoder::OutputEncoder(bool detailed) : useDetailedOutput(detailed) {}

Matrix OutputEncoder::encode(GameState state) {
    int size = getOutputSize();
    Matrix output(size, 1);
    output.fill(0.0);
    
    if (useDetailedOutput) {
        // 5 sorties: Nothing, Check White, Check Black, Checkmate White, Checkmate Black
        switch (state) {
            case GameState::NOTHING:
                output(0, 0) = 1.0;
                break;
            case GameState::CHECK_WHITE:
                output(1, 0) = 1.0;
                break;
            case GameState::CHECK_BLACK:
                output(2, 0) = 1.0;
                break;
            case GameState::CHECKMATE_WHITE:
                output(3, 0) = 1.0;
                break;
            case GameState::CHECKMATE_BLACK:
                output(4, 0) = 1.0;
                break;
            default:
                output(0, 0) = 1.0;
        }
    } else {
        // 3 sorties: Nothing, Check, Checkmate
        switch (state) {
            case GameState::NOTHING:
                output(0, 0) = 1.0;
                break;
            case GameState::CHECK:
            case GameState::CHECK_WHITE:
            case GameState::CHECK_BLACK:
                output(1, 0) = 1.0;
                break;
            case GameState::CHECKMATE:
            case GameState::CHECKMATE_WHITE:
            case GameState::CHECKMATE_BLACK:
                output(2, 0) = 1.0;
                break;
        }
    }
    
    return output;
}

Matrix OutputEncoder::encode(const std::string& stateStr) {
    return encode(parseState(stateStr));
}

OutputEncoder::GameState OutputEncoder::decode(const Matrix& output) {
    if (output.getCols() != 1) {
        throw std::invalid_argument("Output must be a column vector");
    }
    
    int maxIdx = 0;
    double maxVal = output(0, 0);
    
    for (size_t i = 1; i < output.getRows(); i++) {
        if (output(i, 0) > maxVal) {
            maxVal = output(i, 0);
            maxIdx = i;
        }
    }
    
    if (useDetailedOutput) {
        switch (maxIdx) {
            case 0: return GameState::NOTHING;
            case 1: return GameState::CHECK_WHITE;
            case 2: return GameState::CHECK_BLACK;
            case 3: return GameState::CHECKMATE_WHITE;
            case 4: return GameState::CHECKMATE_BLACK;
            default: return GameState::NOTHING;
        }
    } else {
        switch (maxIdx) {
            case 0: return GameState::NOTHING;
            case 1: return GameState::CHECK;
            case 2: return GameState::CHECKMATE;
            default: return GameState::NOTHING;
        }
    }
}

std::string OutputEncoder::decodeToString(const Matrix& output) {
    return stateToString(decode(output));
}

int OutputEncoder::getOutputSize() const {
    return useDetailedOutput ? 5 : 3;
}

OutputEncoder::GameState OutputEncoder::parseState(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "nothing") return GameState::NOTHING;
    if (lower == "check") return GameState::CHECK;
    if (lower == "checkmate") return GameState::CHECKMATE;
    if (lower == "check white") return GameState::CHECK_WHITE;
    if (lower == "check black") return GameState::CHECK_BLACK;
    if (lower == "checkmate white") return GameState::CHECKMATE_WHITE;
    if (lower == "checkmate black") return GameState::CHECKMATE_BLACK;
    
    return GameState::NOTHING;
}

std::string OutputEncoder::stateToString(GameState state) {
    switch (state) {
        case GameState::NOTHING: return "Nothing";
        case GameState::CHECK: return "Check";
        case GameState::CHECKMATE: return "Checkmate";
        case GameState::CHECK_WHITE: return "Check White";
        case GameState::CHECK_BLACK: return "Check Black";
        case GameState::CHECKMATE_WHITE: return "Checkmate White";
        case GameState::CHECKMATE_BLACK: return "Checkmate Black";
        default: return "Nothing";
    }
}
