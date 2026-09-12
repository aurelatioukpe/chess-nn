#include "../include/fenparser.hpp"

FENParser::FENParser() 
    : board(8, std::vector<char>(8, '.')), 
      activePlayer('w'),
      castlingRights("-"),
      enPassant("-"),
      halfMoveClock(0),
      fullMoveNumber(1)
{}

bool FENParser::validatePiece(char c) {
    // Pièces valides : p,r,n,b,q,k (noir) et P,R,N,B,Q,K (blanc)
    return std::string("prnbqkPRNBQK").find(c) != std::string::npos;
}

bool FENParser::validateRow(const std::string& row) {
    int squareCount = 0;
    for (char c : row) {
        if (isdigit(c)) {
            int num = c - '0';
            if (num < 1 || num > 8) return false;
            squareCount += num;
        } else if (validatePiece(c)) {
            squareCount++;
        } else {
            return false;
        }
    }
    return squareCount == 8;
}

bool FENParser::parse(const std::string& fen) {
    std::istringstream iss(fen);
    std::string token;
    std::vector<std::string> parts;
    
    // Split FEN string par espaces
    while (std::getline(iss, token, ' ')) {
        parts.push_back(token);
    }
    
    if (parts.size() != 6) {
        std::cerr << "Error: FEN must have 6 parts, got " << parts.size() << std::endl;
        return false;
    }
    
    // Réinitialiser le board
    board = std::vector<std::vector<char>>(8, std::vector<char>(8, '.'));
    
    // Parse board (partie 1)
    std::istringstream boardStream(parts[0]);
    std::string row;
    int rowIdx = 0;
    
    while (std::getline(boardStream, row, '/') && rowIdx < 8) {
        if (!validateRow(row)) {
            std::cerr << "Error: Invalid row format: " << row << std::endl;
            return false;
        }
        
        int colIdx = 0;
        for (char c : row) {
            if (isdigit(c)) {
                // Cases vides : avancer de N cases
                int emptySquares = c - '0';
                colIdx += emptySquares;
            } else if (validatePiece(c)) {
                board[rowIdx][colIdx] = c;
                colIdx++;
            }
        }
        rowIdx++;
    }
    
    if (rowIdx != 8) {
        std::cerr << "Error: Board must have 8 rows, got " << rowIdx << std::endl;
        return false;
    }
    
    // Parse active player (partie 2)
    if (parts[1] == "w" || parts[1] == "b") {
        activePlayer = parts[1][0];
    } else {
        std::cerr << "Error: Invalid active player '" << parts[1] << "'" << std::endl;
        return false;
    }
    
    // Parse castling rights (partie 3)
    castlingRights = parts[2];
    if (castlingRights != "-" && 
        castlingRights.find_first_not_of("KQkq") != std::string::npos) {
        std::cerr << "Error: Invalid castling rights '" << castlingRights << "'" << std::endl;
        return false;
    }
    
    // Parse en passant (partie 4)
    enPassant = parts[3];
    if (enPassant != "-") {
        // Validation basique : format 'e3', 'a6', etc.
        if (enPassant.length() != 2 || 
            enPassant[0] < 'a' || enPassant[0] > 'h' ||
            enPassant[1] < '1' || enPassant[1] > '8') {
            std::cerr << "Error: Invalid en passant square '" << enPassant << "'" << std::endl;
            return false;
        }
    }
    
    // Parse half move clock et full move number (parties 5 et 6)
    try {
        halfMoveClock = std::stoi(parts[4]);
        fullMoveNumber = std::stoi(parts[5]);
        
        if (halfMoveClock < 0 || fullMoveNumber < 1) {
            std::cerr << "Error: Invalid move numbers" << std::endl;
            return false;
        }
    } catch (...) {
        std::cerr << "Error: Cannot parse move numbers" << std::endl;
        return false;
    }
    
    return true;
}

void FENParser::printBoard() const {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    for (int i = 0; i < 8; i++) {
        std::cout << (8 - i) << " |";
        for (int j = 0; j < 8; j++) {
            std::cout << " " << board[i][j] << " |";
        }
        std::cout << " " << (8 - i) << "\n";
        std::cout << "  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "    a   b   c   d   e   f   g   h\n\n";
}

void FENParser::printInfo() const {
    std::cout << "Active player: " << activePlayer << std::endl;
    std::cout << "Castling rights: " << castlingRights << std::endl;
    std::cout << "En passant: " << enPassant << std::endl;
    std::cout << "Half move clock: " << halfMoveClock << std::endl;
    std::cout << "Full move number: " << fullMoveNumber << std::endl;
}