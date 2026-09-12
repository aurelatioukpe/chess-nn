#ifndef FENPARSER_HPP
#define FENPARSER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cctype>

class FENParser {
private:
    std::vector<std::vector<char>> board;
    char activePlayer;
    std::string castlingRights;
    std::string enPassant;
    int halfMoveClock;
    int fullMoveNumber;

    bool validatePiece(char c);
    bool validateRow(const std::string& row);

public:
    FENParser();
    bool parse(const std::string& fen);
    
    // Getters
    const std::vector<std::vector<char>>& getBoard() const { return board; }
    char getActivePlayer() const { return activePlayer; }
    std::string getCastlingRights() const { return castlingRights; }
    // Backward-compatible alias (used by older code)
    std::string getCastling() const { return castlingRights; }
    std::string getEnPassant() const { return enPassant; }
    int getHalfMoveClock() const { return halfMoveClock; }
    int getFullMoveNumber() const { return fullMoveNumber; }
    
    // Affichage
    void printBoard() const;
    void printInfo() const;
};

#endif // FENPARSER_HPP