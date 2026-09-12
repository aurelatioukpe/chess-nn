#include "../../include/cli/HelpDisplay.hpp"
#include <iostream>

void printHelp() {
    std::cout << "USAGE\n";
    std::cout << "    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE\n\n";
    std::cout << "DESCRIPTION\n";
    std::cout << "    --train       Launch the neural network in training mode. Each chessboard in FILE must\n";
    std::cout << "                  contain inputs to send to the neural network in FEN notation and the expected\n";
    std::cout << "                  output separated by space. If specified, the newly trained neural network\n";
    std::cout << "                  will be saved in SAVEFILE. Otherwise, it will be saved in the original LOADFILE.\n\n";
    std::cout << "    --predict     Launch the neural network in prediction mode. Each chessboard in FILE must\n";
    std::cout << "                  contain inputs to send to the neural network in FEN notation, and optionally\n";
    std::cout << "                  an expected output.\n\n";
    std::cout << "    --save        Save neural network into SAVEFILE. Only works in train mode.\n\n";
    std::cout << "    LOADFILE      File containing an artificial neural network\n";
    std::cout << "    CHESSFILE     File containing chessboards\n";
}
