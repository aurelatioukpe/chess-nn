#ifndef ANALYZER_HPP
#define ANALYZER_HPP

#include <string>
#include <vector>

struct AnalyzerConfig {
    bool trainMode;
    bool predictMode;
    bool saveSpecified;
    std::string loadFile;
    std::string chessFile;
    std::string saveFile;
    
    AnalyzerConfig() : trainMode(false), predictMode(false), saveSpecified(false) {}
};

class Analyzer {
private:
    AnalyzerConfig config;
    
    void printHelp() const;
    bool parseArguments(int argc, char* argv[]);
    
public:
    Analyzer();
    
    int run(int argc, char* argv[]);
    int runPredictMode();
    int runTrainMode();
};

#endif
