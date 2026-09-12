NAME    = my_torch_analyzer

SRC_DIR = ./src
INC_DIR = ./include

SRC_ANALYZER = $(SRC_DIR)/main_analyzer.cpp $(SRC_DIR)/Analyzer.cpp \
               $(SRC_DIR)/NeuralNetwork.cpp $(SRC_DIR)/FENEncoder.cpp \
               $(SRC_DIR)/fenparser.cpp $(SRC_DIR)/Matrix.cpp \
               $(SRC_DIR)/Operators.cpp $(SRC_DIR)/Multiply_Matrix.cpp \
               $(SRC_DIR)/Analyse.cpp

SRC_TEST = $(SRC_DIR)/main.cpp $(SRC_DIR)/NeuralNetwork.cpp \
           $(SRC_DIR)/Matrix.cpp $(SRC_DIR)/Operators.cpp \
           $(SRC_DIR)/Multiply_Matrix.cpp $(SRC_DIR)/Analyse.cpp

OBJ_ANALYZER = $(SRC_ANALYZER:.cpp=.o)
OBJ_TEST = $(SRC_TEST:.cpp=.o)

CXX     = g++
CXXFLAGS = -I $(INC_DIR) -Wall -Wextra -std=c++17

all: $(NAME)

$(NAME): $(OBJ_ANALYZER)
	$(CXX) $(CXXFLAGS) -o $@ $^

test: $(OBJ_TEST)
	$(CXX) $(CXXFLAGS) -o nn_test $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o
	rm -f tools/create_network tools/create_network_* tools/test_*

fclean: clean
	rm -f $(NAME) nn_test

re: fclean all

