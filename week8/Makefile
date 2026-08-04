CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp

# Die SME-Kernel aus Woche 6 werden mitgebaut (mini_jit::Unary / mini_jit::Gemm).
# Sie erzeugen ihre Instruktionen zur Laufzeit als Maschinenwoerter, deshalb
# braucht die Uebersetzung hier KEIN -march=...+sme.
SRCS = main.cpp teir_parser.cpp teir_evaluator.cpp teir_optimizer.cpp \
       ../week6/Unary.cpp ../week6/Gemm.cpp
OBJS = $(notdir $(SRCS:.cpp=.o))
TARGET = teir_runtime

vpath %.cpp . ../week6

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Tests: pruefen gegen eine UNABHAENGIGE Referenz, nicht gegen den Evaluator
# selbst. main.cpp vergleicht nur Baseline gegen optimiert -- beide laufen durch
# dieselben Kernel, ein falscher Kernel faellt dort nicht auf.
TEST_TARGET = test_week8
TEST_OBJS   = test_week8.o teir_parser.o teir_evaluator.o teir_optimizer.o Unary.o Gemm.o

$(TEST_TARGET): $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	rm -f $(OBJS) $(TARGET) $(TEST_OBJS) $(TEST_TARGET)

run: $(TARGET)
	./$(TARGET)

benchmark: $(TARGET)
	./$(TARGET)

.PHONY: all clean run benchmark test
