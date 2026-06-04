CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp

SRCS = main.cpp teir_parser.cpp teir_evaluator.cpp teir_optimizer.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = teir_runtime

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

benchmark: $(TARGET)
	./$(TARGET)

.PHONY: all clean run benchmark
