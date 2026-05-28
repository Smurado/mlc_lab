# Compiler configuration for Apple Clang
CXX      := clang++
CXXFLAGS := -std=c++17 -O3 -Xpreprocessor -fopenmp

# Homebrew paths for Apple Silicon (Change to /usr/local if on an Intel Mac)
BREW_PREFIX := /opt/homebrew
INCLUDES    := -I$(BREW_PREFIX)/opt/libomp/include
LDFLAGS     := -L$(BREW_PREFIX)/opt/libomp/lib -lomp -ldl

# Target configurations
TARGET := teir_runtime
SRCS   := main.cpp teir_parser.cpp teir_compiler.cpp
OBJS   := $(SRCS:.cpp=.o)

.PHONY: all clean run test bench

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

run: all
	./$(TARGET)

test: all
	./$(TARGET) --tests-only

bench: all
	./$(TARGET) --bench-only

clean:
	rm -f $(TARGET) $(OBJS) teir_jit_*.cpp teir_jit_*.so