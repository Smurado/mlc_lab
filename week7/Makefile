# Compiler configuration for Apple Clang
CXX      := clang++
CXXFLAGS := -std=c++17 -O3 -Xpreprocessor -fopenmp

# Homebrew paths for Apple Silicon (Change to /usr/local if on an Intel Mac)
BREW_PREFIX := /opt/homebrew
INCLUDES    := -I$(BREW_PREFIX)/opt/libomp/include
LDFLAGS     := -L$(BREW_PREFIX)/opt/libomp/lib -lomp -ldl

# Target configurations
TARGET := teir_runtime
SRCS   := main.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET) $(LDFLAGS)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET) teir_jit_*.cpp teir_jit_*.so