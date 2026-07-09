CXX = clang++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp

# JIT Parameter an den C++ Code durchreichen
JIT_CXX = "$(CXX)"
JIT_FLAGS = "-O3 -march=native $(CXXFLAGS)"
JIT_LDFLAGS = "-shared -fPIC $(LDFLAGS)"
DEFINES = -DJIT_CXX=\"$(JIT_CXX)\" -DJIT_FLAGS=\"$(JIT_FLAGS)\" -DJIT_LDFLAGS=\"$(JIT_LDFLAGS)\"

TARGET = teir_compiler

SRCS = main.cpp parser.cpp passes.cpp benchmark.cpp autotuner.cpp codegen.cpp einsum.cpp cost_model.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	@$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) $(DEFINES) -c $< -o $@

# C2: Akzeptanztest fuer die Stichproben-Validierung (Layout-Bug wird gefangen).
test: einsum.o
	@$(CXX) $(CXXFLAGS) test_c2_validator.cpp einsum.o -o _test_c2
	@./_test_c2

clean:
	@echo "Cleaning up..."
	@rm -f $(OBJS) $(TARGET) results.csv autotuner_results.csv generated_kernel.cpp generated_kernel.so _trial_*.cpp _trial_*.so _test_c2

.PHONY: all clean test