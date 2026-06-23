CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2
TARGET = teir_compiler

SRCS = main.cpp parser.cpp passes.cpp benchmark.cpp autotuner.cpp codegen.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) results.csv autotuner_results.csv generated_kernel.cpp generated_kernel.so

.PHONY: all clean