CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -I../lib
# Apple Silicon (M4) unterstützt SME, aber KEIN normales SVE (SVE-Befehle sind
# nur im Streaming-Mode legal). Wir kompilieren den C++-Code daher für die
# CPU direkt — das verhindert SVE-Auto-Vektorisierung in der STL. Die
# SME-Instruktionen werden ausschließlich JIT-emittiert; der Assembler muss
# sie also gar nicht kennen.
ASMFLAGS = -mcpu=apple-m3

TARGETS = main main_test

all: $(TARGETS)

main: main.cpp Unary.cpp Gemm.cpp
	$(CXX) $(CXXFLAGS) $(ASMFLAGS) -o $@ $^

main_test: main_test.cpp Unary.cpp Gemm.cpp
	$(CXX) $(CXXFLAGS) $(ASMFLAGS) -o $@ $^

clean:
	rm -f $(TARGETS)

.PHONY: all clean