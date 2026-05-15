CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=armv8-a -Wall -Wextra

all: main

run: main
	./main

benchmark: main
	./main --benchmark

main: main.o Unary.o Gemm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

main.o: main.cpp Unary.h Gemm.h
	$(CXX) $(CXXFLAGS) -c $<

Unary.o: Unary.cpp Unary.h
	$(CXX) $(CXXFLAGS) -c $<

Gemm.o: Gemm.cpp Gemm.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.o main
