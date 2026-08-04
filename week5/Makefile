CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=armv8-a -Wall -Wextra

all: main main_test

run: main
	./main

benchmark: main
	./main --benchmark

# Unit-Tests: pruefen die JIT-erzeugten Kernel elementweise gegen eine naive
# Referenzimplementierung. Ein falsches Instruktionswort erzeugt keinen
# Compilerfehler -- nur ein Vergleich gegen die Referenz faengt so etwas.
test: main_test
	./main_test

main_test: main_test.o Unary.o Gemm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

main_test.o: main_test.cpp Unary.h Gemm.h
	$(CXX) $(CXXFLAGS) -c $<

main: main.o Unary.o Gemm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

main.o: main.cpp Unary.h Gemm.h
	$(CXX) $(CXXFLAGS) -c $<

Unary.o: Unary.cpp Unary.h
	$(CXX) $(CXXFLAGS) -c $<

Gemm.o: Gemm.cpp Gemm.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.o main main_test
