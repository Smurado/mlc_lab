# clang++ statt g++: Auf dem Mac ist "g++" je nach PATH entweder Apple clang
# oder Homebrew GCC. Die beiden melden unterschiedliche Warnungen -- ein
# fehlerhafter Clobber-Eintrag (x18, das Platform-Register) blieb lokal unter
# GCC unsichtbar und fiel erst in der CI unter Apple clang auf.
#
# Ein A/B-Vergleich ueber 64 Messpunkte hat gezeigt, dass die Wahl des
# Compilers die Messwerte nicht verschiebt (Median +0,3 %, weit unter der
# Streuung zwischen Wiederholungen). Die Rechenarbeit steckt in
# SME-Instruktionen, die der JIT zur Laufzeit schreibt.
CXX = clang++
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
