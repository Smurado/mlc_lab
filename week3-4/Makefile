CXX = clang++

# WICHTIG: Der C++-Teil wird OHNE SME-/SVE-Flags uebersetzt.
#
# Apple Silicon (M4) implementiert SME, aber KEIN eigenstaendiges SVE -- SVE-
# Instruktionen sind nur INNERHALB des Streaming-Modus gueltig. Uebersetzt man
# main.cpp mit `-march=armv9.2-a+sme`, haelt clang SVE fuer generell verfuegbar
# und autovektorisiert ab -O2 den normalen C++-Code damit. Diese Instruktionen
# laufen dann ausserhalb des Streaming-Modus und die Anwendung stirbt mit SIGILL
# (Exit-Code 132).
#
# Nachweis: `clang++ -O2 -march=armv9.2-a+sme -S main.cpp` erzeugt 29 SVE-
# Instruktionen, bei -O0 sind es 0. Deshalb wird das SME-Flag ausschliesslich
# beim Assemblieren der .s-Dateien gesetzt.
CXXFLAGS  = -std=c++17 -Wall -Wextra -O2
ASMFLAGS  = -march=armv9.2-a+sme

TARGET    = main_test
OBJS      = main.o functions_sve.o functions_sme.o

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	@$(CXX) $(OBJS) -o $(TARGET)

main.o: main.cpp
	@echo "Compiling main.cpp (ohne SVE/SME-Flags)..."
	@$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

functions_sve.o: functions_sve.s
	@echo "Assembling functions_sve.s..."
	@$(CXX) $(ASMFLAGS) -c functions_sve.s -o functions_sve.o

functions_sme.o: functions_sme.s
	@echo "Assembling functions_sme.s..."
	@$(CXX) $(ASMFLAGS) -c functions_sme.s -o functions_sme.o

# Korrektheitstests + Leistungsmessung (GFLOPS fuer GEMM, GiB/s fuer die Unaries).
test: $(TARGET)
	@./$(TARGET)

# Nur die Korrektheitstests, ohne die Messungen.
check: $(TARGET)
	@./$(TARGET) "~[benchmark]"

clean:
	@echo "Cleaning up..."
	@rm -rf $(TARGET) *.o *.dSYM

.PHONY: all test check clean
