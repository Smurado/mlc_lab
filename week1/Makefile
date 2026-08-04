CXX      = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

TARGET   = test_run
SRCS     = main.cpp data/base_math.cpp functions.s

all: $(TARGET)

$(TARGET): $(SRCS)
	@echo "Building $(TARGET)..."
	@$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Unit-Tests: Assembly gegen die Referenzimplementierung aus data/base_math.cpp.
test: $(TARGET)
	@./$(TARGET)

# Debug-Build fuer die GDB-Sitzung (Aufgabe 2). Ohne Optimierung, damit sich der
# Assembly-Code Schritt fuer Schritt nachvollziehen laesst.
debug: $(SRCS)
	@echo "Building test_run_debug..."
	@$(CXX) -std=c++17 -Wall -Wextra -O0 -g $(SRCS) -o test_run_debug

clean:
	@echo "Cleaning up..."
	@rm -rf $(TARGET) test_run_debug *.o *.dSYM test_run_debug.dSYM

.PHONY: all test debug clean
