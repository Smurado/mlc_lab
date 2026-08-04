CXX      = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

TARGET   = test_run
SRCS     = main.cpp functions.s

all: $(TARGET)

$(TARGET): $(SRCS)
	@echo "Building $(TARGET)..."
	@$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Korrektheitstests + Durchsatzmessungen (FMADD/FMLA) + Bandbreite der
# Permutation in GiB/s in Abhaengigkeit von |c|.
test: $(TARGET)
	@./$(TARGET)

# Nur die Korrektheitstests, ohne die langlaufenden Messungen.
check: $(TARGET)
	@./$(TARGET) "[perm_neon_abc_cba]"

clean:
	@echo "Cleaning up..."
	@rm -rf $(TARGET) *.o *.dSYM

.PHONY: all test check clean
