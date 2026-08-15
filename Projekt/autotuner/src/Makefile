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

# Unit-Tests. `make test` laeuft alle.
#   C2:    Akzeptanztest fuer die Stichproben-Validierung (Layout-Bug wird gefangen).
#   bench: Messschleife (bench_loop.hpp) -- Deckel-Einhaltung, Blockgroesse,
#          Messwert-Identitaet alt/adaptiv.
#   guard: OOM-Guard (tensor_guard.hpp) -- Grenze, Env-Auswertung, die 3 GETT-Faelle.
#   valid: Kernel-Validierung (kernel_validation.hpp) -- Fuellmuster, Toleranz,
#          Referenz-/Stichproben-Schwelle, Erkennung von Layout-/Nullausgabe-Fehlern.
#   einsum: Einstein-Notation -- Zerlegung, gesplittete Achsen, Strides, isGEMMForm.
#   parser: CSV -> IR (Eingangspunkt), inkl. Quotes und Fehlerfaelle.
#   passes: IR-Transformationen -- Split, Reorder, Parallelisierung.
#   cost:   Cost-Modell -- Kalibrierungs-Defaults, Schaetz-Invarianten, Breakdown.
test: test-c2 test-bench test-guard test-valid test-einsum test-parser test-passes test-cost

test-c2: einsum.o
	@$(CXX) $(CXXFLAGS) test_c2_validator.cpp einsum.o -o _test_c2
	@./_test_c2

test-bench:
	@$(CXX) $(CXXFLAGS) test_bench_loop.cpp -o _test_bench
	@./_test_bench

test-guard:
	@$(CXX) $(CXXFLAGS) test_tensor_guard.cpp -o _test_guard
	@./_test_guard

test-valid: einsum.o
	@$(CXX) $(CXXFLAGS) test_kernel_validation.cpp einsum.o -o _test_valid
	@./_test_valid

test-einsum: einsum.o
	@$(CXX) $(CXXFLAGS) test_einsum.cpp einsum.o -o _test_einsum
	@./_test_einsum

test-parser: parser.o
	@$(CXX) $(CXXFLAGS) test_parser.cpp parser.o -o _test_parser
	@./_test_parser

test-passes: passes.o
	@$(CXX) $(CXXFLAGS) test_passes.cpp passes.o -o _test_passes
	@./_test_passes

test-cost: cost_model.o einsum.o
	@$(CXX) $(CXXFLAGS) test_cost_model.cpp cost_model.o einsum.o -o _test_cost $(LDFLAGS)
	@./_test_cost

clean:
	@echo "Cleaning up..."
	@rm -f $(OBJS) $(TARGET) results.csv autotuner_results.csv generated_kernel.cpp generated_kernel.so _trial_*.cpp _trial_*.so _test_c2 _test_bench _test_guard _test_valid _test_einsum _test_parser _test_passes _test_cost

.PHONY: all clean test test-c2 test-bench test-guard test-valid test-einsum test-parser test-passes test-cost
