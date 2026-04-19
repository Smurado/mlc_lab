#define CATCH_CONFIG_MAIN  // Weist Catch2 an, eine main-Funktion zu generieren
#include "../lib/catch.hpp"
#include <cstdint>
#include <vector>
#include <chrono>
#include <iostream>
#include <locale>

// Deklaration der Assembly-Funktionen
extern "C" {
    void micro_benchmark(uint64_t iterations);
    void perm_neon_abc_cba( int64_t size_c, 
                            float const * abc, 
                            float       * cba);
}

TEST_CASE("Benchmark: micro_benchmark Funktion", "[micro_benchmark_bench]") {
    const uint64_t num_iterations = 100'000'000; // 100 Millionen Iterationen
    
    // Warmup
    micro_benchmark(10);
    
    // Benchmark mit Chrono
    auto start = std::chrono::high_resolution_clock::now();
    micro_benchmark(num_iterations);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double avg_ns = static_cast<double>(duration_ns.count()) / (num_iterations * 50*2*28);
    
    // Tausendertrennzeichen für die Ausgabe aktivieren
    std::cout.imbue(std::locale("de_DE.UTF-8"));

    std::cout << "\n=== Micro Benchmark Results ===\n";
    std::cout << "Iterations: " << num_iterations << "\n";
    std::cout << "Total Time: " << duration_ms.count() << " ms\n";
    std::cout << "Total Time: " << duration_us.count() << " us\n";
    std::cout << "Average Time per Call: " << avg_ns << " ns\n";
    std::cout << "==============================\n";
}

TEST_CASE("Einsum Permutation abc -> cba wird korrekt berechnet", "[perm_neon_abc_cba]") {
    // Gemaess Aufgabenstellung: Dimension a hat Groesse |a| = 8, Dimension b hat Groesse |b| = 4.
    const int A = 8;
    const int B = 4;
    
    // Wir testen verschiedene Groessen fuer C, da es ein Parameter ist
    const std::vector<int64_t> size_c_tests = {1, 2, 4, 8, 15, 32}; 

    for (int64_t size_c : size_c_tests) {
        const int total_elements = A * B * size_c;

        float abc[total_elements];
        float cba[total_elements];

        // Array mit Testwerten initialisieren
        for (int i = 0; i < total_elements; ++i) {
            abc[i] = static_cast<float>(i + 1);
            cba[i] = 0.0f; // Ergebnis-Array nullen
        }

        // Aufrufen der NEON Assembly-Funktion
        perm_neon_abc_cba(size_c, abc, cba);

        // Erwartetes Ergebnis ueberpruefen:
        // Tensor abc (Row-Major): Index(a, b, c) = a * (B * C) + b * C + c
        // Tensor cba (Row-Major): Index(c, b, a) = c * (B * A) + b * A + a
        for (int a = 0; a < A; ++a) {
            for (int b = 0; b < B; ++b) {
                for (int c = 0; c < size_c; ++c) {
                    int abc_idx = a * (B * size_c) + b * size_c + c;
                    int cba_idx = c * (B * A) + b * A + a;
                    
                    REQUIRE(cba[cba_idx] == abc[abc_idx]);
                }
            }
        }
    }
}

/*
TEST_CASE("Test Print abc to cba", "[print_perm]") {
    const int A = 8;
    const int B = 4;
    const int64_t size_c = 4;
    const int total_elements = A * B * size_c;

    float abc[total_elements];
    float cba[total_elements];

    // Array mit Testwerten initialisieren
    for (int i = 0; i < total_elements; ++i) {
        abc[i] = static_cast<float>(i + 1);
        cba[i] = 0.0f;
    }

    std::cout << "\nVorher: abc Array (" << A << "x" << B << "x" << size_c << "):\n\n";
    for (int a = 0; a < A; ++a) {
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < size_c; ++c) {
                int idx = a * (B * size_c) + b * size_c + c;
                std::cout << abc[idx] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n\n";
    }

    // Aufrufen der NEON Assembly-Funktion
    perm_neon_abc_cba(size_c, abc, cba);

    std::cout << "Nachher: cba Array (" << size_c << "x" << B << "x" << A << "):\n\n";
    for (int c = 0; c < size_c; ++c) {
        for (int b = 0; b < B; ++b) {
            for (int a = 0; a < A; ++a) {
                int idx = c * (B * A) + b * A + a;
                std::cout << cba[idx] << " ";
            }
            std::cout << "\n ";
        }
        std::cout << "\n\n";
    }
}
*/