#define CATCH_CONFIG_MAIN  // Weist Catch2 an, eine main-Funktion zu generieren
#include "../lib/catch.hpp"
#include <cstdint>
#include <vector>
#include <chrono>
#include <iostream>

// Deklaration der Assembly-Funktionen
extern "C" {
    void micro_benchmark(uint64_t iterations);
    void micro_benchmark_fmla_4s(uint64_t iterations);
    void micro_benchmark_fmla_2s(uint64_t iterations);
    void perm_neon_abc_cba(int64_t size_c, float const *abc, float *cba);
}


TEST_CASE("FMADD (scalar, FP32) Throughput", "[fmadd_bench]") {
    // Pro Iteration fuehrt die Assembly 50 * 28 = 1400 FMADDs aus.
    const uint64_t iterations = 100'000'000;
    const uint64_t instr      = iterations * 50 * 28;

    micro_benchmark(10); // Warmup

    auto start = std::chrono::high_resolution_clock::now();
    micro_benchmark(iterations);
    auto end   = std::chrono::high_resolution_clock::now();

    double ns     = std::chrono::duration<double, std::nano>(end - start).count();
    double gflops = (instr / ns) * 2.0; // 1 FMADD = 2 FLOPs (mul + add)

    std::cout << "\n[FMADD scalar] " << iterations << " iter, "
              << (ns / 1e6) << " ms, " << gflops << " GFLOP/s\n";
}

TEST_CASE("FMLA (vector) 4S Throughput", "[fmla4s_bench]") {
    const uint64_t iterations = 50'000'000;
    const uint64_t instr      = iterations * 50 * 28;

    micro_benchmark_fmla_4s(10);

    auto start = std::chrono::high_resolution_clock::now();
    micro_benchmark_fmla_4s(iterations);
    auto end   = std::chrono::high_resolution_clock::now();

    double ns     = std::chrono::duration<double, std::nano>(end - start).count();
    double gflops = (instr / ns) * 8.0; // 4 Lanes * (mul + add)

    std::cout << "\n[FMLA 4S]      " << iterations << " iter, "
              << (ns / 1e6) << " ms, " << gflops << " GFLOP/s\n";
}

TEST_CASE("FMLA (vector) 2S Throughput", "[fmla2s_bench]") {
    const uint64_t iterations = 50'000'000;
    const uint64_t instr      = iterations * 50 * 28;

    micro_benchmark_fmla_2s(10);

    auto start = std::chrono::high_resolution_clock::now();
    micro_benchmark_fmla_2s(iterations);
    auto end   = std::chrono::high_resolution_clock::now();

    double ns     = std::chrono::duration<double, std::nano>(end - start).count();
    double gflops = (instr / ns) * 4.0; // 2 Lanes * (mul + add)

    std::cout << "\n[FMLA 2S]      " << iterations << " iter, "
              << (ns / 1e6) << " ms, " << gflops << " GFLOP/s\n";
}

TEST_CASE("Permutation abc -> cba wird korrekt berechnet", "[perm_neon_abc_cba]") {
    // Laut Aufgabe: |a| = 8, |b| = 4, |c| ist Parameter.
    const int A = 8;
    const int B = 4;

    const std::vector<int64_t> size_c_tests = {1, 2, 4, 8, 15, 32};

    for (int64_t size_c : size_c_tests) {
        const int total = A * B * size_c;
        std::vector<float> abc(total), cba(total, 0.0f);

        for (int i = 0; i < total; ++i) {
            abc[i] = static_cast<float>(i + 1);
        }

        perm_neon_abc_cba(size_c, abc.data(), cba.data());

        // Erwartetes Ergebnis pruefen:
        // abc row-major: a*(B*C) + b*C + c
        // cba row-major: c*(B*A) + b*A + a
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

TEST_CASE("Permutation abc -> cba: Visualisierung", "[perm_print]") {
    // Kleines Beispiel zur visuellen Kontrolle: wir drucken abc
    // (A x B x C) und danach das permutierte cba (C x B x A).
    const int A = 8;
    const int B = 4;
    const int64_t size_c = 4;
    const int total = A * B * size_c;

    std::vector<float> abc(total), cba(total, 0.0f);
    for (int i = 0; i < total; ++i) {
        abc[i] = static_cast<float>(i + 1); // 1, 2, 3, ...
    }

    std::cout << "\n--- abc (A=" << A << ", B=" << B
              << ", C=" << size_c << ") ---\n";
    for (int a = 0; a < A; ++a) {
        std::cout << "a=" << a << ":\n";
        for (int b = 0; b < B; ++b) {
            std::cout << "  b=" << b << ": ";
            for (int c = 0; c < size_c; ++c) {
                int idx = a * (B * size_c) + b * size_c + c;
                std::cout << abc[idx] << " ";
            }
            std::cout << "\n";
        }
    }

    perm_neon_abc_cba(size_c, abc.data(), cba.data());

    std::cout << "\n--- cba nach Permutation (C=" << size_c
              << ", B=" << B << ", A=" << A << ") ---\n";
    for (int c = 0; c < size_c; ++c) {
        std::cout << "c=" << c << ":\n";
        for (int b = 0; b < B; ++b) {
            std::cout << "  b=" << b << ": ";
            for (int a = 0; a < A; ++a) {
                int idx = c * (B * A) + b * A + a;
                std::cout << cba[idx] << " ";
            }
            std::cout << "\n";
        }
    }
}

TEST_CASE("Permutation abc -> cba: Bandbreite in GiB/s", "[perm_bench]") {
    const int A = 8;
    const int B = 4;

    // Verschiedene |c| fuer den Performance-Report
    const std::vector<int64_t> sizes = {
        4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 16384, 65536, 262144
    };

    std::cout << "\n[Permutation] |c|  reps  time_ms  GiB/s\n";

    for (int64_t size_c : sizes) {
        const size_t total = static_cast<size_t>(A) * B * size_c;
        std::vector<float> abc(total), cba(total);
        for (size_t i = 0; i < total; ++i) abc[i] = static_cast<float>(i);

        // Ziel: pro Messung ca. 2^28 Elemente beruehren
        uint64_t reps = (1ULL << 28) / total;
        if (reps < 10) reps = 10;

        // Warmup
        for (int i = 0; i < 3; ++i) {
            perm_neon_abc_cba(size_c, abc.data(), cba.data());
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < reps; ++i) {
            perm_neon_abc_cba(size_c, abc.data(), cba.data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(t1 - t0).count();
        double ms      = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // read + write = 2 * total * sizeof(float)
        double bytes   = 2.0 * total * sizeof(float) * reps;
        double gib_s   = bytes / seconds / (1024.0 * 1024.0 * 1024.0);

        std::cout << "              " << size_c << "  " << reps
                  << "  " << ms << "  " << gib_s << "\n";
    }
}
