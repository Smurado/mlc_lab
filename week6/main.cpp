#include "Unary.h"
#include "Gemm.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <string>

using namespace mini_jit;

// =====================================================================
// Verifikation der Unary-Kernel
// Jede Verifikationsfunktion testet die Ausgabe des JIT-kompilierten
// Kernels gegen eine einfache C++-Referenzimplementierung für kleine Sizes.
// =====================================================================
static bool verify_zero(uint32_t m, uint32_t n) {
    // Initialisiere Vektor B mit 1.0f als Dummy-Wert.
    std::vector<float> B(m * n, 1.0f);
    Unary kernel_gen;
    
    // Generiere den Kernel für Zero (setzt alle Werte auf 0).
    if (kernel_gen.generate(m, n, 0, Unary::dtype_t::fp32, Unary::ptype_t::zero) != Unary::error_t::success) return false;
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return false;
    
    // ld_a wird für 'zero' nicht verwendet (a = nullptr). ld_b = m (column-major).
    kernel(nullptr, B.data(), m, m);
    
    // Verifiziere, dass wirklich alles 0.0f ist.
    for (float val : B) if (val != 0.0f) return false;
    return true;
}

static bool verify_identity(uint32_t m, uint32_t n) {
    // A erhält ansteigende Werte als Input, B initialisieren wir mit 0.
    std::vector<float> A(m * n);
    std::vector<float> B(m * n, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(i);

    Unary kernel_gen;
    // Generiere den Identity Kernel (kopiert A nach B).
    if (kernel_gen.generate(m, n, 0, Unary::dtype_t::fp32, Unary::ptype_t::identity) != Unary::error_t::success) return false;
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return false;
    
    kernel(A.data(), B.data(), m, m);
    
    // Verifiziere, dass B nun A entspricht.
    for (size_t i = 0; i < A.size(); ++i) if (A[i] != B[i]) return false;
    return true;
}

static bool verify_relu(uint32_t m, uint32_t n) {
    // A enthält zur Hälfte negative und zur Hälfte positive Werte.
    std::vector<float> A(m * n);
    std::vector<float> B(m * n, 0.0f);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = static_cast<float>(i) - static_cast<float>(A.size() / 2);

    Unary kernel_gen;
    // Generiere den ReLU Kernel (max(x, 0)).
    if (kernel_gen.generate(m, n, 0, Unary::dtype_t::fp32, Unary::ptype_t::relu) != Unary::error_t::success) return false;
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return false;
    
    kernel(A.data(), B.data(), m, m);
    
    // Verifiziere gegen std::max(0.0f, A).
    for (size_t i = 0; i < A.size(); ++i) {
        float expected = std::max(0.0f, A[i]);
        if (B[i] != expected) return false;
    }
    return true;
}

// =====================================================================
// Verifikation der GEMM-Kernel
// Layout:  A col-major (MxK), B row-major (KxN), C col-major (MxN).
// Aufruf:  kernel(A, B, C, ld_a=M, ld_b=N, ld_c=M).
// =====================================================================
static bool verify_gemm(uint32_t m, uint32_t n, uint32_t k) {
    std::vector<float> A(static_cast<size_t>(m) * k);
    std::vector<float> B(static_cast<size_t>(k) * n);
    std::vector<float> C(static_cast<size_t>(m) * n, 0.0f);
    std::vector<float> C_ref(static_cast<size_t>(m) * n, 0.0f);

    // Deterministische, nicht-triviale Werte (damit Layout-Fehler auch wirklich auffallen).
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>((i % 7) - 3) * 0.5f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = static_cast<float>((i % 5) - 2) * 0.25f;

    // C_ref += A * B
    //   A column-major: A[i, l] = A[i + l*m]
    //   B row-major   : B[l, j] = B[l*n + j]
    //   C column-major: C[i, j] = C[i + j*m]
    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t l = 0; l < k; ++l) {
            float b_lj = B[static_cast<size_t>(l) * n + j];
            for (uint32_t i = 0; i < m; ++i) {
                C_ref[static_cast<size_t>(i) + static_cast<size_t>(j) * m] +=
                    A[static_cast<size_t>(i) + static_cast<size_t>(l) * m] * b_lj;
            }
        }
    }

    Gemm kernel_gen;
    if (kernel_gen.generate(m, n, k, 0, 0, 0, Gemm::dtype_t::fp32) != Gemm::error_t::success) return false;
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return false;

    kernel(A.data(), B.data(), C.data(), m, n, m);

    // Toleranz proportional zu k (Akkumulationsfehler bei FP32).
    const float tol = 1e-3f * static_cast<float>(k);
    for (size_t i = 0; i < C.size(); ++i) {
        if (std::abs(C[i] - C_ref[i]) > tol) return false;
    }
    return true;
}

// =====================================================================
// Benchmarks
// Messmethodik zum Performance-Vorgleich. Um Fluktuationen und Init-Kosten 
// zu verringern, gibt es optionale Runs (--benchmark) und einen Warm-up.
// =====================================================================
static void benchmark_unary(uint32_t m, uint32_t n, Unary::ptype_t ptype,
                            const std::string& name, bool benchmark_mode) {
    Unary kernel_gen;
    if (kernel_gen.generate(m, n, 0, Unary::dtype_t::fp32, ptype) != Unary::error_t::success) {
        std::cout << name << " (" << m << "x" << n << "): Generation failed" << std::endl;
        return;
    }
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return;

    std::vector<float> A(static_cast<size_t>(m) * n, 1.0f);
    std::vector<float> B(static_cast<size_t>(m) * n, 0.0f);

    int num_runs    = 10000;
    int outer_runs  = benchmark_mode ? 10 : 1;
    double total_gib_per_sec = 0.0;

    for (int outer = 0; outer < outer_runs; ++outer) {
        // Warm-up, um Caches aufzuwärmen und Taktung anzupassen
        for (int i = 0; i < 100; ++i) kernel(A.data(), B.data(), m, m);

        auto start = std::chrono::high_resolution_clock::now();
        // Eigentlicher Mess-Durchlauf
        for (int i = 0; i < num_runs; ++i) kernel(A.data(), B.data(), m, m);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        // Berechnung der Speicherbandbreite. 'zero' liest nicht von A (nur out), 
        // Identity und ReLU lesen und schreiben (2x I/O).
        double bytes_per_run = (ptype == Unary::ptype_t::zero)
            ? static_cast<double>(m) * n * sizeof(float)
            : 2.0 * m * n * sizeof(float);
        double total_bytes = bytes_per_run * num_runs;
        double gib_per_sec = (total_bytes / diff.count()) / (1024.0 * 1024.0 * 1024.0);
        total_gib_per_sec += gib_per_sec;
    }

    double avg = total_gib_per_sec / outer_runs;
    std::cout << std::left << std::setw(10) << name
              << " (" << std::setw(3) << m << "x" << std::setw(3) << n << "): "
              << std::fixed << std::setprecision(3) << avg << " GiB/s"
              << (benchmark_mode ? " (avg of 10)" : "") << std::endl;
}

static void benchmark_gemm(uint32_t m, uint32_t n, uint32_t k, bool benchmark_mode) {
    Gemm kernel_gen;
    if (kernel_gen.generate(m, n, k, 0, 0, 0, Gemm::dtype_t::fp32) != Gemm::error_t::success) {
        std::cout << "GEMM (" << m << "x" << n << "x" << k << "): Generation failed" << std::endl;
        return;
    }
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return;

    std::vector<float> A(static_cast<size_t>(m) * k, 1.0f);
    std::vector<float> B(static_cast<size_t>(k) * n, 1.0f);
    std::vector<float> C(static_cast<size_t>(m) * n, 0.0f);

    // Wiederholungen dynamisch so wählen, dass für jede Matrixgröße 
    // ca. ~5 GFLOPs Rechen-Arbeit als Grundlast anliegen.
    const double target_flops = 5e9; 
    double flops_per_run = 2.0 * m * n * k; // 2 Operations (Multiply + Add) pro Skalar
    int num_runs = std::max(20, static_cast<int>(target_flops / flops_per_run));

    int outer_runs = benchmark_mode ? 10 : 1;
    double total_gflops_per_sec = 0.0;

    for (int outer = 0; outer < outer_runs; ++outer) {
        // Kurzer Warm-up für SME 
        for (int i = 0; i < 5; ++i) kernel(A.data(), B.data(), C.data(), m, n, m);

        auto start = std::chrono::high_resolution_clock::now();
        // Zeitmessung für die tatsächliche GEMM Workload
        for (int i = 0; i < num_runs; ++i) kernel(A.data(), B.data(), C.data(), m, n, m);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        // Gesamt GFLOPS iterieren...
        double total_gflops = (flops_per_run * num_runs) / 1e9;
        double gflops_per_sec = total_gflops / diff.count();
        total_gflops_per_sec += gflops_per_sec;
    }

    double avg = total_gflops_per_sec / outer_runs;
    std::cout << std::left << std::setw(6) << "GEMM"
              << " (" << std::setw(3) << m << "x" << std::setw(3) << n
              << "x" << std::setw(3) << k << "): "
              << std::fixed << std::setprecision(2) << avg << " GFLOPS"
              << (benchmark_mode ? " (avg of 10)" : "") << std::endl;
}

int main(int argc, char** argv) {
    bool benchmark_mode = false;
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        benchmark_mode = true;
        std::cout << "Starting BENCHMARK MODE (10 runs averaged)..." << std::endl;
    }

    if (!benchmark_mode) {
        std::cout << "--- Verify Kernels ---" << std::endl;
        std::cout << "Zero 16x16:        " << (verify_zero(16, 16)      ? "PASS" : "FAIL") << std::endl;
        std::cout << "Identity 16x16:    " << (verify_identity(16, 16)  ? "PASS" : "FAIL") << std::endl;
        std::cout << "ReLU 16x16:        " << (verify_relu(16, 16)      ? "PASS" : "FAIL") << std::endl;
        std::cout << "Identity 64x128:   " << (verify_identity(64, 128) ? "PASS" : "FAIL") << std::endl;
        std::cout << "GEMM 64x64x64:     " << (verify_gemm(64, 64, 64)   ? "PASS" : "FAIL") << std::endl;
        std::cout << "GEMM 128x64x32:    " << (verify_gemm(128, 64, 32)  ? "PASS" : "FAIL") << std::endl;
        std::cout << "GEMM 512x512x512:  " << (verify_gemm(512, 512, 512)? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // -------- Unary: 9 Settings (M x N) ------------------------------
    std::cout << "--- Unary Benchmarks (9 settings each) ---" << std::endl;
    std::vector<uint32_t> sizes = {64, 128, 512};
    for (auto m_dim : sizes) {
        for (auto n_dim : sizes) {
            benchmark_unary(m_dim, n_dim, Unary::ptype_t::zero,     "Zero",     benchmark_mode);
            benchmark_unary(m_dim, n_dim, Unary::ptype_t::identity, "Identity", benchmark_mode);
            benchmark_unary(m_dim, n_dim, Unary::ptype_t::relu,     "ReLU",     benchmark_mode);
            std::cout << std::endl;
        }
    }

    // -------- GEMM: 27 Settings (M x N x K) --------------------------
    std::cout << "--- GEMM Benchmarks (27 settings) ---" << std::endl;
    for (auto m_dim : sizes) {
        for (auto n_dim : sizes) {
            for (auto k_dim : sizes) {
                benchmark_gemm(m_dim, n_dim, k_dim, benchmark_mode);
            }
        }
    }

    return 0;
}

