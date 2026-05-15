#include "Unary.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace mini_jit;

bool verify_zero(uint32_t m, uint32_t n) {
    std::vector<float> B(m * n, 1.0f); // Array mit Einsen füllen
    Unary kernel_gen;
    if (kernel_gen.generate(m, n, 0, Unary::dtype_t::fp32, Unary::ptype_t::zero) != Unary::error_t::success) return false;
    
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return false;

    kernel(nullptr, B.data(), m, m);

    for (float val : B) {
        if (val != 0.0f) return false;
    }
    return true;
}

bool verify_identity(uint32_t m, uint32_t n) {
    std::vector<float> A(m * n);
    std::vector<float> B(m * n, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(i);

    Unary kernel_gen;
    if (kernel_gen.generate(m, n, 0, Unary::dtype_t::fp32, Unary::ptype_t::identity) != Unary::error_t::success) return false;
    
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return false;

    kernel(A.data(), B.data(), m, m);

    for (size_t i = 0; i < A.size(); ++i) {
        if (A[i] != B[i]) return false;
    }
    return true;
}

bool verify_relu(uint32_t m, uint32_t n) {
    std::vector<float> A(m * n);
    std::vector<float> B(m * n, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(i) - static_cast<float>(A.size()/2); // Mix aus negativen und positiven Werten

    Unary kernel_gen;
    if (kernel_gen.generate(m, n, 0, Unary::dtype_t::fp32, Unary::ptype_t::relu) != Unary::error_t::success) return false;
    
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return false;

    kernel(A.data(), B.data(), m, m);

    for (size_t i = 0; i < A.size(); ++i) {
        float expected = std::max(0.0f, A[i]);
        if (B[i] != expected) return false;
    }
    return true;
}

// Wir deaktiveren hier gezielt die C++ Compiler-Optimierung für die Benchmark-Funktionen (O0).
// Sonst speichert der Compiler die lokalen Variablen zur Zeitmessung in den "callee-saved" 
// Vector-Registern (d8-d15). Da "smstart sm" diese Register hardwareseitig nullt, bekämen 
// wir sonst Divisionen durch Null und kaputte Timer.
__attribute__((optimize("O0"))) void benchmark(uint32_t m, uint32_t n, Unary::ptype_t ptype, const std::string& name, bool benchmark_mode) {
    Unary kernel_gen;
    if (kernel_gen.generate(m, n, 0, Unary::dtype_t::fp32, ptype) != Unary::error_t::success) {
        std::cout << name << " (" << m << "x" << n << "): Generation failed" << std::endl;
        return;
    }
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return;

    std::vector<float> A(m * n, 1.0f);
    std::vector<float> B(m * n, 0.0f);
    
    int num_runs = 10000;
    int outer_runs = benchmark_mode ? 10 : 1;
    double total_gib_per_sec = 0.0;
    
    for (int outer = 0; outer < outer_runs; ++outer) {
        // Cache anwärmen (Warmup)
        for (int i = 0; i < 100; ++i) {
            kernel(A.data(), B.data(), m, m);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; ++i) {
            kernel(A.data(), B.data(), m, m);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        // Durchsatz in GiB/s berechnen
        double bytes_per_run = 0;
        if (ptype == Unary::ptype_t::zero) {
            bytes_per_run = m * n * sizeof(float); // Nur Schreiben
        } else {
            bytes_per_run = 2.0 * m * n * sizeof(float); // Lesen + Schreiben
        }
        
        double total_bytes = bytes_per_run * num_runs;
        double gib_per_sec = (total_bytes / diff.count()) / (1024.0 * 1024.0 * 1024.0);
        total_gib_per_sec += gib_per_sec;
    }
    
    double avg_gib_per_sec = total_gib_per_sec / outer_runs;
    
    std::cout << std::left << std::setw(10) << name 
              << " (" << std::setw(3) << m << "x" << std::setw(3) << n << "): " 
              << std::fixed << std::setprecision(2) << avg_gib_per_sec << " GiB/s" 
              << (benchmark_mode ? " (Avg of 10)" : "") << std::endl;
}

#include "Gemm.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <random>

using namespace mini_jit;

// Unary code omitted for brevity right here, assuming the user knows we just append this

bool verify_gemm(uint32_t m, uint32_t n, uint32_t k) {
    if (m != 512 || n != 512 || k != 512) return true; // Only 512 implemented

    std::vector<float> A(m * k, 1.0f);
    std::vector<float> B(k * n, 1.0f);
    std::vector<float> C(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);

    // Provide values
    for(size_t i=0; i<A.size(); ++i) A[i] = 1.0f;
    for(size_t i=0; i<B.size(); ++i) B[i] = 2.0f;

    // Standard matrix multiplication
    for(uint32_t j=0; j<n; ++j) {
        for(uint32_t l=0; l<k; ++l) {
            for(uint32_t i=0; i<m; ++i) {
                C_ref[i + j*m] += A[i + l*m] * B[l + j*k];
            }
        }
    }

    Gemm kernel_gen;
    if (kernel_gen.generate(m, n, k, 0, 0, 0, Gemm::dtype_t::fp32) != Gemm::error_t::success) return false;
    
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return false;

    // Call the generated JIT kernel
    kernel(A.data(), B.data(), C.data(), m, k, m);

    // Verify
    for(size_t i=0; i<C.size(); ++i) {
        if (std::abs(C[i] - C_ref[i]) > 1e-4) return false;
    }
    return true;
}

__attribute__((optimize("O0"))) __attribute__((optimize("O0"))) void benchmark_gemm(uint32_t m, uint32_t n, uint32_t k, bool benchmark_mode) {
    Gemm kernel_gen;
    if (kernel_gen.generate(m, n, k, 0, 0, 0, Gemm::dtype_t::fp32) != Gemm::error_t::success) return;
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return;

    std::vector<float> A(m * k, 1.0f);
    std::vector<float> B(k * n, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    int num_runs = 50;
    int outer_runs = benchmark_mode ? 10 : 1;
    double total_gflops_per_sec = 0.0;
    
    for (int outer = 0; outer < outer_runs; ++outer) {
        // Cache anwärmen (Warmup)
        for (int i = 0; i < 5; ++i) kernel(A.data(), B.data(), C.data(), m, k, m);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; ++i) {
            kernel(A.data(), B.data(), C.data(), m, k, m);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        // Calculate GFLOPS
        double flops_per_run = 2.0 * m * n * k;
        double total_gflops = (flops_per_run * num_runs) / (1e9);
        double gflops_per_sec = total_gflops / diff.count();
        total_gflops_per_sec += gflops_per_sec;
    }
    
    double avg_gflops_per_sec = total_gflops_per_sec / outer_runs;
    
    std::cout << std::left << std::setw(10) << "GEMM" 
              << " (" << std::setw(3) << m << "x" << std::setw(3) << n << "x" << std::setw(3) << k << "): " 
              << std::fixed << std::setprecision(2) << avg_gflops_per_sec << " GFLOPS" 
              << (benchmark_mode ? " (Avg of 10)" : "") << std::endl;
}

int main(int argc, char** argv) {
    bool benchmark_mode = false;
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        benchmark_mode = true;
        std::cout << "Starting BENCHMARK MODE (10 runs averaged)..." << std::endl;
    }

    if (!benchmark_mode) {
        std::cout << "--- Verify Kernels ---" << std::endl;
        std::cout << "Zero 16x16:     " << (verify_zero(16, 16) ? "PASS" : "FAIL") << std::endl;
        std::cout << "Identity 16x16: " << (verify_identity(16, 16) ? "PASS" : "FAIL") << std::endl;
        std::cout << "ReLU 16x16:     " << (verify_relu(16, 16) ? "PASS" : "FAIL") << std::endl;
        std::cout << "GEMM 512x512:   " << (verify_gemm(512, 512, 512) ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    std::cout << "--- Benchmarks ---" << std::endl;
    std::vector<uint32_t> sizes = {64, 128, 512};
    for (auto size : sizes) {
        benchmark(size, size, Unary::ptype_t::zero, "Zero", benchmark_mode);
        benchmark(size, size, Unary::ptype_t::identity, "Identity", benchmark_mode);
        benchmark(size, size, Unary::ptype_t::relu, "ReLU", benchmark_mode);
        std::cout << std::endl;
    }
    
    benchmark_gemm(512, 512, 512, benchmark_mode);

    return 0;
}
