// compilieren mit: /opt/homebrew/opt/llvm/bin/clang++ main.cpp functions.s -o main -march=armv9.2-a+sme

#define CATCH_CONFIG_MAIN  // Weist Catch2 an, eine main-Funktion zu generieren
#include "../lib/catch.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>

// Helper: Erstellt eine Matrix mit Testdaten
std::vector<float> create_matrix(int rows, int cols, float start_val = 1.0f) {
    std::vector<float> m(rows * cols);
    for (int i = 0; i < rows * cols; ++i) m[i] = start_val + i;
    return m;
}

// Deklaration der Assembly-Funktionen
extern "C" {
    void identity_16_16( float const * a,
                       float       * b,
                       int64_t       ld_a,
                       int64_t       ld_b,
                       int32_t       trans_b );
    void zero_16_16( float const * a,
                   int64_t       ld_a );
    void relu_16_16( float const * a,
                   float       * b,
                   int64_t       ld_a,
                   int64_t       ld_b,
                   int32_t       trans_b );
    void gemm_32_32_1( float   const * a,
                      float   const * b,
                      float         * c,
                      int64_t         ld_a,
                      int64_t         ld_b,
                      int64_t         ld_c );
    void gemm_32_32_512( float   const * a,
                        float   const * b,
                        float         * c,
                        int64_t         ld_a,
                        int64_t         ld_b,
                        int64_t         ld_c );
    void gemm_512_32_512( float   const * a,
                         float   const * b,
                         float         * c,
                         int64_t         ld_a,
                         int64_t         ld_b,
                         int64_t         ld_c );
    void gemm_512_512_512( float   const * a,
                          float   const * b,
                          float         * c,
                          int64_t         ld_a,
                          int64_t         ld_b,
                          int64_t         ld_c );
}


TEST_CASE("identity_16_16: Kopiert Matrix korrekt", "[identity_16_16]") {
    auto a = create_matrix(16, 16);
    std::vector<float> b(16 * 16, 0.0f);

    SECTION("Column-major zu Column-major (trans_b = 0)") {
        identity_16_16(a.data(), b.data(), 16, 16, 0);
        REQUIRE(b == a);
    }

    SECTION("Column-major zu Row-major (trans_b = 1)") {
        identity_16_16(a.data(), b.data(), 16, 16, 1);
        for(int i=0; i<16; ++i)
            for(int j=0; j<16; ++j)
                CHECK(b[i * 16 + j] == a[j * 16 + i]);
    }
}


TEST_CASE("zero_16_16: Setzt Matrix auf Null", "[zero_16_16]") {
    auto a = create_matrix(16, 16);
    zero_16_16(a.data(), 16);
    
    for (float val : a) {
        REQUIRE(val == 0.0f);
    }
}

TEST_CASE("relu_16_16: Aktivierungsfunktion", "[relu_16_16]") {
    std::vector<float> a(16 * 16);
    for(int i=0; i<256; ++i) a[i] = (i % 2 == 0) ? (float)i : -(float)i;
    std::vector<float> b(16 * 16, 0.0f);

    relu_16_16(a.data(), b.data(), 16, 16, 0);

    for(int i=0; i<256; ++i) {
        float expected = std::max(0.0f, a[i]);
        CHECK(b[i] == expected);
    }
}

TEST_CASE("gemm_32_32_1: Matrix-Multiplikation (Rank-1 Update)", "[gemm_32_32_1]") {
    // 32x1 * 1x32 = 32x32
    auto a = create_matrix(32, 1);
    auto b = create_matrix(1, 32);
    std::vector<float> c(32 * 32, 0.0f);

    gemm_32_32_1(a.data(), b.data(), c.data(), 32, 1, 32);

    // C column-major: C[i,j] = c[j*32 + i] = A[i,0] * B[0,j] = a[i] * b[j]
    for (int j = 0; j < 32; ++j) {
        for (int i = 0; i < 32; ++i) {
            CHECK(c[j * 32 + i] == a[i] * b[j]);
        }
    }
}

// Helper: Naive Referenz-GEMM
// A column-major (M x K), B row-major (K x N), C column-major (M x N)
static void gemm_ref(const float* a, const float* b, float* c,
                     int M, int N, int K,
                     int64_t ld_a, int64_t ld_b, int64_t ld_c) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                // A[i,k] = a[k*ld_a + i]   (column-major)
                // B[k,j] = b[k*ld_b + j]   (row-major)
                acc += a[k * ld_a + i] * b[k * ld_b + j];
            }
            c[j * ld_c + i] += acc;
        }
    }
}

TEST_CASE("gemm_32_32_512: K-Loop Microkernel", "[gemm_32_32_512]") {
    const int M = 32, N = 32, K = 512;
    std::vector<float> a(M * K), b(K * N), c(M * N, 0.0f), c_ref(M * N, 0.0f);
    for (int i = 0; i < (int)a.size(); ++i) a[i] = (i % 13) * 0.125f - 0.5f;
    for (int i = 0; i < (int)b.size(); ++i) b[i] = (i % 17) * 0.0625f - 0.25f;

    gemm_32_32_512(a.data(), b.data(), c.data(), M, N, M);
    gemm_ref(a.data(), b.data(), c_ref.data(), M, N, K, M, N, M);

    for (int i = 0; i < M * N; ++i) {
        CHECK(c[i] == Approx(c_ref[i]).margin(1e-2));
    }
}

TEST_CASE("gemm_512_32_512: M-Loop", "[gemm_512_32_512]") {
    const int M = 512, N = 32, K = 512;
    std::vector<float> a(M * K), b(K * N), c(M * N, 0.0f), c_ref(M * N, 0.0f);
    for (int i = 0; i < (int)a.size(); ++i) a[i] = (i % 13) * 0.125f - 0.5f;
    for (int i = 0; i < (int)b.size(); ++i) b[i] = (i % 17) * 0.0625f - 0.25f;

    gemm_512_32_512(a.data(), b.data(), c.data(), M, N, M);
    gemm_ref(a.data(), b.data(), c_ref.data(), M, N, K, M, N, M);

    for (int i = 0; i < M * N; ++i) {
        REQUIRE(c[i] == Approx(c_ref[i]).margin(1e-1));
    }
}

TEST_CASE("gemm_512_512_512: Voller GEMM (N-Loop)", "[gemm_512_512_512]") {
    const int M = 512, N = 512, K = 512;
    std::vector<float> a(M * K), b(K * N), c(M * N, 0.0f), c_ref(M * N, 0.0f);
    for (int i = 0; i < (int)a.size(); ++i) a[i] = (i % 13) * 0.125f - 0.5f;
    for (int i = 0; i < (int)b.size(); ++i) b[i] = (i % 17) * 0.0625f - 0.25f;

    gemm_512_512_512(a.data(), b.data(), c.data(), M, N, M);
    gemm_ref(a.data(), b.data(), c_ref.data(), M, N, K, M, N, M);

    for (int i = 0; i < M * N; ++i) {
        REQUIRE(c[i] == Approx(c_ref[i]).margin(1e-1));
    }
}

TEST_CASE("Performance Benchmarks (SME GEMM)", "[benchmark]") {
    std::cout << "\n========================================\n";
    std::cout << "           SME GEMM BENCHMARKS          \n";
    std::cout << "========================================\n";

    auto run_gemm_bench = [](const char* name, auto func, int M, int N, int K, int runs = 100) {
        std::vector<float> a(M * K, 1.0f), b(K * N, 1.0f), c(M * N, 0.0f);
        
        // Warmup
        func(a.data(), b.data(), c.data(), M, N, M);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; ++i) {
            func(a.data(), b.data(), c.data(), M, N, M);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        double avg_seconds = diff.count() / runs;
        double flops = 2.0 * M * N * K;
        double gflops = (flops / avg_seconds) / 1e9;
        
        std::cout << std::left << std::setw(20) << name 
                  << ": " << std::fixed << std::setprecision(2) << std::setw(8) << gflops << " GFLOPS" 
                  << " (Avg Time: " << avg_seconds * 1000.0 << " ms)\n";
    };

    run_gemm_bench("gemm_32_32_1", gemm_32_32_1, 32, 32, 1, 10000);
    run_gemm_bench("gemm_32_32_512", gemm_32_32_512, 32, 32, 512, 1000);
    run_gemm_bench("gemm_512_32_512", gemm_512_32_512, 512, 32, 512, 100);
    run_gemm_bench("gemm_512_512_512", gemm_512_512_512, 512, 512, 512, 10);
    
    std::cout << "========================================\n\n";
}

TEST_CASE("Performance Benchmarks (SVE Unary)", "[benchmark]") {
    std::cout << "\n========================================\n";
    std::cout << "          SVE UNARY BENCHMARKS          \n";
    std::cout << "========================================\n";

    auto run_unary_bench = [](const char* name, auto func, int M, int N, int bytes_accessed, int runs = 100000) {
        std::vector<float> a(M * N, 1.0f);
        std::vector<float> b(M * N, 0.0f);
        
        // Warmup
        func(a.data(), b.data(), M, M, 0);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; ++i) {
            func(a.data(), b.data(), M, M, 0);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        double avg_seconds = diff.count() / runs;
        double total_bytes = bytes_accessed;
        double gib_per_sec = (total_bytes / avg_seconds) / (1024.0 * 1024.0 * 1024.0);
        
        std::cout << std::left << std::setw(20) << name 
                  << ": " << std::fixed << std::setprecision(2) << std::setw(8) << gib_per_sec << " GiB/s" 
                  << " (Avg Time: " << avg_seconds * 1000000.0 << " us)\n";
    };

    auto run_zero_bench = [](const char* name, auto func, int M, int N, int bytes_accessed, int runs = 100000) {
        std::vector<float> a(M * N, 1.0f);
        
        // Warmup
        func(a.data(), M);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; ++i) {
            func(a.data(), M);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        double avg_seconds = diff.count() / runs;
        double total_bytes = bytes_accessed;
        double gib_per_sec = (total_bytes / avg_seconds) / (1024.0 * 1024.0 * 1024.0);
        
        std::cout << std::left << std::setw(20) << name 
                  << ": " << std::fixed << std::setprecision(2) << std::setw(8) << gib_per_sec << " GiB/s" 
                  << " (Avg Time: " << avg_seconds * 1000000.0 << " us)\n";
    };

    // identity: read A (16x16x4), write B (16x16x4) -> 2048 bytes
    run_unary_bench("identity_16_16", identity_16_16, 16, 16, 2048);
    // zero: write A (16x16x4) -> 1024 bytes
    run_zero_bench("zero_16_16", zero_16_16, 16, 16, 1024);
    // relu: read A (16x16x4), write B (16x16x4) -> 2048 bytes
    run_unary_bench("relu_16_16", relu_16_16, 16, 16, 2048);
    
    std::cout << "========================================\n\n";
}
