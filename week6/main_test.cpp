#define CATCH_CONFIG_MAIN
#include "../lib/catch.hpp"
#include "Unary.h"
#include "Gemm.h"
#include <vector>
#include <cmath>
#include <cstddef>

using namespace mini_jit;

// ---------------------------------------------------------------------
// Hilfs-Referenzen (naive C++ Implementierungen).
// ---------------------------------------------------------------------
namespace {

void ref_zero(std::vector<float>& B) {
    std::fill(B.begin(), B.end(), 0.0f);
}

void ref_identity(const std::vector<float>& A, std::vector<float>& B) {
    REQUIRE(A.size() == B.size());
    for (size_t i = 0; i < A.size(); ++i) B[i] = A[i];
}

void ref_relu(const std::vector<float>& A, std::vector<float>& B) {
    REQUIRE(A.size() == B.size());
    for (size_t i = 0; i < A.size(); ++i) B[i] = std::max(0.0f, A[i]);
}

// A col-major (MxK), B row-major (KxN), C col-major (MxN).
void ref_gemm(uint32_t m, uint32_t n, uint32_t k,
              const std::vector<float>& A,
              const std::vector<float>& B,
              std::vector<float>& C) {
    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t l = 0; l < k; ++l) {
            float b_lj = B[static_cast<size_t>(l) * n + j];
            for (uint32_t i = 0; i < m; ++i) {
                C[static_cast<size_t>(i) + static_cast<size_t>(j) * m] +=
                    A[static_cast<size_t>(i) + static_cast<size_t>(l) * m] * b_lj;
            }
        }
    }
}

} // namespace

// =====================================================================
// Unary-Kernel
// =====================================================================
TEST_CASE("Unary Kernel Verification", "[unary]") {
    SECTION("Identity Kernel (64x64)") {
        const uint32_t M = 64, N = 64;
        std::vector<float> A(M * N), B(M * N, -1.0f), Bref(M * N);
        for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(i) - 100.0f;
        ref_identity(A, Bref);

        Unary gen;
        REQUIRE(gen.generate(M, N, 0, Unary::dtype_t::fp32, Unary::ptype_t::identity) == Unary::error_t::success);
        auto k = gen.get_kernel();
        REQUIRE(k != nullptr);
        k(A.data(), B.data(), M, M);

        for (size_t i = 0; i < B.size(); ++i) REQUIRE(B[i] == Bref[i]);
    }

    SECTION("Zero Kernel (128x128)") {
        const uint32_t M = 128, N = 128;
        std::vector<float> B(M * N, 42.0f), Bref(M * N, 42.0f);
        ref_zero(Bref);

        Unary gen;
        REQUIRE(gen.generate(M, N, 0, Unary::dtype_t::fp32, Unary::ptype_t::zero) == Unary::error_t::success);
        auto k = gen.get_kernel();
        REQUIRE(k != nullptr);
        k(nullptr, B.data(), M, M);

        for (size_t i = 0; i < B.size(); ++i) REQUIRE(B[i] == Bref[i]);
    }

    SECTION("ReLU Kernel (512x512)") {
        const uint32_t M = 512, N = 512;
        std::vector<float> A(M * N), B(M * N, 0.0f), Bref(M * N);
        for (size_t i = 0; i < A.size(); ++i)
            A[i] = static_cast<float>(static_cast<long long>(i) - static_cast<long long>(A.size() / 2));
        ref_relu(A, Bref);

        Unary gen;
        REQUIRE(gen.generate(M, N, 0, Unary::dtype_t::fp32, Unary::ptype_t::relu) == Unary::error_t::success);
        auto k = gen.get_kernel();
        REQUIRE(k != nullptr);
        k(A.data(), B.data(), M, M);

        for (size_t i = 0; i < B.size(); ++i) REQUIRE(B[i] == Bref[i]);
    }

    SECTION("Identity rechteckig (64x128)") {
        const uint32_t M = 64, N = 128;
        std::vector<float> A(M * N), B(M * N, -1.0f), Bref(M * N);
        for (size_t i = 0; i < A.size(); ++i) A[i] = std::sin(static_cast<float>(i));
        ref_identity(A, Bref);

        Unary gen;
        REQUIRE(gen.generate(M, N, 0, Unary::dtype_t::fp32, Unary::ptype_t::identity) == Unary::error_t::success);
        auto k = gen.get_kernel();
        REQUIRE(k != nullptr);
        k(A.data(), B.data(), M, M);

        for (size_t i = 0; i < B.size(); ++i) REQUIRE(B[i] == Bref[i]);
    }
}

// =====================================================================
// GEMM-Kernel
// =====================================================================
namespace {
void run_gemm_check(uint32_t M, uint32_t N, uint32_t K) {
    std::vector<float> A(static_cast<size_t>(M) * K);
    std::vector<float> B(static_cast<size_t>(K) * N);
    std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> Cref(static_cast<size_t>(M) * N, 0.0f);

    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>((i % 11) - 5) * 0.125f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = static_cast<float>((i % 13) - 6) * 0.0625f;

    ref_gemm(M, N, K, A, B, Cref);

    Gemm gen;
    REQUIRE(gen.generate(M, N, K, 0, 0, 0, Gemm::dtype_t::fp32) == Gemm::error_t::success);
    auto k = gen.get_kernel();
    REQUIRE(k != nullptr);
    k(A.data(), B.data(), C.data(), M, N, M);

    const float tol = 1e-3f * static_cast<float>(K);
    for (size_t i = 0; i < C.size(); ++i) {
        REQUIRE(std::abs(C[i] - Cref[i]) <= tol);
    }
}
} // namespace

TEST_CASE("GEMM Kernel Verification", "[gemm]") {
    SECTION("GEMM Small (64x64x64)")           { run_gemm_check(64, 64, 64); }
    SECTION("GEMM Mixed (64x128x32)")          { run_gemm_check(64, 128, 32); }
    SECTION("GEMM Medium (128x128x128)")       { run_gemm_check(128, 128, 128); }
    SECTION("GEMM Skinny K (128x64x16)")       { run_gemm_check(128, 64, 16); }
    SECTION("GEMM Large (512x512x512)")        { run_gemm_check(512, 512, 512); }
}
