// compilieren mit: /opt/homebrew/opt/llvm/bin/clang++ main.cpp functions.s -o main -march=armv9.2-a+sme

#define CATCH_CONFIG_MAIN  // Weist Catch2 an, eine main-Funktion zu generieren
#include "lib/catch.hpp"

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

/*
TEST_CASE("identity_16_16: Kopiert Matrix korrekt", "[identity_16_16]") {
    auto a = create_matrix(16, 16);
    std::vector<float> b(16 * 16, 0.0f);

    // SECTION("Column-major zu Column-major (trans_b = 0)") {
    //     identity_16_16(a.data(), b.data(), 16, 16, 0);
    //     REQUIRE(b == a);
    // }

    SECTION("Column-major zu Row-major (trans_b = 1)") {
        identity_16_16(a.data(), b.data(), 16, 16, 1);
        for(int i=0; i<16; ++i)
            for(int j=0; j<16; ++j)
                CHECK(b[i * 16 + j] == a[j * 16 + i]);
    }
}
*/

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
/*
TEST_CASE("gemm_32_32_1: Matrix-Multiplikation (Rank-1 Update)", "[gemm_32_32_1]") {
    // 32x1 * 1x32 = 32x32
    auto a = create_matrix(32, 1);
    auto b = create_matrix(1, 32);
    std::vector<float> c(32 * 32, 0.0f);

    gemm_32_32_1(a.data(), b.data(), c.data(), 32, 1, 32);

    // Stichprobenprüfung: C[i,j] = A[i,0] * B[0,j]
    CHECK(c[0] == a[0] * b[0]);
    CHECK(c[32 * 31 + 31] == a[31] * b[31]);
}

TEST_CASE("Leading Dimension Handling", "[ld_checks]") {
    // Testet, ob ld_a > rows korrekt übersprungen wird
    std::vector<float> a(16 * 20, 1.0f); // 16 Zeilen, 20 LD
    std::vector<float> b(16 * 16, 0.0f);
    
    // Setze markante Werte in die ersten 16x16
    for(int j=0; j<16; ++j) a[j * 20] = 99.0f; 

    identity_16_16(a.data(), b.data(), 20, 16, 0);
    
    CHECK(b[0] == 99.0f);
    CHECK(b[16] == 99.0f);
}
*/