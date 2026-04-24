#define CATCH_CONFIG_MAIN  // Weist Catch2 an, eine main-Funktion zu generieren
#include "../lib/catch.hpp"

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


TEST_CASE("", "[identity_16_16]") {
}
TEST_CASE("", "[zero_16_16]") {
}
TEST_CASE("", "[relu_16_16]") {
}
TEST_CASE("", "[gemm_32_32_1]") {
}
TEST_CASE("", "[gemm_32_32_512]") {
}
TEST_CASE("", "[gemm_512_32_512]") {
}
TEST_CASE("", "[gemm_512_512_512]") {
}