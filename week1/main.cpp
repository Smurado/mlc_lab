#define CATCH_CONFIG_MAIN  // Weist Catch2 an, eine main-Funktion zu generieren
#include "../lib/catch.hpp"
#include <cstdint>

// Deklaration der Assembly-Funktionen
extern "C" {
    int64_t inner_product(uint32_t const *i_a, uint32_t const *i_b, uint32_t i_size);
    void outer_product(uint32_t const *i_a, uint32_t const *i_b, uint32_t i_size, uint64_t *o_c);
}

TEST_CASE("Inner Product wird korrekt berechnet", "[inner_product]") {
    uint32_t a[] = {1, 2, 3, 4};
    uint32_t b[] = {5, 6, 7, 8};
    uint32_t size = 4;

    // Erwartetes Ergebnis: 1*5 + 2*6 + 3*7 + 4*8 = 70
    REQUIRE(inner_product(a, b, size) == 70);
}

TEST_CASE("Outer Product wird korrekt berechnet", "[outer_product]") {
    uint32_t a[] = {1, 2};
    uint32_t b[] = {3, 4};
    uint32_t size = 2;
    uint64_t c[4] = {0}; // Ergebnis-Array (size x size = 4 Elemente)

    // Aufrufen der Assembly-Funktion
    outer_product(a, b, size, c);

    // Erwartete Ergebnisse fuer a={1, 2} und b={3, 4}:
    // 1*3 = 3, 1*4 = 4
    // 2*3 = 6, 2*4 = 8
    REQUIRE(c[0] == 3);
    REQUIRE(c[1] == 4);
    REQUIRE(c[2] == 6);
    REQUIRE(c[3] == 8);
}