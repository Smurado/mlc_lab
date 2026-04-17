#define CATCH_CONFIG_MAIN  // Weist Catch2 an, eine main-Funktion zu generieren
#include "catch.hpp"
#include <cstdint>
#include <vector>

// Deklaration der Assembly-Funktionen
extern "C" {
    void perm_neon_abc_cba( int64_t size_c, 
                            float const * abc, 
                            float       * cba);
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