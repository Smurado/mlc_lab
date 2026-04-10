#include <iostream>
#include <cstdint>

// Deklaration der Assembly-Funktion
extern "C" int64_t inner_product(uint32_t const *i_a, uint32_t const *i_b, uint32_t i_size);

int main() {
    // Test-Arrays
    uint32_t a[] = {1, 2, 3, 4};
    uint32_t b[] = {5, 6, 7, 8};
    uint32_t size = 4;

    // Aufruf der Funktion
    int64_t result = inner_product(a, b, size);
    std::cout << "Inner Product Result: " << result << std::endl;

    // Erwartetes Ergebnis: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    return 0;
}