#define CATCH_CONFIG_MAIN  // Weist Catch2 an, eine main-Funktion zu generieren
#include "../lib/catch.hpp"
#include <cstdint>
#include <random>
#include <vector>

// Deklaration der Assembly-Funktionen
extern "C" {
    int64_t inner_product(uint32_t const *i_a, uint32_t const *i_b, uint32_t i_size);
    void outer_product(uint32_t const *i_a, uint32_t const *i_b, uint32_t i_size, uint64_t *o_c);
}

// Referenzimplementierung aus data/base_math.cpp.
int64_t inner_product_cpp(uint32_t const *i_a, uint32_t const *i_b, uint32_t i_size);
void outer_product_cpp(uint32_t const *i_a, uint32_t const *i_b, uint32_t i_size, uint64_t *o_c);

// ---------------------------------------------------------------------------
// Feste Erwartungswerte
//
// Diese beiden Faelle sind von Hand nachgerechnet und dokumentieren, WAS die
// Funktionen tun sollen. Sie pruefen aber nur je eine Eingabe -- der eigentliche
// Nachweis kommt aus dem Vergleich gegen die Referenz weiter unten.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Vergleich gegen die Referenzimplementierung (data/base_math.cpp)
//
// Der Vorteil gegenueber festen Erwartungswerten: das Ergebnis muss nicht vorab
// bekannt sein. Damit lassen sich beliebig viele Eingaben pruefen -- vor allem
// zufaellige, auf die man beim Nachrechnen von Hand nie kaeme.
// ---------------------------------------------------------------------------

namespace {

// Fester Seed: schlaegt ein Test fehl, ist er exakt reproduzierbar.
std::mt19937 makeRng() { return std::mt19937(20250803u); }

// Zufallswerte auf 16 Bit begrenzt, damit die Produkte im aeusseren Produkt
// (32x32 -> 64 Bit) mit Sicherheit darstellbar bleiben.
std::vector<uint32_t> randomVector(std::mt19937 &rng, uint32_t size) {
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFu);
    std::vector<uint32_t> v(size);
    for (auto &x : v) x = dist(rng);
    return v;
}

} // namespace

TEST_CASE("Inner Product stimmt mit der Referenz ueberein", "[inner_product][reference]") {
    auto rng = makeRng();

    // Randfaelle bewusst dabei: 0 (Schleife laeuft nie), 1 (nur ein Durchlauf),
    // ungerade Laengen sowie groessere Vektoren.
    const std::vector<uint32_t> sizes = {0, 1, 2, 3, 7, 16, 63, 64, 255, 1024};

    for (uint32_t size : sizes) {
        auto a = randomVector(rng, size);
        auto b = randomVector(rng, size);

        const int64_t got      = inner_product(a.data(), b.data(), size);
        const int64_t expected = inner_product_cpp(a.data(), b.data(), size);

        INFO("size = " << size);
        REQUIRE(got == expected);
    }
}

TEST_CASE("Outer Product stimmt mit der Referenz ueberein", "[outer_product][reference]") {
    auto rng = makeRng();

    const std::vector<uint32_t> sizes = {0, 1, 2, 3, 7, 16, 33, 64};

    for (uint32_t size : sizes) {
        auto a = randomVector(rng, size);
        auto b = randomVector(rng, size);

        // Fuellwert ungleich null: so faellt auf, wenn der Kernel ein Element gar
        // nicht beschreibt, statt es nur falsch zu beschreiben.
        std::vector<uint64_t> got(static_cast<size_t>(size) * size, 0xDEADBEEFull);
        std::vector<uint64_t> expected(static_cast<size_t>(size) * size, 0);

        outer_product(a.data(), b.data(), size, got.data());
        outer_product_cpp(a.data(), b.data(), size, expected.data());

        INFO("size = " << size);
        REQUIRE(got == expected);
    }
}

TEST_CASE("Inner Product: viele zufaellige Durchlaeufe", "[inner_product][reference]") {
    auto rng = makeRng();
    std::uniform_int_distribution<uint32_t> sizeDist(1, 200);

    for (int run = 0; run < 200; ++run) {
        const uint32_t size = sizeDist(rng);
        auto a = randomVector(rng, size);
        auto b = randomVector(rng, size);

        INFO("run = " << run << ", size = " << size);
        REQUIRE(inner_product(a.data(), b.data(), size)
                == inner_product_cpp(a.data(), b.data(), size));
    }
}

// ---------------------------------------------------------------------------
// Grenzwerte
// ---------------------------------------------------------------------------

TEST_CASE("Inner Product: groesste 32-Bit-Werte", "[inner_product][limits]") {
    // Die Assembly nutzt `umull` (unsigned 32x32 -> 64 Bit) und summiert in 64 Bit.
    // Hier wird geprueft, dass die Erweiterung VOR der Multiplikation passiert --
    // andernfalls wuerde das Produkt schon in 32 Bit abgeschnitten.
    const uint32_t maxv = 0xFFFFFFFFu;
    std::vector<uint32_t> a(4, maxv);
    std::vector<uint32_t> b(4, maxv);

    REQUIRE(inner_product(a.data(), b.data(), 4)
            == inner_product_cpp(a.data(), b.data(), 4));
}

TEST_CASE("Outer Product: groesste 32-Bit-Werte", "[outer_product][limits]") {
    const uint32_t maxv = 0xFFFFFFFFu;
    std::vector<uint32_t> a = {maxv, 1};
    std::vector<uint32_t> b = {maxv, 2};

    std::vector<uint64_t> got(4, 0);
    std::vector<uint64_t> expected(4, 0);

    outer_product(a.data(), b.data(), 2, got.data());
    outer_product_cpp(a.data(), b.data(), 2, expected.data());

    REQUIRE(got == expected);
    // Explizit: das groesste Produkt passt gerade noch in 64 Bit.
    REQUIRE(got[0] == static_cast<uint64_t>(maxv) * static_cast<uint64_t>(maxv));
}

TEST_CASE("Groesse 0: Rueckgabe 0, Puffer unangetastet", "[limits]") {
    uint32_t a[] = {1, 2, 3};
    uint32_t b[] = {4, 5, 6};

    REQUIRE(inner_product(a, b, 0) == 0);

    std::vector<uint64_t> c(4, 0xABCDull);
    outer_product(a, b, 0, c.data());
    for (uint64_t v : c) REQUIRE(v == 0xABCDull);
}
