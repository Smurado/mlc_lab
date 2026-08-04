// Unit-Tests fuer die JIT-erzeugten Kernel (Woche 5).
//
// Warum das hier besonders wichtig ist: Ein falsches Instruktionswort erzeugt
// KEINEN Compilerfehler. Es entsteht stillschweigend falscher Maschinencode, der
// entweder abstuerzt oder -- schlimmer -- plausible falsche Zahlen liefert.
// Deshalb wird jede Kernel-Ausgabe elementweise gegen eine naive Referenz
// geprueft, nicht gegen von Hand ausgerechnete Konstanten.
//
// Aufruf: make test
#define CATCH_CONFIG_MAIN
#include "../lib/catch.hpp"

#include "Unary.h"
#include "Gemm.h"

#include <cstdint>
#include <random>
#include <vector>

using mini_jit::Unary;
using mini_jit::Gemm;

namespace {

// Fester Seed: schlaegt ein Test fehl, ist er exakt reproduzierbar.
std::mt19937 makeRng() { return std::mt19937(20250803u); }

// Werte mit Vorzeichenwechsel -- sonst wuerde ReLU nie einen Wert abschneiden
// und der Test liefe versehentlich gegen Identity.
std::vector<float> randomMatrix(std::mt19937 &rng, size_t count) {
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<float> v(count);
    for (auto &x : v) x = dist(rng);
    return v;
}

// --- Referenzimplementierungen (bewusst naiv und offensichtlich korrekt) ----

void referenceZero(std::vector<float> &b, uint32_t m, uint32_t n, int64_t ld_b) {
    for (uint32_t j = 0; j < n; ++j)
        for (uint32_t i = 0; i < m; ++i)
            b[static_cast<size_t>(j) * ld_b + i] = 0.0f;
}

void referenceIdentity(const std::vector<float> &a, std::vector<float> &b,
                       uint32_t m, uint32_t n, int64_t ld_a, int64_t ld_b) {
    for (uint32_t j = 0; j < n; ++j)
        for (uint32_t i = 0; i < m; ++i)
            b[static_cast<size_t>(j) * ld_b + i] = a[static_cast<size_t>(j) * ld_a + i];
}

void referenceRelu(const std::vector<float> &a, std::vector<float> &b,
                   uint32_t m, uint32_t n, int64_t ld_a, int64_t ld_b) {
    for (uint32_t j = 0; j < n; ++j)
        for (uint32_t i = 0; i < m; ++i) {
            const float v = a[static_cast<size_t>(j) * ld_a + i];
            b[static_cast<size_t>(j) * ld_b + i] = v > 0.0f ? v : 0.0f;
        }
}

// C += A * B.
//
// Layout laut Aufgabenstellung: A und C SPALTENWEISE, B ZEILENWEISE.
//   A(i,l) = a[l * ld_a + i]     (spaltenweise)
//   B(l,j) = b[l * ld_b + j]     (zeilenweise!)
//   C(i,j) = c[j * ld_c + i]     (spaltenweise)
//
// Die B-Indizierung ist der entscheidende Punkt: `verify_gemm` in main.cpp
// rechnete hier spaltenweise (b[j * ld_b + l]) und lag damit falsch. Aufgefallen
// ist das nie, weil dort B konstant mit 2.0 gefuellt wird -- bei konstanten
// Werten liefern beide Layouts dasselbe Ergebnis.
void referenceGemm(const std::vector<float> &a, const std::vector<float> &b,
                   std::vector<float> &c, uint32_t m, uint32_t n, uint32_t k,
                   int64_t ld_a, int64_t ld_b, int64_t ld_c) {
    for (uint32_t j = 0; j < n; ++j)
        for (uint32_t i = 0; i < m; ++i) {
            float acc = 0.0f;
            for (uint32_t l = 0; l < k; ++l)
                acc += a[static_cast<size_t>(l) * ld_a + i]
                     * b[static_cast<size_t>(l) * ld_b + j];
            c[static_cast<size_t>(j) * ld_c + i] += acc;
        }
}

// Aufruf des generierten Kernels.
//
// Frueher war dafuer ein Inline-Assembler-Wrapper noetig, weil der Kernel per
// `smstart` die callee-saved Register v8-v15 zerstoerte, ohne sie zu sichern.
// Der Generator sichert sie inzwischen selbst (Prolog/Epilog in Unary.cpp),
// deshalb genuegt ein gewoehnlicher Aufruf.
void callUnary(Unary::kernel_t kernel, const float *a, float *b,
               int64_t ld_a, int64_t ld_b) {
    kernel(a, b, ld_a, ld_b);
}

// Aufruf des GEMM-Kernels mit deklarierten Clobbern.
//
// Der GEMM-Kernel ist ein festes Instruktionsarray (Gemm.cpp) und sichert
// d8-d15 nicht, die `smstart` zerstoert. Anders als beim Unary-Generator laesst
// sich das dort nicht nachruesten, ohne die relativen Sprungziele im Array zu
// verschieben. Ohne diese Deklaration darf der Compiler Gleitkommawerte ueber
// den Aufruf hinweg in d8-d15 halten und liest danach Nullen.
void callGemm(Gemm::kernel_t kern, const float *a, const float *b, float *c,
              int64_t ld_a, int64_t ld_b, int64_t ld_c) {
    register const float *x0 asm("x0") = a;
    register const float *x1 asm("x1") = b;
    register float       *x2 asm("x2") = c;
    register int64_t      x3 asm("x3") = ld_a;
    register int64_t      x4 asm("x4") = ld_b;
    register int64_t      x5 asm("x5") = ld_c;
    Gemm::kernel_t fn = kern;

    asm volatile(
        "blr %[fn]\n"
        : "+r"(x0), "+r"(x1), "+r"(x2), "+r"(x3), "+r"(x4), "+r"(x5)
        : [fn] "r"(fn)
        : "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
          // x18 steht bewusst NICHT hier: auf Apple-Plattformen ist es das
          // Platform-Register und darf nicht ueberschrieben werden. Der
          // Kernel fasst es in keinem seiner 114 Instruktionswoerter an.
          "x16", "x17", "x30",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "memory", "cc"
    );
}

// Groessen laut Aufgabe: Vielfache von 16.
const std::vector<uint32_t> kSizes = {16, 32, 64, 128};

} // namespace

// ---------------------------------------------------------------------------
TEST_CASE("Unary zero: setzt alle Eintraege auf null", "[unary][zero]") {
    auto rng = makeRng();

    for (uint32_t size : kSizes) {
        Unary gen;
        REQUIRE(gen.generate(size, size, 0, Unary::dtype_t::fp32,
                             Unary::ptype_t::zero) == Unary::error_t::success);
        auto kernel = gen.get_kernel();
        REQUIRE(kernel != nullptr);

        auto a = randomMatrix(rng, static_cast<size_t>(size) * size);
        // Fuellwert ungleich null: so faellt auf, wenn ein Element gar nicht
        // beschrieben wird, statt nur falsch beschrieben zu werden.
        std::vector<float> got(static_cast<size_t>(size) * size, 7.5f);
        std::vector<float> expected = got;

        callUnary(kernel, a.data(), got.data(), size, size);
        referenceZero(expected, size, size, size);

        INFO("size = " << size);
        REQUIRE(got == expected);
    }
}

TEST_CASE("Unary identity: kopiert A nach B", "[unary][identity]") {
    auto rng = makeRng();

    for (uint32_t size : kSizes) {
        Unary gen;
        REQUIRE(gen.generate(size, size, 0, Unary::dtype_t::fp32,
                             Unary::ptype_t::identity) == Unary::error_t::success);
        auto kernel = gen.get_kernel();
        REQUIRE(kernel != nullptr);

        auto a = randomMatrix(rng, static_cast<size_t>(size) * size);
        std::vector<float> got(static_cast<size_t>(size) * size, 7.5f);
        std::vector<float> expected = got;

        callUnary(kernel, a.data(), got.data(), size, size);
        referenceIdentity(a, expected, size, size, size, size);

        INFO("size = " << size);
        REQUIRE(got == expected);
    }
}

TEST_CASE("Unary relu: schneidet negative Werte ab", "[unary][relu]") {
    auto rng = makeRng();

    for (uint32_t size : kSizes) {
        Unary gen;
        REQUIRE(gen.generate(size, size, 0, Unary::dtype_t::fp32,
                             Unary::ptype_t::relu) == Unary::error_t::success);
        auto kernel = gen.get_kernel();
        REQUIRE(kernel != nullptr);

        auto a = randomMatrix(rng, static_cast<size_t>(size) * size);
        std::vector<float> got(static_cast<size_t>(size) * size, 7.5f);
        std::vector<float> expected = got;

        callUnary(kernel, a.data(), got.data(), size, size);
        referenceRelu(a, expected, size, size, size, size);

        INFO("size = " << size);
        REQUIRE(got == expected);

        // Gegenprobe, dass der Testfall ueberhaupt etwas abschneidet -- sonst
        // waere er von Identity nicht zu unterscheiden.
        bool hasNegativeInput = false;
        for (float v : a) if (v < 0.0f) { hasNegativeInput = true; break; }
        REQUIRE(hasNegativeInput);
    }
}

TEST_CASE("Unary: unterschiedliche m und n", "[unary][identity]") {
    auto rng = makeRng();

    // Nicht nur quadratisch -- eine vertauschte Schleifengrenze faellt sonst
    // nicht auf.
    const std::vector<std::pair<uint32_t, uint32_t>> shapes = {
        {16, 32}, {32, 16}, {16, 64}, {64, 16}, {32, 64}
    };

    for (auto [m, n] : shapes) {
        Unary gen;
        REQUIRE(gen.generate(m, n, 0, Unary::dtype_t::fp32,
                             Unary::ptype_t::identity) == Unary::error_t::success);
        auto kernel = gen.get_kernel();
        REQUIRE(kernel != nullptr);

        auto a = randomMatrix(rng, static_cast<size_t>(m) * n);
        std::vector<float> got(static_cast<size_t>(m) * n, 7.5f);
        std::vector<float> expected = got;

        callUnary(kernel, a.data(), got.data(), m, m);
        referenceIdentity(a, expected, m, n, m, m);

        INFO("m = " << m << ", n = " << n);
        REQUIRE(got == expected);
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("Gemm: C += A*B stimmt mit der Referenz ueberein", "[gemm]") {
    auto rng = makeRng();

    const uint32_t m = 512, n = 512, k = 512;

    Gemm gen;
    REQUIRE(gen.generate(m, n, k, 0, 0, 0, Gemm::dtype_t::fp32)
            == Gemm::error_t::success);
    auto kernel = gen.get_kernel();
    REQUIRE(kernel != nullptr);

    auto a = randomMatrix(rng, static_cast<size_t>(m) * k);
    auto b = randomMatrix(rng, static_cast<size_t>(k) * n);

    // C ungleich null vorbelegen: der Kernel soll AKKUMULIEREN (C += A*B).
    // Startet C bei null, waere ein Kernel, der C ueberschreibt statt zu
    // akkumulieren, nicht von einem korrekten zu unterscheiden.
    auto c0 = randomMatrix(rng, static_cast<size_t>(m) * n);
    std::vector<float> got = c0;
    std::vector<float> expected = c0;

    callGemm(kernel, a.data(), b.data(), got.data(), m, k, m);
    referenceGemm(a, b, expected, m, n, k, m, k, m);

    // Toleranz: die Summationsreihenfolge unterscheidet sich zwischen Kernel
    // und Referenz, bei k=512 summieren sich Rundungsfehler auf.
    for (size_t i = 0; i < got.size(); ++i) {
        INFO("Index " << i);
        REQUIRE(got[i] == Approx(expected[i]).epsilon(1e-4));
    }
}

TEST_CASE("Gemm: akkumuliert, statt zu ueberschreiben", "[gemm][accumulate]") {
    auto rng = makeRng();
    const uint32_t m = 512, n = 512, k = 512;

    Gemm gen;
    REQUIRE(gen.generate(m, n, k, 0, 0, 0, Gemm::dtype_t::fp32)
            == Gemm::error_t::success);
    auto kernel = gen.get_kernel();
    REQUIRE(kernel != nullptr);

    auto a = randomMatrix(rng, static_cast<size_t>(m) * k);
    auto b = randomMatrix(rng, static_cast<size_t>(k) * n);

    // Zweimal auf dasselbe C anwenden muss denselben Zuwachs ergeben wie
    // einmal -- das prueft die Akkumulation direkt.
    std::vector<float> once(static_cast<size_t>(m) * n, 0.0f);
    callGemm(kernel, a.data(), b.data(), once.data(), m, k, m);

    std::vector<float> twice(static_cast<size_t>(m) * n, 0.0f);
    callGemm(kernel, a.data(), b.data(), twice.data(), m, k, m);
    callGemm(kernel, a.data(), b.data(), twice.data(), m, k, m);

    for (size_t i = 0; i < once.size(); ++i) {
        INFO("Index " << i);
        REQUIRE(twice[i] == Approx(2.0f * once[i]).epsilon(1e-4));
    }
}
