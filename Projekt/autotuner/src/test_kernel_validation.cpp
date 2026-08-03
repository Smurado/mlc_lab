// Tests fuer die gemeinsame Kernel-Validierung (kernel_validation.hpp).
//
// Hintergrund: Fuellmuster, Referenz-/Stichproben-Entscheidung und Toleranz
// standen wortgleich doppelt im Code (benchmark.cpp fuer jeden Such-Trial,
// main.cpp fuer den Gewinner vor der Endmessung). Beide MUESSEN dasselbe
// pruefen, sonst akzeptiert die Suche Kernel, die die Endmessung ablehnt.
// Diese Tests sichern das Verhalten der jetzt einzigen Implementierung ab.
#include "kernel_validation.hpp"

#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_run = 0;

static void check(bool ok, const std::string& name, const std::string& detail = "")
{
    ++g_run;
    if (ok) {
        std::printf("  [ok]   %s\n", name.c_str());
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s   %s\n", name.c_str(), detail.c_str());
    }
}

// Minimale IR fuer einen Einsum-Fall.
static TEIR makeIr(const std::string& einsum, const std::vector<Axis>& axes)
{
    TEIR ir;
    ir.name = "test";
    ir.einsum = einsum;
    ir.axes = axes;
    return ir;
}

int main()
{
    std::printf("\n=== Kernel-Validierung (kernel_validation.hpp) ===\n");

    // ---------------------------------------------------------------
    std::printf("\n-- Fuellmuster --\n");
    // ---------------------------------------------------------------
    {
        std::vector<float> a(30, 0.0f), b(30, 0.0f);
        fillEinsumInputs(a, b);
        // in0: (i%13+1)/13 -> 1/13, 2/13, ... 13/13, dann wieder 1/13
        check(std::abs(a[0] - 1.0f / 13.0f) < 1e-6f, "in0[0] = 1/13");
        check(std::abs(a[12] - 13.0f / 13.0f) < 1e-6f, "in0[12] = 13/13");
        check(std::abs(a[13] - 1.0f / 13.0f) < 1e-6f, "in0 wiederholt sich nach 13");
        check(std::abs(b[0] - 1.0f / 7.0f) < 1e-6f, "in1[0] = 1/7");
        check(std::abs(b[6] - 7.0f / 7.0f) < 1e-6f, "in1[6] = 7/7");
        check(std::abs(b[7] - 1.0f / 7.0f) < 1e-6f, "in1 wiederholt sich nach 7");
        // Kein Element darf 0 sein: eine 0 wuerde Layout-Fehler verdecken.
        bool anyZero = false;
        for (float v : a) if (v == 0.0f) anyZero = true;
        for (float v : b) if (v == 0.0f) anyZero = true;
        check(!anyZero, "kein Fuellwert ist 0 (wuerde Layout-Fehler verdecken)");
        // Perioden teilerfremd -> Produktmuster wiederholt sich erst nach 91.
        check(13 % 7 != 0 && 7 % 13 != 0, "Perioden 13 und 7 sind teilerfremd");
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Toleranz --\n");
    // ---------------------------------------------------------------
    check(withinTolerance(1.0f, 1.0f), "identische Werte sind gueltig");
    check(withinTolerance(1.005f, 1.0f), "Abweichung unter 1 % ist gueltig");
    check(!withinTolerance(1.02f, 1.0f), "Abweichung ueber 1 % ist ungueltig");
    // Absolute Untergrenze: bei winzigen Referenzwerten gilt 1e-2 absolut,
    // sonst wuerde eine rein relative Schranke bei ref~0 immer scheitern.
    check(withinTolerance(0.005f, 0.0f), "kleiner Absolutfehler bei ref=0 ist gueltig");
    check(!withinTolerance(0.02f, 0.0f), "grosser Absolutfehler bei ref=0 ist ungueltig");
    // Bei grossen Werten skaliert die Toleranz mit.
    check(withinTolerance(1000.0f, 1005.0f), "relative Toleranz skaliert mit ref");
    check(!withinTolerance(1000.0f, 1020.0f), "2 % bei grossem ref ist ungueltig");

    // ---------------------------------------------------------------
    std::printf("\n-- Referenz vs. Stichprobe --\n");
    // ---------------------------------------------------------------
    check(kFullReferenceMaxIters == 100000000.0,
          "Schwelle liegt bei 1e8 Iterationen");
    {
        // ab-ac-cb @ 32: 32^3 = 32768 Iterationen -> volle Referenz
        TEIR small = makeIr("ab-ac-cb", {{"a", 32}, {"b", 32}, {"c", 32}});
        check(useFullReference(small), "kleiner Fall nutzt die volle Referenz",
              "iters=" + std::to_string(einsumFlops(small) / 2.0));
        // ab-ac-cb @ 1024: 1024^3 = 1.07e9 -> Stichprobe
        TEIR big = makeIr("ab-ac-cb", {{"a", 1024}, {"b", 1024}, {"c", 1024}});
        check(!useFullReference(big), "grosser Fall nutzt die Stichprobe",
              "iters=" + std::to_string(einsumFlops(big) / 2.0));
        // Genau auf der Schwelle: 464^3 = 99.9e6 < 1e8 -> noch volle Referenz
        TEIR edge = makeIr("ab-ac-cb", {{"a", 464}, {"b", 464}, {"c", 464}});
        check(useFullReference(edge), "knapp unter der Schwelle: volle Referenz",
              "iters=" + std::to_string(einsumFlops(edge) / 2.0));
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Ende-zu-Ende gegen die Referenz --\n");
    // ---------------------------------------------------------------
    {
        // Kleiner GEMM-foermiger Fall, voll nachrechenbar.
        TEIR ir = makeIr("ab-ac-cb", {{"a", 4}, {"b", 5}, {"c", 6}});
        const int aN = 4, bN = 5, cN = 6;
        std::vector<float> in0(aN * cN), in1(cN * bN), out(aN * bN, 0.0f);
        fillEinsumInputs(in0, in1);

        // Korrekte Ausgabe = genau die Referenz.
        referenceEinsum(ir, in0.data(), in1.data(), out.data());
        check(validateEinsumOutput(ir, in0.data(), in1.data(), out.data(), aN * bN),
              "korrekte Ausgabe besteht die Pruefung");

        // Ein einzelnes verfaelschtes Element muss auffallen.
        std::vector<float> wrong = out;
        wrong[0] += 1.0f;
        check(!validateEinsumOutput(ir, in0.data(), in1.data(), wrong.data(), aN * bN),
              "ein verfaelschtes Element wird erkannt");

        // Transponierte Ausgabe (klassischer Layout-Bug) muss auffallen.
        std::vector<float> transposed(aN * bN, 0.0f);
        for (int i = 0; i < aN; ++i)
            for (int j = 0; j < bN; ++j)
                transposed[j * aN + i] = out[i * bN + j];
        check(!validateEinsumOutput(ir, in0.data(), in1.data(), transposed.data(), aN * bN),
              "transponierte Ausgabe (Layout-Bug) wird erkannt");

        // Alles-Null muss auffallen -- der Fall, den ein Konstanten-Check verpasst.
        std::vector<float> zeros(aN * bN, 0.0f);
        check(!validateEinsumOutput(ir, in0.data(), in1.data(), zeros.data(), aN * bN),
              "Null-Ausgabe wird erkannt");
    }

    std::printf("\n=== %d Pruefungen, %d Fehler ===\n\n", g_run, g_fail);
    return g_fail == 0 ? 0 : 1;
}
