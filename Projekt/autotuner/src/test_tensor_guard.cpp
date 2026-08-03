// Tests fuer den OOM-Guard des Benchmark-Harness (tensor_guard.hpp).
//
// Hintergrund: die Groessengrenze stand doppelt im Code (benchmark.cpp fuer die
// Such-Messung, main.cpp fuer die Endmessung). Beim Einbau von TEIR_MAX_TENSOR
// wurde nur eine Kopie angepasst -> drei GETT-Faelle durchliefen die Suche, aber
// die Endmessung brach ab ("partial" ohne [PERFORMANCE]). Diese Tests sichern das
// Verhalten der jetzt einzigen Implementierung ab.
#include "tensor_guard.hpp"

#include <cstdio>
#include <string>

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

int main()
{
    std::printf("\n=== OOM-Guard (tensor_guard.hpp) ===\n");

    // ---------------------------------------------------------------
    std::printf("\n-- Default-Grenze --\n");
    // ---------------------------------------------------------------
    check(kDefaultMaxTensorElements == 100000000,
          "Default sind 100 Mio. Elemente (~400 MB float)");
    check(parseMaxTensorElements(nullptr) == kDefaultMaxTensorElements,
          "ohne Env-Variable gilt der Default");

    // ---------------------------------------------------------------
    std::printf("\n-- Env-Auswertung --\n");
    // ---------------------------------------------------------------
    check(parseMaxTensorElements("200000000") == 200000000,
          "Env-Wert wird uebernommen");
    check(parseMaxTensorElements("1") == 1,
          "kleinster gueltiger Wert");
    // Konvention der uebrigen TEIR_*-Schalter: std::max(1, atoi(...)).
    check(parseMaxTensorElements("0") == 1,
          "0 wird auf 1 geklemmt (keine Grenze <= 0)");
    check(parseMaxTensorElements("-5") == 1,
          "negativer Wert wird auf 1 geklemmt");
    check(parseMaxTensorElements("abc") == 1,
          "nicht parsbarer Wert -> 1 (atoi-Konvention, alles gilt als zu gross)");

    // ---------------------------------------------------------------
    std::printf("\n-- Groessenpruefung --\n");
    // ---------------------------------------------------------------
    const int L = 100000000;
    check(!tensorTooLarge(1, 1, 1, L), "winzige Tensoren passen");
    check(!tensorTooLarge(L, L, L, L), "exakt auf der Grenze ist erlaubt");
    check(tensorTooLarge(L + 1, 1, 1, L), "in0 ueber der Grenze schlaegt an");
    check(tensorTooLarge(1, L + 1, 1, L), "in1 ueber der Grenze schlaegt an");
    check(tensorTooLarge(1, 1, L + 1, L), "out ueber der Grenze schlaegt an");

    // ---------------------------------------------------------------
    std::printf("\n-- Die drei real betroffenen GETT-Faelle --\n");
    // ---------------------------------------------------------------
    // in0-Groessen aus data/input_gett.csv (a:48;c:24;b:36;e:48;d:36;f:36 usw.)
    struct Case { const char* name; int in0; int in1; int out; };
    const Case cases[] = {
        {"abcde-efbad-cf", 107495424,  864, 71663616},
        {"abcde-ecbfa-fd", 143327232, 1152, 71663616},
        {"abcde-efcad-bf", 107495424,  864, 71663616},
    };
    for (const auto& c : cases) {
        check(tensorTooLarge(c.in0, c.in1, c.out, kDefaultMaxTensorElements),
              std::string(c.name) + ": mit Default-Grenze abgelehnt (war der Bug)",
              "in0=" + std::to_string(c.in0));
        check(!tensorTooLarge(c.in0, c.in1, c.out, 200000000),
              std::string(c.name) + ": mit TEIR_MAX_TENSOR=200000000 erlaubt");
    }

    std::printf("\n=== %d Pruefungen, %d Fehler ===\n\n", g_run, g_fail);
    return g_fail == 0 ? 0 : 1;
}
