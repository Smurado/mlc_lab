// Tests fuer die IR-Transformationen (passes.cpp).
//
// Diese drei Passes bauen aus der Start-IR die Konfiguration, die der Codegen
// dann uebersetzt -- ein Fehler hier erzeugt einen falschen Kernel, ohne dass
// irgendwo eine Fehlermeldung auftaucht. Besonders splitOuterAxis: es ersetzt
// eine Achse durch zwei (a -> a0/a1), und genau dieses Namensschema muss
// extentOfChar in einsum.cpp wieder zusammenrechnen (siehe test_einsum.cpp).
#include "passes.hpp"

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

static TEIR makeIr()
{
    TEIR ir;
    ir.name = "t";
    ir.axes = {{"a", 32}, {"b", 16}, {"c", 8}};
    ir.schedule = {{"a", Policy::Sequential},
                   {"b", Policy::Sequential},
                   {"c", Policy::Sequential}};
    return ir;
}

static std::string axisNames(const TEIR& ir)
{
    std::string s;
    for (const auto& a : ir.axes) s += a.name + " ";
    return s;
}

static std::string schedNames(const TEIR& ir)
{
    std::string s;
    for (const auto& i : ir.schedule) s += i.axis + " ";
    return s;
}

static int extentOf(const TEIR& ir, const std::string& name)
{
    for (const auto& a : ir.axes)
        if (a.name == name) return a.extent;
    return -1;
}

int main()
{
    std::printf("\n=== IR-Transformationen (passes.cpp) ===\n");

    // ---------------------------------------------------------------
    std::printf("\n-- splitOuterAxis --\n");
    // ---------------------------------------------------------------
    {
        TEIR ir = makeIr();
        splitOuterAxis(ir, "a", 4);   // a:32 -> a0:8, a1:4

        check(axisNames(ir) == "a0 a1 b c ",
              "a wird durch a0/a1 an derselben Position ersetzt", axisNames(ir));
        check(extentOf(ir, "a0") == 8, "aeusseres Extent = original / Faktor",
              std::to_string(extentOf(ir, "a0")));
        check(extentOf(ir, "a1") == 4, "inneres Extent = Faktor",
              std::to_string(extentOf(ir, "a1")));
        check(extentOf(ir, "a0") * extentOf(ir, "a1") == 32,
              "Produkt bleibt das Original-Extent");
        check(extentOf(ir, "a") == -1, "die Original-Achse ist weg");

        check(schedNames(ir) == "a0 a1 b c ",
              "Schedule bekommt beide Schleifen an derselben Stelle",
              schedNames(ir));
    }
    {
        // Policy-Vererbung: die AEUSSERE Schleife erbt die Policy, die innere
        // wird sequentiell -- sonst wuerde ein Split versehentlich eine zweite
        // parallele Ebene erzeugen.
        TEIR ir = makeIr();
        makeParallel(ir, "a");
        splitOuterAxis(ir, "a", 4);
        check(ir.schedule[0].axis == "a0" && ir.schedule[0].policy == Policy::Parallel,
              "aeussere Schleife erbt die Policy (parallel)");
        check(ir.schedule[1].axis == "a1" && ir.schedule[1].policy == Policy::Sequential,
              "innere Schleife wird sequentiell");
    }
    {
        // Unbekannte Achse: no-op statt Absturz.
        TEIR ir = makeIr();
        const std::string before = axisNames(ir);
        splitOuterAxis(ir, "zzz", 4);
        check(axisNames(ir) == before, "unbekannte Achse laesst die IR unveraendert");
    }
    {
        // Nicht teilender Faktor: warnt, teilt aber trotzdem (Ganzzahl-Division).
        // Festgehalten, damit die Konsequenz sichtbar ist: 8/3 = 2, 2*3 = 6 != 8.
        TEIR ir = makeIr();
        splitOuterAxis(ir, "c", 3);   // c:8 -> c0:2, c1:3
        check(extentOf(ir, "c0") == 2 && extentOf(ir, "c1") == 3,
              "nicht teilender Faktor: Ganzzahl-Division, Produkt < Original",
              std::to_string(extentOf(ir, "c0")) + "*" + std::to_string(extentOf(ir, "c1")));
    }

    // ---------------------------------------------------------------
    std::printf("\n-- reorderSchedule --\n");
    // ---------------------------------------------------------------
    {
        TEIR ir = makeIr();
        reorderSchedule(ir, {"c", "a", "b"});
        check(schedNames(ir) == "c a b ", "Schleifen in der gewuenschten Reihenfolge",
              schedNames(ir));
        check(ir.axes[0].name == "a", "Achsen-Definitionen bleiben unberuehrt");
    }
    {
        // Unvollstaendige Angabe: der Rest wird hinten angehaengt, nicht verworfen.
        // Sonst wuerde eine Schleife lautlos verschwinden.
        TEIR ir = makeIr();
        reorderSchedule(ir, {"c"});
        check(schedNames(ir) == "c a b ",
              "nicht genannte Schleifen bleiben erhalten (hinten)", schedNames(ir));
        check(ir.schedule.size() == 3, "keine Schleife geht verloren",
              std::to_string(ir.schedule.size()));
    }
    {
        // Unbekannte Namen werden ignoriert, nicht eingefuegt.
        TEIR ir = makeIr();
        reorderSchedule(ir, {"zzz", "b"});
        check(schedNames(ir) == "b a c ", "unbekannter Name wird uebersprungen",
              schedNames(ir));
        check(ir.schedule.size() == 3, "kein Phantom-Eintrag entsteht",
              std::to_string(ir.schedule.size()));
    }
    {
        // Policies ueberleben das Umsortieren.
        TEIR ir = makeIr();
        makeParallel(ir, "b");
        reorderSchedule(ir, {"c", "b", "a"});
        check(ir.schedule[1].axis == "b" && ir.schedule[1].policy == Policy::Parallel,
              "Policy wandert mit der Schleife mit");
    }

    // ---------------------------------------------------------------
    std::printf("\n-- makeParallel --\n");
    // ---------------------------------------------------------------
    {
        TEIR ir = makeIr();
        makeParallel(ir, "b");
        check(ir.schedule[1].policy == Policy::Parallel, "b wird parallel");
        check(ir.schedule[0].policy == Policy::Sequential &&
              ir.schedule[2].policy == Policy::Sequential,
              "die uebrigen Schleifen bleiben sequentiell");
    }
    {
        TEIR ir = makeIr();
        makeParallel(ir, "zzz");
        int par = 0;
        for (const auto& i : ir.schedule)
            if (i.policy == Policy::Parallel) ++par;
        check(par == 0, "unbekannte Achse parallelisiert nichts");
    }
    {
        // Zweimal aufrufen darf nicht zwei parallele Ebenen erzeugen.
        TEIR ir = makeIr();
        makeParallel(ir, "a");
        makeParallel(ir, "a");
        int par = 0;
        for (const auto& i : ir.schedule)
            if (i.policy == Policy::Parallel) ++par;
        check(par == 1, "idempotent: genau eine parallele Schleife",
              std::to_string(par));
    }

    std::printf("\n=== %d Pruefungen, %d Fehler ===\n\n", g_run, g_fail);
    return g_fail == 0 ? 0 : 1;
}
