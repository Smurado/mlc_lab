// Tests fuer das analytische Cost-Modell (cost_model.cpp).
//
// Das Modell entscheidet als Vorfilter, welche Kandidaten ueberhaupt gemessen
// werden, und liefert den Warmstart-Punkt der Suche. Ein Fehler hier faellt
// nirgends als Absturz auf, sondern nur als schlechteres Suchergebnis -- genau
// deshalb brauchen die reinen Schaetz-Anteile eigene Tests. Die Kalibrierung
// selbst (Messung auf der Maschine) bleibt aussen vor; getestet wird mit den
// dokumentierten Defaults (TEIR_CALIBRATE=0).
#include "cost_model.hpp"
#include "einsum.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// GEMM-foermige Kontraktion out[a,b] = sum_c in0[a,c] * in1[c,b].
static TEIR makeGemmIr(int extent)
{
    TEIR ir;
    ir.name = "t";
    ir.einsum = "ab-ac-cb";
    ir.axes = {{"a", extent}, {"b", extent}, {"c", extent}};
    ir.schedule = {{"a", Policy::Sequential},
                   {"b", Policy::Sequential},
                   {"c", Policy::Sequential}};
    return ir;
}

static TuningConfig makeConfig()
{
    TuningConfig cfg;
    cfg.loop_order = {"a", "b", "c"};
    return cfg;
}

int main()
{
    std::printf("\n=== Cost-Modell (cost_model.cpp) ===\n");

    // ---------------------------------------------------------------
    std::printf("\n-- Kalibrierungs-Defaults --\n");
    // ---------------------------------------------------------------
    check(costModelPeakGflops() == 10.0,
          "Peak-Default 10 GFLOPS vor jeder Kalibrierung");
    check(costModelParallelOverheadMs() == 0.5,
          "Overhead-Default 0.5 ms vor jeder Kalibrierung");

    // TEIR_CALIBRATE=0 ist der dokumentierte Weg zu reproduzierbaren
    // Ablationen: die Defaults muessen die Kalibrierung ueberleben.
    setenv("TEIR_CALIBRATE", "0", 1);
    calibrateCostModel();
    check(costModelPeakGflops() == 10.0 && costModelParallelOverheadMs() == 0.5,
          "TEIR_CALIBRATE=0 laesst die Defaults unveraendert");

    // ---------------------------------------------------------------
    std::printf("\n-- Grundverhalten --\n");
    // ---------------------------------------------------------------
    TEIR ir = makeGemmIr(64);
    TuningConfig cfg = makeConfig();

    const double cost = estimateCost(ir, cfg);
    check(std::isfinite(cost) && cost > 0.0,
          "plausible Konfiguration ergibt endliche, positive Kosten");

    TEIR nonEinsum;
    nonEinsum.name = "legacy";
    check(estimateCost(nonEinsum, cfg) == 1.0,
          "IR ohne Einsum faellt auf den neutralen Wert 1.0 zurueck");

    TuningConfig empty;
    check(estimateCost(ir, empty) >= 1e17,
          "leere Schleifenreihenfolge wird prohibitiv teuer bewertet");

    // ---------------------------------------------------------------
    std::printf("\n-- Skalierung --\n");
    // ---------------------------------------------------------------
    check(estimateCost(makeGemmIr(128), cfg) > cost,
          "achtfaches FLOP-Volumen kostet mehr als das kleine Problem");

    // ---------------------------------------------------------------
    std::printf("\n-- Parallel-Overhead --\n");
    // ---------------------------------------------------------------
    // 4^3 = 128 FLOPs: weit unter der 1e6-Schwelle pro Thread. Das Modell
    // muss Parallelisierung hier bestrafen (Spawn-Kosten > Rechenzeit).
    TEIR tiny = makeGemmIr(4);
    TuningConfig par = makeConfig();
    par.parallel_axis = "a";
    check(estimateCost(tiny, par) > estimateCost(tiny, cfg),
          "Mikro-Workload: parallel ist teurer als sequentiell");

    // ---------------------------------------------------------------
    std::printf("\n-- Breakdown-Konsistenz --\n");
    // ---------------------------------------------------------------
    const CostBreakdown cb = estimateCostDetailed(ir, cfg);
    check(cb.estimated_ms == cost,
          "estimateCost entspricht estimated_ms des Breakdowns");
    check(cb.total_flops == einsumFlops(ir),
          "total_flops stimmt mit einsumFlops ueberein");

    std::printf("\n=== %d Pruefungen, %d Fehler ===\n", g_run, g_fail);
    return g_fail ? 1 : 0;
}
