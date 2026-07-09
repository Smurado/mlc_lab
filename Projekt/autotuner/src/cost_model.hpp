#pragma once

#include "teir.hpp"
#include "autotuner.hpp"

double estimateCost(const TEIR& ir, const TuningConfig& config);

// C3: Kalibrierung. Statt peak_gflops und Thread-Spawn-Overhead hart zu kodieren,
// werden sie einmalig auf der Zielmaschine gemessen (Mini-GEMM fuer die Peak-Rate,
// leerer OpenMP-Parallelblock fuer den Overhead). Bis calibrateCostModel() laeuft,
// gelten konservative Defaults (10 GFLOPS / 0.5 ms). Abschaltbar per TEIR_CALIBRATE=0
// (dann bleiben die Defaults — fuer reproduzierbare Ablationen).
void calibrateCostModel();

// Aktuell aktive Kalibrierungswerte (fuer Reporting/Tests).
double costModelPeakGflops();
double costModelParallelOverheadMs();

struct CostBreakdown {
    double total_flops;
    double mem_penalty;
    double parallel_factor;
    double parallel_overhead_ms;
    double split_factor_cost;
    double unroll_factor;
    double estimated_ms;
};

CostBreakdown estimateCostDetailed(const TEIR& ir, const TuningConfig& config);
