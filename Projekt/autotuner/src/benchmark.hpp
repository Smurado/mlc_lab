#pragma once
#include "teir.hpp"
#include <string>

struct BenchmarkResult {
    double runtime_ms;
    double gflops;
};

// Interpretiert den aktuellen Schedule und misst die Zeit
BenchmarkResult benchmark(const TEIR& ir);

// A1 Kernel-Cache: Statistik über Treffer/Fehltreffer. Semantisch identische
// Schedules erzeugen identischen Kernel-Code und werden nur einmal JIT-kompiliert.
long benchmarkCacheHits();
long benchmarkCacheMisses();
// A5: Treffer im persistenten Cross-Run-Disk-Cache (Compile+ggf. Bench gespart).
long benchmarkPersistHits();
void resetBenchmarkCacheStats();

// Speichert das Ergebnis sauber in eine CSV-Datei (mit Strategie-Spalte).
// C3: costEstimateMs = CostModel-Schaetzung (ms) fuer diesen Trial, damit sich
// offline die Rang-Korrelation "geschaetzt vs. gemessen" berechnen laesst.
void saveToCSV(const std::string& filename, const std::string& configName,
               const BenchmarkResult& res, const std::string& strategy,
               double costEstimateMs);

// D1: Haengt eine Zeile an das Konvergenz-Log (fuer Konvergenzkurven). Erfasst
// pro Trial den Index, das GFLOPS dieses Trials (0 = inkorrekt) und das beste
// GFLOPS bisher — gruppiert nach Strategie und Seed.
void appendTrialLog(const std::string& filename, const std::string& strategy,
                    unsigned seed, int trialIndex, double trialGflops,
                    double bestGflops);