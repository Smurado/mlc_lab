#pragma once
#include "teir.hpp"
#include <string>

struct BenchmarkResult {
    double runtime_ms;
    double gflops;
};

// Interpretiert den aktuellen Schedule und misst die Zeit
BenchmarkResult benchmark(const TEIR& ir);

// Speichert das Ergebnis sauber in eine CSV-Datei (mit Strategie-Spalte)
void saveToCSV(const std::string& filename, const std::string& configName,
               const BenchmarkResult& res, const std::string& strategy);