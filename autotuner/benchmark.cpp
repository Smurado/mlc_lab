#include "benchmark.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>

// Rekursiver Loop-Interpreter zur Simulation des Schedules
void runLoopNest(const TEIR& ir, size_t depth, int currentStrideIndex, const std::vector<int>& extents, float* dummyMem, volatile float& accumulator) {
    if (depth == ir.schedule.size()) {
        // Innere Kernel-Berechnung (Simulierter Flop-Inhalt)
        // Nutze den berechneten Stride-Index, um Cache-Lokalität zu erzwingen
        accumulator += dummyMem[currentStrideIndex & 0xFFFFF]; // Begrenzt auf 1MB Cache-Größe
        return;
    }

    std::string currentAxis = ir.schedule[depth].axis;
    int extent = 1;
    // Finde das passende Extent für die aktuelle Achse
    for (const auto& ax : ir.axes) {
        if (ax.name == currentAxis) {
            extent = ax.extent;
            break;
        }
    }

    // Künstlicher Speicher-Stride basierend auf dem Achsennamen zur Cache-Simulation
    // Achsen bekommen unterschiedliche Strides, um Reordering-Effekte sichtbar zu machen
    int stride = 1;
    if (!currentAxis.empty()) {
        stride = static_cast<int>(currentAxis.back()) * 7; // Pseudo-Zufälliger, aber fester Stride per Achse
    }

    for (int i = 0; i < extent; ++i) {
        runLoopNest(ir, depth + 1, currentStrideIndex + (i * stride), extents, dummyMem, accumulator);
    }
}

BenchmarkResult benchmark(const TEIR& ir) {
    // Berechne die totalen Flops (Jede innere Schleifeninstanz simuliert 2 FLOPs: MAC-Operation)
    double totalIterations = 1.0;
    std::vector<int> extents;
    for (const auto& iter : ir.schedule) {
        for (const auto& ax : ir.axes) {
            if (ax.name == iter.axis) {
                totalIterations *= ax.extent;
                extents.push_back(ax.extent);
                break;
            }
        }
    }
    double totalFlops = totalIterations * 2.0;

    // Allokiere Dummy-Speicher für die Cache-Simulation (ca. 4MB)
    size_t memSize = 1024 * 1024;
    std::vector<float> dummyMem(memSize, 1.0f);
    volatile float accumulator = 0.0f;

    // Zeitmessung starten
    auto start = std::chrono::high_resolution_clock::now();
    
    // Starte den simulierten Loop-Nest-Interpreter
    runLoopNest(ir, 0, 0, extents, dummyMem.data(), accumulator);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    double runtime_ms = duration.count();
    // GFLOPS = (Total Flops / 10^9) / (Runtime in Sekunden)
    double runtime_sec = runtime_ms / 1000.0;
    double gflops = (runtime_sec > 0) ? (totalFlops / 1e9) / runtime_sec : 0.0;

    return {runtime_ms, gflops};
}

void saveToCSV(const std::string& filename, const std::string& configName, const BenchmarkResult& res) {
    std::ofstream file;
    // Prüfe, ob Datei bereits existiert, um Header zu schreiben
    std::ifstream check(filename);
    bool exists = check.good();
    check.close();

    file.open(filename, std::ios_base::app);
    if (!exists) {
        file << "config,runtime_ms,gflops\n";
    }
    file << configName << "," << res.runtime_ms << "," << res.gflops << "\n";
    file.close();
}