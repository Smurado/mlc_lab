#include "autotuner.hpp"
#include "passes.hpp"
#include "benchmark.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>

// Generiert den diskreten Suchraum basierend auf Regeln und filtert (Pruning)
std::vector<TuningConfig> generateSearchSpace(const TEIR& baseIr) {
    std::vector<TuningConfig> space;

    // 1. Mögliche Split-Faktoren definieren
    std::vector<int> potentialFactors = {2, 4, 8, 16, 32, 64};
    
    // Finde das originale Extent der Achse 'p' für die Teilungsregel
    int pExtent = 128; 
    for (const auto& ax : baseIr.axes) {
        if (ax.name == "p") { pExtent = ax.extent; break; }
    }

    // 2. Mögliche Loop-Reihenfolgen (Permutationen) definieren
    // Da p gesplittet wird in p0 und p1, betrachten wir Permutationen dieser Ketten
    std::vector<std::vector<std::string>> orders = {
        {"p0", "p1", "r", "t"},
        {"r", "p0", "p1", "t"},
        {"r", "t", "p0", "p1"},
        {"p0", "r", "p1", "t"}
    };

    // 3. Mögliche Achsen für Parallelisierung
    std::vector<std::string> parallelOptions = {"", "p0", "r", "t", "p1"};

    // Verschachtelte Generierung des Suchraums
    for (int factor : potentialFactors) {
        // --- RULE-BASED PRUNING 1 ---
        // Teste nur Faktoren, die das Achsen-Extent mathematisch exakt teilen
        if (pExtent % factor != 0) continue; 

        for (const auto& order : orders) {
            for (const auto& parallelAxis : parallelOptions) {
                
                // --- RULE-BASED PRUNING 2 ---
                // Paralleles Loop-Pruning: Parallelisiere niemals die innerste Schleife!
                // Wenn die gewählte Achse ganz hinten in der Kette steht, verwerfe den Trial.
                if (!parallelAxis.empty() && !order.empty() && order.back() == parallelAxis) {
                    continue; 
                }

                space.push_back({factor, order, parallelAxis});
            }
        }
    }
    return space;
}

TuningConfig runAutotuner(const TEIR& baseIr) {
    return runAutotuner(baseIr, AutotunerOptions{});
}

TuningConfig runAutotuner(const TEIR& baseIr, const AutotunerOptions& opts) {
    auto searchSpace = generateSearchSpace(baseIr);

    // Reproduzierbares Shuffle, damit Early-Stopping nicht durch eine
    // deterministisch schlechte Reihenfolge getaeuscht wird.
    if (opts.shuffle) {
        std::mt19937 rng(opts.seed);
        std::shuffle(searchSpace.begin(), searchSpace.end(), rng);
    }

    std::cout << "[AUTOTUNER] Suchraum generiert. Teste " << searchSpace.size()
              << " valide Kandidaten (nach Pruning).\n";
    std::cout << "[AUTOTUNER] Early-Stopping: patience=" << opts.patience
              << ", minImprovement=" << (opts.minImprovementRel * 100.0) << "%"
              << ", timeBudget=" << (opts.timeBudgetMs / 1000.0) << "s"
              << ", shuffle=" << (opts.shuffle ? "on" : "off") << "\n";
    std::cout << "[AUTOTUNER] Jeder Trial wird JIT-kompiliert, auf Korrektheit geprueft und real gebenchmarkt...\n\n";

    TuningConfig bestConfig;
    BenchmarkResult bestResult = { std::numeric_limits<double>::infinity(), 0.0 };
    int trialCount = 0;
    int validCount = 0;
    int rejectedCount = 0;
    int trialsSinceImprovement = 0;
    bool stoppedEarly = false;
    const char* stopReason = "";

    const auto startTime = std::chrono::high_resolution_clock::now();

    for (const auto& config : searchSpace) {
        trialCount++;

        // --- Zeit-Budget pruefen (Sicherheitsnetz) ---
        const auto now = std::chrono::high_resolution_clock::now();
        const double elapsedMs =
            std::chrono::duration<double, std::milli>(now - startTime).count();
        if (elapsedMs >= opts.timeBudgetMs) {
            stoppedEarly = true;
            stopReason = "Zeit-Budget erreicht";
            break;
        }

        // --- Patience-Budget pruefen (Stagnations-Abbruch) ---
        if (trialsSinceImprovement >= opts.patience) {
            stoppedEarly = true;
            stopReason = "Patience erschöpft (Stagnation)";
            break;
        }

        // Klone die Basis-IR für diesen spezifischen Trial
        TEIR trialIr = baseIr;

        // Wende die Transformationen an
        splitOuterAxis(trialIr, "p", config.split_factor);
        reorderSchedule(trialIr, config.loop_order);

        // Genau EINE Achse soll parallel sein: alle zuruecksetzen, dann gezielt setzen.
        // (Split/Input koennen sonst mehrere Achsen als parallel hinterlassen -> Races.)
        for (auto& it : trialIr.schedule) it.policy = Policy::Sequential;
        if (!config.parallel_axis.empty()) {
            makeParallel(trialIr, config.parallel_axis);
        }

        // Real benchmarken (kompiliert, verifiziert, misst echte Zeit)
        BenchmarkResult res = benchmark(trialIr);

        // Inkorrekte / racy Konfigurationen verwerfen (runtime == +inf)
        if (!std::isfinite(res.runtime_ms)) {
            rejectedCount++;
            continue;
        }
        validCount++;

        // Nur valide Trials in CSV loggen
        saveToCSV("autotuner_results.csv", config.toString(), res);

        // Evaluieren, ob es der bisher beste Trial ist (mit Min-Delta gegen Jitter)
        // Sonderfall: initialer Best ist +inf -> jede finite Zeit ist eine Verbesserung.
        const bool isInitialBest = !std::isfinite(bestResult.runtime_ms);
        const double improvementAbs = bestResult.runtime_ms - res.runtime_ms;
        const double improvementRel =
            (std::isfinite(bestResult.runtime_ms) && bestResult.runtime_ms > 0.0)
                ? improvementAbs / bestResult.runtime_ms
                : 1.0;

        if (isInitialBest || improvementRel >= opts.minImprovementRel) {
            bestResult = res;
            bestConfig = config;
            trialsSinceImprovement = 0;
            std::cout << "[NEW BEST] Trial #" << trialCount << "/" << searchSpace.size() << " | "
                      << "SF=" << config.split_factor << ", Parallel=" << (config.parallel_axis.empty() ? "none" : config.parallel_axis)
                      << " -> Zeit: " << res.runtime_ms << " ms (" << res.gflops << " GFLOPS)\n";
        } else {
            trialsSinceImprovement++;
        }
    }

    const auto endTime = std::chrono::high_resolution_clock::now();
    const double totalMs =
        std::chrono::duration<double, std::milli>(endTime - startTime).count();

    std::cout << "\n==================================================\n";
    std::cout << "   AUTOTUNING FERTIG!\n";
    std::cout << "==================================================\n";
    std::cout << "Getestet: " << trialCount << "/" << searchSpace.size()
              << " | Valide: " << validCount
              << " | Verworfen (inkorrekt/Race): " << rejectedCount << "\n";
    if (stoppedEarly) {
        std::cout << "[EARLY STOP] Abbruch nach " << trialCount << " Trials: "
                  << stopReason << "\n";
        std::cout << "             Uebersprungen: " << (searchSpace.size() - trialCount)
                  << " Kandidaten.\n";
    }
    std::cout << "Gesamtlaufzeit: " << (totalMs / 1000.0) << " s\n";
    std::cout << "Beste Konfiguration:\n";
    std::cout << "  - Split Factor: " << bestConfig.split_factor << "\n";
    std::cout << "  - Loop Order:   ";
    for (const auto& o : bestConfig.loop_order) std::cout << o << " ";
    std::cout << "\n  - Parallel Axis: " << (bestConfig.parallel_axis.empty() ? "none" : bestConfig.parallel_axis) << "\n";
    std::cout << "  - Performance:   " << bestResult.runtime_ms << " ms (" << bestResult.gflops << " GFLOPS)\n";
    std::cout << "Alle validen Profile wurden in 'autotuner_results.csv' abgelegt.\n";

    return bestConfig;
}