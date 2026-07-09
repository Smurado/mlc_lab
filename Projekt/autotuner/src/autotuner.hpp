#pragma once
#include "teir.hpp"
#include <string>
#include <vector>
struct TuningConfig {
    std::string split_axis = "";   // Reduktions-Achse die gesplittet wird ("" = kein Split)
    int split_factor = 1;          // Split-Faktor (1 = kein Split)
    std::vector<std::string> loop_order;
    std::string parallel_axis = "";
    int unroll_factor = 1;

    std::string toString() const {
        std::string orderStr = "";
        for (const auto& o : loop_order) orderStr += o + "-";
        if (!orderStr.empty()) orderStr.pop_back();
        return "split_" + split_axis + std::to_string(split_factor) +
               "_order_" + orderStr +
               "_parallel_" + (parallel_axis.empty() ? "none" : parallel_axis) +
               "_unroll_" + std::to_string(unroll_factor);
    }

    bool operator==(const TuningConfig& other) const {
        return split_axis == other.split_axis &&
               split_factor == other.split_factor &&
               loop_order == other.loop_order &&
               parallel_axis == other.parallel_axis &&
               unroll_factor == other.unroll_factor;
    }
};

// Verfügbare Suchstrategien für den Autotuner.
// - RANDOM_SEARCH:        stochastische Baseline (Ablation-Vergleich)
// - SIMULATED_ANNEALING:  TVM-Default, probabilistisches Akzeptieren von Verschlechterungen
// - GENETIC:              populationsbasiert mit Crossover/Mutation
enum class SearchStrategy { RANDOM_SEARCH, SIMULATED_ANNEALING, GENETIC };

struct AutotunerOptions {
    int patience = 20;               // Trials ohne Verbesserung bis zum Abbruch
    double minImprovementRel = 0.01; // relative Schwelle (1 %) fuer "neuer Best"
    unsigned seed = 42;              // Seed fuer reproduzierbare Zufallszahl
    bool shuffle = true;             // Suchraum vor der Iteration mischen
    double timeBudgetMs = 60000.0;   // harte Abbruchgrenze (Sicherheitsnetz)
    int maxTrials = 0;               // max. Anzahl JIT-Trials (0 = deaktiviert).
                                     // Fuer FAIRE Strategie-Vergleiche: gleiche
                                     // Trial-Zahl statt gleicher Wanduhrzeit.
    SearchStrategy strategy = SearchStrategy::SIMULATED_ANNEALING;
    Backend backend = Backend::Scalar; // Codegen-Backend (Scalar, NEON oder SME)

    // Simulated-Annealing-Parameter
    double saInitialTemp = 1.0;      // initiale Temperatur (relativ zur ersten Zeit)
    double saCoolingRate = 0.95;     // geometrische Abkuehlung pro Schritt
    int saNeighborsPerTemp = 5;      // Nachbarn pro Temperaturniveau

    // Genetic-Algorithm-Parameter
    int gaPopulationSize = 12;       // Groesse der Population
    int gaGenerations = 8;           // Anzahl Generationen
    double gaMutationRate = 0.3;     // Wahrscheinlichkeit fuer Mutation eines Gens
    double gaEliteFraction = 0.25;   // Anteil der Elite (unveraendert uebernommen)
    // B2: max. Remutationen, um ein bereits besuchtes Kind durch ein unbesuchtes
    // zu ersetzen (statt es ersatzlos zu ueberspringen -> Leerlauf). 0 = altes
    // Verhalten (ueberspringen), fuer die Ablation.
    int gaRemutateTries = 25;

    // CostModel-Vorfilter: nur die Top X% der Kandidaten (nach Schaetzung)
    // werden JIT-kompiliert. 0.0 = deaktiviert, 0.5 = Top 50%, 1.0 = alle.
    double costModelFilterPct = 0.3;

    // B3: SA/GA starten am Cost-Model-Optimum (bestplatzierte gefilterte Config)
    // statt an einer zufaelligen. Lokales Suchen zahlt sich erst mit gutem Start
    // aus. Random Search bleibt bewusst uninformiert (faire Baseline).
    bool warmStart = true;
};

// Startet den Autotuning-Prozess basierend auf einer Start-IR und liefert die
// beste (real gemessene) Konfiguration zurueck. Verwendet Default-Optionen.
TuningConfig runAutotuner(const TEIR& baseIr);

// Startet den Autotuning-Prozess mit expliziten Optionen (Suchstrategie,
// Early-Stopping, Shuffle, Zeit-Budget).
TuningConfig runAutotuner(const TEIR& baseIr, const AutotunerOptions& opts);

// Liefert einen menschenlesbaren Namen fuer eine Suchstrategie.
std::string strategyName(SearchStrategy s);