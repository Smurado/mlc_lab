#pragma once
#include "teir.hpp"
#include <string>
#include <vector>

struct TuningConfig {
    int split_factor;
    std::vector<std::string> loop_order;
    std::string parallel_axis;

    // Generiert einen kompakten String-Namen für die CSV/Ausgabe
    std::string toString() const {
        std::string orderStr = "";
        for (const auto& o : loop_order) orderStr += o + "-";
        if (!orderStr.empty()) orderStr.pop_back();
        return "split_" + std::to_string(split_factor) + "_order_" + orderStr + "_parallel_" + (parallel_axis.empty() ? "none" : parallel_axis);
    }

    bool operator==(const TuningConfig& other) const {
        return split_factor == other.split_factor &&
               loop_order == other.loop_order &&
               parallel_axis == other.parallel_axis;
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
    SearchStrategy strategy = SearchStrategy::SIMULATED_ANNEALING;

    // Simulated-Annealing-Parameter
    double saInitialTemp = 1.0;      // initiale Temperatur (relativ zur ersten Zeit)
    double saCoolingRate = 0.95;     // geometrische Abkuehlung pro Schritt
    int saNeighborsPerTemp = 5;      // Nachbarn pro Temperaturniveau

    // Genetic-Algorithm-Parameter
    int gaPopulationSize = 12;       // Groesse der Population
    int gaGenerations = 8;           // Anzahl Generationen
    double gaMutationRate = 0.3;     // Wahrscheinlichkeit fuer Mutation eines Gens
    double gaEliteFraction = 0.25;   // Anteil der Elite (unveraendert uebernommen)
};

// Startet den Autotuning-Prozess basierend auf einer Start-IR und liefert die
// beste (real gemessene) Konfiguration zurueck. Verwendet Default-Optionen.
TuningConfig runAutotuner(const TEIR& baseIr);

// Startet den Autotuning-Prozess mit expliziten Optionen (Suchstrategie,
// Early-Stopping, Shuffle, Zeit-Budget).
TuningConfig runAutotuner(const TEIR& baseIr, const AutotunerOptions& opts);

// Liefert einen menschenlesbaren Namen fuer eine Suchstrategie.
std::string strategyName(SearchStrategy s);