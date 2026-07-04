#include "autotuner.hpp"
#include "passes.hpp"
#include "benchmark.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <numeric>
#include <unordered_set>

// =====================================================================
// Suchraum-Generierung (wie bisher, mit regelbasiertem Pruning)
// =====================================================================

std::vector<TuningConfig> generateSearchSpace(const TEIR& baseIr) {
    std::vector<TuningConfig> space;

    // Erweiterte Split-Faktoren (inkl. 1 = kein Split, 128 = voll)
    std::vector<int> potentialFactors = {1, 2, 4, 8, 16, 32, 64, 128};

    int pExtent = 128;
    for (const auto& ax : baseIr.axes) {
        if (ax.name == "p") { pExtent = ax.extent; break; }
    }

    // Alle 24 Permutationen von {p0, p1, r, t} via next_permutation
    std::vector<std::string> baseOrder = {"p0", "p1", "r", "t"};
    std::sort(baseOrder.begin(), baseOrder.end());
    std::vector<std::vector<std::string>> orders;
    do {
        orders.push_back(baseOrder);
    } while (std::next_permutation(baseOrder.begin(), baseOrder.end()));

    std::vector<std::string> parallelOptions = {"", "p0", "r", "t", "p1"};

    // Unroll-Faktoren fuer die innerste Schleife
    std::vector<int> unrollFactors = {1, 2, 4, 8, 16};

    for (int factor : potentialFactors) {
        if (pExtent % factor != 0) continue;

        for (const auto& order : orders) {
            for (const auto& parallelAxis : parallelOptions) {
                // Pruning: parallelisiere nie die innerste Schleife
                if (!parallelAxis.empty() && !order.empty() && order.back() == parallelAxis) {
                    continue;
                }
                for (int unroll : unrollFactors) {
                    space.push_back({factor, order, parallelAxis, unroll});
                }
            }
        }
    }
    return space;
}

// =====================================================================
// Gemeinsamer Kontext: Best-Tracking, Budget-Checks, Trial-Evaluation
// =====================================================================

namespace {

struct SearchContext {
    const TEIR& baseIr;
    const AutotunerOptions& opts;
    std::mt19937 rng;
    std::chrono::high_resolution_clock::time_point startTime;

    // Best-Ergebnis-Tracking
    TuningConfig bestConfig;
    BenchmarkResult bestResult;
    int trialsSinceImprovement = 0;

    // Statistik
    int trialCount = 0;
    int validCount = 0;
    int rejectedCount = 0;
    std::unordered_set<size_t> visited; // Hashes der besuchten Configs

    bool stoppedEarly = false;
    std::string stopReason;

    SearchContext(const TEIR& ir, const AutotunerOptions& o)
        : baseIr(ir), opts(o), rng(o.seed),
          startTime(std::chrono::high_resolution_clock::now()),
          bestResult{std::numeric_limits<double>::infinity(), 0.0} {}

    double elapsedMs() const {
        const auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - startTime).count();
    }

    bool budgetExceeded() const {
        return elapsedMs() >= opts.timeBudgetMs;
    }

    bool patienceExceeded() const {
        return trialsSinceImprovement >= opts.patience;
    }

    // Hash einer Config, um besuchte Trials nicht doppelt zu evaluieren.
    size_t hashConfig(const TuningConfig& c) const {
        size_t h = std::hash<int>{}(c.split_factor);
        h ^= std::hash<int>{}(c.unroll_factor) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<std::string>{}(c.parallel_axis) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (const auto& o : c.loop_order) {
            h ^= std::hash<std::string>{}(o) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }

    bool alreadyVisited(const TuningConfig& c) const {
        return visited.count(hashConfig(c)) > 0;
    }

    // Wendet die Transformationen auf einen Klon der Basis-IR an und benchmarkt.
    // Liefert finite BenchmarkResult fuer valide Configs, +inf fuer inkorrekte.
    BenchmarkResult evaluateTrial(const TuningConfig& config) {
        TEIR trialIr = baseIr;

        splitOuterAxis(trialIr, "p", config.split_factor);
        reorderSchedule(trialIr, config.loop_order);

        for (auto& it : trialIr.schedule) it.policy = Policy::Sequential;
        if (!config.parallel_axis.empty()) {
            makeParallel(trialIr, config.parallel_axis);
        }

        trialIr.unrollFactor = config.unroll_factor;
        trialIr.backend = opts.backend;

        return benchmark(trialIr);
    }

    // Nimmt ein Benchmark-Ergebnis entgegen, prueft Korrektheit, loggt in CSV
    // und aktualisiert ggf. den bisher besten Trial. Liefert true, wenn ein
    // neuer Best gefunden wurde.
    bool absorbTrial(const TuningConfig& config, const BenchmarkResult& res,
                     const std::string& strategyName) {
        trialCount++;
        visited.insert(hashConfig(config));

        if (!std::isfinite(res.runtime_ms)) {
            rejectedCount++;
            return false;
        }
        validCount++;

        saveToCSV("autotuner_results.csv", config.toString(), res, strategyName);

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
            std::cout << "[NEW BEST] Trial #" << trialCount << " | "
                      << "SF=" << config.split_factor
                      << ", Parallel=" << (config.parallel_axis.empty() ? "none" : config.parallel_axis)
                      << " -> Zeit: " << res.runtime_ms << " ms ("
                      << res.gflops << " GFLOPS)\n";
            return true;
        }
        trialsSinceImprovement++;
        return false;
    }
};

// Prueft die Abbruchbedingungen und setzt stopReason/stopReason bei Trigger.
// Liefert true, wenn die Suche abgebrochen werden soll.
bool checkStop(SearchContext& ctx) {
    if (ctx.budgetExceeded()) {
        ctx.stoppedEarly = true;
        ctx.stopReason = "Zeit-Budget erreicht";
        return true;
    }
    if (ctx.patienceExceeded()) {
        ctx.stoppedEarly = true;
        ctx.stopReason = "Patience erschöpft (Stagnation)";
        return true;
    }
    return false;
}

void printSummary(const SearchContext& ctx, const std::string& strategyName,
                  int spaceSize) {
    const double totalMs = ctx.elapsedMs();

    std::cout << "\n==================================================\n";
    std::cout << "   AUTOTUNING FERTIG! [" << strategyName << "]\n";
    std::cout << "==================================================\n";
    std::cout << "Getestet: " << ctx.trialCount << "/" << spaceSize
              << " | Valide: " << ctx.validCount
              << " | Verworfen (inkorrekt/Race): " << ctx.rejectedCount << "\n";
    if (ctx.stoppedEarly) {
        std::cout << "[EARLY STOP] Abbruch nach " << ctx.trialCount << " Trials: "
                  << ctx.stopReason << "\n";
        std::cout << "             Uebersprungen: " << (spaceSize - ctx.trialCount)
                  << " Kandidaten.\n";
    }
    std::cout << "Gesamtlaufzeit: " << (totalMs / 1000.0) << " s\n";
    std::cout << "Beste Konfiguration:\n";
    std::cout << "  - Split Factor: " << ctx.bestConfig.split_factor << "\n";
    std::cout << "  - Loop Order:   ";
    for (const auto& o : ctx.bestConfig.loop_order) std::cout << o << " ";
    std::cout << "\n  - Parallel Axis: "
              << (ctx.bestConfig.parallel_axis.empty() ? "none" : ctx.bestConfig.parallel_axis) << "\n";
    std::cout << "  - Performance:   " << ctx.bestResult.runtime_ms << " ms ("
              << ctx.bestResult.gflops << " GFLOPS)\n";
    std::cout << "Alle validen Profile wurden in 'autotuner_results.csv' abgelegt.\n";
}

// =====================================================================
// Strategie 1: Random Search (stochastische Baseline)
// =====================================================================

TuningConfig runRandomSearch(SearchContext& ctx,
                             const std::vector<TuningConfig>& space) {
    const std::string name = strategyName(SearchStrategy::RANDOM_SEARCH);

    // Shuffle-Kopie, damit wir zufaellig, aber ohne Wiederholung ziehen koennen.
    std::vector<TuningConfig> queue = space;
    std::shuffle(queue.begin(), queue.end(), ctx.rng);

    for (const auto& config : queue) {
        if (checkStop(ctx)) break;

        const BenchmarkResult res = ctx.evaluateTrial(config);
        ctx.absorbTrial(config, res, name);
    }
    return ctx.bestConfig;
}

// =====================================================================
// Strategie 2: Simulated Annealing
// =====================================================================

// Erzeugt einen zufaelligen Nachbarn einer Config durch kleine Mutation.
TuningConfig mutateNeighbor(std::mt19937& rng, const TuningConfig& base,
                            const std::vector<TuningConfig>& space) {
    // Wir waehlen einen zufaelligen Nachbarn aus dem Suchraum, der mindestens
    // ein Gen mit base teilt (gleicher Split-Faktor ODER gleiche Loop-Order).
    std::vector<const TuningConfig*> candidates;
    for (const auto& c : space) {
        if (c.split_factor == base.split_factor ||
            c.loop_order == base.loop_order) {
            candidates.push_back(&c);
        }
    }
    if (candidates.empty()) return base;
    std::uniform_int_distribution<int> pick(0, (int)candidates.size() - 1);
    return *candidates[pick(rng)];
}

TuningConfig runSimulatedAnnealing(SearchContext& ctx,
                                   const std::vector<TuningConfig>& space) {
    const std::string name = strategyName(SearchStrategy::SIMULATED_ANNEALING);

    // Start: zufaellige initiale Config
    std::uniform_int_distribution<int> pickSpace(0, (int)space.size() - 1);
    TuningConfig current = space[pickSpace(ctx.rng)];
    BenchmarkResult currentRes = ctx.evaluateTrial(current);
    ctx.absorbTrial(current, currentRes, name);

    // Temperatur relativ zur ersten gemessenen Zeit skalieren, damit
    // exp(-delta/temp) auch bei Mikrosekunden-Kernels sinnvoll ist.
    double temp = ctx.opts.saInitialTemp * std::max(1.0, currentRes.runtime_ms);
    const double cooling = ctx.opts.saCoolingRate;
    const int neighborsPerTemp = ctx.opts.saNeighborsPerTemp;

    while (temp > 1e-3) {
        for (int i = 0; i < neighborsPerTemp; ++i) {
            if (checkStop(ctx)) goto sa_done;

            const TuningConfig neighbor = mutateNeighbor(ctx.rng, current, space);
            if (ctx.alreadyVisited(neighbor)) continue;

            const BenchmarkResult neighborRes = ctx.evaluateTrial(neighbor);

            // Jeder evaluierte Trial wird in CSV/Best-Tracking aufgenommen.
            ctx.absorbTrial(neighbor, neighborRes, name);

            // Akzeptanz-Entscheidung fuer den SA-Random-Walk.
            if (!std::isfinite(neighborRes.runtime_ms)) continue; // inkorrekt

            const double delta = neighborRes.runtime_ms - currentRes.runtime_ms;
            bool accept = false;
            if (delta < 0) {
                accept = true;
            } else {
                std::uniform_real_distribution<double> prob(0.0, 1.0);
                const double p = std::exp(-delta / temp);
                accept = (prob(ctx.rng) < p);
            }

            if (accept) {
                current = neighbor;
                currentRes = neighborRes;
            }
        }
        temp *= cooling;
    }
sa_done:
    return ctx.bestConfig;
}

// =====================================================================
// Strategie 3: Genetic Algorithm
// =====================================================================

struct Individual {
    TuningConfig config;
    BenchmarkResult fitness; // runtime_ms als Fitness (niedriger = besser)
    bool evaluated = false;
};

// Erzeugt zwei Nachkommen aus zwei Eltern via Crossover (Gen-fuer-Gen).
// Da Gene (split_factor, loop_order, parallel_axis) einzeln sind, mischen wir
// sie komponentenweise.
std::pair<TuningConfig, TuningConfig> crossover(std::mt19937& rng,
                                                const TuningConfig& a,
                                                const TuningConfig& b) {
    TuningConfig child1 = a, child2 = b;
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    // Split-Faktor
    if (coin(rng) < 0.5) {
        child1.split_factor = b.split_factor;
        child2.split_factor = a.split_factor;
    }
    // Loop-Order
    if (coin(rng) < 0.5) {
        child1.loop_order = b.loop_order;
        child2.loop_order = a.loop_order;
    }
    // Parallel-Axis
    if (coin(rng) < 0.5) {
        child1.parallel_axis = b.parallel_axis;
        child2.parallel_axis = a.parallel_axis;
    }
    return {child1, child2};
}

// Mutiert ein Gen zufaellig, bezogen auf den Suchraum.
void mutate(std::mt19937& rng, TuningConfig& c,
            const std::vector<TuningConfig>& space) {
    std::uniform_int_distribution<int> pickGene(0, 2);
    std::uniform_int_distribution<int> pickSpace(0, (int)space.size() - 1);

    const TuningConfig& donor = space[pickSpace(rng)];
    switch (pickGene(rng)) {
        case 0: c.split_factor = donor.split_factor; break;
        case 1: c.loop_order = donor.loop_order; break;
        case 2: c.parallel_axis = donor.parallel_axis; break;
    }
}

// Liefert eine valide, noch nicht besuchte Config nahe der Eingabe.
// Falls Crossover/Mutation eine ungueltige Config ergeben, fallen wir auf
// eine zufaellige valide aus dem Suchraum zurueck.
TuningConfig repair(std::mt19937& rng, TuningConfig c,
                    const std::vector<TuningConfig>& space,
                    [[maybe_unused]] const SearchContext& ctx) {
    // Pruefe, ob c im Suchraum existiert (sonst ist sie ungueltig).
    auto it = std::find(space.begin(), space.end(), c);
    if (it == space.end()) {
        std::uniform_int_distribution<int> pick(0, (int)space.size() - 1);
        return space[pick(rng)];
    }
    return c;
}

TuningConfig runGeneticAlgorithm(SearchContext& ctx,
                                 const std::vector<TuningConfig>& space) {
    const std::string name = strategyName(SearchStrategy::GENETIC);

    // Initial Population: zufaellige Stichprobe aus dem Suchraum
    const int popSize = std::min(ctx.opts.gaPopulationSize, (int)space.size());
    std::vector<Individual> population(popSize);
    {
        std::vector<int> indices(space.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), ctx.rng);
        for (int i = 0; i < popSize; ++i) {
            population[i].config = space[indices[i]];
        }
    }

    // Evaluiere initiale Population
    for (auto& ind : population) {
        if (checkStop(ctx)) goto ga_done;
        ind.fitness = ctx.evaluateTrial(ind.config);
        ind.evaluated = true;
        ctx.absorbTrial(ind.config, ind.fitness, name);
    }

    for (int gen = 0; gen < ctx.opts.gaGenerations; ++gen) {
        if (checkStop(ctx)) break;

        // Sortiere nach Fitness (aufsteigend = beste zuerst)
        std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.fitness.runtime_ms < b.fitness.runtime_ms;
        });

        // Selektion: Elite + Nachkommen
        const int eliteCount = std::max(1, (int)(popSize * ctx.opts.gaEliteFraction));
        std::vector<Individual> nextGen;
        nextGen.reserve(popSize);

        // Elite unverändert übernehmen
        for (int i = 0; i < eliteCount && i < (int)population.size(); ++i) {
            nextGen.push_back(population[i]);
        }

        // Nachkommen via Tournament-Selektion + Crossover + Mutation
        std::uniform_int_distribution<int> pickPop(0, (int)population.size() - 1);
        while ((int)nextGen.size() < popSize) {
            if (checkStop(ctx)) break;

            // Tournament (Größe 2): bessere von zwei zufälligen Eltern
            const Individual& parentA = [&]() -> const Individual& {
                const Individual& c1 = population[pickPop(ctx.rng)];
                const Individual& c2 = population[pickPop(ctx.rng)];
                return (c1.fitness.runtime_ms < c2.fitness.runtime_ms) ? c1 : c2;
            }();
            const Individual& parentB = [&]() -> const Individual& {
                const Individual& c1 = population[pickPop(ctx.rng)];
                const Individual& c2 = population[pickPop(ctx.rng)];
                return (c1.fitness.runtime_ms < c2.fitness.runtime_ms) ? c1 : c2;
            }();

            auto [c1, c2] = crossover(ctx.rng, parentA.config, parentB.config);
            std::uniform_real_distribution<double> mutProb(0.0, 1.0);
            if (mutProb(ctx.rng) < ctx.opts.gaMutationRate) mutate(ctx.rng, c1, space);
            if (mutProb(ctx.rng) < ctx.opts.gaMutationRate) mutate(ctx.rng, c2, space);

            c1 = repair(ctx.rng, c1, space, ctx);
            c2 = repair(ctx.rng, c2, space, ctx);

            for (const TuningConfig& childCfg : {c1, c2}) {
                if ((int)nextGen.size() >= popSize) break;
                if (ctx.alreadyVisited(childCfg)) {
                    // Schon evaluiert: aus Population holen oder überspringen.
                    // Wir nehmen einfach die Config direkt als evaluiert an,
                    // wenn sie schon im Best ist, sonst neu bewerten.
                    continue;
                }
                Individual child;
                child.config = childCfg;
                child.fitness = ctx.evaluateTrial(childCfg);
                child.evaluated = true;
                ctx.absorbTrial(child.config, child.fitness, name);
                nextGen.push_back(child);
            }
        }

        if (checkStop(ctx)) break;
        population = std::move(nextGen);
    }

ga_done:
    return ctx.bestConfig;
}

} // namespace

// =====================================================================
// Oeffentliche API
// =====================================================================

std::string strategyName(SearchStrategy s) {
    switch (s) {
        case SearchStrategy::RANDOM_SEARCH:        return "RandomSearch";
        case SearchStrategy::SIMULATED_ANNEALING:  return "SimulatedAnnealing";
        case SearchStrategy::GENETIC:              return "GeneticAlgorithm";
    }
    return "Unknown";
}

TuningConfig runAutotuner(const TEIR& baseIr) {
    return runAutotuner(baseIr, AutotunerOptions{});
}

TuningConfig runAutotuner(const TEIR& baseIr, const AutotunerOptions& opts) {
    auto searchSpace = generateSearchSpace(baseIr);

    std::cout << "[AUTOTUNER] Suchraum generiert. " << searchSpace.size()
              << " valide Kandidaten (nach Pruning).\n";
    std::cout << "[AUTOTUNER] Strategie: " << strategyName(opts.strategy)
              << " | patience=" << opts.patience
              << ", minImprovement=" << (opts.minImprovementRel * 100.0) << "%"
              << ", timeBudget=" << (opts.timeBudgetMs / 1000.0) << "s\n";
    std::cout << "[AUTOTUNER] Jeder Trial wird JIT-kompiliert, auf Korrektheit geprueft und real gebenchmarkt...\n\n";

    SearchContext ctx(baseIr, opts);
    TuningConfig best;

    switch (opts.strategy) {
        case SearchStrategy::RANDOM_SEARCH:
            best = runRandomSearch(ctx, searchSpace);
            break;
        case SearchStrategy::SIMULATED_ANNEALING:
            best = runSimulatedAnnealing(ctx, searchSpace);
            break;
        case SearchStrategy::GENETIC:
            best = runGeneticAlgorithm(ctx, searchSpace);
            break;
    }

    printSummary(ctx, strategyName(opts.strategy), (int)searchSpace.size());
    return best;
}
