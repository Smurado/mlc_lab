#include "autotuner.hpp"
#include "passes.hpp"
#include "benchmark.hpp"
#include "einsum.hpp"
#include "cost_model.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <numeric>
#include <unordered_set>
#include <cstdlib>

// =====================================================================
// Suchraum-Generierung
// GEMM-Modus (kein einsum): hardcoded p0/p1/r/t wie bisher.
// Einsum-Modus: Suchraum aus Einsum abgeleitet - Reduktions-Achsen innen,
// Output-Achsen aussen, Split auf einer Reduktions-Achse, Parallel auf
// Output-Achsen. Vermeidet Suchraum-Explosion durch Out!×Red! statt N!.
// =====================================================================

static std::vector<TuningConfig> generateSearchSpaceGEMM(const TEIR& baseIr) {
    std::vector<TuningConfig> space;

    std::vector<int> potentialFactors = {1, 2, 4, 8, 16, 32, 64, 128};

    int pExtent = 128;
    for (const auto& ax : baseIr.axes) {
        if (ax.name == "p") { pExtent = ax.extent; break; }
    }

    std::vector<std::string> baseOrder = {"p0", "p1", "r", "t"};
    std::sort(baseOrder.begin(), baseOrder.end());
    std::vector<std::vector<std::string>> orders;
    do {
        orders.push_back(baseOrder);
    } while (std::next_permutation(baseOrder.begin(), baseOrder.end()));

    std::vector<std::string> parallelOptions = {"", "p0", "r", "t", "p1"};
    std::vector<int> unrollFactors = {1, 2, 4, 8, 16};

    for (int factor : potentialFactors) {
        if (pExtent % factor != 0) continue;
        for (const auto& order : orders) {
            for (const auto& parallelAxis : parallelOptions) {
                if (!parallelAxis.empty() && !order.empty() && order.back() == parallelAxis)
                    continue;
                for (int unroll : unrollFactors) {
                    space.push_back({"p", factor, order, parallelAxis, unroll});
                }
            }
        }
    }
    return space;
}

static std::vector<TuningConfig> generateSearchSpaceEinsum(const TEIR& baseIr) {
    std::vector<TuningConfig> space;

    EinsumSpec spec = parseEinsum(baseIr.einsum);

    std::vector<std::string> outAxes, reduceAxes;
    for (char c : spec.out_axes) outAxes.push_back(std::string(1, c));
    for (char c : spec.reduce_axes) reduceAxes.push_back(std::string(1, c));

    std::vector<int> splitFactors = {1, 2, 4, 8, 16, 32, 64};
    std::vector<int> unrollFactors = {1, 2, 4, 8, 16};

    // Permutationen der Output-Achsen (außen)
    std::sort(outAxes.begin(), outAxes.end());
    std::vector<std::vector<std::string>> outPerms;
    do {
        outPerms.push_back(outAxes);
    } while (std::next_permutation(outAxes.begin(), outAxes.end()));

    // Permutationen der Reduktions-Achsen (innen, ohne Split)
    std::sort(reduceAxes.begin(), reduceAxes.end());
    std::vector<std::vector<std::string>> redPermsBase;
    do {
        redPermsBase.push_back(reduceAxes);
    } while (std::next_permutation(reduceAxes.begin(), reduceAxes.end()));

    for (const std::string& splitAxis : reduceAxes) {
        int splitExtent = extentOfChar(baseIr, splitAxis[0]);

        for (int factor : splitFactors) {
            if (factor > 1 && splitExtent % factor != 0) continue;

            std::vector<std::string> otherRed;
            for (const auto& r : reduceAxes)
                if (r != splitAxis) otherRed.push_back(r);

            std::sort(otherRed.begin(), otherRed.end());
            std::vector<std::vector<std::string>> otherRedPerms;
            do {
                otherRedPerms.push_back(otherRed);
            } while (std::next_permutation(otherRed.begin(), otherRed.end()));

            for (const auto& outPerm : outPerms) {
                for (const auto& otherRedPerm : otherRedPerms) {
                    std::vector<std::string> loopOrder = outPerm;
                    std::string split0 = splitAxis + "0";
                    std::string split1 = splitAxis + "1";
                    loopOrder.push_back(split0);
                    loopOrder.push_back(split1);
                    for (const auto& r : otherRedPerm) loopOrder.push_back(r);

                    std::vector<std::string> parallelOptions;
                    parallelOptions.push_back("");
                    for (const auto& o : outPerm) parallelOptions.push_back(o);
                    parallelOptions.push_back(split0);

                    for (const auto& parallel : parallelOptions) {
                        if (!parallel.empty() && loopOrder.back() == parallel)
                            continue;
                        for (int unroll : unrollFactors) {
                            space.push_back({splitAxis, factor, loopOrder, parallel, unroll});
                        }
                    }
                }
            }
        }
    }
    return space;
}

std::vector<TuningConfig> generateSearchSpace(const TEIR& baseIr) {
    if (!baseIr.einsum.empty())
        return generateSearchSpaceEinsum(baseIr);
    return generateSearchSpaceGEMM(baseIr);
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

    // B2: GA-Leerlauf-Statistik (nur die GA-Strategie befuellt diese).
    long gaDuplicateChildren = 0; // Kinder, die schon besucht waren (waeren Leerlaeufe)
    long gaResolved = 0;          // davon per Remutation zu unbesucht aufgeloest
    long gaGaveUp = 0;            // davon nach maxTries aufgegeben
    long gaRemutations = 0;       // gesamte Remutations-Versuche
    long gaStallBreaks = 0;       // Generationen, die wegen erschoepftem Raum abbrachen

    // D1: optionaler Pfad fuer das Konvergenz-Log (Env TEIR_TRIAL_LOG).
    std::string trialLogPath;

    SearchContext(const TEIR& ir, const AutotunerOptions& o)
        : baseIr(ir), opts(o), rng(o.seed),
          startTime(std::chrono::high_resolution_clock::now()),
          bestResult{std::numeric_limits<double>::infinity(), 0.0} {
        if (const char* p = std::getenv("TEIR_TRIAL_LOG")) trialLogPath = p;
    }

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
        size_t h = std::hash<std::string>{}(c.split_axis);
        h ^= std::hash<int>{}(c.split_factor) + 0x9e3779b9 + (h << 6) + (h >> 2);
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

        if (config.split_factor > 1 && !config.split_axis.empty()) {
            splitOuterAxis(trialIr, config.split_axis, config.split_factor);
        }
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

        bool isNewBest = false;

        if (std::isfinite(res.runtime_ms)) {
            validCount++;
            // C3: CostModel-Schaetzung mitloggen (fuer die Rang-Korrelation
            // geschaetzt vs. gemessen). Identische Basis wie der Vorfilter.
            const double costEst = estimateCost(baseIr, config);
            saveToCSV("autotuner_results.csv", config.toString(), res, strategyName, costEst);

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
                isNewBest = true;
                std::cout << "[NEW BEST] Trial #" << trialCount << " | "
                          << "SF=" << config.split_factor
                          << ", Parallel=" << (config.parallel_axis.empty() ? "none" : config.parallel_axis)
                          << " -> Zeit: " << res.runtime_ms << " ms ("
                          << res.gflops << " GFLOPS)\n";
            } else {
                trialsSinceImprovement++;
            }
        } else {
            rejectedCount++;
        }

        // D1: Konvergenz-Log — jeder Trial (valide wie inkorrekt, da beide ins
        // Budget zaehlen). trial_gflops=0 fuer inkorrekte, best_gflops = bisher Bestes.
        if (!trialLogPath.empty()) {
            const double trialG = std::isfinite(res.runtime_ms) ? res.gflops : 0.0;
            const double bestG = std::isfinite(bestResult.runtime_ms) ? bestResult.gflops : 0.0;
            appendTrialLog(trialLogPath, strategyName, opts.seed, trialCount, trialG, bestG);
        }

        return isNewBest;
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
    if (ctx.opts.maxTrials > 0 && ctx.trialCount >= ctx.opts.maxTrials) {
        ctx.stoppedEarly = true;
        ctx.stopReason = "Trial-Budget erreicht";
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
    {
        const long hits = benchmarkCacheHits();
        const long misses = benchmarkCacheMisses();
        const long total = hits + misses;
        const double hitRate = (total > 0) ? (100.0 * hits / total) : 0.0;
        std::cout << "Kernel-Cache: " << hits << " Treffer / " << misses
                  << " JIT-Kompilate (" << hitRate << "% gespart)\n";
        // A5: persistenter Cross-Run-Cache (nur wenn aktiv/getroffen).
        const long phits = benchmarkPersistHits();
        if (phits > 0)
            std::cout << "Persistenter Cache: " << phits
                      << " Cross-Run-Treffer (JIT/Bench uebersprungen)\n";
    }
    // B2: GA-Leerlauf-Vermeidung (nur relevant, wenn ueberhaupt Duplikate auftraten).
    if (ctx.gaDuplicateChildren > 0) {
        std::cout << "GA-Leerlaeufe: " << ctx.gaDuplicateChildren
                  << " doppelte Kinder -> " << ctx.gaResolved
                  << " per Remutation ersetzt (" << ctx.gaRemutations << " Versuche), "
                  << ctx.gaGaveUp << " aufgegeben";
        if (ctx.gaStallBreaks > 0)
            std::cout << ", " << ctx.gaStallBreaks << " Generation(en) wegen erschoepftem Raum beendet";
        std::cout << "\n";
    }
    if (ctx.stoppedEarly) {
        std::cout << "[EARLY STOP] Abbruch nach " << ctx.trialCount << " Trials: "
                  << ctx.stopReason << "\n";
        std::cout << "             Uebersprungen: " << (spaceSize - ctx.trialCount)
                  << " Kandidaten.\n";
    }
    std::cout << "Gesamtlaufzeit: " << (totalMs / 1000.0) << " s\n";
    std::cout << "Beste Konfiguration:\n";
    std::cout << "  - Split Axis:   " << (ctx.bestConfig.split_axis.empty() ? "none" : ctx.bestConfig.split_axis) << "\n";
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

// Liefert den Index des EINEN Gens, in dem sich a und b unterscheiden, oder -1
// falls sie in 0 oder >1 Genen differieren. Gen-Indizes:
// 0=split_axis, 1=split_factor, 2=loop_order, 3=parallel_axis, 4=unroll_factor.
static int singleGeneDiff(const TuningConfig& a, const TuningConfig& b) {
    int diffs = 0, which = -1;
    if (a.split_axis    != b.split_axis)    { ++diffs; which = 0; }
    if (a.split_factor  != b.split_factor)  { ++diffs; which = 1; }
    if (a.loop_order    != b.loop_order)    { ++diffs; which = 2; }
    if (a.parallel_axis != b.parallel_axis) { ++diffs; which = 3; }
    if (a.unroll_factor != b.unroll_factor) { ++diffs; which = 4; }
    return (diffs == 1) ? which : -1;
}

// True, wenn sich zwei Loop-Orders in genau zwei Positionen unterscheiden und
// diese vertauscht sind (Einzel-Transposition) — die lokalste Permutations-
// Mutation.
static bool isSingleSwap(const std::vector<std::string>& a,
                         const std::vector<std::string>& b) {
    if (a.size() != b.size()) return false;
    int p0 = -1, p1 = -1, diffs = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            ++diffs;
            if (p0 < 0) p0 = (int)i; else if (p1 < 0) p1 = (int)i;
        }
    }
    if (diffs != 2) return false;
    return a[p0] == b[p1] && a[p1] == b[p0];
}

// Erzeugt einen ECHTEN lokalen Nachbarn: die zurueckgegebene Config unterscheidet
// sich vom Ausgangspunkt in GENAU EINEM Gen und liegt im (gefilterten) Suchraum.
// Fuer geordnete Gene (Split-Faktor, Unroll) wird der naechstgelegene Wert
// bevorzugt (kleine Schritte), fuer die Loop-Order eine Einzel-Transposition.
// (Vorher: OR-Bedingung ueber den ganzen Raum -> globale Spruenge, faktisch
// Random Search. Das war der Grund, warum SA sich nicht von Random abhob.)
TuningConfig mutateNeighbor(std::mt19937& rng, const TuningConfig& base,
                            const std::vector<TuningConfig>& space) {
    // Ein-Gen-Nachbarn, gruppiert nach Gen-Index.
    std::vector<const TuningConfig*> byGene[5];
    for (const auto& c : space) {
        const int g = singleGeneDiff(base, c);
        if (g >= 0) byGene[g].push_back(&c);
    }

    // Gene in zufaelliger Reihenfolge probieren, damit keines bevorzugt wird.
    int order[5] = {0, 1, 2, 3, 4};
    std::shuffle(order, order + 5, rng);

    auto absDiff = [](int x, int y) { return x > y ? x - y : y - x; };

    for (int g : order) {
        auto& cand = byGene[g];
        if (cand.empty()) continue;

        // Geordnete Gene: naechstgelegenen Wert waehlen (echter kleiner Schritt).
        if (g == 1 || g == 4) {
            const int baseVal = (g == 1) ? base.split_factor : base.unroll_factor;
            int bestDist = std::numeric_limits<int>::max();
            for (const auto* c : cand) {
                const int v = (g == 1) ? c->split_factor : c->unroll_factor;
                bestDist = std::min(bestDist, absDiff(v, baseVal));
            }
            std::vector<const TuningConfig*> nearest;
            for (const auto* c : cand) {
                const int v = (g == 1) ? c->split_factor : c->unroll_factor;
                if (absDiff(v, baseVal) == bestDist) nearest.push_back(c);
            }
            std::uniform_int_distribution<int> pick(0, (int)nearest.size() - 1);
            return *nearest[pick(rng)];
        }

        // Loop-Order: Einzel-Swaps bevorzugen, sonst irgendeine andere Order.
        if (g == 2) {
            std::vector<const TuningConfig*> swaps;
            for (const auto* c : cand)
                if (isSingleSwap(base.loop_order, c->loop_order)) swaps.push_back(c);
            const auto& pool = swaps.empty() ? cand : swaps;
            std::uniform_int_distribution<int> pick(0, (int)pool.size() - 1);
            return *pool[pick(rng)];
        }

        // Kategorische Gene (split_axis, parallel_axis): gleichverteilt.
        std::uniform_int_distribution<int> pick(0, (int)cand.size() - 1);
        return *cand[pick(rng)];
    }

    // Kein Ein-Gen-Nachbar im gefilterten Raum -> keine Bewegung.
    return base;
}

TuningConfig runSimulatedAnnealing(SearchContext& ctx,
                                   const std::vector<TuningConfig>& space) {
    const std::string name = strategyName(SearchStrategy::SIMULATED_ANNEALING);

    // Start: B3-Warmstart am Cost-Model-Optimum (space[0], da der Suchraum
    // kostensortiert ist), sonst zufaellig. Lokales Bergsteigen von einem guten
    // Startpunkt schlaegt breites Random-Sampling; von einem schlechten nicht.
    TuningConfig current;
    if (ctx.opts.warmStart && !space.empty()) {
        current = space[0];
    } else {
        std::uniform_int_distribution<int> pickSpace(0, (int)space.size() - 1);
        current = space[pickSpace(ctx.rng)];
    }
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

    if (coin(rng) < 0.5) {
        child1.split_axis = b.split_axis;
        child2.split_axis = a.split_axis;
    }
    if (coin(rng) < 0.5) {
        child1.split_factor = b.split_factor;
        child2.split_factor = a.split_factor;
    }
    if (coin(rng) < 0.5) {
        child1.loop_order = b.loop_order;
        child2.loop_order = a.loop_order;
    }
    if (coin(rng) < 0.5) {
        child1.parallel_axis = b.parallel_axis;
        child2.parallel_axis = a.parallel_axis;
    }
    return {child1, child2};
}

void mutate(std::mt19937& rng, TuningConfig& c,
            const std::vector<TuningConfig>& space) {
    std::uniform_int_distribution<int> pickGene(0, 3);
    std::uniform_int_distribution<int> pickSpace(0, (int)space.size() - 1);

    const TuningConfig& donor = space[pickSpace(rng)];
    switch (pickGene(rng)) {
        case 0: c.split_axis = donor.split_axis; break;
        case 1: c.split_factor = donor.split_factor; break;
        case 2: c.loop_order = donor.loop_order; break;
        case 3: c.parallel_axis = donor.parallel_axis; break;
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

// B2: Ersetzt ein bereits besuchtes Kind durch eine noch NICHT besuchte, strukturell
// valide Config, indem es wiederholt mutiert + repariert. Liefert true (und veraendert
// `child` in-place), sobald ein unbesuchtes gefunden wird; false nach maxAttempts
// (Suchraum praktisch erschoepft). Ohne diesen Ersatz wuerde die GA-while-Schleife
// bei konvergierter Population/kleinem Raum Crossover ohne Fortschritt wiederholen,
// bis das Zeit-Budget greift — reine Leerlaeufe. `remutations` zaehlt die Versuche.
static bool ensureUnvisited(std::mt19937& rng, TuningConfig& child,
                            const std::vector<TuningConfig>& space,
                            const SearchContext& ctx, int maxAttempts,
                            long& remutations) {
    int attempt = 0;
    while (ctx.alreadyVisited(child) && attempt < maxAttempts) {
        mutate(rng, child, space);
        child = repair(rng, child, space, ctx);
        ++remutations;
        ++attempt;
    }
    return !ctx.alreadyVisited(child);
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
        // B3: Warmstart — das obere Drittel der Startpopulation mit den besten
        // Cost-Model-Kandidaten impfen (space ist kostensortiert). Der Rest bleibt
        // zufaellig, um Diversitaet fuer Crossover zu erhalten.
        if (ctx.opts.warmStart) {
            const int seeded = std::max(1, popSize / 3);
            for (int i = 0; i < seeded && i < popSize && i < (int)space.size(); ++i)
                population[i].config = space[i];
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
        // B2: Abbruch-Zaehler gegen Endlosschleife. Wenn ueber viele Iterationen in
        // Folge KEIN neues Individuum entsteht (Suchraum erschoepft), ist die
        // Population nicht fuellbar -> Generation vorzeitig beenden statt bis zum
        // Zeit-Budget leerzulaufen.
        int stall = 0;
        const int STALL_LIMIT = std::max(8, popSize);
        while ((int)nextGen.size() < popSize) {
            if (checkStop(ctx)) break;

            const size_t before = nextGen.size();

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

                TuningConfig cfg = childCfg;
                if (ctx.alreadyVisited(cfg)) {
                    ctx.gaDuplicateChildren++;
                    // B2: statt Leerlauf gezielt zu einem unbesuchten Kind remutieren.
                    // gaRemutateTries=0 stellt das alte Ueberspringen wieder her (Ablation).
                    const bool ok = ctx.opts.gaRemutateTries > 0 &&
                                    ensureUnvisited(ctx.rng, cfg, space, ctx,
                                                    ctx.opts.gaRemutateTries, ctx.gaRemutations);
                    if (!ok) {
                        ctx.gaGaveUp++;
                        continue; // Suchraum erschoepft -> Stall-Zaehler greift unten
                    }
                    ctx.gaResolved++;
                }

                Individual child;
                child.config = cfg;
                child.fitness = ctx.evaluateTrial(cfg);
                child.evaluated = true;
                ctx.absorbTrial(child.config, child.fitness, name);
                nextGen.push_back(child);
            }

            if (nextGen.size() == before) {
                if (++stall >= STALL_LIMIT) {
                    ctx.gaStallBreaks++;
                    break; // Population nicht fuellbar (Raum erschoepft)
                }
            } else {
                stall = 0;
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
    auto fullSpace = generateSearchSpace(baseIr);

    std::vector<TuningConfig> searchSpace;
    if (!baseIr.einsum.empty() && opts.costModelFilterPct > 0.0 && opts.costModelFilterPct < 1.0) {
        std::vector<std::pair<double, int>> ranked;
        ranked.reserve(fullSpace.size());
        for (int i = 0; i < (int)fullSpace.size(); ++i)
            ranked.push_back({estimateCost(baseIr, fullSpace[i]), i});
        std::sort(ranked.begin(), ranked.end());

        // Strukturell absurde Kandidaten (Cost == 1e18, z.B. Parallelisierung
        // eines winzigen Workloads) werden KOMPLETT verworfen — nicht nur nach
        // hinten sortiert. Sonst leaken sie in die Top X%, wenn es zu wenige
        // gute Configs gibt (bei 6out: nur 3600 sequenzielle von 28800), und der
        // Random-Start landet auf einer katastrophalen parallelen Config.
        constexpr double ABSURD_COST = 1e17;
        std::vector<int> viable;
        viable.reserve(ranked.size());
        for (const auto& [cost, idx] : ranked) {
            if (cost >= ABSURD_COST) break; // ranked ist sortiert -> ab hier nur noch absurd
            viable.push_back(idx);
        }
        if (viable.empty()) { // Fallback: nichts als viabel eingestuft
            for (const auto& [cost, idx] : ranked) viable.push_back(idx);
        }
        const int absurdCount = (int)fullSpace.size() - (int)viable.size();

        int keepCount = std::max(1, (int)(viable.size() * opts.costModelFilterPct));
        searchSpace.reserve(keepCount);
        for (int i = 0; i < keepCount; ++i)
            searchSpace.push_back(fullSpace[viable[i]]);

        std::cout << "[AUTOTUNER] Suchraum generiert. " << fullSpace.size()
                  << " valide Kandidaten (nach Pruning).\n";
        if (absurdCount > 0)
            std::cout << "[COSTMODEL] " << absurdCount
                      << " strukturell absurde Configs verworfen "
                      << "(z.B. Parallelisierung winziger Workloads).\n";
        std::cout << "[COSTMODEL] Vorfilter: Top " << (opts.costModelFilterPct * 100.0)
                  << "% (" << keepCount << " von " << viable.size()
                  << " viablen) werden JIT-kompiliert.\n";
    } else {
        searchSpace = fullSpace;
        std::cout << "[AUTOTUNER] Suchraum generiert. " << searchSpace.size()
                  << " valide Kandidaten (nach Pruning).\n";
    }

    std::cout << "[AUTOTUNER] Strategie: " << strategyName(opts.strategy)
              << " | patience=" << opts.patience
              << ", minImprovement=" << (opts.minImprovementRel * 100.0) << "%"
              << ", timeBudget=" << (opts.timeBudgetMs / 1000.0) << "s";
    if (opts.maxTrials > 0) std::cout << ", maxTrials=" << opts.maxTrials;
    std::cout << "\n";
    std::cout << "[AUTOTUNER] Jeder Trial wird JIT-kompiliert, auf Korrektheit geprueft und real gebenchmarkt...\n\n";

    resetBenchmarkCacheStats();
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
