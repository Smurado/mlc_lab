#include "benchmark.hpp"
#include "bench_loop.hpp"
#include "kernel_validation.hpp"
#include "tensor_guard.hpp"
#include "codegen.hpp"
#include "einsum.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <dlfcn.h>
#include <unordered_map>
#include <string>
#include <filesystem>

// A1 Kernel-Cache: Zaehler fuer Treffer/Fehltreffer.
static long g_benchmarkCacheHits = 0;
static long g_benchmarkCacheMisses = 0;
static long g_persistCacheHits = 0; // A5: Disk-Cache-Treffer (Compile+Bench gespart)
long benchmarkCacheHits() { return g_benchmarkCacheHits; }
long benchmarkCacheMisses() { return g_benchmarkCacheMisses; }
long benchmarkPersistHits() { return g_persistCacheHits; }
void resetBenchmarkCacheStats() {
    g_benchmarkCacheHits = 0; g_benchmarkCacheMisses = 0; g_persistCacheHits = 0;
}

// =====================================================================
// A5: Persistenter Cross-Run-Cache. Keyed auf die Kernel-Quelle (codeHash).
// Persistiert das Korrektheits-Verdikt (maschinenUNabhaengig -> immer vertrauens-
// wuerdig) und optional das Timing (maschinen-/lastabhaengig -> nur mit explizitem
// TEIR_CACHE_TRUST_TIMINGS vertraut, sonst neu vermessen). Nutzen: die Regressions-
// Suite startet 4 Prozesse/Kontraktion, die aktuell dieselben Kernel neu kompilieren.
// =====================================================================
namespace {
bool persistCacheEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("TEIR_PERSIST_CACHE");
        return e && (std::string(e) == "1" || std::string(e) == "true" || std::string(e) == "on");
    }();
    return on;
}
bool persistTrustTimings() {
    static const bool on = [] {
        const char* e = std::getenv("TEIR_CACHE_TRUST_TIMINGS");
        return e && (std::string(e) == "1" || std::string(e) == "true" || std::string(e) == "on");
    }();
    return on;
}
const std::string& persistCacheDir() {
    static const std::string dir = [] {
        const char* e = std::getenv("TEIR_CACHE_DIR");
        return std::string(e ? e : ".teir_cache");
    }();
    return dir;
}
enum class DiskVerdict { Miss, Invalid, Valid };

DiskVerdict diskLookup(size_t hash, BenchmarkResult& out) {
    const std::string path = persistCacheDir() + "/" + std::to_string(hash) + ".txt";
    std::ifstream f(path);
    if (!f.good()) return DiskVerdict::Miss;
    std::string verdict;
    if (!(f >> verdict)) return DiskVerdict::Miss;
    if (verdict == "INVALID") return DiskVerdict::Invalid;
    if (verdict == "VALID") {
        double ms = 0.0, g = 0.0;
        if (f >> ms >> g) { out = { ms, g }; return DiskVerdict::Valid; }
    }
    return DiskVerdict::Miss;
}

void diskStore(size_t hash, const BenchmarkResult& r) {
    std::error_code ec;
    std::filesystem::create_directories(persistCacheDir(), ec);
    const std::string path = persistCacheDir() + "/" + std::to_string(hash) + ".txt";
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp);
        if (!f.good()) return;
        if (std::isfinite(r.runtime_ms))
            f << "VALID " << r.runtime_ms << " " << r.gflops << "\n";
        else
            f << "INVALID\n";
    }
    std::rename(tmp.c_str(), path.c_str()); // atomar -> keine halben Dateien
}
} // namespace

// Liefert das Extent einer Achse anhand ihres Namens (oder fallback, falls nicht vorhanden)
static int extentOf(const TEIR& ir, const std::string& name, int fallback = 1) {
    for (const auto& ax : ir.axes) {
        if (ax.name == name) return ax.extent;
    }
    return fallback;
}

// Echtes Benchmarking: Fuer das uebergebene (transformierte) Schedule wird ein
// konkreter Kernel generiert, JIT-kompiliert, auf Korrektheit geprueft und auf
// realen Tensor-Daten zeitlich vermessen. Inkorrekte Konfigurationen (z.B. ein
// Race durch Parallelisierung der Reduktionsachse) werden mit runtime = +inf
// verworfen, damit der Autotuner sie nicht auswaehlt.
BenchmarkResult benchmark(const TEIR& ir) {
    static int trialId = 0;

    // --- A1: Kernel-Cache -------------------------------------------------
    // Semantisch identische Schedules erzeugen identischen Kernel-Code (z.B.
    // Unroll-Faktor groesser als das innere Extent, oder Split-Faktor 1). Solche
    // Kernel werden nur EINMAL JIT-kompiliert und vermessen; jede Wiederholung
    // liefert das gecachte Ergebnis in ~0ms. Das ist der groesste Durchsatz-Hebel,
    // weil SA/GA denselben Kernel oft mehrfach erzeugen.
    const std::string source = generateSourceCode(ir);
    const size_t codeHash = std::hash<std::string>{}(source);

    const BenchmarkResult INVALID = { std::numeric_limits<double>::infinity(), 0.0 };

    static std::unordered_map<size_t, BenchmarkResult> kernelCache;
    {
        auto it = kernelCache.find(codeHash);
        if (it != kernelCache.end()) {
            ++g_benchmarkCacheHits;
            return it->second;
        }
    }

    // A5: Persistenter Disk-Cache VOR dem JIT. INVALID-Verdikte werden immer
    // vertraut (maschinenunabhaengig) -> der Compile bekannter Race-Kernel entfaellt.
    // Gecachte Timings nur bei TEIR_CACHE_TRUST_TIMINGS (sonst neu vermessen).
    if (persistCacheEnabled()) {
        BenchmarkResult cached{};
        switch (diskLookup(codeHash, cached)) {
            case DiskVerdict::Invalid:
                ++g_persistCacheHits;
                kernelCache[codeHash] = INVALID;
                return INVALID;
            case DiskVerdict::Valid:
                if (persistTrustTimings()) {
                    ++g_persistCacheHits;
                    kernelCache[codeHash] = cached;
                    return cached;
                }
                break; // bekannt-valide, aber Timing neu vermessen
            case DiskVerdict::Miss:
                break;
        }
    }

    ++g_benchmarkCacheMisses;

    const int id = trialId++;

    const BenchmarkResult result = [&]() -> BenchmarkResult {

    const std::string src = "_trial_" + std::to_string(id) + ".cpp";
    const std::string lib = "./_trial_" + std::to_string(id) + ".so";
    {
        std::ofstream f(src);
        f << source;
    }

    std::string jitFlags = JIT_FLAGS;

    // A2: Optimierungslevel fuer Such-JIT konfigurierbar. In der Praxis ist der
    // Compile hier NICHT der Flaschenhals (Messung dominiert, s.u.), daher Default
    // -O3 (keine Auswahl-Verzerrung). Der Hook bleibt fuer sehr grosse Kernels, bei
    // denen -O3-Compile teuer wird: TEIR_SEARCH_OPT="-O2".
    {
        const char* envOpt = std::getenv("TEIR_SEARCH_OPT");
        const std::string level = envOpt ? std::string(envOpt) : "-O3";
        const size_t pos = jitFlags.find("-O3");
        if (pos != std::string::npos && !level.empty())
            jitFlags.replace(pos, 3, level);
    }

    if (ir.backend == Backend::SME) {
        const std::string native = "-march=native";
        const std::string sme = "-march=armv9.2-a+sme";
        size_t pos = jitFlags.find(native);
        if (pos != std::string::npos) {
            jitFlags.replace(pos, native.length(), sme);
        } else {
            jitFlags += " " + sme;
        }
    }
    const std::string cmd = std::string(JIT_CXX) + " " + jitFlags + " " + JIT_LDFLAGS + " " + src + " -o " + lib + " 2>/dev/null";
    if (std::system(cmd.c_str()) != 0) {
        std::remove(src.c_str());
        return INVALID;
    }

    void* handle = dlopen(lib.c_str(), RTLD_NOW);
    if (!handle) {
        std::remove(src.c_str());
        std::remove(lib.c_str());
        return INVALID;
    }
    typedef void (*kernel_t)(const float*, const float*, float*);
    auto kernel = reinterpret_cast<kernel_t>(dlsym(handle, ("teir_" + ir.name).c_str()));
    if (!kernel) {
        dlclose(handle);
        std::remove(src.c_str());
        std::remove(lib.c_str());
        return INVALID;
    }

    // --- Tensor-Groessen + FLOPs ---
    int in0_size, in1_size, out_size;
    double total_flops;
    int reduction_size = 0;
    int gemm_P = 0;

    if (ir.einsum.empty()) {
        const int R = extentOf(ir, "r");
        const int T = extentOf(ir, "t");
        if (extentOf(ir, "p", -1) != -1) gemm_P = extentOf(ir, "p");
        else gemm_P = extentOf(ir, "p0") * extentOf(ir, "p1");
        in0_size = R * gemm_P;
        in1_size = gemm_P * T;
        out_size = R * T;
        total_flops = 2.0 * R * gemm_P * T;
    } else {
        EinsumSpec spec = parseEinsum(ir.einsum);
        in0_size = tensorElements(spec.in0_idx, ir);
        in1_size = tensorElements(spec.in1_idx, ir);
        out_size = tensorElements(spec.out_idx, ir);
        total_flops = einsumFlops(ir);

        reduction_size = 1;
        for (char c : spec.reduce_axes) reduction_size *= extentOfChar(ir, c);

        // OOM-Schutz, Grenze siehe tensor_guard.hpp (TEIR_MAX_TENSOR).
        if (tensorTooLarge(in0_size, in1_size, out_size)) {
            dlclose(handle);
            std::remove(src.c_str());
            std::remove(lib.c_str());
            return INVALID;
        }
    }

    // --- Tensor-Daten allokieren + fuellen ---
    std::vector<float> in0(in0_size), in1(in1_size), out(out_size, 0.0f);

    if (ir.einsum.empty()) {
        // GEMM-Pfad: konstante Fuellung (eigene Konstanten-Pruefung unten).
        std::fill(in0.begin(), in0.end(), 1.0f);
        std::fill(in1.begin(), in1.end(), 2.0f);
    } else {
        fillEinsumInputs(in0, in1);   // siehe kernel_validation.hpp
    }

    // --- Warmup + Korrektheitspruefung ---
    kernel(in0.data(), in1.data(), out.data());

    auto cleanup_invalid = [&]() {
        dlclose(handle);
        std::remove(src.c_str());
        std::remove(lib.c_str());
    };

    if (ir.einsum.empty()) {
        const float expected = 2.0f * static_cast<float>(gemm_P);
        for (size_t i = 0; i < out.size(); ++i) {
            if (std::abs(out[i] - expected) > 1e-2f) {
                cleanup_invalid();
                return INVALID;
            }
        }
    } else {
        // Volle Referenz oder C2-Stichprobe — Entscheidung und Toleranz stehen in
        // kernel_validation.hpp, damit die Endmessung in main.cpp exakt dasselbe
        // prueft wie die Suche hier.
        if (!validateEinsumOutput(ir, in0.data(), in1.data(), out.data(), out_size)) {
            cleanup_invalid();
            return INVALID;
        }
        (void)reduction_size;
    }

    // --- Zeitmessung ------------------------------------------------------
    // C1: Statt einer FIXEN Iterationszahl wird bis zu einem Mindest-Messfenster
    // (BENCH_MIN_MS) gemessen. Sonst ist ein winziger Kernel (z.B. 6out mit 4374
    // FLOPs, ~0.1us/Aufruf) nach 2000 Iterationen erst 0.2ms gelaufen -> unter
    // Timer-/Loop-Rausch, GFLOPS instabil. Mit Mindestfenster wird jeder Kernel
    // lang genug wiederholt, um eine stabile Zahl zu liefern. Obergrenze bleibt
    // BENCH_CAP_MS (Durchsatz). A2': zum RANKING reicht ein kurzes Fenster; der
    // finale Gewinner wird in main.cpp separat und praeziser neu vermessen.
    static const int BENCH_REPEATS = [] {
        const char* e = std::getenv("TEIR_BENCH_REPEATS");
        return e ? std::max(1, std::atoi(e)) : 3;
    }();
    static const double BENCH_MIN_MS = [] {
        const char* e = std::getenv("TEIR_BENCH_MIN_MS");
        return e ? std::max(0.1, std::atof(e)) : 20.0;
    }();
    static const double BENCH_CAP_MS = [] {
        const char* e = std::getenv("TEIR_BENCH_CAP_MS");
        return e ? std::max(1.0, std::atof(e)) : 300.0;
    }();
    // A6: Adaptive Blockgroesse. Der Cap kann nur ZWISCHEN zwei Bloecken greifen,
    // also kostet ein fixer Block von 64 bei langsamen Kernels ein Vielfaches des
    // Caps (64 x 633 ms = 40 s statt 300 ms) -- und zwar dreimal, wegen
    // BENCH_REPEATS. Genau das hat die GETT-Matrix auf 1-2 Trials pro Fall
    // gedrosselt. Der Block existiert nur, um die Timer-Ablesung (~25 ns) zu
    // amortisieren; bei einem 633-ms-Kernel ist dafuer EIN Aufruf genug.
    // Adaptiv wird die Blockgroesse aus der gemessenen Kernel-Zeit bestimmt.
    // Default = 0 = altes Verhalten (fixer Block), damit alte Zahlen
    // reproduzierbar bleiben.
    static const bool BENCH_ADAPTIVE = [] {
        const char* e = std::getenv("TEIR_BENCH_ADAPTIVE");
        return e && std::atoi(e) != 0;
    }();

    // Zeit-Cap/Messfenster werden INKREMENTELL (in Bloecken) geprueft, nicht erst
    // nach einer festen Iterationszahl. Jeder rep laeuft mindestens BENCH_MIN_MS
    // (Stabilitaet) und hoechstens BENCH_CAP_MS (Durchsatz). Schleife + Block-
    // groesse stehen in bench_loop.hpp -- gemeinsam mit der Endmessung in
    // main.cpp und dort per Unit-Test abgedeckt (make test).
    BlockPolicy pol;
    pol.adaptive    = BENCH_ADAPTIVE;
    pol.minWindowMs = BENCH_MIN_MS;
    pol.capMs       = BENCH_CAP_MS;

    volatile double best_ms = std::numeric_limits<double>::infinity();
    for (int rep = 0; rep < BENCH_REPEATS; ++rep) {
        const auto start = std::chrono::high_resolution_clock::now();
        const BlockResult r = runBlockLoop(
            pol,
            [&](long n) {
                for (long k = 0; k < n; ++k)
                    kernel(in0.data(), in1.data(), out.data());
            },
            [&] {
                return std::chrono::duration<double, std::milli>(
                           std::chrono::high_resolution_clock::now() - start).count();
            });
        if (r.perCallMs() < best_ms) best_ms = r.perCallMs();
        if (r.elapsedMs >= BENCH_CAP_MS) break;
    }

    dlclose(handle);
    std::remove(src.c_str());
    std::remove(lib.c_str());

    const double gflops = (best_ms > 0.0) ? (total_flops / 1e9) / (best_ms / 1000.0) : 0.0;
    return { best_ms, gflops };
    }();

    // A5: Verdikt (und Timing) fuer Cross-Run-Wiederverwendung persistieren.
    if (persistCacheEnabled()) diskStore(codeHash, result);

    kernelCache[codeHash] = result;
    return result;
}

void saveToCSV(const std::string& filename, const std::string& configName,
               const BenchmarkResult& res, const std::string& strategy,
               double costEstimateMs) {
    std::ofstream file;
    // Prüfe, ob Datei bereits existiert, um Header zu schreiben
    std::ifstream check(filename);
    bool exists = check.good();
    check.close();

    file.open(filename, std::ios_base::app);
    if (!exists) {
        file << "strategy,config,runtime_ms,gflops,cost_estimate_ms\n";
    }
    file << strategy << "," << configName << "," << res.runtime_ms << ","
         << res.gflops << "," << costEstimateMs << "\n";
    file.close();
}

void appendTrialLog(const std::string& filename, const std::string& strategy,
                    unsigned seed, int trialIndex, double trialGflops,
                    double bestGflops) {
    std::ifstream check(filename);
    const bool exists = check.good();
    check.close();

    std::ofstream file(filename, std::ios_base::app);
    if (!exists) {
        file << "strategy,seed,trial,gflops,best_gflops\n";
    }
    file << strategy << "," << seed << "," << trialIndex << ","
         << trialGflops << "," << bestGflops << "\n";
    file.close();
}