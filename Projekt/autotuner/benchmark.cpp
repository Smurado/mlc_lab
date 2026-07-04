#include "benchmark.hpp"
#include "codegen.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <dlfcn.h>

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
    const int id = trialId++;

    const BenchmarkResult INVALID = { std::numeric_limits<double>::infinity(), 0.0 };

    // --- Problemdimensionen ableiten ---
    const int R = extentOf(ir, "r");
    const int T = extentOf(ir, "t");
    int P;
    if (extentOf(ir, "p", -1) != -1) P = extentOf(ir, "p");
    else P = extentOf(ir, "p0") * extentOf(ir, "p1");

    // --- 1. Kernel-Quelle generieren und schreiben (eindeutiger Name pro Trial) ---
    const std::string src = "_trial_" + std::to_string(id) + ".cpp";
    const std::string lib = "./_trial_" + std::to_string(id) + ".so";
    {
        std::ofstream f(src);
        f << generateSourceCode(ir);
    }

    // --- 2. JIT-Kompilierung mit echten Optimierungen ---
    // Der Compiler-Befehl wird dynamisch aus den Makefile-Parametern zusammengebaut (-D Macros).
    // Fuer SME-Backend: ersetze -march=native durch -march=armv9.2-a+sme
    std::string jitFlags = JIT_FLAGS;
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

    // --- 3. Dynamisch laden und Symbol binden ---
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

    // --- 4. Echte Tensor-Daten allokieren ---
    std::vector<float> in0(static_cast<size_t>(R) * P, 1.0f);
    std::vector<float> in1(static_cast<size_t>(P) * T, 2.0f);
    std::vector<float> out(static_cast<size_t>(R) * T, 0.0f);

    // --- 5. Warmup + Korrektheitspruefung ---
    // Erwartung: jedes out-Element = sum_p (1.0 * 2.0) = 2.0 * P
    kernel(in0.data(), in1.data(), out.data());
    const float expected = 2.0f * static_cast<float>(P);
    for (size_t i = 0; i < out.size(); ++i) {
        if (std::abs(out[i] - expected) > 1e-2f) {
            dlclose(handle);
            std::remove(src.c_str());
            std::remove(lib.c_str());
            return INVALID; // inkorrekte / racy Konfiguration verwerfen
        }
    }

    // --- 6. Zeitmessung: Best-of mehrerer Wiederholungen (stabiler gegen Jitter) ---
    // Per-Repeat-Cap verhindert, dass ein pathological Trial (z.B. extreme
    // Langsamkeit durch schlechte Parallelisierung) den gesamten Autotuner
    // blockiert. Sobald ein Repeat-Lauf laenger als REPEAT_CAP_MS braucht,
    // wird das Ergebnis direkt uebernommen und nicht weiter wiederholt.
    // ITERS ist hoeher gewaehlt, damit auch sehr schnelle SME-Kernels messbar sind.
    const int ITERS = 10000;
    const int REPEATS = 5;
    const double REPEAT_CAP_MS = 2000.0; // 2 s Cap pro Repeat-Block
    volatile double best_ms = std::numeric_limits<double>::infinity();
    for (int rep = 0; rep < REPEATS; ++rep) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) {
            kernel(in0.data(), in1.data(), out.data());
        }
        auto end = std::chrono::high_resolution_clock::now();
        const double blockMs =
            std::chrono::duration<double, std::milli>(end - start).count();
        const double ms = blockMs / ITERS;
        if (ms < best_ms) best_ms = ms;
        if (blockMs >= REPEAT_CAP_MS) break; // Trial zu langsam, nicht weiter messen
    }

    dlclose(handle);
    std::remove(src.c_str());
    std::remove(lib.c_str());

    const double flops = 2.0 * R * P * T;
    const double gflops = (best_ms > 0.0) ? (flops / 1e9) / (best_ms / 1000.0) : 0.0;
    return { best_ms, gflops };
}

void saveToCSV(const std::string& filename, const std::string& configName,
               const BenchmarkResult& res, const std::string& strategy) {
    std::ofstream file;
    // Prüfe, ob Datei bereits existiert, um Header zu schreiben
    std::ifstream check(filename);
    bool exists = check.good();
    check.close();

    file.open(filename, std::ios_base::app);
    if (!exists) {
        file << "strategy,config,runtime_ms,gflops\n";
    }
    file << strategy << "," << configName << "," << res.runtime_ms << "," << res.gflops << "\n";
    file.close();
}