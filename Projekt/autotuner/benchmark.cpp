#include "benchmark.hpp"
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

    const std::string src = "_trial_" + std::to_string(id) + ".cpp";
    const std::string lib = "./_trial_" + std::to_string(id) + ".so";
    {
        std::ofstream f(src);
        f << generateSourceCode(ir);
    }

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
    bool use_reference = false;
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

        double total_iters = total_flops / 2.0;
        use_reference = (total_iters <= 100000000.0);

        reduction_size = 1;
        for (char c : spec.reduce_axes) reduction_size *= extentOfChar(ir, c);

        const int MAX_TENSOR = 100000000;
        if (in0_size > MAX_TENSOR || in1_size > MAX_TENSOR || out_size > MAX_TENSOR) {
            dlclose(handle);
            std::remove(src.c_str());
            std::remove(lib.c_str());
            return INVALID;
        }
    }

    // --- Tensor-Daten allokieren + fuellen ---
    std::vector<float> in0(in0_size), in1(in1_size), out(out_size, 0.0f);

    if (use_reference) {
        for (int i = 0; i < in0_size; ++i) in0[i] = (float)((i % 13) + 1) / 13.0f;
        for (int i = 0; i < in1_size; ++i) in1[i] = (float)((i % 7) + 1) / 7.0f;
    } else {
        std::fill(in0.begin(), in0.end(), 1.0f);
        std::fill(in1.begin(), in1.end(), 2.0f);
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
    } else if (use_reference) {
        std::vector<float> ref(out_size, 0.0f);
        referenceEinsum(ir, in0.data(), in1.data(), ref.data());
        for (int i = 0; i < out_size; ++i) {
            float tol = 1e-2f * std::max(1.0f, std::abs(ref[i]));
            if (std::abs(out[i] - ref[i]) > tol) {
                cleanup_invalid();
                return INVALID;
            }
        }
    } else {
        float expected = 2.0f * static_cast<float>(reduction_size);
        float tol = 1e-1f * std::max(1.0f, std::abs(expected));
        for (int i = 0; i < out_size; ++i) {
            if (std::abs(out[i] - expected) > tol) {
                cleanup_invalid();
                return INVALID;
            }
        }
    }

    // --- Zeitmessung ---
    int ITERS;
    if (ir.einsum.empty()) {
        ITERS = 10000;
    } else {
        double est_ms = total_flops / 1e9;
        ITERS = std::max(1, std::min(10000, (int)(200.0 / est_ms)));
    }

    const int REPEATS = 5;
    const double REPEAT_CAP_MS = 2000.0;
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
        if (blockMs >= REPEAT_CAP_MS) break;
    }

    dlclose(handle);
    std::remove(src.c_str());
    std::remove(lib.c_str());

    const double gflops = (best_ms > 0.0) ? (total_flops / 1e9) / (best_ms / 1000.0) : 0.0;
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