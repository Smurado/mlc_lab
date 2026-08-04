#include "include/teir.h"
#include "include/teir_parser.h"
#include "include/teir_evaluator.h"
#include "include/teir_optimizer.h"

#include <chrono>
#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <utility>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <random>

size_t get_tensor_size(const TEIRProgram& prog, const std::string& tname) {
    size_t max_offset = 0;
    for (auto const& [ax_name, ax] : prog.axes) {
        auto it = ax.strides.find(tname);
        if (it != ax.strides.end()) {
            max_offset += (size_t)(ax.extent - 1) * it->second;
        }
    }
    return max_offset + 4; 
}

void init_tensor_random(void* ptr, size_t bytes) {
    float* arr = static_cast<float*>(ptr);
    size_t elements = bytes / sizeof(float);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < elements; ++i) {
        arr[i] = dis(gen);
    }
}

bool verify_tensors(void* base_ptr, void* opt_ptr, size_t bytes) {
    float* base = static_cast<float*>(base_ptr);
    float* opt = static_cast<float*>(opt_ptr);
    size_t elements = bytes / sizeof(float);
    for (size_t i = 0; i < elements; ++i) {
        if (std::abs(base[i] - opt[i]) > 1e-3) {
            std::cout << "[ERROR] Mismatch at index " << i << ": Base=" << base[i] << " Opt=" << opt[i] << "\n";
            return false;
        }
    }
    return true;
}

// Rechenoperationen des Programms.
//
// Eine Contraction fuehrt je (M, N, K)-Kombination eine Multiplikation und eine
// Addition aus, also 2 Flop. Die aeusseren Schleifenachsen vervielfachen das.
// Zero und Copy zaehlen nicht als Rechenoperationen -- fuer reine
// Datenbewegung waere GB/s die passende Kennzahl, die wird separat gemeldet.
double count_flops(const TEIRProgram& prog) {
    double flops = 0.0;
    for (auto const& [pname, prim] : prog.primitives) {
        if (prim.kind != "Contraction") continue;

        // Achsen, die das Primitive selbst abarbeitet.
        double inner = 1.0;
        std::vector<std::string> prim_axes;
        for (auto const& role : {"M", "N", "K"}) {
            auto it = prim.axes_map.find(role);
            if (it == prim.axes_map.end()) continue;
            for (auto const& a : it->second) {
                const std::string ax = (a[0] == '@') ? a.substr(1) : a;
                prim_axes.push_back(ax);
                auto x = prog.axes.find(ax);
                if (x != prog.axes.end()) inner *= x->second.extent;
            }
        }

        // Alle uebrigen Achsen sind Schleifen um das Primitive herum.
        double outer = 1.0;
        for (auto const& [name, axis] : prog.axes)
            if (std::find(prim_axes.begin(), prim_axes.end(), name) == prim_axes.end())
                outer *= axis.extent;

        flops += 2.0 * inner * outer;
    }
    return flops;
}

// Bewegte Bytes fuer reine Datenbewegung (Copy): einmal lesen, einmal schreiben.
double count_bytes_moved(const TEIRProgram& prog) {
    double bytes = 0.0;
    for (auto const& [pname, prim] : prog.primitives) {
        if (prim.kind != "Copy") continue;
        double elems = 1.0;
        for (auto const& [name, axis] : prog.axes) elems *= axis.extent;
        bytes += elems * 4.0 * 2.0;
    }
    return bytes;
}

// Durchsatz passend zur Art des Programms: GFLOPS fuer Rechnung, GB/s fuer
// reine Datenbewegung.
std::string throughput(const TEIRProgram& prog, double ms) {
    char buf[64];
    const double flops = count_flops(prog);
    if (flops > 0.0) {
        std::snprintf(buf, sizeof buf, "%.1f GFLOPS", flops / (ms * 1e6));
        return buf;
    }
    const double bytes = count_bytes_moved(prog);
    if (bytes > 0.0) {
        std::snprintf(buf, sizeof buf, "%.2f GB/s", bytes / (ms * 1e6));
        return buf;
    }
    return "--";
}

void run_ablation_study(const std::string& filepath, const std::string& model_name) {
    std::cout << "\n========================================\n";
    std::cout << "Model: " << model_name << " (" << filepath << ")\n";
    std::cout << "========================================\n";

    TEIRProgram prog_base;
    try {
        prog_base = load_teir(filepath);
    } catch (...) {
        std::cout << "[WARN] File not found or parsing failed.\n";
        return;
    }

    std::vector<void*> base_args, opt_args;
    size_t out_idx = -1;
    for (size_t i = 0; i < prog_base.tensors.size(); ++i) {
        const auto& tname = prog_base.tensors[i];
        if (tname == "out") out_idx = i;
        size_t t_bytes = get_tensor_size(prog_base, tname);
        void* b_ptr = std::malloc(t_bytes);
        void* o_ptr = std::malloc(t_bytes);
        
        if (tname != "out") {
            init_tensor_random(b_ptr, t_bytes);
            std::memcpy(o_ptr, b_ptr, t_bytes);
        } else {
            std::memset(b_ptr, 0, t_bytes);
            std::memset(o_ptr, 0, t_bytes);
        }
        base_args.push_back(b_ptr);
        opt_args.push_back(o_ptr);
    }

    std::cout << ">> Stage 0: Baseline (Unoptimized)\n";
    TEIREvaluator baseline_eval(prog_base);
    auto t0 = std::chrono::high_resolution_clock::now();
    baseline_eval.evaluate(base_args.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double base_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    // Sichtbar machen, welcher Kernel je Primitive tatsaechlich lief. Ohne das
    // sieht man der Zeit nicht an, ob SME gegriffen hat oder zurueckgefallen wurde.
    // NACH der Zeitnahme, damit die Ausgabe die Messung nicht verfaelscht.
    for (const auto& line : baseline_eval.plan()) std::cout << "   " << line << "\n";
    std::cout << "   Time: " << base_ms << " ms  |  " << throughput(prog_base, base_ms) << "\n";

    for (int stage = 1; stage <= 3; ++stage) {
        std::cout << "\n>> Stage " << stage << ": ";
        if (stage == 1) std::cout << "Parallel Only\n";
        if (stage == 2) std::cout << "Cache Blocking Only\n";
        if (stage == 3) std::cout << "Combined (Parallel + Blocking)\n";

        TEIRProgram prog_opt = load_teir(filepath);
        TEIROptimizer opt(prog_opt);

        if (stage == 1 || stage == 3) {
            opt.expose_parallelism();
        }
        if (stage == 2 || stage == 3) {
            // Generisch: Kachelkante aus L1-Groesse und ZA-Kachel abgeleitet, die
            // Achsen aus den Strides. Kein Bezug mehr auf model_name.
            opt.apply_cache_blocking();
        }

        std::memset(opt_args[out_idx], 0, get_tensor_size(prog_base, "out"));

        TEIREvaluator optimized_eval(prog_opt);
        auto t2 = std::chrono::high_resolution_clock::now();
        optimized_eval.evaluate(opt_args.data());
        auto t3 = std::chrono::high_resolution_clock::now();
        double opt_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        
        std::cout << "   Time: " << opt_ms << " ms  |  " << throughput(prog_opt, opt_ms)
                  << "  |  Speedup vs Base: " << (base_ms / opt_ms) << "x\n";

        bool correct = verify_tensors(base_args[out_idx], opt_args[out_idx], get_tensor_size(prog_base, "out"));
        std::cout << "   Verification: " << (correct ? "PASSED" : "FAILED") << "\n";
    }

    for (size_t i = 0; i < base_args.size(); ++i) {
        std::free(base_args[i]);
        std::free(opt_args[i]);
    }
}

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "[Week 8] TEIR Ablation Study (In-Memory AST)\n";
    std::cout << "========================================\n";

    // Ohne Argument laufen alle Modelle. Mit Argument nur die genannten --
    // einsum allein braucht ueber acht Minuten, das macht jede Fehlersuche
    // an einem der anderen Modelle unnoetig zaeh.
    const std::vector<std::pair<std::string, std::string>> models = {
        {"matmul",        "data/matmul.teir"},
        {"contraction",   "data/contraction.teir"},
        {"einsum",        "data/einsum.teir"},
        {"transposition", "data/transposition.teir"},
    };

    std::vector<std::string> wanted(argv + 1, argv + argc);
    for (auto const& m : models) {
        if (!wanted.empty() &&
            std::find(wanted.begin(), wanted.end(), m.first) == wanted.end()) continue;
        run_ablation_study(m.second, m.first);
    }

    return 0;
}
