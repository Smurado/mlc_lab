#include "teir.hpp"
#include "parser.hpp"
#include "autotuner.hpp"
#include "passes.hpp"
#include "codegen.hpp"
#include "einsum.hpp"
#include "cost_model.hpp"
#include "bench_loop.hpp"
#include "kernel_validation.hpp"
#include "tensor_guard.hpp"

#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <dlfcn.h>

int main()
{
    try
    {
        // stdout sofort flushen (kein Block-Buffering bei Umleitung in Datei/Pipe),
        // damit externe Tools den Fortschritt ([NEW BEST] etc.) LIVE mitlesen koennen.
        std::cout << std::unitbuf;

        std::cout << "======================================================\n";
        std::cout << "   TEIR Autotuner - Echtes JIT-Benchmarking\n";
        std::cout << "======================================================\n\n";

        //----------------------------------------------------------
        // IRs laden
        //----------------------------------------------------------

        const char* envInput = std::getenv("TEIR_INPUT");
        const std::string inputFile = envInput ? envInput : "input.csv";

        std::cout << "[INFO] Lade IRs aus CSV: " << inputFile << "\n";

        TEIR ir = parseCSV(inputFile);

        //----------------------------------------------------------
        // Alle IRs bearbeiten
        //----------------------------------------------------------

        std::cout << "\n======================================================\n";
        std::cout << "Running IR: " << ir.name << "\n";
        std::cout << "======================================================\n";

        //------------------------------------------------------
        // Autotuner konfigurieren
        //------------------------------------------------------

        AutotunerOptions opts;

        if (const char* envStrategy = std::getenv("TEIR_STRATEGY"))
        {
            std::string s(envStrategy);

            if (s == "random" || s == "RANDOM")
                opts.strategy = SearchStrategy::RANDOM_SEARCH;
            else if (s == "sa" || s == "SA" || s == "annealing")
                opts.strategy = SearchStrategy::SIMULATED_ANNEALING;
            else if (s == "ga" || s == "GA" || s == "genetic")
                opts.strategy = SearchStrategy::GENETIC;
        }

        if (const char* envBudget = std::getenv("TEIR_TIME_BUDGET_MS"))
        {
            opts.timeBudgetMs = std::stod(envBudget);
        }
        if (const char* envMaxTrials = std::getenv("TEIR_MAX_TRIALS"))
        {
            opts.maxTrials = std::stoi(envMaxTrials);
        }
        if (const char* envSeed = std::getenv("TEIR_SEED"))
        {
            opts.seed = static_cast<unsigned>(std::stoul(envSeed));
        }
        if (const char* envWarm = std::getenv("TEIR_WARMSTART"))
        {
            std::string w(envWarm);
            opts.warmStart = !(w == "0" || w == "false" || w == "off");
        }
        if (const char* envCostFilter = std::getenv("TEIR_COST_FILTER"))
        {
            opts.costModelFilterPct = std::stod(envCostFilter);
        }
        if (const char* envGaRemut = std::getenv("TEIR_GA_REMUTATE_TRIES"))
        {
            opts.gaRemutateTries = std::stoi(envGaRemut);
        }
        // Backend-Auswahl: scalar (Default), neon oder sme
        if (const char* envBackend = std::getenv("TEIR_BACKEND"))
        {
            std::string b(envBackend);

            if (b == "sme" || b == "SME")
                opts.backend = Backend::SME;
            else if (b == "neon" || b == "NEON")
                opts.backend = Backend::NEON;
        }

        //------------------------------------------------------
        // Autotuning (oder Default-Schedule benchmarken)
        //------------------------------------------------------

        TEIR bestIr = ir;
        bestIr.backend = opts.backend;

        const char* envNoAutotune = std::getenv("TEIR_NO_AUTOTUNE");

        if (envNoAutotune && (std::string(envNoAutotune) == "1" || std::string(envNoAutotune) == "true"))
        {
            std::cout << "[INFO] TEIR_NO_AUTOTUNE=1 — benchmarke Default-Schedule ohne Autotuning.\n";
        }
        else
        {
            // C3: CostModel-Parameter (peak_gflops, Thread-Overhead) einmalig auf
            // dieser Maschine messen, statt sie hart zu kodieren. Beeinflusst nur
            // den CostModel-Vorfilter/Warmstart, nicht die realen Messungen.
            calibrateCostModel();

            TuningConfig best = runAutotuner(ir, opts);

            std::cout << "\n[INFO] Wende das gefundene Optimum auf die IR an...\n";

            if (best.split_factor > 1 && !best.split_axis.empty())
                splitOuterAxis(bestIr, best.split_axis, best.split_factor);
            reorderSchedule(bestIr, best.loop_order);

            for (auto& it : bestIr.schedule)
                it.policy = Policy::Sequential;

            if (!best.parallel_axis.empty())
                makeParallel(bestIr, best.parallel_axis);

            bestIr.unrollFactor = best.unroll_factor;
        }

        //------------------------------------------------------
        // Code generieren
        //------------------------------------------------------

        std::string kernelCode = generateSourceCode(bestIr);
        writeCodeToFile("generated_kernel.cpp", kernelCode);

        //------------------------------------------------------
        // JIT
        //------------------------------------------------------

        std::cout << "\n[JIT] Kompiliere Kernel...\n";

        std::string cmd =
            std::string(JIT_CXX) + " " +
            JIT_FLAGS + " " +
            JIT_LDFLAGS +
            " generated_kernel.cpp -o generated_kernel.so";

        if (std::system(cmd.c_str()) != 0)
            throw std::runtime_error("JIT compilation failed.");

        void* handle = dlopen("./generated_kernel.so", RTLD_NOW);

        if (!handle)
            throw std::runtime_error(dlerror());

        //------------------------------------------------------
        // Symbol laden
        //------------------------------------------------------

        std::string symbolName = "teir_" + ir.name;

        typedef void (*kernel_func_t)(
            const float*,
            const float*,
            float*);

        kernel_func_t kernel =
            reinterpret_cast<kernel_func_t>(
                dlsym(handle, symbolName.c_str()));

        if (const char* err = dlerror())
        {
            dlclose(handle);
            throw std::runtime_error(err);
        }

        std::cout << "[JIT SUCCESS] Symbol "
                    << symbolName
                    << " geladen.\n";

        //------------------------------------------------------
        // Validierung
        //------------------------------------------------------

        auto extentOf =
            [&](const std::string& name, int fallback = 1)
        {
            for (const auto& ax : ir.axes)
                if (ax.name == name)
                    return ax.extent;

            return fallback;
        };

        bool ok = true;

        if (ir.einsum.empty())
        {
            const int P = extentOf("p");
            const int R = extentOf("r");
            const int T = extentOf("t");

            std::vector<float> in0(R * P, 1.0f);
            std::vector<float> in1(P * T, 2.0f);
            std::vector<float> out(R * T, 0.0f);

            kernel(in0.data(), in1.data(), out.data());

            float expected = 2.0f * P;

            for (float v : out)
            {
                if (std::abs(v - expected) > 1e-4f)
                {
                    ok = false;
                    break;
                }
            }
        }
        else
        {
            EinsumSpec spec = parseEinsum(ir.einsum);
            int in0_size = tensorElements(spec.in0_idx, ir);
            int in1_size = tensorElements(spec.in1_idx, ir);
            int out_size = tensorElements(spec.out_idx, ir);

            // OOM-Schutz, Grenze siehe tensor_guard.hpp (TEIR_MAX_TENSOR).
            // Bewusst identisch mit der Pruefung in benchmark.cpp: sonst laeuft die
            // Suche durch, aber die Endmessung bricht hier ab -> "partial" ohne
            // [PERFORMANCE]-Zeile.
            if (tensorTooLarge(in0_size, in1_size, out_size))
            {
                std::cout << "[SKIP] Tensor zu gross fuer Validierung ("
                          << in0_size << " / " << in1_size << " / " << out_size
                          << " Elemente, Grenze " << maxTensorElements()
                          << " -- anhebbar via TEIR_MAX_TENSOR).\n";
                dlclose(handle);
                return 0;
            }

            std::vector<float> in0(in0_size), in1(in1_size), out(out_size, 0.0f);

            fillEinsumInputs(in0, in1);   // siehe kernel_validation.hpp

            kernel(in0.data(), in1.data(), out.data());

            // Exakt dieselbe Pruefung wie in der Suche (benchmark.cpp) -- volle
            // Referenz oder C2-Stichprobe, Entscheidung und Toleranz zentral in
            // kernel_validation.hpp.
            ok = validateEinsumOutput(ir, in0.data(), in1.data(), out.data(), out_size);
        }

        if (ok)
        {
            std::cout << "[SUCCESS] Validation passed.\n";

            // Timing-Messung (auch fuer TEIR_NO_AUTOTUNE, damit der Default-
            // Schedule vergleichbar ist)
            double total_flops;
            if (ir.einsum.empty())
            {
                const int P2 = extentOf("p");
                const int R2 = extentOf("r");
                const int T2 = extentOf("t");
                total_flops = 2.0 * R2 * P2 * T2;
            }
            else
            {
                total_flops = einsumFlops(ir);
            }

            if (total_flops > 0 && ok)
            {
                // Eigene Daten fuer Timing (unabhaengig vom Validierungs-Scope)
                int t_in0, t_in1, t_out;
                if (ir.einsum.empty())
                {
                    t_in0 = extentOf("r") * extentOf("p");
                    t_in1 = extentOf("p") * extentOf("t");
                    t_out = extentOf("r") * extentOf("t");
                }
                else
                {
                    EinsumSpec spec = parseEinsum(ir.einsum);
                    t_in0 = tensorElements(spec.in0_idx, ir);
                    t_in1 = tensorElements(spec.in1_idx, ir);
                    t_out = tensorElements(spec.out_idx, ir);
                }

                std::vector<float> t_a(t_in0, 1.0f);
                std::vector<float> t_b(t_in1, 2.0f);
                std::vector<float> t_c(t_out, 0.0f);

                // C1: Bis zu einem stabilen Messfenster wiederholen statt fixer
                // Iterationszahl. Ein winziger Kernel wuerde sonst nach 1000
                // Aufrufen unter Timer-/Loop-Rausch gemessen -> instabile GFLOPS
                // ueber Seeds. Mindestfenster = Stabilitaet, Obergrenze = Zeitschutz.
                // A6: Diese Schleife liefert die [PERFORMANCE]-Zahl, also den
                // Wert, der in allen Auswertungen landet -- und sie ist teurer
                // als die Such-Messung: 5 Wiederholungen, und der Deckel bricht
                // nur die innere Schleife ab. Bei einem 763-ms-Kernel sind das
                // 5 x 64 x 763 ms = 244 s fuer EINE Endmessung. Adaptiv bleiben
                // es 5 Wiederholungen (Statistik unveraendert), aber mit je
                // einem Aufruf statt 64. Schleife: bench_loop.hpp (make test).
                BlockPolicy pol;
                pol.adaptive = [] {
                    const char* e = std::getenv("TEIR_BENCH_ADAPTIVE");
                    return e && std::atoi(e) != 0;
                }();
                pol.minWindowMs = 50.0;
                pol.capMs       = 1000.0;

                volatile double best_ms = 1e18;
                long usedIters = 0;
                for (int rep = 0; rep < 5; ++rep)
                {
                    const auto t0 = std::chrono::high_resolution_clock::now();
                    const BlockResult r = runBlockLoop(
                        pol,
                        [&](long n) {
                            for (long k = 0; k < n; ++k)
                                kernel(t_a.data(), t_b.data(), t_c.data());
                        },
                        [&] {
                            return std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - t0).count();
                        });
                    if (r.perCallMs() < best_ms) best_ms = r.perCallMs();
                    usedIters = r.calls;
                }
                double gflops = (total_flops / 1e9) / (best_ms / 1000.0);
                std::cout << "[PERFORMANCE] " << best_ms << " ms ("
                          << gflops << " GFLOPS)\n";
                // C1: Mikro-Workloads klar kennzeichnen, statt Rausch-GFLOPS als
                // belastbare Zahl auszugeben. Schwelle: unter ~1e5 FLOPs ist die
                // reine Rechenzeit selbst bei Spitzenleistung im Sub-Mikrosekunden-
                // Bereich -> nicht sinnvoll tunebar, Zahl nur bedingt aussagekraeftig.
                if (total_flops < 1e5)
                    std::cout << "[WARN] Mikro-Workload (" << (long)total_flops
                              << " FLOPs): unter Messgrenze — GFLOPS nur bedingt "
                              << "aussagekraeftig (gemessen ueber " << usedIters
                              << " Iterationen/Block).\n";
            }
        }
        else
        {
            std::cout << "[FAILED] Validation failed.\n";
        }

        dlclose(handle);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[ERROR] " << e.what() << '\n';
        return 1;
    }

    return 0;
}