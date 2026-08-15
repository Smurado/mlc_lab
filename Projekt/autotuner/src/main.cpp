// Ablaufsteuerung des Autotuners: CSV laden, Optionen aus der Umgebung lesen,
// Suche starten, das Optimum anwenden, den finalen Kernel per JIT bauen und
// gegen die Referenz validieren. Die [PERFORMANCE]-Zeile am Ende ist die
// Endmessung, auf der alle Auswertungen beruhen.
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
#include <dlfcn.h>

// Numerische Env-Variablen mit verstaendlicher Fehlermeldung parsen. std::stod
// alleine wirft bei TEIR_MAX_TRIALS=abc nur "stoi: no conversion" -- daraus geht
// nicht hervor, welche Variable falsch gesetzt war.
static double envToDouble(const char* name, const char* value)
{
    try
    {
        size_t used = 0;
        double d = std::stod(value, &used);
        if (used != std::string(value).size())
            throw std::invalid_argument("");
        return d;
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(std::string(name) + ": ungueltiger Wert '" +
                                 value + "' (Zahl erwartet)");
    }
}

static int envToInt(const char* name, const char* value)
{
    double d = envToDouble(name, value);
    if (d != (int)d)
        throw std::runtime_error(std::string(name) + ": ungueltiger Wert '" +
                                 value + "' (Ganzzahl erwartet)");
    return (int)d;
}

static void printHelp()
{
    std::cout <<
        "TEIR-Autotuner: tunt die Kontraktion aus der Eingabe-CSV per JIT-Suche.\n"
        "Aufruf: teir_compiler [--help]\n"
        "Konfiguration ueber Umgebungsvariablen (Default in Klammern):\n"
        "\n"
        "Eingabe und Ablauf\n"
        "  TEIR_INPUT            Pfad zur Eingabe-CSV (input.csv)\n"
        "  TEIR_BACKEND          Codegen-Backend: scalar | neon | sme (scalar)\n"
        "  TEIR_NO_AUTOTUNE      1 = Default-Schedule nur messen, keine Suche (aus)\n"
        "\n"
        "Suche\n"
        "  TEIR_STRATEGY         sa | ga | random (sa)\n"
        "  TEIR_MAX_TRIALS       max. Anzahl JIT-Trials, 0 = kein Limit (0)\n"
        "  TEIR_TIME_BUDGET_MS   hartes Zeitlimit der Suche in ms (60000)\n"
        "  TEIR_SEED             Seed fuer reproduzierbare Laeufe (42)\n"
        "  TEIR_WARMSTART        0 = Suche startet zufaellig statt am\n"
        "                        Cost-Modell-Optimum (an)\n"
        "  TEIR_COST_FILTER      Anteil der Kandidaten, der nach Cost-Modell\n"
        "                        gemessen wird; ausserhalb (0,1) = Filter aus (0.3)\n"
        "  TEIR_GA_REMUTATE_TRIES  GA: Mutationsversuche gegen Duplikate (3)\n"
        "  TEIR_SEARCH_OPT       Compiler-Flags der Such-JITs, z. B. -O2 (-O3)\n"
        "  TEIR_CALIBRATE        0 = Cost-Modell-Kalibrierung ueberspringen (an)\n"
        "\n"
        "Messung und Validierung\n"
        "  TEIR_BENCH_ADAPTIVE   1 = adaptive Blockgroesse der Messschleife (aus)\n"
        "  TEIR_BENCH_REPEATS    Wiederholungen der Such-Messung (3)\n"
        "  TEIR_BENCH_MIN_MS     Mindest-Messfenster in ms (50)\n"
        "  TEIR_BENCH_CAP_MS     Zeitdeckel je Messung in ms (1000)\n"
        "  TEIR_MAX_TENSOR       OOM-Grenze in Tensor-Elementen (1e8)\n"
        "  TEIR_VALIDATE_SAMPLES Stichprobengroesse der Validierung (64)\n"
        "\n"
        "Kernel-Cache und Logs\n"
        "  TEIR_PERSIST_CACHE    1 = uebersetzte Kernel prozessuebergreifend cachen (aus)\n"
        "  TEIR_CACHE_DIR        Ablageort des Caches (.teir_cache)\n"
        "  TEIR_CACHE_TRUST_TIMINGS  1 = gecachten Timings vertrauen statt\n"
        "                        neu zu messen (aus)\n"
        "  TEIR_TRIAL_LOG        Pfad fuer das Trial-Log der Konvergenzanalyse (aus)\n";
}

int main(int argc, char* argv[])
{
    if (argc > 1)
    {
        const std::string arg = argv[1];
        if (arg == "--help" || arg == "-h")
        {
            printHelp();
            return 0;
        }
        std::cerr << "[ERROR] Unbekanntes Argument '" << arg
                  << "'. Konfiguration laeuft ueber Umgebungsvariablen, siehe --help.\n";
        return 1;
    }

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
            else
                std::cout << "[WARN] TEIR_STRATEGY='" << s << "' unbekannt "
                          << "(gueltig: sa, ga, random) -- nutze Default sa.\n";
        }

        if (const char* envBudget = std::getenv("TEIR_TIME_BUDGET_MS"))
        {
            opts.timeBudgetMs = envToDouble("TEIR_TIME_BUDGET_MS", envBudget);
            if (opts.timeBudgetMs <= 0)
                throw std::runtime_error("TEIR_TIME_BUDGET_MS muss positiv sein");
        }
        if (const char* envMaxTrials = std::getenv("TEIR_MAX_TRIALS"))
        {
            opts.maxTrials = envToInt("TEIR_MAX_TRIALS", envMaxTrials);
            if (opts.maxTrials < 0)
                throw std::runtime_error("TEIR_MAX_TRIALS darf nicht negativ sein (0 = kein Limit)");
        }
        if (const char* envSeed = std::getenv("TEIR_SEED"))
        {
            opts.seed = static_cast<unsigned>(envToInt("TEIR_SEED", envSeed));
        }
        if (const char* envWarm = std::getenv("TEIR_WARMSTART"))
        {
            std::string w(envWarm);
            opts.warmStart = !(w == "0" || w == "false" || w == "off");
        }
        if (const char* envCostFilter = std::getenv("TEIR_COST_FILTER"))
        {
            opts.costModelFilterPct = envToDouble("TEIR_COST_FILTER", envCostFilter);
            // Werte ausserhalb (0,1) schalten den Vorfilter ab (autotuner.cpp
            // prueft 0 < pct < 1). Absichtlich erlaubt, aber sichtbar machen.
            if (opts.costModelFilterPct <= 0 || opts.costModelFilterPct >= 1)
                std::cout << "[WARN] TEIR_COST_FILTER=" << opts.costModelFilterPct
                          << " liegt ausserhalb (0,1) -- Vorfilter ist damit aus.\n";
        }
        if (const char* envGaRemut = std::getenv("TEIR_GA_REMUTATE_TRIES"))
        {
            opts.gaRemutateTries = envToInt("TEIR_GA_REMUTATE_TRIES", envGaRemut);
        }
        // Backend-Auswahl: scalar (Default), neon oder sme
        if (const char* envBackend = std::getenv("TEIR_BACKEND"))
        {
            std::string b(envBackend);

            if (b == "sme" || b == "SME")
                opts.backend = Backend::SME;
            else if (b == "neon" || b == "NEON")
                opts.backend = Backend::NEON;
            else if (b != "scalar" && b != "SCALAR")
                std::cout << "[WARN] TEIR_BACKEND='" << b << "' unbekannt "
                          << "(gueltig: scalar, neon, sme) -- nutze Default scalar.\n";
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
            dlclose(handle);
            return 1; // fuer Skripte: fehlgeschlagene Validierung != Erfolg
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