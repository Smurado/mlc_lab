#include "teir.hpp"
#include "parser.hpp"
#include "autotuner.hpp"
#include "passes.hpp"
#include "codegen.hpp"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include <dlfcn.h>

int main()
{
    try
    {
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
        // Autotuning
        //------------------------------------------------------

        TuningConfig best = runAutotuner(ir, opts);

        std::cout << "\n[INFO] Wende das gefundene Optimum auf die IR an...\n";

        TEIR bestIr = ir;

        splitOuterAxis(bestIr, "p", best.split_factor);
        reorderSchedule(bestIr, best.loop_order);

        for (auto& it : bestIr.schedule)
            it.policy = Policy::Sequential;

        if (!best.parallel_axis.empty())
            makeParallel(bestIr, best.parallel_axis);

        bestIr.unrollFactor = best.unroll_factor;
        bestIr.backend = opts.backend;

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

        const int P = extentOf("p");
        const int R = extentOf("r");
        const int T = extentOf("t");

        std::vector<float> in0(R * P, 1.0f);
        std::vector<float> in1(P * T, 2.0f);
        std::vector<float> out(R * T, 0.0f);

        kernel(in0.data(), in1.data(), out.data());

        float expected = 2.0f * P;

        bool ok = true;

        for (float v : out)
        {
            if (std::abs(v - expected) > 1e-4f)
            {
                ok = false;
                break;
            }
        }

        if (ok)
        {
            std::cout << "[SUCCESS] Validation passed.\n";
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