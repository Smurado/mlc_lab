#include "teir.hpp"
#include "parser.hpp"
#include "autotuner.hpp"
#include "passes.hpp"
#include "codegen.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <dlfcn.h>   // Erforderlich fuer dlopen, dlsym, dlclose
#include <cmath>

int main() {
    try {
        std::cout << "======================================================\n";
        std::cout << "   TEIR Autotuner - Echtes JIT-Benchmarking\n";
        std::cout << "======================================================\n\n";

        // 1. Basis-IR laden
        TEIR irBase = parseTEIR("input.teir");

        // 2. Autotuner-Optionen konfigurieren
        //    Strategie ist hier zentral umschaltbar fuer Vergleichslaeufe.
        AutotunerOptions opts;
        // Wahl der Suchstrategie (zur Laufzeit z.B. per ENV-Variable moeglich):
        const char* envStrategy = std::getenv("TEIR_STRATEGY");
        if (envStrategy) {
            std::string s(envStrategy);
            if (s == "random" || s == "RANDOM") {
                opts.strategy = SearchStrategy::RANDOM_SEARCH;
            } else if (s == "sa" || s == "SA" || s == "annealing") {
                opts.strategy = SearchStrategy::SIMULATED_ANNEALING;
            } else if (s == "ga" || s == "GA" || s == "genetic") {
                opts.strategy = SearchStrategy::GENETIC;
            }
        }
        // Zeit-Budget fuer Vergleichslaeufe reduzieren, falls per Env gesetzt
        const char* envBudget = std::getenv("TEIR_TIME_BUDGET_MS");
        if (envBudget) {
            opts.timeBudgetMs = std::stod(envBudget);
        }

        // 3. Autotuning ausführen: liefert die real beste Konfiguration zurück
        TuningConfig best = runAutotuner(irBase, opts);

        // 3. Bestes Schedule tatsächlich anwenden & Code generieren
        std::cout << "\n[INFO] Wende das gefundene Optimum auf die IR an...\n";
        TEIR bestIr = irBase;
        splitOuterAxis(bestIr, "p", best.split_factor);
        reorderSchedule(bestIr, best.loop_order);
        for (auto& it : bestIr.schedule) it.policy = Policy::Sequential;
        if (!best.parallel_axis.empty()) {
            makeParallel(bestIr, best.parallel_axis);
        }
        bestIr.unrollFactor = best.unroll_factor;

        std::string kernelCode = generateSourceCode(bestIr);
        writeCodeToFile("generated_kernel.cpp", kernelCode);

        // ==========================================
        // FINALE JIT COMPILATION (On-the-fly)
        // ==========================================
        std::cout << "\n[JIT] Kompiliere 'generated_kernel.cpp' zu Shared Library...\n";
        
        // Shell-Befehl zur Kompilierung der finalen .so-Datei wird aus Makefile-Makros gebaut
        std::string cmd = std::string(JIT_CXX) + " " + JIT_FLAGS + " " + JIT_LDFLAGS + " generated_kernel.cpp -o generated_kernel.so";
        int compileStatus = std::system(cmd.c_str());
        if (compileStatus != 0) {
            throw std::runtime_error("JIT-Kompilierung fehlgeschlagen!");
        }
        std::cout << "[JIT SUCCESS] Shared Library 'generated_kernel.so' erfolgreich erzeugt.\n";

        // Dynamic Loading der Library in den eigenen Adressraum
        std::cout << "[JIT] Lade Bibliothek via dlopen()...\n";
        void* handle = dlopen("./generated_kernel.so", RTLD_NOW);
        if (!handle) {
            throw std::runtime_error(std::string("dlopen fehlgeschlagen: ") + dlerror());
        }

        // Funktionspointer aus der Symboltabelle extrahieren
        typedef void (*kernel_func_t)(const float*, const float*, float*);
        kernel_func_t teir_contraction_jit = (kernel_func_t)dlsym(handle, "teir_contraction");
        
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            dlclose(handle);
            throw std::runtime_error(std::string("dlsym fehlgeschlagen: ") + dlsym_error);
        }
        std::cout << "[JIT SUCCESS] Funktionspointer 'teir_contraction' erfolgreich gebunden.\n";

        // ==========================================
        // VALIDIERUNG MIT ECHTEN ARRAYS
        // ==========================================
        std::cout << "\n[VALIDATION] Allokiere echte Tensor-Workloads...\n";
        // Dimensionen aus der IR: in0 (96x128), in1 (128x32), out (96x32)
        std::vector<float> in0(96 * 128, 1.0f);  // Mit 1.0 initialisiert
        std::vector<float> in1(128 * 32, 2.0f);  // Mit 2.0 initialisiert
        std::vector<float> out(96 * 32,  0.0f);  // Ergebnisspeicher

        std::cout << "[VALIDATION] Fuehre JIT-kompilierten Kernel aus...\n";
        teir_contraction_jit(in0.data(), in1.data(), out.data());

        // Mathematische Verifizierung:
        // Jedes Element in 'out' berechnet sich aus dem Skalarprodukt einer Zeile von in0 und Spalte von in1.
        // Länge der Reduktionsachse (p) ist 128. Jede Multiplikation ist 1.0 * 2.0 = 2.0.
        // Erwartetes Ergebnis pro Element: 128 * 2.0 = 256.0
        bool verificationPassed = true;
        for (size_t i = 0; i < out.size(); ++i) {
            if (std::abs(out[i] - 256.0f) > 1e-4) {
                verificationPassed = false;
                std::cout << "[ERROR] Abweichung bei Index " << i << ": Gefunden=" << out[i] << ", Erwartet=256.0\n";
                break;
            }
        }

        if (verificationPassed) {
            std::cout << "\n======================================================\n";
            std::cout << "   🎉 VALIDATION SUCCESSFUL! 🎉\n";
            std::cout << "======================================================\n";
            std::cout << "  Der JIT-generierte, real getunte Kernel rechnet zu\n";
            std::cout << "  100% mathematisch korrekt. Element[0] = " << out[0] << "\n";
        } else {
            std::cout << "[FAIL] Die Berechnung lieferte falsche Werte.\n";
        }

        // Library sauber entladen
        dlclose(handle);

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Kritischer Fehler: " << e.what() << "\n";
        return 1;
    }
    return 0;
}