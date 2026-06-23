#include "teir.hpp"
#include "parser.hpp"
#include "autotuner.hpp"
#include "passes.hpp"
#include "codegen.hpp"
#include <iostream>

int main() {
    try {
        std::cout << "==================================================\n";
        std::cout << "   TEIR Autotuner - Schritt 4: Code Generation\n";
        std::cout << "==================================================\n\n";

        // 1. Basis-IR einlesen
        std::cout << "[INFO] Lese 'input.teir' ein...\n";
        TEIR irBase = parseTEIR("input.teir");
        std::cout << "[SUCCESS] Basis-IR erfolgreich geladen.\n\n";

        // 2. Autotuning-Suche ausfuehren (findet das Optimum)
        runAutotuner(irBase);

        // 3. Generierung des besten Codes (Wir wenden das gefundene Optimum manuell an)
        std::cout << "\n[INFO] Generiere C++ Quellcode fuer die beste Konfiguration...\n";
        TEIR bestIr = irBase;
        
        // Wir wenden hier die Werte an, die dein Autotuner als Optimum ermittelt hat
        splitOuterAxis(bestIr, "p", 64);
        reorderSchedule(bestIr, {"r", "t", "p0", "p1"});
        makeParallel(bestIr, "r");

        // Code generieren
        std::string generatedCode = generateSourceCode(bestIr);
        
        // In Datei schreiben
        writeCodeToFile("generated_kernel.cpp", generatedCode);

        std::cout << "\n[SUCCESS] Der Compiler-Autotuner-Pipelinezyklus ist abgeschlossen!\n";

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Kritischer Fehler: " << e.what() << "\n";
        return 1;
    }
    return 0;
}