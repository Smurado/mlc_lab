#include "codegen.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

// Liefert das Extent einer Achse anhand ihres Namens (oder fallback, falls nicht vorhanden)
static int extentOf(const TEIR& ir, const std::string& name, int fallback = 1) {
    for (const auto& ax : ir.axes) {
        if (ax.name == name) return ax.extent;
    }
    return fallback;
}

// Generiert korrekten, parametrischen C++-Code fuer die Kontraktion
//   out[r, t] = sum_p in0[r, p] * in1[p, t]
// Schleifenreihenfolge, Tiling (Split von p) und Parallelisierungsachse stammen
// direkt aus der (vom Autotuner) transformierten IR.
std::string generateSourceCode(const TEIR& ir) {
    std::stringstream ss;

    // --- Problemdimensionen aus der IR ableiten ---
    const int R = extentOf(ir, "r");
    const int T = extentOf(ir, "t");

    bool split = false;
    int F = 1;   // Extent der inneren Tile-Achse p1 (= Split-Faktor)
    int P;       // Volles Extent der Reduktionsachse p
    if (extentOf(ir, "p", -1) != -1) {
        P = extentOf(ir, "p");
    } else {
        // p wurde in p0 (aussen) und p1 (innen) gesplittet
        F = extentOf(ir, "p1");
        P = extentOf(ir, "p0") * extentOf(ir, "p1");
        split = true;
    }

    ss << "// Auto-generiert vom TEIR-Autotuner (parametrisch, echtes Benchmarking)\n";
    ss << "#include <omp.h>\n\n";
    ss << "extern \"C\" {\n";
    ss << "void teir_" << ir.name
       << "(const float* __restrict__ in0, const float* __restrict__ in1, float* __restrict__ out) {\n";

    // @zero: Ausgabe initialisieren
    ss << "    for (int i = 0; i < " << (R * T) << "; ++i) out[i] = 0.0f;\n\n";

    // --- Schleifennest exakt in Schedule-Reihenfolge erzeugen ---
    int indent = 4;
    for (const auto& iter : ir.schedule) {
        const std::string pad(indent, ' ');
        const int extent = extentOf(ir, iter.axis);
        if (iter.policy == Policy::Parallel) {
            ss << pad << "#pragma omp parallel for\n";
        }
        ss << pad << "for (int " << iter.axis << " = 0; " << iter.axis << " < "
           << extent << "; ++" << iter.axis << ") {\n";
        indent += 4;
    }

    // --- Innerster Kern: MAC-Operation ---
    const std::string pad(indent, ' ');
    if (split) {
        // Globalen p-Index aus den Tile-Achsen rekonstruieren: p = p0 * F + p1
        ss << pad << "int p = p0 * " << F << " + p1;\n";
    }
    ss << pad << "out[r * " << T << " + t] += in0[r * " << P << " + p] * in1[p * " << T << " + t];\n";

    // --- Schleifen schliessen ---
    while (indent > 4) {
        indent -= 4;
        const std::string cpad(indent, ' ');
        ss << cpad << "}\n";
    }

    ss << "}\n}\n"; // Funktion + extern "C" schliessen
    return ss.str();
}

void writeCodeToFile(const std::string& filename, const std::string& code) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) throw std::runtime_error("Fehler beim Schreiben: " + filename);
    outfile << code;
    outfile.close();
    std::cout << "[CODEGEN] Parametrischer C++-Code exportiert nach: " << filename << "\n";
}