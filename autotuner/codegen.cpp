#include "codegen.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

std::string generateSourceCode(const TEIR& ir) {
    std::stringstream ss;

    // Header-Inkludierungen generieren
    ss << "// Automatisch generiert durch den TEIR-Compiler Autotuner\n";
    ss << "#include <iostream>\n";
    ss << "#include <vector>\n";
    ss << "#include <omp.h>\n\n";

    // Funktionssignatur basierend auf den Tensoren bauen
    ss << "void teir_" << ir.name << "(\n";
    for (size_t i = 0; i < ir.tensors.size(); ++i) {
        ss << "    float* " << ir.tensors[i].name;
        if (i < ir.tensors.size() - 1) ss << ",\n";
    }
    ss << "\n) {\n";

    // Schleifen-Verschachtelung (Loop Nest) auf Basis des Schedules generieren
    int indent = 4;
    for (const auto& iter : ir.schedule) {
        // Finde das zugehörige Extent der Achse
        int extent = 1;
        for (const auto& ax : ir.axes) {
            if (ax.name == iter.axis) {
                extent = ax.extent;
                break;
            }
        }

        std::string indentStr(indent, ' ');

        // Phase 5: OpenMP Parallelisierung einfügen, falls die Policy es verlangt
        if (iter.policy == Policy::Parallel) {
            ss << indentStr << "#pragma omp parallel for\n";
        }

        // Phase 3: Eigentliche Schleife generieren
        ss << indentStr << "for (int " << iter.axis << " = 0; " << iter.axis << " < " << extent << "; ++" << iter.axis << ") {\n";
        indent += 4;
    }

    // Innersten Kernel (Primitives) simulieren / einfügen
    std::string innerIndent(indent, ' ');
    ss << innerIndent << "// Invoke Primitives Kernels\n";
    for (const auto& inv : ir.invokes) {
        ss << innerIndent << "// execute_primitive_" << inv << "();\n";
    }
    
    // Einfache Dummy-Rechenoperation im Inneren, damit der Code valide rechnet
    ss << innerIndent << "out[0] += in0[0] * in1[0]; // Platzhalter fuer Tensor Contraction\n";

    // Schleifenklammern wieder schließen
    while (indent > 4) {
        indent -= 4;
        std::string closeIndent(indent, ' ');
        ss << closeIndent << "}\n";
    }

    ss << "}\n";
    return ss.str();
}

void writeCodeToFile(const std::string& filename, const std::string& code) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Fehler beim Schreiben der Code-Datei: " + filename);
    }
    outfile << code;
    outfile.close();
    std::cout << "[CODEGEN] C++ Quellcode erfolgreich exportiert nach: " << filename << "\n";
}