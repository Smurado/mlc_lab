#include "codegen.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

std::string generateSourceCode(const TEIR& ir) {
    std::stringstream ss;

    // Header inklusive ARM NEON Header generieren
    ss << "// Automatisch generiert durch den TEIR-Compiler Autotuner (NEON SIMD Edition)\n";
    ss << "#include <iostream>\n";
    ss << "#include <vector>\n";
    ss << "#include <omp.h>\n";
    ss << "#include <arm_neon.h> // Phase 6: Hardware-Intrinsics\n\n";

    // C-Linkage erzwingen, damit dlsym den Funktionsnamen im JIT-Schritt sauber findet
    ss << "extern \"C\" {\n";
    ss << "void teir_" << ir.name << "(\n";
    for (size_t i = 0; i < ir.tensors.size(); ++i) {
        ss << "    const float* __restrict__ " << ir.tensors[i].name;
        if (ir.tensors[i].name == "out") {
            ss.str(""); // Reset für saubere Pointer-Deklaration (out muss beschreibbar sein)
            ss << "// Automatisch generiert durch den TEIR-Compiler Autotuner (NEON SIMD Edition)\n#include <arm_neon.h>\nextern \"C\" {\nvoid teir_" << ir.name << "(const float* __restrict__ in0, const float* __restrict__ in1, float* __restrict__ out) {\n";
            break;
        }
    }

    int indent = 4;
    std::string indentStr(indent, ' ');

    // Primitiv: @zero Initialisierung
    ss << indentStr << "for (int i = 0; i < (96 * 32); ++i) out[i] = 0.0f;\n\n";

    // Schleifen-Generierung fuer die aeusseren Loops (r, t, p0)
    for (const auto& iter : ir.schedule) {
        if (iter.axis == "p1") continue; // Die innerste Achse wird voneinander isoliert und vektorisiert!

        int extent = 1;
        for (const auto& ax : ir.axes) {
            if (ax.name == iter.axis) { extent = ax.extent; break; }
        }

        std::string currentIndent(indent, ' ');
        if (iter.policy == Policy::Parallel) {
            ss << currentIndent << "#pragma omp parallel for\n";
        }
        ss << currentIndent << "for (int " << iter.axis << " = 0; " << iter.axis << " < " << extent << "; ++" << iter.axis << ") {\n";
        indent += 4;
    }

    std::string innerIndent(indent, ' ');
    ss << innerIndent << "// --- Phase 6: ARM64 NEON Vektorkern ---\n";
    ss << innerIndent << "float32x4_t v_accum = vdupq_n_f32(0.0f);\n\n";

    // Vektorschleife über p1 mit Stride 4 (da 4 x float32 in einem NEON-Register liegen)
    ss << innerIndent << "for (int p1 = 0; p1 < 64; p1 += 4) {\n";
    ss << innerIndent << "    // Kontinuierlicher Vektor-Load aus in0 (Row-Major Zugriff)\n";
    ss << innerIndent << "    float32x4_t v_in0 = vld1q_f32(&in0[r * 128 + p0 * 64 + p1]);\n\n";
    
    ss << innerIndent << "    // Strided-Load aus in1 (rekonstruierte p-Indizes)\n";
    ss << innerIndent << "    int base_p = p0 * 64 + p1;\n";
    ss << innerIndent << "    float32x4_t v_in1 = {\n";
    ss << innerIndent << "        in1[(base_p + 0) * 32 + t],\n";
    ss << innerIndent << "        in1[(base_p + 1) * 32 + t],\n";
    ss << innerIndent << "        in1[(base_p + 2) * 32 + t],\n";
    ss << innerIndent << "        in1[(base_p + 3) * 32 + t]\n";
    ss << innerIndent << "    };\n\n";

    ss << innerIndent << "    // Fused Multiply-Accumulate in Hardware\n";
    ss << innerIndent << "    v_accum = vmlaq_f32(v_accum, v_in0, v_in1);\n";
    ss << innerIndent << "}\n\n";

    ss << innerIndent << "// Horizontale Reduktion des NEON-Registers in das Ausgabetensor-Element\n";
    ss << innerIndent << "out[r * 32 + t] += vaddvq_f32(v_accum);\n";

    // Schleifen wieder schließen
    while (indent > 4) {
        indent -= 4;
        std::string closeIndent(indent, ' ');
        ss << closeIndent << "}\n";
    }

    ss << "}\n}\n"; // Schließe Funktion und extern "C"
    return ss.str();
}

void writeCodeToFile(const std::string& filename, const std::string& code) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) throw std::runtime_error("Fehler beim Schreiben: " + filename);
    outfile << code;
    outfile.close();
    std::cout << "[CODEGEN] NEON-optimierter C++ Code exportiert nach: " << filename << "\n";
}