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

// ============================================================================
// SME-Backend: Generiert C++ mit Inline-ASM fuer 32x32 GEMM-Bloecke
// out[R,T] += in0[R,P] * in1[P,T]
//
// Da in0 row-major ist, sind A-Spalten (festes k) nicht zusammenhaengend.
// Daher transponieren wir in0 in ein temporaeres Buffer [P,R] (A_t[p*R+r]),
// sodass A_t-Spalten (festes k, aufeinanderfolgende r) zusammenhaengend sind
// und mit ld1w geladen werden koennen (wie in week3-4 wo A column-major war).
// in1 ist row-major: B-Zeilen (festes k, aufeinanderfolgende t) sind bereits
// zusammenhaengend.
// ============================================================================
static std::string generateSMEKernel(const TEIR& ir) {
    std::stringstream ss;

    const int R = extentOf(ir, "r");
    const int T = extentOf(ir, "t");
    const int P = extentOf(ir, "p", -1) != -1 ? extentOf(ir, "p")
                 : extentOf(ir, "p0") * extentOf(ir, "p1");

    const int mBlocks = (R + 31) / 32;
    const int nBlocks = (T + 31) / 32;

    ss << "// Auto-generiert vom TEIR-Autotuner (SME-Backend)\n";
    ss << "// GEMM: out[" << R << "," << T << "] += in0[" << R << "," << P
       << "] * in1[" << P << "," << T << "]\n";
    ss << "#include <arm_neon.h>\n";
    ss << "#include <cstdlib>\n\n";
    ss << "extern \"C\" {\n";
    ss << "__attribute__((target(\"sme\")))\n";
    ss << "void teir_" << ir.name
       << "(const float* __restrict__ in0, const float* __restrict__ in1, float* __restrict__ out) {\n";

    // Ausgabe initialisieren
    ss << "    for (int i = 0; i < " << (R * T) << "; ++i) out[i] = 0.0f;\n\n";

    // A transponieren: A_t[p*R + r] = in0[r*P + p]
    ss << "    // Transponiere in0[R,P] -> A_t[P,R] fuer zusammenhaengende ld1w-Loads\n";
    ss << "    float* A_t = (float*)malloc(" << (P * R) << " * sizeof(float));\n";
    ss << "    for (int r = 0; r < " << R << "; ++r)\n";
    ss << "        for (int p = 0; p < " << P << "; ++p)\n";
    ss << "            A_t[p * " << R << " + r] = in0[r * " << P << " + p];\n\n";

    // smstart einmal aussen (Performance: kein wiederholter Mode-Wechsel)
    ss << "    asm volatile(\"smstart\");\n";
    ss << "    asm volatile(\"ptrue p0.s\");\n\n";

    // M/N-Loops in C++ (wie gemm_512_512_512 aus week3-4)
    ss << "    for (int mb = 0; mb < " << mBlocks << "; ++mb) {\n";
    ss << "        for (int nb = 0; nb < " << nBlocks << "; ++nb) {\n";
    ss << "            const int m = mb * 32;\n";
    ss << "            const int n = nb * 32;\n\n";

    // ZA-Tile nullen
    ss << "            asm volatile(\"zero {za}\");\n\n";

    // K-Loop ueber P: pro k ein Rank-1-Update via 4x fmopa
    // A-Vektor: A_t[k*R + m..m+31] (zusammenhaengend, 2 Z-Register)
    // B-Vektor: in1[k*T + n..n+31] (zusammenhaengend, 2 Z-Register)
    ss << "            for (int k = 0; k < " << P << "; ++k) {\n";
    ss << "                const float* a_ptr = A_t + k * " << R << " + m;\n";
    ss << "                const float* b_ptr = in1 + k * " << T << " + n;\n\n";

    // Laden: z0 = A_t[k, m..m+15], z1 = A_t[k, m+16..m+31]
    //        z2 = in1[k, n..n+15], z3 = in1[k, n+16..n+31]
    ss << "                asm volatile(\"ld1w {z0.s}, p0/z, [%0]\\n\\t\"\n";
    ss << "                             \"ld1w {z1.s}, p0/z, [%1]\\n\\t\"\n";
    ss << "                             \"ld1w {z2.s}, p0/z, [%2]\\n\\t\"\n";
    ss << "                             \"ld1w {z3.s}, p0/z, [%3]\\n\\t\"\n";
    ss << "                             : : \"r\"(a_ptr), \"r\"(a_ptr + 16),\n";
    ss << "                                 \"r\"(b_ptr), \"r\"(b_ptr + 16)\n";
    ss << "                             : \"z0\", \"z1\", \"z2\", \"z3\", \"memory\");\n\n";

    // 4x fmopa (Outer Product): za0/za1/za2/za3 = 16x16 Sub-Tiles des 32x32 ZA
    ss << "                asm volatile(\n";
    ss << "                    \"fmopa za0.s, p0/m, p0/m, z0.s, z2.s\\n\\t\"\n";
    ss << "                    \"fmopa za1.s, p0/m, p0/m, z1.s, z2.s\\n\\t\"\n";
    ss << "                    \"fmopa za2.s, p0/m, p0/m, z0.s, z3.s\\n\\t\"\n";
    ss << "                    \"fmopa za3.s, p0/m, p0/m, z1.s, z3.s\\n\\t\"\n";
    ss << "                    : : : \"memory\");\n";
    ss << "            }\n\n"; // Ende K-Loop

    // Store: ZA-Tile zurueck in out (mit Akkumulation via fadd)
    // mova braucht den Row-Index in w12-w15, daher feste Register wie in week3-4.
    ss << "            // ZA-Tile in out zurueckschreiben (out += za)\n";
    ss << "            for (int row = 0; row < 16; ++row) {\n";
    ss << "                float* out_ptr0 = out + (m + row) * " << T << " + n;\n";
    ss << "                float* out_ptr1 = out_ptr0 + 16;\n";
    ss << "                register int row_reg asm(\"w12\") = row;\n";
    ss << "                asm volatile(\n";
    ss << "                    \"mova z4.s, p0/m, za0v.s[w12, 0]\\n\\t\"\n";
    ss << "                    \"ld1w {z6.s}, p0/z, [%[o0]]\\n\\t\"\n";
    ss << "                    \"fadd z6.s, z6.s, z4.s\\n\\t\"\n";
    ss << "                    \"st1w {z6.s}, p0, [%[o0]]\\n\\t\"\n";
    ss << "                    \"mova z4.s, p0/m, za2v.s[w12, 0]\\n\\t\"\n";
    ss << "                    \"ld1w {z6.s}, p0/z, [%[o1]]\\n\\t\"\n";
    ss << "                    \"fadd z6.s, z6.s, z4.s\\n\\t\"\n";
    ss << "                    \"st1w {z6.s}, p0, [%[o1]]\\n\\t\"\n";
    ss << "                    : \n";
    ss << "                    : [o0] \"r\"(out_ptr0), [o1] \"r\"(out_ptr1)\n";
    ss << "                    : \"z4\", \"z6\", \"w12\", \"memory\");\n";
    ss << "            }\n";
    // Zweite Haelfte (Zeilen 16..31): za1/za3
    ss << "            for (int row = 0; row < 16; ++row) {\n";
    ss << "                float* out_ptr0 = out + (m + 16 + row) * " << T << " + n;\n";
    ss << "                float* out_ptr1 = out_ptr0 + 16;\n";
    ss << "                register int row_reg asm(\"w12\") = row;\n";
    ss << "                asm volatile(\n";
    ss << "                    \"mova z4.s, p0/m, za1v.s[w12, 0]\\n\\t\"\n";
    ss << "                    \"ld1w {z6.s}, p0/z, [%[o0]]\\n\\t\"\n";
    ss << "                    \"fadd z6.s, z6.s, z4.s\\n\\t\"\n";
    ss << "                    \"st1w {z6.s}, p0, [%[o0]]\\n\\t\"\n";
    ss << "                    \"mova z4.s, p0/m, za3v.s[w12, 0]\\n\\t\"\n";
    ss << "                    \"ld1w {z6.s}, p0/z, [%[o1]]\\n\\t\"\n";
    ss << "                    \"fadd z6.s, z6.s, z4.s\\n\\t\"\n";
    ss << "                    \"st1w {z6.s}, p0, [%[o1]]\\n\\t\"\n";
    ss << "                    : \n";
    ss << "                    : [o0] \"r\"(out_ptr0), [o1] \"r\"(out_ptr1)\n";
    ss << "                    : \"z4\", \"z6\", \"w12\", \"memory\");\n";
    ss << "            }\n";
    ss << "        }\n"; // Ende N-Loop
    ss << "    }\n\n"; // Ende M-Loop

    ss << "    asm volatile(\"smstop\");\n";
    ss << "    free(A_t);\n";
    ss << "}\n}\n";
    return ss.str();
}

// ============================================================================
// Scalar-Backend: parametrisches Schleifennest mit OpenMP + Unroll
// ============================================================================

// Generiert korrekten, parametrischen C++-Code fuer die Kontraktion
//   out[r, t] = sum_p in0[r, p] * in1[p, t]
// Schleifenreihenfolge, Tiling (Split von p) und Parallelisierungsachse stammen
// direkt aus der (vom Autotuner) transformierten IR.
static std::string generateScalarKernel(const TEIR& ir) {
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

    ss << "// Auto-generiert vom TEIR-Autotuner (Scalar-Backend, parametrisch)\n";
    ss << "#include <omp.h>\n\n";
    ss << "extern \"C\" {\n";
    ss << "void teir_" << ir.name
       << "(const float* __restrict__ in0, const float* __restrict__ in1, float* __restrict__ out) {\n";

    // @zero: Ausgabe initialisieren
    ss << "    for (int i = 0; i < " << (R * T) << "; ++i) out[i] = 0.0f;\n\n";

    // --- Schleifennest exakt in Schedule-Reihenfolge erzeugen ---
    int indent = 4;
    const int nIters = static_cast<int>(ir.schedule.size());
    for (int i = 0; i < nIters; ++i) {
        const auto& iter = ir.schedule[i];
        const std::string pad(indent, ' ');
        const int extent = extentOf(ir, iter.axis);
        if (iter.policy == Policy::Parallel) {
            ss << pad << "#pragma omp parallel for\n";
        }
        // Unroll-Pragma nur fuer die innerste Schleife, falls factor > 1
        if (i == nIters - 1 && ir.unrollFactor > 1) {
            ss << pad << "#pragma GCC unroll(" << ir.unrollFactor << ")\n";
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

std::string generateSourceCode(const TEIR& ir) {
    if (ir.backend == Backend::SME) {
        return generateSMEKernel(ir);
    }
    return generateScalarKernel(ir);
}

void writeCodeToFile(const std::string& filename, const std::string& code) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) throw std::runtime_error("Fehler beim Schreiben: " + filename);
    outfile << code;
    outfile.close();
    std::cout << "[CODEGEN] Parametrischer C++-Code exportiert nach: " << filename << "\n";
}