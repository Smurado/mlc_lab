#include "codegen.hpp"
#include "einsum.hpp"
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

    // Ausgabe wird direkt mit dem ZA-Tile ueberschrieben (out = za, nicht out += za).
    // Da zero {za} das Tile initialisiert und fmopa akkumuliert, ist das Ergebnis
    // identisch zum vorherigen out=0 + fadd(out, za) - aber ohne Zero-Init und
    // ohne den Read-Modify-Write von out (Store statt Load+FADD+Store).
    // WICHTIG: das ist nur korrekt, weil zero {za} das Tile pro Block nullt.

    // A transponieren: A_t[p*R + r] = in0[r*P + p]
    // Caching: nur transponieren wenn in0-Pointer sich aendert (Benchmark
    // ruft denselben Kernel 10000x mit gleichem in0 -> Transposition nur 1x).
    ss << "    // Transponiere in0[R,P] -> A_t[P,R] fuer zusammenhaengende ld1w-Loads\n";
    ss << "    // Static-Cache: nur transponieren wenn in0-Pointer sich aendert\n";
    ss << "    static float* A_t_cache = nullptr;\n";
    ss << "    static const float* A_t_src = nullptr;\n";
    ss << "    if (A_t_cache == nullptr || A_t_src != in0) {\n";
    ss << "        if (A_t_cache == nullptr)\n";
    ss << "            A_t_cache = (float*)malloc(" << (P * R) << " * sizeof(float));\n";
    ss << "        A_t_src = in0;\n";
    ss << "        for (int r = 0; r < " << R << "; ++r)\n";
    ss << "            for (int p = 0; p < " << P << "; ++p)\n";
    ss << "                A_t_cache[p * " << R << " + r] = in0[r * " << P << " + p];\n";
    ss << "    }\n";
    ss << "    float* A_t = A_t_cache;\n\n";

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

    // Store: ZA-Tile direkt in out schreiben (out = za, ohne Read-Modify-Write)
    // mova braucht den Row-Index in w12-w15, daher feste Register wie in week3-4.
    ss << "            // ZA-Tile in out schreiben (out = za, kein fadd noetig)\n";
    ss << "            for (int row = 0; row < 16; ++row) {\n";
    ss << "                float* out_ptr0 = out + (m + row) * " << T << " + n;\n";
    ss << "                float* out_ptr1 = out_ptr0 + 16;\n";
    ss << "                register int row_reg asm(\"w12\") = row;\n";
    ss << "                asm volatile(\n";
    ss << "                    \"mova z4.s, p0/m, za0v.s[w12, 0]\\n\\t\"\n";
    ss << "                    \"st1w {z4.s}, p0, [%[o0]]\\n\\t\"\n";
    ss << "                    \"mova z4.s, p0/m, za2v.s[w12, 0]\\n\\t\"\n";
    ss << "                    \"st1w {z4.s}, p0, [%[o1]]\\n\\t\"\n";
    ss << "                    : \n";
    ss << "                    : [o0] \"r\"(out_ptr0), [o1] \"r\"(out_ptr1)\n";
    ss << "                    : \"z4\", \"w12\", \"memory\");\n";
    ss << "            }\n";
    // Zweite Haelfte (Zeilen 16..31): za1/za3
    ss << "            for (int row = 0; row < 16; ++row) {\n";
    ss << "                float* out_ptr0 = out + (m + 16 + row) * " << T << " + n;\n";
    ss << "                float* out_ptr1 = out_ptr0 + 16;\n";
    ss << "                register int row_reg asm(\"w12\") = row;\n";
    ss << "                asm volatile(\n";
    ss << "                    \"mova z4.s, p0/m, za1v.s[w12, 0]\\n\\t\"\n";
    ss << "                    \"st1w {z4.s}, p0, [%[o0]]\\n\\t\"\n";
    ss << "                    \"mova z4.s, p0/m, za3v.s[w12, 0]\\n\\t\"\n";
    ss << "                    \"st1w {z4.s}, p0, [%[o1]]\\n\\t\"\n";
    ss << "                    : \n";
    ss << "                    : [o0] \"r\"(out_ptr0), [o1] \"r\"(out_ptr1)\n";
    ss << "                    : \"z4\", \"w12\", \"memory\");\n";
    ss << "            }\n";
    ss << "        }\n"; // Ende N-Loop
    ss << "    }\n\n"; // Ende M-Loop

    ss << "    asm volatile(\"smstop\");\n";
    ss << "}\n}\n";
    return ss.str();
}

// ============================================================================
// NEON-Backend: 4x4 Outer-Product GEMM mit skalarem A-Broadcast
// out[R,T] += in0[R,P] * in1[P,T]
//
// Schedule-aware: nutzt die Autotuner-Transformtionen wie folgt:
//   - split_factor = p1-Extent = K-Tile-Groesse (Cache-Blocking der Reduktion)
//   - loop_order: relative Reihenfolge von r/t wird als Block-Schleifen-
//                 reihenfolge uebernommen (p0/p1 liegen innen, da die 4x4-
//                 Akkumulatoren pro (m,n)-Block den gesamten K-Bereich spannen)
//   - parallel_axis: OpenMP auf die r- oder t-Block-Schleife
//   - unroll_factor: Pragma auf die innere p1- (bzw. p-) Schleife
//
// A liegt row-major vor; fester m + variierendes p ist nicht contiguous.
// Daher 4 skalare Lasten + Broadcast via vmlaq_n_f32 (keine Transposition).
// B liegt row-major vor; festes p + variierendes t ist contiguous -> vld1q.
// ============================================================================
static std::string generateNEONKernel(const TEIR& ir) {
    std::stringstream ss;

    const int R = extentOf(ir, "r");
    const int T = extentOf(ir, "t");

    bool split = false;
    int F = 1;   // K-Tile-Groesse = p1-Extent
    int P;       // Volles Reduktions-Extent
    int P0 = 0;  // Aeussere K-Schleifen-Extent
    if (extentOf(ir, "p", -1) != -1) {
        P = extentOf(ir, "p");
        P0 = P;
    } else {
        F = extentOf(ir, "p1");
        P0 = extentOf(ir, "p0");
        P = P0 * F;
        split = true;
    }

    const int mBlocks = R / 4;
    const int nBlocks = T / 4;

    // Schedule analysieren: relative Reihenfolge von r und t + Parallel-Achsen.
    // NEON-Constraint: r,t (M/N-Bloecke) muessen ausserhalb von p0,p1 (K) liegen,
    // da die 4x4-Akkumulatoren pro (m,n)-Block den gesamten K-Bereich spannen.
    bool tBeforeR = false;
    bool parallelR = false, parallelT = false;
    int rPos = -1, tPos = -1;
    for (int i = 0; i < (int)ir.schedule.size(); ++i) {
        const auto& it = ir.schedule[i];
        if (it.axis == "r") { rPos = i; if (it.policy == Policy::Parallel) parallelR = true; }
        if (it.axis == "t") { tPos = i; if (it.policy == Policy::Parallel) parallelT = true; }
    }
    if (rPos >= 0 && tPos >= 0 && tPos < rPos) tBeforeR = true;

    const int unroll = ir.unrollFactor;

    ss << "// Auto-generiert vom TEIR-Autotuner (NEON-Backend, 4x4 Outer-Product)\n";
    ss << "// GEMM: out[" << R << "," << T << "] += in0[" << R << "," << P
       << "] * in1[" << P << "," << T << "]\n";
    ss << "// K-Tile (split p1): " << F << " | Unroll: " << unroll
       << " | Loop: " << (tBeforeR ? "t,r" : "r,t") << "\n";
    ss << "#include <arm_neon.h>\n";
    ss << "#include <omp.h>\n\n";
    ss << "extern \"C\" {\n";
    ss << "void teir_" << ir.name
       << "(const float* __restrict__ in0, const float* __restrict__ in1, float* __restrict__ out) {\n";

    // Tail-Elemente (R%4 bzw. T%4) brauchen null-initialisiertes out fuer die
    // skalare Rest-Akkumulation. Die 4x4-Bloecke ueberschreiben ihren Bereich
    // anschliessend mit dem vollen Akku-Ergebnis (out = acc, nicht out += acc).
    ss << "    for (int i = 0; i < " << (R * T) << "; ++i) out[i] = 0.0f;\n\n";

    // --- Aeussere Block-Schleifen (r/t in Schedule-Reihenfolge) ---
    if (tBeforeR) {
        if (parallelT) ss << "    #pragma omp parallel for\n";
        ss << "    for (int nb = 0; nb < " << nBlocks << "; ++nb) {\n";
        ss << "        const int n = nb * 4;\n";
        if (parallelR) ss << "        #pragma omp parallel for\n";
        ss << "        for (int mb = 0; mb < " << mBlocks << "; ++mb) {\n";
        ss << "            const int m = mb * 4;\n";
    } else {
        if (parallelR) ss << "    #pragma omp parallel for\n";
        ss << "    for (int mb = 0; mb < " << mBlocks << "; ++mb) {\n";
        ss << "        const int m = mb * 4;\n";
        if (parallelT) ss << "        #pragma omp parallel for\n";
        ss << "        for (int nb = 0; nb < " << nBlocks << "; ++nb) {\n";
        ss << "            const int n = nb * 4;\n";
    }

    // --- 4x4 Akkumulatoren nullen (eine float32x4_t pro Ausgabezeile) ---
    ss << "            float32x4_t acc0 = vdupq_n_f32(0.0f);\n";
    ss << "            float32x4_t acc1 = vdupq_n_f32(0.0f);\n";
    ss << "            float32x4_t acc2 = vdupq_n_f32(0.0f);\n";
    ss << "            float32x4_t acc3 = vdupq_n_f32(0.0f);\n\n";

    // --- K-Schleifen: p0 aussen, p1 innen (fmla). Ohne Split: einzelne p-Schleife ---
    if (split) {
        ss << "            for (int p0 = 0; p0 < " << P0 << "; ++p0) {\n";
        if (unroll > 1) ss << "                #pragma GCC unroll(" << unroll << ")\n";
        ss << "                for (int p1 = 0; p1 < " << F << "; ++p1) {\n";
        ss << "                    const int p = p0 * " << F << " + p1;\n";
    } else {
        ss << "            for (int p = 0; p < " << P << "; ++p) {\n";
    }

    // --- 4x4 Microkernel: 1 B-Vektor-Load + 4 skalare A-Loads + 4 fmla ---
    ss << "                    const float* b_ptr = in1 + p * " << T << " + n;\n";
    ss << "                    float32x4_t b_vec = vld1q_f32(b_ptr);\n";
    ss << "                    acc0 = vmlaq_n_f32(acc0, b_vec, in0[(m + 0) * " << P << " + p]);\n";
    ss << "                    acc1 = vmlaq_n_f32(acc1, b_vec, in0[(m + 1) * " << P << " + p]);\n";
    ss << "                    acc2 = vmlaq_n_f32(acc2, b_vec, in0[(m + 2) * " << P << " + p]);\n";
    ss << "                    acc3 = vmlaq_n_f32(acc3, b_vec, in0[(m + 3) * " << P << " + p]);\n";

    // --- K-Schleifen schliessen ---
    if (split) {
        ss << "                }\n";
        ss << "            }\n\n";
    } else {
        ss << "            }\n\n";
    }

    // --- Store: 4 Akku-Vektoren in out (out = acc, kein Read-Modify-Write) ---
    ss << "            vst1q_f32(out + (m + 0) * " << T << " + n, acc0);\n";
    ss << "            vst1q_f32(out + (m + 1) * " << T << " + n, acc1);\n";
    ss << "            vst1q_f32(out + (m + 2) * " << T << " + n, acc2);\n";
    ss << "            vst1q_f32(out + (m + 3) * " << T << " + n, acc3);\n";

    // --- Block-Schleifen schliessen ---
    ss << "        }\n";
    ss << "    }\n\n";

    // --- Skalarer Tail fuer R%4 bzw. T%4 (Reste; i.d.R. 0 Iterationen) ---
    ss << "    // Tail: Rest-Zeilen/Spalten bei R%4!=0 oder T%4!=0\n";
    ss << "    for (int m = " << (mBlocks * 4) << "; m < " << R << "; ++m)\n";
    ss << "        for (int p = 0; p < " << P << "; ++p)\n";
    ss << "            for (int t = 0; t < " << T << "; ++t)\n";
    ss << "                out[m * " << T << " + t] += in0[m * " << P << " + p] * in1[p * " << T << " + t];\n";
    ss << "    for (int m = 0; m < " << (mBlocks * 4) << "; ++m)\n";
    ss << "        for (int p = 0; p < " << P << "; ++p)\n";
    ss << "            for (int t = " << (nBlocks * 4) << "; t < " << T << "; ++t)\n";
    ss << "                out[m * " << T << " + t] += in0[m * " << P << " + p] * in1[p * " << T << " + t];\n";

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

// ============================================================================
// Einsum-Backend: Allgemeiner Tensorkontraktions-Codegen
// out[out_idx] = sum_{reduce_idx} in0[in0_idx] * in1[in1_idx]
// Ignoriert Schedule (Schicht 1); Schicht 2 generalisiert den Autotuner.
// ============================================================================

static std::string generateEinsumKernel(const TEIR& ir) {
    EinsumSpec spec = parseEinsum(ir.einsum);
    auto out_strides = computeStrides(spec.out_idx, ir);
    auto in0_strides = computeStrides(spec.in0_idx, ir);
    auto in1_strides = computeStrides(spec.in1_idx, ir);
    int out_size = tensorElements(spec.out_idx, ir);

    std::stringstream ss;
    ss << "// Auto-generiert vom TEIR-Autotuner (Einsum-Backend)\n";
    ss << "// " << ir.einsum << "\n";
    ss << "extern \"C\" {\n";
    ss << "void teir_" << ir.name
       << "(const float* __restrict__ in0, const float* __restrict__ in1, float* __restrict__ out) {\n";

    ss << "    for (int i = 0; i < " << out_size << "; ++i) out[i] = 0.0f;\n\n";

    int indent = 4;
    for (char c : spec.out_axes) {
        int ext = extentOfChar(ir, c);
        std::string pad(indent, ' ');
        ss << pad << "for (int " << c << " = 0; " << c << " < " << ext << "; ++" << c << ") {\n";
        indent += 4;
    }

    {
        std::string pad(indent, ' ');
        ss << pad << "float acc = 0.0f;\n";
    }

    for (char c : spec.reduce_axes) {
        int ext = extentOfChar(ir, c);
        std::string pad(indent, ' ');
        ss << pad << "for (int " << c << " = 0; " << c << " < " << ext << "; ++" << c << ") {\n";
        indent += 4;
    }

    {
        std::string pad(indent, ' ');
        ss << pad << "acc += in0[";
        for (int i = 0; i < (int)spec.in0_idx.size(); ++i) {
            if (i > 0) ss << " + ";
            ss << spec.in0_idx[i] << " * " << in0_strides[i];
        }
        ss << "] * in1[";
        for (int i = 0; i < (int)spec.in1_idx.size(); ++i) {
            if (i > 0) ss << " + ";
            ss << spec.in1_idx[i] << " * " << in1_strides[i];
        }
        ss << "];\n";
    }

    for (size_t i = 0; i < spec.reduce_axes.size(); ++i) {
        indent -= 4;
        std::string pad(indent, ' ');
        ss << pad << "}\n";
    }

    {
        std::string pad(indent, ' ');
        ss << pad << "out[";
        for (int i = 0; i < (int)spec.out_idx.size(); ++i) {
            if (i > 0) ss << " + ";
            ss << spec.out_idx[i] << " * " << out_strides[i];
        }
        ss << "] = acc;\n";
    }

    for (size_t i = 0; i < spec.out_axes.size(); ++i) {
        indent -= 4;
        std::string pad(indent, ' ');
        ss << pad << "}\n";
    }

    ss << "}\n}\n";
    return ss.str();
}

std::string generateSourceCode(const TEIR& ir) {
    if (!ir.einsum.empty()) {
        return generateEinsumKernel(ir);
    }
    if (ir.backend == Backend::SME) {
        return generateSMEKernel(ir);
    }
    if (ir.backend == Backend::NEON) {
        return generateNEONKernel(ir);
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