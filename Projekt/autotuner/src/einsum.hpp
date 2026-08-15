// Einstein-Notation: Zerlegung der Kontraktions-Strings ("ab-ac-cb"), daraus
// abgeleitete Groessen (Extents, Strides, FLOPs) sowie die naive Referenz-
// implementierung, gegen die jeder generierte Kernel validiert wird.
#pragma once

#include "teir.hpp"

#include <string>
#include <vector>

// Zerlegter Einsum-String: Index-Strings der drei Tensoren plus die daraus
// abgeleiteten Achsmengen (Ausgabe-, Reduktions- und alle Achsen).
struct EinsumSpec {
    std::string out_idx;
    std::string in0_idx;
    std::string in1_idx;
    std::vector<char> out_axes;
    std::vector<char> reduce_axes;
    std::vector<char> all_axes;
};

EinsumSpec parseEinsum(const std::string& einsum);

// Extent einer Einsum-Achse. Rechnet gesplittete Achsen (a -> a0/a1) wieder
// zum Original zusammen; das Namensschema stammt aus splitOuterAxis.
int extentOfChar(const TEIR& ir, char c);

// Zeilen-major-Strides fuer den Index-String eines Tensors.
std::vector<int> computeStrides(const std::string& idx, const TEIR& ir);

// Elementzahl des Tensors mit diesem Index-String.
int tensorElements(const std::string& idx, const TEIR& ir);

// 2 * Produkt aller Achs-Extents (multiply-add je innerster Iteration).
double einsumFlops(const TEIR& ir);

// Naive Referenz (Triple-Loop-Verallgemeinerung), bewusst ohne jede
// Optimierung: Massstab fuer die Korrektheit, nicht fuer die Zeit.
void referenceEinsum(const TEIR& ir, const float* in0, const float* in1, float* out);

// C2: Stichproben-Validierung fuer grosse Kontraktionen (VOLLE Referenz zu teuer).
// Erwartet GEMUSTERTE (nicht konstante) Eingaben und rechnet eine Stichprobe von
// Output-Elementen exakt gegen die Referenz nach. Faengt Layout-/Stride-Fehler, die
// ein konstanter Fuell-/Pruefwert nicht sieht. numSamples <= 0 => Env
// TEIR_VALIDATE_SAMPLES (Default 64). Liefert true, wenn alle Stichproben passen.
bool validateEinsumSample(const TEIR& ir, const float* in0, const float* in1,
                          const float* out, int numSamples = 0);

// Index-Ausdruck einer Achse fuer den Codegen; bei gesplitteten Achsen
// "(a0 * extent_a1 + a1)", sonst der Schleifenname selbst.
std::string axisExpr(const TEIR& ir, char c);

// Gehoert diese Schedule-Achse (auch als Split-Teil a0/a1) zu den
// Reduktionsachsen der Kontraktion?
bool isReduceAxis(const std::string& schedAxis, const EinsumSpec& spec);

// GEMM-Form: exakt out[a,b] = sum_c in0[a,c] * in1[c,b], inklusive der
// Achsreihenfolge in den Eingaben. Nur dann greifen die NEON-/SME-Kernel.
bool isGEMMForm(const EinsumSpec& spec);

struct GEMMMapping {
    char out0_axis;
    char out1_axis;
    char reduce_axis;
};

// Ordnet die drei Achsen einer GEMM-foermigen Kontraktion den Rollen M/N/K zu.
GEMMMapping extractGEMMMapping(const EinsumSpec& spec);
