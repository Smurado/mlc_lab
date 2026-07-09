#pragma once

#include "teir.hpp"

#include <string>
#include <vector>

struct EinsumSpec {
    std::string out_idx;
    std::string in0_idx;
    std::string in1_idx;
    std::vector<char> out_axes;
    std::vector<char> reduce_axes;
    std::vector<char> all_axes;
};

EinsumSpec parseEinsum(const std::string& einsum);

int extentOfChar(const TEIR& ir, char c);

std::vector<int> computeStrides(const std::string& idx, const TEIR& ir);

int tensorElements(const std::string& idx, const TEIR& ir);

double einsumFlops(const TEIR& ir);

void referenceEinsum(const TEIR& ir, const float* in0, const float* in1, float* out);

// C2: Stichproben-Validierung fuer grosse Kontraktionen (VOLLE Referenz zu teuer).
// Erwartet GEMUSTERTE (nicht konstante) Eingaben und rechnet eine Stichprobe von
// Output-Elementen exakt gegen die Referenz nach. Faengt Layout-/Stride-Fehler, die
// ein konstanter Fuell-/Pruefwert nicht sieht. numSamples <= 0 => Env
// TEIR_VALIDATE_SAMPLES (Default 64). Liefert true, wenn alle Stichproben passen.
bool validateEinsumSample(const TEIR& ir, const float* in0, const float* in1,
                          const float* out, int numSamples = 0);

std::string axisExpr(const TEIR& ir, char c);

bool isReduceAxis(const std::string& schedAxis, const EinsumSpec& spec);

bool isGEMMForm(const EinsumSpec& spec);

struct GEMMMapping {
    char out0_axis;
    char out1_axis;
    char reduce_axis;
};

GEMMMapping extractGEMMMapping(const EinsumSpec& spec);
