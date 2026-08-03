#pragma once
// Korrektheitspruefung eines JIT-kompilierten Einsum-Kernels.
//
// Warum ein eigener Header: diese Logik stand wortgleich zweimal im Code --
// in benchmark.cpp (Pruefung jedes Such-Trials) und in main.cpp (Pruefung des
// Gewinners vor der Endmessung). Beides muss zwingend dasselbe pruefen, sonst
// akzeptiert die Suche Kernel, die die Endmessung ablehnt (oder umgekehrt).
// Genau diese Doppelung hat bereits zweimal zu stillen Divergenzen gefuehrt
// (Blockgroesse der Messschleife, OOM-Guard).
//
// Nicht enthalten: der GEMM-Pfad (ir.einsum leer). Der ist in beiden Dateien
// unterschiedlich gewachsen -- benchmark.cpp prueft mit Toleranz 1e-2, main.cpp
// mit 1e-4 -- und wird hier bewusst NICHT vereinheitlicht, um das Verhalten der
// bereits erhobenen Messreihen nicht zu veraendern.

#include "teir.hpp"
#include "einsum.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// Ab dieser Iterationszahl ist die volle Referenzrechnung zu teuer und es wird
// auf die Stichproben-Validierung (C2) umgeschaltet.
constexpr double kFullReferenceMaxIters = 100000000.0;

// Volle Referenz oder Stichprobe?
inline bool useFullReference(const TEIR& ir)
{
    return (einsumFlops(ir) / 2.0) <= kFullReferenceMaxIters;
}

// C2: Eingaben IMMER gemustert fuellen -- sowohl die volle Referenz als auch die
// Stichprobe brauchen variierende Werte, damit ein falsches Layout nicht
// zufaellig eine Konstante trifft und durchrutscht. Die Perioden 13 und 7 sind
// teilerfremd, damit sich das Muster ueber in0 x in1 nicht frueh wiederholt.
inline void fillEinsumInputs(std::vector<float>& in0, std::vector<float>& in1)
{
    for (std::size_t i = 0; i < in0.size(); ++i)
        in0[i] = static_cast<float>((i % 13) + 1) / 13.0f;
    for (std::size_t i = 0; i < in1.size(); ++i)
        in1[i] = static_cast<float>((i % 7) + 1) / 7.0f;
}

// Relative Toleranz mit absoluter Untergrenze: kleine Referenzwerte duerfen
// nicht an einer rein relativen Schranke scheitern.
inline bool withinTolerance(float got, float ref)
{
    const float tol = 1e-2f * std::max(1.0f, std::abs(ref));
    return std::abs(got - ref) <= tol;
}

// Vollstaendige Pruefung der Kernel-Ausgabe (Einsum-Pfad).
// Klein genug -> jedes Element gegen die Referenz; sonst Stichprobe.
inline bool validateEinsumOutput(const TEIR& ir, const float* in0, const float* in1,
                                 const float* out, int outSize)
{
    if (useFullReference(ir)) {
        std::vector<float> ref(static_cast<std::size_t>(outSize), 0.0f);
        referenceEinsum(ir, in0, in1, ref.data());
        for (int i = 0; i < outSize; ++i)
            if (!withinTolerance(out[i], ref[i]))
                return false;
        return true;
    }
    return validateEinsumSample(ir, in0, in1, out);
}
