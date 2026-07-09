// C2-Akzeptanztest: Faengt die Stichproben-Validierung einen bewusst eingebauten
// Layout-Bug (transponierte Ausgabe), den der alte Konstanten-Check durchrutschen
// liess? Baut eine kleine Kontraktion, erzeugt eine korrekte und eine korrupte
// Ausgabe und prueft beide Validatoren.
//
// Bauen:  make && clang++ -std=c++20 -O2 test_c2_validator.cpp einsum.o -o _test_c2 && ./_test_c2
#include "teir.hpp"
#include "einsum.hpp"
#include <vector>
#include <cstdio>
#include <cmath>

int main() {
    // Kleine, GEMM-foermige Kontraktion: out[a,b] = sum_c in0[a,c] * in1[c,b].
    const int N = 8;
    TEIR ir;
    ir.name = "c2test";
    ir.einsum = "ab-ac-cb";
    ir.axes = {{"a", N}, {"b", N}, {"c", N}};

    std::vector<float> in0(N * N), in1(N * N);
    // Muster (variierend, nicht konstant) — genau was C2 fuer den Layout-Test braucht.
    for (int i = 0; i < N * N; ++i) in0[i] = (float)((i % 13) + 1) / 13.0f;
    for (int i = 0; i < N * N; ++i) in1[i] = (float)((i % 7) + 1) / 7.0f;

    std::vector<float> correct(N * N, 0.0f);
    referenceEinsum(ir, in0.data(), in1.data(), correct.data());

    // Layout-Bug: transponierte Ausgabe  outBad[a,b] = correct[b,a].
    std::vector<float> transposed(N * N, 0.0f);
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b)
            transposed[a * N + b] = correct[b * N + a];

    bool okCorrect = validateEinsumSample(ir, in0.data(), in1.data(), correct.data(), N * N);
    bool okBad     = validateEinsumSample(ir, in0.data(), in1.data(), transposed.data(), N * N);

    // Gegenprobe: der ALTE Konstanten-Check. Mit KONSTANTEN Eingaben (1.0/2.0) ist
    // jedes korrekte Output-Element = 2*N; die transponierte Ausgabe ist dann
    // ebenfalls ueberall 2*N -> der alte Check "out == Konstante" bemerkt den Bug NICHT.
    std::vector<float> c0(N * N, 1.0f), c1(N * N, 2.0f), cConst(N * N, 0.0f);
    referenceEinsum(ir, c0.data(), c1.data(), cConst.data());
    std::vector<float> cConstT(N * N, 0.0f);
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b)
            cConstT[a * N + b] = cConst[b * N + a];
    const float konst = 2.0f * N;
    bool oldCheckMissesBug = true; // alter Check auf transponierter Konstanten-Ausgabe
    for (int i = 0; i < N * N; ++i)
        if (std::fabs(cConstT[i] - konst) > 1e-1f) { oldCheckMissesBug = false; break; }

    std::printf("C2-Akzeptanztest (%s):\n", ir.einsum.c_str());
    std::printf("  neue Stichprobe: korrekte Ausgabe  -> %s (erwartet PASS)\n",
                okCorrect ? "PASS" : "FAIL");
    std::printf("  neue Stichprobe: LAYOUT-BUG (transp.) -> %s (erwartet FAIL)\n",
                okBad ? "PASS" : "FAIL");
    std::printf("  alter Konstanten-Check auf demselben Bug -> %s\n",
                oldCheckMissesBug ? "PASS (Bug uebersehen!)" : "FAIL (Bug bemerkt)");

    bool testPass = okCorrect && !okBad && oldCheckMissesBug;
    std::printf("\n  => %s: die neue Validierung faengt den Bug, den der alte Check "
                "durchrutschen liess.\n", testPass ? "BESTANDEN" : "DURCHGEFALLEN");
    return testPass ? 0 : 1;
}
