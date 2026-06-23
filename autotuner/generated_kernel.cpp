// Automatisch generiert durch den TEIR-Compiler Autotuner
#include <iostream>
#include <vector>
#include <omp.h>

void teir_contraction(
    float* in0,
    float* in1,
    float* out
) {
    #pragma omp parallel for
    for (int r = 0; r < 96; ++r) {
        for (int t = 0; t < 32; ++t) {
            #pragma omp parallel for
            for (int p0 = 0; p0 < 2; ++p0) {
                for (int p1 = 0; p1 < 64; ++p1) {
                    // Invoke Primitives Kernels
                    // execute_primitive_zero();
                    // execute_primitive_gemm();
                    out[0] += in0[0] * in1[0]; // Platzhalter fuer Tensor Contraction
                }
            }
        }
    }
}
