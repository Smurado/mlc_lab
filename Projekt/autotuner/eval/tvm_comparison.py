#!/usr/bin/env python3
"""
D3-Zusatz: TEIR-Autotuner vs. Apache TVM (MetaSchedule) auf demselben GEMM.

Anders als der torch-Vergleich (BLAS laeuft ueber den AMX-Matrix-Coprozessor) ist
TVM der FAIRE Vergleich: TVM generiert wie TEIR CPU-Code ueber LLVM und tuned Schedules
per Suche (MetaSchedule = TVMs Autotuner). Gleicher Hardwarepfad (NEON, LLVM), also misst
"X % von TVM" tatsaechlich Autotuner+Codegen gegen Autotuner+Codegen.

Laeuft im conda-Env 'tvm-bench' mit SELBST GEBAUTEM TVM v0.17 (das pip-Wheel 0.25 hat
keinen Tuner), editierbar installiert aus ~/tvm-0.17 — 'import tvm' funktioniert direkt,
kein PYTHONPATH noetig. Setup/Neubau: siehe TVM_SETUP.md.

Aufruf:  <conda>/envs/tvm-bench/bin/python tvm_comparison.py
Budget:  TVM_TRIALS (Default 8 = Smoke-Test; fuer echte Zahlen z.B. 512 setzen)
         TVM_SIZES  (Default "256,512")
"""
import os
import sys
import tempfile
import numpy as np

import tvm
from tvm import te
from tvm import meta_schedule as ms

NCPU = os.cpu_count() or 1
TRIALS = int(os.environ.get("TVM_TRIALS", "8"))
SIZES = [int(s) for s in os.environ.get("TVM_SIZES", "256,512").split(",")]
# apple-m1 = sichere NEON-Basis, die LLVM 18 kennt (M4 hat dieselbe 128-bit-NEON-Breite).
TARGET = tvm.target.Target(f"llvm -num-cores {NCPU} -mcpu=apple-m1")


def matmul_mod(M, N, K):
    A = te.placeholder((M, K), "float32", name="A")
    B = te.placeholder((K, N), "float32", name="B")
    k = te.reduce_axis((0, K), "k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    return tvm.IRModule({"main": te.create_prim_func([A, B, C])})


def gflops(lib, M, N, K, fname="main"):
    dev = tvm.cpu()
    a = tvm.nd.array(np.random.randn(M, K).astype("float32"), dev)
    b = tvm.nd.array(np.random.randn(K, N).astype("float32"), dev)
    c = tvm.nd.array(np.zeros((M, N), "float32"), dev)
    ev = lib.time_evaluator(fname, dev, number=50, repeat=5)
    t = ev(a, b, c).min  # Sekunden, Best-of-repeat
    return 2.0 * M * N * K / t / 1e9


def main():
    print("=" * 66)
    print(f"  TVM {tvm.__version__} (MetaSchedule) vs. TEIR — GEMM")
    print(f"  Target: {TARGET}  | Tuning-Budget: {TRIALS} Trials/Groesse")
    print("=" * 66)

    results = []
    for size in SIZES:
        M = N = K = size
        mod = matmul_mod(M, N, K)

        # Naiver (ungeschedulter) TVM-Build als Referenz-Untergrenze.
        naive = tvm.build(mod, target=TARGET)
        gf_naive = gflops(naive, M, N, K)

        # MetaSchedule: TVMs Autotuner sucht Tile-/Vectorize-/Parallel-Schedules.
        with tempfile.TemporaryDirectory() as work:
            db = ms.tir_integration.tune_tir(
                mod=mod, target=TARGET, work_dir=work,
                max_trials_global=TRIALS,
                num_trials_per_iter=min(64, TRIALS),
            )
            sch = ms.tir_integration.compile_tir(db, mod, TARGET)
            tuned = tvm.build(sch.mod, target=TARGET)
        gf_tuned = gflops(tuned, M, N, K)

        results.append((size, gf_naive, gf_tuned))
        print(f"\n  {size}^3:  TVM naiv = {gf_naive:7.1f} GFLOPS   |   "
              f"TVM tuned = {gf_tuned:7.1f} GFLOPS")

    print("\n" + "=" * 66)
    print(f"  {'Groesse':>8} {'TVM naiv':>10} {'TVM tuned':>11}")
    for size, gn, gt in results:
        print(f"  {size:>6}^3 {gn:>10.1f} {gt:>11.1f}")
    print("\n  (TEIR-Zahlen aus pytorch_comparison.py separat gegenstellen.)")


if __name__ == "__main__":
    main()
