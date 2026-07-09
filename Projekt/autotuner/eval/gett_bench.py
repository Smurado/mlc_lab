#!/usr/bin/env python3
"""
Per-Fall-Worker fuer die GETT-Vergleichsmatrix (torch ODER TVM).
Wird vom Orchestrator run_gett_matrix.py als SUBPROZESS aufgerufen (Isolation +
Timeout + Fehler-Skip). Liest EINE GETT-CSV-Zeile, misst eine Engine, druckt:
    RESULT <gflops>        bei Erfolg
    RESULT FAIL <grund>    bei Fehler

torch-Worker laeuft unter dem .venv-Python (torch 2.13), TVM-Worker unter dem
conda-Env tvm-bench (selbst gebautes TVM 0.17). Imports sind lazy — jede Engine
zieht nur ihre eigene Abhaengigkeit.

Aufruf:  python gett_bench.py --engine torch|tvm --row "<csv-zeile>"
Budget:  TVM via Env TVM_TRIALS (Default 200).
"""
import os
import sys
import time


def parse_row(line):
    """GETT-Zeile -> (name, out_idx, in0_idx, in1_idx, ext{char:int}, flops)."""
    f = [c.strip() for c in line.strip().split(",")]  # CRLF-robust
    name, axes_field, einsum = f[0], f[2], f[6]
    ext = {}
    for pair in axes_field.split(";"):
        c, e = pair.split(":")
        ext[c] = int(e)
    out_idx, in0_idx, in1_idx = einsum.split("-")
    flops = 2.0
    for c in set(out_idx + in0_idx + in1_idx):
        flops *= ext[c]
    return name, out_idx, in0_idx, in1_idx, ext, flops


# --------------------------------------------------------------------------- torch
def run_torch(out_idx, in0_idx, in1_idx, ext, flops):
    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    eq = f"{in0_idx},{in1_idx}->{out_idx}"
    A = torch.randn(*[ext[c] for c in in0_idx], dtype=torch.float32)
    B = torch.randn(*[ext[c] for c in in1_idx], dtype=torch.float32)
    fn = lambda: torch.einsum(eq, A, B)
    for _ in range(10):
        fn()
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        best = min(best, (time.perf_counter() - t0) / 20)
    return flops / best / 1e9


# --------------------------------------------------------------------------- TVM
def run_tvm(out_idx, in0_idx, in1_idx, ext, flops):
    import tempfile
    import numpy as np
    import tvm
    from tvm import te
    from tvm import meta_schedule as ms

    ncpu = os.cpu_count() or 1
    trials = int(os.environ.get("TVM_TRIALS", "200"))
    target = tvm.target.Target(f"llvm -num-cores {ncpu} -mcpu=apple-m1")

    # Generischer Einsum -> TensorIR-PrimFunc. Reduktionsachsen = in Inputs, nicht in Out.
    A = te.placeholder(tuple(ext[c] for c in in0_idx), "float32", "A")
    B = te.placeholder(tuple(ext[c] for c in in1_idx), "float32", "B")
    red = [c for c in dict.fromkeys(in0_idx + in1_idx) if c not in out_idx]
    rax = {c: te.reduce_axis((0, ext[c]), c) for c in red}

    def compute(*outc):
        idx = dict(zip(out_idx, outc))
        idx.update(rax)
        a = A[tuple(idx[c] for c in in0_idx)]
        b = B[tuple(idx[c] for c in in1_idx)]
        prod = a * b
        return te.sum(prod, axis=list(rax.values())) if rax else prod

    C = te.compute(tuple(ext[c] for c in out_idx), compute, name="C")
    mod = tvm.IRModule({"main": te.create_prim_func([A, B, C])})

    with tempfile.TemporaryDirectory() as work:
        db = ms.tir_integration.tune_tir(
            mod=mod, target=target, work_dir=work,
            max_trials_global=trials, num_trials_per_iter=min(64, trials),
        )
        sch = ms.tir_integration.compile_tir(db, mod, target)
        lib = tvm.build(sch.mod, target=target)

    dev = tvm.cpu()
    args = [tvm.nd.array(np.random.randn(*[ext[c] for c in idx]).astype("float32"), dev)
            for idx in (in0_idx, in1_idx)]
    args.append(tvm.nd.array(np.zeros([ext[c] for c in out_idx], "float32"), dev))
    ev = lib.time_evaluator("main", dev, number=20, repeat=5)
    t = ev(*args).min
    return flops / t / 1e9


def main():
    argv = sys.argv[1:]
    engine = argv[argv.index("--engine") + 1]
    row = argv[argv.index("--row") + 1]
    try:
        _, out_idx, in0_idx, in1_idx, ext, flops = parse_row(row)
        gf = run_torch(out_idx, in0_idx, in1_idx, ext, flops) if engine == "torch" \
            else run_tvm(out_idx, in0_idx, in1_idx, ext, flops)
        print(f"RESULT {gf:.4f}")
    except Exception as e:
        print(f"RESULT FAIL {type(e).__name__}:{str(e)[:100]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
