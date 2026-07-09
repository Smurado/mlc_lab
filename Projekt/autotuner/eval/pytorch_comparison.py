#!/usr/bin/env python3
"""
D3 - Realitaetsanker: Wie viel Prozent von PyTorch erreicht der getunte TEIR-Kernel?

Fuer die GEMM-foermige Teilmenge (out[a,b] = sum_c in0[a,c]*in1[c,b], isGEMMForm)
wird DERSELBE Fall durch beide Pfade getrieben:

  1. TEIR: der eigene Autotuner tuned den Kernel real (JIT + Benchmark). Es wird die
     EHRLICHE [PERFORMANCE]-Endmessung geparst (nicht die grobe Such-CSV), fuer die
     Backends scalar UND neon — berichtet wird das bessere ("TEIRs beste Seite").
  2. torch.matmul: Apples hand-optimiertes BLAS (Accelerate) auf identischen Extents
     — der ehrliche Anker. Gemessen mit 1 Thread (fairer Kernel-Vergleich) UND mit
     allen Kernen (echter BLAS-Peak).
  3. TorchInductor: PyTorchs eigener JIT-Autotuner (torch.compile) — das direkte
     Gegenstueck zu TEIRs Autotuner. Best-effort; wird uebersprungen, wenn er nicht baut.

Ausgegeben wird pro Groesse "TEIR = X % von <Referenz>". Erwartung: X ist KLEIN gegen
Multi-Thread-BLAS — das ist die ehrliche, lehrreiche Aussage (Roadmap D3), kein Sieg.

Aufruf:  .venv/bin/python pytorch_comparison.py
"""
import os
import re
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

DIR = os.path.dirname(os.path.abspath(__file__))          # eval/
COMPILER = os.path.join(os.path.dirname(DIR), "src", "teir_compiler")

try:
    import torch
except ImportError:
    sys.exit("PyTorch nicht gefunden. Installieren: .venv/bin/pip install torch")

# ------------------------------- Konfiguration -------------------------------
# GEMM-foermiger Einsum: out[a,b] = sum_c in0[a,c] * in1[c,b]  (M=a, N=b, K=c).
EINSUM = "ab-ac-cb"
SIZES = [256, 512]              # quadratisch: M = N = K = size
BACKENDS = ["scalar", "neon"]  # TEIRs beide GEMM-Pfade; berichtet wird der bessere
TEIR_TRIALS = 20               # kurzes, faires Tuning-Budget je Backend
WARMUP, ITERS, REPEATS = 30, 200, 5
NCPU = os.cpu_count() or 1


def make_csv(size):
    path = os.path.join(DIR, "_d3_input.csv")
    axes = f"a:{size};b:{size};c:{size}"
    with open(path, "w") as f:
        f.write("name,tensors,axes,primitives,schedule,invokes,einsum\n")
        f.write(f'contraction_d3,"in0:f32;in1:f32;out:f32","{axes}",'
                f'"zero;gemm","a:sequential;b:sequential;c:sequential","zero;gemm","{EINSUM}"\n')
    return path


def run_teir(csv_path, backend):
    """Tuned den Kernel und liefert die ehrliche [PERFORMANCE]-GFLOPS (oder None)."""
    env = {**os.environ,
           "TEIR_INPUT": csv_path, "TEIR_STRATEGY": "sa", "TEIR_BACKEND": backend,
           "TEIR_MAX_TRIALS": str(TEIR_TRIALS), "TEIR_TIME_BUDGET_MS": "600000",
           "TEIR_SEARCH_OPT": "-O2", "TEIR_SEED": "1"}
    try:
        r = subprocess.run([COMPILER], capture_output=True, text=True,
                           timeout=1200, env=env, cwd=DIR)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout + r.stderr
    if "[SUCCESS]" not in out:
        return None
    m = re.search(r"\[PERFORMANCE\]\s+[\d.eE+-]+\s+ms\s+\(([\d.eE+-]+)\s+GFLOPS\)", out)
    return float(m.group(1)) if m else None


def bench(fn):
    for _ in range(WARMUP):
        fn()
    best_ms = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        ms = (time.perf_counter() - t0) * 1e3 / ITERS
        best_ms = min(best_ms, ms)
    return best_ms


def torch_gflops(fn, flops):
    return flops / (bench(fn) * 1e-3) / 1e9


def pct(a, b):
    return f"{100.0 * a / b:5.1f}%" if (a and b) else "  n/a"


def main():
    print("=" * 70)
    print("  D3 - TEIR vs. PyTorch (GEMM-foermig)   Einsum: " + EINSUM)
    print(f"  PyTorch {torch.__version__} | {NCPU} CPU-Kerne | Tuning: SA, {TEIR_TRIALS} Trials")
    print("=" * 70)

    # TorchInductor einmalig vorbereiten (best-effort).
    inductor_fn = None
    try:
        @torch.compile(backend="inductor", mode="max-autotune")
        def _cmm(a, b):
            return torch.mm(a, b)
        inductor_fn = _cmm
    except Exception as e:
        print(f"  [WARN] TorchInductor nicht verfuegbar: {e}")

    summary = []
    for size in SIZES:
        M = N = K = size
        flops = 2.0 * M * N * K
        csv_path = make_csv(size)

        # --- TEIR: beide Backends, bestes berichten ---
        teir = {}
        for be in BACKENDS:
            print(f"  [{size}^3] TEIR tuning ({be}) ...", flush=True)
            teir[be] = run_teir(csv_path, be)
        teir_best = max((v for v in teir.values() if v), default=None)
        teir_be = max(teir, key=lambda b: teir[b] or -1) if teir_best else "-"

        # --- torch.matmul: 1 Thread und alle Kerne ---
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(K, N, dtype=torch.float32)

        torch.set_num_threads(1)
        gf_mm1 = torch_gflops(lambda: torch.mm(A, B), flops)

        torch.set_num_threads(NCPU)
        gf_mmN = torch_gflops(lambda: torch.mm(A, B), flops)

        gf_ind = None
        if inductor_fn is not None:
            try:
                inductor_fn(A, B)  # Trigger-Compile (zaehlt nicht)
                gf_ind = torch_gflops(lambda: inductor_fn(A, B), flops)
            except Exception:
                gf_ind = None

        summary.append((size, teir_best, teir_be, gf_mm1, gf_mmN, gf_ind))

        print(f"\n  --- {size}^3  (FLOPs = {flops:.2e}) ---")
        print(f"    TEIR best ({teir_be:6s})   : {(f'{teir_best:8.2f}' if teir_best else '   FAIL')} GFLOPS")
        print(f"    torch.matmul   1 Thread : {gf_mm1:8.2f} GFLOPS   -> TEIR = {pct(teir_best, gf_mm1)}")
        print(f"    torch.matmul  {NCPU:2d} Thr.  : {gf_mmN:8.2f} GFLOPS   -> TEIR = {pct(teir_best, gf_mmN)}")
        if gf_ind:
            print(f"    TorchInductor {NCPU:2d} Thr.  : {gf_ind:8.2f} GFLOPS   -> TEIR = {pct(teir_best, gf_ind)}")
        print()

    if os.path.exists(os.path.join(DIR, "_d3_input.csv")):
        os.remove(os.path.join(DIR, "_d3_input.csv"))

    # ------------------------------- Fazit -------------------------------
    print("=" * 70)
    print("  Zusammenfassung (TEIR-best als % der jeweiligen Referenz)")
    print("=" * 70)
    print(f"  {'Groesse':>8} {'TEIR':>10} {'%(1thr)':>9} {'%(Nthr)':>9} {'%(Induct)':>10}")
    for size, tb, be, g1, gn, gi in summary:
        tb_s = f"{tb:.1f}" if tb else "FAIL"
        print(f"  {size:>6}^3 {tb_s:>10} {pct(tb, g1):>9} {pct(tb, gn):>9} "
              f"{(pct(tb, gi) if gi else '   n/a'):>10}")
    print()
    print("  Ehrliche Einordnung: gegen Multi-Thread-BLAS (Accelerate, geblockt,")
    print("  hand-optimiert) erreicht ein getunter TEIR-Kernel nur einen kleinen")
    print("  Bruchteil. Der Single-Thread-Vergleich ist der faire Kernel-zu-Kernel-")
    print("  Massstab. Ziel von D3 ist der Lerneffekt, nicht der Sieg (Roadmap).")


if __name__ == "__main__":
    main()
