#!/usr/bin/env python3
"""
TEIR Autotuner vs. TorchInductor – GEMM Benchmark
==================================================
Vergleicht drei Varianten auf dem SELBEN Input (input.csv):

  1. TEIR (eigene Implementierung)
     Ergebnisse aus autotuner_results.csv – JIT-Kernel, der durch
     Simulated Annealing / Genetic Algorithm / Random Search getuned wurde.

  2. TorchInductor (bestehender Autotuner, torch.compile)
     PyTorchs eingebetteter Compiler-Stack: Dynamo + Inductor generiert
     zur Laufzeit hardware-spezifischen C++/OpenMP-Code und wählt
     automatisch tile sizes und loop-Transformationen – konzeptuell das
     Gleiche wie TEIR, aber als etabliertes Framework.

  3. Baseline: torch.mm (plain BLAS, kein Autotuner)
     Gibt die Obergrenze vor (Accelerate / OpenBLAS hand-optimiert).

Operation: out[R, T] = in0[R, P] @ in1[P, T]
Dimensionen aus input.csv: R=96, P=128, T=32
"""

import time
import csv
import sys
import os
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
except ImportError:
    sys.exit("PyTorch nicht gefunden. Bitte mit 'pip3 install torch' installieren.")

# ---------------------------------------------------------------------------
# Problemdimensionen (aus input.csv: p=128, r=96, t=32)
# ---------------------------------------------------------------------------
R = 96
P = 128
T = 32
FLOPS_PER_CALL = 2 * R * P * T   # 786 432 FLOPs (2 × MACs)

# ---------------------------------------------------------------------------
# Benchmark-Konfiguration
# ---------------------------------------------------------------------------
WARMUP_ITERS = 100
BENCH_ITERS  = 500
REPEATS      = 5          # Best-of-N reduziert Messjitter
RESULTS_CSV  = os.path.join(os.path.dirname(__file__), "autotuner_results.csv")


def bench(fn, warmup: int = WARMUP_ITERS,
          iters: int = BENCH_ITERS, repeats: int = REPEATS):
    """Gibt (best_ms, gflops) zurück."""
    for _ in range(warmup):
        fn()
    best_ms = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        elapsed_ms = (time.perf_counter() - t0) * 1e3 / iters
        if elapsed_ms < best_ms:
            best_ms = elapsed_ms
    gflops = FLOPS_PER_CALL / (best_ms * 1e-3) / 1e9
    return best_ms, gflops


def load_teir_results(path: str):
    rows = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["gflops"]     = float(row["gflops"])
                row["runtime_ms"] = float(row["runtime_ms"])
                rows.append(row)
    except FileNotFoundError:
        print(f"[WARN] {path} nicht gefunden – TEIR-Ergebnisse werden übersprungen.")
    return rows


def print_table(rows, col_w=(30, 14, 12)):
    header = f"{'Methode':<{col_w[0]}}{'Zeit (ms)':>{col_w[1]}}{'GFLOPS':>{col_w[2]}}"
    sep    = "-" * sum(col_w)
    print(header)
    print(sep)
    for name, ms, gf in rows:
        print(f"{name:<{col_w[0]}}{ms:>{col_w[1]}.5f}{gf:>{col_w[2]}.3f}")


# ---------------------------------------------------------------------------
# Haupt-Benchmark
# ---------------------------------------------------------------------------
def main():
    torch.set_num_threads(os.cpu_count() or 1)

    print("=" * 60)
    print("  TEIR vs. TorchInductor – GEMM-Benchmark")
    print("=" * 60)
    print(f"  Input     : out[{R},{T}] = in0[{R},{P}] @ in1[{P},{T}]")
    print(f"  FLOPs     : {FLOPS_PER_CALL:,}  (2 × R × P × T)")
    print(f"  PyTorch   : {torch.__version__}  |  {torch.get_num_threads()} CPU-Threads")
    print(f"  Messung   : Best-of-{REPEATS} × {BENCH_ITERS} Iterationen")
    print("=" * 60)
    print()

    in0 = torch.ones(R, P, dtype=torch.float32)
    in1 = torch.full((P, T), 2.0, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Abschnitt A: Baseline – plain torch.mm  (kein Autotuner, nur BLAS)
    # ------------------------------------------------------------------
    ms_mm, gf_mm = bench(lambda: torch.mm(in0, in1))

    # ------------------------------------------------------------------
    # Abschnitt B: TorchInductor (bestehender Autotuner)
    # torch.compile analysiert den Compute-Graph und generiert zur
    # Laufzeit optimierten C++/OpenMP-Code mit automatischer Tile-
    # und Loop-Auswahl – konzeptuell identisch zu TEIR.
    # ------------------------------------------------------------------
    print("[INFO] torch.compile (TorchInductor) wird kompiliert ... ", end="", flush=True)

    @torch.compile(backend="inductor", mode="max-autotune")
    def compiled_mm(a, b):
        return torch.mm(a, b)

    # Erster Aufruf triggert die Kompilierung (zählt nicht zur Messung)
    _ = compiled_mm(in0, in1)
    print("fertig.")

    ms_ind, gf_ind = bench(lambda: compiled_mm(in0, in1))

    # Korrektheit beider Varianten prüfen
    expected = 2.0 * P
    ref = torch.mm(in0, in1)
    assert torch.allclose(ref, torch.full((R, T), expected)), "torch.mm Validierung fehlgeschlagen!"
    assert torch.allclose(compiled_mm(in0, in1), torch.full((R, T), expected)), \
        "TorchInductor Validierung fehlgeschlagen!"
    print("[OK] Validierung erfolgreich (alle Ausgaben = 256.0)\n")

    # ------------------------------------------------------------------
    # Abschnitt C: TEIR (eigene Implementierung)
    # ------------------------------------------------------------------
    teir_rows = load_teir_results(RESULTS_CSV)
    best_teir = max(teir_rows, key=lambda r: r["gflops"]) if teir_rows else None

    # ------------------------------------------------------------------
    # Ausgabe-Tabelle
    # ------------------------------------------------------------------
    print("┌─ Baseline (kein Autotuner) ─────────────────────────────")
    print_table([
        ("torch.mm  [plain BLAS]", ms_mm, gf_mm),
    ])
    print()

    print("┌─ Bestehender Autotuner: TorchInductor ──────────────────")
    print_table([
        ("torch.compile [Inductor]", ms_ind, gf_ind),
    ])
    print()

    if teir_rows:
        top5 = sorted(teir_rows, key=lambda r: r["gflops"], reverse=True)[:5]
        print("┌─ Eigene Implementierung: TEIR Autotuner ─────────────────")
        print_table([
            (f"TEIR [{r['strategy'][:8]}] {r['config'][:18]}", r["runtime_ms"], r["gflops"])
            for r in top5
        ])
        print()

    # ------------------------------------------------------------------
    # Fazit & Speedup-Vergleich
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Zusammenfassung")
    print("=" * 60)
    print(f"  Baseline  (torch.mm)         : {gf_mm:>8.3f} GFLOPS  ({ms_mm:.5f} ms)")
    print(f"  Inductor  (torch.compile)    : {gf_ind:>8.3f} GFLOPS  ({ms_ind:.5f} ms)")

    if best_teir:
        gf_t = best_teir["gflops"]
        ms_t = best_teir["runtime_ms"]
        print(f"  TEIR best (eigener Autotuner): {gf_t:>8.3f} GFLOPS  ({ms_t:.5f} ms)")
        print()
        print(f"  TEIR vs. Inductor : ", end="")
        ratio = gf_t / gf_ind
        if ratio >= 1.0:
            print(f"TEIR ist {ratio:.2f}× schneller")
        else:
            print(f"Inductor ist {1/ratio:.2f}× schneller")

        print(f"  TEIR vs. Baseline : ", end="")
        ratio2 = gf_t / gf_mm
        if ratio2 >= 1.0:
            print(f"TEIR ist {ratio2:.2f}× schneller")
        else:
            print(f"Baseline ist {1/ratio2:.2f}× schneller")

    print()
    print("Hinweis:")
    print("  TorchInductor  = JIT-Compiler + auto tile selection (wie TEIR,")
    print("  aber nutzt Apples Accelerate/BLAS als Fallback-Codepfad).")
    print("  Baseline torch.mm = manuell hand-optimierte BLAS-Routine.")
    print()


if __name__ == "__main__":
    main()
