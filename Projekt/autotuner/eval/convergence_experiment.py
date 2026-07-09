#!/usr/bin/env python3
"""
D1 - Konvergenzkurven: SA vs. GA vs. Random bei FAIRER Trial-Zahl.

Fuehrt fuer jede Strategie mehrere Seeds aus, sammelt pro Trial das beste GFLOPS
bisher (aus dem TEIR_TRIAL_LOG des Autotuners) und plottet Median + IQR-Band ueber
den Trial-Index. Das macht den Kernbefund aus Phase B sichtbar: SA ist nicht
schneller im Bestfall, aber robuster (engeres Band) als Random.

Aufruf:  python3 convergence_experiment.py
Ausgabe: convergence.png  +  Konsolen-Zusammenfassung (Median @ letztem Trial).

Alles ueber die Konstanten unten steuerbar. Bewusst schlank gehalten (nur numpy +
matplotlib), damit es ohne pandas laeuft.
"""
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib fehlt. Installiere mit: pip install matplotlib numpy")
    sys.exit(1)

# --------------------------- Konfiguration ---------------------------
DIR = Path(__file__).resolve().parent          # eval/
AUTO = DIR.parent                              # autotuner/
COMPILER = AUTO / "src" / "teir_compiler"

# Testfall: 3D-Kontraktion mit echter Varianz (nicht Mikro, GEMM-artig genug).
EINSUM = "abc-bda-dc"
AXES = "a:32;b:32;c:32;d:32"

STRATEGIES = ["sa", "ga", "random"]   # alle bei gleicher Trial-Zahl
SEEDS = [1, 2, 3, 4, 5]
MAX_TRIALS = 20
COST_FILTER = "0.3"
SEARCH_OPT = "-O2"                    # schnellerer Such-Compile (fair, s. Roadmap A2)
WARMSTART = "1"                        # SA/GA warmgestartet; Random bleibt uninformiert
OUT_PNG = DIR / "notebooks" / "convergence.png"

LABELS = {"sa": "SA (warm)", "ga": "GA (warm)", "random": "Random"}
COLORS = {"sa": "#1f77b4", "ga": "#2ca02c", "random": "#d62728"}


def make_csv(tmpdir):
    out_idx, in0_idx, in1_idx = EINSUM.split("-")
    reduce_idx = (set(in0_idx) | set(in1_idx)) - set(out_idx)
    parts, seen = [], set()
    for c in out_idx:
        if c not in seen:
            parts.append(f"{c}:sequential"); seen.add(c)
    for c in sorted(reduce_idx):
        parts.append(f"{c}:sequential")
    schedule = ";".join(parts)
    path = Path(tmpdir) / "_conv_input.csv"
    with open(path, "w") as f:
        f.write("name,tensors,axes,primitives,schedule,invokes,einsum\n")
        f.write(f'contraction_conv,"in0:f32;in1:f32;out:f32","{AXES}",'
                f'"zero;gemm","{schedule}","zero;gemm","{EINSUM}"\n')
    return path


def run_one(csv_path, strategy, seed, log_path):
    env = {**os.environ,
           "TEIR_INPUT": str(csv_path), "TEIR_STRATEGY": strategy,
           "TEIR_BACKEND": "scalar", "TEIR_MAX_TRIALS": str(MAX_TRIALS),
           "TEIR_TIME_BUDGET_MS": "600000", "TEIR_COST_FILTER": COST_FILTER,
           "TEIR_SEARCH_OPT": SEARCH_OPT, "TEIR_WARMSTART": WARMSTART,
           "TEIR_SEED": str(seed), "TEIR_TRIAL_LOG": str(log_path)}
    if strategy == "random":
        env["TEIR_WARMSTART"] = "0"  # Baseline bewusst uninformiert
    subprocess.run([str(COMPILER)], capture_output=True, text=True,
                   timeout=1200, env=env, cwd=str(COMPILER.parent))


def read_best_curve(log_path, max_trials):
    """Liest best_gflops je Trial; auf max_trials aufgefuellt (letzter Wert fortgeschrieben)."""
    best = {}
    with open(log_path) as f:
        for row in csv.DictReader(f):
            best[int(row["trial"])] = float(row["best_gflops"])
    if not best:
        return None
    curve, last = [], 0.0
    for t in range(1, max_trials + 1):
        last = best.get(t, last)
        curve.append(last)
    return curve


def main():
    if not COMPILER.exists():
        print("[ERROR] teir_compiler nicht gefunden. Erst `make` ausfuehren.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = make_csv(tmp)
        # curves[strategy] = np.array [n_seeds, max_trials]
        curves = {s: [] for s in STRATEGIES}
        for strat in STRATEGIES:
            for seed in SEEDS:
                log = Path(tmp) / f"_log_{strat}_{seed}.csv"
                if log.exists():
                    log.unlink()
                print(f"  running {LABELS[strat]:12s} seed={seed} ...", flush=True)
                run_one(csv_path, strat, seed, log)
                curve = read_best_curve(log, MAX_TRIALS)
                if curve is None:
                    print(f"    [WARN] kein Log fuer {strat} seed={seed}")
                    continue
                curves[strat].append(curve)

    plt.figure(figsize=(8, 5))
    print("\n  Median best-GFLOPS @ letztem Trial:")
    for strat in STRATEGIES:
        data = np.array(curves[strat])
        if data.size == 0:
            continue
        x = np.arange(1, data.shape[1] + 1)
        med = np.median(data, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        plt.plot(x, med, label=LABELS[strat], color=COLORS[strat], linewidth=2)
        plt.fill_between(x, q25, q75, color=COLORS[strat], alpha=0.15)
        print(f"    {LABELS[strat]:12s} {med[-1]:6.2f} GFLOPS  "
              f"(IQR {q25[-1]:.2f}-{q75[-1]:.2f}, {data.shape[0]} Seeds)")

    plt.xlabel("Trial-Index (gleiches Budget fuer alle Strategien)")
    plt.ylabel("Bestes GFLOPS bisher")
    plt.title(f"Konvergenz: {EINSUM} @ {AXES.split(';')[0].split(':')[1]}  "
              f"({len(SEEDS)} Seeds, Median + IQR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130)
    print(f"\n  Abbildung gespeichert: {OUT_PNG}")


if __name__ == "__main__":
    main()
