#!/usr/bin/env python3
"""
C3 - CostModel-Kalibrierung: Wie gut ORDNET das Modell die Configs?

Fragestellung: Bringt die Start-Kalibrierung (peak_gflops + Thread-Overhead auf
der Zielmaschine gemessen, statt hart kodiert) ein besseres RANKING als die alten
Konstanten? Das Modell muss nicht exakt sein — es muss nur die Reihenfolge treffen.

Methode: gleicher realer Fall, breite Random-Stichprobe (CostModel-Filter AUS, damit
der ganze Suchraum gesampelt wird), einmal kalibriert (TEIR_CALIBRATE=1) und einmal
mit den alten Defaults (TEIR_CALIBRATE=0). Fuer jeden Lauf die Spearman-Rang-
Korrelation zwischen geschaetzter Zeit (cost_estimate_ms) und gemessener Zeit
(runtime_ms). Positiv = das Modell ordnet richtig (kleine Schaetzung -> kleine Zeit).

Aufruf:  .venv/bin/python costmodel_correlation.py
Kein scipy noetig — Spearman = Pearson auf mittleren Raengen (Ties gemittelt).
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent          # eval/
COMPILER = DIR.parent / "src" / "teir_compiler"
RESULTS = DIR / "autotuner_results.csv"          # teir schreibt es ins cwd (=DIR), s.u.

# Realer Fall mit echter Varianz (nicht Mikro), s. convergence_experiment.py.
EINSUM = "abc-bda-dc"
AXES = "a:32;b:32;c:32;d:32"
MAX_TRIALS = 80          # breite Stichprobe fuer eine belastbare Korrelation
SEED = 7


def make_csv(path):
    out_idx, in0_idx, in1_idx = EINSUM.split("-")
    reduce_idx = (set(in0_idx) | set(in1_idx)) - set(out_idx)
    parts, seen = [], set()
    for c in out_idx:
        if c not in seen:
            parts.append(f"{c}:sequential"); seen.add(c)
    for c in sorted(reduce_idx):
        parts.append(f"{c}:sequential")
    schedule = ";".join(parts)
    with open(path, "w") as f:
        f.write("name,tensors,axes,primitives,schedule,invokes,einsum\n")
        f.write(f'contraction_c3,"in0:f32;in1:f32;out:f32","{AXES}",'
                f'"zero;gemm","{schedule}","zero;gemm","{EINSUM}"\n')


def rankdata(a):
    """Mittlere Raenge (Ties gemittelt) — wie scipy.stats.rankdata('average')."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-basiert
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def run(calibrate):
    csv_in = DIR / "_c3_corr_input.csv"
    make_csv(csv_in)
    if RESULTS.exists():
        RESULTS.unlink()
    env = {**os.environ,
           "TEIR_INPUT": str(csv_in),
           "TEIR_STRATEGY": "random",     # breit sampeln, uninformiert
           "TEIR_BACKEND": "scalar",
           "TEIR_MAX_TRIALS": str(MAX_TRIALS),
           "TEIR_TIME_BUDGET_MS": "600000",
           "TEIR_COST_FILTER": "1.0",      # Filter AUS -> ganzer Raum
           "TEIR_WARMSTART": "0",
           "TEIR_SEED": str(SEED),
           "TEIR_CALIBRATE": "1" if calibrate else "0"}
    proc = subprocess.run([str(COMPILER)], capture_output=True, text=True,
                          timeout=1200, env=env, cwd=str(DIR))
    peak = overhead = None
    for line in (proc.stdout + proc.stderr).splitlines():
        if "peak=" in line:
            # "... peak=87.71 GFLOPS ... thread_overhead=0.0446 ms ..."
            try:
                peak = float(line.split("peak=")[1].split()[0])
                overhead = float(line.split("thread_overhead=")[1].split()[0])
            except (IndexError, ValueError):
                pass
    est, meas = [], []
    with open(RESULTS, newline="") as f:
        for row in csv.DictReader(f):
            e = float(row["cost_estimate_ms"]); m = float(row["runtime_ms"])
            if np.isfinite(e) and np.isfinite(m):
                est.append(e); meas.append(m)
    csv_in.unlink(missing_ok=True)
    return np.array(est), np.array(meas), peak, overhead


def report(tag, est, meas, peak, overhead):
    rho = spearman(est, meas)
    # Kontroll-Metrik: haette das Modell die Top-5 (schnellste gemessen) erkannt?
    n = len(meas)
    k = min(5, n)
    best_meas = set(np.argsort(meas)[:k])
    best_est = set(np.argsort(est)[:k])
    overlap = len(best_meas & best_est)
    print(f"  [{tag}]  n={n}  peak={peak}  overhead={overhead}")
    print(f"      Spearman(est, gemessen) = {rho:+.3f}   "
          f"(1.0 = perfektes Ranking, 0 = Zufall)")
    print(f"      Top-{k}-Ueberlappung (schnellste geschaetzt vs. gemessen): "
          f"{overlap}/{k}")
    return rho


def main():
    if not COMPILER.exists():
        print("[ERROR] teir_compiler nicht gebaut (make)."); sys.exit(1)
    print("=" * 70)
    print("  C3 - Ordnet das kalibrierte CostModel besser als die Defaults?")
    print(f"  Fall: {EINSUM} @ {AXES} | Random, Filter AUS, {MAX_TRIALS} Trials")
    print("=" * 70)

    e1, m1, p1, o1 = run(calibrate=True)
    rho_cal = report("KALIBRIERT ", e1, m1, p1, o1)
    print()
    e0, m0, p0, o0 = run(calibrate=False)
    rho_def = report("DEFAULT    ", e0, m0, p0, o0)

    print("\n" + "-" * 70)
    print(f"  Spearman kalibriert: {rho_cal:+.3f}   |   Default: {rho_def:+.3f}")
    delta = rho_cal - rho_def
    if np.isnan(rho_cal) or np.isnan(rho_def):
        verdict = "nicht bewertbar (zu wenige Datenpunkte)"
    elif delta > 0.02:
        verdict = f"Kalibrierung VERBESSERT das Ranking (+{delta:.3f})"
    elif delta < -0.02:
        verdict = f"Kalibrierung VERSCHLECHTERT das Ranking ({delta:.3f})"
    else:
        verdict = f"kein relevanter Unterschied im Ranking (Δ={delta:+.3f})"
    print(f"  Urteil: {verdict}")
    print(f"  Akzeptanz C3 (Rang-Korr. > 0 auf einem realen Fall): "
          f"{'ERFUELLT' if (rho_cal > 0 or rho_def > 0) else 'NICHT erfuellt'}")
    print("-" * 70)


if __name__ == "__main__":
    main()
