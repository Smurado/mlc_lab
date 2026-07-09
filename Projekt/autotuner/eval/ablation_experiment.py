#!/usr/bin/env python3
"""
D2 - Ablationstabelle: Was traegt jede Komponente zum Ergebnis bei?

One-Factor-at-a-Time: Basis ist der volle SA-Lauf (Warmstart + CostModel-Filter +
Kalibrierung). Jede Zeile schaltet GENAU EINE Komponente ab und misst, was passiert.
Zusaetzlich die uninformierte Random-Baseline. Fuer jede Konfiguration ueber mehrere
Seeds bei GLEICHER Trial-Zahl:
  - Median best-GFLOPS @ letztem Trial   (Endqualitaet)
  - Median Trials bis 90 % des eigenen Optimums   (Konvergenztempo)

Jede Zeile enthaelt den exakten Kommandozeilen-Aufruf zum Nachstellen (ein Seed).

Aufruf:  .venv/bin/python ablation_experiment.py
Steuerung ueber die Konstanten unten. Nur numpy noetig (kein pandas/scipy).
"""
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent          # eval/
COMPILER = DIR.parent / "src" / "teir_compiler"

# Gleicher Fall wie die Konvergenzstudie: echte Varianz, nicht Mikro.
EINSUM = "abc-bda-dc"
AXES = "a:32;b:32;c:32;d:32"
SEEDS = [1, 2, 3, 4, 5]
MAX_TRIALS = 20
SEARCH_OPT = "-O2"      # schnellerer Such-Compile (fair, s. Roadmap A2)

# Basis-Env (alle Komponenten AN). Eine Ablation ueberschreibt gezielt Schluessel.
BASE = {
    "TEIR_STRATEGY": "sa",
    "TEIR_WARMSTART": "1",
    "TEIR_COST_FILTER": "0.3",
    "TEIR_CALIBRATE": "1",
}

# (Label, Overrides, kurze Interpretationshilfe). None-Wert = Schluessel entfernen/Default.
ABLATIONS = [
    ("SA voll (Basis)",        {}),
    ("  − Warmstart",          {"TEIR_WARMSTART": "0"}),
    ("  − CostModel-Filter",   {"TEIR_COST_FILTER": "1.0"}),
    ("  − Kalibrierung",       {"TEIR_CALIBRATE": "0"}),
    ("Random (uninformiert)",  {"TEIR_STRATEGY": "random", "TEIR_WARMSTART": "0"}),
]


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
    path = Path(tmpdir) / "_ablation_input.csv"
    with open(path, "w") as f:
        f.write("name,tensors,axes,primitives,schedule,invokes,einsum\n")
        f.write(f'contraction_abl,"in0:f32;in1:f32;out:f32","{AXES}",'
                f'"zero;gemm","{schedule}","zero;gemm","{EINSUM}"\n')
    return path


def env_for(overrides):
    e = dict(BASE)
    e.update(overrides)
    return e


def run_one(csv_path, cfg_env, seed, log_path):
    env = {**os.environ,
           "TEIR_INPUT": str(csv_path),
           "TEIR_BACKEND": "scalar",
           "TEIR_MAX_TRIALS": str(MAX_TRIALS),
           "TEIR_TIME_BUDGET_MS": "600000",
           "TEIR_SEARCH_OPT": SEARCH_OPT,
           "TEIR_SEED": str(seed),
           "TEIR_TRIAL_LOG": str(log_path),
           **cfg_env}
    subprocess.run([str(COMPILER)], capture_output=True, text=True,
                   timeout=1200, env=env, cwd=str(DIR))


def read_best_curve(log_path, max_trials):
    best = {}
    try:
        with open(log_path) as f:
            for row in csv.DictReader(f):
                best[int(row["trial"])] = float(row["best_gflops"])
    except FileNotFoundError:
        return None
    if not best:
        return None
    curve, last = [], 0.0
    for t in range(1, max_trials + 1):
        last = best.get(t, last)
        curve.append(last)
    return curve


def trials_to_90(curve):
    """1-basierter Trial-Index, ab dem 90 % des eigenen Endoptimums erreicht sind."""
    final = curve[-1]
    if final <= 0:
        return float("nan")
    target = 0.9 * final
    for i, v in enumerate(curve, start=1):
        if v >= target:
            return i
    return len(curve)


def repro_cmd(cfg_env):
    parts = [f'{k}={v}' for k, v in env_for(cfg_env).items()]
    parts += [f'TEIR_INPUT=<case>.csv TEIR_MAX_TRIALS={MAX_TRIALS}',
              f'TEIR_SEARCH_OPT={SEARCH_OPT} TEIR_SEED=1']
    return " ".join(parts) + " ./teir_compiler"


def main():
    if not COMPILER.exists():
        print("[ERROR] teir_compiler nicht gebaut (make)."); sys.exit(1)

    print("=" * 74)
    print("  D2 - Ablation:  " + EINSUM + " @ " + AXES)
    print(f"  {len(SEEDS)} Seeds, {MAX_TRIALS} Trials, SEARCH_OPT={SEARCH_OPT}, "
          f"One-Factor-at-a-Time")
    print("=" * 74)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = make_csv(tmp)
        for label, overrides in ABLATIONS:
            cfg = env_for(overrides)
            finals, conv = [], []
            for seed in SEEDS:
                log = Path(tmp) / f"_abl_{seed}.csv"
                if log.exists():
                    log.unlink()
                print(f"  running {label.strip():22s} seed={seed} ...", flush=True)
                run_one(csv_path, cfg, seed, log)
                curve = read_best_curve(log, MAX_TRIALS)
                if curve is None:
                    print(f"    [WARN] kein Log ({label.strip()} seed={seed})")
                    continue
                finals.append(curve[-1])
                conv.append(trials_to_90(curve))
            if not finals:
                continue
            rows.append({
                "label": label,
                "median_gflops": float(np.median(finals)),
                "iqr_lo": float(np.percentile(finals, 25)),
                "iqr_hi": float(np.percentile(finals, 75)),
                "median_conv": float(np.median(conv)),
                "n": len(finals),
                "cmd": repro_cmd(overrides),
            })

    # ---- Tabelle ----
    print("\n" + "=" * 74)
    print("  ERGEBNIS (Median ueber Seeds)")
    print("=" * 74)
    hdr = f"  {'Konfiguration':24s} {'GFLOPS':>8s}  {'IQR':>13s}  {'Trials→90%':>10s}"
    print(hdr)
    print("  " + "-" * 70)
    base_g = rows[0]["median_gflops"] if rows else None
    for r in rows:
        delta = ""
        if base_g and r["label"] != rows[0]["label"]:
            d = 100.0 * (r["median_gflops"] - base_g) / base_g
            delta = f"  ({d:+.1f}% vs. Basis)"
        print(f"  {r['label']:24s} {r['median_gflops']:8.2f}  "
              f"{r['iqr_lo']:5.1f}-{r['iqr_hi']:<5.1f}  {r['median_conv']:10.1f}{delta}")

    print("\n  Nachstell-Aufrufe (ein Seed; <case>.csv = " + EINSUM + " @ " + AXES + "):")
    for r in rows:
        print(f"    # {r['label'].strip()}")
        print(f"    {r['cmd']}")

    print("\n  Lesehilfe:")
    print("    - GFLOPS   = Endqualitaet nach " + str(MAX_TRIALS) + " Trials (hoeher besser)")
    print("    - Trials→90% = wie schnell 90 % des EIGENEN Optimums erreicht sind (niedriger besser)")
    print("    - Ein negativer Δ heisst: diese Komponente WEGzunehmen schadet -> sie hilft.")


if __name__ == "__main__":
    main()
