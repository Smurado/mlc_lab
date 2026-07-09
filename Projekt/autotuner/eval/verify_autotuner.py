#!/usr/bin/env python3
"""
TEIR Autotuner Verifikations-Suite
Testet, ob der Autotuner gut ist, durch drei Test-Kategorien:

1. Speedup-Test: Default-Schedule vs. getunte Config (tut der Tuner was?)
2. Optimalitaets-Test: Exhaustive vs. Autotuner bei kleinem Suchraum
3. Strategie-Vergleich: SA vs. GA vs. Random bei gleichem Budget

Ausgabe: results-Tabelle + CSV-Datei
"""

import subprocess
import re
import csv
import os
import sys
import time
from pathlib import Path

AUTOTUNER_DIR = Path(__file__).resolve().parent            # eval/
COMPILER = AUTOTUNER_DIR.parent / "src" / "teir_compiler"
RESULTS_CSV = AUTOTUNER_DIR.parent / "results" / "verification_results.csv"

# Globaler Fortschritt-Tracker
_total_steps = 0
_done_steps = 0
_test_start = 0.0

def init_progress(total):
    global _total_steps, _done_steps
    _total_steps = total
    _done_steps = 0

def step_done():
    global _done_steps
    _done_steps += 1

def fmt_duration(s):
    if s < 60: return f"{s:.0f}s"
    m, s = divmod(int(s), 60)
    return f"{m}m{s:02d}s"

def progress_line(label, detail=""):
    """Zeigt Fortschritt als [3/12] mit Prozent + Vergangene Zeit."""
    global _test_start
    elapsed = time.time() - _test_start
    pct = (_done_steps / _total_steps * 100) if _total_steps > 0 else 0
    eta = (elapsed / _done_steps * (_total_steps - _done_steps)) if _done_steps > 0 else 0
    bar_len = 20
    filled = int(bar_len * _done_steps / _total_steps) if _total_steps > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(f"\r  [{_done_steps}/{_total_steps}] {bar} {pct:5.1f}% | "
                     f"{fmt_duration(elapsed)} elapsed | ~{fmt_duration(eta)} remaining | {label}")
    if detail:
        sys.stdout.write(f" — {detail}")
    sys.stdout.flush()

# Kontraktionen fuer Tests (aus input_gett.csv, ausgewaehlt nach Suchraum-Groesse/Diversitaet)
# Dimensionen bewusst klein gewaehlt, damit Trials schnell durchlaufen
TEST_CONTRACTIONS = [
    ("abc-bda-dc",          "a:8;b:8;c:8;d:8",          "small_perm"),    # 3 Out, 1 Red, Permutation
    ("abcd-aebf-dfce",      "a:4;b:4;c:4;d:4;e:4;f:4",  "2red_perm"),     # 4 Out, 2 Red
    ("abcdef-dega-gfbc",    "a:3;b:3;c:3;d:3;e:3;f:3;g:3", "6out"),       # 6 Out, 7 Achsen
    ("ab-ac-cb",            "a:16;b:16;c:16",           "gemm"),          # GEMM-form
    ("abc-adc-bd",          "a:8;b:8;c:8;d:8",          "3out_1red"),     # 3 Out, 1 Red
]

def make_csv(einsum, axes_str, name="verify"):
    """Erstellt eine temporaere CSV-Input-Datei fuer eine Kontraktion."""
    out_idx, in0_idx, in1_idx = einsum.split("-")
    all_idx = set(out_idx + in0_idx + in1_idx)
    reduce_idx = (set(in0_idx) | set(in1_idx)) - set(out_idx)

    schedule_parts = []
    seen = set()
    for c in out_idx:
        if c not in seen:
            schedule_parts.append(f"{c}:sequential")
            seen.add(c)
    for c in sorted(reduce_idx):
        schedule_parts.append(f"{c}:sequential")  # Reduktion nie parallel (Race)
    schedule_str = ";".join(schedule_parts)

    csv_path = AUTOTUNER_DIR / f"_verify_{name}.csv"
    with open(csv_path, "w") as f:
        f.write("name,tensors,axes,primitives,schedule,invokes,einsum\n")
        f.write(f'contraction_{name},"in0:f32;in1:f32;out:f32","{axes_str}",'
                f'"zero;gemm","{schedule_str}","zero;gemm","{einsum}"\n')
    return csv_path

def run_autotuner(csv_path, strategy="random", backend="scalar",
                  time_budget_ms=5000, cost_filter="0.3", extra_env=None):
    """Fuehrt den teir_compiler aus und parst die Ausgabe."""
    env = {
        "TEIR_INPUT": str(csv_path),
        "TEIR_STRATEGY": strategy,
        "TEIR_BACKEND": backend,
        "TEIR_TIME_BUDGET_MS": str(time_budget_ms),
        "TEIR_COST_FILTER": cost_filter,
    }
    if extra_env:
        env.update(extra_env)

    try:
        result = subprocess.run(
            [str(COMPILER)],
            capture_output=True, text=True, timeout=600, env={**os.environ, **env},
            cwd=str(COMPILER.parent)  # Scratch bleibt in src/
        )
    except subprocess.TimeoutExpired:
        return {"validation": "TIMEOUT", "best_gflops": None, "getestet": 0,
                "suchraum": None, "costmodel_kept": None, "valide": 0, "verworfen": 0,
                "best_runtime_ms": None, "best_split_axis": None, "best_split_factor": None,
                "best_loop_order": None, "best_parallel": None, "early_stop": None,
                "output": ""}
    output = result.stdout + result.stderr

    # Parse Ergebnisse
    parsed = {
        "output": output,
        "suchraum": None,
        "costmodel_kept": None,
        "getestet": None,
        "valide": None,
        "verworfen": None,
        "best_gflops": None,
        "best_runtime_ms": None,
        "best_split_axis": None,
        "best_split_factor": None,
        "best_loop_order": None,
        "best_parallel": None,
        "validation": None,
        "early_stop": None,
    }

    m = re.search(r"Suchraum generiert\.\s+(\d+)", output)
    if m: parsed["suchraum"] = int(m.group(1))

    m = re.search(r"COSTMODEL\] Vorfilter: Top \S+ \((\d+)", output)
    if m: parsed["costmodel_kept"] = int(m.group(1))

    m = re.search(r"Getestet:\s+(\d+)/(\d+)\s+\|\s+Valide:\s+(\d+)\s+\|\s+Verworfen.*?:\s+(\d+)", output)
    if m:
        parsed["getestet"] = int(m.group(1))
        parsed["valide"] = int(m.group(3))
        parsed["verworfen"] = int(m.group(4))

    m = re.search(r"Split Axis:\s+(\S+)", output)
    if m: parsed["best_split_axis"] = m.group(1)

    m = re.search(r"Split Factor:\s+(\d+)", output)
    if m: parsed["best_split_factor"] = int(m.group(1))

    m = re.search(r"Loop Order:\s+(.+)", output)
    if m: parsed["best_loop_order"] = m.group(1).strip()

    m = re.search(r"Parallel Axis:\s+(\S+)", output)
    if m: parsed["best_parallel"] = m.group(1)

    m = re.search(r"Performance:\s+([\d.e+-]+)\s+ms\s+\(([\d.e+-]+)\s+GFLOPS\)", output)
    if m:
        parsed["best_runtime_ms"] = float(m.group(1))
        parsed["best_gflops"] = float(m.group(2))

    if "[SUCCESS]" in output:
        parsed["validation"] = "PASS"
    elif "[FAILED]" in output:
        parsed["validation"] = "FAIL"
    elif "[SKIP]" in output:
        parsed["validation"] = "SKIP"

    if "EARLY STOP" in output:
        parsed["early_stop"] = True

    return parsed

def run_default_benchmark(csv_path, backend="scalar"):
    """Benchmarkt die echte Default-Schedule aus der CSV (kein Autotuning).
    Nutzt TEIR_NO_AUTOTUNE=1, um den Autotuner zu ueberspringen."""
    return run_autotuner(csv_path, strategy="random", backend=backend,
                         time_budget_ms=0, cost_filter="1.0",
                         extra_env={"TEIR_NO_AUTOTUNE": "1"})

def format_gflops(g):
    if g is None: return "N/A"
    if g < 0.001: return f"{g:.2e}"
    if g < 1: return f"{g:.4f}"
    return f"{g:.2f}"

def print_table(headers, rows):
    """Gibt eine einfache ASCII-Tabelle aus."""
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows if i < len(r)), default=0))
              for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(sep)
    print("| " + " | ".join(f"{str(h):<{w}}" for h, w in zip(headers, widths)) + " |")
    print(sep)
    for row in rows:
        print("| " + " | ".join(f"{str(c):<{w}}" for c, w in zip(row, widths)) + " |")
    print(sep)

def test_speedup():
    """Test 1: Default-Schedule vs. getunte Config."""
    global _test_start
    print("\n" + "=" * 70)
    print("  TEST 1: Speedup (Default vs. Autotuned)")
    print("=" * 70)
    print("  Frage: Verbessert der Autotuner die Performance ueberhaupt?\n")

    n_contractions = len(TEST_CONTRACTIONS)
    init_progress(n_contractions * 2)  # Default + Tuned pro Kontraktion
    _test_start = time.time()

    rows = []
    for einsum, axes, label in TEST_CONTRACTIONS:
        csv_path = make_csv(einsum, axes, f"speedup_{label}")

        progress_line(f"Default {label}", einsum)
        default = run_default_benchmark(csv_path)
        step_done()

        progress_line(f"Tuned   {label}", einsum)
        tuned = run_autotuner(csv_path, strategy="sa", time_budget_ms=5000, cost_filter="0.3")
        step_done()

        default_gflops = default["best_gflops"]
        tuned_gflops = tuned["best_gflops"]
        speedup = tuned_gflops / default_gflops if (default_gflops and default_gflops > 0) else None

        status = "OK" if tuned["validation"] == "PASS" else tuned["validation"]
        sys.stdout.write("\r" + " " * 100 + "\r")
        print(f"  [{label}] {einsum:20s} Default: {format_gflops(default_gflops):>10s}"
              f" -> Tuned: {format_gflops(tuned_gflops):>10s} GFLOPS"
              f"  Speedup: {f'{speedup:.2f}x' if speedup else 'N/A':>7s}  ({status})")

        rows.append([label, einsum, format_gflops(default_gflops), format_gflops(tuned_gflops),
                     f"{speedup:.2f}x" if speedup else "N/A", status])

    print()
    headers = ["Kontraktion", "Einsum", "Default GFLOPS", "Tuned GFLOPS", "Speedup", "Status"]
    print_table(headers, rows)
    return rows

def test_optimality():
    """Test 2: Exhaustive vs. Autotuner bei kleinem Suchraum."""
    global _test_start
    print("\n" + "=" * 70)
    print("  TEST 2: Optimalitaet (Exhaustive vs. Autotuner)")
    print("=" * 70)
    print("  Frage: Findet der Autotuner das echte Optimum?\n")

    test_cases = [
        ("abc-bda-dc", "a:8;b:8;c:8;d:8", "abc_bda_dc_8"),
        ("ab-ac-cb",   "a:8;b:8;c:8",     "ab_ac_cb_8"),
    ]

    init_progress(len(test_cases) * 2)  # Exhaustive + Autotuned pro Kontraktion
    _test_start = time.time()

    rows = []
    for einsum, axes, label in test_cases:
        csv_path = make_csv(einsum, axes, f"opt_{label}")

        progress_line(f"Exhaustive {label}", f"{einsum} (alle Trials, bis zu 120s)")
        exhaustive = run_autotuner(csv_path, strategy="random", time_budget_ms=120000, cost_filter="1.0")
        step_done()

        progress_line(f"Autotuned  {label}", f"{einsum} (5s Budget + CostModel)")
        autotuned = run_autotuner(csv_path, strategy="sa", time_budget_ms=5000, cost_filter="0.3")
        step_done()

        exhaustive_gflops = exhaustive["best_gflops"]
        autotuned_gflops = autotuned["best_gflops"]
        exhaustive_trials = exhaustive["getestet"]
        exhaustive_space = exhaustive["suchraum"]
        autotuned_trials = autotuned["getestet"]

        ratio = autotuned_gflops / exhaustive_gflops if (exhaustive_gflops and exhaustive_gflops > 0) else None

        sys.stdout.write("\r" + " " * 120 + "\r")
        print(f"  [{label}] {einsum:20s} Exhaustive: {format_gflops(exhaustive_gflops):>10s}"
              f" ({exhaustive_trials} trials) -> Autotuned: {format_gflops(autotuned_gflops):>10s}"
              f" ({autotuned_trials} trials)  Opt: {f'{ratio:.1%}' if ratio else 'N/A'}")

        rows.append([label, einsum,
                     str(exhaustive_space), str(exhaustive_trials),
                     format_gflops(exhaustive_gflops), format_gflops(autotuned_gflops),
                     f"{ratio:.2%}" if ratio else "N/A", str(autotuned_trials)])

    print()
    headers = ["Test", "Einsum", "Suchraum", "Exh.Trials", "Exh.GFLOPS", "Auto.GFLOPS", "Optimalitaet", "Auto.Trials"]
    print_table(headers, rows)
    return rows

def test_strategies():
    """Test 3: SA vs. GA vs. Random bei gleichem Zeit-Budget."""
    global _test_start
    print("\n" + "=" * 70)
    print("  TEST 3: Strategie-Vergleich (SA vs. GA vs. Random)")
    print("=" * 70)
    print("  Frage: Welcher Heuristik ist am besten (bei gleichem Budget)?\n")

    test_cases = [
        ("abc-bda-dc",       "a:16;b:16;c:16;d:16",  "medium_perm"),
        ("abcd-aebf-dfce",   "a:4;b:4;c:4;d:4;e:4;f:4", "2red"),
    ]

    strategies = ["random", "sa", "ga"]
    init_progress(len(test_cases) * len(strategies))
    _test_start = time.time()

    rows = []
    for einsum, axes, label in test_cases:
        csv_path = make_csv(einsum, axes, f"strat_{label}")
        strat_results = []

        for strat in strategies:
            progress_line(f"{strat.upper()} {label}", einsum)
            res = run_autotuner(csv_path, strategy=strat, time_budget_ms=5000, cost_filter="0.3")
            step_done()
            gflops = res["best_gflops"]
            trials = res["getestet"]
            status = res["validation"]

            sys.stdout.write("\r" + " " * 100 + "\r")
            print(f"  [{label:12s}] {strat.upper():6s} {einsum:20s}"
                  f"  {format_gflops(gflops):>10s} GFLOPS  ({trials} trials, {status})")
            strat_results.append((strat, gflops, trials, status))

        best_strat = max(strat_results, key=lambda x: x[1] if x[1] else 0)
        row = [label, einsum]
        for strat, gflops, trials, status in strat_results:
            row.append(f"{format_gflops(gflops)} ({trials})")
        row.append(best_strat[0].upper())
        rows.append(row)

    print()
    headers = ["Test", "Einsum", "Random", "SA", "GA", "Beste"]
    print_table(headers, rows)
    return rows

def main():
    if not COMPILER.exists():
        print("[ERROR] teir_compiler nicht gefunden. Bitte erst 'make' ausfuehren.")
        sys.exit(1)

    print("=" * 70)
    print("  TEIR Autotuner - Verifikations-Suite")
    print("=" * 70)
    print(f"  Hardware: {subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()}")
    print(f"  Compiler: {Path(__file__).parent}")

    # Schaetzung der Gesamtlaufzeit
    n_speedup = len(TEST_CONTRACTIONS) * 2
    n_opt = 2 * 2
    n_strat = 2 * 3
    n_total = n_speedup + n_opt + n_strat
    print(f"\n  Tests: {n_total} Einzellaeufe")
    print(f"    Test 1 (Speedup):    {n_speedup} Laeufe  (~3-8 Min)")
    print(f"    Test 2 (Optimalitaet): {n_opt} Laeufe  (~5-15 Min, exhaustive kann dauern)")
    print(f"    Test 3 (Strategien):  {n_strat} Laeufe  (~3-8 Min)")
    print(f"  Geschaetzte Dauer: ~10-30 Min (je nach Hardware & Suchraum)")
    print()

    all_rows = []
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "label", "einsum", "metric1", "metric2", "metric3", "status"])

    t0 = time.time()

    # Test 1: Speedup
    print("\n" + "─" * 70)
    print(f"  TEST 1/3 startet (Speedup)")
    print("─" * 70)
    t1 = time.time()
    speedup_rows = test_speedup()
    print(f"  Test 1 Dauer: {fmt_duration(time.time() - t1)}")
    for row in speedup_rows:
        all_rows.append(("speedup", row))

    # Test 2: Optimalitaet
    print("\n" + "─" * 70)
    print(f"  TEST 2/3 startet (Optimalitaet)")
    print("─" * 70)
    t2 = time.time()
    optimality_rows = test_optimality()
    print(f"  Test 2 Dauer: {fmt_duration(time.time() - t2)}")
    for row in optimality_rows:
        all_rows.append(("optimality", row))

    # Test 3: Strategien
    print("\n" + "─" * 70)
    print(f"  TEST 3/3 startet (Strategien)")
    print("─" * 70)
    t3 = time.time()
    strategy_rows = test_strategies()
    print(f"  Test 3 Dauer: {fmt_duration(time.time() - t3)}")
    for row in strategy_rows:
        all_rows.append(("strategies", row))

    elapsed = time.time() - t0

    # Zusammenfassung
    print("\n" + "=" * 70)
    print("  ZUSAMMENFASSUNG")
    print("=" * 70)
    print(f"  Gesamtlaufzeit: {fmt_duration(elapsed)}")
    print(f"  Ergebnisse in: {RESULTS_CSV.name}")

    # Aufraeumen
    for f in AUTOTUNER_DIR.glob("_verify_*.csv"):
        f.unlink()

    print("\n  Done.")

if __name__ == "__main__":
    main()
