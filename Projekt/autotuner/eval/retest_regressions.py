#!/usr/bin/env python3
"""
Fokussierter Re-Test der 2 Regressionen aus der Verifikations-Suite.
"""
import subprocess, re, os, sys, time
from pathlib import Path

DIR = Path(__file__).resolve().parent          # eval/
AUTO = DIR.parent                              # autotuner/
COMPILER = AUTO / "src" / "teir_compiler"       # Binary liegt in src/

REGRESSIONS = [
    ("abcdef-dega-gfbc", "a:3;b:3;c:3;d:3;e:3;f:3;g:3", "6out"),
    ("ab-ac-cb",         "a:16;b:16;c:16",              "gemm"),
]

def make_csv(einsum, axes_str, name):
    out_idx, in0_idx, in1_idx = einsum.split("-")
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
    csv_path = DIR / f"_retest_{name}.csv"
    with open(csv_path, "w") as f:
        f.write("name,tensors,axes,primitives,schedule,invokes,einsum\n")
        f.write(f'contraction_{name},"in0:f32;in1:f32;out:f32","{axes_str}",'
                f'"zero;gemm","{schedule_str}","zero;gemm","{einsum}"\n')
    return csv_path

def run(csv_path, strategy="sa", time_budget_ms=5000, cost_filter="0.3", no_autotune=False):
    env = {**os.environ,
           "TEIR_INPUT": str(csv_path), "TEIR_STRATEGY": strategy,
           "TEIR_BACKEND": "scalar", "TEIR_TIME_BUDGET_MS": str(time_budget_ms),
           "TEIR_COST_FILTER": cost_filter}
    if no_autotune:
        env["TEIR_NO_AUTOTUNE"] = "1"
    try:
        r = subprocess.run([str(COMPILER)], capture_output=True, text=True, timeout=600, env=env,
                           cwd=str(COMPILER.parent))  # Scratch (_trial_*, generated_kernel) bleibt in src/
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    out = r.stdout + r.stderr
    m = re.search(r"Performance:\s+([\d.e+-]+)\s+ms\s+\(([\d.e+-]+)\s+GFLOPS\)", out)
    if not m:
        m = re.search(r"\[PERFORMANCE\]\s+([\d.e+-]+)\s+ms\s+\(([\d.e+-]+)\s+GFLOPS\)", out)
    gflops = float(m.group(2)) if m else None
    status = "PASS" if "[SUCCESS]" in out else "FAIL"
    return gflops, status

def fmt(g):
    if g is None: return "N/A"
    if g < 0.001: return f"{g:.2e}"
    if g < 1: return f"{g:.4f}"
    return f"{g:.2f}"

def main():
    if not COMPILER.exists():
        print("[ERROR] teir_compiler nicht gefunden."); sys.exit(1)

    print("=" * 70)
    print("  Re-Test: 2 Regressionen (6out + GEMM)")
    print("=" * 70)
    print(f"  Fixes: TEIR_NO_AUTOTUNE (echte Default-Schedule) + CostModel parallel penalty\n")

    for einsum, axes, label in REGRESSIONS:
        csv_path = make_csv(einsum, axes, label)

        print(f"  [{label}] {einsum}")
        t0 = time.time()
        print(f"    Default (echte CSV-Schedule) ...", end=" ", flush=True)
        default_g, default_s = run(csv_path, no_autotune=True)
        print(f"{fmt(default_g)} GFLOPS ({default_s})  [{time.time()-t0:.0f}s]")

        t1 = time.time()
        print(f"    Tuned SA (5s, CostModel 30%)  ...", end=" ", flush=True)
        tuned_g, tuned_s = run(csv_path, strategy="sa", time_budget_ms=5000, cost_filter="0.3")
        print(f"{fmt(tuned_g)} GFLOPS ({tuned_s})  [{time.time()-t1:.0f}s]")

        t2 = time.time()
        print(f"    Tuned GA (5s, CostModel 30%)  ...", end=" ", flush=True)
        ga_g, ga_s = run(csv_path, strategy="ga", time_budget_ms=5000, cost_filter="0.3")
        print(f"{fmt(ga_g)} GFLOPS ({ga_s})  [{time.time()-t2:.0f}s]")

        t3 = time.time()
        print(f"    Tuned Random (5s, CostModel 30%) ...", end=" ", flush=True)
        rand_g, rand_s = run(csv_path, strategy="random", time_budget_ms=5000, cost_filter="0.3")
        print(f"{fmt(rand_g)} GFLOPS ({rand_s})  [{time.time()-t3:.0f}s]")

        speedup_sa = tuned_g / default_g if (tuned_g and default_g and default_g > 0) else None
        speedup_ga = ga_g / default_g if (ga_g and default_g and default_g > 0) else None
        speedup_rand = rand_g / default_g if (rand_g and default_g and default_g > 0) else None

        best = max([("SA", tuned_g), ("GA", ga_g), ("Random", rand_g)], key=lambda x: x[1] if x[1] else 0)
        best_speedup = best[1] / default_g if (best[1] and default_g and default_g > 0) else None

        print(f"\n    Summary:")
        print(f"      Default:  {fmt(default_g)} GFLOPS")
        print(f"      SA:       {fmt(tuned_g)} GFLOPS  ({f'{speedup_sa:.2f}x' if speedup_sa else 'N/A'})")
        print(f"      GA:       {fmt(ga_g)} GFLOPS  ({f'{speedup_ga:.2f}x' if speedup_ga else 'N/A'})")
        print(f"      Random:   {fmt(rand_g)} GFLOPS  ({f'{speedup_rand:.2f}x' if speedup_rand else 'N/A'})")
        print(f"      Beste:    {best[0]} ({fmt(best[1])} GFLOPS, {f'{best_speedup:.2f}x' if best_speedup else 'N/A'})")
        print()

        csv_path.unlink()

    print("  Done.")

if __name__ == "__main__":
    main()
