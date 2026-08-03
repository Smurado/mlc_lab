#!/usr/bin/env python3
"""A/B: TEIR_BENCH_ADAPTIVE=0 (alt, fixer Block 64) vs. =1 (adaptiv).

Test A - reine Methodendifferenz: TEIR_NO_AUTOTUNE=1, identischer Kernel, nur die
         Messschleife unterscheidet sich. Zwei Schedule-Varianten:
           seq = alle Achsen sequential  -> 1 Thread, saubere Isolation
           par = aeussere Achse parallel -> alle Kerne, so entstehen die echten Zahlen
Test B - Ende-zu-Ende: volle SA-Suche, Seed 42, 12 Trials (= GETT-Budget).

Fortschritt: jeder Lauf meldet START, alle HEARTBEAT s ein Lebenszeichen mit der
letzten relevanten Zeile des Compilers, danach das Ergebnis. Kopfzeile nennt die
Gesamtzahl der Laeufe, jede Zeile den Stand [i/N] + bisherige Gesamtzeit + ETA.
"""
import os, re, subprocess, time, tempfile
from pathlib import Path

AUTO = Path("/Users/justin/Developer/Repositories/mlc_lab/Projekt/autotuner")
COMPILER = AUTO / "src" / "teir_compiler"
HEADER = "name,tensors,axes,primitives,schedule,invokes,einsum\n"
HEARTBEAT = 15.0
REPS = 3
PERF = re.compile(r"\[PERFORMANCE\]\s+([\d.eE+-]+)\s+ms\s+\(([\d.eE+-]+)\s+GFLOPS\)")
INTERESTING = ("NEW BEST", "Trial", "[PERFORMANCE]", "[AUTOTUNER]", "[COSTMODEL]")

# VARIANTS je Fall: welche Schedule-Varianten in Test A gemessen werden.
# Der grosse Fall wird NICHT sequenziell gemessen: die all-sequentielle IR ist
# nur der STARTPUNKT der Suche (so steht sie in input_gett.csv), der Autotuner
# verwirft sie sofort. Real gemessen wurden in der GETT-Matrix parallele Configs
# (633 ms), nicht die 11+ s der sequentiellen Variante. Ein A/B darauf wuerde
# eine Konfiguration vergleichen, die in keinem Ergebnis vorkommt -- und 35 min
# kosten.
CASES = [
    ("klein abc-bda-dc @ 32",    "a:32;b:32;c:32;d:32",    "abc-bda-dc", "a",
     ["seq", "par"]),
    ("gross abc-dca-bd @ 384er", "a:384;c:376;b:24;d:384", "abc-dca-bd", "a",
     ["par"]),
]


def mkrow(axes, einsum, par_axis):
    out_idx, in0_idx, in1_idx = einsum.split("-")
    order = list(dict.fromkeys(out_idx)) + sorted(
        (set(in0_idx) | set(in1_idx)) - set(out_idx))
    sched = ";".join(f"{c}:{'parallel' if c == par_axis else 'sequential'}"
                     for c in order)
    return (f"c,in0:f32;in1:f32;out:f32,{axes},zero;gemm,{sched},zero;gemm,{einsum}")


def say(*a):
    print(*a, flush=True)


def hhmm(s):
    s = int(s)
    return f"{s//60:d}m{s%60:02d}s" if s < 3600 else f"{s//3600:d}h{(s%3600)//60:02d}m"


# --------------------------- Laufplan aufbauen ---------------------------
RUNS = []   # (phase, case_label, variant, mode, adaptive, row, env, timeout)
for label, axes, einsum, par, variants in CASES:
    for variant in variants:
        pax = par if variant == "par" else None
        for mode, adaptive in [("alt", "0"), ("adaptiv", "1")]:
            for i in range(REPS):
                RUNS.append(("A", label, variant, f"{mode} #{i+1}", adaptive,
                             mkrow(axes, einsum, pax),
                             {"TEIR_NO_AUTOTUNE": "1"}, 1800))
# Test B: harte Obergrenze 1800 s je Lauf. Wenn der alte Modus 12 Trials darin
# NICHT schafft, ist genau das das Ergebnis (= das Problem der GETT-Matrix).
for label, axes, einsum, par, _ in CASES:
    for mode, adaptive in [("alt", "0"), ("adaptiv", "1")]:
        RUNS.append(("B", label, "suche", mode, adaptive,
                     mkrow(axes, einsum, None),
                     {"TEIR_STRATEGY": "sa", "TEIR_SEED": "42",
                      "TEIR_MAX_TRIALS": "12", "TEIR_TIME_BUDGET_MS": "3600000"},
                     1800))

N = len(RUNS)
T0 = time.time()
say("=" * 84)
say(f"A/B-MESSUNG   {N} Laeufe gesamt   "
    f"(Test A: {sum(1 for r in RUNS if r[0]=='A')}, "
    f"Test B: {sum(1 for r in RUNS if r[0]=='B')})")
say("  Test A = NO_AUTOTUNE, identischer Kernel, nur Messschleife unterschiedlich")
say("  Test B = volle SA-Suche, Seed 42, 12 Trials")
say("  Teuer sind: Test A 'gross/alt' und Test B 'gross/alt' (fixer Block 64).")
say("=" * 84)

results = {}   # (phase, case, variant, mode_base) -> list of (gflops, wall)


def run_one(idx, phase, label, variant, mode, adaptive, row, extra, timeout):
    key = f"[{idx}/{N}]"
    say(f"\n{key} {phase} | {label} | {variant} | {mode} | "
        f"BENCH_ADAPTIVE={adaptive}  ... START   "
        f"(gesamt bisher {hhmm(time.time()-T0)})")
    with tempfile.TemporaryDirectory() as tmp:
        csvp = Path(tmp) / "case.csv"
        csvp.write_text(HEADER + row + "\n")
        logp = Path(tmp) / "out.txt"
        env = {**os.environ, "TEIR_INPUT": str(csvp), "TEIR_BACKEND": "scalar",
               "TEIR_SEARCH_OPT": "-O2", "TEIR_BENCH_ADAPTIVE": adaptive, **extra}
        t0 = time.time()
        with open(logp, "w") as fh:
            p = subprocess.Popen([str(COMPILER)], stdout=fh,
                                 stderr=subprocess.STDOUT, env=env,
                                 cwd=str(COMPILER.parent))
            nxt, timed_out = HEARTBEAT, False
            while True:
                try:
                    p.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    pass
                el = time.time() - t0
                if el >= nxt:
                    last = ""
                    try:
                        for ln in logp.read_text(errors="replace").splitlines():
                            if any(t in ln for t in INTERESTING):
                                last = ln.strip()[:88]
                    except OSError:
                        pass
                    say(f"      · laeuft [{el:6.0f}s] … {last}")
                    nxt += HEARTBEAT
                if el > timeout:
                    p.kill(); p.wait(); timed_out = True
                    break
        wall = time.time() - t0
        out = logp.read_text(errors="replace")
    m = PERF.search(out)
    gf = float(m.group(2)) if m else None
    ms = float(m.group(1)) if m else None
    if timed_out:
        say(f"      -> TIMEOUT nach {hhmm(wall)}")
    elif gf is None:
        say(f"      -> kein [PERFORMANCE] nach {hhmm(wall)}")
    else:
        say(f"      -> {gf:9.3f} GFLOPS   Kernel {ms:10.4f} ms   "
            f"Wandzeit {wall:7.1f}s")
    done, tot_el = idx, time.time() - T0
    say(f"      ({done}/{N} fertig, gesamt {hhmm(tot_el)}, "
        f"grobe ETA Rest {hhmm(tot_el/done*(N-done))})")
    return gf, wall


for i, (phase, label, variant, mode, adaptive, row, extra, to) in enumerate(RUNS, 1):
    gf, wall = run_one(i, phase, label, variant, mode, adaptive, row, extra, to)
    if gf is not None:
        results.setdefault((phase, label, variant, mode.split(" #")[0]),
                           []).append((gf, wall))

# --------------------------- Auswertung ---------------------------
say("")
say("=" * 84)
say("AUSWERTUNG")
say("=" * 84)
for phase in ("A", "B"):
    for label in (c[0] for c in CASES):
        for variant in ("seq", "par", "suche"):
            a = results.get((phase, label, variant, "alt"))
            n = results.get((phase, label, variant, "adaptiv"))
            if not a or not n:
                continue
            ga = sum(x[0] for x in a) / len(a); wa = sum(x[1] for x in a) / len(a)
            gn = sum(x[0] for x in n) / len(n); wn = sum(x[1] for x in n) / len(n)
            sa = f"{min(x[0] for x in a):.3f}-{max(x[0] for x in a):.3f}"
            sn = f"{min(x[0] for x in n):.3f}-{max(x[0] for x in n):.3f}"
            say(f"\nTest {phase} | {label} | {variant}")
            say(f"  alt     {ga:9.3f} GFLOPS  (Spanne {sa})  {wa:8.1f}s")
            say(f"  adaptiv {gn:9.3f} GFLOPS  (Spanne {sn})  {wn:8.1f}s")
            say(f"  DELTA   {100*(gn-ga)/ga:+7.2f} % GFLOPS   |   "
                f"x{wa/max(wn,1e-9):.1f} schneller ({wa:.0f}s -> {wn:.0f}s)")
say(f"\nGesamtdauer: {hhmm(time.time()-T0)}")
