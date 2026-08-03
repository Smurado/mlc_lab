#!/usr/bin/env python3
"""Kontrollversuch auf EINEM Kern: aendert die adaptive Blockgroesse den
gemessenen Wert?

OMP_NUM_THREADS=1 -> identisches Kompilat wie die parallelen Laeufe (die
#pragma-omp-Zeile bleibt drin), aber kein Thread-Scheduling, keine
Lastverteilung, kein Runtertakten unter Dauerlast. Was hier an Differenz
uebrigbleibt, ist die Messschleife und sonst nichts.

Groessenleiter statt eines Falls: die Kernelzeit laeuft von "weit unter dem
300-ms-Deckel" bis "Deckel greift". Damit sieht man nicht nur OB, sondern AB
WANN der fixe Block etwas veraendert.

TEIR_NO_AUTOTUNE=1 -> keine Suche, identischer Kernel in beiden Modi.
"""
import os, re, subprocess, time, tempfile
from pathlib import Path

AUTO = Path("/Users/justin/Developer/Repositories/mlc_lab/Projekt/autotuner")
COMPILER = AUTO / "src" / "teir_compiler"
HEADER = "name,tensors,axes,primitives,schedule,invokes,einsum\n"
PERF = re.compile(r"\[PERFORMANCE\]\s+([\d.eE+-]+)\s+ms\s+\(([\d.eE+-]+)\s+GFLOPS\)")
EINSUM = "abc-bda-dc"
# Kernelzeit einkernig ~0.2 ms (32) bis ~130 ms (160). Ab 64 ueberschreitet ein
# Block von 64 Aufrufen bereits den 1000-ms-Deckel der Endmessung -- der Effekt
# ist also ueber die ganze Leiter sichtbar. Groesser als 160 waere im ALTEN
# Modus unbezahlbar (5 x 64 Aufrufe je Wiederholung).
SIZES = [32, 64, 96, 160]
REPS = 5                        # einkernig billig -> mehr Wiederholungen


def mkrow(n):
    out_idx, in0_idx, in1_idx = EINSUM.split("-")
    order = list(dict.fromkeys(out_idx)) + sorted(
        (set(in0_idx) | set(in1_idx)) - set(out_idx))
    # aeussere Achse parallel markiert wie im Ernstfall; OMP_NUM_THREADS=1
    # sorgt fuer den einen Kern, das Kompilat bleibt identisch.
    sched = ";".join(f"{c}:{'parallel' if i == 0 else 'sequential'}"
                     for i, c in enumerate(order))
    axes = ";".join(f"{c}:{n}" for c in order)
    return f"c,in0:f32;in1:f32;out:f32,{axes},zero;gemm,{sched},zero;gemm,{EINSUM}"


def say(*a):
    print(*a, flush=True)


def run(n, adaptive, timeout=1800):
    with tempfile.TemporaryDirectory() as tmp:
        csvp = Path(tmp) / "case.csv"
        csvp.write_text(HEADER + mkrow(n) + "\n")
        env = {**os.environ, "TEIR_INPUT": str(csvp), "TEIR_BACKEND": "scalar",
               "TEIR_SEARCH_OPT": "-O2", "TEIR_NO_AUTOTUNE": "1",
               "TEIR_BENCH_ADAPTIVE": adaptive,
               "OMP_NUM_THREADS": "1"}
        t0 = time.time()
        try:
            p = subprocess.run([str(COMPILER)], capture_output=True, text=True,
                               timeout=timeout, env=env, cwd=str(COMPILER.parent))
        except subprocess.TimeoutExpired:
            return None, None, time.time() - t0
        wall = time.time() - t0
        m = PERF.search(p.stdout)
        return (float(m.group(1)), float(m.group(2)), wall) if m else (None, None, wall)


T0 = time.time()
N = len(SIZES) * 2 * REPS
say("=" * 84)
say(f"KONTROLLVERSUCH EIN KERN (OMP_NUM_THREADS=1)   {N} Laeufe")
say(f"  {EINSUM}, Groessen {SIZES}, {REPS} Wiederholungen je Modus")
say("  TEIR_NO_AUTOTUNE=1 -> identischer Kernel, nur die Messschleife variiert")
say("=" * 84)

i = 0
rows = []
for n in SIZES:
    say(f"\n--- Groesse {n} " + "-" * 60)
    acc = {}
    for mode, adaptive in [("alt (Block 64)", "0"), ("adaptiv", "1")]:
        gs, ws, kms = [], [], []
        for r in range(REPS):
            i += 1
            ms, gf, wall = run(n, adaptive)
            if gf is None:
                say(f"  [{i}/{N}] {mode:15s} #{r+1}: kein Ergebnis ({wall:.1f}s)")
                continue
            gs.append(gf); ws.append(wall); kms.append(ms)
            say(f"  [{i}/{N}] {mode:15s} #{r+1}: {gf:9.3f} GFLOPS   "
                f"Kernel {ms:11.5f} ms   Lauf {wall:7.1f}s")
        if gs:
            acc[mode] = (sum(gs)/len(gs), min(gs), max(gs),
                         sum(ws)/len(ws), sum(kms)/len(kms))
    if len(acc) == 2:
        ga, gamin, gamax, wa, kma = acc["alt (Block 64)"]
        gn, gnmin, gnmax, wn, kmn = acc["adaptiv"]
        overlap = max(gamin, gnmin) <= min(gamax, gnmax)
        say(f"  => alt     {ga:9.3f} GFLOPS  (Spanne {gamin:.3f}-{gamax:.3f})  "
            f"Lauf {wa:6.1f}s")
        say(f"  => adaptiv {gn:9.3f} GFLOPS  (Spanne {gnmin:.3f}-{gnmax:.3f})  "
            f"Lauf {wn:6.1f}s")
        say(f"  => Kernelzeit ~{kma:.4f} ms   DELTA {100*(gn-ga)/ga:+6.2f} %   "
            f"Spannen ueberlappen: {'JA (= Rauschen)' if overlap else 'NEIN'}   "
            f"Messung x{wa/max(wn,1e-9):.1f} schneller")
        rows.append((n, kma, ga, gn, 100*(gn-ga)/ga, overlap, wa, wn))

say("")
say("=" * 84)
say("ZUSAMMENFASSUNG (ein Kern)")
say("=" * 84)
say(f"{'Groesse':>8} {'Kernel ms':>11} {'alt GFLOPS':>11} {'adaptiv':>10} "
    f"{'Delta %':>9} {'Rauschen?':>10} {'Lauf alt':>9} {'Lauf neu':>9}")
for n, km, ga, gn, d, ov, wa, wn in rows:
    say(f"{n:>8} {km:>11.5f} {ga:>11.3f} {gn:>10.3f} {d:>+9.2f} "
        f"{('ja' if ov else 'NEIN'):>10} {wa:>8.1f}s {wn:>8.1f}s")
say(f"\nGesamtdauer: {time.time()-T0:.0f}s")
