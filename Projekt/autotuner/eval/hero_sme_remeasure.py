#!/usr/bin/env python3
"""Hero-Fall neu messen, nachdem das SME-Backend repariert wurde.

HINTERGRUND
Die bisherigen Hero-Zahlen (naiv 3 / TEIR 35 / TVM 505 / PyTorch 2371 GFLOPS)
wurden mit dem NEON-Backend erhoben. Das SME-Backend war defekt: es lieferte
falsche Ergebnisse und wurde von der Validierung verworfen, sodass in der Suche
kein einziger Kandidat gueltig war. Drei Fehler in src/codegen.cpp (falsche
ZA-Scheibenrichtung, w12 nicht als Operand uebergeben, smstart ohne
Clobber-Liste) sind behoben.

WICHTIG: Der Autotuner sucht das Backend NICHT -- `backend` ist eine feste
Option, kein Teil der TuningConfig. "TEIR mit SME" heisst also: mit
TEIR_BACKEND=sme starten. Das gehoert im Bericht so benannt.

WAS NEU GEMESSEN WIRD
Nur die TEIR-Varianten. TVM und PyTorch sind von der Aenderung nicht betroffen;
ihre Werte werden aus hero_results.json uebernommen, damit die Maschine nicht
unnoetig belegt wird.

Ergebnis: results/hero_sme.json + Konsolentabelle.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent
AUTO = DIR.parent
COMPILER = AUTO / "src" / "teir_compiler"
OUT_JSON = AUTO / "results" / "hero_sme.json"
OLD_JSON = DIR / "notebooks" / "hero_results.json"

N = 512                    # Hero-Fall: ab-ac-cb @ 512^3
TRIALS = 30
PERF = re.compile(r"\[PERFORMANCE\]\s+([\d.eE+-]+)\s+ms\s+\(([\d.eE+-]+)\s+GFLOPS\)")


def say(*a):
    print(*a, flush=True)


def run_teir(backend, tuned):
    """Liefert (GFLOPS, Wandzeit) oder (None, Wandzeit)."""
    with tempfile.TemporaryDirectory() as tmp:
        csvp = Path(tmp) / "hero.csv"
        csvp.write_text(
            "name,tensors,axes,primitives,schedule,invokes,einsum\n"
            f"hero,in0:f32;in1:f32;out:f32,a:{N};b:{N};c:{N},zero;gemm,"
            "a:sequential;b:sequential;c:sequential,zero;gemm,ab-ac-cb\n")
        env = {**os.environ,
               "TEIR_INPUT": str(csvp),
               "TEIR_BACKEND": backend,
               "TEIR_BENCH_ADAPTIVE": "1"}
        if tuned:
            env.update({"TEIR_STRATEGY": "sa", "TEIR_MAX_TRIALS": str(TRIALS),
                        "TEIR_TIME_BUDGET_MS": "600000", "TEIR_SEARCH_OPT": "-O2",
                        "TEIR_SEED": "42"})
        else:
            env["TEIR_NO_AUTOTUNE"] = "1"

        t0 = time.time()
        try:
            p = subprocess.run([str(COMPILER)], capture_output=True, text=True,
                               timeout=3600, env=env, cwd=str(COMPILER.parent))
        except subprocess.TimeoutExpired:
            return None, time.time() - t0
        wall = time.time() - t0
        m = PERF.search(p.stdout)
        if not m:
            # Validierung fehlgeschlagen o.ae. -- Grund mit ausgeben.
            fail = [l for l in p.stdout.splitlines() if "FAILED" in l or "Valide" in l]
            say("      Hinweis: " + (fail[-1].strip() if fail else "keine [PERFORMANCE]-Zeile"))
            return None, wall
        return float(m.group(2)), wall


def main():
    if not COMPILER.exists():
        say("[ERROR] teir_compiler fehlt. Erst `make -C src` ausfuehren.")
        sys.exit(1)

    say("=" * 72)
    say(f"Hero-Fall ab-ac-cb @ {N}^3 -- Neumessung nach dem SME-Fix")
    say(f"  SA, {TRIALS} Trials, Seed 42, adaptiver Messblock")
    say("=" * 72)

    variants = [
        ("naiv (ungetunt, scalar)", "scalar", False),
        ("TEIR getunt, scalar",     "scalar", True),
        ("TEIR getunt, NEON",       "neon",   True),
        ("TEIR getunt, SME",        "sme",    True),
    ]

    results = {}
    for label, backend, tuned in variants:
        say(f"\n{label} ...")
        g, wall = run_teir(backend, tuned)
        results[label] = g
        say(f"   -> {'%.1f GFLOPS' % g if g is not None else 'KEIN ERGEBNIS'}"
            f"   ({wall:.0f}s)")

    # TVM und PyTorch unveraendert uebernehmen.
    if OLD_JSON.exists():
        old = json.loads(OLD_JSON.read_text())
        for k, v in old.items():
            if "TVM" in k or "PyTorch" in k:
                results[k.replace("\n", " ") + " (uebernommen)"] = v

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=1, ensure_ascii=False))

    say("\n" + "=" * 72)
    say("ERGEBNIS")
    say("=" * 72)
    width = max(len(k) for k in results)
    for k, v in results.items():
        say(f"  {k:<{width}}  {('%10.1f GFLOPS' % v) if v is not None else '   kein Ergebnis'}")
    say(f"\nGespeichert: {OUT_JSON}")
    say("\nHinweis fuer den Bericht: Der Autotuner waehlt das Backend NICHT selbst.")
    say("Die SME-Zahl gilt fuer einen Lauf mit TEIR_BACKEND=sme und nur fuer")
    say("GEMM-foermige Kontraktionen mit durch 32 teilbaren M und N.")


if __name__ == "__main__":
    main()
