#!/usr/bin/env python3
"""
Batch-Check: laeuft JEDE Zeile einer Input-CSV (z.B. input_gett.csv) einzeln durch den
teir_compiler und protokolliert, ob sie parst, kompiliert, validiert und laeuft.

Hintergrund: teir_compiler verarbeitet pro Aufruf nur die ERSTE Datenzeile (parser.cpp).
Dieses Skript schreibt daher pro Kontraktion eine 1-Zeilen-CSV und ruft den Compiler auf.

Default: TEIR_NO_AUTOTUNE=1 (nur Default-Schedule -> schnell; prueft Korrektheit+Lauf).
Mit --tune wird stattdessen echt getuned (langsam).
Mit --cap N werden alle Achsen-Extents auf <=N gedeckelt (Einsum-MUSTER bleibt gleich,
nur die Groesse schrumpft) -> schneller Korrektheits-/Durchlauf-Check aller Faelle,
weil der naive Default-Schedule auf den echten (grossen) Extents sonst austimet.

Aufruf:  .venv/bin/python run_all_gett.py [input_gett.csv] [--tune] [--cap 16]
"""
import csv
import os
import re
import subprocess
import sys
import time

DIR = os.path.dirname(os.path.abspath(__file__))          # eval/
AUTO = os.path.dirname(DIR)                                # autotuner/
COMPILER = os.path.join(AUTO, "src", "teir_compiler")
HEADER = "name,tensors,axes,primitives,schedule,invokes,einsum\n"
PER_CASE_TIMEOUT = 120  # s, Sicherheitsnetz


def cap_axes(line, cap):
    """Deckelt jede Achse (Feld 2, 'a:384;c:24;...') auf <= cap. Muster bleibt gleich."""
    fields = line.split(",")
    axpairs = fields[2].split(";")
    capped = []
    for p in axpairs:
        name, ext = p.split(":")
        capped.append(f"{name}:{min(int(ext), cap)}")
    fields[2] = ";".join(capped)
    return ",".join(fields)


def run_case(row_line, tune):
    tmp = os.path.join(DIR, "_gett_one.csv")
    with open(tmp, "w") as f:
        f.write(HEADER)
        f.write(row_line if row_line.endswith("\n") else row_line + "\n")
    env = {**os.environ, "TEIR_INPUT": tmp, "TEIR_BACKEND": "scalar"}
    if tune:
        env.update({"TEIR_STRATEGY": "sa", "TEIR_MAX_TRIALS": "15",
                    "TEIR_TIME_BUDGET_MS": "600000", "TEIR_SEARCH_OPT": "-O2"})
    else:
        env["TEIR_NO_AUTOTUNE"] = "1"
    t0 = time.time()
    try:
        r = subprocess.run([COMPILER], capture_output=True, text=True,
                           timeout=PER_CASE_TIMEOUT, env=env, cwd=os.path.dirname(COMPILER))
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        os.remove(tmp)
        return "TIMEOUT", None, time.time() - t0, ""
    os.remove(tmp)
    dt = time.time() - t0

    if "[SUCCESS]" in out:
        status = "PASS"
    elif "[SKIP]" in out:
        status = "SKIP"
    elif "[FAILED]" in out:
        status = "FAIL(validation)"
    elif "[ERROR]" in out:
        status = "ERROR"
    else:
        status = "ERROR(no-marker)"
    m = re.search(r"\[PERFORMANCE\]\s+[\d.eE+-]+\s+ms\s+\(([\d.eE+-]+)\s+GFLOPS\)", out)
    gflops = float(m.group(1)) if m else None
    err = ""
    if status.startswith(("FAIL", "ERROR")):
        em = re.search(r"\[ERROR\][^\n]*", out)
        err = em.group(0)[:80] if em else out.strip().splitlines()[-1][:80] if out.strip() else ""
    return status, gflops, dt, err


def main():
    argv = sys.argv[1:]
    tune = "--tune" in argv
    cap = None
    if "--cap" in argv:
        cap = int(argv[argv.index("--cap") + 1])
        argv = [a for i, a in enumerate(argv)
                if a != "--cap" and (i == 0 or argv[i - 1] != "--cap")]
    args = [a for a in argv if not a.startswith("--")]
    inp = args[0] if args else os.path.join(AUTO, "data", "input_gett.csv")

    if not os.path.exists(COMPILER):
        sys.exit("teir_compiler nicht gebaut (make).")
    with open(inp, newline="") as f:
        lines = f.read().splitlines()
    data = [l for l in lines[1:] if l.strip()]
    if cap is not None:
        data = [cap_axes(l, cap) for l in data]

    mode = "TUNED (SA, 15 Trials)" if tune else "Default-Schedule (NO_AUTOTUNE)"
    if cap is not None:
        mode += f", Extents<= {cap}"
    print("=" * 78)
    print(f"  Batch-Check: {len(data)} Kontraktionen aus {os.path.basename(inp)}  |  {mode}")
    print("=" * 78)
    print(f"  {'#':>2} {'Name':38s} {'Status':16s} {'GFLOPS':>8} {'s':>5}")
    print("  " + "-" * 74)

    counts, fails = {}, []
    for i, line in enumerate(data, 1):
        name = line.split(",", 1)[0]
        status, gf, dt, err = run_case(line, tune)
        counts[status.split("(")[0]] = counts.get(status.split("(")[0], 0) + 1
        gfs = f"{gf:8.1f}" if gf else "     -  "
        print(f"  {i:>2} {name[:38]:38s} {status:16s} {gfs} {dt:5.0f}", flush=True)
        if not status.startswith(("PASS", "SKIP")):
            fails.append((name, status, err))

    print("  " + "-" * 74)
    print("  Zusammenfassung:", "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if fails:
        print(f"\n  {len(fails)} nicht bestanden:")
        for name, status, err in fails:
            print(f"    - {name}: {status}  {err}")
    else:
        print("  Alle Kontraktionen laufen sauber durch. ✓")


if __name__ == "__main__":
    main()
