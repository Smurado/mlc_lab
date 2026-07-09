#!/usr/bin/env python3
"""
Orchestrator fuer die grosse GETT-Vergleichsmatrix (alle 48 Kontraktionen):
  Stufe 1  torch        (torch.einsum, .venv-Python)
  Stufe 2  TEIR SA       (eigener Autotuner, Simulated Annealing, warmgestartet)
  Stufe 3  TEIR GA       (Genetic Algorithm, warmgestartet)
  Stufe 4  TEIR Random   (uninformierte Baseline) -> Strategie-Vergleich ueber alle 48
  Stufe 5  TVM           (MetaSchedule, conda-Env tvm-bench)  <- laeuft als LETZTES

Waehrend eines laufenden Falls gibt die Konsole alle GETT_HEARTBEAT s (Default 30)
ein Lebenszeichen aus (inkl. letzter Worker-Zeile) -> man sieht, dass nichts haengt.

Stufen laufen SEQUENZIELL: torch + TEIR liefern frueh, der lange TVM-Lauf haengt
hinten dran ohne den Rest zu blockieren. Jede Stufe schreibt INKREMENTELL (pro Fall,
mit flush+fsync) nach gett_results/<engine>.csv und ist RESUMIERBAR (bereits erledigte
Faelle werden uebersprungen). Jeder Fall laeuft als Subprozess mit Timeout + Fehler-
Skip, damit ein Ausreisser den Nachtlauf nicht killt.

Pro Fall wird protokolliert: name, gflops, status, SEKUNDEN, TIMESTAMP. So sieht man
morgens genau, welche Faelle wie lange dauerten. Ausserdem zeigt gett_results/_current.txt
live den GERADE laufenden Fall (bei Haenger erkennbar). Zu langsame Faelle beim erneuten
Lauf gezielt rausnehmen:  --skip name1,name2

Aufruf (typisch, ueber Nacht):
    .venv/bin/python run_gett_matrix.py                       # alle Stufen
    .venv/bin/python run_gett_matrix.py --stages tvm          # nur TVM erneut
    .venv/bin/python run_gett_matrix.py --stages tvm --skip contraction_ab_ac_cb

Env-Knoepfe (Defaults in Klammern):
    GETT_TEIR_TRIALS (30)   GETT_TVM_TRIALS/TVM_TRIALS (200)   GETT_HEARTBEAT (30) [s]
    GETT_TORCH_TIMEOUT (300)  GETT_TEIR_TIMEOUT (1800)  GETT_TVM_TIMEOUT (1800)  [Sekunden]
    GETT_TVM_PY  (Pfad zum conda tvm-bench python)
"""
import datetime
import os
import re
import subprocess
import sys
import time

DIR = os.path.dirname(os.path.abspath(__file__))          # eval/
AUTO = os.path.dirname(DIR)                                # autotuner/
COMPILER = os.path.join(AUTO, "src", "teir_compiler")
VENV_PY = sys.executable  # der Orchestrator laeuft unter .venv (hat torch)
WORKER = os.path.join(DIR, "gett_bench.py")               # Worker liegt neben diesem Skript (eval/)
INPUT = os.environ.get("GETT_INPUT", os.path.join(AUTO, "data", "input_gett.csv"))
RESULTS = os.environ.get("GETT_RESULTS", os.path.join(AUTO, "results", "gett_results"))
HEADER1 = "name,tensors,axes,primitives,schedule,invokes,einsum\n"

TVM_PY = os.environ.get(
    "GETT_TVM_PY",
    "/opt/homebrew/Caskroom/miniconda/base/envs/tvm-bench/bin/python")
TEIR_TRIALS = os.environ.get("GETT_TEIR_TRIALS", "30")
TVM_TRIALS = os.environ.get("GETT_TVM_TRIALS", os.environ.get("TVM_TRIALS", "200"))
TO_TORCH = int(os.environ.get("GETT_TORCH_TIMEOUT", "300"))
TO_TEIR = int(os.environ.get("GETT_TEIR_TIMEOUT", "1800"))
TO_TVM = int(os.environ.get("GETT_TVM_TIMEOUT", "1800"))


def data_rows():
    with open(INPUT, newline="") as f:
        return [l for l in f.read().splitlines()[1:] if l.strip()]


def done_names(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {l.split(",")[0] for l in f.read().splitlines()[1:] if l.strip()}


def append(path, name, gflops, status, seconds, ts):
    """Schreibt eine Ergebniszeile mit Dauer + Timestamp, sofort geflusht."""
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("name,gflops,status,seconds,timestamp\n")
        gf = f"{gflops:.4f}" if gflops is not None else ""
        f.write(f"{name},{gf},{status},{seconds:.1f},{ts}\n")
        f.flush()
        os.fsync(f.fileno())


def parse_result(out):
    """-> (gflops|None, status)."""
    m = re.search(r"RESULT\s+([\d.eE+-]+)\s*$", out.strip(), re.M)
    if m:
        try:
            return float(m.group(1)), "ok"
        except ValueError:
            pass
    m = re.search(r"RESULT FAIL\s+(.*)", out)
    return None, (f"FAIL:{m.group(1)[:60]}" if m else "FAIL:no-result")


HEARTBEAT = int(os.environ.get("GETT_HEARTBEAT", "30"))  # s zwischen Lebenszeichen


def _last_line(path):
    try:
        with open(path, errors="replace") as f:
            lines = [l.strip() for l in f.read().splitlines() if l.strip()]
        return lines[-1][:90] if lines else ""
    except OSError:
        return ""


def run_capture(cmd, timeout, env, label, cwd=None):
    """Startet cmd, streamt Output in eine Temp-Datei und gibt ALLE HEARTBEAT s ein
    Lebenszeichen aus (inkl. letzter Worker-Zeile) -> Konsole zeigt, dass es laeuft.
    Liefert (Output-Text, timed_out)."""
    logf = os.path.join(RESULTS, "_worker.out")
    with open(logf, "w") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    start = time.time()
    next_hb = HEARTBEAT
    timed_out = False
    while True:
        try:
            p.wait(timeout=2)
            break
        except subprocess.TimeoutExpired:
            pass
        el = time.time() - start
        if el >= next_hb:
            print(f"        · {label} laeuft [{el:5.0f}s] … {_last_line(logf)}", flush=True)
            next_hb += HEARTBEAT
        if el > timeout:
            p.kill()
            p.wait()
            timed_out = True
            break
    with open(logf, errors="replace") as f:
        return f.read(), timed_out


def tvm_partial(out):
    """Bestes bis dahin gemessenes GFLOPS aus der MetaSchedule-Fortschrittstabelle
    (Spalte 'Speed (GFLOPS)'), damit ein TVM-Timeout eine Zahl liefert statt nichts.
    Tabellenzeile: '  0 | main | <FLOP> | 1 | 703.9113 | <lat> | ... '."""
    best = 0.0
    for line in out.splitlines():
        if "| main |" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5:
            try:
                best = max(best, float(parts[4]))
            except ValueError:
                pass  # 'N/A' (noch keine Messung)
    return (best, "partial-timeout") if best > 0 else (None, "TIMEOUT")


def run_worker(py, engine, row, timeout, label, extra_env=None):
    env = {**os.environ}
    if extra_env:
        env.update(extra_env)
    out, to = run_capture([py, WORKER, "--engine", engine, "--row", row], timeout, env, label)
    if to:
        return tvm_partial(out) if engine == "tvm" else (None, "TIMEOUT")
    return parse_result(out)


def run_teir(row, timeout, strategy, warmstart, label):
    tmp = os.path.join(DIR, "_gett_matrix_one.csv")
    with open(tmp, "w") as f:
        f.write(HEADER1)
        f.write(row + "\n")
    env = {**os.environ, "TEIR_INPUT": tmp, "TEIR_BACKEND": "scalar",
           "TEIR_STRATEGY": strategy, "TEIR_WARMSTART": warmstart,
           "TEIR_MAX_TRIALS": TEIR_TRIALS, "TEIR_TIME_BUDGET_MS": "600000",
           "TEIR_SEARCH_OPT": "-O2"}
    out, to = run_capture([COMPILER], timeout, env, label, cwd=os.path.dirname(COMPILER))
    os.path.exists(tmp) and os.remove(tmp)
    # Voll durchgelaufen -> praezise Endmessung.
    m = re.search(r"\[PERFORMANCE\]\s+[\d.eE+-]+\s+ms\s+\(([\d.eE+-]+)\s+GFLOPS\)", out)
    if m:
        return float(m.group(1)), "ok"
    # Sonst: bestes bis dahin gefundenes GFLOPS aus der Suche (grobe Such-Messung),
    # damit auch abgebrochene Faelle EINE Zahl liefern statt gar keiner.
    bests = re.findall(r"\[NEW BEST\].*?\(([\d.eE+-]+)\s+GFLOPS\)", out)
    if bests:
        return float(bests[-1]), ("partial-timeout" if to else "partial")
    if to:
        return None, "TIMEOUT"
    return None, "FAIL:no-success"


# Reihenfolge: torch (schnell) -> TEIR SA/GA/Random (mittel) -> TVM (lang, zuletzt).
STAGES = {
    "torch": dict(path="torch.csv",
                  fn=lambda row, lbl: run_worker(VENV_PY, "torch", row, TO_TORCH, lbl)),
    "teir_sa": dict(path="teir_sa.csv",
                    fn=lambda row, lbl: run_teir(row, TO_TEIR, "sa", "1", lbl)),
    "teir_ga": dict(path="teir_ga.csv",
                    fn=lambda row, lbl: run_teir(row, TO_TEIR, "ga", "1", lbl)),
    "teir_random": dict(path="teir_random.csv",
                        fn=lambda row, lbl: run_teir(row, TO_TEIR, "random", "0", lbl)),
    "tvm": dict(path="tvm.csv",
                fn=lambda row, lbl: run_worker(TVM_PY, "tvm", row, TO_TVM, lbl,
                                               {"TVM_TRIALS": TVM_TRIALS})),
}
DEFAULT_STAGES = ["torch", "teir_sa", "teir_ga", "teir_random", "tvm"]


def main():
    argv = sys.argv[1:]
    stages = list(DEFAULT_STAGES)
    if "--stages" in argv:
        stages = argv[argv.index("--stages") + 1].split(",")
    skip = set()
    if "--skip" in argv:
        skip = {s.strip() for s in argv[argv.index("--skip") + 1].split(",") if s.strip()}
    os.makedirs(RESULTS, exist_ok=True)
    rows = data_rows()

    def now():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"=== GETT-Matrix: {len(rows)} Faelle | Stufen: {stages} | Start {now()} ===")
    print(f"    Budgets: TEIR={TEIR_TRIALS} Trials, TVM={TVM_TRIALS} Trials")
    print(f"    Timeouts[s]: torch={TO_TORCH} teir={TO_TEIR} tvm={TO_TVM}")
    if skip:
        print(f"    Uebersprungen (--skip): {sorted(skip)}")
    print(flush=True)

    for stage in stages:
        cfg = STAGES[stage]
        path = os.path.join(RESULTS, cfg["path"])
        done = done_names(path)
        print(f"----- Stufe {stage.upper()} -> {cfg['path']} "
              f"({len(done)} bereits erledigt) @ {now()} -----", flush=True)
        t_stage = time.time()
        for i, row in enumerate(rows, 1):
            name = row.split(",")[0].strip()
            if name in done:
                continue
            if name in skip:
                append(path, name, None, "SKIPPED(manual)", 0.0, now())
                print(f"  [{stage}] {i:>2}/{len(rows)} {name[:38]:38s} SKIPPED (--skip)", flush=True)
                continue
            # "gerade in Arbeit"-Marker: zeigt morgens, welcher Fall haengt.
            with open(os.path.join(RESULTS, "_current.txt"), "w") as cf:
                cf.write(f"{stage} | {name} | started {now()}\n")
            print(f"  [{stage}] {i:>2}/{len(rows)} {name[:38]:38s} ... START @ {now()}", flush=True)
            t0 = time.time()
            gf, status = cfg["fn"](row, f"{stage}:{name[:28]}")
            dt = time.time() - t0
            append(path, name, gf, status, dt, now())
            tag = f"{gf:.1f} GFLOPS [{status}]" if gf is not None else status
            print(f"  [{stage}] {i:>2}/{len(rows)} {name[:38]:38s} {tag:28s} "
                  f"{dt:6.0f}s  DONE @ {now()}", flush=True)
        print(f"----- Stufe {stage.upper()} fertig in "
              f"{(time.time()-t_stage)/60:.1f} min @ {now()} -----\n", flush=True)

    # Aufraeumen des In-Arbeit-Markers am Ende.
    cur = os.path.join(RESULTS, "_current.txt")
    if os.path.exists(cur):
        with open(cur, "w") as cf:
            cf.write(f"IDLE | alle Stufen fertig | {now()}\n")
    print(f"=== Matrix komplett @ {now()}. Ergebnisse in gett_results/{{torch,teir,tvm}}.csv ===")


if __name__ == "__main__":
    main()
