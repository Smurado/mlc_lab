#!/usr/bin/env python3
"""Schritt 1: Trial-Budget fuer die vollstaendige GETT-Matrix bestimmen.

FRAGE: Wie viele Trials braucht die Suche, bis sie nicht mehr besser wird?
Ohne diese Zahl waere jedes Budget fuer den grossen Lauf willkuerlich -- und im
Bericht nicht begruendbar.

AUFBAU
  Faelle:      je ein Vertreter pro Achsenzahl (4/5/6/7). Die Konvergenz haengt
               am SUCHRAUM (Achsenzahl), nicht am Datenvolumen -- deshalb ist in
               jeder Klasse der GUENSTIGSTE Fall gewaehlt. Beim teuersten Fall
               gleicher Achsenzahl kostet ein Trial ~130 s statt ~1,5 s.
  Strategien:  alle drei (SA, GA, Random) -- das Budget soll fuer jede von
               ihnen reichen. Random konvergiert am langsamsten und bestimmt
               daher die Untergrenze.
  Trials:      100 (Default) -- das 8-fache des alten Budgets von 12.
  Seeds:       3 -- eine Budget-Entscheidung aus einem einzigen Lauf waere
               geraten, nicht gemessen.
  Zeitlimit:   600 s je Lauf. Random-Laeufe reissen es teils; solche Kurven
               werden als ABGEBROCHEN gekennzeichnet und aus den
               Konvergenzaussagen ausgeschlossen (nicht fortgeschrieben).
  Rest:        identisch zum geplanten grossen Lauf (Cost-Filter 0.3, Warmstart
               fuer SA / aus fuer Random, -O2-Suchkompilate, adaptiver Block).

AUSGABE
  results/trial_budget/curves.json  Rohdaten (best-so-far je Trial). Wird
                                    inkrementell geschrieben -> ein Abbruch
                                    verliert nur den laufenden Fall.
  results/trial_budget/REPORT.md    Auswertung + Empfehlung, als Beleg fuer die
                                    Budget-Entscheidung im Bericht.

Erneut ausfuehren ueberspringt bereits vermessene Kombinationen (--force = neu).
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AUTO = Path(__file__).resolve().parent.parent
COMPILER = AUTO / "src" / "teir_compiler"
GETT = AUTO / "data" / "input_gett.csv"
OUTDIR = AUTO / "results" / "trial_budget"
CURVES = OUTDIR / "curves.json"
REPORT = OUTDIR / "REPORT.md"

# Ueberschreibbar: PROBE_TRIALS=3 PROBE_SEEDS=1 PROBE_CASE=<name> PROBE_TIMEOUT=600
# 100 Trials = das 8-fache des alten Budgets (12), genug um ein Plateau zu sehen.
# Mehr waere bei Random unbezahlbar: dessen Trials kosten ~20x so viel wie die
# von SA, weil es gleichverteilt zieht und dabei katastrophale Schleifen-
# ordnungen trifft, waehrend SA warm startet und lokal bleibt.
MAX_TRIALS = int(os.environ.get("PROBE_TRIALS", "100"))
SEEDS = [int(s) for s in os.environ.get("PROBE_SEEDS", "1,2,3").split(",")]
STRATEGIES = ["sa", "ga", "random"]
COST_FILTER = "0.3"
SEARCH_OPT = "-O2"
PER_RUN_TIMEOUT = int(os.environ.get("PROBE_TIMEOUT", "600"))

# Je ein guenstiger Vertreter pro Achsenzahl.
PROBE_CASES = [
    "contraction_abc_dca_bd",        # 4 Achsen, Suchraum 1050
    "contraction_abcd_deca_be",      # 5 Achsen, Suchraum 4320
    "contraction_abcde_efbad_cf",    # 6 Achsen
    "contraction_abcdef_dega_gfbc",  # 7 Achsen
]

if os.environ.get("PROBE_CASE"):
    PROBE_CASES = [os.environ["PROBE_CASE"]]

FORCE = "--force" in sys.argv


def say(*a):
    print(*a, flush=True)


def hhmm(s):
    s = int(s)
    return f"{s//60}m{s%60:02d}s" if s < 3600 else f"{s//3600}h{(s%3600)//60:02d}m"


def load_rows():
    rows = {r["name"]: r for r in csv.DictReader(open(GETT))}
    hdr = ",".join(next(csv.reader(open(GETT))))
    return hdr, rows


def axis_count(row):
    return len(row["axes"].split(";"))


def volume(row):
    v = 1
    for a in row["axes"].split(";"):
        v *= int(a.split(":")[1])
    return v


def read_curve(log_path):
    """best_gflops je Trial. Laenge = hoechster tatsaechlich geloggter Trial --
    NICHT auf MAX_TRIALS aufgefuellt. Ein abgebrochener Lauf wuerde sonst eine
    flache Linie bis 200 zeigen und faelschlich wie konvergiert aussehen."""
    best = {}
    try:
        with open(log_path) as f:
            for row in csv.DictReader(f):
                best[int(row["trial"])] = float(row["best_gflops"])
    except OSError:
        return None
    if not best:
        return None
    curve, last = [], 0.0
    for t in range(1, max(best) + 1):   # Luecken INNERHALB des Bereichs fuellen
        last = best.get(t, last)
        curve.append(last)
    return curve


def run_one(hdr, row, strategy, seed):
    """Liefert (curve, wall, truncated). Bei Timeout wird das Trial-Log
    trotzdem ausgewertet -- eine Teilkurve ist brauchbar, sie muss nur als
    solche gekennzeichnet sein."""
    with tempfile.TemporaryDirectory() as tmp:
        csvp = Path(tmp) / "case.csv"
        csvp.write_text(hdr + "\n" + ",".join(row[k] for k in row) + "\n")
        log = Path(tmp) / "trials.csv"
        env = {**os.environ,
               "TEIR_INPUT": str(csvp), "TEIR_BACKEND": "scalar",
               "TEIR_STRATEGY": strategy, "TEIR_SEED": str(seed),
               "TEIR_MAX_TRIALS": str(MAX_TRIALS),
               "TEIR_TIME_BUDGET_MS": "36000000",
               "TEIR_COST_FILTER": COST_FILTER,
               "TEIR_SEARCH_OPT": SEARCH_OPT,
               "TEIR_BENCH_ADAPTIVE": "1",
               "TEIR_WARMSTART": "0" if strategy == "random" else "1",
               "TEIR_TRIAL_LOG": str(log)}
        t0 = time.time()
        truncated = False
        try:
            subprocess.run([str(COMPILER)], capture_output=True, text=True,
                           timeout=PER_RUN_TIMEOUT, env=env,
                           cwd=str(COMPILER.parent))
        except subprocess.TimeoutExpired:
            truncated = True
        curve = read_curve(log)
        if curve is not None and len(curve) < MAX_TRIALS:
            truncated = True
        return curve, time.time() - t0, truncated


# ----------------------------- Auswertung -----------------------------
def trials_to(curve, frac):
    """Erster Trial, ab dem best-so-far >= frac * Endwert ist."""
    if not curve or curve[-1] <= 0:
        return None
    target = frac * curve[-1]
    for i, v in enumerate(curve, 1):
        if v >= target:
            return i
    return len(curve)


def last_improvement(curve):
    last = 1
    for i in range(1, len(curve)):
        if curve[i] > curve[i - 1] * 1.0001:
            last = i + 1
    return last


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def main():
    if not COMPILER.exists():
        say("[ERROR] teir_compiler fehlt. Erst `make -C src` ausfuehren.")
        sys.exit(1)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    hdr, rows = load_rows()

    data = {}
    if CURVES.exists() and not FORCE:
        data = json.loads(CURVES.read_text())

    plan = [(c, s, sd) for c in PROBE_CASES for s in STRATEGIES for sd in SEEDS]
    todo = [p for p in plan if f"{p[0]}|{p[1]}|{p[2]}|n{MAX_TRIALS}" not in data]
    say("=" * 84)
    say(f"TRIAL-BUDGET-SONDIERUNG   {len(plan)} Laeufe "
        f"({len(plan)-len(todo)} bereits vorhanden, {len(todo)} offen)")
    say(f"  {MAX_TRIALS} Trials | Strategien {STRATEGIES} | Seeds {SEEDS}")
    say(f"  Cost-Filter {COST_FILTER} | Warmstart: SA=1 Random=0 | adaptiver Block")
    say("=" * 84)

    T0 = time.time()
    for i, (case, strat, seed) in enumerate(todo, 1):
        row = rows[case]
        # Budget im Schluessel: sonst wuerde ein Lauf mit anderer Trial-Zahl
        # (z.B. ein Rauchtest) faelschlich als "schon vermessen" gelten.
        key = f"{case}|{strat}|{seed}|n{MAX_TRIALS}"
        say(f"\n[{i}/{len(todo)}] {case} | {strat} | seed {seed} "
            f"({axis_count(row)} Achsen, Iter={volume(row):,})  ... START "
            f"(gesamt {hhmm(time.time()-T0)})")
        curve, wall, trunc = run_one(hdr, row, strat, seed)
        if curve is None:
            say(f"      -> kein Trial-Log nach {hhmm(wall)} (verworfen)")
            continue
        data[key] = {"case": case, "strategy": strat, "seed": seed,
                     "axes": axis_count(row), "volume": volume(row),
                     "wall_s": round(wall, 1), "trials": len(curve),
                     "truncated": trunc, "curve": curve}
        CURVES.write_text(json.dumps(data, indent=1))
        say(f"      -> {len(curve)}/{MAX_TRIALS} Trials"
            f"{'  [ABGEBROCHEN]' if trunc else ''} | "
            f"Endwert {curve[-1]:.2f} GFLOPS | "
            f"95 % ab Trial {trials_to(curve,0.95)} | "
            f"letzte Verbesserung Trial {last_improvement(curve)} | "
            f"{wall:.0f}s ({wall/len(curve):.2f}s/Trial)")

    say(f"\nMessungen fertig nach {hhmm(time.time()-T0)}. Schreibe Auswertung ...")
    write_report(data, rows)
    say(f"  -> {CURVES}")
    say(f"  -> {REPORT}")


def write_report(data, rows):
    entries = list(data.values())
    L = []
    L.append("# Trial-Budget der GETT-Matrix — Herleitung\n")
    L.append("Automatisch erzeugt von `eval/trial_budget_probe.py`. "
             "Rohdaten: `curves.json`.\n")
    L.append("## Frage\n")
    L.append("Wie viele Trials braucht die Suche, bis weitere Trials nichts mehr "
             "bringen? Der vollstaendige Suchraum ist nicht ausschoepfbar "
             "(1 050 Kandidaten bei 4 Achsen, 115 200 bei 7), das Budget muss "
             "also gemessen statt geraten werden.\n")
    L.append("## Aufbau\n")
    L.append(f"- {MAX_TRIALS} Trials, Seeds {SEEDS}, Strategien {STRATEGIES}\n")
    L.append("- Je ein Fall pro Achsenzahl (4/5/6/7). Die Konvergenz haengt am "
             "Suchraum, also an der Achsenzahl — deshalb pro Klasse der "
             "guenstigste Fall, was 200 Trials ueberhaupt bezahlbar macht.\n")
    L.append("- Random ist bewusst dabei: es konvergiert am langsamsten und "
             "bestimmt damit das Budget, das einen FAIREN Vergleich aller drei "
             "Strategien erlaubt.\n")
    L.append("- Uebrige Einstellungen identisch zum geplanten grossen Lauf "
             f"(Cost-Filter {COST_FILTER}, Warmstart SA=1/Random=0, "
             f"Suchkompilate {SEARCH_OPT}, adaptiver Messblock).\n")

    trunc = [e for e in entries if e.get("truncated")]
    if trunc:
        L.append(f"\n> **{len(trunc)} von {len(entries)} Laeufen wurden vor "
                 f"{MAX_TRIALS} Trials abgebrochen** (Zeitlimit "
                 f"{PER_RUN_TIMEOUT} s). Ihre Kurven sind nur bis zum "
                 "erreichten Trial ausgewertet, nicht fortgeschrieben. "
                 "Konvergenzaussagen daraus sind untere Schranken.\n")

    L.append("\n## Messung\n")
    L.append("| Fall | Achsen | Strategie | Trials | Endwert GFLOPS | 90 % ab | "
             "95 % ab | 99 % ab | letzte Verbesserung | s/Trial |\n")
    L.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for case in PROBE_CASES:
        for strat in STRATEGIES:
            es = [e for e in entries if e["case"] == case and e["strategy"] == strat]
            if not es:
                continue
            nt = median([len(e["curve"]) for e in es])
            mark = "*" if any(e.get("truncated") for e in es) else ""
            fin = median([e["curve"][-1] for e in es])
            t90 = median([trials_to(e["curve"], 0.90) for e in es])
            t95 = median([trials_to(e["curve"], 0.95) for e in es])
            t99 = median([trials_to(e["curve"], 0.99) for e in es])
            li = median([last_improvement(e["curve"]) for e in es])
            sp = median([e["wall_s"] / len(e["curve"]) for e in es])
            L.append(f"| {case} | {es[0]['axes']} | {strat} | {nt:g}{mark} | "
                     f"{fin:.2f} | {t90} | {t95} | {t99} | {li} | {sp:.2f} |\n")
    L.append("\nMediane über die Seeds; `*` = mindestens ein Lauf abgebrochen. "
             "„95 % ab“ = erster Trial, ab dem der beste bisher gefundene Wert "
             "mindestens 95 % des am Ende dieses Laufs erreichten Werts hat.\n")

    full = [e for e in entries if not e.get("truncated")]
    base = full if full else entries

    def mx(sel, frac):
        v = [trials_to(e["curve"], frac) for e in base if sel(e)]
        v = [x for x in v if x is not None]
        # Kein Wert heisst hier: alle passenden Laeufe wurden abgebrochen und
        # sind bewusst ausgeschlossen -- nicht "0 Trials noetig".
        return str(max(v)) if v else "— (alle betreffenden Laeufe abgebrochen)"

    li = [last_improvement(e["curve"]) for e in base]
    mli = max([x for x in li if x is not None], default=None)

    L.append("\n## Ableitung\n")
    if not full:
        L.append("_Alle Laeufe abgebrochen — die folgenden Zahlen sind untere "
                 "Schranken._\n")
    L.append(f"- 95 % des Endwerts: spaetestens ab Trial "
             f"**{mx(lambda e: True, 0.95)}** (alle Faelle/Strategien/Seeds).\n")
    L.append(f"- Nur Random (der langsamste Konvergierer): spaetestens Trial "
             f"**{mx(lambda e: e['strategy']=='random', 0.95)}**.\n")
    L.append(f"- 99 % des Endwerts: spaetestens ab Trial "
             f"**{mx(lambda e: True, 0.99)}**.\n")
    L.append(f"- Letzte beobachtete Verbesserung ueberhaupt: Trial **{mli}**.\n")

    L.append("\n## Kosten der vollen Matrix\n")
    L.append("Die Trial-Kosten haengen stark von der Strategie ab: SA startet "
             "warm und bleibt in brauchbaren Konfigurationen, Random zieht "
             "gleichverteilt und trifft dabei Schleifenordnungen mit "
             "katastrophalem Cache-Verhalten. Gemessene Mediane aus dieser "
             "Sondierung (guenstige Faelle!):\n\n")
    L.append("| Strategie | s/Trial (Median) |\n|---|---|\n")
    cost = {}
    for strat in STRATEGIES:
        es = [e for e in entries if e["strategy"] == strat]
        if es:
            cost[strat] = median([e["wall_s"] / len(e["curve"]) for e in es])
            L.append(f"| {strat} | {cost[strat]:.2f} |\n")
    L.append("\nDie 48 Faelle der Matrix reichen von ~1,3 Mrd. bis ~524 Mrd. "
             "Iterationen; zwoelf teure Faelle machen ~94 % der Rechenzeit aus. "
             "Analytische Hochrechnung (8 GFLOPS je Kernel, 1 s Compile) ergibt "
             "~25,1 min pro Trial ueber alle 48 Faelle:\n\n")
    L.append("| N | 48 Faelle x 3 Strategien (analytisch) |\n|---|---|\n")
    for n in (10, 20, 30, 50, 100):
        L.append(f"| {n} | {25.1*n*3/60:.1f} h |\n")
    if cost.get("random") and cost.get("sa"):
        L.append(f"\nDiese Hochrechnung unterstellt gleiche Kosten je Strategie. "
                 f"Gemessen kostet Random hier das "
                 f"{cost['random']/cost['sa']:.0f}-fache von SA — der reale "
                 f"Random-Anteil liegt also deutlich darueber.\n")
    L.append("\n## Entscheidung\n")
    L.append("_(von Hand zu ergaenzen, sobald Budget und Laufzeit abgewogen "
             "sind — die Messung oben liefert die Begruendung.)_\n")
    REPORT.write_text("".join(L))


if __name__ == "__main__":
    main()
