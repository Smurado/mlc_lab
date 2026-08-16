"""Erzeugt die Abbildung für Abschnitt 5.3 des Projektberichts.

Datenquelle: Projekt/autotuner/eval/notebooks/convergence_data.json, je fünf
Startwerte pro Strategie. Dargestellt ist der Median des bis dahin besten
Ergebnisses; das schattierte Band zeigt Minimum und Maximum über die fünf
Läufe und damit, wie stark eine Strategie zwischen Startwerten schwankt.
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/Users/justin/Developer/Repositories/mlc_lab/Projekt/autotuner/"
            "eval/notebooks/convergence_data.json")
OUT = Path("/Users/justin/Developer/Repositories/mlc_lab/docs/projekt/konvergenz.png")

STYLE = {
    "sa":     ("Simulated Annealing", "#08519c", "-"),
    "ga":     ("Genetischer Algorithmus", "#41ab5d", "-"),
    "random": ("Zufallssuche", "#d95f02", "--"),
}

runs = json.loads(DATA.read_text())
fig, ax = plt.subplots(figsize=(9, 5))

for key, (label, color, ls) in STYLE.items():
    series = runs[key]
    n = min(len(r) for r in series)
    x = range(1, n + 1)
    med = [st.median(r[i] for r in series) for i in range(n)]
    lo = [min(r[i] for r in series) for i in range(n)]
    hi = [max(r[i] for r in series) for i in range(n)]

    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, med, ls, color=color, linewidth=2.2, label=label)

ax.set_xlabel("Trial")
ax.set_ylabel("bestes Ergebnis bis dahin [GFLOPS]")
ax.set_title("Konvergenz der Suchstrategien (Median aus 5 Startwerten, "
             "Band = Min bis Max)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
ax.set_xlim(1, min(len(r) for s in runs.values() for r in s))
ax.set_xticks(range(2, 21, 2))   # Trials sind ganzzahlig
ax.set_ylim(0, 47)               # Luft nach oben fuer die Warmstart-Notiz

# Der Warmstart ist die Kernaussage: SA und GA liegen ab Trial 1 oben. Text
# und Pfeil liegen oberhalb der Kurven, damit sie keine Linie kreuzen.
ax.annotate("Warmstart: SA und GA starten\nbereits am Cost-Modell-Optimum",
            xy=(1.4, 40.8), xytext=(2.6, 44), fontsize=10.5, color="#333333",
            va="center",
            arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1})

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("geschrieben:", OUT)
