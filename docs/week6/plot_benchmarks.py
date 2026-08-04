"""Erzeugt die Abbildung für docs/week6 aus den gemessenen Werten.

Datenquelle: Ausgabe von week6/main (27 GEMM-Settings, 12 Unary-Settings),
gemessen am 2026-08-04 auf Apple M4 Max.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (M, N) -> {K: GFLOPS}
GEMM = {
    (64, 64):   {64: 1323.35, 128: 1592.74, 512: 1844.87},
    (64, 128):  {64: 1335.01, 128: 1563.36, 512: 1890.93},
    (64, 512):  {64: 1326.51, 128: 1560.91, 512: 1886.65},
    (128, 64):  {64: 1325.45, 128: 1573.69, 512: 1888.78},
    (128, 128): {64: 1296.58, 128: 1605.01, 512: 1882.38},
    (128, 512): {64: 1308.46, 128: 1605.01, 512: 1829.12},
    (512, 64):  {64: 1139.12, 128: 1433.79, 512: 1843.51},
    (512, 128): {64: 1169.50, 128: 1440.81, 512: 1848.87},
    (512, 512): {64: 1109.05, 128: 1494.90, 512: 1800.37},
}

UNARY = {
    "128x512": {"Zero": 226.812, "Identity": 447.185, "ReLU": 304.377},
    "512x64":  {"Zero": 199.657, "Identity": 403.738, "ReLU": 302.079},
    "512x128": {"Zero": 206.461, "Identity": 407.240, "ReLU": 303.261},
    "512x512": {"Zero": 207.026, "Identity": 409.993, "ReLU": 304.272},
}

KS = [64, 128, 512]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- links: GEMM, gruppiert nach M x N, eine Säule je K --------------------
labels = [f"{m}x{n}" for (m, n) in GEMM]
x = range(len(labels))
width = 0.27
colors = ["#9ecae1", "#4292c6", "#08519c"]

for i, k in enumerate(KS):
    vals = [GEMM[mn][k] for mn in GEMM]
    ax1.bar([p + (i - 1) * width for p in x], vals, width,
            label=f"K = {k}", color=colors[i])

# Erwartungsbereich aus der Aufgabenstellung
band = ax1.axhspan(1500, 1800, color="#666666", alpha=0.12, zorder=0,
                   label="Zielbereich 1,5 bis 1,8 TFLOPS")

ax1.set_xticks(list(x))
ax1.set_xticklabels(labels, rotation=45, ha="right")
ax1.set_xlabel("M x N")
ax1.set_ylabel("GFLOPS")
ax1.set_title("GEMM: Durchsatz über 27 Konfigurationen")
ax1.legend(loc="lower left", fontsize=8, framealpha=0.95)
ax1.grid(axis="y", alpha=0.3)
ax1.set_ylim(0, 2050)

# --- rechts: Unary-Bandbreiten --------------------------------------------
sizes = list(UNARY)
ops = ["Zero", "Identity", "ReLU"]
x2 = range(len(sizes))
for i, op in enumerate(ops):
    vals = [UNARY[s][op] for s in sizes]
    ax2.bar([p + (i - 1) * width for p in x2], vals, width,
            label=op, color=colors[i])

ax2.set_xticks(list(x2))
ax2.set_xticklabels(sizes)
ax2.set_xlabel("Matrixgröße")
ax2.set_ylabel("GiB/s")
ax2.set_title("Unary: Bandbreite")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, 500)

fig.tight_layout()
out = "/Users/justin/Developer/Repositories/mlc_lab/docs/week6/benchmarks.png"
fig.savefig(out, dpi=150)
print("geschrieben:", out)
