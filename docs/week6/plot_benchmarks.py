"""Erzeugt die Abbildung für docs/week6 aus der Messkampagne.

Datenquelle: week6/main, 15 Läufe je Messpunkt, Apple M4 Max, Apple clang 21.
Angegeben ist der Median; die Fehlerbalken zeigen Minimum und Maximum über
dieselben Läufe. Sie machen sichtbar, dass kleine Unary-Matrizen nicht
belastbar messbar sind, während die GEMM-Werte stabil liegen.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (M, N) -> {K: (Median, Minimum, Maximum)}
GEMM = {
    (64, 64): {64: (1324.7, 1299.0, 1339.7), 128: (1589.4, 1559.3, 1605.4), 512: (1867.0, 1822.0, 1898.2)},
    (64, 128): {64: (1322.4, 1294.3, 1340.7), 128: (1580.6, 1537.0, 1604.8), 512: (1868.3, 1816.6, 1900.7)},
    (64, 512): {64: (1317.8, 1298.0, 1342.4), 128: (1588.1, 1551.5, 1610.5), 512: (1847.8, 1790.5, 1895.1)},
    (128, 64): {64: (1328.0, 1298.3, 1345.2), 128: (1587.2, 1562.5, 1605.6), 512: (1872.9, 1812.2, 1901.3)},
    (128, 128): {64: (1324.7, 1289.4, 1343.1), 128: (1578.0, 1521.2, 1607.0), 512: (1876.6, 1833.5, 1902.0)},
    (128, 512): {64: (1320.9, 1290.2, 1334.5), 128: (1583.4, 1532.0, 1612.4), 512: (1872.7, 1822.2, 1891.2)},
    (512, 64): {64: (1129.7, 1074.9, 1163.7), 128: (1455.5, 1430.9, 1489.3), 512: (1827.8, 1769.2, 1854.5)},
    (512, 128): {64: (1146.5, 1103.6, 1177.8), 128: (1461.7, 1434.3, 1487.6), 512: (1822.2, 1741.4, 1852.1)},
    (512, 512): {64: (1137.4, 1071.3, 1177.7), 128: (1457.8, 1381.9, 1491.9), 512: (1796.0, 1755.2, 1846.7)},
}

# Größe -> {Operation: Median}
UNARY = {
    "128x512": {"Zero": 225.1, "Identity": 436.9, "ReLU": 303.2},
    "512x64": {"Zero": 202.0, "Identity": 402.8, "ReLU": 302.0},
    "512x128": {"Zero": 204.3, "Identity": 405.4, "ReLU": 303.6},
    "512x512": {"Zero": 204.7, "Identity": 409.1, "ReLU": 303.9},
}

# Größe -> größte Spannweite über die drei Operationen, in Prozent
UNARY_SPREAD = {"128x512": 21.6, "512x64": 9.5, "512x128": 4.7, "512x512": 4.8}

KS = [64, 128, 512]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- links: GEMM, gruppiert nach M x N, eine Säule pro K --------------------
labels = [f"{m}x{n}" for (m, n) in GEMM]
x = range(len(labels))
width = 0.27
colors = ["#9ecae1", "#4292c6", "#08519c"]

for i, k in enumerate(KS):
    med = [GEMM[mn][k][0] for mn in GEMM]
    lo = [GEMM[mn][k][0] - GEMM[mn][k][1] for mn in GEMM]
    hi = [GEMM[mn][k][2] - GEMM[mn][k][0] for mn in GEMM]
    ax1.bar([p + (i - 1) * width for p in x], med, width,
            label=f"K = {k}", color=colors[i],
            yerr=[lo, hi], capsize=2, ecolor="#333333", error_kw={"lw": 0.8})

ax1.axhspan(1500, 1800, color="#666666", alpha=0.12, zorder=0,
            label="Zielbereich 1,5 bis 1,8 TFLOPS")

ax1.set_xticks(list(x))
ax1.set_xticklabels(labels, rotation=45, ha="right")
ax1.set_xlabel("M x N")
ax1.set_ylabel("GFLOPS")
ax1.set_title("GEMM: Median aus 15 Läufen, Balken zeigen Min und Max")
ax1.legend(loc="lower left", fontsize=8, framealpha=0.95)
ax1.grid(axis="y", alpha=0.3)
ax1.set_ylim(0, 2050)

# --- rechts: Unary-Bandbreiten ---------------------------------------------
sizes = list(UNARY)
ops = ["Zero", "Identity", "ReLU"]
x2 = range(len(sizes))
for i, op in enumerate(ops):
    vals = [UNARY[s][op] for s in sizes]
    ax2.bar([p + (i - 1) * width for p in x2], vals, width,
            label=op, color=colors[i])

for i, s in enumerate(sizes):
    ax2.text(i, 468, f"Spannweite {UNARY_SPREAD[s]:.1f} %".replace(".", ","),
             ha="center", fontsize=8, color="#444444")

ax2.set_xticks(list(x2))
ax2.set_xticklabels(sizes)
ax2.set_xlabel("Matrixgröße")
ax2.set_ylabel("GiB/s")
ax2.set_title("Unary: Median aus 15 Läufen")
ax2.legend(fontsize=9, loc="center right")
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, 500)

fig.tight_layout()
out = "/Users/justin/Developer/Repositories/mlc_lab/docs/week6/benchmarks.png"
fig.savefig(out, dpi=150)
print("geschrieben:", out)
