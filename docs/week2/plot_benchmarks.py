"""Erzeugt die Abbildung für docs/week2 aus den gemessenen Werten.

Datenquelle: week2/main.cpp, Benchmarks [perm_bench], [fmadd_bench],
[fmla4s_bench], [fmla2s_bench] auf Apple M4 Max.

Arbeitssatz der Permutation: beide Tensoren zusammen, also
2 * |a| * |b| * |c| * 4 Byte = 256 * |c| Byte.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# |c| -> GiB/s
BW = {
    4: 120.0, 8: 143.7, 16: 186.6, 32: 190.0, 128: 194.3, 256: 195.4,
    512: 192.2, 1024: 86.5, 4096: 53.5, 16384: 47.4, 65536: 54.4,
    262144: 27.2,
}

THROUGHPUT = {
    "FMADD\n(scalar)": 34.4,
    "FMLA 2S\n(vector)": 59.0,
    "FMLA 4S\n(vector)": 109.1,
}

BYTES_PER_C = 256          # beide Tensoren zusammen
L1_BYTES    = 65536        # hw.l1dcachesize
L2_BYTES    = 4194304      # hw.l2cachesize

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                               gridspec_kw={"width_ratios": [1.7, 1]})

# --- links: Bandbreite über |c| -------------------------------------------
cs   = sorted(BW)
vals = [BW[c] for c in cs]

ax1.plot(cs, vals, marker="o", color="#08519c", linewidth=2, markersize=6)
ax1.set_xscale("log", base=2)
ax1.set_xlabel("|c|")
ax1.set_ylabel("GiB/s")
ax1.set_title("Permutation abc nach cba: Bandbreite über |c|")
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 220)

# Cache-Grenzen: |c|, ab dem der Arbeitssatz die jeweilige Größe überschreitet
for limit, name, color in ((L1_BYTES, "L1d 64 KiB", "#d95f02"),
                           (L2_BYTES, "L2 4 MiB", "#7570b3")):
    c_limit = limit / BYTES_PER_C
    ax1.axvline(c_limit, color=color, linestyle="--", linewidth=1.4)
    ax1.text(c_limit * 1.12, 202, name, fontsize=9, color=color, va="top")

peak_c = max(BW, key=BW.get)
ax1.annotate(f"Peak {BW[peak_c]} GiB/s", xy=(peak_c, BW[peak_c]),
             xytext=(peak_c / 24, BW[peak_c] + 12), fontsize=9, ha="left",
             color="#08519c")

# --- rechts: Instruktionsdurchsatz ----------------------------------------
names = list(THROUGHPUT)
ax2.bar(names, [THROUGHPUT[n] for n in names],
        color=["#9ecae1", "#4292c6", "#08519c"], width=0.6)
for i, n in enumerate(names):
    ax2.text(i, THROUGHPUT[n] + 2, f"{THROUGHPUT[n]:.1f}",
             ha="center", fontsize=9)
ax2.set_ylabel("GFLOP/s")
ax2.set_title("Instruktionsdurchsatz")
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, 125)

fig.tight_layout()
out = "/Users/justin/Developer/Repositories/mlc_lab/docs/week2/benchmarks.png"
fig.savefig(out, dpi=150)
print("geschrieben:", out)
