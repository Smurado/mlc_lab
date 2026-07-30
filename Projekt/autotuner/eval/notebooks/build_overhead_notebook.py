#!/usr/bin/env python3
"""Baut overhead_comparison.ipynb — der Fall, in dem PyTorch schlecht und TEIR gut ist:
kleine GEMMs, bei denen der Framework-/Dispatch-Overhead PyTorch ausbremst, waehrend
ein kompilierter TEIR-Kernel schlank bleibt. Zeigt die Overhead-Crossover-Kurve.
Alle Zahlen frisch messbar (kein TVM -> schnell)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Kleine Kernels: Wenn Framework-Overhead PyTorch ausbremst\n"
    "\n"
    "Der **umgekehrte** Fall zum Hero-GEMM: bei **kleinen** Matrix-Multiplikationen "
    "dominiert bei PyTorch der **Dispatch-/Setup-Overhead** (Python-Aufruf, BLAS-Setup, "
    "AMX-Anlauf) — der lässt sich auf einem winzigen 16×16-Produkt nicht amortisieren. "
    "Ein **kompilierter TEIR-Kernel** ist dagegen ein direkter C-Funktionsaufruf in einer "
    "engen Schleife → **hier gewinnt TEIR deutlich**.\n"
    "\n"
    "Wir fahren `ab-ac-cb` über wachsende Größen $N$ und suchen den **Crossover**: "
    "ab welcher Größe holt PyTorchs BLAS/AMX seinen Overhead wieder rein?\n"
    "\n"
    "**Hardware:** Apple M4 Max. Frisch gemessen (kein TVM → schnell).\n"
))

c.append(nbf.v4.new_code_cell(
    "import os, subprocess, re, time\n"
    "import torch\n"
    "torch.set_num_threads(os.cpu_count() or 1)\n"
    "\n"
    "SIZES = [16, 24, 32, 48, 64, 96, 128]\n"
    "DIR = os.getcwd()\n"
    "AUTO = os.path.dirname(os.path.dirname(DIR))  # notebooks/ -> eval/ -> autotuner/\n"
    "COMPILER = os.path.join(AUTO, 'src', 'teir_compiler')\n"
    "\n"
    "def bench(fn, warmup=30, iters=50, reps=5):\n"
    "    for _ in range(warmup):\n"
    "        fn()\n"
    "    best = float('inf')\n"
    "    for _ in range(reps):\n"
    "        t0 = time.perf_counter()\n"
    "        for _ in range(iters):\n"
    "            fn()\n"
    "        best = min(best, (time.perf_counter() - t0) / iters)\n"
    "    return best\n"
    "\n"
    "def teir(N):\n"
    "    csv = os.path.join(DIR, '_ovh_one.csv')\n"
    "    with open(csv, 'w') as f:\n"
    "        f.write('name,tensors,axes,primitives,schedule,invokes,einsum\\n')\n"
    "        f.write(f'o,in0:f32;in1:f32;out:f32,a:{N};b:{N};c:{N},zero;gemm,'\n"
    "                f'a:sequential;b:sequential;c:sequential,zero;gemm,ab-ac-cb\\n')\n"
    "    env = {**os.environ, 'TEIR_INPUT': csv, 'TEIR_STRATEGY': 'sa', 'TEIR_BACKEND': 'neon',\n"
    "           'TEIR_MAX_TRIALS': '15', 'TEIR_TIME_BUDGET_MS': '600000', 'TEIR_SEARCH_OPT': '-O2'}\n"
    "    out = subprocess.run([COMPILER], capture_output=True,\n"
    "                         text=True, env=env, timeout=None).stdout  # keine Zeitgrenze\n"
    "    m = re.search(r'\\[PERFORMANCE\\]\\s+[\\d.eE+-]+\\s+ms\\s+\\(([\\d.eE+-]+)', out)\n"
    "    return float(m.group(1)) if m else float('nan')\n"
    "\n"
    "import json\n"
    "RESULTS_JSON = os.path.join(DIR, 'overhead_results.json')\n"
    "REMEASURE = False  # True = neu messen (Sekunden pro Groesse); False = gespeicherte laden\n"
    "if REMEASURE or not os.path.exists(RESULTS_JSON):\n"
    "    rows = []\n"
    "    for N in SIZES:\n"
    "        A = torch.randn(N, N); B = torch.randn(N, N); F = 2 * N**3\n"
    "        mm = F / bench(lambda: torch.matmul(A, B)) / 1e9\n"
    "        es = F / bench(lambda: torch.einsum('ac,cb->ab', A, B)) / 1e9\n"
    "        tr = teir(N)\n"
    "        rows.append((N, mm, es, tr))\n"
    "        print(f'N={N:>4}:  torch.mm={mm:8.1f}  einsum={es:8.1f}  TEIR={tr:8.1f} GFLOPS', flush=True)\n"
    "    with open(RESULTS_JSON, 'w') as f:\n"
    "        json.dump(rows, f, indent=1)\n"
    "    print('Gemessen + gespeichert ->', os.path.basename(RESULTS_JSON))\n"
    "else:\n"
    "    with open(RESULTS_JSON) as f:\n"
    "        rows = [tuple(r) for r in json.load(f)]\n"
    "    print('Geladen aus', os.path.basename(RESULTS_JSON), '(REMEASURE=True zum Neumessen)')\n"
))

c.append(nbf.v4.new_markdown_cell("## Tabelle — wer gewinnt bei welcher Größe?"))

c.append(nbf.v4.new_code_cell(
    "import pandas as pd\n"
    "df = pd.DataFrame(rows, columns=['N', 'torch.matmul', 'torch.einsum', 'TEIR (neon)'])\n"
    "df['Sieger'] = df[['torch.matmul', 'torch.einsum', 'TEIR (neon)']].idxmax(axis=1)\n"
    "df['TEIR / torch.mm'] = (df['TEIR (neon)'] / df['torch.matmul']).round(2)\n"
    "df\n"
))

c.append(nbf.v4.new_markdown_cell("## Crossover-Kurve"))

c.append(nbf.v4.new_code_cell(
    "import matplotlib.pyplot as plt\n"
    "N = [r[0] for r in rows]\n"
    "fig, ax = plt.subplots(figsize=(9.5, 6))\n"
    "ax.plot(N, [r[1] for r in rows], 'o-', label='PyTorch torch.matmul', color='#d62728', lw=2.5, markersize=7)\n"
    "ax.plot(N, [r[2] for r in rows], 's--', label='PyTorch torch.einsum', color='#ff9896', lw=2.5, markersize=7)\n"
    "ax.plot(N, [r[3] for r in rows], '^-', label='TEIR (getunt, NEON)', color='#1f77b4', lw=2.5, markersize=8)\n"
    "ax.set_xscale('log', base=2); ax.set_yscale('log')\n"
    "ax.set_xticks(N); ax.set_xticklabels(N, fontsize=12)\n"
    "ax.tick_params(axis='y', labelsize=12)\n"
    "ax.set_xlabel('Matrixgröße N  (GEMM N\\u00d7N\\u00d7N)', fontsize=14)\n"
    "ax.set_ylabel('GFLOPS  (log)', fontsize=14)\n"
    "ax.set_title('Overhead-Crossover: kleine GEMMs  \\u2014  Apple M4 Max', fontsize=15, fontweight='bold')\n"
    "ax.grid(True, which='both', alpha=0.3)\n"
    "ax.legend(fontsize=12)\n"
    "# Bereich markieren, in dem TEIR fuehrt\n"
    "lead = [n for n, r in zip(N, rows) if r[3] > r[1]]\n"
    "if lead:\n"
    "    ax.axvspan(min(N), max(lead) * 1.15, color='#1f77b4', alpha=0.06)\n"
    "    ax.text(min(N) * 1.05, ax.get_ylim()[1] * 0.5, 'TEIR führt\\n(Overhead-Regime)',\n"
    "            color='#1f77b4', fontsize=12, fontweight='bold', va='top')\n"
    "plt.tight_layout(); plt.savefig('overhead_comparison.png', dpi=140); plt.show()\n"
))

c.append(nbf.v4.new_markdown_cell(
    "## Was das zeigt (ehrlich)\n"
    "\n"
    "- **Bei kleinen N gewinnt TEIR — teils drastisch** (bei 16³ rund **10× vor `torch.matmul`**, "
    "~75× vor `torch.einsum`). Grund: PyTorchs fixe Kosten pro Aufruf (Python-Dispatch, "
    "BLAS-/AMX-Setup) überwiegen die winzige Rechenlast; TEIRs kompilierter Kernel hat "
    "diesen Overhead nicht.\n"
    "- **Ab dem Crossover (~32–64) dreht sich das:** die Rechenlast wird groß genug, dass "
    "PyTorchs BLAS/AMX seinen Overhead amortisiert und davonzieht (siehe Hero-Notebook).\n"
    "- **`torch.einsum` liegt durchweg klar unter `torch.matmul`** — selbst innerhalb PyTorch "
    "kostet der einsum-Pfad (Planung + Umformen) spürbar.\n"
    "\n"
    "**Ehrlicher Caveat:** bei sehr kleinen N sind die GFLOPS nahe am Messrauschen "
    "(Sub-Mikrosekunden-Rechenzeit) — der Sieg heißt genau genommen *„weniger Overhead\", nicht "
    "„besserer Compute\"*. Aber es ist ein **echter, praktischer Vorteil** für kleine oder "
    "einmalige Kernels: ein spezialisierter, kompilierter Kernel schlägt den Framework-Aufruf. "
    "Genau das ist eine Stärke des Codegen-/Autotuning-Ansatzes.\n"
))

nb['cells'] = c
with open('overhead_comparison.ipynb', 'w') as f:
    nbf.write(nb, f)
print('overhead_comparison.ipynb geschrieben:', len(c), 'Zellen')
