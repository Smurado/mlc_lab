# MLC Lab

Kurs-Repository von Julian und Justin. Es enthält die wöchentlichen Aufgaben des
MLC Lab sowie das Abschlussprojekt: **TEIR**, einen JIT-Autotuner für
Tensorkontraktionen auf Apple Silicon.

Der vollständige Projektbericht und alle Wochenberichte sind als gebaute
Dokumentation online: <https://smurado.github.io/mlc_lab/>

## Aufbau des Repositories

| Pfad                  | Inhalt                                                      |
|-----------------------|-------------------------------------------------------------|
| `Projekt/autotuner/`  | Abschlussprojekt: Quellcode (`src/`), Evaluations-Skripte (`eval/`), Eingabedaten (`data/`), Messrohdaten (`results/`) |
| `week1/` bis `week8/` | Wochenaufgaben, jeweils mit eigenem Makefile               |
| `docs/`               | Sphinx-Quellen der Berichte (Projekt + Wochen)              |
| `lib/`                | Catch2-Header für die Wochen-Tests                          |
| `.github/workflows/`  | CI (Autotuner und Wochen) sowie das Pages-Deployment        |

## Projekt: TEIR-Autotuner

Der Autotuner nimmt eine Einsum-Beschreibung entgegen, erzeugt Schedule-Kandidaten
(Kachelung, Schleifenreihenfolge, Parallelisierung, Unrolling), übersetzt sie per
JIT und misst sie real. Drei Backends: Scalar, NEON und SME. Details, Messmethodik
und Ergebnisse stehen im [Projektbericht](https://smurado.github.io/mlc_lab/projekt/index.html).

Bauen und ausführen (Apple Silicon, benötigt `brew install libomp`):

```bash
cd Projekt/autotuner/src
make               # baut teir_compiler
make test          # 8 Unit-Test-Suiten
./teir_compiler --help   # Referenz aller TEIR_*-Umgebungsvariablen
TEIR_INPUT=../data/input.csv ./teir_compiler
```

Der End-zu-End-Rauchtest liegt in `Projekt/autotuner/eval/smoke_test.sh`; die
Regressionssuite (`eval/retest_regressions.py`) läuft bewusst nur lokal auf der
idlen Referenzmaschine, weil Messwerte von geteilten CI-Runnern wertlos sind.

## Wochenaufgaben

Jede Woche baut mit `make` im jeweiligen Ordner. Die ausführlichen Berichte mit
Messwerten stehen in der Dokumentation; hier nur die Themen:

| Woche    | Thema                                                        |
|----------|--------------------------------------------------------------|
| Woche 1  | AArch64-Assembly-Grundlagen, Debugging mit LLDB              |
| Woche 2  | Execution Throughput (FMADD/FMLA), NEON-Permutation abc→cba  |
| Woche 3+4| SME: SSVE-Unary-Primitives und GEMM-Microkernel (fmopa)      |
| Woche 5  | Codegenerator-Basis, Portierung der Kernel                   |
| Woche 6  | GEMM- und Unary-Primitives über den Codegenerator            |
| Woche 7  | TEIR-Laufzeitumgebung mit JIT und OpenMP                     |
| Woche 8  | In-Memory-AST-Evaluator und Optimizer-Passes                 |

## Dokumentation lokal bauen

```bash
python3 -m venv .venv && .venv/bin/pip install sphinx
.venv/bin/sphinx-build -b html docs docs/_build/html
```

Bei jedem Push auf `main` baut GitHub Actions die Doku und veröffentlicht sie
auf GitHub Pages; zusätzlich laufen die CI-Workflows für den Autotuner
(macOS arm64: Build ohne Warnungen, Unit-Tests, Rauchtest) und die Wochen
(Cross-Compile + QEMU).
