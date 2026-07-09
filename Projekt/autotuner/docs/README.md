# TEIR: Dynamic Tensor Compiler & Autotuner

TEIR ist ein experimentelles Framework zur automatischen Generierung, Optimierung und JIT-Kompilierung von Tensorkontraktions-Kernels. Das Ziel des Projekts ist es, ein "High-Level" IR (Intermediate Representation) zu definieren, dieses dynamisch in optimierten C++ Code zu transformieren und zur Laufzeit als Shared Library zu kompilieren und auszuführen.

## Der aktuelle Pipeline-Prozess

Das System durchläuft derzeit folgende Schritte für jeden Workload:

1.  **IR Definition:** Programmatischer Aufbau der Tensorkontraktion (Achsen, Tensoren, Schedule).
2.  **Autotuning & Passes:** Anwendung von Optimierungen wie Axis-Splitting für Cache-Blocking und Parallelisierung.
3.  **Code-Generierung:** Umwandlung der TEIR-Struktur in C++ Quellcode.
4.  **JIT-Kompilierung:** Kompilierung des generierten Codes mittels `g++` (`-O3`, `-march=native`) zur Laufzeit.
5.  **Runtime-Integration:** Dynamisches Laden der `.so` Library via `dlopen` und Bindung des Kernel-Symbols mittels `dlsym`.
6.  **Speichermanagement:** Dynamische Berechnung der benötigten Puffergrößen basierend auf den Achsendefinitionen und Allokation im RAM.
7.  **Execution & Verification:** Ausführung des JIT-Kernels auf den Eingabedaten und mathematische Validierung der Ergebnisse.

## Projektstatus: V.0.1 (Integration der JIT-Laufzeit)

Mit den letzten Updates haben wir die Lücke zwischen Kompilierung und Ausführung geschlossen:

* **Dynamische Allokation:** Das System berechnet nun eigenständig die notwendigen Puffer-Größen basierend auf den `axisDefinitions` für Eingabe- und Ausgabetensoren.
* **JIT-Runtime:** Durch die Integration von `dlfcn.h` kann das System generierte Bibliotheken sofort laden und als `void*` Handle verarbeiten.
* **Robustheit:** Die `main.cpp` verwendet nun eine sichere Typ-Initialisierung für alle TEIR-Strukturen, um Kompatibilitätsprobleme mit modernen Apple-Clang Compilern zu vermeiden.
* **Verifikation:** Jeder Kernel-Durchlauf wird automatisch gegen ein mathematisches Modell validiert, um Korrektheit sicherzustellen.

## Voraussetzungen

* C++20 kompatibler Compiler (getestet mit Apple Clang / G++)
* Standard-Linux/macOS Umgebung mit `g++` und `dlfcn` (POSIX)
* Make

## Projektstruktur

```
autotuner/
├── src/       C++-Kern: alle .cpp/.hpp, Makefile, test_c2_validator.cpp
├── eval/      Python-Experimente & Harness (convergence_/ablation_/tvm_/pytorch_comparison,
│   │          run_gett_matrix, gett_bench, verify_autotuner, retest_regressions, …)
│   └── notebooks/  Vergleichs-Notebooks (hero/overhead/tuningtime) + Builder + Charts
├── data/      Input-CSVs (input.csv, input_gett.csv, *.teir)
├── results/   Testergebnisse (gett_results/, autotuner_results.csv, verification_results.csv)
├── docs/      README, TVM_SETUP
├── archive/   alte Debug-/Scratch-Dateien
└── .venv/     Python-Umgebung (numpy, matplotlib, torch)
```

## Build & Run

Der C++-Kern wird über das Makefile in `src/` gebaut:

1.  **Bauen** (aus `src/`):
    ```bash
    cd src && make        # erzeugt src/teir_compiler
    ```

2.  **Einzelnen Kernel tunen** (aus dem autotuner-Root):
    ```bash
    TEIR_INPUT=data/input_gett.csv src/teir_compiler
    ```

3.  **Experimente/Vergleiche** (aus dem autotuner-Root, mit lokalem venv):
    ```bash
    .venv/bin/python eval/retest_regressions.py         # Regressionstest (6out + GEMM)
    .venv/bin/python eval/run_gett_matrix.py            # 48-Fälle-Matrix torch/TEIR/TVM
    .venv/bin/jupyter notebook eval/notebooks/          # Vergleichs-Notebooks
    ```
    Die Skripte finden `src/teir_compiler` und `data/`/`results/` automatisch (relative Pfade).
    TVM-Setup separat: siehe `docs/TVM_SETUP.md`.

## Roadmap

- [x] Grundlegende IR-Struktur
- [x] Code-Generierung
- [x] JIT-Kompilierung (`g++`)
- [x] Dynamische Speicherallokation
- [x] Symbol-Bindung via `dlsym`
- [ ] Erweiterung des Autotuners (automatisches Finden optimaler Split-Faktoren)
- [ ] Support für weitere Hardware-Backends (z.B. CUDA/OpenCL)