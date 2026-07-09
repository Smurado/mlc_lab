# TVM-Vergleich (D3-Zusatz): Setup & Reproduktion

`tvm_comparison.py` stellt den getunten TEIR-Kernel gegen **Apache TVM (MetaSchedule)**
auf demselben GEMM. TVM ist der *faire* Anker: es generiert wie TEIR CPU-Code über
**LLVM** und tuned Schedules per Suche — gleicher Hardwarepfad (NEON/LLVM), kein
AMX-Coprozessor wie bei torch/BLAS.

## Warum ein Quelltext-Build?

Das einzige pip-Wheel `apache-tvm==0.25` ist die refaktorierte „relax/FFI"-Variante und
enthält **keinen** Tuner (`tvm.meta_schedule`, `tvm.autotvm`, `tvm.auto_scheduler` fehlen
alle; `tvm.tir`/`tvm.nd` sind wegrefaktoriert). Der klassische Autotuner lebt in
TVM **≤ 0.17** — deshalb Source-Build von `v0.17.0`.

## Rezept (macOS arm64, M4 Max)

```bash
# 1. Codegen-Backend: LLVM 18 (passt zu TVM 0.17; neuere LLVM brechen den Build)
brew install llvm@18

# 2. Isoliertes conda-Env mit Python 3.12 (venv-3.14 ist zu neu für TVM)
conda create -y -n tvm-bench python=3.12
PY="$(conda info --base)/envs/tvm-bench/bin/python"

# 3. MetaSchedule-Abhängigkeiten. WICHTIG: xgboost < 2.0 — TVM 0.17 nutzt
#    `from xgboost import rabit`, das in xgboost >= 2.0 entfernt wurde.
"$PY" -m pip install numpy decorator attrs psutil cloudpickle ml_dtypes tornado scipy "xgboost<2"

# 4. TVM v0.17 klonen (mit Submodulen) und bauen  (TVM_DIR = fester Ort, z.B. ~/tvm-0.17)
git clone --recursive --branch v0.17.0 --depth 1 https://github.com/apache/tvm ~/tvm-0.17
cd ~/tvm-0.17 && mkdir build && cd build
cp ../cmake/config.cmake .
{ echo 'set(USE_LLVM "/opt/homebrew/opt/llvm@18/bin/llvm-config")'
  echo 'set(CMAKE_BUILD_TYPE Release)'
  echo 'set(USE_LIBBACKTRACE OFF)'; } >> config.cmake
export CMAKE_POLICY_VERSION_MINIMUM=3.5   # cmake 4.x akzeptiert sonst TVMs alte Submodul-Skripte nicht
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja                                     # ~10-15 min auf 16 Cores -> build/libtvm.dylib

# 5. TVM editierbar ins Env installieren -> 'import tvm' findet die Lib dauerhaft selbst
"$PY" -m pip install -e ~/tvm-0.17/python

# 6. Vergleich fahren (kein PYTHONPATH noetig)
TVM_TRIALS=512 TVM_SIZES=256,512 "$PY" tvm_comparison.py
```

Steuerung: `TVM_TRIALS` (Tuning-Budget/Größe, Default 8 = Smoke-Test), `TVM_SIZES`
(z. B. `256,512`).

## Ergebnis (dieser Maschine, 512 Trials, `ab-ac-cb`)

| Variante (gleicher HW-Pfad) | 256³ | 512³ |
|---|---|---|
| TVM naiv (ungeschedult) | 3.4 | 2.9 |
| **TEIR-best (unser Autotuner, neon)** | **37.5** | **34.3** |
| **TVM tuned (MetaSchedule)** | **214.8** | **573.6** |

TEIR schlägt TVM-naiv ~11×, erreicht aber nur ~17 % (256³) / ~6 % (512³) von TVM-tuned.
Der Gap ist Codegen-Reichtum (mehrstufiges Tiling/Register-Blocking, das TEIRs Suchraum
nicht ausdrückt), nicht das Suchbudget — Details in `Roadmap_neu.md`, Phase D (D3-Zusatz).

## Persistenz (bereits eingerichtet)

Der Build liegt dauerhaft unter **`~/tvm-0.17`** und ist per `pip install -e` editierbar
ins conda-Env `tvm-bench` eingehängt. `import tvm` findet `~/tvm-0.17/build/libtvm.dylib`
automatisch — **kein `PYTHONPATH`/`TVM_LIBRARY_PATH` nötig**. Aufruf einfach:

```bash
PY="$(conda info --base)/envs/tvm-bench/bin/python"
TVM_TRIALS=512 TVM_SIZES=256,512 "$PY" tvm_comparison.py
```

Nach Änderungen am C++-Teil von TVM: `cd ~/tvm-0.17/build && ninja` genügt (editable
install zeigt schon auf diesen Ordner, kein Reinstall nötig). Reine Python-Änderungen an
TVM greifen durch den editable install ohnehin sofort.
