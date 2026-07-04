# Roadmap & Fortschritt: TEIR Auto-Tuning Optimizer

Chronologische Abarbeitungsliste für das Abschlussprojekt. Basiert auf der Vision in [Ideen.md](Ideen.md) und dem aktuellen Code-Stand im [autotuner/](autotuner)-Ordner.

**Legende:** ✅ fertig · 🟡 teilweise / mit Mängeln · ⬜ offen

Stand: 2026-07-04

---

## Gesamtfortschritt auf einen Blick

| Stufe | Thema | Status |
|-------|-------|--------|
| 1 | MVP – Parametrisierte JIT-Kompilierung | ✅ fertig |
| 2 | Profiler & Suchraum (Trials) | ✅ fertig |
| 3 | Kür – Heuristiken & Framework-Vergleich | ⬜ offen |

---

## Stufe 1 – MVP: Parametrisierte JIT-Kompilierung

- [x] TEIR-IR-Struktur (`Axis`, `Iteration`, `Policy`, `Tensor`) — [teir.hpp](autotuner/teir.hpp)
- [x] Parser für `.teir`-Dateien — [parser.cpp](autotuner/parser.cpp), [input.teir](autotuner/input.teir)
- [x] Parametrisierte Passes statt fester Werte — [passes.cpp](autotuner/passes.cpp)
  - [x] `splitOuterAxis` (Tiling / Cache-Blocking)
  - [x] `reorderSchedule` (Loop-Nesting)
  - [x] `makeParallel` (OpenMP-Parallelisierung)
- [x] Code-Generierung mit ARM NEON-Intrinsics — [codegen.cpp](autotuner/codegen.cpp)
- [x] JIT-Kompilierung zur Laufzeit (`g++ -O3 -shared -fPIC`) — [main.cpp](autotuner/main.cpp)
- [x] Dynamisches Laden via `dlopen` / `dlsym`
- [x] Mathematische Verifikation des Kernel-Ergebnisses

---

## Stufe 2 – Profiler & Suchraum (Trials)

- [x] Diskreter Suchraum aus Split-Faktoren × Loop-Orders × Parallel-Achsen — [autotuner.cpp](autotuner/autotuner.cpp)
- [x] Regelbasiertes Pruning
  - [x] Nur Split-Faktoren, die das Achsen-Extent exakt teilen
  - [x] Innerste Schleife wird nie parallelisiert
- [x] Trial-Schleife mit Auswahl des besten Kandidaten
- [x] Ergebnis-Logging nach `autotuner_results.csv` — [benchmark.cpp](autotuner/benchmark.cpp)
- [x] **Benchmark misst den ECHTEN JIT-Kernel** — jeder Trial wird mit `g++ -O3 -march=native -fopenmp` kompiliert, auf echten Daten vermessen (Best-of-5 × 200 Iterationen) und liefert reale GFLOPS
- [x] **Korrektheits-Validierung pro Trial** — inkorrekte/racy Konfigurationen (parallelisierte Reduktionsachse) werden automatisch verworfen
- [x] **Codegen verwendet die getunten Parameter** — `generateSourceCode` ist parametrisch (Split, Loop-Order, Parallel-Achse); `main.cpp` verbaut die real beste Konfiguration
- [x] **Early-Stopping / „Regress-Verfall"** — Patience-Counter (Stagnations-Abbruch), Min-Delta gegen Jitter, reproduzierbares Shuffle, hartes Zeit-Budget als Sicherheitsnetz — [autotuner.cpp](autotuner/autotuner.cpp)

---

## Stufe 3 – Die Kür: Heuristiken & Benchmarking

- [ ] Intelligente Such-Heuristik statt Brute-Force
  - [ ] z.B. Simulated Annealing **oder** Genetic Algorithm
  - [ ] **oder** abstraktes Kostenmodell zum Vorfiltern
- [ ] SME-Backend zusätzlich zu NEON (Ideen.md fordert „speziell SME und NEON")
- [ ] Benchmark-Visualisierung (Notebook / Plots)
- [ ] Vergleich gegen etablierte Frameworks
  - [ ] PyTorch mit `accelerate`
  - [ ] Apache TVM

---

## Empfohlene chronologische Reihenfolge der nächsten Schritte

1. ~~**Benchmark auf echten JIT-Kernel umstellen**~~ ✅ erledigt — reale Messung + Korrektheits-Check pro Trial.
2. ~~**Codegen parametrisieren**~~ ✅ erledigt — getunte Parameter fließen in den Kernel, Best-Config wird verbaut.
3. ~~**Early-Stopping**~~ ✅ erledigt — Patience + Min-Delta + Shuffle + Zeit-Budget.
4. **Suchheuristik** (Simulated Annealing o.ä.) auf den jetzt validen Suchraum aufsetzen (Stufe 3).
5. **SME-/NEON-Intrinsics-Codegen-Pfad** ergänzen (Stufe 3) — aktuell skalarer Kern mit `-O3 -march=native`-Autovektorisierung.
6. **Framework-Vergleich + Visualisierung** als Abschluss-Evaluation (Stufe 3).

---

## Offene Notizen / Risiken

- ✅ Der größte konzeptionelle Bruch (simuliertes Benchmark + hartcodierter Codegen) ist behoben: Tuning, Messung und generierter Kernel hängen jetzt zusammen.
- Laufzeit: ~96 echte `g++ -O3`-Kompilate → voller Lauf dauert ~2 min (compile-gebunden). Early-Stopping / Caching könnte das reduzieren.
- OpenMP läuft real (Homebrew GCC 15, 16 Threads). Bei diesem kleinen Workload (~0.79 MFLOP/Kernel) ist seriell aktuell am schnellsten — parallele Configs verlieren durch Thread-Overhead.
- Aktueller Kern ist skalar (Autovektorisierung via `-march=native`). Echte NEON-/SME-Intrinsics bleiben als Stufe-3-Erweiterung offen.
- SME-Support evtl. abhängig von Hardware-/Compiler-Verfügbarkeit auf dem Zielgerät (M-Serie + passender Toolchain) prüfen.
