Week 7:
====================

1. Introduction
----------------

In dieser Woche haben wir eine komplette Runtime-Umgebung implementiert, die TEIR Spezifikationen verarbeitet und in wiederverwendbare Function-Pointer kompiliert. 
Dabei lag der Fokus darauf, den Loop-Nest rekursiv aus den Iteration-Nodes zu erzeugen, OpenMP für die Shared-Memory-Parallelisierung zu nutzen 
und einen eigenen Code-Generator für den innersten Kernel einzubinden.

2. TEIR-AST und JIT-Codegen
---------------------------

Die Programm-Repräsentation erfolgt über einen abstrakten Syntaxbaum (TEIR-AST), der über den entwickelten Parser aus den ``.teir``-Dateien aufgebaut wird.

- Der rekursive Codegenerator wandert durch den Schedule-Baum und erzeugt C++-Code (``teir_jit_<name>.cpp``).
- Im Anschluss wird dieser Code per externem Aufruf (unter anderem mit ``clang++`` und `-O3`) in eine Shared Library (``.so``) überführt.
- Der typsichere Function-Pointer wird über ``dlopen`` und ``dlsym`` geladen und führt die dynamisch kompilierten Instruktionen aus.

3. OpenMP-Parallelisierung
--------------------------

Für eine bestmögliche Shared-Memory-Skalierung auf der CPU musste die Handhabung der OpenMP-Befehle im Baum optimiert werden:

- Ursprünglich führte die Übersetzung jeder aneinanderreihenden ``policy parallel``-Iteration zu starker Verschachtelung und erheblichem Overhead (Thread-Explosion).
- Dies wurde behoben, indem wir im Codegen aufeinanderfolgende ``parallel``-Knoten zusammenfassen und dafür eine einzelne ``#pragma omp parallel for collapse(N)`` Direktive generieren.
- Mit diesem Ansatz skaliert beispielsweise die Matrixmultiplikation linear und fehlerfrei über die verfügbaren Hardware-Threads (z.B. 16 Threads).

4. NEON-Microkernel
-------------------

Für das innerste Primitiv in Tensor-Kontraktionen wurde ein eigener, handgerollter NEON-Microkernel mit Register-Blocking integriert.

- Anstelle einer ineffizienten skalaren Schleife haben wir das Problem in Tiles von ``M_R = 4`` und ``N_R = 16`` unterteilt.
- Während der inneren K-Schleife verbleiben die Ergebnisse in den 16 Akkumulatoren-Registern (mittels ``vfmaq_n_f32``), was unnötiges Lesen und Schreiben in den Speicher verhindert.
- Diese Spezialisierung greift automatisch bei passenden Strides und Problemgrößen und bringt enorme Leistungsvorteile: Die reine Contraction-Workload steigerte sich bei 16 Threads von anfänglichen ca. 200 GFLOPS auf erstaunliche ~987 GFLOPS.

5. Schedule-Anpassung (TEIR-Fix)
--------------------------------

Während der Verifikation stellten wir fest, dass die ``matmul.teir`` Spec einen logischen Fehler enthielt:

- Das ``Zero``-Primitiv, zuständig für das Nullsetzen der Speicherbereiche vor der Matrixmultiplikation, lief in jeder K0-Iteration, wodurch vorherige Akkumulationen wieder gelöscht wurden. Im Endeffekt überlebte somit nur das Ergebnis der letzten Iteration.
- In Analogie zur funktionierenden ``contraction.teir`` haben wir dieses Verhalten durch Hinzufügen des passenden Guards (``guard first(@k0)``) korrigiert, wodurch alle Benchmarks nun erfolgreich ("PASS") und mit der erwarteten Präzision durchlaufen.

6. Architektur-Verbesserungen und Optimierungen
-----------------------------------------------

Im Rahmen der aktuellen Entwicklung haben wir unsere Architektur grundlegend überarbeitet und verbessert:

- **Zero-Kernel & Akkumulation:** Wir trennen Initialisierung und Berechnung nun strikt. Die Nullsetzung erfolgt *einmalig* durch die ``@zero`` Primitive (via ``guard first(...)``). Der eigentliche Rechenkernel (``@gemm``) lädt den Output vor der K-Schleife (mittels ``vld1q_f32`` im NEON-Pfad), akkumuliert darauf (``vfmaq_n_f32``) und speichert ihn wieder (``vst1q_f32``).
- **Transponierungen:** Wir verwenden nun generische Offset-Berechnungen über **Strides**. Ein Tauschen von Achsen (z.B. für eine Transponierung) wird nativ durch die Anpassung der Strides in der ``.teir`` Konfiguration modelliert (wie bei der Datei ``transposition.teir``). Spezifische Transponierungs-Abfragen auf Code-Ebene fallen damit gänzlich weg.

7. Performance-Werte (Benchmarks)
---------------------------------

Zur detaillierten Auswertung dokumentieren wir die Performance-Ergebnisse der Runtime übersichtlich mit Tabellen, um den erzeugten JIT-Code messbar und verständlich aufzubereiten:

**Aktuelle Messwerte (Lokales Deployment, Apple Silicon M-Series, FP32, 16 Threads)**

.. table:: Messwerte der TEIR Runtime Module

   ==================== ============= ============= ===========================================
   Workload             Korrektheit   Beste Zeit    Beste Performance (Durchsatz)
   ==================== ============= ============= ===========================================
   Transposition        PASS          ~ 7.8 ms      ~ 19 GB/s
   Matmul (8192^3)      PASS          ~ 1.19 s      ~ 926 GFLOPS (mit NEON-Microkernel)
   Contraction          PASS          ~ 1.25 s      ~ 987 GFLOPS (mit NEON-Microkernel)
   ==================== ============= ============= ===========================================

Diese Werte vergleichen den Fallback-Codegen (skalar) mit unserem neu erstellten NEON-Codegenerator. Während die speicherlimiterte Transposition gleich aufbleibt, konnte die Performance der Tensor Contraction durch den `try_emit_neon_contraction`-Pfad von ehemals ca. 200 GFLOPS (skalar) auf knapp **1 TFLOP** fast verfünffacht werden.

.. image:: pictures/output.png
   :width: 100%
   :alt: Benchmark Graphen

Graphen und Vergleiche der Thread-Sweeps (z.B. {4, 8, 10, max_threads}) sind über die Konsolenausgaben des Benchmark-Flags zu entnehmen und spiegeln dort die lineare Parallelisierung durch OpenMP wider.

8. Contributions
----------------

In dieser Woche haben wir uns die Teilaufgaben wie folgt aufgeteilt:

**Justin Bergmann:**

- Verfassen der ausführlichen wöchentlichen Dokumentation zu Architektur, Optimierungen und dem Bugfix der TEIR-Spezifikation.
- Entwicklung und Automatisierung der umfassenden Test-Suite (inklusive der Phase-1-Korrektheitstests mit Random-Inputs und der Sanity-Checks in Phase 2).
- Validierung der generierten JIT-Kernel auf korrekte Verarbeitung der Strides und mathematische Präzision (Abgleich gegen naive Referenz-Implementierungen).

**Julian Müller:**

- Vollständige Implementierung der ``main.cpp`` als zentralem Laufzeit-Hub für unseren TEIR-Compiler.
- Integration von Modulen wie dem AST-Aufbau, Parser und JIT-Codegenerator innerhalb der Hauptanwendung.
- Bereitstellung der Ausführungs- und Benchmark-Logik in C++ (inkl. Thread-Sweeps, Allokation der Tensoren und Performance-Logging in GFLOPS/GB/s).