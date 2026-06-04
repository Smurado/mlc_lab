Week 8: In-Memory AST Evaluator & Ablation Study
================================================

1. Introduction
----------------

In Woche 8 haben wir uns von der JIT-Kompilierung entfernt und stattdessen eine direkte In-Memory Evaluierung des TEIR-Abstrakten Syntaxbaums (AST) implementiert. Das Ziel war es, Programmtransformationen (wie Cache Blocking und Loop-Parallelisierung) direkt auf der AST-Datenstruktur auszuführen (als Optimizer-Pass) und anschließend einen Interpreter darüberlaufen zu lassen. Um die Wirksamkeit zu messen, wurde eine umfassende Ablation Study durchgeführt, welche Ausführungszeiten stufenweise vergleicht.

2. In-Memory AST Evaluator
--------------------------

Die Berechnung erfolgt nun zur Laufzeit durch rekursives Traversieren des ASTs (``teir_evaluator.cpp``). 
Ursprünglich hat der Evaluator in jedem einzelnen Berechnungsschritt rekursiv Variablen über String-Hashmaps (``std::map``) sowie Strides aufgelöst. 

Da dies bei Milliarden von Rechenoperationen (z.B. Einsum) astronomischen Overhead verursachte und das System mit Out-of-Memory-Crashes zum Erliegen brachte, haben wir eine "Fast-Fallback"-Abkürzung eingebaut. Wenn das Speicherlayout einer Operation (z.B. Matmul) gutartig ist, werden die Daten-Offsets einmalig pro Ausführung aufgelöst und in rohe Pointer umgewandelt. Das ermöglicht es den tiefen C-Schleifen, im nativen Apple-Silicon den integrierten NEON-Coprozessor optimal auszureizen.

3. AST Optimierungen
--------------------

Der ``TEIROptimizer`` manipuliert den geladenen Knotenbaum direkt im Speicher, bevor dieser interpretiert wird. Wir haben zwei wesentliche Passes implementiert:

- **Cache Blocking:** Größere Schleifen über Tensor-Achsen werden in eine äußere Iteration und einen inneren Cache-Block zerlegt (z.B. Splitting durch Faktoren wie 2 oder 4). Dies geschieht durch tiefgreifende Knoten-Verlinkung und Erzeugung neuer ``Iteration``-Nodes in C++.
- **Parallelization:** Hierbei werden AST-Knoten auf der äußersten Iterationsebene mit einem Multi-Threading-Flag (OpenMP) assembliert, sodass der Evaluator diesen Baum parallel abarbeitet.

4. Ablation Study
-----------------

Die Test-Infrastruktur (``main.cpp``) durchläuft für jedes spezifizierte ``.teir``-Modell automatisch vier Ausführungs-Stufen (Stages) und prüft im Anschluss die Richtigkeit der berechneten Tensoren mittels eines Fehlertoleranz-Checks (``std::abs(a - b) < 1e-3``). 

- **Stage 0:** Unoptimierte Baseline
- **Stage 1:** Parallelisierung (OpenMP)
- **Stage 2:** Cache Blocking
- **Stage 3:** Combined (Parallelisierung + Cache Blocking)

**Erkenntnis:** Durch die tiefgreifende Apple-Hardware-Beschleunigung in unserer Baseline-Fallbackschnittstelle erzielen die AST-Optimierungen hier momentan keine großen Speedup-Vorteile (Speedup liegt meist bei ca. ~1.0x). Faktisch bedeutet Cache-Blocking auf C++ Ebene aktuell eher einen Störfaktor, da die Hardware durch zerteilte Arbeitsblöcke an den schnellen Vektor-Instruktionen gehindert wird, anstatt kohärente Datenmengen am Stück zu verarbeiten.

5. Makefile und Jupyter Analytics
---------------------------------

Um lästige Kopier-Aktionen zwischen Terminals zu verhindern, haben wir die Build-Pipeline mittels ``make run`` und ``make benchmark`` voll umfänglich automatisiert. Offene OpenMP-Bibliothekspfade wurden sauber in das MacOS Homebrew-Ecosystem eingebettet.

Die Ergebnisse der Ablation-Schleife werden direkt in Form von Log-Outputs generiert. Ein beiliegendes Jupyter Notebook (``benchmarks.ipynb``) extrahiert diese Terminal-Ausgaben über Subprocess-Aufrufe dynamisch und generiert detaillierte Balkendiagramme über die Performance-Multiplikatoren, welche auch abgebrochene Runs (wie bspw. OOM-Kills durch das OS) robust abfangen.

6. Contributions
----------------

In dieser Woche haben Justin Bergmann und Julian Müller alle Aufgaben eng im Team und zum Großteil im Pair-Programming gemeinsam bearbeitet. Es gab daher keine strikte Trennung, sondern eine gemeinsame Verantwortung für die gesamte Pipeline:

**Justin Bergmann & Julian Müller:**

- **Architektur & Implementierung:** Gemeinsame Entwicklung des In-Memory Evaluators (``teir_evaluator.cpp``) sowie Ausarbeitung der Fast-Fallback Kernel zur Performance-Steigerung und Vermeidung von OOM-Crashes.
- **AST & Optimierungen:** Kooperative Code-Gestaltung der TEIR-Optimizer Passes (``teir_optimizer.cpp``) für Cache-Blocking und Loop-Parallelisierung direkt auf AST-Ebene.
- **Testing & Ablation Study:** Abstimmung und Bau der Test-Infrastruktur in der ``main.cpp`` zur automatisierten Verifikation der 4 Stages.
- **Infrastruktur & Automatisierung:** Gemeinsames Troubleshooting der Build-Umgebung, das Fixen der Makefile-Abhängigkeiten und Aufbau der Jupyter-Notebook-Visualisierung (``benchmarks.ipynb``).
- **Dokumentation:** Geteiltes Verfassen des vorliegenden Reports, sowie gemeinsame Diskussion und Analyse der Performance-Resultate.
