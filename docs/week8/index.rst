Week 8:
================================================

1. Introduction
----------------

In Woche 8 werten wir den TEIR-Syntaxbaum direkt im Speicher aus, statt daraus ein eigenes
Programm zu erzeugen. Die Rechenkerne stammen weiterhin aus den JIT-Generatoren der Woche 6.

Das Ziel war es, Programmtransformationen (wie Cache Blocking und Loop-Parallelisierung) 
direkt auf der AST-Datenstruktur auszuführen (als Optimizer-Pass) und anschließend einen Interpreter darüberlaufen zu lassen. 
Um zu messen, wie viel diese Optimierungen tatsächlich bringen, haben wir die Ausführungszeiten der einzelnen Stufen schrittweise verglichen.

2. In-Memory AST Evaluator
--------------------------

Die Berechnung erfolgt zur Laufzeit, indem der AST schrittweise abgearbeitet wird
(``teir_evaluator.cpp``). Die Offsets der aktiven Schleifen werden dabei über die Achsennamen
aufgelöst.

**Kernel-Auswahl.** Für die Primitive verwenden wir die Codegeneratoren aus Woche 6, statt in der
Auswertung selbst zu rechnen. Vor dem ersten Lauf wird je Primitive ein Plan erstellt:

- ``Contraction`` über ``mini_jit::Gemm``, sofern ``out`` und ``in1`` N-zusammenhängend sind und
  M sowie N durch 16 teilbar sind. A wird in ein M-zusammenhängendes Layout gepackt.
- ``Zero`` und ``Copy`` über ``mini_jit::Unary``. Bei ``Copy`` bestimmt der Vergleich der
  Strides, ob der transponierende Pfad (``trans_b = 1``) gewählt wird.
- Passt das Layout nicht, greift eine NEON-Kachel mit ``M_R = 4`` und ``N_R = 16``, darunter eine
  skalare Schleife.

Der Plan wird einmal vor der Ausführung erzeugt und danach nur noch gelesen. Das ist die
Voraussetzung dafür, dass mehrere Threads gleichzeitig darauf zugreifen dürfen. Welcher Weg
tatsächlich gewählt wurde, gibt der Evaluator aus, damit einer Laufzeit anzusehen ist, ob der
SME-Kernel gegriffen hat oder ob zurückgefallen wurde.

3. AST Optimierungen
--------------------

Der ``TEIROptimizer`` manipuliert den geladenen Knotenbaum direkt im Speicher, bevor dieser
interpretiert wird. Alle fünf von TEIR vorgesehenen Transformationen sind umgesetzt:
``split_iteration``, ``fuse_iteration_nodes``, ``reorder_schedule_chain``, ``set_policy`` und
``promote_to_primitive``.

**Zielplattform-Parameter.** Die Heuristiken leiten ihre Faktoren aus den Eigenschaften der
Hardware ab, nicht aus den Namen der Modelle:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Parameter
     - Wert (M4 Max)
     - Quelle
   * - P-Kerne
     - 12
     - ``hw.perflevel0.physicalcpu``
   * - E-Kerne
     - 4
     - ``hw.perflevel1.physicalcpu``
   * - L1-Datencache
     - 65 536 Byte
     - ``hw.l1dcachesize``
   * - L2-Cache
     - 4 194 304 Byte
     - ``hw.l2cachesize``
   * - Cache-Line
     - 128 Byte
     - ``hw.cachelinesize``
   * - ZA-Kachel (fp32)
     - 16 x 16, vier Kacheln
     - SME, SVL 512 Bit

**Cache Blocking.** Gesucht wird die größte Kachelkante :math:`t` als Vielfaches von 16, für die
der Arbeitssatz noch in den L1-Datencache passt:

.. math::

    (t^2 + 2 \cdot t \cdot 16) \cdot 4 \leq 65536

Daraus ergibt sich :math:`t = 112`. Gesplittet werden nur Achsen mit der Rolle M oder N, deren
Extent größer als die Kachel und ohne Rest durch sie teilbar ist.

**Parallelisierung.** Es wird entlang der äußeren Schleifenkette abgestiegen und so lange
parallel geschaltet, bis das Produkt der Extents die Zahl der P-Kerne erreicht. Eine einzelne
Schwelle je Achse genügt nicht: Bei ``einsum`` sind die äußeren Achsen 4, 4 und 3 groß, zusammen
also 48 unabhängige Iterationen, von denen keine für sich allein 12 erreicht.

Zwei Bedingungen schließen Achsen aus:

- **Reduktionsachsen** (Rolle K) schreiben mehrfach auf dieselbe Ausgabezelle.
- **Achsen mit überlappenden Schreibbereichen.** Die Schrittweite der Achse in ``out`` muss
  größer sein als die Spanne, die alle weiter innen liegenden Achsen zusammen überdecken. Ist
  sie das nicht, hängt das Ergebnis von der Reihenfolge ab.

Bei ``transposition.teir`` greift die zweite Bedingung: Die Achse ``a`` hat in ``out`` die
Schrittweite 48, während ``b``, ``c`` und ``d`` zusammen 18 083 567 Elemente überdecken. Die
Datei bildet 18 874 368 Eingabewerte auf 18 088 128 Ausgabezellen ab, Kollisionen sind daher
unvermeidlich. Die Achse wird deshalb nicht parallelisiert.

Der Evaluator fasst eine Kette direkt geschachtelter paralleler Achsen zu einer einzigen
Schleife über das Produkt zusammen. Ohne das bliebe jede innere Ebene wirkungslos, weil OpenMP
geschachtelte Parallelität standardmäßig abschaltet.

4. Ablation Study & Performance Analysis
----------------------------------------

Die Test-Infrastruktur (``main.cpp``) durchläuft für jedes spezifizierte ``.teir``-Modell automatisch vier Ausführungs-Stufen und prüft im Anschluss die Richtigkeit der berechneten Tensoren mittels eines Fehlertoleranz-Checks (``std::abs(a - b) < 1e-3``). 

- **Stage 0:** Unoptimierte Baseline
- **Stage 1:** Parallelisierung (OpenMP)
- **Stage 2:** Cache Blocking
- **Stage 3:** Combined (Parallelisierung + Cache Blocking)

.. figure:: ablation_study_split.png
   :alt: Ablation Study Split Diagram
   :align: center
   :width: 100%

   Vergleich der relativen Speedups (Baseline = 1.0x) für alle Modelle in Stage 1, Stage 2 und Stage 3.

**Messwerte (Apple Silicon M4 Max, FP32, 16 Threads)**

Alle zwölf Konfigurationen bestehen die Verifikation gegen die unoptimierte Baseline.

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 19 19

   * - Workload
     - Baseline (Stage 0)
     - Durchsatz
     - Stage 1
     - Stage 3
   * - Matmul (8192^3)
     - 451,8 ms
     - 2434 GFLOPS
     - 1,00x
     - 1,00x
   * - Contraction
     - 2267,5 ms
     - 546 GFLOPS
     - 1,01x
     - 1,00x
   * - Einsum
     - 133 581 ms
     - 5,2 GFLOPS
     - **12,04x**
     - 11,74x
   * - Transposition
     - 102,7 ms
     - 1,47 GB/s
     - 0,98x
     - 0,97x

Bei ``einsum`` steigt der Durchsatz in Stage 1 von 5,2 auf 62,7 GFLOPS.

**Warum die Speedups so unterschiedlich ausfallen.** Die Wirkung eines Passes hängt davon ab,
was in der jeweiligen Datei noch zu holen ist:

- ``matmul.teir`` und ``contraction.teir`` markieren ihre äußeren Achsen bereits als
  ``parallel``. Die Heuristik findet dort nichts, was nicht schon gesetzt wäre, und die
  Abweichungen von 1,00x liegen im Rauschen.
- ``einsum.teir`` ist vollständig sequenziell und lässt sich über das Produkt der äußeren
  Achsen auf 16 Iterationen bringen. Das ist der einzige Fall mit einem echten Gewinn aus der
  Parallelisierung.
- ``transposition.teir`` wird aus dem oben genannten Grund nicht parallelisiert.

**Cache Blocking bringt bei keinem Modell einen Gewinn** (0,97x bis 1,00x). Der Engpass liegt
nicht in der Größe des Arbeitssatzes: Bei ``matmul`` und ``contraction`` arbeitet der SME-Kernel
ohnehin auf Kacheln, die in den L1 passen, und bei ``einsum`` verhindert das Layout den Einsatz
des Kernels. Der Pass ist implementiert und messbar, sein Nutzen auf diesen vier Modellen ist
jedoch nicht nachweisbar.

**Vergleich zu Woche 7.** Der Wert von 2434 GFLOPS für Matmul liegt nahe an den 2547 GFLOPS der
Laufzeitumgebung aus Woche 7, die dasselbe Problem mit denselben Kerneln rechnet. Die Differenz
von 4,4 Prozent entspricht dem Aufwand für die Baumauswertung. Der Abstand zwischen ``einsum``
mit 5,2 GFLOPS und ``matmul`` mit 2434 GFLOPS zeigt, was der SME-Kernel gegenüber dem
Rückfallweg ausmacht, gemessen im selben Programm.

5. Makefile und Jupyter Analytics
---------------------------------

Um Kopier-Aktionen zwischen Terminals zu verhindern, haben wir die Build-Pipeline mittels ``make run`` und ``make benchmark`` voll umfänglich automatisiert. Offene OpenMP-Bibliothekspfade wurden sauber in das MacOS Homebrew-Ecosystem eingebettet.

Die Ergebnisse der Ablation-Schleife werden direkt in Form von Log-Outputs generiert. Ein beiliegendes Jupyter Notebook (``benchmarks.ipynb``) extrahiert diese Terminal-Ausgaben über Subprocess-Aufrufe dynamisch und generiert detaillierte Balkendiagramme über die Performance-Multiplikatoren, welche auch abgebrochene Runs (wie bspw. OOM-Kills durch das OS) robust abfangen.

6. Contributions
----------------

In dieser Woche haben Justin Bergmann und Julian Müller alle Aufgaben eng im Team und zum Großteil im Pair-Programming gemeinsam bearbeitet. 
Es gab daher keine strikte Trennung, sondern eine gemeinsame Verantwortung für die gesamte Pipeline:

**Justin Bergmann & Julian Müller:**

- **Architektur & Implementierung:** Gemeinsame Entwicklung des In-Memory Evaluators (``teir_evaluator.cpp``) samt Anbindung der eigenen SME-Kernel aus Woche 6 und der Rückfallpfade.
- **AST & Optimierungen:** Kooperative Code-Gestaltung der TEIR-Optimizer Passes (``teir_optimizer.cpp``) für Cache-Blocking und Loop-Parallelisierung direkt auf AST-Ebene.
- **Testing & Ablation Study:** Bau der Test-Infrastruktur in ``main.cpp`` zur Verifikation der vier Stages sowie der Unit-Tests in ``test_week8.cpp`` (28 Prüfungen zu Kernel-Auswahl, Transformationen und Vergleich gegen eine unabhängige Referenz).
- **Infrastruktur & Automatisierung:** Gemeinsames Troubleshooting der Build-Umgebung, das Fixen der Makefile-Abhängigkeiten und Aufbau der Jupyter-Notebook-Visualisierung (``benchmarks.ipynb``).
- **Dokumentation:** Geteiltes Verfassen des vorliegenden Reports, sowie gemeinsame Diskussion und Analyse der Performance-Resultate.
