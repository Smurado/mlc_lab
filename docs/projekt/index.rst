Projekt: TEIR-Autotuner
=======================

.. note::

   **Gerüst, noch kein fertiger Bericht.** Die Tabellen und Kennzahlen unten sind gemessen
   und belegt. Die Fließtexte fehlen bewusst: Sie sind selbst zu schreiben, GenAI ist laut
   Kursvorgabe nur zum Korrekturlesen zugelassen, und die Nutzung ist offenzulegen.

   Jeder Abschnitt beginnt mit einem Hinweis, was dort hineingehört und welche Belege
   vorliegen. Diese Hinweise sind vor der Abgabe zu entfernen.

1. Einleitung
-------------

.. hint::

   Was gehört hinein: Problem (Tensorkontraktionen sind der Rechenkern moderner
   Netze, die passende Schleifenreihenfolge hängt von Form und Hardware ab), Ziel des
   Projekts, Abgrenzung zu den Wochenaufgaben. Zwei bis drei Absätze.

2. Grundlagen
-------------

.. hint::

   Einstein-Notation, TEIR als Zwischendarstellung, ARM SME und das ZA-Array. Kurz
   halten und auf die Wochenberichte verweisen, statt sie zu wiederholen.

3. Design und Architektur
-------------------------

.. hint::

   Pipeline von der Eingabe bis zum ausgeführten Kernel: Parser, IR, Transformationen,
   Cost-Modell als Vorfilter, Suchstrategie, Codegenerator, Messschleife, Validierung.
   Ein Diagramm hilft hier mehr als Prosa.

**Belegte Eckdaten**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Eigenschaft
     - Wert
   * - Suchstrategien
     - Simulated Annealing, genetischer Algorithmus, Zufallssuche
   * - Warmstart
     - Startpunkt ist das Optimum des Cost-Modells
   * - Backends
     - Scalar, NEON, SME
   * - Zielplattform
     - Apple M4 Max, 12 P-Kerne, 4 E-Kerne, L1d 64 KiB, L2 4 MiB

4. Methodik
-----------

.. hint::

   Dieser Abschnitt trägt den Bericht. Er erklärt, warum den Zahlen zu trauen ist.
   Reihenfolge je Befund: Symptom, Ursache mit Beleg, Änderung, Nachweis.

4.1 Der Fehler in der Messschleife
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Die GETT-Matrix lieferte **0 von 144** Läufen mit Status ``ok``
   * - Ursache
     - Der Zeitdeckel wurde erst **nach 64 Kernel-Aufrufen** geprüft. Bei einem
       Kernel von 763 ms dauerte eine Endmessung dadurch 244 s statt 3,8 s
   * - Änderung
     - Adaptive Blockgröße, über ``TEIR_BENCH_ADAPTIVE=1`` zuschaltbar,
       Default bleibt das alte Verhalten
   * - Nachweis
     - Bis **64-fach** schnellere Messung; Abweichung der Messwerte über sieben
       Vergleichspunkte **-2,3 % bis +1,8 % mit wechselndem Vorzeichen**

Die wechselnden Vorzeichen sind der eigentliche Beleg: Ein systematischer Fehler hätte
ein festes Vorzeichen. Die verbleibende Unsicherheit liegt bei rund 2 Prozent.

.. hint::

   Konsequenz, die hier hingehört: Aussagen, die auf Unterschieden unter 3 Prozent
   beruhen, dürfen die beiden Messverfahren nicht mischen.

4.2 Streuung und Wiederholungen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::

   Aus den Wochenberichten übernehmen: Messpunkte mit kurzer Laufzeit streuen stark
   (bis 85 Prozent bei 16x16-Kernen), große Probleme liegen unter 10 Prozent. Daraus
   folgt die Entscheidung, Mediane aus mehreren Läufen anzugeben statt Einzelwerte.

4.3 Validierung der Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::

   Vergleich gegen eine naive Referenz statt gegen Konstanten; Stichprobenprüfung bei
   großen Fällen. Belege aus dem Projekt: ``test_kernel_validation.cpp`` weist nach,
   dass verfälschte Elemente, transponierte Ausgaben und Nullausgaben erkannt werden.

   Hier gehört auch das wiederkehrende Muster hinein: **Dreimal bestand eine Prüfung
   nur deshalb, weil eine andere Stelle den Fehler verdeckte.** Konstante Testfüllung
   in Woche 5, nie gesetztes ``trans_b`` in Woche 6, und ein Zero-Kernel in Woche 8,
   der zu wenig nullte, was durch ein vorheriges ``memset`` unsichtbar blieb.

5. Ergebnisse
-------------

5.1 Hero-Fall
^^^^^^^^^^^^^

``ab-ac-cb`` bei 512³, Simulated Annealing, 30 Trials, Seed 42, adaptiver Messblock.
Rohdaten: ``Projekt/autotuner/results/hero_sme.json``.

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Variante
     - GFLOPS
     - Suchdauer
   * - naiv, ungetunt, Scalar
     - 3,1
     - 1 s
   * - TEIR getunt, Scalar
     - 2,3
     - 602 s
   * - TEIR getunt, NEON
     - 35,2
     - 7 s
   * - TEIR getunt, SME
     - **1428,3**
     - 2 s
   * - TVM, MetaSchedule
     - 505,1
     - übernommen
   * - PyTorch, BLAS/AMX
     - 2370,9
     - übernommen

.. hint::

   Zwei Aussagen, die hier ausformuliert gehören:

   1. Mit SME liegt TEIR bei 2,8-fachem TVM und erreicht 60 Prozent von PyTorch. Die
      frühere Aussage "6 bis 17 Prozent von TVM" galt ausschließlich für NEON.
   2. **Die Suche kann schaden.** Mit Scalar-Backend liefert die getunte Konfiguration
      2,3 GFLOPS gegen 3,1 ungetunt, nach 602 Sekunden Suche. Das ist kein Ausreißer,
      auch die Regressionstests zeigen Faktoren von 0,86 bis 0,99. Mögliche Ursachen:
      schwacher Cost-Modell-Vorfilter und ein Warmstart, der die Suche um einen
      schlechten Punkt kreisen lässt.

5.2 Einordnung gegen die Wochenaufgaben
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alle Werte auf derselben Maschine und im selben Semester gemessen.

.. list-table::
   :header-rows: 1
   :widths: 45 30 25

   * - Quelle
     - Kennzahl
     - Wert
   * - Woche 3+4, handgeschriebene SME-Assembly
     - GEMM 512³, Median aus 15 Läufen
     - 1655 GFLOPS
   * - Woche 6, Codegenerator
     - Spitze über 27 Einstellungen
     - 1877 GFLOPS
   * - Woche 7, TEIR-Laufzeitumgebung
     - matmul 8192³, 16 Threads
     - 2547 GFLOPS
   * - Woche 8, AST-Evaluator
     - matmul 8192³
     - 2434 GFLOPS
   * - **Projekt-Autotuner mit SME**
     - ``ab-ac-cb`` bei 512³
     - **1428 GFLOPS**

.. hint::

   Der Autotuner liegt in derselben Größenordnung wie die handoptimierte Assembly. Das
   ist die stärkere Aussage gegenüber der früheren Fassung. Ehrlich dazusagen: Woche 7
   und 8 lösen ein anderes, größeres Problem und sind deshalb nicht direkt vergleichbar.

5.3 Cost-Modell und Suchstrategien
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::

   Belege, die vorliegen: Rangkorrelation des Cost-Modells rund 0,08; in der Ablation
   kostet der Vorfilter etwa 6 Prozent Endqualität. Dazu der Vergleich SA gegen GA
   gegen Zufall und die Konvergenzkurven aus ``eval/notebooks``.

6. Limitierungen
----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Einschränkung
     - Belegt durch
   * - SME nur bei GEMM-Form und M, N durch 32 teilbar
     - 1 von 48 Fällen der GETT-Matrix
   * - Das Backend wird nicht gesucht, es ist eine Vorgabe
     - ``backend`` ist Teil von ``AutotunerOptions``, nicht der ``TuningConfig``
   * - Die Suche kann das Ergebnis verschlechtern
     - Hero-Fall Scalar: 2,3 gegen 3,1 GFLOPS
   * - Cost-Modell trägt wenig
     - Rangkorrelation 0,08, Ablation rund 6 Prozent
   * - Suchraum-Abdeckung
     - 30 Trials decken 2,9 Prozent bei 4 Achsen ab, aber nur 0,026 Prozent bei 7
   * - 12 schwere GETT-Fälle nicht vollständig getunt
     - 459 bis 524 Mrd. Iterationen, ein Trial kostet dort Minuten
   * - Die CI belegt die SME-Korrektheit nicht
     - GitHub-Runner haben kein SME, die Tests laufen nur lokal auf dem M4

7. Fazit
--------

.. hint::

   Kurz halten. Was funktioniert, was nicht, und was der nächste Schritt wäre.

8. Offenlegung der GenAI-Nutzung
--------------------------------

.. hint::

   Kursvorgabe: weniger als eine halbe Seite. Konkret benennen, wofür ein Assistent
   eingesetzt wurde und wofür nicht. Zu diesem Bericht gehört mindestens: Fehlersuche
   und Analyse im Code, Aufbereitung der Messdaten, Erstellung der Abbildungen und
   dieses Kapitelgerüst.
