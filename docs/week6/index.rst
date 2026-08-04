Week 6:
====================

1. Introduction
----------------

In dieser Woche haben wir die in Woche 5 geschaffenen Grundlagen für die dynamische Code Generation (JIT) erweitert. 
Die generierten Kernel sind nun nicht mehr statisch auf eine feste Problemgröße limitiert, sondern vollständig parametrisierbar. 
Das Ziel war die Implementierung flexibler JIT-Generatoren für die Unary- und GEMM-Primitive, die skalierten Matrizen-Dimensionen (wie z.B. Vielfache von 16) zur Laufzeit nativ unterstützen.

2. Unary Primitives (SSVE)
--------------------------

Für die Unary-Primitive (``zero``, ``identity``, ``relu``) wurde die Codegenerierung so angepasst, dass ``M`` und ``N`` als beliebige Vielfache von 16 unterstützt werden. 

- Die Schleifengrenzen werden nun dynamisch während der Ausführung in die Hardware-Register geschrieben (z.B. per JIT emittierten ``mov x9, n``).
- Die Pointer für den Speicherzugriff auf die Arrays werden pro Durchlauf entsprechend der Stride-Längen korrekt mit dem berechneten Byte-Offset verschoben.
- Die in der ``main.cpp`` aufgesetzten Tests und Benchmarks über 9 verschiedene Konfigurationen (Kombinationen aus Matrixgrößen 64, 128, 512) liefen vollständig fehlerfrei ("PASS").

**Transponierung**

Über den Parameter ``trans_b`` lässt sich wählen, wie B im Speicher liegt. Bei ``trans_b = 0``
sind Spalten zusammenhängend, bei ``trans_b = 1`` Zeilen. Der Generator erzeugt für beide Fälle
eine eigene Schleife:

.. math::

    \text{trans\_b} = 0: \quad B(i,j) = b[i + j \cdot ld_b]

    \text{trans\_b} = 1: \quad B(i,j) = b[i \cdot ld_b + j]

Gelesen wird in beiden Fällen 16 Werte am Stück aus A. Bei ``trans_b = 1`` schreibt der Kernel
die Werte einzeln mit ``st1w`` und einem Zeigerinkrement von ``ld_b``, weil die Zielelemente im
Speicher nicht mehr benachbart sind. Umgesetzt ist das für ``identity`` und ``relu``. Für
``zero`` spielt die Anordnung keine Rolle, dort wird der Parameter ignoriert.

**Messwerte Unary (GiB/s)**

Angegeben ist der Median aus 15 Läufen, dazu die Spannweite (Maximum minus Minimum,
bezogen auf den Median) über dieselben Läufe.

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - Größe
     - Zero
     - Identity
     - ReLU
     - größte Spannweite
   * - 128x512
     - 225,1
     - 436,9
     - 303,2
     - 21,6 %
   * - 512x64
     - 202,0
     - 402,8
     - 302,0
     - 9,5 %
   * - 512x128
     - 204,3
     - 405,4
     - 303,6
     - 4,7 %
   * - 512x512
     - 204,7
     - 409,1
     - 303,9
     - 4,8 %

``identity`` liegt bei etwa dem Doppelten von ``zero``, weil beide dieselbe Anzahl Elemente
schreiben, ``identity`` sie aber zusätzlich liest und die gemessene Bandbreite beide Richtungen
zählt.

**Kleine Matrizen sind nicht belastbar messbar.** Die Spannweite wächst, je kleiner die Matrix
wird: bei 512x512 liegt sie unter 5 Prozent, bei 128x128 bereits bei 29 Prozent und bei 64x64
bei 52 Prozent. Der Kernel läuft dort so kurz, dass Aufrufaufwand, Zeitmessung und
Hintergrundlast das Ergebnis bestimmen und nicht mehr die Bandbreite. Für Vergleiche taugen
deshalb nur die Zeilen ab 512 in einer Dimension; die kleineren Größen sind in der Tabelle
bewusst nicht aufgeführt.

3. GEMM Primitive (SME)
-----------------------

Für das anspruchsvollere Matrizenmultiplikations-Primitive (GEMM) wurde in der ``Gemm.cpp`` der Code Generator so verallgemeinert, 
dass die Matrixgrößen ``M`` und ``N`` als Vielfache von 16 sowie eine beliebige Problemgröße für ``K`` unterstützt werden. 

- Das Einbinden variabler Parameter verlangte das dynamische Erstellen und Handling der Outer-, Middle- und Inner-Loops in Assembly. Der emittierte Maschinencode kalkuliert somit Laufzeit-Offsets über die übergebenen Stride-Argumente.
- Für die Validierung iterieren die Benchmarks über 27 Kombinationen (Variation von M, N, K jeweils für {64, 128, 512}).

**Akkumulation im ZA-Array**

Der Kernel lädt C zu Beginn in das ZA-Array, statt das Array mit ``zero {za}`` zu nullen.
``fmopa`` akkumuliert dann direkt auf den vorhandenen Werten, und am Ende wird das Ergebnis
einmal zurückgeschrieben. Ein Nullen mit anschließendem Addieren von C wäre ein zusätzlicher
Durchlauf über die Ausgabekachel. Für das Setzen von C auf Null gibt es den ``zero``-Kernel aus
Abschnitt 2, der an dieser Stelle nicht gebraucht wird.

Sind M und N durch 32 teilbar, verwendet der Kernel alle vier ZA-Kacheln als 32x32-Akkumulator.
Andernfalls fällt er auf eine einzelne 16x16-Kachel zurück:

.. code-block:: cpp

    const bool use32 = (m % 32 == 0) && (n % 32 == 0);

**Messwerte GEMM (GFLOPS)**

Median aus 15 Läufen. Die Spannweite liegt hier über alle 27 Konfigurationen zwischen 2,7 und
9,4 Prozent, die Werte sind also deutlich stabiler als die Unary-Messungen kleiner Matrizen.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - M x N
     - K = 64
     - K = 128
     - K = 512
   * - 64 x 64
     - 1324,7
     - 1589,4
     - 1867,0
   * - 64 x 128
     - 1322,4
     - 1580,6
     - 1868,3
   * - 64 x 512
     - 1317,8
     - 1588,1
     - 1847,8
   * - 128 x 64
     - 1328,0
     - 1587,2
     - 1872,9
   * - 128 x 128
     - 1324,7
     - 1578,0
     - 1876,6
   * - 128 x 512
     - 1320,9
     - 1583,4
     - 1872,7
   * - 512 x 64
     - 1129,7
     - 1455,5
     - 1827,8
   * - 512 x 128
     - 1146,5
     - 1461,7
     - 1822,2
   * - 512 x 512
     - 1137,4
     - 1457,8
     - 1796,0

.. figure:: benchmarks.png
   :alt: GEMM-Durchsatz über 27 Konfigurationen und Unary-Bandbreiten
   :align: center
   :width: 100%

   Links: GEMM-Durchsatz, gruppiert nach M x N mit je einer Säule pro K. Der graue Bereich
   markiert die in der Aufgabenstellung genannte Zielgröße von 1,5 bis 1,8 TFLOPS. Rechts:
   Bandbreite der Unary-Kernel.

Der Durchsatz hängt vor allem von K ab, nicht von M und N. Bei K = 64 liegt der Mittelwert bei
1261 GFLOPS, bei K = 128 bei 1542 und bei K = 512 bei 1850. Der Grund ist das Verhältnis von
Rechenarbeit zu Randaufwand: Das Laden von C in das ZA-Array und das Zurückschreiben fallen je
Ausgabekachel einmal an, unabhängig von K. Je mehr ``fmopa``-Schritte dazwischen liegen, desto
weniger fällt dieser Anteil ins Gewicht. Der höchste gemessene Median ist 1876,6 GFLOPS bei
128x128x512.

**Messbedingungen.** Alle Werte stammen von einem Apple M4 Max, übersetzt mit Apple clang 21
und den Flags aus ``week6/Makefile``. Je Messpunkt 15 Läufe, angegeben ist der Median. Die
Maschine war während der Messung nicht vollständig lastfrei; das erklärt einen Teil der
Spannweite, nicht aber deren Abhängigkeit von der Problemgröße.

4. Callee-Saved Registers (Lösung zum -O3 Bug)
----------------------------------------------

In der vorherigen Woche musste für korrekte Benchmark-Messwerte in C++ auf das ``-O0``-Flag (Optimierungsverzicht) zurückgegriffen werden. 
Grund war der SME Setup Command ``smstart sm``, der implizit das Vector-Register-Set (inklusive der FPR ``d8`` bis ``d15``) nullt, 
welche wiederum vom Compiler in der Host-Language für lokale Variablen genutzt wurden.

Dieses Problem auf der Ebene des Application Binary Interface wurde im JIT-Compiler dieser Woche
an der Quelle behoben:

- Die generierten Binary-Instruktionen beinhalten nun einen korrekten ARMv8 (AAPCS64) **Prologue und Epilogue**.
- Wichtige Callee-Saved Register (unter anderem ``d8``...``d15`` sowie für Pointer-Backups verwendete Register wie ``x19``...``x22``) werden vor der Kernel-Ausführung mittels ``stp`` (Store Pair) auf dem Stack gesichert. 
- Am Ende des Programms erfolgt über die passenden ``ldp`` (Load Pair) Anweisungen die vollständige Speicher-Restrukturierung.
- Die Benchmarks laufen dadurch mit voller ``-O3``-Optimierung, ohne dass Messvariablen überschrieben werden.

5. Contributions
----------------

In dieser Woche wurden die Aufgaben wie folgt zwischen uns aufgeteilt:

**Justin Bergmann:**

- Implementierung der dynamischen ``generate``-Funktion für die SME GEMM-Primitive (Unterstützung für variable Parameter M, N, K).
- Entwicklung und Auswertung der Test- und Benchmarking-Logik über 27 Settings (Ausgabe der Performance in GFLOPS).

**Julian Müller:**

- Implementierung der flexiblen SSVE Code-Generation (``generate``) für die Unary-Primitive (``zero``, ``identity``, ``relu``).
- Automatisierung der Unary-Tests und Bandbreiten-Benchmarks über 9 verschiedene Settings (Ausgabe in GiB/s).
