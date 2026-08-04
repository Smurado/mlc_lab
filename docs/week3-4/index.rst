Week 3-4:
====================

1. Introduction
----------------
Die dritte Woche drehte sich hauptsächlich um die ARM Scalable Matrix Extension (SME). Ziel war es, Kernel für Matrizen-Multiplikationen (GEMM) in AArch64 Assembly zu schreiben und diese nach und nach zu optimieren.
Außerdem haben wir GFLOPS-Benchmarks in C++ eingebaut, um die Performance-Unterschiede der einzelnen Optimierungsschritte gut vergleichen zu können. Die Unary Primitives für SSVE (Identity, Zero, ReLU) wurden wie in Task 1 gefordert ebenfalls umgesetzt und runden als Basis-Vorarbeit das Ganze ab.

2. SSVE Unary Primitives (Einfache Matrix-Operationen)
------------------------------------------------------

Für die Basis haben wir einfache unäre Operationen für 16x16 Matrizen geschrieben. Die nutzen **SSVE** (Streaming SVE), wodurch wir die vollen 512-bit Vektorregister ausnutzen können.

**Implementierte Kernel**

- ``identity_16_16``: Kopiert eine Matrix oder führt eine Transposition durch. Dabei wird
  horizontal in das ZA-Tile geladen (``za0h``) und vertikal wieder gespeichert (``za0v``).
- ``zero_16_16``: Setzt alle Elemente des Arrays auf Null.
- ``relu_16_16``: Berechnet :math:`\max(0, x)`. Bei transponierter Ausgabe erfolgt die
  Operation direkt beim Laden in das ZA-Tile.

**Messwerte**

Gemessen wird in GiB/s, weil diese Operationen durch den Speicher beziehungsweise den
L1-Cache begrenzt sind und nicht durch die Rechenwerke.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Funktion
     - Durchsatz
     - Mittlere Zeit
   * - ``identity_16_16``
     - 128,47 GiB/s
     - 0,01 µs
   * - ``zero_16_16``
     - 63,32 GiB/s
     - 0,02 µs
   * - ``relu_16_16``
     - 87,15 GiB/s
     - 0,02 µs

**Beobachtungen**

- Die Durchsätze liegen zwischen 63 und 128 GiB/s. ``identity_16_16`` erreicht den höchsten
  Wert, ``zero_16_16`` mit 63,32 GiB/s etwa die Hälfte davon. ``zero`` schreibt nur, während
  ``identity`` zusätzlich liest und die Messung beide Richtungen zählt.
- Der Weg über das ZA-Tile vermeidet einzelne Scatter-Stores. Das ist nicht nur schneller,
  sondern umgeht auch Ausrichtungsfehler, die sonst zu SIGILL führen können.

3. Implementation & Microkernels
--------------------------------
Die Matrizen-Multiplikationen wurden schrittweise in Assembly umgesetzt (``functions_sme.s``). Dadurch konnte man das Problem in kleinere Stücke zerteilen und Step by Step optimieren.

**Rank-1 Update (gemm_32_32_1)**

Macht im Grunde nur ein Rank-1 Update: ``C += A * B``.

- Verwendet noch keine Schleifen
- Lädt komplette Spalten von A und Reihen von B
- Macht die Multiply-Accumulate-Rechnung direkt im ZA-Array mittels Outer-Product (``fmopa``)

**K-Loop (gemm_32_32_512)**

Hier kommt eine Kachel-Schleife über K=512 auf C(32x32) dazu.

- Schleife läuft 512 Iterationen lang
- Wirft das Outer-Product kontinuierlich als Schleife im SME-Modus auf das ZA-Array

**M-Loop & N-Loop Wrappers (gemm_512_32_512 & gemm_512_512_512)**

Packen den K-Loop nochmal in äußere Schleifen über die Dimensionen M=512 und N=512 (bauen sie quasi ein).

4. Optimierung (smstart/smstop Overhead)
----------------------------------------
Die volle Matrix-Multiplikation ``gemm_512_512_512`` ruft die kleineren Sub-Kernel wiederholt auf. Jeder dieser Aufrufe führt ein eigenes ``smstart`` und ``smstop`` aus, bei 512x512 also 256 Wechsel des Hardware-Zustands pro Durchlauf. Jeder Wechsel kostet Zeit, in der nicht gerechnet wird.

Wir haben deshalb interne "Fast"-Varianten der Kernel geschrieben. Diese Hilfsfunktionen übernehmen den SME-Zustand des Aufrufers, statt ihn selbst zu setzen. Das ZA-Tile wird dadurch einmal zu Beginn aktiviert und am Ende wieder beendet, unabhängig von der Anzahl der Sub-Kernel-Aufrufe.

5. Performance Benchmarks
-------------------------
Gemessen wurde das Ganze in C++ mit ``std::chrono::high_resolution_clock`` über unsere Catch2-Testfälle. So sieht man auch direkt, ob neben der Performance auch das Ergebnis noch stimmt. 

Für den GFLOPS-Wert haben wir 2 FLOPs (Multiply + Add) pro Schleifendurchlauf angenommen. Die Messungen basieren auf einem Apple Silicon Chip in unserer Testumgebung:

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Kernel
     - GFLOPS
     - Anmerkung
   * - gemm_32_32_1
     - 6,09
     - ein einzelnes Rank-1-Update, die Laufzeit besteht überwiegend aus dem Auf- und Abbau des SME-Zustands
   * - gemm_32_32_512 (K-Loop)
     - 1658,48
     - Arbeitssatz liegt im Cache
   * - gemm_512_32_512 (M-Loop)
     - 1652,05
     - 0,4 Prozent unter dem K-Loop
   * - gemm_512_512_512 (voller N-Loop)
     - 1625,03
     - 2,0 Prozent unter dem K-Loop

Der Abstand zwischen ``gemm_32_32_1`` und den übrigen Kerneln zeigt die Kosten der
Zustandswechsel: Bei 512 K-Iterationen verteilt sich derselbe Aufwand auf 512-mal mehr
Rechenarbeit, der Durchsatz steigt um den Faktor 272. Von dort bis zur vollen
512x512x512-Multiplikation fällt der Wert nur noch um 2,0 Prozent, obwohl 256-mal mehr
Sub-Kernel-Aufrufe stattfinden. Ohne die "Fast"-Varianten aus Abschnitt 4 würde jeder
dieser Aufrufe einen eigenen Zustandswechsel auslösen.

6. Build
--------

Der Build läuft über das Makefile im Ordner ``week3-4``:

.. code-block:: bash

    make          # baut main_test
    make test     # baut und führt die Tests aus
    make clean

Dabei werden C++-Quellen und Assembly getrennt übersetzt. Die SME-Flags werden
ausschließlich beim Assemblieren der ``.s``-Dateien gesetzt:

.. code-block:: make

    CXXFLAGS = -std=c++17 -Wall -Wextra -O2
    ASMFLAGS = -march=armv9.2-a+sme

Der Grund ist eine Eigenheit von Apple Silicon: Der M4 implementiert SME, aber kein
eigenständiges SVE. SVE-Instruktionen sind nur innerhalb des Streaming-Modus gültig.
Übersetzt man ``main.cpp`` mit ``-march=armv9.2-a+sme``, hält clang SVE für allgemein
verfügbar und vektorisiert ab ``-O2`` auch den normalen C++-Code damit. Diese
Instruktionen laufen dann außerhalb des Streaming-Modus, und das Programm endet mit
SIGILL (Exit-Code 132).

Nachweis: ``clang++ -O2 -march=armv9.2-a+sme -S main.cpp`` erzeugt 29 SVE-Instruktionen,
bei ``-O0`` sind es keine.
