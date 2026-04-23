Week 2:
====================

1. Introduction
----------------
Die Aufgabe in Woche 2 bestand aus zwei Teilen:

1. **Execution Throughput** - Microbenchmarks für den Befehls-Durchsatz einzelner
   Fließkomma-Instruktionen:

   - ``FMADD`` (scalar), FP32-Variante
   - Optional: ``FMLA`` (vector) mit Arrangement ``4S``
   - Optional: ``FMLA`` (vector) mit Arrangement ``2S``

2. **Permutation** - Implementierung eines NEON-Kernels, der eine Tensor-Permutation
   :math:`abc \rightarrow cba` durchführt (row-major, :math:`|a| = 8`, :math:`|b| = 4`,
   :math:`|c|` als Parameter). Anschließend Messung der Bandbreite in GiB/s in
   Abhängigkeit von :math:`|c|`.

2. Execution Throughput
------------------------
Für jede Instruktion gibt es in ``functions.s`` eine eigene Benchmark-Funktion.
Pro Iteration werden ``50 * 28 = 1400`` unabhängige Instruktionen ausgeführt
(50-fach unrolled, 28 unterschiedliche Zielregister). Dadurch entstehen keine
Abhängigkeitsketten zwischen aufeinanderfolgenden Instruktionen und der
Prozessor kann mehrere FP-Pipelines parallel füllen.

**Aufbau der Benchmark-Schleife (FMADD scalar)**

Hier ist der grundlegende Aufbau der Benchmark-Schleife:

Pseudocode::

    for iter in range(iterations):
        for _ in range(50):                # .rep 50 (unroll)
            fmadd s0,  s29, s30, s31
            fmadd s1,  s29, s30, s31
            ...
            fmadd s27, s29, s30, s31       # 28 unabhaengige Zielregister

Für ``FMLA 4S`` und ``FMLA 2S`` sieht die Struktur identisch aus, nur dass
statt ``fmadd sN, ...`` Vektor-Instruktionen der Form ``fmla vN.4s, v29.4s, v30.4s``
bzw. ``fmla vN.2s, v29.2s, v30.2s`` verwendet werden.

**FLOPs-Umrechnung**

- ``FMADD`` (scalar) = 2 FLOPs (1 mul + 1 add)
- ``FMLA v.4s``      = 8 FLOPs (4 Lanes × (1 mul + 1 add))
- ``FMLA v.2s``      = 4 FLOPs (2 Lanes × (1 mul + 1 add))

**Messwerte (Apple Silicon, -O2)**

Gemessen wurde auf einem Apple Silicon mit M4 Max.

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 25

   * - Instruktion
     - Iterationen
     - Zeit
     - Throughput
   * - FMADD (scalar)
     - 100 000 000
     - 8 140 ms
     - 34,4 GFLOP/s
   * - FMLA 4S (vector)
     - 50 000 000
     - 5 135 ms
     - 109,1 GFLOP/s
   * - FMLA 2S (vector)
     - 50 000 000
     - 4 749 ms
     - 59,0 GFLOP/s

Die 4S-Variante liefert erwartungsgemäß etwa das **Doppelte** der 2S-Rate:
beide Varianten sind instruktionsraten-limitiert, 4S bearbeitet aber pro
Instruktion viermal statt zweimal so viele Lanes. Die skalare Variante liegt
deutlich darunter, weil pro Instruktion nur eine Lane aktiv ist.


3. Permutation :math:`abc \rightarrow cba`
-------------------------------------------
Die Einsum-Definition lautet:

.. math::

    \text{cba}[c, b, a] = \text{abc}[a, b, c]

Die Werte werden also nur umsortiert, das heißt es wird nichts gerechnet oder
zusammengefasst. Aus der Form :math:`(A, B, C)` wird :math:`(C, B, A)`.

**Kernel-Idee**

Anstatt einfach drei verschachtelte Schleifen zu nutzen - was beim Schreiben riesige Speichersprünge von :math:`B \cdot A \cdot 4 = 128` Bytes zur Folge hätte und die Cache-Lines ignoriert - haben wir das Ganze blockweise mit NEON umgesetzt.

Wir iterieren außen über ``b`` und bearbeiten das ``c`` in 4er-Schritten. Wir laden für alle 8 ``a``-Zeilen mit ``ld1`` direkt 4 Werte auf einmal in die Register. So erhalten wir zwei 4x4-Blöcke, die wir anschließend mit den Befehlen ``trn1``/``trn2`` und ``zip1``/``zip2`` "umkippen". 

Durch das Transponieren liegen die Daten danach genau so in den Registern, dass wir für ein bestimmtes ``c`` alle 8 ``a``-Werte direkt mit einem einzigen ``stp q, q`` Befehl am Stück in ``cba`` wegschreiben können. 
Falls am Ende noch Elemente übrig bleiben (wenn :math:`|c|` nicht ohne Rest durch 4 teilbar ist), kopiert eine kurze skalare Schleife einfach den Rest.

Pseudocode::

    for b in range(B=4):
        c = 0
        while c + 4 <= C:
            # 8 Vektoren mit je 4 c-Werten laden (fuer a = 0..7)
            v0..v7 = load 4 c-Werte fuer a = 0..7
            # zwei 4x4-Transposes
            transponiere(v0..v3) -> v20..v23   # c0..c3, a=0..3
            transponiere(v4..v7) -> v28..v31   # c0..c3, a=4..7
            # 8-float-Bloecke an cba[c+k, b, 0..7] schreiben
            for k in 0..3:
                store_q_q(cba + (c+k)*B*A + b*A, v(20+k), v(28+k))
            c += 4
        while c < C:        # skalarer Tail
            for a in 0..7:
                cba[c, b, a] = abc[a, b, c]
            c += 1

**Korrektheit**

Der TEST_CASE ``[perm_neon_abc_cba]`` prüft jedes einzelne Element gegen die
Einsum-Formel und durchläuft dabei :math:`|c| \in \{1, 2, 4, 8, 15, 32\}`
(insgesamt 1984 Assertions). Zusätzlich gibt es einen
Visualisierungs-Test ``[perm_print]``, der ``abc`` und das permutierte ``cba``
für ein kleines Beispiel (``A=8, B=4, C=4``) ausdruckt.

**Bandbreiten-Messung**

Mit dem Benchmark ``[perm_bench]`` haben wir getestet, wie hoch die tatsächliche
Speicherbandbreite bei unterschiedlichen Größen für :math:`|c|` ist. Die bewegten
Daten (einmal lesen, einmal schreiben) berechnen sich so:

.. math::

    \text{bytes} = 2 \cdot |a| \cdot |b| \cdot |c| \cdot \text{sizeof(float)}

Damit die Messungen vergleichbar bleiben, passen wir die Anzahl der
Schleifendurchläufe immer so an, dass am Ende insgesamt knapp :math:`2^{28}`
Elemente verarbeitet werden. Außerdem machen wir vor der eigentlichen Zeitnahme
immer 3 Warmup-Calls.

.. list-table::
   :header-rows: 1
   :widths: 15 20 15 25

   * - :math:`|c|`
     - Wiederholungen
     - Zeit
     - Bandbreite
   * - 4
     - 2 097 152
     - 16,7 ms
     - 120,0 GiB/s
   * - 8
     - 1 048 576
     - 13,9 ms
     - 143,7 GiB/s
   * - 16
     - 524 288
     - 10,7 ms
     - 186,6 GiB/s
   * - 32
     - 262 144
     - 10,5 ms
     - 190,0 GiB/s
   * - 128
     - 65 536
     - 10,3 ms
     - 194,3 GiB/s
   * - 256
     - 32 768
     - 10,2 ms
     - **195,4 GiB/s** (Peak)
   * - 512
     - 16 384
     - 10,4 ms
     - 192,2 GiB/s
   * - 1 024
     - 8 192
     - 23,1 ms
     - 86,5 GiB/s
   * - 4 096
     - 2 048
     - 37,4 ms
     - 53,5 GiB/s
   * - 16 384
     - 512
     - 42,2 ms
     - 47,4 GiB/s
   * - 65 536
     - 128
     - 36,7 ms
     - 54,4 GiB/s
   * - 262 144
     - 32
     - 73,5 ms
     - 27,2 GiB/s

**Interpretation der Ergebnisse**

Wenn wir uns die gemessenen Werte anschauen, sieht man folgendes:

- Für Werte von :math:`|c|` zwischen 128 und 512 erreichen wir die höchste Bandbreite (den Peak) mit knapp **195 GiB/s**. Das liegt daran, dass beide Tensoren hier noch komplett in den extrem schnellen L1-Cache passen (sie sind da nur 32 bis 128 KiB groß). Der Kernel wird in dem Bereich also nur durch den L1-Cache limitiert.
- Sobald :math:`|c|` größer oder gleich 1024 wird, werden die Daten zu groß für den L1- und irgendwann auch für den L2-Cache. Das sieht man sofort, weil die Bandbreite stark einbricht und auf **30 bis 55 GiB/s** abfällt. Ab hier müssen die Daten aus dem deutlich langsameren Arbeitsspeicher (DRAM) geholt werden, was dann auch den Flaschenhals bildet.
- Bei kleinen Werten wie :math:`|c| = 4` oder :math:`|c| = 8` ist die reine Ausführung so schnell vorbei, dass der Overhead für den Funktionsaufruf und das Einrichten der Schleife zu stark ins Gewicht fallen. Deswegen erreichen wir da noch nicht den vollen Peak vom L1-Cache.


4. Unit Tests und Benchmarks mit Catch2
----------------------------------------
Alle Tests und Benchmarks liegen in ``week2/main.cpp`` und sind über Tags
getrennt aufrufbar:

.. code-block:: bash

    # Alles ausfuehren
    ./test_run

    # Nur Korrektheitstests
    ./test_run "[perm_neon_abc_cba]"

    # Nur einzelne Benchmarks
    ./test_run "[fmadd_bench]"
    ./test_run "[fmla4s_bench]"
    ./test_run "[fmla2s_bench]"
    ./test_run "[perm_bench]"

    # Visualisierung der Permutation
    ./test_run "[perm_print]"
