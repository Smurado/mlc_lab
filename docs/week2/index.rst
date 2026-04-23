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

Ein naiver Triple-Loop hätte zwar sequenzielle Reads aus ``abc``, müsste aber
in ``cba`` mit einem Schreib-Stride von :math:`B \cdot A \cdot 4 = 128` Bytes
zwischen aufeinanderfolgenden ``c``-Werten arbeiten - pro Store wird dann nur
ein Bruchteil einer Cache Line genutzt.

Stattdessen arbeiten wir in der NEON-Implementierung mit **4×4-Transpose-Tiles**:

- Äußerer Loop über ``b``.
- Innerer Loop über ``c`` in 4er-Blöcken.
- Für jede der 8 ``a``-Zeilen wird mit ``ld1 {v.4s}`` ein 4-Float-Vektor
  (vier aufeinanderfolgende ``c``-Werte) geladen.
- Zwei 4×4-Tiles (``a=0..3`` und ``a=4..7``) werden mit ``trn1``/``trn2`` und
  ``zip1``/``zip2`` transponiert. Danach enthält jedes Ziel-Register 4 ``a``-Werte
  für ein festes ``c``.
- Pro ``c`` schreibt ein einziges ``stp q, q`` 8 ``a``-Werte zusammenhängend
  nach ``cba``.
- Der Rest (:math:`|c| \bmod 4`) wird mit einem einfachen skalaren Loop kopiert.

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

Der Benchmark ``[perm_bench]`` misst für verschiedene :math:`|c|` die effektive
Speicherbandbreite:

.. math::

    \text{bytes} = 2 \cdot |a| \cdot |b| \cdot |c| \cdot \text{sizeof(float)}
                   \quad (\text{read + write})

Die Anzahl der Wiederholungen wird pro Messpunkt so gewählt, dass insgesamt
ca. :math:`2^{28}` Elemente berührt werden. Vor jeder Messung laufen 3
Warmup-Calls.

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

**Interpretation**

- :math:`|c| = 128 \ldots 512`: Beide Tensoren passen vollständig in den L1-Cache
  (Gesamtgröße 32-128 KiB pro Tensor). Hier erreichen wir den **Peak von ca.
  195 GiB/s** - der Kernel ist durch die L1-Bandbreite limitiert.
- :math:`|c| \geq 1024`: Arbeitsmenge sprengt L1 und später L2. Die Bandbreite
  fällt auf DRAM-nahe **30-55 GiB/s** - hier ist der Kernel klar
  speicherbandbreiten-limitiert.
- :math:`|c| = 4` und :math:`|c| = 8`: Pro-Call-Overhead (Prolog, Loop-Setup)
  dominiert, der L1-Peak wird noch nicht erreicht.


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

Die CI-Pipeline in ``.github/workflows/tests.yml`` überspringt die Benchmark-Tags
(``~[*_bench]``, ``~[perm_print]``), da diese unter QEMU-Emulation zu langsam
wären; geprüft werden dort ausschließlich die Korrektheitstests.
