Week 3-4:
====================

1. Introduction
----------------
Die dritte Woche drehte sich hauptsächlich um die ARM Scalable Matrix Extension (SME). Ziel war es, Kernel für Matrizen-Multiplikationen (GEMM) in AArch64 Assembly zu schreiben und diese nach und nach zu optimieren.
Außerdem haben wir GFLOPS-Benchmarks in C++ eingebaut, um die Performance-Unterschiede der einzelnen Optimierungsschritte gut vergleichen zu können. Die Unary Primitives für SSVE (Identity, Zero, ReLU) wurden wie in Task 1 gefordert ebenfalls umgesetzt und runden als Basis-Vorarbeit das Ganze ab.

2. SSVE Unary Primitives (Einfache Matrix-Operationen)
----------------

Für die Basis haben wir einfache unäre Operationen für 16x16 Matrizen geschrieben. Die nutzen **SSVE** (Streaming SVE), wodurch wir die vollen 512-bit Vektorregister ausnutzen können.

### Implementierte Kernels
- **identity_16_16**: Kopiert eine Matrix oder führt eine Transposition durch. Hier kann man schön den SME-Trick mit horizontalem Laden (`za0h`) und vertikalem Speichern (`za0v`) nutzen.
- **zero_16_16**: Setzt einfach alle Elemente im Array auf Null.
- **relu_16_16**: Berechnet die ReLU-Aktivierung ($max(0, x)$). Wenn transponiert wird, passiert die Operation direkt beim Laden in das ZA-Tile.

### Performance-Ergebnisse
Hier messen wir in **GiB/s**, weil diese Operationen eigentlich fast immer vom Speicher bzw. L1-Cache ausgebremst werden (Memory Bound).

| Funktion | Durchsatz (GiB/s) | Durchschnittliche Zeit |
| :--- | :--- | :--- |
| **identity_16_16** | **128.47 GiB/s** | 0.01 µs |
| **zero_16_16**     | **63.32 GiB/s** | 0.02 µs |
| **relu_16_16**     | **87.15 GiB/s** | 0.02 µs |

### Beobachtungen & Besonderheiten
- **Speicherauslastung**: Die gemessenen Durchsätze liegen im Bereich von ca. 63 bis 128 GiB/s. `identity_16_16` schneidet hier mit knapp über 128 GiB/s am besten ab.
- **SME Transposition**: Durch das Laden/Speichern über das ZA-Tile umgehen wir langsame Scatter/Store-Instruktionen. Das macht es nicht nur schneller, sondern verhindert auch eklige Abstürze (`SIGILL`), die auf dem Mac bei fehlerhaftem Memory-Alignment gerne mal auftreten.

3. Implementation & Microkernels
------------------
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
-------------------------
Wir haben ziemlich schnell gemerkt, dass die volle Matrix-Multiplikation ``gemm_512_512_512`` total ausgebremst wird. Grund dafür: Durch das wiederholte Aufrufen der kleineren Sub-Kernel wurde in jedem Durchlauf (256 mal!) ``smstart`` und ``smstop`` getriggert. Solche ständigen State-Wechsel der Hardware fressen ordentlich Performance.

Als simplen Fix haben wir interne "Fast"-Varianten der Kernels geschrieben. Die Idee ist, dass die Hilfsfunktionen den SME-State des Callers einfach übernehmen. So müssen wir das ZA-Tile nur einmal ganz am Anfang hochfahren und am Ende wieder beenden.

5. Performance Benchmarks
----------------------
Gemessen wurde das Ganze in C++ mit ``std::chrono::high_resolution_clock`` über unsere Catch2-Testfälle. So sieht man auch direkt, ob neben der Performance auch das Ergebnis noch stimmt. 

Für den GFLOPS-Wert haben wir 2 FLOPs (Multiply + Add) pro Schleifendurchlauf angenommen. Die Messungen basieren auf einem Apple Silicon Chip in unserer Testumgebung:

- **gemm_32_32_1**: Recht bescheidene **6.09 GFLOPS**, aber das liegt einfach daran, dass die Funktion quasi nur State aufbaut und direkt wieder schliesst (Setup zu Teuer für die kurze Laufzeit).
- **gemm_32_32_512 (K-Loop)**: Sehr hohe Performance von **1658.48 GFLOPS**. Hier passiert fast alles im Cache, weshalb die Hardware perfekt ausgereizt wird.
- **gemm_512_32_512 (M-Loop)**: Erreicht rund **1652.05 GFLOPS**.
- **gemm_512_512_512 (Voller N-Loop)**: Hier macht sich unser Fix für den `smstart`-Overhead stark bemerkbar. Die vollständige Matrixberechnung erreicht extrem starke **1625.03 GFLOPS**.
