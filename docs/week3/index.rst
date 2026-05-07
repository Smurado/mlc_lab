Week 3:
====================

1. Introduction
----------------
In Woche 3 stand die ARM Scalable Matrix Extension (SME) im Fokus. Das Ziel war die Implementierung und Optimierung von Kerneln fuer Matrizen-Multiplikationen (GEMM) in Hardware-nahem Assembly.
Zusaetzlich sollten die Benchmarks (GFLOPS) in C++ eingebaut werden, um die Performance der unterschiedlichen Optimierungsstufen zu evaluieren. Eine Umsetzung der *Unary Primitives* fuer SSVE wurde bewusst ausgelassen, da die Anforderung beschraenkt wurde.

2. SSVE Unary Primitives (Einfache Matrix-Operationen)
----------------

Zuerst wurden grundlegende unäre Operationen für 16x16 Matrizen implementiert. Diese nutzen **SSVE** (Streaming SVE), um die volle Vektorbreite der Hardware (512-bit auf Apple Silicon) auszuschöpfen.

### Implementierte Kernels
- **identity_16_16**: Kopiert eine Matrix oder führt eine Transposition durch. Hierbei wird die SME-Besonderheit genutzt, Daten horizontal zu laden (`za0h`) und vertikal zu speichern (`za0v`).
- **zero_16_16**: Setzt alle Matrixelemente effizient auf Null.
- **relu_16_16**: Berechnet die ReLU-Aktivierung ($max(0, x)$). Im Transpose-Pfad wird die Operation direkt beim Laden in das ZA-Array durchgeführt.

### Performance-Ergebnisse
Die Messung erfolgt in **GiB/s**, da diese Operationen primär durch die Speicherbandbreite des L1-Caches limitiert sind.

| Funktion | Durchsatz (GiB/s) | Durchschnittliche Zeit |
| :--- | :--- | :--- |
| **identity_16_16** | **126.70 GiB/s** | 0.02 µs |
| **relu_16_16** | **110.79 GiB/s** | 0.02 µs |
| **zero_16_16** | **62.25 GiB/s** | 0.02 µs |

### Analyse & Besonderheiten
- **Maximale Bandbreite**: Werte von über 110 GiB/s zeigen, dass die SSVE-Implementierung den Datenpfad des Prozessors nahezu vollständig sättigt.
- **SME Transposition**: Die hardwareseitige Transposition über das ZA-Tile vermeidet teure Scatter-Store-Befehle. Dies sorgt für eine stabile Performance und verhindert Abstürze (`SIGILL`), die bei nicht-standardkonformen SVE-Zugriffen auf Apple-Hardware auftreten können.
- **Zero-Durchsatz**: Der rechnerisch niedrigere Wert bei `zero_16_16` resultiert daraus, dass keine Quelldaten gelesen werden müssen. Da GiB/s den gesamten Datenfluss (Read + Write) beschreibt, halbiert sich der Wert gegenüber den bidirektionalen Kernels (Identity/ReLU) nominell.

3. Implementation & Microkernels
------------------
Die Implementierung der Matrizen-Multiplikation erfolgte inkrementell in AArch64 SME-Assembly (``functions_sme.s``).
Auf diese Weise laesst sich die GEMM-Problematik Schritt fuer Schritt optimieren.

**Rank-1 Update (gemm_32_32_1)**

Berechnet das Rank-1 Update einer Matrix: ``C += A * B``.

- Ohne Schleifen
- Laedt Spalten von A und Reihen von B
- FMA-Berechnung in das ZA-Array mittels Outer-Product (``fmopa``)

**K-Loop (gemm_32_32_512)**

Erweitert das Update um eine Kachel-Schleife ueber K=512 auf C(32x32).

- Initialisiert Schleife ueber die Laenge 512
- Durchgaengige Outer-Product Operation der Block-Slices im aktiven SME-Stadium

**M-Loop & N-Loop Wrappers (gemm_512_32_512 & gemm_512_512_512)**

Erweitern den K-Loop um weitere Dimensionen auf M=512 und N=512, jeweils als Wrappert-Funktionen.

4. Optimierung (smstart/smstop Overhead)
-------------------------
Bei der vollen Multiplikationsfunktion ``gemm_512_512_512`` wurde ein enormer Overhead festgestellt, da durch die wiederholten Aufrufe permanent ``smstart`` und ``smstop`` ausgefuehrt wurde (256 mal pro vollem Zyklus). Dies erzeugte erhebliche Moduswechselkosten in der Hardware.

Als Loesung wurden interne "Fast"-Varianten der Base-Funktionen geschrieben, bei denen die Subfunktionen den SME Status der Elternfunktion erben, was dazu fuehrte, dass das ZA-Array im Prozessor nur *ein* einziges mal hoch- und runtergefahren wird.

5. Performance Benchmarks
----------------------
Die finalen Messungen in C++ erfolgten mittels ``std::chrono::high_resolution_clock`` und Catch2-Testfaellen, um sicherzustellen, dass Korrektheit und Geschwindigkeit parallel ueberprueft werden. 

Bei der Berechnung wurden 2 FLOPs (Multiply-Add) pro Einzelaufruf angesetzt. Hier sind die resultierenden Werte bei Ausfuehrung unter ``-O3`` auf dem Apple Silicon System:

- **gemm_32_32_1**: Geringer GFLOPS Wert aufgrund des hohen Setup-Overheads bei kurzen Routinen (~10 GFLOPS).
- **gemm_32_32_512 (K-Loop)**: Exzellenter Wert dank optimalem Caching und sauberer FMA-Ratio (~893 GFLOPS).
- **gemm_512_512_512 (Voller N-Loop)**: Maximale Performance durch Vermeidung redundanter ``smstart`` Operationen. Der Benchmark erreichte ca. **1001 GFLOPS**.
