Week 6:
====================

1. Introduction
----------------

In dieser Woche haben wir die in Woche 5 geschaffenen Grundlagen für die dynamische Code Generation (JIT) erweitert. 
Die generierten Kernel sind nun nicht mehr statisch auf eine feste Problemgröße limitiert, sondern vollständig parametrisierbar. 
Das Ziel war die Implementierung flexibler JIT-Generatoren für die Unary- und GEMM-Primitive, die skalierten Matrizen-Dimensionen (wie z.B. Vielfache von 16) zur Laufzeit nativ unterstützen.

2. Unary Primitives (SSVE)
--------------------------

Für die Unary-Primitive (``zero``, ``identity``, ``relu``) wurde die Codegenerierung so angepasst, dass `M` und `N` als beliebige Vielfache von 16 unterstützt werden. 

- Die Schleifengrenzen werden nun dynamisch während der Ausführung in die Hardware-Register geschrieben (z.B. per JIT emittierten ``mov x9, n``).
- Die Pointer für den Speicherzugriff auf die Arrays werden pro Durchlauf entsprechend der Stride-Längen korrekt mit dem berechneten Byte-Offset verschoben.
- Die in der ``main.cpp`` aufgesetzten Tests und Benchmarks über 9 verschiedene Konfigurationen (Kombinationen aus Matrixgrößen 64, 128, 512) liefen vollständig fehlerfrei ("PASS"). 
- Die erzielte Performance der skalierbaren Unary-Kernel lag in den Benchmarks im Idealfall bei Werten um ca. 400 GiB/s.

3. GEMM Primitive (SME)
-----------------------

Für das anspruchsvollere Matrizenmultiplikations-Primitive (GEMM) wurde in der ``Gemm.cpp`` der Code Generator so verallgemeinert, 
dass die Matrixgrößen `M` und `N` als Vielfache von 16 sowie eine beliebige Problemgröße für `K` unterstützt werden. 

- Das Einbinden variabler Parameter verlangte das dynamische Erstellen und Handling der Outer-, Middle- und Inner-Loops in Assembly. Der emittierte Maschinencode kalkuliert somit Laufzeit-Offsets über die übergebenen Stride-Argumente.
- Für die Validierung iterieren die Benchmarks über 27 Kombinationen (Variation von M, N, K jeweils für {64, 128, 512}).
- Die Benchmark-Läufe offenbaren – gestützt durch die dedizierte Apple Silicon SME Hardware – konstante Metriken von rund 480-520 GFLOPS über alle Größenordnungen hinweg.

4. Callee-Saved Registers (Lösung zum -O3 Bug)
----------------------------------------------

In der vorherigen Woche musste für korrekte Benchmark-Messwerte in C++ auf das ``-O0``-Flag (Optimierungsverzicht) zurückgegriffen werden. 
Grund war der SME Setup Command ``smstart sm``, der implizit das Vector-Register-Set (inklusive der FPR ``d8`` bis ``d15``) nullt, 
welche wiederum vom Compiler in der Host-Language für lokale Variablen genutzt wurden.

Dieses konzeptionelle Problem auf der ABI (Application Binary Interface)-Ebene wurde im JIT-Compiler dieser Woche 
"professionell" behoben:    

- Die generierten Binary-Instruktionen beinhalten nun einen korrekten ARMv8 (AAPCS64) **Prologue und Epilogue**.
- Wichtige Callee-Saved Register (unter anderem ``d8``...``d15`` sowie für Pointer-Backups verwendete Register wie ``x19``...``x22``) werden vor der Kernel-Ausführung mittels ``stp`` (Store Pair) auf dem Stack gesichert. 
- Am Ende des Programms erfolgt über die passenden ``ldp`` (Load Pair) Anweisungen die vollständige Speicher-Restrukturierung.
- Dank dieser sauberen Stack-Operationen konnten die Performance-Benchmarks wieder mit voller ``-O3`` Compiler-Optimierung durchgeführt werden, ohne Laufzeitabstürze oder berechnete "NaN"-Fehler bei Variablen riskieren zu müssen.

5. Contributions
----------------

In dieser Woche wurden die Aufgaben wie folgt zwischen uns aufgeteilt:

**Justin Bergmann:**

- Implementierung der dynamischen ``generate``Funktion für die SME GEMM-Primitive (Unterstützung für variable Parameter M, N, K).
- Entwicklung und Auswertung der Test- und Benchmarking-Logik über 27 Settings (Ausgabe der Performance in GFLOPS).

**Julian Müller:**

- Implementierung der flexiblen SSVE Code-Generation (``generate``) für die Unary-Primitive (``zero``, ``identity``, ``relu``).
- Automatisierung der Unary-Tests und Bandbreiten-Benchmarks über 9 verschiedene Settings (Ausgabe in GiB/s).
