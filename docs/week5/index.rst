Week 5:
====================

1. Introduction
----------------

In dieser Woche lag der Fokus auf Code Generation zur Laufzeit (Runtime Code Generation). 
Ziel war es, unsere bisherigen ARM SME-Kernel nicht mehr statisch aus Assembly-Dateien aufzurufen, 
sondern den Maschinencode dynamisch im C++ Programm zu generieren und im Speicher auszuführen. 
Die in Woche3-4 geschriebenen Unary- und GEMM-Kernel wurden dafür in entsprechende Code-Generator-Klassen überführt.

2. Code Generation Basis
------------------------

Um Maschinencode zur Laufzeit auszuführen, muss zunächst geeigneter Speicher bereitgestellt werden. 
Die Code-Generator-Klasse nutzt dafür ``mmap`` in Kombination mit den Flags ``PROT_READ | PROT_WRITE``. 
Sobald die hexadezimalen Opcodes in diesen Bereich übertragen wurden, werden die Rechte über 
``mprotect`` auf ``PROT_READ | PROT_EXEC`` geändert. Danach lässt sich der Speicherbereich über einen Funktionszeiger 
als nativer Maschinencode ausführen.

3. Portierung der Kernel
------------------------

Die in den vorherigen Wochen geschriebenen ARM Assembly-Instruktionen wurden 
in den passenden hexadezimalen Maschinencode umgewandelt.

- **Unary Kernel**: Hier werden die Schleifen dynamisch je nach Parameter in den Maschinencode eingefügt. 
  Die relativen Sprungadressen (Offsets) für Branches mussten dafür zur Laufzeit genau berechnet werden, 
  um eine korrekte Ausführung sicherzustellen.
- **GEMM Kernel**: Für die Matrizen-Multiplikation (512x512x512) wurden die 114 Instruktionen exakt in Hexadezimal-Werte 
  übersetzt und in Serie per emit-Funktion geladen.

4. Benchmarks und Fehleranalyse
-------------------------------

**Symptom.** Das Benchmarking in der ``main.cpp`` lieferte ``0.00 GiB/s`` und ``nan GFLOPS``.

**Ursache.** Die Instruktion ``smstart sm`` aktiviert nicht nur die Matrix-Features, sondern
setzt laut ARM-Architektur alle Vektorregister zurück. Dazu gehören ``d8`` bis ``d15``, die
nach AAPCS64 vom Aufgerufenen zu sichern sind. Mit ``-O3`` legt der Compiler die lokalen
Variablen der Zeitmessung genau dort ab. Nach dem Kernel-Aufruf war die gemessene Zeitdifferenz
0, die Division dadurch undefiniert.

**Änderung.** Der generierte Kernel sichert ``d8`` bis ``d15`` selbst, wie es die
Aufrufkonvention verlangt. Prolog und Epilog werden vom Generator mit ausgegeben:

.. code-block:: cpp

    emit(0x6DBC27E8);   // stp d8,  d9,  [sp, #-64]!
    emit(0x6D012FEA);   // stp d10, d11, [sp, #16]
    emit(0x6D0237EC);   // stp d12, d13, [sp, #32]
    emit(0x6D033FEE);   // stp d14, d15, [sp, #48]
    // ... Kernel ...
    emit(0x6CC427E8);   // ldp d8,  d9,  [sp], #64

Damit liegt die Verantwortung dort, wo die Aufrufkonvention sie vorsieht, und der Aufrufer
braucht keine Sonderbehandlung. Der GEMM-Kernel besteht aus einem festen Instruktionsfeld mit
bereits berechneten Sprungweiten, das sich nicht ohne Verschieben der Offsets erweitern lässt.
Er wird deshalb über eine Inline-Assembly-Hülle aufgerufen, die ``d8`` bis ``d15`` als
zerstört deklariert. Der Compiler sichert die betroffenen Variablen dann von sich aus.

**Nachweis.** Beide Kernel liefern reproduzierbare Werte, und die Benchmark-Funktionen laufen
mit voller Optimierung. Eine Übersetzung mit ``-O0`` ist nicht mehr nötig.

Die Zeitmessung besteht aus einer inneren Schleife über viele Ausführungen und einem Mittel
über 10 unabhängige Durchläufe. Das begrenzt den Einfluss von System-Interrupts, Thermal
Throttling und Cache-Evictions auf das Ergebnis.

5. Unit Tests
--------------

Die Tests liegen in ``week5/main_test.cpp`` und laufen über ``make test``. Sie prüfen die
generierten Kernel gegen eine naive C++-Referenz, nicht gegen fest eingetragene Erwartungswerte.

- **Unary**: ``identity``, ``relu`` und ``zero`` über verschiedene Matrixgrößen, jeweils
  elementweise gegen die Referenz. Geprüft wird zusätzlich mit einer Leading Dimension, die
  größer als die Matrixbreite ist, damit ein Kernel auffällt, der die Zeilenlänge mit dem
  Speicherabstand verwechselt.
- **GEMM**: 512x512x512 gegen eine dreifach verschachtelte Referenzschleife.

Die Eingaben sind gemusterte Werte, keine Konstanten. Eine Füllung mit lauter gleichen Zahlen
liefert bei vertauschten Indizes dasselbe Ergebnis wie bei richtigen und würde einen
Layout-Fehler nicht anzeigen.

Insgesamt umfasst die Testsuite 524 347 Assertions in 6 Testfällen.

6. Lessons Learned
------------------

Die Lessons Learned sollen dazu genutzt werden, auf Probleme hinzuweisen, die wir bei der Entwicklung der Runtime Code Generation und der Benchmarks hatten.

- **Hardware-State & Calling Conventions:** Ein direkter Sprung auf die native Hardwareebene kann zu unerwarteten Konflikten bei den *Calling Conventions* führen. 
  Man darf sich nicht darauf verlassen, dass Register ihre Werte behalten, wenn man systemnahe Status-Instruktionen (wie ``smstart sm``) ausführt.

- **Code Generation vs Static Compilation:** Sobald Maschinencode zur Laufzeit erzeugt wird,
  kennt der C++-Compiler dessen Wirkung auf die Register nicht. Der generierte Code muss die
  Aufrufkonvention deshalb selbst einhalten. Wo das nicht möglich ist, etwa bei einem festen
  Instruktionsfeld mit berechneten Sprungweiten, muss die Aufrufstelle die betroffenen Register
  als zerstört deklarieren. Die Optimierung des umgebenden C++-Codes abzuschalten behebt das
  Symptom, aber nicht die Ursache.

- **Verzerrungen im Benchmarking:** Eine noch so große Anzahl an Loop-Iterationen ist unzureichend, 
  wenn sie fortlaufend (am selben Datenblock) ausgeführt wird und gerade ein L3-Cache-Miss oder ein Kontextwechsel des Betriebssystems passiert. 
  Das Wiederholen in mehreren unabhängigen Blöcken (Avg of 10) ist zwingend nötig für verlässliche GFLOPS/GiB-Werte.

7. Aufgabenverteilung
---------------------

Die Umsetzung der fünften Woche wurde im Team folgendermassen bearbeitet:

**Teammitglied 1: Justin Bergmann**
- Implementierung der ``mmap`` und ``mprotect`` Speicherverwaltung für ausführbaren Code
- Umsetzung der präzisen Benchmark-Logik (10-Run Average)
- Analyse und Bug-Fix des Callee-Saved Register-Resets

**Teammitglied 2: Julian Müller**
- Übersetzung der ARM Assembly-Opcodes in Hexadezimal-Maschinencode
- Dynamische Berechnung der relativen Branch-Offsets für die Schleifen im Unary-Kernel
- Transfer des GEMM-Kernels direkt in das C++ Emit-Muster