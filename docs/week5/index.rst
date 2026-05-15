Week 5:
====================

1. Introduction
----------------

In dieser Woche lag der Fokus auf Just-In-Time (JIT) Kompilierung. Ziel war es, unsere bisherigen ARM SME-Kernel nicht mehr logisch aus statischen Assembly-Dateien aufzurufen, sondern den Maschinencode zur Laufzeit direkt im C++ Programm zu generieren und im Speicher auszuführen. Die zuvor geschriebenen Unary- und GEMM-Kernel wurden dafür in JIT-Klassen überführt.

2. JIT Compiler Basis
----------------

Für die JIT-Ausführung muss zunächst Speicher bereitgestellt werden, der beschrieben und ausgeführt werden darf. Die JIT-Klasse nutzt dafür ``mmap`` in Kombination mit den Flags ``PROT_READ | PROT_WRITE``. Sobald die hexadezimalen Opcodes der Instruktionen in diesen Bereich übertragen wurden, werden die Rechte über ``mprotect`` auf ``PROT_READ | PROT_EXEC`` geändert. Danach lässt sich der Speicherbereich über einen Funktionszeiger regulär ausführen.

3. Portierung der Kernel
----------------

Die in den vorherigen Wochen geschriebenen ARM Assembly-Instruktionen wurden in den passenden hexadezimalen Maschinencode umgewandelt.

- **Unary Kernel**: Hier werden die Schleifen dynamisch je nach Parameter in den Maschinencode eingefügt. Die relativen Sprungadressen (Offsets) für Branches mussten dafür zur Laufzeit genau berechnet werden, um saubere Durchläufe sicherzustellen.
- **GEMM Kernel**: Für die Matrizen-Multiplikation (512x512x512) wurden die 114 Instruktionen exakt in Hexadezimal-Werte übersetzt und in Serie per emit-Funktion geladen. Da wir uns in dieser Stufe noch auf eine feste feste Problemgrösse beschränken, war hier vorerst keine dynamische Offset-Berechnung innerhalb der Schleifen nötig.

4. Benchmarks und Fehleranalyse
----------------

Beim Benchmarking in der ``main.cpp`` trat zunächst ein interessantes Phänomen auf: Die Ergebnisse zeigten konsequent ``0.00 GiB/s`` und ``nan GFLOPS``. 

Die Ursache lag in einer Wechselwirkung zwischen dem C++ Compiler und der SME-Hardware: Die Hardware-Instruktion ``smstart sm`` zur Aktivierung des SME-Modus fährt nicht nur die Matrix-Features hoch, sondern leert dabei (wie von der ARM Architektur vorgegeben) sicherheitshalber sämtliche Vector-Register. Dazu gehören auch die "callee-saved" Register ``d8`` bis ``d15``.
Da wir beim Kompilieren das ``-O3`` Flag genutzt haben, hatte der Host-C++-Compiler die lokalen Variablen für die Laufzeitmessungen direkt in diese Register verlagert. Der anschliessende Reset durch den neu generierten JIT-SME-Kernel führte beim Errechnen der finalen Metriken schliesslich zu einer Division durch Null.

Zur Lösung wurden die relevanten Benchmark-Funktionen mit ``__attribute__((optimize("O0")))`` dekoriert. Somit sichert der Compiler die zeitkritischen Variablen im Arbeitsspeicher/Stack ab, was das Zerstören durch den Hardware-Reset verhindert.

Ausserdem wurde die Messmethodik verfeinert: Die Zeitmessung umfasst jetzt nicht nur eine innere Schleife über viele Ausführungen, sondern zusätzlich einen "Average of 10" (äußere Schleife). Dadurch werden zufällige Rauschfaktoren wie System-Interrupts (OS-Jitter), Thermal Throttling und Cache-Evictions effektiv herausgeglättet.

5. Lessons Learned
------------------

- **Hardware-State & Calling Conventions:** Ein direkter Sprung auf die Hardwareebene (SME via JIT) kann böse Überraschungen bei den *Calling Conventions* bereithalten. Man darf sich nicht darauf verlassen, dass Register ihre Werte behalten, wenn man systemnahe Status-Instruktionen (wie ``smstart sm``) ausführt.
- **JIT vs AOT:** Sobald man den Compiler-Pfad zum Teil verlässt und Maschinencode direkt injiziert, greifen die Sicherheitsnetze und Register-Allokationen des C++-Compilers nicht mehr für die injizierten Blöcke. Manuelles Eingreifen (wie ``__attribute__((optimize("O0")))``) ist dann unerlässlich.
- **Verzerrungen im Benchmarking:** Eine noch so große Anzahl an Loop-Iterationen ist nutzlos, wenn sie am Stück ausgeführt wird und gerade ein L3-Cache-Miss oder ein Kontextwechsel des Betriebssystems passiert. Das Wiederholen in mehreren unabhängigen Blöcken (Avg of 10) ist zwingend nötig für verlässliche GFLOPS/GiB-Werte.

6. Aufgabenverteilung
----------------

Die Umsetzung der fünften Woche wurde im Team folgendermassen bearbeitet:

**Teammitglied 1: Justin Bergmann**
- Implementierung der JIT-Klasse (Speicherallokation via mmap und Executable-Rechte via mprotect)
- Umsetzung der präzisen Benchmark-Logik (10-Run Average)
- Analyse und Bug-Fix des Callee-Saved Register-Resets

**Teammitglied 2: Julian Müller**
- Übersetzung der ARM Assembly-Opcodes in Hexadezimal-Maschinencode
- Dynamische Berechnung der relativen Branch-Offsets für die Schleifen im Unary-Kernel
- Transfer des GEMM-Kernels direkt in das C++ Emit-Muster