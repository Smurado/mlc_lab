Team Contributions
==================

Week 1
------

**Teammitglied 1: Justin Bergmann**

* Implementierung der ``outer_product`` Funktion in Assembly (``functions.s``).
* Entwicklung der entsprechenden Unit-Tests für das Outer Product in der ``main.cpp``.

**Teammitglied 2: Julian Müller**

* Implementierung der ``inner_product`` Funktion in Assembly (``functions.s``).
* Entwicklung der entsprechenden Unit-Tests für das Inner Product in der ``main.cpp``.


Week 2
------

**Teammitglied 1: Justin Bergmann**

* Implementierung der Benchmarks für den Execution Throughput der Instruktionen:

  - ``FMADD`` (scalar), FP32-Variante
  - ``FMLA`` (vector) mit Arrangement Specifier ``4S``
  - ``FMLA`` (vector) mit Arrangement Specifier ``2S``

**Teammitglied 2: Julian Müller**

* Implementierung der Basisfunktion für die Benchmarks
* Implementierung der Matrixtransformation

  - ``abc -> cba``

Week 3-4
--------

**Teammitglied 1: Justin Bergmann**

* Implementierung der ARM SME GEMM Microkernel
* Optimierung der SME Kernel-Ausführung durch "Fast"-Routinen
* Integration und Auswertung der Catch2 C++ Performance-Benchmarks (GFLOPS)

**Teammitglied 2: Julian Müller**

* Implementierung der SSVE Unary-Kernel
* Implementierung der Unit-Tests für Kernel-Funktionen

Week 5
------

**Teammitglied 1: Justin Bergmann**

* Implementierung der JIT-Klasse (Speicherallokation via mmap und Executable-Rechte via mprotect)
* Umsetzung der Benchmark-Logik (10-Run Average)
* Analyse und Bug-Fix des Callee-Saved Register-Resets

**Teammitglied 2: Julian Müller**

* Übersetzung der ARM Assembly-Opcodes in Hexadezimal-Maschinencode
* Dynamische Berechnung der relativen Branch-Offsets für die Schleifen im Unary-Kernel
* Umbau des GEMM-Kernels auf das C++ Emit-Muster

Week 6
------

**Teammitglied 1: Justin Bergmann**

* Implementierung der dynamischen ``generate``Funktion für die SME GEMM-Primitive (Unterstützung für variable Parameter M, N, K).
* Entwicklung und Auswertung der Test- und Benchmarking-Logik über 27 Settings (Ausgabe der Performance in GFLOPS).

**Teammitglied 2: Julian Müller**

* Implementierung der flexiblen SSVE Code-Generation (``generate``) für die Unary-Primitive (``zero``, ``identity``, ``relu``).
* Automatisierung der Unary-Tests und Bandbreiten-Benchmarks über 9 verschiedene Settings (Ausgabe in GiB/s).