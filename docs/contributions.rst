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

Week 3
------

**Teammitglied 1: Justin Bergmann**

* Implementierung der ARM SME GEMM Microkernel
* Optimierung der SME Kernel-Ausführung durch "Fast"-Routinen
* Integration und Auswertung der Catch2 C++ Performance-Benchmarks (GFLOPS)

**Teammitglied 2: Julian Müller**
* Implementierung der SSVE Unary-Kernel
* Implementierung der Unit-Tests für Kernel-Funktionen


