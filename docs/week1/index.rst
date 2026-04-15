Week 1:
====================

1. Introduction
----------------
Die Aufgabe in Woche 1 bestand darin, `inner_product` und `outer_product` in assembly zu implementieren.
Außerdem sollten dazu Unit-Tests in C++ erstellt werden, um die Funktionalität der Implementierungen zu überprüfen.
Zum Schluss sollte der GNU Project Debugger (GDB) verwendet werden, um durch die Implementierung der `inner_product` Funktion zu debuggen.

2. Implementation
------------------


3. Unit Tests mit Catch2
-------------------------
Die Unit-Tests wurden mit dem Catch2 Framework erstellt und validieren die Korrektheit der Implementierungen.

4. Debugging mit GDB
----------------------
Der GDB wurde verwendet, um die `inner_product` Funktion Schritt für Schritt zu durchlaufen.
Wir sind hierbei wie folgt vorgegangen (mittels GDB unter Ubuntu/Linux in einem Docker Container):
- Kompilieren des C++ und AArch64-Assembly Codes mit Debug-Informationen (`-g` und `-O0` Flags).
- Starten von GDB mit dem kompilierten Programm.
- Setzen eines Breakpoints auf das Label `inner_product`.
- Schrittgweises Durchlaufen (`stepi`) und Überprüfen der Registerwerte (`info registers`).

Hier ist ein exemplarischer Auszug unserer Debugging-Session:

.. code-block:: text

    (gdb) break _inner_product
    Breakpoint 1 at 0xe8974: file week1/functions.s, line 5.
    (gdb) run
    Starting program: /usr/src/app/week1/test_run_debug 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/aarch64-linux-gnu/libthread_db.so.1".

    Breakpoint 1, inner_product () at week1/functions.s:5
    5           mov x8, #0
    (gdb) info registers x8
    x8             0x101010101010101   72340172838076673
    (gdb)  stepi
    6           mov w4, #0                  // Zaehler initialisieren
    (gdb) info registers w4
    w4             0x1f                31
    (gdb) stepi
    loop_start () at week1/functions.s:9
    9           cmp w4, w2
    (gdb) info registers w4 w2
    w4             0x0                 0
    w2             0x4                 4
    (gdb) stepi
    10          b.ge loop_end               // Springe zum Ende, falls Zaehler >= Groesse
    (gdb) info registers x0
    x0             0xffffffffea08      281474976705032

Auswertung der Debugger-Session
----------
Der Debugger-Output liefert uns drei kurze, aber wichtige Erkenntnisse:

1. **Notwendige Initialisierung:** Vor den `mov`-Befehlen stehen noch alte Speicherwerte in den Registern. Das beweist, dass Register manuell auf `0` gesetzt werden müssen.
2. **Korrekte Parameterübergabe:** C++ übergibt die Parameter exakt nach ARM-Standard: `x0` enthält die Speicheradresse des Arrays, und in `w2` liegt unsere Array-Größe `4`.
3. **Schleifenlogik:** Der Zählervergleich (`cmp w4, w2`) prüft korrekt `0` gegen `4`, womit die Schleife fehlerfrei startet.