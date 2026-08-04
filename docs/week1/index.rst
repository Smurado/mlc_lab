Week 1:
====================

1. Introduction
----------------
Die Aufgabe in Woche 1 bestand darin, ``inner_product`` und ``outer_product`` in assembly zu implementieren.
Außerdem sollten dazu Unit-Tests in C++ erstellt werden, um die Funktionalität der Implementierungen zu überprüfen.
Zum Schluss sollte der GNU Project Debugger (GDB) verwendet werden, um durch die Implementierung der ``inner_product`` Funktion zu debuggen.

2. Implementation
------------------
Die Funktionen ``inner_product`` und ``outer_product`` wurden in AArch64-Assembly umgesetzt. 
Die Übergabe der Parameter erfolgt über Register (z.B. ``x0``, ``x1``, ``w2``).

**Inner Product**

Berechnet das Skalarprodukt zweier Arrays.

- Schleife über alle Elemente
- Multiplikation der Werte
- Aufsummieren der Ergebnisse
- Rückgabe der Summe

Pseudocode::

    sum = 0
    for i in range(size):
        sum += a[i] * b[i]
    return sum


**Outer Product**

Berechnet das äußere Produkt zweier Arrays.

- Zwei Schleifen (für beide Arrays)
- Jedes Element aus ``a`` wird mit jedem aus ``b`` multipliziert
- Ergebnis wird im Array ``c`` gespeichert

Pseudocode::

    for i in range(size):
        for j in range(size):
            c[i * size + j] = a[i] * b[j]

3. Unit Tests mit Catch2
-------------------------
Die Unit-Tests wurden mit dem Catch2 Framework erstellt und validieren die Korrektheit der Implementierungen.

**Vergleich gegen eine Referenzimplementierung**

Ein Test gegen von Hand ausgerechnete Werte hat eine Schwäche: Trifft eine fehlerhafte
Implementierung den erwarteten Wert zufällig, bleibt der Fehler unbemerkt. Wir prüfen deshalb
elementweise gegen ``data/base_math.cpp``, eine bewusst naive Umsetzung derselben Rechnung in C++.
Sie rechnet mit 32x32 auf 64 Bit ohne Vorzeichen, passend zu ``umull`` in der Assembly. Die Datei
ist von uns geschrieben und damit selbst Teil des zu prüfenden Codes, keine vorgegebene Referenz.

Die Testeingaben erzeugen wir zufällig, aber mit festem Startwert (``std::mt19937(20250803u)``).
Damit sind die Läufe reproduzierbar, ohne dass die Tests auf bestimmte Zahlen zugeschnitten sind.
Dazu kommen Randfälle mit Nullen, mit Maximalwerten und mit der Länge 1.

Vor jedem Aufruf füllen wir den Ausgabepuffer mit ``0xDEADBEEF``. Elemente, die die Routine nicht
beschreibt, fallen dadurch auf, statt zufällig den richtigen Wert zu enthalten.

Insgesamt umfasst die Testsuite 231 Assertions in 8 Testfällen.

**Gegenprobe durch Mutation**

Ein Test, der nie fehlschlägt, sagt nichts aus. Wir bauen deshalb gezielt Fehler in ``functions.s``
ein und prüfen, ob die Tests sie melden.

.. list-table::
   :header-rows: 1

   * - Änderung im Assembly-Code
     - vom Test erkannt
   * - Schleifengrenze um 1 verschoben
     - ja
   * - Ergebnisregister vertauscht
     - ja
   * - ``umull`` durch ``mul`` ersetzt
     - nein

Die dritte Zeile ist keine Lücke in den Tests. ``ldr w5`` erweitert den geladenen Wert bereits mit
Nullen auf 64 Bit, deshalb liefern ``mul`` und ``umull`` an dieser Stelle dasselbe Ergebnis. Die
Änderung ist semantisch neutral und kann von keinem Test erkannt werden.

Die Tests laufen über ``make test`` im Ordner ``week1``.

4. Debugging mit GDB
----------------------
Der GDB wurde verwendet, um die ``inner_product`` Funktion Schritt für Schritt zu durchlaufen.
Wir sind hierbei wie folgt vorgegangen (mittels GDB unter Ubuntu/Linux in einem Docker Container):

- Kompilieren des C++ und AArch64-Assembly Codes mit Debug-Informationen (``-g`` und ``-O0``).
- Starten von GDB mit dem kompilierten Programm.
- Setzen eines Breakpoints auf das Label ``inner_product``.
- Schrittweises Durchlaufen (``stepi``) und Überprüfen der Registerwerte (``info registers``).

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

5. Auswertung der Debugger-Session
----------------------------------
Der Debugger-Output liefert uns drei kurze, aber wichtige Erkenntnisse:

1. **Notwendige Initialisierung:** Vor den ``mov``-Befehlen stehen noch alte Speicherwerte in den Registern. Das beweist, dass Register manuell auf 0 gesetzt werden müssen.
2. **Korrekte Parameterübergabe:** C++ übergibt die Parameter exakt nach ARM-Standard: ``x0`` enthält die Speicheradresse des Arrays, und in ``w2`` liegt unsere Array-Größe 4.
3. **Schleifenlogik:** Der Zählervergleich (``cmp w4, w2``) prüft korrekt 0 gegen 4, womit die Schleife fehlerfrei startet.
