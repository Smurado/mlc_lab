# MLC Lab

## Hinweise zur Abgabe
- Das Datum in Dateinamen muss immer im Format **YYYY.MM.DD** angegeben werden.
- Die **Nachnamen der Teammitglieder** müssen ebenfalls im Dateinamen enthalten sein.

## Week 1: Debugging Report (inner_product | outer_product)

Die Aufgabe verlangte das Debuggen eines Aufrufs der `inner_product` Funktion mit dem GNU Debugger (GDB). Da wir auf Apple Silicon (AArch64) arbeiten, wurde das Äquivalent **LLDB** genutzt. 

Um das Programm durchschreiten (step through) zu können, wurde es zunächst mit Debug-Symbolen (`-g`) kompiliert:
`clang++ -g -arch arm64 week1/main.cpp week1/functions.s -o week1/test_run_debug`

### Vorgehensweise und Beobachtungen:
1. **Debugger starten & Haltepunkt setzen**
   - Mit `lldb ./week1/test_run_debug` wurde das Programm geladen.
   - Mit `b inner_product` wurde ein Breakpoint genau am Beginn der Assembler-Funktion gesetzt, damit das Programm aus C++ heraus direkt beim Einsprung pausiert.
2. **Programm ausführen (`run`)**
   - Das Test-Framework (Catch2) lief bis zu dem Punkt, an dem `inner_product` aufgerufen wurde, und fror dort ein. Wir befanden uns direkt in der CPU beim ersten `mov`-Befehl (`mov x8, #0x0`).
3. **Schrittweises Durchlaufen (`ni`)**
   - Mit dem Befehl `ni` (*next instruction*) sind wir den Maschinencode Anweisung für Anweisung durchgegangen.
   - Wir konnten live beobachten, wie die Array-Werte geladen (`ldr`), multipliziert (`umull`) und summiert (`add`) wurden.
4. **Beobachtungen in den Registern (`register read`)**
   - **Parameter-Übergabe:** Nach einigen Schritten überprüften wir das Register `w2`, in dem laut AAPCS64 Calling Convention das dritte Argument (die Array-Größe) stehen sollte. Mit `register read w2` bekamen wir die Ausgabe `0x00000004` (4), was exakt der Variable `size = 4` aus unserem C++ Code entspricht.
   - **Akkumulation:** Durch das Auslesen des Registers `x8` (unsere finale Summe) konnten wir nach jeder inneren Schleifeniteration zusehen, wie sich das korrekte innere Produkt Stück für Stück aufbaute, bis es schließlich den erwarteten Wert `70` erreichte.

## Info zu Catch2
Für das Testing-Framework Catch2 (`catch.hpp`) haben wir einen eigenen Ordner `lib` im Hauptverzeichnis des Projekts angelegt. Darin liegt nun die Header-Datei, sodass beide Wochen-Ordner strukturiert darauf zugreifen können.