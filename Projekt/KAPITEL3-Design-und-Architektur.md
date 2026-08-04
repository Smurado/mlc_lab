# Kapitel 3: Design und Architektur

Kurzbeschreibung für den Projektbericht. Ziel ist, dass jemand nach dem Lesen weiß, aus
welchen Bausteinen der Autotuner besteht und was ein einzelner Trial durchläuft.

**Wo es hingehört:** `docs/projekt/index.rst`, Abschnitt 3. Dort steht aktuell ein
Platzhalter plus eine Tabelle mit Eckdaten, die stehen bleiben kann. Alles andere in dem
Abschnitt ist zu ersetzen.

**Umfang:** ungefähr zwei Seiten plus ein Diagramm. Die Kursvorgabe lautet „keep the
project report to the point" — lieber knapp und mit Diagramm als ausführlich in Prosa.

---

## Reihenfolge

### 3.1 Überblick der Pipeline

Ein Diagramm, das den Weg von der Eingabe bis zum fertigen Kernel zeigt. Das ist das
Wichtigste am Kapitel; der Text drumherum kann kurz bleiben.

```
CSV-Eingabe
   ↓  parser.cpp
interne Darstellung (teir.hpp)
   ↓  autotuner.cpp
Suchraum aufspannen
   ↓  cost_model.cpp
Vorfilter auf die besten 30 %
   ↓  autotuner.cpp
Suchschleife (SA / GA / Random)
   │
   ├──→ passes.cpp      Konfiguration auf die IR anwenden
   ├──→ codegen.cpp     C++-Kernel erzeugen
   ├──→ JIT             übersetzen und laden
   ├──→ Validierung     gegen naive Referenz prüfen
   └──→ benchmark.cpp   real messen
   ↓
beste Konfiguration + validierter Kernel
```

Der Satz, der dabei hängen bleiben soll: **Jeder Kandidat wird tatsächlich übersetzt,
auf Korrektheit geprüft und gemessen. Es wird nichts geschätzt.**

### 3.2 Die interne Darstellung

Kurz erklären, was der Autotuner überhaupt verändert. Kernpunkt: Die Beschreibung der
Rechnung (Tensoren, Achsen mit Extents und Schrittweiten, Primitive) bleibt unangetastet.
Verändert wird ausschließlich der Schedule, also Schleifenreihenfolge, Kachelung und
Parallelisierung. Deshalb kann eine Transformation die Laufzeit ändern, aber nie das
Ergebnis.

Datei: `src/teir.hpp`, geparst in `src/parser.cpp`.

### 3.3 Der Suchraum

Die Konfiguration hat fünf Stellschrauben. Aus `src/autotuner.hpp`:

| Feld | Bedeutung |
|---|---|
| `split_axis` | welche Reduktionsachse gekachelt wird, leer = keine |
| `split_factor` | Kachelgröße, 1 = kein Split |
| `loop_order` | Reihenfolge der Schleifen |
| `parallel_axis` | welche Achse parallel läuft |
| `unroll_factor` | Entrollungsgrad der inneren Schleife |

Hier die Größe des Raums nennen: rund 28 800 Kandidaten bei sechs Achsen, faktoriell
wachsend mit der Achsenzahl. Daraus folgt, warum vollständiges Durchprobieren ausscheidet
und stattdessen gesucht wird.

### 3.4 Cost-Modell als Vorfilter

Nur die Rolle in der Architektur beschreiben, **nicht** die Bewertung. Die Zahlen und die
Kritik stehen in Abschnitt 5.3, das gehört nicht dupliziert.

Für das Kapitel reicht: Eine analytische Heuristik schätzt die Kosten aus
Speicherzugriffsmuster, Parallelisierungsaufwand, Arbeitssatzgröße und Entrollung. Sie
kalibriert sich beim Start selbst, indem sie die erreichbare Spitzenrate und den Aufwand
für den Thread-Start auf der laufenden Maschine misst. Zwei Aufgaben: Vorfilter auf die
besten 30 Prozent und Startpunkt für SA und GA.

Datei: `src/cost_model.cpp`.

### 3.5 Suchstrategien

Je zwei bis drei Sätze, was die Verfahren tun. Ergebnisse gehören auch hier nach 5.3.

- **Simulated Annealing** (Standard): lokale Nachbarschaft, Warmstart am Optimum des
  Cost-Modells
- **Genetischer Algorithmus**: Population mit Kreuzung und Mutation
- **Zufallssuche**: bewusst uninformiert, dient als faire Vergleichsbasis

Datei: `src/autotuner.cpp`.

### 3.6 Codegenerator und JIT

Was pro Trial passiert: Die Konfiguration wird auf die IR angewendet (`src/passes.cpp`),
daraus erzeugt `src/codegen.cpp` C++-Quelltext, der zur Laufzeit übersetzt und geladen
wird. Drei Backends: Scalar, NEON und SME.

Hier gehört eine Einschränkung hin, die später in den Limitierungen wieder auftaucht:
Das Backend ist eine Vorgabe und **kein Teil des Suchraums**. Es steht in
`AutotunerOptions`, nicht in `TuningConfig`.

### 3.7 Messung und Validierung

Kurz halten, das Detail steht in Kapitel 4. Für die Architektur reicht: Jeder erzeugte
Kernel wird erst gegen eine naive Referenz geprüft und dann gemessen. Ist er nicht
korrekt, geht seine Messung nicht in die Suche ein.

Dateien: `src/kernel_validation.hpp`, `src/bench_loop.hpp`, `src/benchmark.cpp`.

---

## Was nicht ins Kapitel gehört

- **Messwerte und Bewertungen.** Die stehen in Kapitel 5. Kapitel 3 beschreibt den
  Aufbau, nicht wie gut er funktioniert.
- **Die gefundenen Fehler.** Messschleife, SME-Backend und Validierung sind Kapitel 4.
- **Wiederholungen aus den Wochenberichten.** Auf sie verweisen statt sie nachzuerzählen.

## Formvorgaben aus dem Betreuer-Feedback

Diese gelten für den ganzen Bericht und sind in Woche 3+4 ausdrücklich angemerkt worden:

- Keine wertenden Verstärker wie „total", „extrem", „enorm". Wo eine Wertung stehen
  soll, gehört eine Zahl hin.
- Fehlerbehebungen nach dem Muster Symptom, Ursache mit Beleg, Änderung, Nachweis.
- Tabellen und Diagramme statt langer Beschreibungen.

## Stand des Codes

`Projekt/autotuner/src/`, aufgeräumt. Die Module sind:

| Datei | Zeilen | Aufgabe |
|---|---|---|
| `autotuner.cpp` | 836 | Suchraum, Strategien, Suchschleife |
| `codegen.cpp` | 760 | Kernel-Erzeugung, drei Backends |
| `benchmark.cpp` | 390 | Messung eines Kandidaten |
| `main.cpp` | 371 | Ablaufsteuerung, Umgebungsvariablen |
| `einsum.cpp` | 261 | Einstein-Notation, Referenzrechnung |
| `cost_model.cpp` | 247 | analytische Heuristik |
| `parser.cpp`, `passes.cpp` | | CSV-Eingabe, IR-Transformationen |

Dazu sieben Testdateien (`test_*.cpp`), die über `make test` laufen.
