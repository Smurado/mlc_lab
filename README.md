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

## Week 2: Execution Throughput & Permutation (abc → cba)

Die Aufgabe bestand aus zwei Teilen: (1) einen Microbenchmark für die Execution Throughput ausgewählter FP-Instruktionen zu schreiben und (2) einen NEON-Kernel zu implementieren, der eine Tensor-Permutation `abc → cba` durchführt, sowie dessen Bandbreite in GiB/s in Abhängigkeit von `|c|` zu vermessen.

Übersetzt wurde wie in Woche 1 direkt mit Clang auf Apple Silicon (AArch64):
`clang++ -O2 -arch arm64 week2/main.cpp week2/functions.s -o week2/test_run`

### 1. Execution Throughput

Für jede Instruktion wurde in `week2/functions.s` eine eigene Benchmark-Funktion geschrieben, die pro Iteration **50 × 28 = 1400** unabhängige Instruktionen ausführt (50-fach unrolled, 28 unabhängige Zielregister, damit keine Abhängigkeitsketten die Pipeline bremsen). Gemessen wird über Catch2-TEST_CASEs mit `std::chrono`.

#### Vorgehensweise und Beobachtungen:
1. **Dependency-Kette vermeiden**
   - Alle FMADD/FMLA-Instruktionen schreiben in ein anderes Zielregister (`s0..s27` bzw. `v0..v27.4s`). So hängt keine Instruktion vom vorherigen Ergebnis ab und der Prozessor kann mehrere FP-Pipelines parallel füllen.
2. **Unrolling**
   - Durch `.rep 50` wird der Schleifen-Overhead (`cbz`, `sub`, `b`) praktisch unsichtbar: pro Schleifendurchlauf werden 1400 FMADDs ausgeführt, aber nur drei Kontrollinstruktionen.
3. **Warmup**
   - Vor der Messung wird `micro_benchmark*(10)` aufgerufen, damit Branch-Predictor und Frequenz-Scaling eingeschwungen sind.
4. **FLOPs-Umrechnung**
   - `FMADD s = a*b + c` und `FMLA v.Ns = v + a*b` werden jeweils als **2 FLOPs pro Lane** gezählt (eine Multiplikation und eine Addition).

#### Messwerte (Apple Silicon, `-O2`):

| Instruktion     | Iterationen  | Zeit     | Throughput         |
|-----------------|-------------:|---------:|-------------------:|
| FMADD (scalar)  | 100 000 000  | 8 140 ms | **34,4 GFLOP/s**   |
| FMLA 4S (vec)   |  50 000 000  | 5 135 ms | **109,1 GFLOP/s**  |
| FMLA 2S (vec)   |  50 000 000  | 4 749 ms | **59,0 GFLOP/s**   |

Die 4S-Variante liefert erwartungsgemäß etwa die doppelte GFLOP/s-Rate von 2S, weil beide instruktionsraten-limitiert sind, 4S aber die vierfache Anzahl Lanes pro Instruktion rechnet. Die skalare Variante liegt klar darunter, da pro Instruktion nur eine Lane aktiv ist.

### 2. Permutation `abc → cba`

Implementiert ist der NEON-Kernel `perm_neon_abc_cba` in `week2/functions.s` mit der Signatur aus der Aufgabenstellung (Dimensionen `|a| = 8`, `|b| = 4`, `|c|` als Parameter; beide Tensoren row-major).

#### Vorgehensweise und Beobachtungen:
1. **Zugriffs­muster**
   - In `abc[a][b][c]` liegen aufeinanderfolgende `c`-Werte **zusammenhängend** im Speicher, in `cba[c][b][a]` die `a`-Werte. Ein naiver Triple-Loop würde also aus abc sequenziell lesen, aber in cba mit Stride `B*A*4 = 128` Byte schreiben – das schlägt die Cache-Line-Granularität kaputt.
2. **4×4-Transpose via NEON**
   - Pro festem `b` laufen wir über `c` in 4er-Blöcken. Für jede der 8 `a`-Zeilen wird ein 4-Float-Vektor (vier aufeinanderfolgende `c`-Werte) mit `ld1 {v.4s}` geladen. Zwei 4×4-Tiles werden mit `trn1/trn2/zip1/zip2` transponiert. Danach enthält jedes Ziel-Register 4 `a`-Werte für ein festes `c`, und die 8 `a`-Werte eines `c`-Slices können mit einem einzigen `stp q,q` geschrieben werden.
3. **Tail-Handling**
   - Ist `|c|` kein Vielfaches von 4, kopiert ein skalarer Rest-Loop die übrigen `c`-Werte einzeln. Der Korrektheitstest in `main.cpp` deckt `|c| ∈ {1, 2, 4, 8, 15, 32}` ab und prüft jedes einzelne Element.
4. **Bandbreiten-Messung**
   - Der Benchmark wählt die Wiederholungszahl so, dass pro Messpunkt ca. `2^28` Elemente bewegt werden. Als Bandbreite wird `bytes = 2 · |a| · |b| · |c| · 4` pro Call gezählt (Read + Write). Vor der Messung gibt es 3 Warmup-Calls.

#### Messwerte (Apple Silicon, `-O2`):

| `\|c\|`       | Wiederholungen | Zeit     | Bandbreite       |
|-------------:|---------------:|---------:|-----------------:|
|            4 |      2 097 152 |  16,7 ms |    120,0 GiB/s   |
|            8 |      1 048 576 |  13,9 ms |    143,7 GiB/s   |
|           16 |        524 288 |  10,7 ms |    186,6 GiB/s   |
|           32 |        262 144 |  10,5 ms |    190,0 GiB/s   |
|          128 |         65 536 |  10,3 ms |    194,3 GiB/s   |
|          256 |         32 768 |  10,2 ms |  **195,4 GiB/s** |
|          512 |         16 384 |  10,4 ms |    192,2 GiB/s   |
|        1 024 |          8 192 |  23,1 ms |     86,5 GiB/s   |
|        4 096 |          2 048 |  37,4 ms |     53,5 GiB/s   |
|       16 384 |            512 |  42,2 ms |     47,4 GiB/s   |
|       65 536 |            128 |  36,7 ms |     54,4 GiB/s   |
|      262 144 |             32 |  73,5 ms |     27,2 GiB/s   |

Der Peak von ca. **195 GiB/s** liegt im Bereich, in dem beide Tensoren komplett in den L1-Cache passen (`|c| ≈ 128..512` → Gesamtgröße 32–128 KiB pro Tensor). Sobald die Arbeitsmenge den L1 und später den L2 sprengt (`|c| ≥ 1024`), fällt die Bandbreite auf DRAM-nahe ~30–55 GiB/s – der Kernel ist dort klar bandbreitenlimitiert. Bei sehr kleinem `|c|` (4, 8) dominieren dagegen die pro-Call-Kosten (Prolog, Loop-Setup), weshalb der L1-Peak noch nicht erreicht wird.