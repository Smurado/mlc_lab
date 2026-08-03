# Trial-Budget der GETT-Matrix — Herleitung
Automatisch erzeugt von `eval/trial_budget_probe.py`. Rohdaten: `curves.json`.
## Frage
Wie viele Trials braucht die Suche, bis weitere Trials nichts mehr bringen? Der vollstaendige Suchraum ist nicht ausschoepfbar (1 050 Kandidaten bei 4 Achsen, 115 200 bei 7), das Budget muss also gemessen statt geraten werden.
## Aufbau
- 100 Trials, Seeds [1, 2, 3], Strategien ['sa', 'ga', 'random']
- Je ein Fall pro Achsenzahl (4/5/6/7). Die Konvergenz haengt am Suchraum, also an der Achsenzahl — deshalb pro Klasse der guenstigste Fall, was 200 Trials ueberhaupt bezahlbar macht.
- Random ist bewusst dabei: es konvergiert am langsamsten und bestimmt damit das Budget, das einen FAIREN Vergleich aller drei Strategien erlaubt.
- Uebrige Einstellungen identisch zum geplanten grossen Lauf (Cost-Filter 0.3, Warmstart SA=1/Random=0, Suchkompilate -O2, adaptiver Messblock).

> **33 von 36 Laeufen wurden vor 100 Trials abgebrochen** (Zeitlimit 600 s). Ihre Kurven sind nur bis zum erreichten Trial ausgewertet, nicht fortgeschrieben. Konvergenzaussagen daraus sind untere Schranken.

## Messung
| Fall | Achsen | Strategie | Trials | Endwert GFLOPS | 90 % ab | 95 % ab | 99 % ab | letzte Verbesserung | s/Trial |
|---|---|---|---|---|---|---|---|---|---|
| contraction_abc_dca_bd | 4 | sa | 19* | 17.97 | 6 | 6 | 6 | 6 | 31.58 |
| contraction_abc_dca_bd | 4 | ga | 8* | 18.56 | 7 | 7 | 7 | 7 | 75.00 |
| contraction_abc_dca_bd | 4 | random | 8* | 18.55 | 7 | 7 | 7 | 7 | 75.00 |
| contraction_abcd_deca_be | 5 | sa | 6* | 5.81 | 1 | 3 | 3 | 3 | 100.00 |
| contraction_abcd_deca_be | 5 | ga | 7* | 7.27 | 5 | 5 | 5 | 5 | 85.71 |
| contraction_abcd_deca_be | 5 | random | 2* | 56.94 | 2 | 2 | 2 | 2 | 300.00 |
| contraction_abcde_efbad_cf | 6 | sa | 18* | 0.00 | None | None | None | 1 | 33.33 |
| contraction_abcde_efbad_cf | 6 | ga | 84* | 0.00 | None | None | None | 1 | 0.40 |
| contraction_abcde_efbad_cf | 6 | random | 100 | 0.00 | None | None | None | 1 | 0.39 |
| contraction_abcdef_dega_gfbc | 7 | sa | 15* | 165.86 | 6 | 6 | 6 | 6 | 40.00 |
| contraction_abcdef_dega_gfbc | 7 | ga | 9* | 68.74 | 1 | 1 | 1 | 1 | 66.67 |
| contraction_abcdef_dega_gfbc | 7 | random | 5* | 105.73 | 2 | 2 | 2 | 2 | 120.00 |

Mediane über die Seeds; `*` = mindestens ein Lauf abgebrochen. „95 % ab“ = erster Trial, ab dem der beste bisher gefundene Wert mindestens 95 % des am Ende dieses Laufs erreichten Werts hat.

## Ableitung
- 95 % des Endwerts: spaetestens ab Trial **— (alle betreffenden Laeufe abgebrochen)** (alle Faelle/Strategien/Seeds).
- Nur Random (der langsamste Konvergierer): spaetestens Trial **— (alle betreffenden Laeufe abgebrochen)**.
- 99 % des Endwerts: spaetestens ab Trial **— (alle betreffenden Laeufe abgebrochen)**.
- Letzte beobachtete Verbesserung ueberhaupt: Trial **1**.

## Kosten der vollen Matrix
Die Trial-Kosten haengen stark von der Strategie ab: SA startet warm und bleibt in brauchbaren Konfigurationen, Random zieht gleichverteilt und trifft dabei Schleifenordnungen mit katastrophalem Cache-Verhalten. Gemessene Mediane aus dieser Sondierung (guenstige Faelle!):

| Strategie | s/Trial (Median) |
|---|---|
| sa | 35.42 |
| ga | 56.41 |
| random | 75.00 |

Die 48 Faelle der Matrix reichen von ~1,3 Mrd. bis ~524 Mrd. Iterationen; zwoelf teure Faelle machen ~94 % der Rechenzeit aus. Analytische Hochrechnung (8 GFLOPS je Kernel, 1 s Compile) ergibt ~25,1 min pro Trial ueber alle 48 Faelle:

| N | 48 Faelle x 3 Strategien (analytisch) |
|---|---|
| 10 | 12.6 h |
| 20 | 25.1 h |
| 30 | 37.6 h |
| 50 | 62.8 h |
| 100 | 125.5 h |

Diese Hochrechnung unterstellt gleiche Kosten je Strategie. Gemessen kostet Random hier das 2-fache von SA — der reale Random-Anteil liegt also deutlich darueber.

## Entscheidung
_(von Hand zu ergaenzen, sobald Budget und Laufzeit abgewogen sind — die Messung oben liefert die Begruendung.)_
