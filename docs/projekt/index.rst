Projekt: TEIR-Autotuner
=======================

.. note::

   **Entwurfsfassung.** Abschnitt 3 fehlt noch und wird gesondert ergaenzt. Die
   Hinweisbloecke sind vor der Abgabe zu entfernen.

1. Einleitung
-------------

Tensorkontraktionen sind der Rechenkern moderner neuronaler Netze. Eine Kontraktion
legt fest, welche Achsen zweier Eingabetensoren verknüpft und welche aufsummiert
werden, sagt aber nichts darüber, in welcher Reihenfolge die Schleifen abgearbeitet
werden, wo gekachelt wird und welche Achse parallel läuft. Genau diese Entscheidungen
bestimmen die Laufzeit, und sie hängen sowohl von der Form der Tensoren als auch von
der Zielhardware ab. Für eine 512er-Matrixmultiplikation ist eine andere
Schleifenordnung günstig als für eine Kontraktion mit sieben Achsen und stark
unterschiedlichen Extents.

Diese Entscheidungen von Hand zu treffen skaliert nicht. Für jede neue Tensorform
müsste man erneut messen, welche Anordnung passt. Ein Autotuner nimmt diese Arbeit ab:
Er erzeugt aus einer Beschreibung der Kontraktion mehrere Kandidaten, misst sie und
gibt die schnellste Konfiguration zurück.

Dieses Projekt setzt einen solchen Autotuner für TEIR um. Der Schwerpunkt liegt nicht
darauf, möglichst viele Suchverfahren zu implementieren, sondern darauf, den
gemessenen Zahlen trauen zu können. Ein Autotuner trifft seine Entscheidungen
ausschließlich auf Grundlage seiner eigenen Messungen. Ist die Messung fehlerhaft,
sucht er zuverlässig die falsche Konfiguration. Der Bericht legt deshalb besonderes
Gewicht auf die Messmethodik und benennt die Stellen, an denen sich das System als
fehlerhaft erwiesen hat.

Abgrenzung zu den Wochenaufgaben: Dort entstanden die einzelnen Bausteine, von der
Assembly über den Codegenerator bis zur Laufzeitumgebung. Das Projekt setzt darauf auf
und ergänzt die Suche über den Konfigurationsraum sowie die Bewertung der Ergebnisse
gegen etablierte Werkzeuge.

2. Grundlagen
-------------

**Einstein-Notation.** Eine Kontraktion wird als Zeichenkette über den Achsennamen
geschrieben, etwa ``ab-ac-cb``. Die drei Teile stehen für Ausgabe, erste Eingabe und
zweite Eingabe. Achsen, die in beiden Eingaben vorkommen, aber nicht in der Ausgabe,
werden aufsummiert. Im Beispiel ist das ``c``, womit die Notation eine gewöhnliche
Matrixmultiplikation beschreibt. Der Vorteil dieser Schreibweise liegt darin, dass sie
die Rechnung festlegt, ohne eine Schleifenreihenfolge vorzugeben. Genau diese Freiheit
nutzt der Autotuner.

**TEIR.** TEIR ist die Zwischendarstellung, in der das Programm vorliegt. Sie trennt
zwei Dinge, die in normalem C++ vermischt sind: was gerechnet wird und wie es
abgearbeitet wird. Die Tensoren, Achsen mit ihren Extents und Schrittweiten sowie die
Primitive beschreiben die Rechnung. Der Schedule beschreibt die Ausführung, also die
Verschachtelung der Schleifen und ihre Ausführungsstrategie. Eine Transformation des
Schedules verändert die Laufzeit, nicht aber das Ergebnis. Der Autotuner arbeitet
ausschließlich auf dem Schedule.

**ARM SME.** Die Scalable Matrix Extension erweitert AArch64 um ein zweidimensionales
Registerfeld, das ZA-Array. Bei einer Vektorlänge von 512 Bit fasst eine Kachel 16 mal
16 Werte in einfacher Genauigkeit; vier Kacheln lassen sich zu einem 32 mal
32-Akkumulator zusammenfassen. Die zentrale Instruktion ist ``fmopa``, die aus zwei
Vektoren ein äußeres Produkt bildet und es auf die Kachel addiert. Eine
Matrixmultiplikation entsteht daraus, indem C in das ZA-Array geladen, über die
K-Achse akkumuliert und das Ergebnis am Ende zurückgeschrieben wird.

SME ist nur im Streaming-Modus verfügbar, der mit ``smstart`` betreten und mit
``smstop`` verlassen wird. Zwei Eigenheiten sind dabei wichtig und haben im Projekt zu
Fehlern geführt. Erstens setzt ``smstart`` alle Vektorregister zurück, einschließlich
der nach AAPCS64 vom Aufgerufenen zu sichernden Register ``d8`` bis ``d15``. Zweitens
implementiert Apple Silicon SME, aber kein eigenständiges SVE: SVE-Instruktionen sind
ausschließlich innerhalb des Streaming-Modus gültig. Übersetzt man gewöhnlichen
C++-Code mit aktiviertem SME-Flag, erzeugt der Compiler SVE-Instruktionen, die
außerhalb des Streaming-Modus zu einer ungültigen Instruktion führen.

Die Umsetzung dieser Bausteine ist in den Wochenberichten beschrieben und wird hier
nicht wiederholt.

3. Design und Architektur
-------------------------

.. hint::

   Pipeline von der Eingabe bis zum ausgeführten Kernel: Parser, IR, Transformationen,
   Cost-Modell als Vorfilter, Suchstrategie, Codegenerator, Messschleife, Validierung.
   Ein Diagramm hilft hier mehr als Prosa.

**Belegte Eckdaten**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Eigenschaft
     - Wert
   * - Suchstrategien
     - Simulated Annealing, genetischer Algorithmus, Zufallssuche
   * - Warmstart
     - Startpunkt ist das Optimum des Cost-Modells
   * - Backends
     - Scalar, NEON, SME
   * - Zielplattform
     - Apple M4 Max, 12 P-Kerne, 4 E-Kerne, L1d 64 KiB, L2 4 MiB

4. Methodik
-----------

Ein Autotuner entscheidet auf Grundlage seiner eigenen Messungen. Dieser Abschnitt
begründet, warum den Zahlen im Ergebnisteil zu trauen ist, und benennt drei Fehler, die
das System zwischenzeitlich unbrauchbar gemacht haben. Die Darstellung folgt jeweils
derselben Reihenfolge: Symptom, belegte Ursache, Änderung, Nachweis.

4.1 Der Fehler in der Messschleife
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Die GETT-Matrix lieferte **0 von 144** Läufen mit Status ``ok``
   * - Ursache
     - Der Zeitdeckel wurde erst **nach 64 Kernel-Aufrufen** geprüft. Bei einem
       Kernel von 763 ms dauerte eine Endmessung dadurch 244 s statt 3,8 s
   * - Änderung
     - Adaptive Blockgröße, über ``TEIR_BENCH_ADAPTIVE=1`` zuschaltbar,
       Default bleibt das alte Verhalten
   * - Nachweis
     - Bis **64-fach** schnellere Messung; Abweichung der Messwerte über sieben
       Vergleichspunkte **-2,3 % bis +1,8 % mit wechselndem Vorzeichen**

Die Messschleife rief den Kernel in Blöcken zu 64 Wiederholungen auf und prüfte erst
nach jedem Block, ob das Zeitbudget erschöpft war. Diese feste Blockgröße stammt aus
der Überlegung, dass ein einzelner Aufruf zu kurz ist, um zuverlässig gemessen zu
werden: Bei einem Kernel im Mikrosekundenbereich liegt die Auflösung des Zeitgebers in
derselben Größenordnung wie die Messgröße. Für schnelle Kernel ist die Wiederholung
also richtig. Für langsame ist sie fatal, weil das Budget erst nach dem 64-fachen der
Kernel-Laufzeit geprüft wird.

Die Lösung passt die Blockgröße an die gemessene Laufzeit an. Ein erster Aufruf
schätzt die Dauer, daraus ergibt sich die Zahl der Wiederholungen für das gewünschte
Messfenster. Langsame Kernel werden dadurch genau einmal aufgerufen, schnelle so oft,
wie für ein stabiles Fenster nötig ist.

Entscheidend für die Vergleichbarkeit ist der Nachweis, dass die Änderung die
Messwerte nicht verschiebt. Über sieben Vergleichspunkte liegt die Abweichung zwischen
altem und neuem Verfahren zwischen -2,3 und +1,8 Prozent, und zwar mit wechselndem
Vorzeichen. Ein systematischer Fehler hätte ein festes Vorzeichen; die Streuung ist
also Rauschen und kein Versatz. Die verbleibende Unsicherheit von rund zwei Prozent ist
die Grenze, unterhalb derer Aussagen aus gemischten Messverfahren nicht mehr belastbar
sind. Konkret heißt das: Ein Vergleich zwischen Suchstrategien, deren Ergebnisse sich
um weniger als drei Prozent unterscheiden, darf nicht auf Zahlen aus beiden Verfahren
beruhen.

4.2 Streuung und Wiederholungen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ein einzelner Messwert sagt wenig darüber aus, wie stabil er ist. Um das zu
quantifizieren, wurden die Kernel aus den Wochenaufgaben je 15-mal gemessen und Median
sowie Spannweite bestimmt. Das Ergebnis zeigt eine klare Abhängigkeit von der
Problemgröße:

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Messpunkt
     - Median
     - Spannweite
   * - ``relu_16_16``
     - 108,8 GiB/s
     - 84,9 %
   * - ``identity_16_16``
     - 109,4 GiB/s
     - 40,7 %
   * - Unary 512x512
     - 409,1 GiB/s
     - 4,8 %
   * - GEMM, alle 27 Konfigurationen
     - 1109 bis 1877 GFLOPS
     - 2,7 bis 9,4 %

Bei einer 16 mal 16-Matrix ist der Kernel nach wenigen Mikrosekunden fertig. Gemessen
wird dann überwiegend der Aufwand für Aufruf und Zeitnahme, nicht die Bandbreite. Der
größte gemessene Wert für ``relu_16_16`` liegt um den Faktor 2,6 über dem kleinsten;
aus solchen Zahlen lässt sich kein Vergleich zwischen Operationen ableiten.

Daraus folgen zwei Entscheidungen für diesen Bericht. Erstens werden Messwerte als
Median mehrerer Läufe angegeben, nicht als Einzelwert. Zweitens steht die Spannweite
mit in der Tabelle, damit ein instabiler Wert nicht denselben Eindruck erweckt wie ein
stabiler.

Die Korrektur dieser Werte hat zwei zuvor getroffene Aussagen widerlegt. Der bislang
für ``gemm_32_32_1`` angegebene Wert von 6,09 GFLOPS lag praktisch am unteren Rand der
Verteilung; der Median liegt bei 21,9. Und die drei großen GEMM-Kernel, die zuvor mit
0,4 und 2,0 Prozent Abstand zueinander beschrieben waren, liegen zwischen 1653 und 1658
GFLOPS und damit innerhalb ihrer eigenen Streuung. Ein Unterschied zwischen ihnen ist
nicht nachweisbar. Diese Korrektur schwächt die Aussage nicht, sondern schärft sie: Die
vollständige 512³-Multiplikation kostet gegenüber dem reinen K-Loop nichts, obwohl
256-mal mehr Sub-Kernel-Aufrufe stattfinden.

4.3 Validierung der Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^

Jeder erzeugte Kernel wird gegen eine naive Referenzimplementierung geprüft, bevor
seine Messung in die Suche eingeht. Bei kleinen Fällen wird vollständig verglichen, bei
großen anhand einer Stichprobe; die Grenze liegt bei 10⁸ Iterationen.

Die Wahl der Testdaten ist dabei nicht nebensächlich. Eine Füllung mit konstanten
Werten liefert bei vertauschten Indizes dasselbe Ergebnis wie bei richtigen und
verdeckt damit genau die Fehlerklasse, die bei Schleifentransformationen am häufigsten
auftritt. Die Eingaben werden deshalb mit teilerfremden Perioden von 13 und 7 gefüllt,
sodass kein Wert doppelt an einer Stelle steht, an der es darauf ankommt. Dass die
Prüfung wirkt, ist selbst getestet: ``test_kernel_validation.cpp`` weist nach, dass ein
einzelnes verfälschtes Element, eine transponierte Ausgabe und eine Nullausgabe
erkannt werden.

Bemerkenswert ist ein Muster, das im Verlauf des Projekts dreimal aufgetreten ist:
**Eine Prüfung bestand jeweils nur deshalb, weil eine andere Stelle den Fehler
verdeckte.**

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Fall
     - Fehler
     - Was ihn verdeckte
   * - Woche 5
     - Referenz indizierte B spaltenweise statt zeilenweise
     - konstante Testfüllung
   * - Woche 6
     - Transponierung war nicht implementiert
     - der Test setzte ``trans_b`` nie
   * - Woche 8
     - Zero-Kernel nullte 2 686 statt 1 769 472 Elemente
     - ein vorheriges ``memset`` im Testrahmen

In allen drei Fällen meldeten die Tests grün. Die Konsequenz für die Praxis ist, dass
eine bestandene Prüfung erst dann etwas aussagt, wenn sie auch fehlschlagen kann. Für
den Autotuner wurde das über Mutationstests abgesichert: Es wurden gezielt Fehler in
den Code eingebaut und geprüft, ob die Testsuite sie meldet.

5. Ergebnisse
-------------

5.1 Hero-Fall
^^^^^^^^^^^^^

``ab-ac-cb`` bei 512³, Simulated Annealing, 30 Trials, Seed 42, adaptiver Messblock.
Rohdaten: ``Projekt/autotuner/results/hero_sme.json``.

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Variante
     - GFLOPS
     - Suchdauer
   * - naiv, ungetunt, Scalar
     - 3,1
     - 1 s
   * - TEIR getunt, Scalar
     - 2,3
     - 602 s
   * - TEIR getunt, NEON
     - 35,2
     - 7 s
   * - TEIR getunt, SME
     - **1428,3**
     - 2 s
   * - TVM, MetaSchedule
     - 505,1
     - übernommen
   * - PyTorch, BLAS/AMX
     - 2370,9
     - übernommen

Mit dem SME-Backend erreicht TEIR das 2,8-fache von TVM und 60 Prozent von PyTorch.
Diese Zahl ist erst nach der Reparatur des SME-Backends entstanden. Zuvor lieferte es
falsche Ergebnisse und wurde von der eigenen Validierung verworfen, sodass bei
``TEIR_BACKEND=sme`` kein einziger Kandidat gültig war. Von außen ist ein defektes
Backend, dessen Kandidaten alle verworfen werden, nicht von einem nicht implementierten
zu unterscheiden. Alle früheren TEIR-Zahlen stammen vom NEON-Backend, und die bis dahin
vertretene Aussage, TEIR erreiche 6 bis 17 Prozent von TVM, galt ausschließlich dafür.

Die Ursachen waren drei unabhängige Fehler in ``codegen.cpp``. Der erste betraf die
Richtung der ZA-Kachelscheibe: ``fmopa`` erzeugt ``za[i][j] = z0[i] * z2[j]``, wobei
``i`` die M- und ``j`` die N-Richtung ist. Eine vertikale Scheibe entspricht einer
Kachelspalte und ist korrekt, wenn C spaltenweise vorliegt, wie in der Assembly aus
Woche 3+4. Der Autotuner hat aber zeilenweises C und braucht die horizontale Scheibe.
Der Code war aus Woche 3+4 übernommen worden, wo er richtig war. Der zweite Fehler
betraf ein Register, das zwar deklariert, aber nicht als Operand an den
Assembler-Block übergeben wurde und zusätzlich in dessen Clobber-Liste stand. Der
dritte war ein ``smstart`` ohne Clobber-Liste, wodurch die Zeitmessvariablen des
Aufrufers zerstört wurden und die gemessene Zeitdifferenz null ergab.

**Die Suche kann das Ergebnis verschlechtern.** Mit Scalar-Backend liefert die getunte
Konfiguration 2,3 GFLOPS gegenüber 3,1 im ungetunten Zustand, und zwar nach 602
Sekunden Suche. Das ist kein Einzelfall: Auch die Regressionstests zeigen regelmäßig
Faktoren zwischen 0,86 und 0,99 gegenüber der Ausgangskonfiguration. Zwei Ursachen
liegen nahe. Der Vorfilter des Cost-Modells siebt Kandidaten aus, bevor sie überhaupt
gemessen werden, und seine Rangkorrelation von rund 0,08 zeigt, dass er die tatsächliche
Reihenfolge kaum trifft. Zusätzlich setzt der Warmstart den Startpunkt der Suche auf
das Optimum des Cost-Modells; ist dieses schlecht gewählt, sucht Simulated Annealing
lokal um einen schlechten Punkt herum. Erschwerend kommt hinzu, dass die
Ausgangskonfiguration kein Strohmann ist, sondern bereits eine vernünftige Schedule.

5.2 Einordnung gegen die Wochenaufgaben
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alle Werte auf derselben Maschine und im selben Semester gemessen.

.. list-table::
   :header-rows: 1
   :widths: 45 30 25

   * - Quelle
     - Kennzahl
     - Wert
   * - Woche 3+4, handgeschriebene SME-Assembly
     - GEMM 512³, Median aus 15 Läufen
     - 1655 GFLOPS
   * - Woche 6, Codegenerator
     - Spitze über 27 Einstellungen
     - 1877 GFLOPS
   * - Woche 7, TEIR-Laufzeitumgebung
     - matmul 8192³, 16 Threads
     - 2547 GFLOPS
   * - Woche 8, AST-Evaluator
     - matmul 8192³
     - 2434 GFLOPS
   * - **Projekt-Autotuner mit SME**
     - ``ab-ac-cb`` bei 512³
     - **1428 GFLOPS**

Der Autotuner liegt damit in derselben Größenordnung wie die von Hand geschriebene
Assembly aus Woche 3+4. Das ist die aussagekräftigere Einordnung als der Vergleich mit
TVM, weil beide Zahlen dieselbe Rechnung auf derselben Maschine beschreiben.

Zwei Einschränkungen gehören dazu. Woche 7 und 8 lösen mit 8192³ ein deutlich größeres
Problem, bei dem sich der Aufwand für Auf- und Abbau des Streaming-Modus über mehr
Rechenarbeit verteilt; die Werte sind deshalb nicht direkt vergleichbar. Und die
Differenz zwischen Woche 7 und Woche 8, also 2547 gegenüber 2434 GFLOPS, entspricht dem
Aufwand für die Auswertung des Syntaxbaums, da beide dieselben Kernel verwenden.

5.3 Cost-Modell und Suchstrategien
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::

   Belege, die vorliegen: Rangkorrelation des Cost-Modells rund 0,08; in der Ablation
   kostet der Vorfilter etwa 6 Prozent Endqualität. Dazu der Vergleich SA gegen GA
   gegen Zufall und die Konvergenzkurven aus ``eval/notebooks``. Hier fehlen noch die
   Diagramme aus den Notebooks, deshalb bleibt der Abschnitt vorerst offen.

6. Limitierungen
----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Einschränkung
     - Belegt durch
   * - SME nur bei GEMM-Form und M, N durch 32 teilbar
     - 1 von 48 Fällen der GETT-Matrix
   * - Das Backend wird nicht gesucht, es ist eine Vorgabe
     - ``backend`` ist Teil von ``AutotunerOptions``, nicht der ``TuningConfig``
   * - Die Suche kann das Ergebnis verschlechtern
     - Hero-Fall Scalar: 2,3 gegen 3,1 GFLOPS
   * - Cost-Modell trägt wenig
     - Rangkorrelation 0,08, Ablation rund 6 Prozent
   * - Suchraum-Abdeckung
     - 30 Trials decken 2,9 Prozent bei 4 Achsen ab, aber nur 0,026 Prozent bei 7
   * - 12 schwere GETT-Fälle nicht vollständig getunt
     - 459 bis 524 Mrd. Iterationen, ein Trial kostet dort Minuten
   * - Die CI belegt die SME-Korrektheit nicht
     - GitHub-Runner haben kein SME, die Tests laufen nur lokal auf dem M4

Zwei dieser Punkte verdienen eine Einordnung, weil sie die Reichweite der Ergebnisse
begrenzen.

**Die Abdeckung des Suchraums fällt mit der Zahl der Achsen dramatisch ab.** Bei vier
Achsen tasten 30 Trials 2,9 Prozent des Raums ab, bei sieben Achsen nur noch 0,026
Prozent. Die Ergebnisse für die kleineren Fälle sagen deshalb wenig über das Verhalten
bei großen Kontraktionen aus. Wer diese Zahl kennt, liest die Ergebnisse anders als
jemand, der nur die GFLOPS-Werte sieht.

**Die Aussage über SME gilt für einen von 48 Fällen.** Der Kernel greift nur bei
GEMM-Form und bei durch 32 teilbaren Werten für M und N. Für alles andere fällt der
Autotuner auf den generischen Weg zurück. Die Zahl 1428 GFLOPS ist damit kein Wert für
die GETT-Matrix insgesamt, sondern für einen bestimmten Fall unter günstigen
Bedingungen.

7. Fazit
--------

Der Autotuner funktioniert für den Fall, für den er gebaut wurde. Mit dem SME-Backend
erreicht er auf dem Hero-Fall 1428 GFLOPS und liegt damit in derselben Größenordnung
wie handgeschriebene Assembly und beim 2,8-fachen von TVM.

Die Grenzen sind allerdings deutlich. Das Cost-Modell trägt mit einer Rangkorrelation
von 0,08 kaum etwas bei, und die Suche liefert im Scalar-Fall ein schlechteres Ergebnis
als die Ausgangskonfiguration. Ein System, das nach zehn Minuten Suche eine schlechtere
Konfiguration zurückgibt als die, mit der es gestartet ist, erfüllt seinen Zweck nicht,
auch wenn es in anderen Fällen gute Zahlen liefert.

Der größere Ertrag des Projekts liegt in der Messmethodik. Drei Fehler haben das System
zwischenzeitlich unbrauchbar gemacht, ohne dass es von außen erkennbar gewesen wäre: die
Messschleife, die den Zeitdeckel um das 64-fache überschritt, ein SME-Backend, dessen
Kandidaten allesamt verworfen wurden, und dreimal eine Prüfung, die nur deshalb bestand,
weil eine andere Stelle den Fehler verdeckte. Jeder dieser Fälle wäre unentdeckt
geblieben, wenn nicht gezielt nachgemessen worden wäre.

Als nächster Schritt bietet sich das Cost-Modell an. Ein Vorfilter, der die Reihenfolge
der Kandidaten kaum trifft, aber ihre Auswahl bestimmt und zugleich den Startpunkt der
Suche vorgibt, ist der wahrscheinlichste Grund dafür, dass die Suche im Scalar-Fall
schadet. Eine Messung ohne Warmstart und ohne Vorfilter würde zeigen, wie viel davon auf
welchen Anteil entfällt.

8. Offenlegung der GenAI-Nutzung
--------------------------------

.. hint::

   Dieser Abschnitt ist zu prüfen und zu ergänzen. Er beschreibt den Stand aus Sicht
   der Werkzeugnutzung; ob er vollständig ist, könnt nur ihr beurteilen.

Für dieses Projekt wurde ein KI-Assistent (Claude) eingesetzt. Die Nutzung umfasste:

- **Fehlersuche und Analyse.** Eingrenzung der drei Fehler im SME-Backend, der
  Messschleife und der Validierung, jeweils mit Prüfung der Hypothesen am Code und an
  Messungen.
- **Aufbereitung der Messdaten.** Skripte für die Messkampagnen, Berechnung von Median
  und Spannweite, Erzeugung der Abbildungen.
- **Erstellung von Textentwürfen** für diesen Bericht sowie für Teile der
  Wochenberichte. Die Entwürfe wurden anschließend überarbeitet.

Nicht eingesetzt wurde der Assistent für die Erhebung der Messwerte selbst; alle Zahlen
stammen aus Läufen auf der Referenzmaschine und liegen als Rohdaten im Repository.
