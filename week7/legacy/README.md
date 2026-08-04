# Alter Weg (Woche 7, vor der Ueberarbeitung)

Diese beiden Dateien erzeugten C++-Quelltext, riefen `clang++` als externen
Prozess auf und luden das Ergebnis per `dlopen`. Das Betreuer-Feedback dazu:

> Der Weg, die C++-Dateien zu erstellen, diese per Command zu kompilieren, das SO
> zu laden und darauf den Pointer zu halten ist ok, war aber nicht der angedachte
> Weg. Es sollte direkt im Arbeitsspeicher gearbeitet werden, entweder per
> rekursiven oder gejitteten Schleifen.

Ersetzt durch `teir_runtime.cpp`: rekursives Ablaufen des Schleifennests zur
Laufzeit, innerste Kernel aus dem eigenen Codegenerator (`mini_jit`, Woche 5/6).
Zum Vergleich aufgehoben, nicht mehr im Build.
