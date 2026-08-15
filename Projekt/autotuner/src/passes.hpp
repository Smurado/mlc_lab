// Die drei Schedule-Transformationen, aus denen jede TuningConfig entsteht:
// Achse kacheln (a -> a0/a1), Schleifen umordnen, eine Achse parallelisieren.
// Sie veraendern nur die Ausfuehrung, nie das Ergebnis (siehe Bericht, Kap. 3).
#pragma once
#include "teir.hpp"
#include <string>
#include <vector>

// Teilt eine Achse in Outer und Inner (Loop Tiling / Blocking)
void splitOuterAxis(TEIR& ir, const std::string& axisName, int factor);

// Ändert die Reihenfolge der Iterations-Schleifen im Schedule
void reorderSchedule(TEIR& ir, const std::vector<std::string>& newOrder);

// Setzt die Ausführungs-Policy einer Achse auf Parallel
void makeParallel(TEIR& ir, const std::string& axisName);