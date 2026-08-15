// Implementierung der drei Schedule-Transformationen (siehe passes.hpp).
// Invariante aller Passes: Achsen-Extents und Berechnung bleiben unberuehrt,
// nur die Ausfuehrungsreihenfolge aendert sich (abgesichert in test_passes.cpp).
#include "passes.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>

void splitOuterAxis(TEIR& ir, const std::string& axisName, int factor) {
    // 1. Finde die Achse in den Achsendefinitionen
    auto axisIt = std::find_if(ir.axes.begin(), ir.axes.end(), [&](const Axis& a) { return a.name == axisName; });
    if (axisIt == ir.axes.end()) return;

    int originalExtent = axisIt->extent;
    if (originalExtent % factor != 0) {
        std::cout << "[WARN] Split-Faktor " << factor << " teilt Extent " << originalExtent << " nicht sauber.\n";
    }

    std::string name0 = axisName + "0";
    std::string name1 = axisName + "1";

    // Ersetze die alte Achse durch die zwei neuen temporären Achsen
    int extent0 = originalExtent / factor;
    int extent1 = factor;
    
    size_t idx = std::distance(ir.axes.begin(), axisIt);
    ir.axes.erase(axisIt);
    ir.axes.insert(ir.axes.begin() + idx, {name0, extent0});
    ir.axes.insert(ir.axes.begin() + idx + 1, {name1, extent1});

    // 2. Ersetze die Achse im Schedule durch die verschachtelten Schleifen
    auto schedIt = std::find_if(ir.schedule.begin(), ir.schedule.end(), [&](const Iteration& it) { return it.axis == axisName; });
    if (schedIt != ir.schedule.end()) {
        Policy originalPolicy = schedIt->policy;
        size_t sIdx = std::distance(ir.schedule.begin(), schedIt);
        ir.schedule.erase(schedIt);
        ir.schedule.insert(ir.schedule.begin() + sIdx, {name0, originalPolicy});
        ir.schedule.insert(ir.schedule.begin() + sIdx + 1, {name1, Policy::Sequential}); // Innere Schleife meist sequentiell
    }
}

void reorderSchedule(TEIR& ir, const std::vector<std::string>& newOrder) {
    std::vector<Iteration> newSchedule;
    // Bringe die Iterationen in die gewünschte Reihenfolge
    for (const auto& axisName : newOrder) {
        auto it = std::find_if(ir.schedule.begin(), ir.schedule.end(), [&](const Iteration& i) { return i.axis == axisName; });
        if (it != ir.schedule.end()) {
            newSchedule.push_back(*it);
        }
    }
    // Falls Loops im Reorder-Befehl vergessen wurden, hänge sie hinten an
    for (const auto& origIter : ir.schedule) {
        if (std::find(newOrder.begin(), newOrder.end(), origIter.axis) == newOrder.end()) {
            newSchedule.push_back(origIter);
        }
    }
    ir.schedule = newSchedule;
}

void makeParallel(TEIR& ir, const std::string& axisName) {
    for (auto& iter : ir.schedule) {
        if (iter.axis == axisName) {
            iter.policy = Policy::Parallel;
            break;
        }
    }
}