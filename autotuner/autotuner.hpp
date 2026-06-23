#pragma once
#include "teir.hpp"
#include <string>
#include <vector>

struct TuningConfig {
    int split_factor;
    std::vector<std::string> loop_order;
    std::string parallel_axis;

    // Generiert einen kompakten String-Namen für die CSV/Ausgabe
    std::string toString() const {
        std::string orderStr = "";
        for (const auto& o : loop_order) orderStr += o + "-";
        if (!orderStr.empty()) orderStr.pop_back();
        return "split_" + std::to_string(split_factor) + "_order_" + orderStr + "_parallel_" + (parallel_axis.empty() ? "none" : parallel_axis);
    }
};

// Startet den Autotuning-Prozess basierend auf einer Start-IR
void runAutotuner(const TEIR& baseIr);