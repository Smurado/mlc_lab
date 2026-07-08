#pragma once

#include "teir.hpp"
#include "autotuner.hpp"

double estimateCost(const TEIR& ir, const TuningConfig& config);

struct CostBreakdown {
    double total_flops;
    double mem_penalty;
    double parallel_factor;
    double split_factor_cost;
    double unroll_factor;
    double estimated_ms;
};

CostBreakdown estimateCostDetailed(const TEIR& ir, const TuningConfig& config);
