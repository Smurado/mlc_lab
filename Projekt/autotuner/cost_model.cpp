#include "cost_model.hpp"
#include "einsum.hpp"
#include <algorithm>
#include <cmath>

static int strideOfChar(const std::string& idx, const std::vector<int>& strides, char c) {
    for (int i = 0; i < (int)idx.size(); ++i)
        if (idx[i] == c) return strides[i];
    return 0;
}

CostBreakdown estimateCostDetailed(const TEIR& ir, const TuningConfig& config) {
    CostBreakdown cb = {0, 1.0, 1.0, 1.0, 1.0, 0.0};

    if (ir.einsum.empty()) {
        cb.estimated_ms = 1.0;
        return cb;
    }

    EinsumSpec spec = parseEinsum(ir.einsum);
    cb.total_flops = einsumFlops(ir);

    if (config.loop_order.empty()) {
        cb.estimated_ms = 1e18;
        return cb;
    }

    const std::string& inner = config.loop_order.back();

    char parentChar = 0;
    bool isSplit1 = false;
    if (inner.size() == 2 && inner[1] == '1') {
        parentChar = inner[0];
        isSplit1 = true;
    } else if (inner.size() == 1) {
        parentChar = inner[0];
    } else {
        cb.estimated_ms = 1e18;
        return cb;
    }

    auto in0_strides = computeStrides(spec.in0_idx, ir);
    auto in1_strides = computeStrides(spec.in1_idx, ir);

    int in0_stride = strideOfChar(spec.in0_idx, in0_strides, parentChar);
    int in1_stride = strideOfChar(spec.in1_idx, in1_strides, parentChar);

    cb.mem_penalty = 1.0;
    if (in0_stride > 1) cb.mem_penalty += 0.5 * std::log2((double)in0_stride);
    if (in1_stride > 1) cb.mem_penalty += 0.5 * std::log2((double)in1_stride);

    int in0_present = (in0_stride > 0) ? 1 : 0;
    int in1_present = (in1_stride > 0) ? 1 : 0;
    if (in0_present + in1_present == 0) {
        cb.estimated_ms = 1e18;
        return cb;
    }

    if (!config.parallel_axis.empty()) {
        int parallel_extent = 1;
        for (const auto& ax : ir.axes)
            if (ax.name == config.parallel_axis) parallel_extent = ax.extent;

        double work_per_thread = cb.total_flops / std::max(1, parallel_extent);
        if (work_per_thread < 1e6)
            cb.parallel_factor += (1e6 - work_per_thread) / 1e6;
        if (parallel_extent > 1)
            cb.parallel_factor *= 0.8;
    }

    cb.split_factor_cost = 1.0;
    if (config.split_factor > 1 && isSplit1) {
        int tile_extent = config.split_factor;
        int working_set_bytes = tile_extent * 4 * (in0_present + in1_present);

        if (working_set_bytes < 8192)
            cb.split_factor_cost = 0.85;
        else if (working_set_bytes < 32768)
            cb.split_factor_cost = 0.95;
        else
            cb.split_factor_cost = 1.1 + std::log2((double)working_set_bytes / 32768) * 0.1;
    }

    cb.unroll_factor = 1.0;
    if (config.unroll_factor > 1) {
        int inner_extent = 1;
        for (const auto& ax : ir.axes)
            if (ax.name == inner) inner_extent = ax.extent;

        if (config.unroll_factor > inner_extent)
            cb.unroll_factor = 1.05;
        else if (config.unroll_factor <= 8)
            cb.unroll_factor = 0.95;
        else
            cb.unroll_factor = 1.05;
    }

    const double peak_gflops = 10.0;
    double effective_flops = cb.total_flops * cb.mem_penalty * cb.parallel_factor
                             * cb.split_factor_cost * cb.unroll_factor;
    cb.estimated_ms = (effective_flops / 1e9) / peak_gflops * 1000.0;

    return cb;
}

double estimateCost(const TEIR& ir, const TuningConfig& config) {
    return estimateCostDetailed(ir, config).estimated_ms;
}
