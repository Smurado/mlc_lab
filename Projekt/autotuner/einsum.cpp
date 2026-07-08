#include "einsum.hpp"

#include <algorithm>
#include <stdexcept>

EinsumSpec parseEinsum(const std::string& einsum) {
    EinsumSpec spec;

    std::vector<std::string> parts;
    size_t prev = 0, pos = 0;
    while ((pos = einsum.find('-', prev)) != std::string::npos) {
        parts.push_back(einsum.substr(prev, pos - prev));
        prev = pos + 1;
    }
    parts.push_back(einsum.substr(prev));

    if (parts.size() != 3)
        throw std::runtime_error("Invalid einsum (expected 3 parts): " + einsum);

    spec.out_idx = parts[0];
    spec.in0_idx = parts[1];
    spec.in1_idx = parts[2];

    for (char c : spec.out_idx) {
        if (std::find(spec.out_axes.begin(), spec.out_axes.end(), c) == spec.out_axes.end())
            spec.out_axes.push_back(c);
    }

    std::vector<char> in_axes;
    for (char c : spec.in0_idx) in_axes.push_back(c);
    for (char c : spec.in1_idx) in_axes.push_back(c);

    for (char c : in_axes) {
        if (spec.out_idx.find(c) == std::string::npos &&
            std::find(spec.reduce_axes.begin(), spec.reduce_axes.end(), c) == spec.reduce_axes.end())
            spec.reduce_axes.push_back(c);
    }

    for (char c : spec.out_axes) spec.all_axes.push_back(c);
    for (char c : spec.reduce_axes) spec.all_axes.push_back(c);

    return spec;
}

int extentOfChar(const TEIR& ir, char c) {
    std::string name(1, c);
    for (const auto& ax : ir.axes)
        if (ax.name == name) return ax.extent;

    std::string name0 = name + "0";
    std::string name1 = name + "1";
    int e0 = 1, e1 = 1;
    bool found0 = false, found1 = false;
    for (const auto& ax : ir.axes) {
        if (ax.name == name0) { e0 = ax.extent; found0 = true; }
        if (ax.name == name1) { e1 = ax.extent; found1 = true; }
    }
    if (found0 && found1) return e0 * e1;

    return 1;
}

std::vector<int> computeStrides(const std::string& idx, const TEIR& ir) {
    int n = (int)idx.size();
    std::vector<int> strides(n);
    if (n == 0) return strides;
    strides[n - 1] = 1;
    for (int i = n - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * extentOfChar(ir, idx[i + 1]);
    return strides;
}

int tensorElements(const std::string& idx, const TEIR& ir) {
    int size = 1;
    for (char c : idx)
        size *= extentOfChar(ir, c);
    return size;
}

double einsumFlops(const TEIR& ir) {
    EinsumSpec spec = parseEinsum(ir.einsum);
    double flops = 2.0;
    for (char c : spec.all_axes)
        flops *= (double)extentOfChar(ir, c);
    return flops;
}

void referenceEinsum(const TEIR& ir, const float* in0, const float* in1, float* out) {
    EinsumSpec spec = parseEinsum(ir.einsum);
    auto in0_strides = computeStrides(spec.in0_idx, ir);
    auto in1_strides = computeStrides(spec.in1_idx, ir);
    auto out_strides = computeStrides(spec.out_idx, ir);

    int out_size = tensorElements(spec.out_idx, ir);
    for (int i = 0; i < out_size; ++i) out[i] = 0.0f;

    int n_axes = (int)spec.all_axes.size();
    std::vector<int> extents(n_axes);
    for (int i = 0; i < n_axes; ++i)
        extents[i] = extentOfChar(ir, spec.all_axes[i]);

    auto axisPos = [&](char c) -> int {
        for (int i = 0; i < n_axes; ++i)
            if (spec.all_axes[i] == c) return i;
        return -1;
    };

    std::vector<int> idx(n_axes, 0);

    while (true) {
        int in0_lin = 0, in1_lin = 0, out_lin = 0;
        for (int i = 0; i < (int)spec.in0_idx.size(); ++i)
            in0_lin += idx[axisPos(spec.in0_idx[i])] * in0_strides[i];
        for (int i = 0; i < (int)spec.in1_idx.size(); ++i)
            in1_lin += idx[axisPos(spec.in1_idx[i])] * in1_strides[i];
        for (int i = 0; i < (int)spec.out_idx.size(); ++i)
            out_lin += idx[axisPos(spec.out_idx[i])] * out_strides[i];

        out[out_lin] += in0[in0_lin] * in1[in1_lin];

        int pos = n_axes - 1;
        while (pos >= 0) {
            idx[pos]++;
            if (idx[pos] < extents[pos]) break;
            idx[pos] = 0;
            pos--;
        }
        if (pos < 0) break;
    }
}

std::string axisExpr(const TEIR& ir, char c) {
    std::string name(1, c);
    for (const auto& ax : ir.axes)
        if (ax.name == name) return name;

    std::string name0 = name + "0";
    std::string name1 = name + "1";
    bool found0 = false, found1 = false;
    int e1 = 1;
    for (const auto& ax : ir.axes) {
        if (ax.name == name0) found0 = true;
        if (ax.name == name1) { found1 = true; e1 = ax.extent; }
    }
    if (found0 && found1)
        return "(" + name0 + " * " + std::to_string(e1) + " + " + name1 + ")";

    return name;
}

bool isReduceAxis(const std::string& schedAxis, const EinsumSpec& spec) {
    if (schedAxis.size() == 1) {
        char c = schedAxis[0];
        return std::find(spec.reduce_axes.begin(), spec.reduce_axes.end(), c) != spec.reduce_axes.end();
    }
    if (schedAxis.size() == 2 && (schedAxis[1] == '0' || schedAxis[1] == '1')) {
        char parent = schedAxis[0];
        return std::find(spec.reduce_axes.begin(), spec.reduce_axes.end(), parent) != spec.reduce_axes.end();
    }
    return false;
}

bool isGEMMForm(const EinsumSpec& spec) {
    if (spec.out_axes.size() != 2 || spec.reduce_axes.size() != 1)
        return false;
    if (spec.in0_idx.size() != 2 || spec.in1_idx.size() != 2)
        return false;
    char out0 = spec.out_axes[0];
    char out1 = spec.out_axes[1];
    char red = spec.reduce_axes[0];
    if (spec.in0_idx[0] == out0 && spec.in0_idx[1] == red &&
        spec.in1_idx[0] == red && spec.in1_idx[1] == out1)
        return true;
    return false;
}

GEMMMapping extractGEMMMapping(const EinsumSpec& spec) {
    GEMMMapping m;
    m.out0_axis = spec.out_axes[0];
    m.out1_axis = spec.out_axes[1];
    m.reduce_axis = spec.reduce_axes[0];
    return m;
}
