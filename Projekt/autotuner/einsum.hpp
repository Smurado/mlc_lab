#pragma once

#include "teir.hpp"

#include <string>
#include <vector>

struct EinsumSpec {
    std::string out_idx;
    std::string in0_idx;
    std::string in1_idx;
    std::vector<char> out_axes;
    std::vector<char> reduce_axes;
    std::vector<char> all_axes;
};

EinsumSpec parseEinsum(const std::string& einsum);

int extentOfChar(const TEIR& ir, char c);

std::vector<int> computeStrides(const std::string& idx, const TEIR& ir);

int tensorElements(const std::string& idx, const TEIR& ir);

double einsumFlops(const TEIR& ir);

void referenceEinsum(const TEIR& ir, const float* in0, const float* in1, float* out);
