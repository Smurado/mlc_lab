#pragma once
#include <string>
#include <vector>
#include <iostream>

enum class Policy {
    Sequential,
    Parallel
};

// Codegen-Backend:
// - SCALAR nutzt Autovektorisierung + OpenMP,
// - NEON generiert echten 4x4 ARM NEON Outer-Product-GEMM (vmlaq_n_f32),
// - SME generiert echten ARM SME Outer-Product-GEMM (smstart/fmopa/smstop).
enum class Backend {
    Scalar,
    NEON,
    SME
};

inline std::string policyToString(Policy p) {
    return p == Policy::Parallel ? "parallel" : "sequential";
}

struct Axis {
    std::string name;
    int extent;
};

struct Tensor {
    std::string name;
    std::string type;
};

struct Iteration {
    std::string axis;
    Policy policy;
};

struct TEIR {
    std::string name;
    std::vector<Tensor> tensors;
    std::vector<Axis> axes;
    std::vector<std::string> primitives;
    std::vector<Iteration> schedule;
    std::vector<std::string> invokes;
    int unrollFactor = 1; // Unroll-Faktor fuer die innerste Schleife (Codegen)
    Backend backend = Backend::Scalar; // Codegen-Backend (Scalar, NEON oder SME)
    std::string einsum; // Optionale Einstein-Notation der Kontraktion (z.B. "abc-bda-dc")

    // Hilfsfunktion zur Verifizierung: Gibt die geladene IR wieder aus
    void print() const {
        std::cout << "teir @" << name << " {\n";
        for (const auto& tensor : tensors) {
            std::cout << "  tensor %" << tensor.name << " : " << tensor.type << "\n";
        }
        for (const auto& axis : axes) {
            std::cout << "  axis @" << axis.name << " extent " << axis.extent << "\n";
        }
        for (const auto& prim : primitives) {
            std::cout << "  primitive @" << prim << "\n";
        }
        std::cout << "  schedule {\n";
        for (const auto& iter : schedule) {
            std::cout << "    iter @" << iter.axis << " policy " << policyToString(iter.policy) << "\n";
        }
        for (const auto& inv : invokes) {
            std::cout << "    invoke @" << inv << "\n";
        }
        std::cout << "  }\n}\n";
    }
};