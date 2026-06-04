#include "include/teir.h"
#include "include/teir_parser.h"
#include "include/teir_evaluator.h"
#include "include/teir_optimizer.h"

#include <chrono>
#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <cmath>
#include <random>

size_t get_tensor_size(const TEIRProgram& prog, const std::string& tname) {
    size_t max_offset = 0;
    for (auto const& [ax_name, ax] : prog.axes) {
        auto it = ax.strides.find(tname);
        if (it != ax.strides.end()) {
            max_offset += (size_t)(ax.extent - 1) * it->second;
        }
    }
    return max_offset + 4; 
}

void init_tensor_random(void* ptr, size_t bytes) {
    float* arr = static_cast<float*>(ptr);
    size_t elements = bytes / sizeof(float);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < elements; ++i) {
        arr[i] = dis(gen);
    }
}

bool verify_tensors(void* base_ptr, void* opt_ptr, size_t bytes) {
    float* base = static_cast<float*>(base_ptr);
    float* opt = static_cast<float*>(opt_ptr);
    size_t elements = bytes / sizeof(float);
    for (size_t i = 0; i < elements; ++i) {
        if (std::abs(base[i] - opt[i]) > 1e-3) {
            std::cout << "[ERROR] Mismatch at index " << i << ": Base=" << base[i] << " Opt=" << opt[i] << "\n";
            return false;
        }
    }
    return true;
}

void run_ablation_study(const std::string& filepath, const std::string& model_name) {
    std::cout << "\n========================================\n";
    std::cout << "Model: " << model_name << " (" << filepath << ")\n";
    std::cout << "========================================\n";

    TEIRProgram prog_base;
    try {
        prog_base = load_teir(filepath);
    } catch (...) {
        std::cout << "[WARN] File not found or parsing failed.\n";
        return;
    }

    std::vector<void*> base_args, opt_args;
    size_t out_idx = -1;
    for (size_t i = 0; i < prog_base.tensors.size(); ++i) {
        const auto& tname = prog_base.tensors[i];
        if (tname == "out") out_idx = i;
        size_t t_bytes = get_tensor_size(prog_base, tname);
        void* b_ptr = std::malloc(t_bytes);
        void* o_ptr = std::malloc(t_bytes);
        
        if (tname != "out") {
            init_tensor_random(b_ptr, t_bytes);
            std::memcpy(o_ptr, b_ptr, t_bytes);
        } else {
            std::memset(b_ptr, 0, t_bytes);
            std::memset(o_ptr, 0, t_bytes);
        }
        base_args.push_back(b_ptr);
        opt_args.push_back(o_ptr);
    }

    std::cout << ">> Stage 0: Baseline (Unoptimized)\n";
    TEIREvaluator baseline_eval(prog_base);
    auto t0 = std::chrono::high_resolution_clock::now();
    baseline_eval.evaluate(base_args.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double base_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "   Time: " << base_ms << " ms\n";

    for (int stage = 1; stage <= 3; ++stage) {
        std::cout << "\n>> Stage " << stage << ": ";
        if (stage == 1) std::cout << "Parallel Only\n";
        if (stage == 2) std::cout << "Cache Blocking Only\n";
        if (stage == 3) std::cout << "Combined (Parallel + Blocking)\n";

        TEIRProgram prog_opt = load_teir(filepath);
        TEIROptimizer opt(prog_opt);

        if (stage == 1 || stage == 3) {
            opt.expose_parallelism();
        }
        if (stage == 2 || stage == 3) {
            if (model_name == "matmul") opt.apply_cache_blocking_matmul();
            else if (model_name == "contraction") opt.apply_cache_blocking_contraction();
            else if (model_name == "einsum") opt.apply_cache_blocking_einsum();
            else if (model_name == "transposition") opt.apply_cache_blocking_transposition();
        }

        std::memset(opt_args[out_idx], 0, get_tensor_size(prog_base, "out"));

        TEIREvaluator optimized_eval(prog_opt);
        auto t2 = std::chrono::high_resolution_clock::now();
        optimized_eval.evaluate(opt_args.data());
        auto t3 = std::chrono::high_resolution_clock::now();
        double opt_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        
        std::cout << "   Time: " << opt_ms << " ms  |  Speedup vs Base: " << (base_ms / opt_ms) << "x\n";

        bool correct = verify_tensors(base_args[out_idx], opt_args[out_idx], get_tensor_size(prog_base, "out"));
        std::cout << "   Verification: " << (correct ? "PASSED" : "FAILED") << "\n";
    }

    for (size_t i = 0; i < base_args.size(); ++i) {
        std::free(base_args[i]);
        std::free(opt_args[i]);
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "[Week 8] TEIR Ablation Study (In-Memory AST)\n";
    std::cout << "========================================\n";

    run_ablation_study("data/matmul.teir", "matmul");
    run_ablation_study("data/contraction.teir", "contraction");
    run_ablation_study("data/einsum.teir", "einsum");
    run_ablation_study("data/transposition.teir", "transposition");

    return 0;
}
