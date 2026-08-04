// Einstiegspunkt: erst Korrektheits-Tests auf kleinen Problemen
// gegen eine naive Referenz, danach die grossen Benchmarks.
#include "include/teir.h"
#include "include/teir_parser.h"
#include "include/teir_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <omp.h>

struct BenchmarkResult {
    double avg_ms;
    double min_ms;
    double median_ms;
};

static BenchmarkResult benchmark_kernel(TEIRRuntime::Kernel kernel, void** args, int iterations,
                                       const std::function<void()>& reset) {
    if (!kernel || iterations <= 0) return {0.0, 0.0, 0.0};

    if (reset) reset();
    kernel(args); // Warmup

    std::vector<double> run_times;
    run_times.reserve(iterations);
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::infinity();

    for (int i = 0; i < iterations; ++i) {
        if (reset) reset();
        auto it_start = std::chrono::high_resolution_clock::now();
        kernel(args);
        auto it_end = std::chrono::high_resolution_clock::now();
        double run_ms = std::chrono::duration<double, std::milli>(it_end - it_start).count();
        run_times.push_back(run_ms);
        total_ms += run_ms;
        min_ms = std::min(min_ms, run_ms);
    }

    std::sort(run_times.begin(), run_times.end());
    double median_ms = run_times[iterations / 2];
    if (iterations % 2 == 0) {
        median_ms = (run_times[iterations / 2 - 1] + run_times[iterations / 2]) * 0.5;
    }

    return { total_ms / iterations, min_ms, median_ms };
}

// Verifiziert nur die durch die Strides logisch adressierten Output-Elemente
// (sonst werden Padding-Holes im Buffer als FAIL gemeldet).
static bool verify_output(const TEIRProgram& prog, const std::string& tensor,
                          const std::vector<float>& buf, float expected,
                          const std::string& name) {
    std::vector<const Axis*> active;
    for (auto const& [n, ax] : prog.axes) {
        (void)n;
        auto it = ax.strides.find(tensor);
        if (it != ax.strides.end() && it->second != 0) active.push_back(&ax);
    }
    if (active.empty()) {
        std::cout << "   [verify " << name << "] no addressable axes on tensor '" << tensor << "' => SKIP\n";
        return false;
    }

    float tol = std::max(1e-4f, std::abs(expected) * 1e-3f);
    float max_err = 0.0f;
    size_t worst_flat = 0;
    float worst_val = expected;
    long long visited = 0;

    std::vector<int> idx(active.size(), 0);
    while (true) {
        long long byte_off = 0;
        for (size_t i = 0; i < active.size(); ++i) {
            byte_off += (long long)idx[i] * active[i]->strides.at(tensor);
        }
        size_t flat = (size_t)(byte_off / (long long)sizeof(float));
        if (flat < buf.size()) {
            float err = std::abs(buf[flat] - expected);
            if (err > max_err) { max_err = err; worst_flat = flat; worst_val = buf[flat]; }
            ++visited;
        }
        size_t carry = 0;
        while (carry < active.size() && ++idx[carry] >= active[carry]->extent) {
            idx[carry] = 0; ++carry;
        }
        if (carry == active.size()) break;
    }

    bool ok = max_err <= tol;
    std::cout << "   [verify " << name << "] expected=" << expected
              << " max_abs_err=" << max_err
              << " (worst flat idx " << worst_flat << " = " << worst_val << ")"
              << " over " << visited << " elements"
              << " => " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

// Vergleicht zwei Buffer elementweise. tol skaliert mit der erwarteten Groesse,
// damit ein laengerer K-Reduktionspfad nicht falsch-positiv FAILt.
static bool compare_buffers(const std::vector<float>& got, const std::vector<float>& ref,
                            float tol, const std::string& name) {
    float max_err = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        float e = std::abs(got[i] - ref[i]);
        if (e > max_err) { max_err = e; worst = i; }
    }
    bool ok = max_err <= tol;
    std::cout << "   [correctness " << name << "] max_abs_err=" << max_err
              << " (worst idx " << worst << ": got=" << got[worst]
              << " ref=" << ref[worst] << ")"
              << " tol=" << tol
              << " => " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

// Korrektheits-Test 1: 64x64 Matmul mit Zufallsdaten gegen naive Referenz.
// Triggert den NEON-Microkernel; deckt M/N/K-Vertauschungen auf, die der
// All-Einsen-Benchmarktest nicht sieht.
static bool correctness_matmul_small() {
    TEIRProgram prog = load_teir("data/matmul_small.teir");
    TEIRRuntime compiler(prog);
    TEIRRuntime::Kernel kernel = compiler.build();
    if (!kernel) { std::cout << "   [correctness matmul_small] compile failed\n"; return false; }

    constexpr int M = 64, N = 64, K = 64;
    std::vector<float> a(M * K), b(K * N), out(M * N, 0.0f), ref(M * N, 0.0f);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : a) x = dist(rng);
    for (auto& x : b) x = dist(rng);

    // Referenz: ijk row-major, unabhaengig vom JIT.
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += a[m * K + k] * b[k * N + n];
            ref[m * N + n] = s;
        }
    }

    void* args[] = { a.data(), b.data(), out.data() };
    kernel(args);

    // Tol relativ zur Reduktionslaenge K (rund 64 FMAs in [-1,1]).
    return compare_buffers(out, ref, 1e-3f, "matmul_small");
}

// Korrektheits-Test 2: Batch-Matmul (B=8, 32x32x32). Prueft, dass die
// aeussere parallel-Achse korrekt vor das Zero+Gemm-Paar gezogen wird.
static bool correctness_contraction_small() {
    TEIRProgram prog = load_teir("data/contraction_small.teir");
    TEIRRuntime compiler(prog);
    TEIRRuntime::Kernel kernel = compiler.build();
    if (!kernel) { std::cout << "   [correctness contraction_small] compile failed\n"; return false; }

    constexpr int B = 8, M = 32, N = 32, K = 32;
    std::vector<float> a(B * M * K), b(B * K * N), out(B * M * N, 0.0f), ref(B * M * N, 0.0f);

    std::mt19937 rng(67890);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : a) x = dist(rng);
    for (auto& x : b) x = dist(rng);

    for (int bi = 0; bi < B; ++bi) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float s = 0.0f;
                for (int k = 0; k < K; ++k) {
                    s += a[bi * M * K + m * K + k] * b[bi * K * N + k * N + n];
                }
                ref[bi * M * N + m * N + n] = s;
            }
        }
    }

    void* args[] = { a.data(), b.data(), out.data() };
    kernel(args);

    return compare_buffers(out, ref, 1e-3f, "contraction_small");
}

int main(int argc, char** argv) {
    // CLI: --tests-only ueberspringt die grossen Benchmarks,
    //      --bench-only ueberspringt die Korrektheits-Tests.
    bool run_tests = true, run_bench = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--tests-only") run_bench = false;
        else if (a == "--bench-only") run_tests = false;
        else { std::cerr << "unknown option: " << a << "\n"; return 2; }
    }

    std::cout << "[TEIR Runtime System Initializing Evaluation Execution Backend]\n\n";

    int max_threads = omp_get_max_threads();
    std::vector<int> thread_counts = {4, 8, 10};
    if (thread_counts.back() != max_threads) {
        thread_counts.push_back(max_threads);
    }

    std::cout << "[OpenMP max threads] " << max_threads << "\n\n";

    // Phase 1: kleine Korrektheits-Tests mit Zufallsdaten gegen naive Referenz.
    // Laufen schnell und decken Bugs auf, die der All-Einsen-Sanity-Check spaeter
    // nicht sehen wuerde (z.B. vertauschte M/N/K-Achsen im NEON-Kernel).
    int correctness_pass = 0, correctness_total = 0;
    if (run_tests) {
        std::cout << "[Phase 1] Korrektheit auf kleinen Problemen\n";
        ++correctness_total; if (correctness_matmul_small())      ++correctness_pass;
        ++correctness_total; if (correctness_contraction_small()) ++correctness_pass;
        std::cout << "\n";
    }
    if (run_bench) {
        std::cout << "[Phase 2] Sanity-Check + Benchmark auf vollen Groessen\n";
    }

    // flops_per_call > 0 => GFLOPS; sonst falls bytes_per_call > 0 => GB/s.
    auto report = [&](const std::string& name, TEIRRuntime::Kernel kernel, void** args, int iterations,
                      const std::function<void()>& reset, double flops_per_call, double bytes_per_call) {
        BenchmarkResult result = benchmark_kernel(kernel, args, iterations, reset);
        std::cout << " -> Benchmark [" << name << "] (threads=" << omp_get_max_threads() << "): avg "
                  << result.avg_ms << " ms, median " << result.median_ms
                  << " ms, best " << result.min_ms << " ms";
        if (flops_per_call > 0.0) {
            double gflops_best   = flops_per_call / (result.min_ms    * 1e6);
            double gflops_median = flops_per_call / (result.median_ms * 1e6);
            std::cout << "  |  " << gflops_median << " GFLOPS (median), "
                      << gflops_best << " GFLOPS (best)";
        } else if (bytes_per_call > 0.0) {
            double gbs_best   = bytes_per_call / (result.min_ms    * 1e6);
            double gbs_median = bytes_per_call / (result.median_ms * 1e6);
            std::cout << "  |  " << gbs_median << " GB/s (median), "
                      << gbs_best << " GB/s (best)";
        }
        std::cout << "  [" << iterations << " runs]\n";
    };

    auto run_benchmark = [&](const std::string& name, TEIRRuntime::Kernel kernel, void** args,
                             const std::function<void()>& reset, int iterations,
                             double flops_per_call, double bytes_per_call) {
        if (!kernel) return;
        std::cout << " -> " << name << " kernel ready. Sweeping thread counts...\n";
        for (int threads : thread_counts) {
            omp_set_num_threads(threads);
            std::cout << "   threads=" << threads << " ... " << std::flush;
            report(name, kernel, args, iterations, reset, flops_per_call, bytes_per_call);
        }
        std::cout << "\n";
    };

    int total_pass = 0, total_tests = 0;

    if (run_bench) {
    // --- Test 1: Transposition ---
    TEIRProgram trans_prog = load_teir("data/transposition.teir");
    TEIRRuntime compiler_trans(trans_prog);
    TEIRRuntime::Kernel trans_kernel = compiler_trans.build();
    constexpr float TRANS_FILL = 1.23f;
    std::vector<float> input_tensor(96LL * 128 * 48 * 32, TRANS_FILL);
    std::vector<float> output_tensor(32LL * 128 * 96 * 48, 0.0f);
    void* trans_args[] = { input_tensor.data(), output_tensor.data() };
    auto reset_trans = [&]() { std::fill(output_tensor.begin(), output_tensor.end(), 0.0f); };
    const double trans_bytes = 2.0 * 4.0 * (double)input_tensor.size();

    if (trans_kernel) {
        std::cout << " -> Compilation Successful: Transposition Kernel Pointer Ready.\n";
        reset_trans();
        trans_kernel(trans_args);
        ++total_tests;
        if (verify_output(trans_prog, "out", output_tensor, TRANS_FILL, "transposition")) ++total_pass;
        run_benchmark("transposition", trans_kernel, trans_args, reset_trans, 10,
                      /*flops*/ 0.0, /*bytes*/ trans_bytes);
    }

    // --- Test 2: Matmul ---
    TEIRProgram matmul_prog = load_teir("data/matmul.teir");
    TEIRRuntime compiler_matmul(matmul_prog);
    TEIRRuntime::Kernel matmul_kernel = compiler_matmul.build();
    std::vector<float> in0_tensor_mm(256LL * 32 * 16 * 512, 1.0f);
    std::vector<float> in1_tensor_mm(16LL * 512 * 128 * 64, 1.0f);
    std::vector<float> out_tensor_mm(256LL * 32 * 128 * 64, 0.0f);
    void* matmul_args[] = { in0_tensor_mm.data(), in1_tensor_mm.data(), out_tensor_mm.data() };
    auto reset_matmul = [&]() { std::fill(out_tensor_mm.begin(), out_tensor_mm.end(), 0.0f); };
    const double matmul_flops = 2.0 * 8192.0 * 8192.0 * 8192.0;
    const float matmul_expected = 16.0f * 512.0f;

    if (matmul_kernel) {
        std::cout << " -> Compilation Successful: Matmul Kernel Pointer Ready.\n";
        reset_matmul();
        matmul_kernel(matmul_args);
        ++total_tests;
        if (verify_output(matmul_prog, "out", out_tensor_mm, matmul_expected, "matmul")) ++total_pass;
        run_benchmark("matmul", matmul_kernel, matmul_args, reset_matmul, 3,
                      /*flops*/ matmul_flops, /*bytes*/ 0.0);
    }

    // --- Test 3: Contraction ---
    TEIRProgram contraction_prog = load_teir("data/contraction.teir");
    TEIRRuntime compiler_contract(contraction_prog);
    TEIRRuntime::Kernel contract_kernel = compiler_contract.build();
    if (contract_kernel) {
        std::cout << " -> Compilation Successful: Contraction Kernel Pointer Ready.\n";
        std::vector<float> in0_tensor_co(128LL * 96 * 32 * 256, 1.0f);
        std::vector<float> in1_tensor_co(32LL * 96 * 256 * 64, 1.0f);
        std::vector<float> out_tensor_co(128LL * 96 * 96 * 64, 0.0f);
        void* contraction_args[] = { in0_tensor_co.data(), in1_tensor_co.data(), out_tensor_co.data() };
        auto reset_contraction = [&]() { std::fill(out_tensor_co.begin(), out_tensor_co.end(), 0.0f); };

        const double contract_flops =
            2.0 * 128.0 * 96.0 * 96.0 * 64.0 * 32.0 * 256.0;
        const float contract_expected = 32.0f * 256.0f;

        reset_contraction();
        contract_kernel(contraction_args);
        ++total_tests;
        if (verify_output(contraction_prog, "out", out_tensor_co, contract_expected, "contraction")) ++total_pass;
        run_benchmark("contraction", contract_kernel, contraction_args, reset_contraction, 3,
                      /*flops*/ contract_flops, /*bytes*/ 0.0);
    }
    } // run_bench

    std::cout << "\n[Unit-Tests] Phase 1 (Korrektheit): " << correctness_pass << "/" << correctness_total
              << " | Phase 2 (Sanity): " << total_pass << "/" << total_tests << "\n";
    int all_pass = correctness_pass + total_pass;
    int all_total = correctness_total + total_tests;
    std::cout << "[Unit-Tests] " << all_pass << "/" << all_total << " bestanden.\n";
    return (all_pass == all_total) ? 0 : 1;
}
