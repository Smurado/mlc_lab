// Analytisches Cost-Modell: schaetzt Kandidaten-Kosten aus Zugriffsmuster,
// Parallelisierung, Arbeitssatz und Unrolling. Kalibriert sich einmalig auf
// der Zielmaschine (TEIR_CALIBRATE=0 schaltet das ab). Bewertung der
// Modellguete: Bericht, Abschnitt 5.3; Invarianten: test_cost_model.cpp.
#include "cost_model.hpp"
#include "einsum.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// C3: Kalibrierte Modellparameter. Defaults sind konservative Schaetzungen und
// gelten, solange calibrateCostModel() nicht lief (oder TEIR_CALIBRATE=0).
// - peak_gflops: sequenzielle Scalar-Peak-Rate der Zielmaschine. Der Divisor,
//   mit dem das Modell die reine Rechenzeit schaetzt.
// - parallel_overhead_ms: Fixkosten eines OpenMP-Parallelblocks (Spawn/Sync).
static double g_peak_gflops = 10.0;
static double g_parallel_overhead_ms = 0.5;

double costModelPeakGflops() { return g_peak_gflops; }
double costModelParallelOverheadMs() { return g_parallel_overhead_ms; }

// Misst eine repraesentative sequenzielle Peak-Rate ueber einen naiven GEMM in
// derselben Compile-Umgebung wie der Autotuner selbst (-O3 -march=native). Das
// ist absichtlich KEIN theoretischer Peak, sondern das, was ein anstaendiger
// skalarer Triple-Loop hier real erreicht — genau das Regime, in dem die
// generierten Kernel laufen.
static double measurePeakGflops() {
    const int N = 96; // 96^3 = ~885k FMA -> ~1.77 MFLOP pro Durchlauf, passt in L2
    std::vector<float> A(N * N), B(N * N), C(N * N, 0.0f);
    for (int i = 0; i < N * N; ++i) { A[i] = (float)((i % 13) + 1) / 13.0f;
                                      B[i] = (float)((i % 7) + 1) / 7.0f; }
    auto gemm = [&]() {
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < N; ++k) {
                const float a = A[i * N + k];
                for (int j = 0; j < N; ++j)
                    C[i * N + j] += a * B[k * N + j];
            }
    };
    gemm(); // Warmup (Caches fuellen, Code laden)

    const double total_flops = 2.0 * N * N * N;
    double best_ms = 1e18;
    for (int rep = 0; rep < 5; ++rep) {
        std::fill(C.begin(), C.end(), 0.0f);
        auto t0 = std::chrono::high_resolution_clock::now();
        int iters = 0;
        double blockMs = 0.0;
        while (blockMs < 30.0) { // mind. 30ms messen -> stabil
            gemm();
            ++iters;
            blockMs = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t0).count();
        }
        const double ms = blockMs / iters;
        if (ms < best_ms) best_ms = ms;
    }
    // Verhindere, dass der Optimizer den GEMM wegoptimiert.
    volatile float sink = C[0]; (void)sink;
    return (best_ms > 0.0) ? (total_flops / 1e9) / (best_ms / 1000.0) : g_peak_gflops;
}

// Misst die reinen Fixkosten eines OpenMP-Parallelblocks (Thread-Spawn/Sync),
// indem ein leerer Parallelblock oft ausgefuehrt und gemittelt wird. Der Aufruf
// von omp_get_thread_num() haelt den Compiler davon ab, den Block zu eliminieren.
static double measureParallelOverheadMs() {
#ifdef _OPENMP
    volatile int sink = 0;
    // Warmup: OpenMP-Threadpool einmalig hochfahren, damit wir die stabilen
    // Wiederhol-Fixkosten messen und nicht die einmalige Pool-Erzeugung.
    for (int i = 0; i < 50; ++i) {
        #pragma omp parallel
        { if (omp_get_thread_num() < 0) sink = -1; }
    }
    const int N = 2000;
    double best_ms = 1e18;
    for (int rep = 0; rep < 5; ++rep) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            #pragma omp parallel
            { if (omp_get_thread_num() < 0) sink = -1; }
        }
        const double blockMs = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() - t0).count();
        const double per = blockMs / N;
        if (per < best_ms) best_ms = per;
    }
    (void)sink;
    return best_ms;
#else
    return g_parallel_overhead_ms;
#endif
}

void calibrateCostModel() {
    const char* env = std::getenv("TEIR_CALIBRATE");
    const bool disabled = env && (std::string(env) == "0" || std::string(env) == "off" ||
                                  std::string(env) == "false");
    if (disabled) {
        std::printf("[COSTMODEL] Kalibrierung deaktiviert (TEIR_CALIBRATE=0) — "
                    "Defaults: peak=%.1f GFLOPS, thread_overhead=%.3f ms\n",
                    g_peak_gflops, g_parallel_overhead_ms);
        return;
    }
    const double peak = measurePeakGflops();
    const double overhead = measureParallelOverheadMs();
    // Nur uebernehmen, wenn die Messung plausibel (>0) ist; sonst Default behalten.
    if (peak > 0.0) g_peak_gflops = peak;
    if (overhead > 0.0) g_parallel_overhead_ms = overhead;
    std::printf("[COSTMODEL] Kalibriert auf Zielmaschine: peak=%.2f GFLOPS "
                "(war 10.0), thread_overhead=%.4f ms (war 0.5)\n",
                g_peak_gflops, g_parallel_overhead_ms);
}

static int strideOfChar(const std::string& idx, const std::vector<int>& strides, char c) {
    for (int i = 0; i < (int)idx.size(); ++i)
        if (idx[i] == c) return strides[i];
    return 0;
}

CostBreakdown estimateCostDetailed(const TEIR& ir, const TuningConfig& config) {
    CostBreakdown cb = {0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0};

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
        if (work_per_thread < 1e6) {
            double ratio = work_per_thread / 1e6;
            cb.parallel_factor *= 1.0 + 9.0 * (1.0 - ratio);
        } else {
            cb.parallel_factor *= 0.8;
        }
        // Fixkosten: Thread-Spawn/Sync eines OpenMP-Blocks, unabhaengig vom
        // Workload. Bei tiny Workloads (<<1ms Compute) dominiert dies komplett.
        // Wert wird beim Start kalibriert (calibrateCostModel), sonst Default.
        cb.parallel_overhead_ms = g_parallel_overhead_ms;
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

    const double peak_gflops = g_peak_gflops; // kalibriert (calibrateCostModel) oder Default

    // Sequenzielle Rechenzeit OHNE Parallel-Faktor. Sie ist die Referenz fuer
    // die Overhead-Pruefung: Thread-Spawn lohnt sich hoechstens, wenn die reine
    // Rechenlast die Fixkosten uebersteigt. (Der Parallel-Faktor darf hier NICHT
    // einfliessen, sonst blaeht er die Compute-Zeit auf und macht den Overhead
    // faelschlich "vernachlaessigbar" — genau der Bug, der Parallelisierung bei
    // winzigen Workloads durchrutschen liess.)
    double seq_flops = cb.total_flops * cb.mem_penalty
                       * cb.split_factor_cost * cb.unroll_factor;
    double seq_compute_ms = (seq_flops / 1e9) / peak_gflops * 1000.0;

    double effective_flops = seq_flops * cb.parallel_factor;
    double compute_ms = (effective_flops / 1e9) / peak_gflops * 1000.0;

    // Best-Case einer Parallelisierung (unendlich viele Threads) ist der reine
    // Overhead — die Rechenzeit geht gegen 0. Ist dieser Overhead bereits >= der
    // sequenziellen Rechenzeit, kann Parallelisierung SIE NIE schlagen. Solche
    // Configs sind strukturell absurd (z.B. 6-Achsen-Kontraktion mit Extent 3:
    // ~microsekunden Compute vs. 0.5ms Thread-Spawn) und werden hart aussortiert,
    // damit sie gar nicht erst in den JIT-Kandidatenpool gelangen.
    if (cb.parallel_overhead_ms > 0.0 && cb.parallel_overhead_ms >= seq_compute_ms) {
        cb.estimated_ms = 1e18;
    } else {
        cb.estimated_ms = compute_ms + cb.parallel_overhead_ms;
    }

    return cb;
}

double estimateCost(const TEIR& ir, const TuningConfig& config) {
    return estimateCostDetailed(ir, config).estimated_ms;
}
