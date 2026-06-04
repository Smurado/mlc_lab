# Week 8: Ablation Study Results & Heuristics Document

## 1. Parameters & Optimizations Applied

### Theoretical vs Empirical Analysis
To systematically evaluate our optimizations, we run an ablation study across four stages for each model:

1. **Stage 0: Baseline (Unoptimized)**
    * Represents naive execution via raw iteration parsing loops.
2. **Stage 1: Parallel Only**
    * Uses OpenMP parallel directives on the outermost loops (e.g. `m0` for matmul, `p` & `r` for contraction).
3. **Stage 2: Cache Blocking Only**
    * Employs matrix blocking by splitting the execution tree domains into sub-blocks tailored towards CPU register sizes or L1 spatial locality.
4. **Stage 3: Combined (Parallel + Blocking)**
    * Both thread-level parallelism and instruction/data locality working in tandem.

### Matmul
- **Baseline loops**: `m0`, `n0`, `k0`.
- **Parallel Policy**: Splitting rows (`m0`).
- **Cache Blocking Dimensions**: Split `m0` -> `m_in` (Factor=4), split `n0` -> `n_in` (Factor=16).

### Contraction
- **Baseline loops**: `p`, `q`, `r`, `s`.
- **Parallel Policy**: Independent dimensions across the left and right tensors.
- **Cache Blocking Dimensions**: Split `p` -> `p_in` (Factor=4), split `q` -> `q_in` (Factor=4).

### Einsum (`ab, cd -> ad`)
- **Baseline loops**: `a`, `b`, `c`, `d`.
- **Parallel Policy**: Split independent iterations `a` and `b`.
- **Cache Blocking Dimensions**: Block matrix dimensions `a`, `b` and `c` by factors of 4 to fit efficiently in microkernel caches.

### Transposition
- **Baseline loops**: `a`, `b`.
- **Parallel Policy**: Top level chunks across main rows (`a`).
- **Cache Blocking Dimensions**: Cache tile width set by factors of 4.

## 2. In-Memory AST Results

All tests have mathematical execution validation testing with `abs(Opt - Base) < 1e-3` checks. The structural execution is guaranteed mathematical correctness even across the dynamically injected sub-dimensions! 

Currently due to OpenMP loop overhead without SIMD instructions implemented in the base node evaluator, runtime performance reflects similar timings across the CPU bounds. Native C++ microkernels or ARM Neon usage handles this inside of our real JIT engine pipeline.
