# Tiled Execution Intermediate Representation

The Tiled Execution Intermediate Representation (TEIR) is a descriptive way to express tensor operations. TEIR consists of axes, primitives and schedules in form of iteration and invocation nodes. See [TEIR](https://tnzr.org/compile/chapters/teir.html) for further details.


## Tasks

 1. Implement a runtime environment that takes TEIR objects and compiles them into a reusable function pointer. The compilation should recursively generate the loop nest from the iteration nodes, use OpenMP for shared-memory parallelization, and use your own code generator for the innermost kernel.
 2. Test and benchmark your implementation using the three examples provided in [data](data/):
    - [transposition.teir](data/transposition.teir): A transposition kernel for $abcd \rightarrow dbac$ with $(96, 128, 48, 32)$
    - [matmul.teir](data/matmul.teir): A blocked matrix multiplication kernel $mk,kn \rightarrow mn$ with $(8192, 8192, 8192)$ as $m_0k_0m_1k_1,k_0n_0k_1n_1 \rightarrow m_0n_0m_1n_1$ with $(256,32,128,64,16,512)$.
    - [contraction.teir](data/contraction.teir): A tensor contraction kernel $pqtu,trus \rightarrow pqrs$ with $(128, 96, 96, 64, 32, 256)$.

## Optimization

TEIR can be adjusted through the following transformations:
 - Split Iteration Node
 - Fuse Iteration Nodes
 - Promote to Primitive
 - Reorder Schedule Chain
 - Set Policy

Use these transformations to write optimization passes for TEIR, e.g., to find suitable kernel axes, to employ cache blocking, or to expose parallelism.

When designing your heuristics, consider the target platform's cache hierarchy (L1/L2/SLC sizes), the number of P-/E-cores, and the SME ZA tile size. Document the parameters and metrics you use to configure your passes.

The unoptimized execution of [matmul.teir](data/matmul.teir) and [contraction.teir](data/contraction.teir) through your runtime serves as the baseline. Additionally, benchmark the einsum $acspx,bspy \rightarrow abcyx$ with $(4,4,3,64,64,1536,1152)$.

### Tasks
 1. Find suitable optimization passes for TEIR.
 2. Test them in an ablation study to measure the impact of individual passes on performance. Verify that every optimized configuration produces numerically correct results compared to the unoptimized baseline.
 3. Report the performance of your runtime in GFLOPS and the speed-up relative to the unoptimized baseline.