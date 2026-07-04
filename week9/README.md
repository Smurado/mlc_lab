# Applications

Supplemental data under this [link](https://cloud.uni-jena.de/s/TskBcqjL3qzcASP)

# 1. Tensor-Ring Reconstruction 

A 5D tensor `T[a, b, c, y, x]` (a 9×9 camera array of 512×512 RGB images) is
stored compressed as five small TR cores of rank `R = 64`. 

```
|a|,|b|,|c|         = 16   (padded with 0)
|y|,|x|             = 512
|p|,|q|,|r|,|s|,|t| = 64
```

Data under `tensor_ring/data/`, all data stored as `FP32` in row-major order.

| file          | shape (row-major) | axes        |
|---------------|-------------------|-------------|
| `eng_Gc.bin`  | `[64, 16, 64]`    | `q, c, r`   |
| `eng_Ga.bin`  | `[64, 16, 64]`    | `r, a, s`   |
| `eng_Gx.bin`  | `[64, 512, 64]`   | `p, x, q`   |
| `eng_Mby.bin` | `[64, 64, 16, 512]` | `p, s, b, y` |

`eng_Mby` is a precomputed intermediate tensor (the result of the `G_y ⊗ G_b` contraction).


Run these three contractions in order:

| step | einsum | sizes |
|------|--------|------|
| S1 | $qcr, ras \rightarrow qcas$     | $(16,16,64,64,64)$ | 
| S2 | $pxq, qcas \rightarrow pxacs$   | $(16,64,64,64,512)$ | 
| S3 | $pxacs, psby \rightarrow xacby$ | $(16,16,16,64,64,512)$ | 


## Tasks

1. Compile the three contractions using the einsums above. Load the four `eng_*.bin` files into aligned `FP32` buffers. Execute them in the given order.
2. Check the following element for reference: T[x=256, a=4, c=0, b=4, y=256] = 0.91
3. Report your performance in GFLOPS
4. Write the reconstructed light field to **PNG**. One **PNG** per camera index. Do not upload the **PNG**. Provide a script to generate the **PNG**.


```

```

---

# 2. Tensorized Neural Network 

A two-layer MLP `x[512] → h[1024] (ReLU) → logits[16]` whose weight matrices are
stored compressed as TT-matrix (MPO) cores joined by a bond axis `c`.

```
|q|,|s|       = 16
|p|,|h|       = 32
|t|           = 64
|c|           = 512 
|o|           = 16   
|b|           = 5000 
```

Data under `tensorized_net/data/`, all data stored as `FP32` in row-major order.


| file       | shape (row-major) | axes      |
|------------|-------------------|-----------|
| `X.bin`    | `[5000, 512]`     | `b, i`    |
| `y.bin`    | `[5000]` (`int32`) | `b`      |
| `G1a.bin`  | `[32, 16, 512]`   | `p, s, c` |
| `G1b.bin`  | `[16, 512, 64]`   | `q, c, t` |
| `G2a.bin`  | `[32, 32]`        | `p, c`    |
| `G2b.bin`  | `[32, 32, 16]`    | `q, c, o` |
| `b1.bin`   | `[1024]`          | `s, t`    |
| `b2.bin`   | `[16]`            | `o`       |

Run four contractions per batch tile (`|b| = TILE`):

| step | einsum | sizes | inputs → output |
|------|--------|------|-----------------|
| c1 | $bqp, psc \rightarrow bqsc$ | $(16,16,32,512,512)$ | `X, G1a → Y1`       |
| c2 | $bqsc, qct \rightarrow bst$ | $(16,16,64,512,512)$ | `Y1, G1b → H[b,o]` |
| c3 | $bhp, pc \rightarrow bhc$   | $(32,32,32,512)$     | `Ht, G2a → Y2`     |
| c4 | $bqc, qco \rightarrow bo$   | $(16,32,32,512)$     | `Y2, G2b → logits` |

Between c2 and c3 the hidden vector is `H[b,o]` (`o = s·O2 + t`): add the bias
and apply ReLU, `H[b,o] = max(0, H[b,o] + b1[o])`, then transpose
`H[b,h1,h2] → Ht[b,h2,h1]` so c3 contracts the minor axis. After c4 add the
output bias to all logits: `logits[b,o] += b2[o]`.

## Tasks

1. Load the data, build the four contractions for `|b| = TILE`, and run them per over the first 10. Report accuracy and GFLOPS.
   batch tile with the biases, ReLU and transpose above.
2. Predict `argmax` over the first 10 logits. Report accuracy
   and GFLOPS.
  

