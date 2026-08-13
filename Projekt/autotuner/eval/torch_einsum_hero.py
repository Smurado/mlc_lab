#!/usr/bin/env python3
"""Hero-Fall (ab-ac-cb @ 512^3): torch.einsum und torch.matmul explizit vergleichen.

Hintergrund: hero_results.json traegt das Label "torch.matmul", gemessen wurde dort
aber ueber gett_bench.run_torch, also torch.einsum("ac,cb->ab"). Dieses Skript misst
beide APIs getrennt mit identischer Methodik (Warmup 10, dann 5 Blaecke a 20 Aufrufe,
bester Blockmittelwert) und legt das Ergebnis als Artefakt in results/ ab.

Aufruf:  .venv/bin/python eval/torch_einsum_hero.py
"""
import json
import os
import time

import torch

N = 512
FLOPS = 2.0 * N * N * N
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "torch_einsum_hero.json")


def bench(fn, warmup=10, blocks=5, iters=20):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        best = min(best, (time.perf_counter() - t0) / iters)
    return FLOPS / best / 1e9


def main():
    torch.set_num_threads(os.cpu_count() or 1)
    A = torch.randn(N, N, dtype=torch.float32)
    B = torch.randn(N, N, dtype=torch.float32)

    results = {
        "case": "ab-ac-cb",
        "size": N,
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "warmup 10, best of 5 blocks x 20 iters, GFLOPS = 2*N^3/t",
        "gflops": {},
    }
    for name, fn in [
        ("torch.einsum(ac,cb->ab)", lambda: torch.einsum("ac,cb->ab", A, B)),
        ("torch.matmul(A,B)", lambda: torch.matmul(A, B)),
    ]:
        val = bench(fn)
        results["gflops"][name] = round(val, 2)
        print(f"{name:28s} {val:8.1f} GFLOPS")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("gespeichert ->", OUT)


if __name__ == "__main__":
    main()
