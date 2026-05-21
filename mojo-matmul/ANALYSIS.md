# mojo-matmul — Mojo naive f32 matmul

**Wave 19 Phase C1.** Naive f32 matmul mirroring `cuda-matmul/matmul.cu`
exactly: 16×16 thread block, one output element per thread, no shared
memory, no tiling. Establishes the scalar-FFMA Mojo baseline so the
TC-enabled `mojo-matmul-tc/` row has something to be 7× faster than.

## Result @ M=N=K ∈ {1024, 2048, 4096}

```
GPU: NVIDIA GeForce RTX 5090 (sm_120, driver 596.21)
Mojo: 1.0.0b1 (a9591de6)
[mojo-matmul] N=1024 iters=10 avg_ms/iter=0.311 TFLOPS=6.90
[mojo-matmul] N=2048 iters=10 avg_ms/iter=2.549 TFLOPS=6.74
[mojo-matmul] N=4096 iters=10 avg_ms/iter=19.28 TFLOPS=7.13
```

## Comparison

| frontend | impl | TFLOPS @ 4096 |
|---|---|---:|
| cuda-matmul (Wave 1) | naive f32 (16×16 BS) | 6.4–7.2 |
| oxide-matmul | unchecked fmuladd | 6.6–7.2 |
| **mojo-matmul (NEW)** | **naive f32** | **7.0–7.1** |

**Parity with nvcc and cuda-oxide naive paths within thermal noise.** All
three frontends produce essentially the same scalar-FFMA SASS for this
algorithm — the GPU SM is fully busy on FFMA throughput, and there's no
algorithmic difference between frontends.

The `mojo-matmul-tc/` (Phase C3) is the more interesting cell — it shows
**how much** Mojo can squeeze out of sm_120 via the `TensorCore` wrapper
(answer: 55.5 TFLOPS, ~7-8× this naive number).

## Algorithm

```
__kernel matmul(A, B, C, n):
    row = blockIdx.y * blockDim.y + threadIdx.y
    col = blockIdx.x * blockDim.x + threadIdx.x
    if row >= n or col >= n: return
    acc = 0.0
    for k in 0..n: acc += A[row*n + k] * B[k*n + col]
    C[row*n + col] = acc
```

Identical to `cuda-matmul/matmul.cu` line for line except for syntax.

## Ceiling

This kernel sits at ~7 TFLOPS on RTX 5090, vs the card's f32 peak of
~104 TFLOPS (rated boost). It's bottlenecked on **uncoalesced B-matrix
loads** (column reads on row-major B) and **no data reuse** (each
A[row,k] and B[k,col] loaded N times). The standard fix sequence is
shared-memory tiling (would push to ~38 TFLOPS — see `cuda-matmul-tiled/`)
then register tiling then tensor cores. We jumped straight to TC
(`mojo-matmul-tc/`) since the f32-microtile interim was already covered
by the existing nvcc/oxide cells.

## Files

- `matmul.mojo` — the kernel + harness
- `run.sh` — reproduce
- `run.log` — fresh run output
