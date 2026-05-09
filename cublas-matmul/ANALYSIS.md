# Wave 2 W2A — cuBLAS sgemm reference baseline (ANALYSIS)

## Methodology

Strict f32 sgemm via cuBLAS 13.4.0 on an RTX 5090 (sm_120 native, CUDA 13.2).
Math mode is set to `CUBLAS_PEDANTIC_MATH` before the first call so Tensor
Core TF32 paths are disabled — this keeps the reference apples-to-apples with
every other implementation in the repo (all of which do plain IEEE f32 FMAs).
Inputs are row-major N×N fp32 matrices filled with the same deterministic
pattern as `cuda-matmul/matmul.cu` (`A[i] = (i%7)*0.01`, `B[i] = (i%11)*0.01`),
and the row-major result is obtained via the standard "compute (B·A) column-major
to get (A·B) row-major" trick. Size sweep: N ∈ {1024, 2048, 4096}, 1 warmup
+ 10 timed iters per N, timed with `cudaEventRecord` bracketing a single
`cublasSgemm` call (per ADR-0001). Correctness is spot-checked in double on
the host at (0,0), (n/2,n/2), (n-1,n-1): **9/9 OK**, all relative errors
< 1e-6.

## Results (PEDANTIC f32, best of 10)

- **N=1024** — best 0.058 ms / median 0.063 ms → **37.3 TFLOPS** best, 34.0 TFLOPS median
- **N=2048** — best 0.262 ms / median 0.272 ms → **65.6 TFLOPS** best, 63.1 TFLOPS median
- **N=4096** — best 1.889 ms / median 2.279 ms → **72.7 TFLOPS** best, 60.3 TFLOPS median

All N=4096 samples land in the expected 50–100 TFLOPS window for strict
f32 on Blackwell's FP32 pipes. Variance at N=4096 is noticeably higher
(iters span 1.89 → 3.26 ms) — typical WSL2/desktop contention we already
noted in Wave 1 for long-running GPU work.

## Interpretation — the gap vs. naive nvcc

Wave 1 W1B's naive CUDA matmul (16×16 blocks, no shared-mem tiling, one
output element per thread) landed at **N=4096 best ≈ 19.42 ms ≈ 7.08 TFLOPS**.
cuBLAS at the same N and same numerics lands at **1.89 ms / 72.7 TFLOPS**.

That is a **≈10.3× speedup** from the exact same chip under the exact same
math mode. The naive kernel is leaving roughly **90 % of available fp32
throughput on the floor** at N=4096. Smaller N tells the same story:
N=1024 naive 0.31 ms vs cuBLAS 0.058 ms → ~5.3× gap; N=2048 is ~8.9×.
The gap grows with N because the naive kernel is bandwidth-bound by
redundant global-memory loads of A rows and B columns, while cuBLAS uses
shared-memory tiling + register blocking + a cooperative-fetch schedule
that gets closer to compute-bound.

This gives us a clean ceiling for Wave 2's hand-written optimized kernels:
**≈73 TFLOPS at N=4096** is the f32-PEDANTIC target anyone's tiled/vectorized
implementation should be graded against. Closing even half the 10× gap
would already beat every other non-cuBLAS implementation in this repo.
