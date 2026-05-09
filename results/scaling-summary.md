# Master scaling results (Waves 1+2+3)

Source: `results/scaling.csv` (240 rows: 8 (impl,kernel) configs × 3 N × 10 iters)

All gpu_ms via cudaEventRecord/cuEventRecord (ADR-0001), nvcc 13.2.78 -arch=sm_120 native (ADR-0002).
**IMPORTANT (Wave 3 fix):** cuda-oxide builds require `CUDA_HOME=/usr/local/cuda` so they pick up the
modern libNVVM 22.0.0 from CUDA 13.2, not the system 2023 libNVVM at `/usr/lib/x86_64-linux-gnu/libnvvm.so.4`
(libNVVM 7.0.1, capped at compute_90). This shipped a wrong number for `oxide/safe` in the v0 README.

Backends:
- `cuda-matmul / matmul` — naive nvcc (1 thread = 1 output, no shared mem)
- `cuda-tiled / matmul_tiled` — register-tiled nvcc (32×32 block + 4×4 register micro-tile)
- `cublas-matmul / sgemm` — cuBLAS sgemm with `CUBLAS_PEDANTIC_MATH` (no TF32)
- `oxide / safe` — cuda-oxide naive, slice-indexed (bounds-checked)
- `oxide / unchecked` — cuda-oxide naive, raw-pointer (no bounds checks)
- `oxide / fmuladd` — cuda-oxide naive with `core::intrinsics::fmuladdf32` (post-link emits `fma.rn.f32`)
- `oxide-tiled / safe` — cuda-oxide 16×16 SharedArray tile, slice-indexed
- `oxide-tiled / unchecked` — cuda-oxide 16×16 SharedArray tile, raw-pointer

## N = 1024  (2.15 GFLOP/iter)

| impl | kernel | best ms | median ms | p95 ms | TFLOPS (med) | × naive nvcc |
|---|---|---:|---:|---:|---:|---:|
| cublas-matmul | sgemm | 0.058 | 0.063 | 0.077 | 33.94 | 4.93× |
| cuda-tiled | matmul_tiled | 0.084 | 0.088 | 0.135 | 24.47 | 3.56× |
| oxide-tiled | unchecked | 0.233 | 0.236 | 0.237 | 9.09 | 1.32× |
| oxide-tiled | safe | 0.240 | 0.242 | 0.246 | 8.89 | 1.29× |
| oxide | unchecked | 0.304 | 0.309 | 0.311 | 6.95 | 1.01× |
| oxide | safe | 0.305 | 0.309 | 0.317 | 6.94 | 1.01× |
| oxide | fmuladd | 0.307 | 0.310 | 0.318 | 6.92 | 1.01× |
| cuda-matmul | matmul | 0.309 | 0.312 | 0.325 | 6.88 | 1.00× |

## N = 2048  (17.18 GFLOP/iter)

| impl | kernel | best ms | median ms | p95 ms | TFLOPS (med) | × naive nvcc |
|---|---|---:|---:|---:|---:|---:|
| cublas-matmul | sgemm | 0.262 | 0.273 | 0.289 | 62.98 | 9.95× |
| cuda-tiled | matmul_tiled | 0.500 | 0.514 | 1.441 | 33.44 | 5.28× |
| oxide-tiled | safe | 1.768 | 2.150 | 2.826 | 7.99 | 1.26× |
| oxide-tiled | unchecked | 1.732 | 2.380 | 2.878 | 7.22 | 1.14× |
| oxide | safe | 2.286 | 2.696 | 3.389 | 6.37 | 1.01× |
| cuda-matmul | matmul | 2.320 | 2.715 | 3.644 | 6.33 | 1.00× |
| oxide | unchecked | 2.309 | 2.957 | 3.368 | 5.81 | 0.92× |
| oxide | fmuladd | 2.299 | 3.080 | 3.933 | 5.58 | 0.88× |

## N = 4096  (137.44 GFLOP/iter)

| impl | kernel | best ms | median ms | p95 ms | TFLOPS (med) | × naive nvcc |
|---|---|---:|---:|---:|---:|---:|
| cublas-matmul | sgemm | 1.889 | 2.297 | 3.256 | 59.83 | 9.60× |
| cuda-tiled | matmul_tiled | 3.639 | 4.897 | 5.393 | 28.07 | 4.50× |
| oxide-tiled | unchecked | 15.882 | 17.371 | 18.559 | 7.91 | 1.27× |
| oxide-tiled | safe | 17.030 | 17.917 | 21.452 | 7.67 | 1.23× |
| cuda-matmul | matmul | 19.423 | 22.057 | 25.179 | 6.23 | 1.00× |
| oxide | fmuladd | 22.336 | 24.125 | 28.352 | 5.70 | 0.91× |
| oxide | unchecked | 22.775 | 26.810 | 28.003 | 5.13 | 0.82× |
| oxide | safe | 22.686 | 28.417 | 30.474 | 4.84 | 0.78× |

## TFLOPS vs N (median)

| impl/kernel \ N | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| cublas-matmul/sgemm | 33.94 | 62.98 | 59.83 |
| cuda-tiled/matmul_tiled | 24.47 | 33.44 | 28.07 |
| oxide-tiled/unchecked | 9.09 | 7.22 | 7.91 |
| oxide-tiled/safe | 8.89 | 7.99 | 7.67 |
| oxide/fmuladd | 6.92 | 5.58 | 5.70 |
| cuda-matmul/matmul | 6.88 | 6.33 | 6.23 |
| oxide/unchecked | 6.95 | 5.81 | 5.13 |
| oxide/safe | 6.94 | 6.37 | 4.84 |

## The Rust safety tax (oxide naive: safe/unchecked TFLOPS ratio)

> v0 reported 2.5× safety tax. After fixing libNVVM (Wave 3), the real number is much smaller.

| N | safe TFLOPS | unchecked TFLOPS | fmuladd TFLOPS | safe/unchecked ratio |
|---|---:|---:|---:|---:|
| 1024 | 6.94 | 6.95 | 6.92 | 1.00× |
| 2048 | 6.37 | 5.81 | 5.58 | 0.91× |
| 4096 | 4.84 | 5.13 | 5.70 | 1.06× |

## Tiling speedup (median TFLOPS, tiled / naive)

> See caveat: nvcc tiled is hand-tuned 32×32+4×4 register micro-tile; oxide-tiled is pure 16×16 SharedArray.

| backend | N=1024 | N=2048 | N=4096 |
|---|---:|---:|---:|
| nvcc CUDA C++ | 3.56× | 5.28× | 4.50× |
| cuda-oxide unchecked | 1.31× | 1.24× | 1.54× |
| cuda-oxide safe | 1.28× | 1.25× | 1.59× |
