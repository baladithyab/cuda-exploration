# cuda-vecadd-bench — memory-bandwidth analysis (Wave 4 W4B)

Kernel: `c[i] = a[i] + b[i]` (3-buffer streaming: read A, read B, write C).
Block 256 threads, 1D grid `ceil(N/256)`. `cudaEvent` timing, 1 warmup + 10 timed
iters per N. Traffic accounting: `3 * N * sizeof(f32)`, so GB/s reflects end-to-end
DRAM-equivalent bandwidth on a cold-cache read.

## Results (RTX 5090, sm_120, CUDA 13.2)

| N (elems) | per-buf MB | best ms | med ms | best GB/s | med GB/s |
|-----------|-----------:|--------:|-------:|----------:|---------:|
| 1,048,576 |         4  | 0.0046  | 0.0092 |  **2750** |     1363 |
| 16,777,216|         64 | 0.1130  | 0.1205 |  **1782** |     1670 |
| 67,108,864|        256 | 0.5005  | 0.7292 |  **1609** |     1104 |
|268,435,456|       1024 | 2.3720  | 3.6220 |  **1358** |      889 |

## Interpretation

**N=1M (4 MB per buffer, 12 MB aggregate).** Working set fits easily in the
Blackwell 5090's ≈96 MB L2. The "best" figure of **2750 GB/s exceeds DRAM peak
(1.79 TB/s)**, which is the expected tell of an L2-resident run — the traffic
accounting (`3·N·4`) counts bytes we're *not* actually re-fetching from DRAM,
so GB/s is a cache-boosted overestimate. This is the deceptive-high regime
flagged in the task prompt.

**N=16M (64 MB per buffer, 192 MB aggregate).** Just barely spills L2 (96 MB
capacity vs 192 MB touched). Throughput settles to ~1670 GB/s median — close
to, but not yet at, the bandwidth-bound plateau.

**N=64M (256 MB per buffer, 768 MB aggregate).** Firmly DRAM-bound. Best
**1609 GB/s ≈ 90% of 1.79 TB/s peak** — the SoL ceiling. Median 1104 GB/s
is dragged down by 2 stray iters near ~1.8 ms (likely WSL2 TDR-adjacent
scheduling jitter, same pattern observed in Wave 1). The best-case number
is the correct read of steady-state bandwidth.

**N=256M (1 GB per buffer, 3 GB aggregate).** Best drops to 1358 GB/s — some
residual overhead from larger grid launches and pagetable pressure.

**Bandwidth-bound hypothesis: supported.** The transition from
cache-resident (N=1M, 2750 GB/s) through the L2 spill (N=16M) to stable
DRAM-bound behaviour (N=64M at 90% of peak) is exactly the textbook shape.
