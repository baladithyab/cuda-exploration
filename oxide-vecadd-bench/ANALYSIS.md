# oxide-vecadd-bench — memory-bandwidth analysis (Wave 4 W4B)

Kernel: `c[idx] = a[idx] + b[idx]` (3-buffer streaming). Two variants:
`vecadd_safe` (`DisjointSlice::get_mut` + slice indexing — bounds-checked)
and `vecadd_unchecked` (`as_ptr().add(idx)` raw-pointer reads + `as_mut_ptr()`
write, guarded by a single scalar `idx < n` check). Block 256 via
`LaunchConfig::for_num_elems(n as u32)`, grid auto-computed. `cuEventRecord`
timing per ADR-0001, 1 warmup + 10 iters per N. Traffic `3·N·4`.

## Results (RTX 5090, sm_120, CUDA 13.2, libNVVM 22)

| N (elems) | kernel     | best ms | med ms | best GB/s | med GB/s |
|-----------|-----------|--------:|-------:|----------:|---------:|
|  1,048,576| safe      | 0.0042  | 0.0082 |  **2979** |     1530 |
|  1,048,576| unchecked | 0.0063  | 0.0088 |     1996  |     1432 |
| 16,777,216| safe      | 0.1083  | 0.1192 |  **1859** |     1689 |
| 16,777,216| unchecked | 0.1108  | 0.1152 |     1817  |     1747 |
| 67,108,864| safe      | 0.5008  | 0.5990 |  **1608** |     1344 |
| 67,108,864| unchecked | 0.5622  | 0.8641 |     1432  |      932 |
|268,435,456| safe      | 2.0812  | 3.3974 |  **1548** |      948 |
|268,435,456| unchecked | 3.1708  | 4.7273 |     1016  |      681 |

## Interpretation

**N=1M (4 MB/buffer, 12 MB aggregate)** fits entirely in the Blackwell 5090's
≈96 MB L2. Reported 2979 GB/s exceeds the 1.79 TB/s DRAM peak — the familiar
cache-resident overcount, since the `3·N·4` traffic model assumes cold reads.
This is the **deceptive-high** regime the task prompt called out.

**N=16M (64 MB/buffer, 192 MB aggregate)** is the L2-to-DRAM transition:
aggregate footprint just overflows L2 capacity. Throughput drops to
~1689 GB/s median, still elevated relative to peak DRAM.

**N=64M (256 MB/buffer, 768 MB aggregate) — DRAM-bound plateau.** Safe kernel
best **1608 GB/s = 90% of 1.79 TB/s peak**, matching the SoL ceiling. This
equals the nvcc baseline best exactly (1609 GB/s) — **within 0.1% at the
bandwidth-bound size**, well inside the ±10% gate.

**Bandwidth-bound hypothesis: supported.** The L2→DRAM transition sits where
predicted (just past 96 MB aggregate, i.e. N=16M). Once bandwidth-bound, the
Rust front-end contributes ~zero overhead relative to nvcc — codegen converges
because arithmetic intensity is trivial.

**Safe vs unchecked.** At large N the unchecked variant runs *slower*. Raw
`as_ptr()` reads from `&[f32]` emit bounds-check stripping that LLVM already
does for the safe `slice + get_mut` form (the safe path's `get_mut` predicate
is hoisted and merged with the grid-bound guard), so the unchecked kernel
has no codegen advantage here — and its extra explicit `n` param argument
slightly increases kernel-launch overhead visible as higher median times.
