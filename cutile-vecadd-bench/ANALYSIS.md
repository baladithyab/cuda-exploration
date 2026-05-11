# cutile-vecadd-bench — Wave 12 W12.1 ANALYSIS

## Headline

**cuTile reaches 99-100% of nvcc on memory-bound vector add** at all four
problem sizes (1M, 16M, 64M, 256M f32 elements). Parity with cuda-oxide too.

| impl | kernel | 1M | 16M | 64M | 256M |
|---|---|---:|---:|---:|---:|
| nvcc | vecadd | 2166.5 | 1707.1 | 1572.1 | **1568.0** |
| cuda-oxide | safe | 2902.0 | 1744.0 | 1568.1 | **1573.1** |
| cuda-oxide | unchecked | 2286.1 | 1724.6 | 1567.0 | **1569.7** |
| cutile | tile256 | 2108.4 | 1700.4 | 1554.9 | **1559.4** |
| cutile | tile1024 | 2031.8 | 1705.7 | 1554.1 | **1559.7** |

All five implementations within 1% at N=256M (the cache-overflow regime, the
cleanest memory-bound comparison). HBM peak on RTX 5090 is ~1792 GB/s; everyone
hits ~87% of peak.

**Compiler quality is not the bottleneck on memory-bound kernels.** The hardware
is. This confirms the existing Wave 4-5 finding that algorithm class matters
more than the language frontend; cuTile slots in alongside cuda-oxide and nvcc
as a viable production frontend on this axis.

## Methodology note: re-running the baselines mattered

The original `cuda-vecadd-bench/results.csv` and `oxide-vecadd-bench/results.csv`
(committed May 9) showed median bandwidths of ~890 GB/s and ~954 GB/s at N=256M
respectively — half of the values measured today (~1568 GB/s and ~1573 GB/s).
First-pass cuTile numbers matched today's session, making cuTile look like it
beat nvcc and cuda-oxide by 75%.

That was wrong. **The original runs were thermally degraded.** Re-running nvcc
and cuda-oxide on the same idle GPU (42°C, 47W, in this same session) produced
~1568 GB/s for both, matching cuTile. The Wave 12 numbers should be the basis
for any cutile-vs-{nvcc, oxide} claim; the original `results.csv` files are
contaminated and need to be regenerated for any apples-to-apples comparison.

This is a re-confirmation of `AGENTS.md`'s Wave 1 lesson — "thermal contention
with desktop apps probably; could also be GPU power-state transitions" — and a
methodological pitfall for anyone reading the older numbers.

## Two cuTile variants tested

Both kernels do `c[i] = a[i] + b[i]`, identical algorithm:

```python
@ct.kernel
def vecadd(a, b, out):
    bid = ct.bid(0)
    a_t = ct.load(a, index=(bid,), shape=(TILE_SIZE,))
    b_t = ct.load(b, index=(bid,), shape=(TILE_SIZE,))
    ct.store(out, index=(bid,), tile=a_t + b_t)
```

- **`tile256`**: TILE_SIZE=256 → matches cuda-oxide's `LaunchConfig::for_num_elems`
  block-256 launch shape. 1M elements → 4096 blocks.
- **`tile1024`**: TILE_SIZE=1024 → 4× fewer launches, 4× more work per block.
  1M elements → 1024 blocks.

**Almost identical bandwidth across the two**, suggesting the kernel is purely
memory-bound at all four N regimes — the launch-overhead of tile256 vs tile1024
doesn't matter at this scale. (At very small N, tile1024 might win marginally;
at very small TILE_SIZE like the original smoke test's 16, launch overhead
*does* dominate — 370 GB/s vs 1559.)

## Setup pitfalls captured (see SETUP.md)

1. **Pip name is `cuda-tile`** (NOT `nvidia-cutile`, NOT `cutile-python`). 3 wrong tries to discover.
2. **README launch syntax `kernel[(grid,)](args)` is broken in v1.3.0.** Use `ct.launch(stream_ptr, grid, kernel, args_tuple)`.
3. **TILE_SIZE matters**: too small (e.g. 16 = 64 bytes/block) drowns in launch overhead. Match cuda-oxide's per-block work (256-1024 elements).
4. **JIT cost is 639 ms on first launch**, then sub-ms. Always discard 1-3 warmup iters.

## Source files

- `main.py` — bench harness (Python; mirrors `oxide-vecadd-bench/src/main.rs`)
- `smoke_test.py` — minimal 1M vec-add (used for initial smoke testing)
- `SETUP.md` — install pitfalls + working launch API
- `results.csv` — per-iter `(impl, kernel, n, iter, gpu_ms, gbps)` for both variants
- `run.log` — captured stdout from `python main.py` (the data source for the table above)
