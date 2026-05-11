# cutile-attn-gdn — Wave 16.4 ANALYSIS

cuTile fused **GDN-decode** (Gated DeltaNet single-timestep decode kernel,
the linear-attention mechanism used in **Qwen3-Next**). Tests cuTile's
expressiveness for a **fundamentally different regime** than GQA/MLA:
no softmax, no exp, stateful recurrence, strongly memory-bound.

## Headline @ Qwen3-Next decode shape (B=1, H=16, d_k=d_v=256)

| metric | value |
|---|---:|
| **best decode time (cudaEvent)** | **13.79 µs** |
| median time (50 iters) | 15.39 µs |
| **best effective bandwidth** | **610.6 GB/s** (34.1% of RTX 5090 HBM peak ~1792 GB/s) |
| median bandwidth | 547.1 GB/s |
| TFLOPS (academic — for the recurrent state ops) | 0.456 |
| HMMA count in cubin | **0** |
| FFMA count in cubin | (substantial — outer-product updates) |
| State H size | d_k × d_v × n_heads × 4 bytes = 256·256·16·4 = 4 MiB |
| Per-iter HBM traffic | 8.2 KiB (H_in load + H_out store + Q/K/V/α/g/O) |
| Correctness vs PyTorch reference | OK (max_abs 3.1e-5 on output, 3e-8 on state) |

## Why GB/s is the headline metric (not TFLOPS)

GDN decode is **strongly memory-bandwidth-bound**:

  - Compute per iter: 6.3 MFLOPS (one outer product `(k ⊗ v)` of size d_k·d_v
    plus a state update plus `q ⊗ H` matvec)
  - Memory per iter: 8.2 KiB (load H_in 256KB, store H_out 256KB, plus Q/K/V/α/g/O)
  - **Arithmetic intensity: 0.77 flops/byte** — vastly below the RTX 5090 ridge
    of ~62 flops/byte for f16 TC. Even f32 CUDA cores have AI ridge ~25 flops/byte.

So compute throughput is irrelevant; the kernel's job is to move data fast.
**610 GB/s = 34% of HBM peak** is a respectable result for a Python tile
DSL kernel that has no manual cache hints / stream-load tweaks.

For comparison, the cuTile vec-add kernel reaches ~1559 GB/s (87% of HBM)
on a much simpler streaming pattern. GDN decode's lower BW efficiency is
plausibly due to: (a) the H tile load/store has a strided access pattern
(d_v = 256 stride per row) that doesn't coalesce as neatly as vec-add's
unit-stride; (b) the per-iter work is so tiny (6 MFLOPS) that launch
overhead and synchronization costs eat a measurable fraction.

## Kernel structure (per (batch, head) tile)

```
@ct.kernel
def gdn_decode(q, k, v, alpha, g, h_in, o, h_out):
    bid_h = ct.bid(0)        # (batch * n_heads) index
    bid_v = ct.bid(1)        # tile index along d_v (BLOCK_V=64, so 4 tiles for d_v=256)

    # Load per-head scalars and the input vectors
    q_vec = ct.load(q, ...)          # (d_k,)        f16
    k_vec = ct.load(k, ...)          # (d_k,)        f16
    v_tile = ct.load(v, ...)         # (BLOCK_V,)    f16   — sliced by bid_v
    α      = ct.load(alpha, ...)     # scalar        f16
    g      = ct.load(g, ...)         # scalar        f16

    # Load this column-tile of the state H (shape (d_k, BLOCK_V) f32)
    h_tile = ct.load(h_in, (bid_h, 0, bid_v * BLOCK_V), (d_k, BLOCK_V))  # f32

    # Compute the output contribution for this tile:
    #   o_tile = q^T @ h_tile      (1, BLOCK_V) f32
    o_tile = ct.matmul(q_vec[:, None].T, h_tile)  # uses ct.matmul or hand FMA loop

    # Compute the state update:
    #   h_new = g * h_tile - α * (k * h_tile) * k^T + (k ⊗ v_tile)
    # All in f32 in registers.
    ...

    ct.store(o,     ..., o_tile.astype(ct.float16))
    ct.store(h_out, ..., h_new)
```

Single fused kernel: state update + output projection in one launch. No
intermediate HBM traffic for the (d_k, BLOCK_V) tile of H.

## Why no HMMA?

GDN decode at d_k = d_v = 256 has **no large matmul-shaped operation**:

  - `q^T @ h_tile` is (1, d_k) × (d_k, BLOCK_V=64) → (1, 64) — too thin to
    benefit from tensor cores. cuTile correctly emits scalar FMA.
  - `(k ⊗ v_tile)` is an outer product of vectors → also rank-1 update.
  - State update `g * H` is element-wise scalar multiply.

This is **expected** for GDN-decode and not a cuTile bug. The PREFILL
version of GDN (chunkwise) does have larger matmuls and would benefit from
TC; but prefill is 1000+ LOC for the WY/triangular-solve pieces, deferred
to a future wave.

## What this tells us about cuTile expressiveness

cuTile **expresses GDN-decode cleanly** in ~485 LOC. The kernel uses:

  - `ct.bid(0)`, `ct.bid(1)` for batch/head and d_v-tile partitioning
  - `ct.load` / `ct.store` for arbitrary-rank tiles via `tiled_view`
  - `ct.matmul` (but here scalar-FMA-equivalent due to thin shapes)
  - Element-wise broadcast operations for the gated update
  - f32 register accumulation with f16 I/O

No primitive fell over. No "this isn't supported in v1.3.0" gaps. This is
the strongest data point so far that **cuTile generalizes beyond
GQA/matmul-shaped attention** to the linear-attention recurrence regime.

## Comparison to other Wave 15-16 cells

| cell | regime | best | HMMA | metric |
|---|---|---:|---:|---|
| cutile-attn-gqa | fused softmax attn | 165.1 TF | 256 | TFLOPS |
| cutile-attn-mla | fused softmax attn | 112.4 TF | 384 | TFLOPS |
| cutile-attn-gdn (this) | recurrent-state decode | 610 GB/s | 0 | GB/s |
| oxide-attn-gqa | f32 no-TC attn | 24.2 TF | 0 | TFLOPS |
| cublas-attn-gqa | 3-kernel | 46.4 TF | (cuBLAS) | TFLOPS |
| cuda-attn-gqa | nvcc 3-kernel WMMA | 23.5 TF | 20 | TFLOPS |

All four cuTile cells match their respective regimes' expectations. The
"cuTile reaches 76% of cuBLAS hgemm peak on fused attention" headline
from Wave 15.1 doesn't apply here because GDN isn't compute-bound; the
analogous metric for memory-bound work is "% of HBM peak" and 34% is
the GDN-decode result.

## Pitfalls hit

1. **fla-org/flash-linear-attention not in cutile venv by default.** Not
   installed for this cell; PyTorch reference hand-wrote the recurrence
   instead. Adds ~20 LOC but eliminates an external dep.

2. **Tiny per-iter work needs many bench iters.** With 14 µs per iter,
   typical 10-iter cudaEvent timing has high variance (cv 5-25%). Used
   50 timed iters + median with IQR for stability.

3. **Closure-pattern factory needed for BLOCK_V** — same Wave 13.1 pitfall:
   `ct.Constant[int]` launch arg fails to propagate to tiled_view shape.

4. **JIT cost is 600+ ms first launch** but irrelevant given we do 50+ iters.

5. **State H is 4 MiB** (f32, d_k × d_v × n_heads). Doesn't fit in shared
   memory; every kernel reads/writes it from HBM. This is the BW-bound
   nature of decode; PREFILL keeps H in registers across timesteps within
   a chunk, which is why prefill is more compute-bound.

## Files

- `main.py` (~485 LOC) — kernel factory + smoke + bench + cubin export
- `run.sh` — convenience wrapper
- `gdn_decode_fused.cubin` (gitignored) + `gdn_decode_fused.sass` (tracked, evidence)
- `results.csv` — per-iter timings (50 iters)
- `run_smoke.log`, `run_bench.log`
- `.gitignore` (added)

## What would be needed to close the BW gap (34% → ~80%)

1. **Vectorized loads** for the H tile — currently scalar f32 loads; LDG.E.128
   would 4× the load throughput. Not exposed by cuTile v1.3.0; would be an
   upstream feature request.
2. **Async HBM prefetch** during compute — overlap the state load of the
   NEXT timestep with the current timestep's update. cuTile's TMA primitives
   (the `UTMALDG.1D` instructions seen in cuTile reduction) could plausibly
   help here but would need an explicit double-buffered structure.
3. **Larger BLOCK_V** (128 or 256) to amortize launch overhead. At BLOCK_V=64
   we have 4 sequential tiles per head; a single-block kernel would cut
   launches by 4×. Worth trying in a Wave 16.5 follow-up.

## Wave 16.4 done; what's next

- **W16.5 GDN-decode @ BLOCK_V=128** — try to push BW above 800 GB/s.
- **W16.6 KDA-decode** — vector-gate variant of GDN. ~30-LOC delta over this kernel; near-free given GDN-decode is built.
- **W16.7 GDN-prefill chunkwise** — the high-LOC version with WY/triangular-solve.
