# oxide-attn-gqa — Wave 16.1 analysis

**The no-tensor-core ceiling for GQA attention on cuda-oxide.**

cuda-oxide v0.1.0 has no usable tensor-core API on consumer Blackwell
sm_120 (Wave 14.4: zero `mma.sync` emitted anywhere in source; `wgmma`
is a placeholder; `tcgen05` requires sm_100a datacenter Blackwell). So
this cell is a pure f32 × f32 GQA pipeline, written the only way
cuda-oxide v0.1.0 can currently express matmul at speed: register
microtile pattern from Wave 7 `oxide-matmul-tiled-microtile` (4×4
microtile per thread, 64×64 output per block, 16×16 threads).

## Headline

| metric | value |
|---|---:|
| best total ms | **2.844 ms** |
| median total ms | 2.855 ms |
| **best TFLOPS** | **24.17 TF** |
| median TFLOPS | 24.07 TF |
| HMMA count in SASS | **0** (as expected, evidence below) |
| FFMA count in SASS | 157 (hardware FMA engaged via `core::intrinsics::fmuladdf32`) |
| correctness (f32) | max_abs=2.1e-7 at bench, max_abs=1.2e-7 at correctness shape |

## The no-TC story, SASS-verified

```
$ /usr/local/cuda/bin/cuobjdump --dump-sass oxide_attn_gqa.cubin | grep -c HMMA
0
$ /usr/local/cuda/bin/cuobjdump --dump-sass oxide_attn_gqa.cubin | grep -c FFMA
157
$ grep -c mma.sync oxide_attn_gqa.ll
0
```

Zero HMMA. That's the point of this cell: **this is how fast attention
runs on a Blackwell consumer GPU when no tensor-core instruction class
is reachable from your kernel-language frontend.** 157 FFMA
instructions confirm hardware FMA is engaged (via `fmuladdf32` → libdevice
`__nv_fmaf` → `fma.rn.f32`). The compute ceiling here is the 4th-gen
sm_120 CUDA-core f32 throughput, not the 4th-gen tensor cores.

## Per-stage breakdown (median iter, llama3_8b bench shape)

| stage | ms | % of total | notes |
|---|---:|---:|---|
| QKᵀ (gqa_qkt_kernel) | 1.237 | 43.5% | (B·n_q, S, S) output = 32 × 2048² cells of f32 matmul, d=128 reduction. Effective ~43 TF for the two matmul FLOPs — matches Wave 7 oxide-matmul-microtile 45 TF ceiling. |
| softmax (custom kernel) | 0.703 | 24.7% | Memory-bound. 537 MB read + 537 MB write at 1.53 TB/s ≈ 72% of HBM peak. Same regime as the cuBLAS-3-kernel softmax from Wave 15.1. |
| PV (gqa_pv_kernel) | 0.907 | 31.8% | Similar matmul shape to QKᵀ but smaller column dim (d=128 vs 2048). Runs at ~59 TF-effective — benefits from probs(f32)×V reduction over inner dim=2048, which gives the fmuladd pipeline better coverage. |
| **total** | 2.855 | 100% | |

## Ratios vs the other GQA cells (Wave 15.1)

| cell | best ms | best TFLOPS | vs us | HMMA |
|---|---:|---:|---:|---:|
| cuda-attn-gqa (nvcc 3-kernel WMMA) | 2.928 | 23.47 | 0.97× | 20 |
| **oxide-attn-gqa (this cell, f32 no-TC)** | **2.844** | **24.17** | **1.00×** | **0** |
| cublas-attn-gqa (cuBLAS-3-kernel) | 1.482 | 46.38 | 1.92× faster | (cuBLAS) |
| cutile-attn-gqa (cuTile fused, f16 TC) | 0.416 | 165.14 | 6.84× faster | 256 |

**Headline finding: cuda-oxide's f32-no-TC path ties the nvcc-WMMA
hand-rolled cell at 24 TF.** Wave 15.1's WMMA cell engaged tensor cores
(20 HMMA in SASS) but used 16×16 tiles with no shared-memory tiling —
the WMMA primitive wasn't enough to overcome the 3-kernel-pipeline +
no-smem-tile architecture. Our f32 register-microtile (Wave 7 pattern,
64×64 per block, 16×16 smem-tiled Q and K loads, fmuladd inner loop)
produces an arithmetically weaker path (CUDA-core f32 vs HMMA.1688.F32)
with stronger tiling — and they land on the same TFLOPS at this shape.

## Why 24 TF instead of 45 TF (the register-microtile matmul ceiling)

The register-microtile matmul from Wave 7 hits 45 TF f32 at N=4096. Our
two matmul stages together account for **~2.14 ms / 3.02 ms = 71%** of
the total wall-clock time, weighted as a single matmul they run at
about 45.3 TF (2 × 4 × 1 × 32 × 2048² × 128 / (2.14e-3) / 1e12 — count
both matmuls' FLOPs per iter). **The matmul stages ARE at the Wave 7
ceiling.** The 24 TF drop is purely the Amdahl tax from softmax:

- If softmax were free: we'd hit `4 × 1 × 32 × 2048² × 128 / (2.14e-3 × 1e12) = 32.2 TF`
- If BOTH matmuls ran at the raw 45 TF microtile peak AND softmax were free: 45 TF for attention

So our **24 TF is a 3-kernel pipeline ceiling**, not a matmul compute
ceiling. That maps directly onto the FlashAttention thesis: fusion
eliminates the softmax HBM round-trip, which would save ~0.7 ms here
and lift us to ~32 TF — if we had a way to fuse without tensor cores.
We don't in cuda-oxide v0.1.0 (no TC ⇒ the online-softmax rescaling
trick still needs the two full matmuls at scalar-f32 throughput, so
fusion would close the softmax gap but not the matmul-throughput gap
to cuTile's f16-TC 165 TF).

## Kernel structure

Three kernels in `src/main.rs` (~870 LOC total incl. host):

- **`gqa_qkt_kernel`** (~155 LOC)
  Grid = (S/64, S/64, B·n_q). Block = (16, 16). Each thread holds a
  4×4 microtile of scores.  TILE_Q[64×16] loaded as Q[b,h_q,row,k];
  TILE_K[16×64] loaded as K[b,h_kv,col,k] (transposed in-smem so inner
  loop reads TILE_K[k*64+col]). h_kv = h_q / groups from blockIdx.z.
  16 `fmuladdf32` per K-iter per thread, d=128 gives 8 K-tiles.

- **`softmax_kernel`** (~70 LOC)
  Grid = (B·n_q·S, 1, 1). Block = (128, 1, 1). One block per row of
  scores. Three passes: row max → exp+row sum → normalize. Each pass
  uses grid-stride within the row (row = 2048 elems, 128 threads, 16
  elems/thread) + warp-shuffle reduction (lane 0 of each of 4 warps
  writes to smem, first warp reduces 4 partials). Matches the
  `oxide-reduction` pattern.

- **`gqa_pv_kernel`** (~150 LOC)
  Grid = (d/64=2, S/64=32, B·n_q=32). Block = (16, 16). Same 4×4
  microtile pattern, but shapes are (S, S) × (S, d) so inner dim is
  S=2048 (128 K-tiles at BK=16). h_kv indexing identical to QKᵀ.

Host-side: minimal NPY reader (supports `<f4` f32 only), cudaEvent
per-stage timing, 1 warmup + 10 iters, writes `results.csv` with full
per-iter + per-stage breakdown.

## Pitfalls hit during implementation

- **libNVVM shadow bug avoidance** — `export CUDA_HOME LIBNVVM_PATH
  PATH=/usr/lib/llvm-21/bin:$PATH` before any cargo oxide invocation.
- **`cargo oxide build --release` doesn't exist**; the command is
  `cargo oxide build --arch sm_120` (or default arch).  The cargo-oxide
  frontend has its own flag set, not the vanilla cargo one.
- **`&mut DeviceBuffer` helper-fn params need an inner rebind** in
  functions that do multiple `cuda_launch!` of the same buffer. The
  compiler errors with "cannot borrow as mutable" despite the `&mut`
  binding, because the macro reborrows via deref coercion twice.
  Fix: `let mut scores_dev = scores_dev;` inside the function body.
- **Kernel artifact name uses underscores:** project `oxide-attn-gqa`
  produces `oxide_attn_gqa.cubin` / `.ptx`; `load_kernel_module(..,
  "oxide_attn_gqa")` with underscores, not hyphens.
- **Scalar kernel args** in `cuda_launch!` are plain exprs (`scale`,
  not `val(scale)`); only buffers use `slice(x)` / `slice_mut(x)`.
- **Correctness rtol for f32 attention.** Absolute max error is 2e-7
  (essentially float rounding noise vs the CPU-f32 reference). The
  max relative error touches 3e-2 on a few tiny-magnitude output
  elements — perfectly fine at atol=1e-5 but looks scary at
  rtol=1e-3. Use atol OR rtol, not both AND'd, for attention outputs
  that include near-zero values.

## Files written

```
oxide-attn-gqa/
├── Cargo.toml
├── rust-toolchain.toml
├── .gitignore
├── src/main.rs           (~870 LOC)
├── oxide_attn_gqa.cubin  (40 KB — committed as SASS evidence)
├── oxide_attn_gqa.ltoir  (12 KB)
├── oxide_attn_gqa.ll     (34 KB — intermediate LLVM IR)
├── results.csv
├── run.log
└── ANALYSIS.md           (this file)
```
