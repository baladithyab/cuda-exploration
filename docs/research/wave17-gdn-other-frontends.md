# Wave 17 — nvcc and cuBLAS feasibility for GDN-decode

**Scope.** Assess whether `nvcc-attn-gdn` (hand-written CUDA C++) and
`cublas-attn-gdn` (cuBLAS) can express the Gated DeltaNet single-timestep
decode recurrence competitively against the existing `cutile-attn-gdn`
baseline (610 GB/s = 34% of HBM peak, 0 HMMA, Qwen3-Next shape
B=1, H=16, d_k=d_v=256, BLOCK_V=64). The kernel is strongly memory-bound
(AI = 0.77 flops/byte, ridge ~25): per-iter HBM traffic ~8.2 KiB,
per-iter compute only 6.3 MFLOPS. The core ops are:

- `S_scaled = α · S_in`                 element-wise (d_k, BLOCK_V) f32
- `u = k · S_scaled`                    matvec (1, d_k) × (d_k, BLOCK_V)
- `S_out = S_scaled + β · k ⊗ (v − u)`  **rank-1 outer product** (d_k, BLOCK_V)
- `o = q · S_out`                       matvec (1, d_k) × (d_k, BLOCK_V)

`cutile-attn-gdn/ANALYSIS.md:78-90` already notes: "no large matmul-shaped
operation… cuTile correctly emits scalar FMA." This doc asks whether the
other two frontends can do the same thing at least as well.

## 1. Evidence from existing CUDA GDN-decode implementations

- **FlashInfer** (`flashinfer-ai/flashinfer`) ships a hand-written CUDA C++
  GDN-decode kernel (`gdn_decode_bf16state_mtp_kernel`, ~600 LOC of .cu
  core, plus Python wrapper). It uses **plain scalar FMA** for the rank-1
  state update, dispatching to `fma_packed_f32x2` on Blackwell (SM100+).
  No cuBLAS. No WMMA for the recurrence. This is the closest production
  reference to our target and directly validates the approach.
- **vLLM** uses a Triton kernel for GDN decode (`fused_recurrent_gated_delta_rule_packed_decode`),
  scalar FMA semantics. No hand-written CUDA for this path.
- **fla-org/flash-linear-attention** is Triton-only for GDN decode.
- **NVIDIA CUDA samples** ship nothing GDN-shaped; the closest analogue is
  the `transpose` / `matrixMul` stateless pattern, not useful here.

Takeaway: the field's answer to "how do you write GDN-decode by hand?" is
**scalar FMA with careful vectorized I/O** — exactly what we'd write in nvcc.

## 2. nvcc-attn-gdn feasibility

**Verdict: worth writing, expected ~500–700 GB/s (competitive with cuTile's
610 GB/s), zero HMMA, plain FFMA.**

### LOC estimate

Anchored on two existing cells in this repo:

| reference cell | kernel LOC | total .cu LOC |
|---|---:|---:|
| `cuda-attn-gqa/attn_gqa.cu` (3-kernel WMMA GQA) | ~210 (kernels) | 493 |
| `cublas-attn-gqa/attn_gqa.cu` (cuBLAS + softmax) | ~180 | 459 |
| FlashInfer GDN-decode (external) | ~600 (kernel only, incl. comments) | n/a |

Our kernel is simpler than FlashInfer's (no MTP, no bf16state, single
timestep, fixed shapes). Estimate:

- GDN-decode kernel body: **~120–160 LOC** (load q/k/v/α/β + state tile,
  three inner loops for matvec/outer/matvec, vectorized store).
- NPY loader + `main()` harness + cudaEvent timing + smoke/bench CSV: **~330 LOC**
  (reuse verbatim from `cuda-attn-gqa/attn_gqa.cu:47-200, 380-493`).
- Total: **~450–500 LOC** single `nvcc_attn_gdn.cu` file — smaller than
  `cuda-attn-gqa` because no WMMA fragments, no softmax, no multi-kernel
  dispatch.

### Expected performance

The kernel is HBM-bound; what matters is load/store throughput, not FFMA
quality. Compared to `cutile-attn-gdn`:

1. **Vectorized 128-bit loads for the H tile.** `cutile-attn-gdn/ANALYSIS.md:152-156`
   flags this as the single biggest BW gap: cuTile v1.3.0 emits scalar
   f32 loads instead of `LDG.E.128`. In nvcc we can directly spell
   `float4 h_vec = *reinterpret_cast<const float4*>(...);` and get the
   wide transaction. This alone is expected to close most of the gap from
   34% → 55–70% of HBM (roughly 990–1250 GB/s theoretical; likely
   500–800 GB/s achieved after accounting for the per-iter launch
   overhead that already bounds cuTile).
2. **Manual register tiling.** One block per (batch·head), one thread per
   output column, keep the full (d_k=256) q/k vectors broadcast via shared
   memory, accumulate state column-tile in registers.
3. **No async prefetch in the baseline** (fine — single-block launch per
   head, no cross-iter pipelining exists in the kernel boundary).

**Estimated ceiling: 500–700 GB/s best-case**, landing at 80–115% of
cuTile's 610 GB/s. The 610 GB/s number is already 34% of HBM; going above
it requires vectorized I/O which nvcc gives trivially but is capped by
the per-launch fixed cost (~14 µs/iter includes kernel-launch latency,
not just data movement — see ANALYSIS.md:39-42). A **realistic target
band is 550–650 GB/s**, i.e. "approximately matches cuTile, may modestly
exceed it via vectorized loads."

No HMMA expected. No WMMA is applicable: the matvecs are (1, 256)×(256, 64)
— the M=1 dim makes any 16×16×16 tile waste ≥94% of the fragment. This
mirrors `cuda-attn-gqa/ANALYSIS.md:96-110` where WMMA on thin tiles hit
only 10.8% of cuBLAS peak; for GDN-decode WMMA is strictly worse than
scalar FFMA.

## 3. cublas-attn-gdn feasibility

**Verdict: SKIP. cuBLAS lacks the right primitives; a cuBLAS implementation
would be strictly slower than both nvcc and cuTile, and would need a
custom CUDA kernel for the element-wise gated update anyway — defeating
the point of a "cuBLAS baseline."**

### Why every GDN-decode op is hostile to cuBLAS

| GDN op | as a GEMM | M×N×K | why cuBLAS loses |
|---|---|---|---|
| `u = k · S_scaled` | `cublasGemmEx` | 1×64×256 | M=1: one warp-wide row. cuBLAS tiles are 128×128+; gets 1–3% of peak. |
| `k ⊗ (v−u)` (outer product) | `cublasGemmEx` | 256×64×**1** | K=1: zero accumulator reuse. Fundamentally serial dot-product of length 1. cuBLAS has no optimized path; `cublasSger` (f32-only, rank-1 update) exists but only for d_k×d_v=256×64=16K element update and would be launch-overhead dominated. |
| `o = q · S_out` | `cublasGemmEx` | 1×64×256 | same as matvec above |
| `α · S + β · k⊗r` | (element-wise) | — | **cuBLAS has no primitive.** Would need a hand-written CUDA kernel. |

### Launch-count arithmetic

A cuBLAS decomposition per-iter per-head:
1. element-wise `α · S_in` → custom kernel OR `cublasSscal` (f32 scalar
   multiply, 256×64 = 16K elements) — launch #1.
2. `k · (α·S_in)` → `cublasGemmEx` M=1 N=64 K=256 — launch #2.
3. element-wise `v − u` → custom kernel — launch #3.
4. rank-1 `β · k ⊗ r` → `cublasSger` or `cublasGemmEx` K=1 — launch #4.
5. element-wise `S_scaled + β·outer` → custom kernel — launch #5.
6. `q · S_out` → `cublasGemmEx` M=1 N=64 K=256 — launch #6.

Per iter per (batch, head): **6 kernel launches**. With B=1, H=16 and no
batching across heads (different S_in per head) that's potentially 96
launches per decode step. Even at 2 µs/launch that's ~200 µs vs
cuTile's 13.8 µs — a **~15× slowdown**. Batched variants (`cublasGemmStridedBatchedEx`)
could fold the head dim together, cutting launches to ~6 total, but the
underlying GEMMs are still M=1 or K=1 which cuBLAS is not tuned for.

### The HBM-traffic argument

Even without launch overhead, each cuBLAS stage writes its intermediate
to HBM and the next stage re-reads it. The **entire point** of the cuTile
fused kernel (and the planned nvcc kernel) is to keep the (d_k, BLOCK_V)
= 64 KB state tile in registers across all four ops. A cuBLAS pipeline
writes and rereads this tile ~4 times, multiplying HBM traffic 4×. For a
memory-bound kernel that is directly ~4× slower.

### What a cuBLAS baseline would actually measure

It would measure "cuBLAS is the wrong tool for this kernel class" — not a
useful comparative data point for Wave 17. The existing cells already
establish cuBLAS's strengths (cublas-half-precision: 218 TF hgemm;
cublas-attn-gqa: 46.4 TF fused attention). Writing `cublas-attn-gdn`
would reach maybe 5–15 GB/s and document only its own unsuitability.

### Recommendation

**Document `cublas-attn-gdn` as "not a meaningful baseline: cuBLAS lacks
primitives for rank-1 recurrent state updates"** and skip the cell.
Replace the slot in Wave 17.5b with a short note in `results/wave17-summary.md`
citing this doc. Budget saved: ~1 day of cell + harness work.

## 4. Reference patterns to crib from

- **FlashInfer `gdn_decode_bf16state_mtp_kernel`** (flashinfer-ai/flashinfer,
  `flashinfer/gdn_kernels/gdn_decode_bf16_state.py` + CuTe-DSL body):
  the canonical scalar-FMA GDN decode; mirrors the structure we want.
  Uses `fma_packed_f32x2` on Blackwell as a micro-optimisation — we can
  adopt this via `__builtin_amdgcn_fma_f32` / `fma2.f32` PTX inline if
  needed but it's a 2% nicety, not first-order.
- **`cuda-attn-gqa/attn_gqa.cu:47-200`** — NPY loader, `CK(...)` macro,
  cudaEvent timing scaffold. Copy verbatim.
- **FLA-org `fused_recurrent_gated_delta_rule`** Triton kernel — semantic
  reference for the recurrence; correctness cross-check only.
- **`cutile-attn-gdn/main.py:100-170`** — our own cuTile kernel body;
  nvcc version is a direct transliteration with `float4` vector loads
  substituted for `ct.load`.

## 5. Summary

| cell | verdict | LOC | expected GB/s | % of cuTile |
|---|---|---:|---:|---:|
| **nvcc-attn-gdn** | **build it** | ~450–500 | 550–650 | 90–107% |
| **cublas-attn-gdn** | **skip, document** | — | ~5–15 | <3% |

The cuTile baseline is already strong because the kernel is memory-bound
and cuTile's scalar FMA + HBM access pattern is close to what a hand-tuned
nvcc would emit. nvcc's one structural win is vectorized 128-bit loads
(`LDG.E.128`) which cuTile v1.3.0 doesn't expose. cuBLAS has no path to
competitive performance on this kernel class and should be skipped with a
pointer to this document.
