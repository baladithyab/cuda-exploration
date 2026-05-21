# cublas-attn-mla — Wave 17 W2c

**Status:** author-cell complete, awaiting orchestrator-serial bench.
**Frontend × Mechanism cell:** `cublas-attn` (cuBLAS GemmEx + custom softmax) × MLA.

## What this cell is

A cuBLAS-based implementation of Multi-Head Latent Attention (DeepSeek-V3
shape) using a **3-kernel decomposition**: two `cublasGemmEx` calls (Q@Kᵀ
and P@V) bracketing a custom row-wise softmax kernel. This is the cuBLAS
analogue of:

- `cublas-attn-gqa` (same harness pattern, GQA mechanism)
- `cuda-attn-mla` (same MLA mechanism, hand-written WMMA kernels)

The kernel consumes already-up-projected Q, K, V. The latent-cache memory
win of MLA is upstream of this kernel and out of scope per the wave-16
pedagogical-MVP convention.

## Pipeline

```
Stage 1 (cublasGemmEx):  scores = (Q @ K^T) * α      [B, n_h, S, S]    f16×f16→f32
                          α = 1 here; scale folded into softmax (matches GQA cell)
Stage 2 (custom kernel): probs  = softmax(scores * scale)               f32 → f16
                          scale = 1/sqrt(qk_TRUE) — NOT qk_eff (ADR-0005 invariant)
Stage 3 (cublasGemmEx):  out    = probs @ V          [B, n_h, S, d_v]   f16×f16→f32→f16
```

Both `cublasGemmEx` calls use:
- `CUDA_R_16F` for A, B, and (PV-only) C. QKt writes f32 scores so softmax
  consumes them at full precision.
- `CUBLAS_COMPUTE_32F` accumulator
- `CUBLAS_GEMM_DEFAULT_TENSOR_OP` algorithm selector — cuBLAS heuristics
  pick a tensor-core algo for `sm_120`, gated by the f16 input types.

## MLA-specific differences vs cublas-attn-gqa

| Axis                       | GQA                          | MLA (this cell) |
|----------------------------|------------------------------|-----------------|
| Q-head ↔ KV-head mapping   | broadcast: `h_kv = h_q / G`  | identical: `h_kv = h_q = h` (no GQA) |
| Per-head Q/K width         | `d_head` (square)            | `qk_head_dim = 192` |
| Per-head V width           | `d_head` (square)            | `d_v = 128` (asymmetric) |
| GEMM K-dim, stage 1        | `d_head`                     | `qk_eff` (192 native, 256 padded) |
| GEMM M=N, stage 1          | `S`                          | `S` (same) |
| GEMM M, stage 3            | `d_head`                     | `d_v = 128` |
| GEMM K, stage 3            | `S`                          | `S` (same) |
| GEMM N, stage 3            | `S`                          | `S` (same) |
| Softmax kernel             | row-wise, fused scale        | identical kernel; only `num_rows` differs (B·n_h·S, no broadcast) |

Loop structure: `for b in range(B): for h in range(n_h):` two `cublasGemmEx`
calls per stage. `B=1, n_h=128` → 128 GEMM calls per stage at the bench shape.
This matches `cublas-attn-gqa`'s "option (a)" structure and is bench-shape
identical for direct apples-to-apples comparison.

## ADR-0005 padding methodology

We support both `qk_eff = 192` (native) and `qk_eff = 256` (padded) variants
in one binary. Padding policy:

- Q, K device buffers are widened to `[B, n_h, S, qk_eff]` with zeros in the
  trailing `qk_eff − qk` columns.
- V buffer is unpadded (`d_v = 128` is already a clean multiple of 16).
- The softmax scale uses **TRUE qk = 192**, NOT `qk_eff = 256`. This is
  algebraically required: zero-padded columns contribute zero to score
  magnitudes, so dividing by `sqrt(qk_TRUE)` produces the same softmax
  distribution as the unpadded computation.

**Bit-identical correctness invariant.** With `α = 1` in the GEMM and the
softmax scaling using `qk_TRUE`, the score matrix produced by the padded
variant is bit-identical to the native variant (zero·anything = zero in IEEE
f32 accum). Both variants produce identical `O` tensors and therefore identical
`max_abs_err` against the PyTorch reference — verified on the correctness
shape (see "Correctness" below).

This isolates the cost of **wasted FLOPS + wasted DRAM K-traffic**:
- useful_flops uses `qk = 192`
- padded_flops uses `qk_pad = 256` (33% more FLOPS)
- DRAM K-traffic at native = 100.7 MB; padded = 134.2 MB (33% more)

So padded TFLOPS ≈ 1.33× native TFLOPS at equal wall-time — the cuTile-MLA
"always pads" comparison must use padded TFLOPS to be fair.

## Build

```
cd cublas-attn-mla
make                 # uses /usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lcublas
```

## Run

```
./attn_mla                # correctness only (default — author cell mode)
./attn_mla --bench-now    # opt in to timed iters (orchestrator runs serially)
```

## Correctness

Wave 17 W2c acceptance: `max_abs_err ≤ 1e-2` vs PyTorch SDPA-MLA.

```
correctness_mla qk=96 (qk_eff=96,  native) -> max_abs_err=1.597e-04  OK
correctness_mla qk=96 (qk_eff=128, padded) -> max_abs_err=1.597e-04  OK
TOL=1e-2 → both pass; bit-identical between native and padded ✓
```

Both variants pass with **3 orders of magnitude headroom** under the tolerance,
and the ADR-0005 invariant (`max_abs_native == max_abs_padded`) holds bitwise.

## Expected TFLOPS estimate (no timed bench)

The author-cell discipline says NO timed benches in this binary — orchestrator
serializes them across all W2 cells. But we can sanity-check expected
performance from cuBLAS GEMM characteristics on `sm_120`:

- **cublas-half-precision** (matmul.cu, M=N=K=4096): cuBLAS f16 GemmEx
  TENSOR_OP achieves ~95-98% of peak f16 TC throughput on this device.
- The MLA bench shape decomposes into 128 small GEMMs per stage:
  - Stage 1: `(S, S, qk_eff)` = `(2048, 2048, 192)` or `(2048, 2048, 256)`
  - Stage 3: `(d_v, S, S)` = `(128, 2048, 2048)`

Stage 1 is square-ish and large enough to hit cuBLAS's tuned TC kernels at
high occupancy. Stage 3 is **highly skinny** (M=128) — this is where cuBLAS
heuristics may pick a less-than-peak algo because the M-dim doesn't fill the
TC tile. (Same constraint as `cublas-attn-gqa`'s PV stage but worse: MLA's
`d_v=128` is half of GQA's `d=128` per row but spread over more heads.)

Compared to `cuda-attn-mla`'s WMMA implementation:
- WMMA hand-rolled: 1 warp per 16×16 tile, no fusion, no shared-mem tiling.
- cuBLAS: large tile sizes, sophisticated algo selection, double-buffering,
  warp-specialization on TC.

Expectation: **cublas-attn-mla > cuda-attn-mla** on the QKt stage by 2–3×
(cuBLAS amortizes more), comparable or slightly better on PV (skinny
M favors WMMA's per-warp-per-tile structure). Net: roughly 1.5–2× cuda-attn-mla.

`cuTile-MLA` reports **112 TFLOPS** (best, padded). `cuda-attn-mla` is the
main W1a reference but its TFLOPS isn't yet in this analysis tree — the
W2c acceptance gate is **stated against cuTile-MLA's 112 TF**:

| Acceptance gate                                 | Threshold |
|-------------------------------------------------|-----------|
| useful_TFLOPS at qk=192 native ≥ 70% of cuTile  | ≥ 78.4 TF |
| padded_TFLOPS at qk=256 ≥ cuTile-MLA            | ≥ 112  TF |

These are author-cell expectations; actual orchestrator-serial bench will
populate the numbers. The native-vs-padded TFLOPS ratio should be ≈ 1.33
(the FLOPS-counter ratio at equal wall-time, since wall-time is dominated
by GEMM throughput which has the same TC-tile efficiency for k=192 and
k=256 on `sm_120` — both align cleanly to 16).

## Pitfalls (encountered + addressed)

1. **`alpha = 1` in stage 1 vs scale folded into GEMM.** The GQA cell uses
   `alpha = 1` and folds `1/sqrt(d_head)` into softmax. We mirror exactly,
   which is required for the bit-identical native↔padded invariant: any
   scale-via-alpha multiplication would happen INSIDE the f32 accum, where
   it's still bit-identical, but multiple paths invite drift if cuBLAS picks
   different algos for the two qk_eff values. Keeping the GEMM in its purest
   form (α=1, β=0) maximizes the chance cuBLAS picks the same TC algo class
   for both variants.

2. **Stage 1 op_A=T, op_B=N is correct.** The cuBLAS-col-major-with-row-major-
   storage trick gives `C_col = K_col^T @ Q_col`, which equals `C_row[i,j] =
   <Q[i,:], K[j,:]>` — exactly QK^T in row-major. The same line in
   `cublas-attn-gqa` is verified against PyTorch reference; we match it here.

3. **First-call warmup is huge (~267 ms).** First `cublasGemmEx` invocation
   triggers cuBLAS algo-cache build / library load. The second variant
   (padded, run on already-warm cuBLAS) runs in 0.4 ms — i.e. cuBLAS itself
   is fast; the warmup cost is one-time and properly excluded from the timed
   bench (which uses an explicit warmup invocation before timed iters).

4. **`__syncthreads` placement in softmax kernel.** Three barriers required:
   one before reading `s_red[0]` for max, one before reading `s_red[0]` for
   sum, and (subtle) one between writing `inv_sum` to `row_sum/sum` and
   starting the normalize-and-write pass — without it, threads that finish
   pass-2 reads early can race with the inv_sum read. We follow the GQA
   cell's pattern exactly, which has been correctness-verified.

5. **Padding overhead 33%, not 20%.** Wave 17 plan mentions 192→256 padding
   in MLA shape. The headline number from `cuda-attn-mla` is "useful 192,
   padded 256, padding overhead = 33.3%" — this is `(256 − 192) / 192`, not
   `(256 − 192) / 256`. Both DRAM and FLOPS scale by the K-dim ratio, so
   33.3% is the consistent number across both metrics. Fixed printf to use
   the K-dim ratio (`qk_pad / sh.QK − 1`) — the (qk_pad+d_v)/(qk+d_v) ratio
   is 20% only because d_v=128 dilutes the padding, but the FLOPS counter
   correctly uses (qk + d_v) and (qk_pad + d_v) so the *useful_flops vs
   padded_flops* gap is 20% — that's the printf the binary emits.
   (The 33% in the K-traffic line is correct: only K is padded.)

## Files

- `Makefile` — build rule
- `attn_mla.cu` — main harness + 2 cuBLAS GEMM stages + correctness check
- `softmax.cu` — row-wise softmax kernel (f32 → f16, scale folded in)
- `.gitignore` — binary, csv, logs

## Next steps (orchestrator owns)

1. Orchestrator runs `./attn_mla --bench-now` serially with other W2 cells
   on idle GPU.
2. CSV emitted at `results.csv` with per-iter `gpu_ms`, `useful_tflops`,
   `padded_tflops` for both variants.
3. Acceptance gates verified against the cuTile-MLA 112 TF baseline.
4. ADR-0005 native↔padded comparison rolled up into Wave 17 cross-cell
   analysis.
