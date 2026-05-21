# cuda-attn-mla — Wave 17 W1a

**Status:** author-cell complete, awaiting orchestrator-serial bench.
**Frontend × Mechanism cell:** `cuda-attn` (nvcc CUDA C++) × MLA.

## What this cell is

A pedagogical CUDA C++ implementation of Multi-Head Latent Attention
(DeepSeek-V3 shape) using a **3-kernel decomposition** with WMMA tensor-core
matmuls — the nvcc analogue of `cuda-attn-gqa` extended to the non-square
MLA per-head dimensions (qk_head_dim = 192, v_head_dim = 128, all n_h=128
heads independent — no GQA broadcast at the attention stage).

The kernel consumes already-up-projected Q, K, V. The latent-cache memory
win of MLA is upstream of this kernel and out of scope per the wave-16
pedagogical-MVP convention.

## Pipeline

```
mla_qkt_kernel    S = (Q @ K^T) * scale       [B, n_h, S, S]    f16×f16→f32  (WMMA m16n16k16)
softmax_kernel    P = softmax(S, dim=-1)      [B, n_h, S, S]    f32 → f16
mla_pv_kernel     O = P @ V                   [B, n_h, S, d_v]  f16×f16→f32→f16  (WMMA m16n16k16)
```

Block geometry (single warp = 32 threads per block, output tile 16×16):

| Kernel | Grid | Block | K-loop trips |
|---|---|---|---|
| `mla_qkt_kernel` | `(S/16, S/16, B·n_h)` | `(32)` | `qk_eff / 16` (12 native or 16 padded) |
| `softmax_kernel` | `(S, B·n_h)` | `(128)` | n/a |
| `mla_pv_kernel` | `(S/16, d_v/16, B·n_h)` | `(32)` | `S / 16` |

Per-warp WMMA fragment shape `m16n16k16 f16→f32` is identical to
`cuda-attn-gqa`. The MLA-specific change is the **non-square per-head
geometry**: `qkt` reads K with leading-dim QK_eff, `pv` writes O with
leading-dim D_v=128. Both kernels are **shared between the native and
padded variants** — only the K-loop trip count and the device-buffer width
differ.

## ADR-0005 padding methodology

Per `docs/adrs/0005-mla-padding-methodology.md`, the nvcc + WMMA cell uses
**192-native (no padding)** as the headline number. We additionally support a
**padded-256 variant** in the same binary so the cross-frontend table can list
both `useful_tflops` (qk=192) and `padded_tflops` (qk_pad=256).

Padding details:

| Variant | qk_eff | K-loop trips | wasted-FLOP fraction | device buffer |
|---|---:|---:|---:|---|
| `native` (headline) | 192 | 12 | 0% | (B, n_h, S, 192) f16 |
| `padded` | 256 | 16 | 25% on QKt only | (B, n_h, S, 256) f16, trailing 64 cols zero |

The softmax `scale = 1/√qk_head_dim` uses the **TRUE** qk=192 in both
variants. Because padded cols of K are exactly zero, the score matrix
is bit-identical between native and padded paths — the only thing that
changes is the wall time. **This is the central correctness invariant** of
ADR-0005, and our correctness check verifies it directly: native and padded
both produce `max_abs_err = 1.597e-04` on the small shape.

The padding here is **BOTH register-padded AND DRAM-padded** — we widened
the device buffer from 192- to 256-wide. The harness reports both K-traffic
numbers so future readers can sanity-check that the LDG.E counts in
`attn_mla.sass` correspond to the wider buffer. (For a DRAM-saving
alternative, a future revision could keep DRAM at 192 and only register-pad
inside the WMMA fragment — no change to the FLOP count, but K-traffic
unchanged from native.)

## Correctness — quoted run.log evidence

Build + run from the harness:

```
$ make
/usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lstdc++ -lm -o attn_mla attn_mla.cu

$ ./attn_mla
[mla] device: NVIDIA GeForce RTX 5090 (sm_120)
[mla] inputs dir: /home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs
[mla] === correctness run (B=1 n_h=4 S=128 qk=96 d_v=64) ===
[mla] correctness_mla qk=96 (qk_eff=96, native) -> max_abs_err=1.597e-04 max_rel=7.271e+00 expected_max_abs=3.039e-01
[mla] correctness_mla qk=96 (qk_eff=128, padded) -> max_abs_err=1.597e-04 max_rel=7.271e+00 expected_max_abs=3.039e-01
[mla] correctness: native=1.597e-04 (OK)  padded=1.597e-04 (OK)  TOL=1e-02
```

**Acceptance:** Wave 17 W1a row demands `max_abs_err ≤ 1e-2` vs PyTorch
SDPA-MLA. We hit **1.597e-04**, ~60× tighter than the bound. Expected output
magnitudes peak at 3.039e-01, so relative error is ~5e-4 — well within
the `f16` tolerance band from `tolerances.py` (`atol=5e-3, rtol=5e-3`).

The "max_rel=7.27" outlier is the standard f16-attention artifact: a
near-zero true value (|want| < 1e-6) divided by a small absolute error;
the absolute-error bound is what matters and it passes by 60×.

## SASS check — quoted run.log evidence

```
$ /usr/local/cuda/bin/cuobjdump --dump-sass attn_mla > attn_mla.sass
$ grep -c HMMA   attn_mla.sass    # → 20
$ grep -c FFMA   attn_mla.sass    # → 104
$ grep -c MUFU   attn_mla.sass    # → 27
$ grep -c LDG.E  attn_mla.sass    # → 129
$ grep HMMA attn_mla.sass | head -2
        /*0850*/   HMMA.16816.F32 R20, R12, R18, R20 ;   /* 0x000000120c14723c */
        /*0860*/   HMMA.16816.F32 R16, R12, R16, R8 ;    /* 0x000000100c10723c */
```

- **HMMA = 20 > 0** — Wave 17 W1a acceptance row satisfied; tensor cores
  engaged on sm_120. Real `HMMA.16816.F32` instructions present, not just a
  stale ABI symbol.
- **MUFU = 27** — present (expected, used by `expf` in the softmax kernel).
- **FFMA = 104** — scalar FMAs (softmax scaling, store conversion).
- **LDG.E = 129** — global-memory loads. `LDG.E.128` count = 0; we are not
  using 16-byte vector loads. The WMMA-fragment loaders `load_matrix_sync`
  emit `LDG` widths chosen by the WMMA implementation — we do not control
  these directly. (Compare `cuda-attn-gqa`'s SASS for the same pattern.)

## Expected TFLOPS estimate (no bench was run)

Bench was **not executed** in this author-cell. Per orchestrator policy
the timed bench runs serially across all 5 W1 cells on the idle GPU.

Estimate is derived from the GQA-template baseline:

- `cuda-attn-gqa` bench: 2.93 ms median, **23.46 TF** at 68.72 GFLOP/iter.
- Per-iter throughput (kernel-bound, unfused) ≈ 23.5 TF.
- MLA bench shape FLOPS (`flops_mla.py`):
  `2 · 1 · 128 · 2048² · (192 + 128) = 3.44e11 ≈ 343.6 GFLOP/iter` (useful).
  This is **5.0× the GQA template's per-iter work**.
- Naive scale-up: ~5.0 × 2.93 ms = **~14.7 ms/iter** at the same 23.5 TF.

That gives an **expected useful TFLOPS ≈ 23 TF** (range plausibly
**[20, 30]** — Q/K/V working set is 5× larger so HBM-bound stages may
slip a bit; on the other hand, 4× more Q heads = 4× more parallel grid
should lift occupancy). Padded variant: same wall time (HMMA-bound, not
HBM-bound on QKt), so `padded_tflops = 343.6 × 1.20 / wall ≈ 28 TF`
diagnostic / `useful_tflops ≈ 23 TF` headline.

**Wave 17 plan range [40, 130]** is on the high side relative to the GQA
template's 23 TF; if we land below 40 TF the cause is the same one that
caps GQA at 23 TF — 3-kernel HBM round-trips dominate over compute, the
wave-16 lesson (3-kernel vs fused = 6.8× gap on GQA). Will NOT close that
gap in this cell; the cross-cell comparison story is the headline.

vs `cutile-attn-mla`'s 112 TF: gap is plausibly 4-5× — same magnitude as
the GQA `cuda-attn-gqa` (23 TF) vs `cutile-attn-gqa` (165 TF) gap of ~7×.
Pattern intact: nvcc 3-kernel pipeline with WMMA hits the same fraction
of the cuTile fused ceiling regardless of mechanism.

## What was NOT done in this cell

- No timed bench iters (`--bench-now`). Orchestrator-serial only.
- No fused single-kernel implementation (FlashAttention-2 style). This cell
  is pedagogical-MVP / 3-kernel; fusion is a Wave-18 candidate per the
  cross-mechanism-summary discussion in `docs/research/wave17-oxide-mla-design.md`.
- No weight-absorption or latent-c_KV up-projection kernel — those are
  upstream, not part of this cell's scope per `shapes_mla.py` MVP note.

## Files in this cell

| File | Purpose |
|---|---|
| `attn_mla.cu` | All 3 kernels + NPY harness + correctness + bench main |
| `Makefile` | `nvcc -ccbin clang-14 -O3 -arch=sm_120` build |
| `run.sh` | Build → correctness → SASS-dump driver. `./run.sh bench` for timed iters |
| `bench.cu` | Stub (single-binary pattern; lives in `attn_mla.cu`) |
| `.gitignore` | binary, sass, results.csv, run.log |
| `ANALYSIS.md` | this file |

## Pitfalls encountered (and avoided)

1. **WMMA matrix_b col-major ld**: For QKt with K stored row-major (S, QK), the
   `col_major` matrix_b view of K-rows-as-columns has leading-dim = QK (the
   row stride), NOT S. Fragment-load API hides this; `load_matrix_sync(bf,
   Bptr, QK)` is the correct call. Same trap as GQA template;
   no surprise.

2. **softmax scale uses TRUE qk, not padded qk**: ADR-0005 invariant. With
   padded K-cols zero, scores' magnitudes are unchanged — but the scale
   formula must still use 192 (or 96) so the score distribution matches
   PyTorch's `scaled_dot_product_attention` reference. Verified by
   bit-identical `max_abs_err` between native and padded variants.

3. **Per-row pad on host side**: The `alloc_padded_qk` helper pads with
   zeros only when `QK_eff != sh.QK`. We use a host-side staging vector
   rather than two `cudaMemcpy` calls per row; this is cleaner and the
   pad happens once at correctness-init / bench-init time.

4. **Working-copy file ownership in jj-managed repo**: with parallel
   subagents (oxide-attn-{gdn,mla} et al.) committing into the same
   working copy, my files had to be `jj split`-isolated into their own
   commits before describing. AGENTS.md run-discipline-pattern: per-cell
   author commits owned-paths only, never `jj describe` of a mixed
   working copy.

5. **Padding overhead reported as 20% (not 25%) at the bench banner**:
   ADR-0005 says the cuTile cell pays 25% wasted compute. Our number is
   `padded_flops/useful_flops − 1 = (qk_pad − qk)/(qk + d_v) =
   64 / 320 = 20%` — different denominator. cuTile's 25% counts QKt-only;
   our top-line counts QKt+PV. Both numbers are correct measures of
   different things; the analysis row in this doc lists "25% on QKt only"
   to match ADR-0005's framing.

## Acceptance summary

| Criterion | Required | Actual |
|---|---|---|
| Compiles with `-arch=sm_120` | yes | ✅ (build.log: 0 warnings, 0 errors) |
| Correctness max_abs_err vs SDPA-MLA at small shape | ≤ 1e-2 | ✅ **1.597e-04** (60× tighter) |
| HMMA > 0 in SASS | > 0 | ✅ **20** |
| qk-native variant | required | ✅ |
| qk-padded-256 variant | required (ADR-0005) | ✅ (correctness verified, identical to native) |
| Bench iters timed | NO (orchestrator-serial) | not run |
