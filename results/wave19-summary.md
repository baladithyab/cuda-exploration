# Wave 19 — Mojo compute-bound matmul (Phase C)

**Status:** ✅ Phase C1 + C3 SHIPPED. C2 (microtile f32) skipped.
**Date:** 2026-05-20 (single-session continuation of Wave 18)
**GPU:** RTX 5090, sm_120, driver 596.21 (idle 42°C, ramp to ~46°C)
**Mojo:** 1.0.0b1 (a9591de6)

## Headline finding

**Mojo's `from layout.tensor_core import TensorCore` wrapper engages
tensor cores on consumer Blackwell sm_120.** SASS shows **64
`HMMA.1684.F32.TF32`** instructions in the warp inner loop, no scalar
FFMA. **No tcgen05** (Mojo correctly takes the legacy `mma.sync` path,
not the sm_100a-only Blackwell tcgen05 path that fails on sm_120 per
issue #5707).

This is the first non-CUDA-C++ frontend other than cuTile to demonstrate
TC reach on RTX 5090 in this repo. cuda-oxide is still shut out.

## Numbers @ M=N=K=4096

| frontend | algorithm | TFLOPS | precision | TC? |
|---|---|---:|---|---|
| cuTile | mma_bf16xbf16_f32acc | **159** | bf16→f32 | ✅ |
| cuTile | mma_tf32xtf32_f32acc | 84 | tf32→f32 | ✅ |
| **Mojo (NEW)** | **tensor_core (TF32)** | **55.5** ⚡ | f32 (TF32 hw) | ✅ |
| nvcc CUDA C++ | tiled f32 | 38 | f32 | ❌ |
| cuTile | mma_f32xf32_f32acc | 8.7 | f32 | ❌ |
| oxide unchecked | fmuladd | 7.0 | f32 | ❌ (Wave 14: shut out) |
| **Mojo (NEW)** | **naive f32** | **7.1** | f32 | ❌ |
| nvcc CUDA C++ | naive f32 | 6.4 | f32 | ❌ |

**Two stories:**
1. Naive Mojo == naive nvcc == naive oxide (parity within thermal noise,
   all are scalar-FFMA bottlenecked on B-load coalescing).
2. Mojo's `TensorCore` wrapper gets ~66% of cuTile's TF32 perf and ~35%
   of cuTile's bf16 perf at the same shape. The bf16 lane is currently
   **unreachable** through the high-level wrapper (compile-time constraint
   `A.dtype == C.dtype` in `TensorCore.store_d`).

## Phases

### C1: `mojo-matmul/` — naive f32 baseline

Direct port of `cuda-matmul/matmul.cu`. 16×16 thread block, no shared
memory, one output per thread. Sizes {1024, 2048, 4096}, 1 warmup +
10 timed iters via `ctx.execution_time`.

**Result:** 6.9 / 6.7 / 7.1 TFLOPS at N ∈ {1024, 2048, 4096}.

Establishes a Mojo-native scalar-FFMA reference so the C3 row has a
fair "+7×" comparison on the same frontend rather than across frontends.

### C2: `mojo-matmul-tiled/` — SKIPPED

Originally planned f32 microtile. Skipped because:
- nvcc's tiled cell (38 TFLOPS) and cuTile's `matmul_tiled_simple`
  (2.5 TFLOPS, blocked on cooperative-tile launch overhead) already
  cover the f32-microtile compute-bound row.
- The Mojo manual's 7-kernel sequence (naive → coalescing → tiled →
  tiled_register → block_tiled → block_tiled_vectorized → tensor_core)
  is identical to Simon Boehm's progression we're already familiar with.
  We don't need to re-prove that block-tiling beats naive on RTX 5090.
- Time budget: C3 (the headline TC question) is the high-information
  kernel. Microtile is filler.

A dedicated `mojo-matmul-block-tiled-vectorized/` (algorithm 6 from the
manual, expected ~70-90 TFLOPS f32 microtile) is a Wave 20 candidate if
someone wants the full progression.

### C3: `mojo-matmul-tc/` — Tensor Core probe ⚡

Direct port of `tensor_core_matrix_multiplication` from the official
Modular tutorial. `f32`/`f32`/`f32` with `MMA_M=16, MMA_N=8, MMA_K=4`
(TF32 m16n8k4 path on Ampere/Ada/Blackwell).

**Result: 55.5 TFLOPS @ 4096³ (2.47 ms/iter).**

SASS captured to `mojo-matmul-tc/matmul_tc.sass` (22KB, 339 lines):
- `.target sm_120a` (native sm_120 with arch-specific extensions)
- 64 × `HMMA.1684.F32.TF32` (the m16n8k4 TF32 SASS lowering of `mma.sync`)
- 0 × `tcgen05.mma` (correct — sm_120 doesn't support these)
- 0 × `UTMALDG` (TMA loads — Mojo `TensorCore` doesn't auto-emit TMA;
  cuTile does, which explains some of the 55.5 vs 84 gap)

## Why bf16-in/f32-acc didn't ship

First attempt used `a_type=DType.bfloat16, c_type=DType.float32` with
`MMA_K=8` (m16n8k8 bf16). Failed at compile time:

```
constraint failed: destination tensor must have the same type
  in TensorCore.store_d at max/kernels/src/layout/tensor_core.mojo:781
```

The high-level `TensorCore` wrapper enforces `A.dtype == C.dtype`. Three
options:
1. Stay with same-dtype TF32 (what we shipped — 55.5 TFLOPS).
2. Drop to `from std.gpu.compute.mma import mma` and emit
   `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` directly.
   This is the Wave 20 candidate.
3. Use `from layout.tensor_core_async import TensorCoreAsync` (the
   newer wrapper used in `matmul_blackwell_iterative/`) — but that
   path uses TMA + tcgen05 and targets sm_100a, not sm_120.

Option 2 is the right path for a future "mojo-matmul-mixed" cell.

## Pitfalls (added to skill)

1. **`divmod(warp_id(), UInt(...))` doesn't compile** in Mojo 1.0.0b1.
   Use `wid // (BN//WN)` and `wid % (BN//WN)` instead — the manual's
   example was written for an older Mojo.
2. **`var x, y = divmod(...)` destructuring** silently produces "unknown
   declaration" errors downstream. Use two separate assignments.
3. **`TensorCore.store_d` requires A.dtype == C.dtype.** No mixed-precision
   through the high-level wrapper. Use `std.gpu.compute.mma` for bf16-in/f32-acc.
4. **`Float32 → BFloat16` implicit cast not allowed** in host buffer init.
   Wrap in `.cast[a_type]()`.

## Re-bench discipline

GPU thermal at start of session: 42°C, 28W (carry-over from Wave 18 close).
At end of mojo-matmul-tc (most expensive run): 42°C, 44W. **Same idle
thermal window as Wave 18 — no fresh re-bench of nvcc/oxide/cuTile cells
needed; numbers from Wave 18 close are still valid.**

## What's next (Wave 20 candidate)

| topic | expected outcome |
|---|---|
| `mojo-matmul-mixed/` — bf16-in/f32-acc via `std.gpu.compute.mma` | 100-130 TFLOPS, closing 60-80% of cuTile's bf16 lead |
| `mojo-matmul-tma/` — add `std.gpu.host.nvidia.tma` to the TF32 path | 70-80 TFLOPS, closing the gap to cuTile's 84 |
| Numerical correctness check vs `vendor_blas.matmul` | atol/rtol bounds for both Mojo TC paths |
| `mojo-matmul-block-tiled-vectorized/` | f32 microtile @ ~70-90 TFLOPS, no TC |

Wave 20 is "Mojo TC depth" — drilling into the `TensorCore` wrapper's
limits and the lower-level `mma` API. By contrast Wave 17 (KDA + GDN +
oxide-MLA) remains its own track.
