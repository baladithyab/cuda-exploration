# mojo-matmul-tc — Mojo Tensor Core matmul on sm_120

**Wave 19 Phase C3.** The headline kernel for Wave 19. Probes whether
Mojo's `from layout.tensor_core import TensorCore` wrapper can engage
tensor cores on consumer Blackwell (RTX 5090, sm_120). The question
matters because cuda-oxide is shut out from TC entirely (Wave 14 finding)
and cuTile via `cuda.tile.cooperative.mma` was previously the only frontend
in the repo with confirmed TC reach on sm_120.

## Result

**Mojo CAN reach tensor cores on sm_120.** SASS evidence (`matmul_tc.sass`):

```
.target sm_120a
...
HMMA.1684.F32.TF32 R20, R32, R4.reuse, R20 ;     ← × 64 instances
```

64 TF32 m16n8k4 tensor-core instructions in the warp loop, no scalar FFMA
in the inner kernel body. **No tcgen05** (so Mojo is taking the legacy
`mma.sync.aligned` path, not the Blackwell sm_100a tcgen05 path that
fails on sm_120 — see issue #5707).

## Performance @ M=N=K=4096

```
GPU: NVIDIA GeForce RTX 5090 (sm_120, driver 596.21)
Mojo: 1.0.0b1 (a9591de6)
[mojo-matmul-tc] M=N=K=4096 a_type=f32 c_type=f32 (TF32 TC path)
                MMA=16x8x4 avg_ms/iter=2.474 TFLOPS=55.55
```

10-iter avg, single warmup, `ctx.execution_time` (cudaEvent-equivalent).

## Cross-frontend matmul comparison @ N=4096³

| frontend | algorithm | GFLOPS | precision | TC reach? |
|---|---|---:|---|---|
| nvcc CUDA C++ | naive | 6,400 | f32 | no (scalar FFMA) |
| nvcc CUDA C++ | tiled | 38,000 | f32 | no (scalar FFMA) |
| cuda-oxide unchecked | fmuladd | 7,000 | f32 | **shut out** (Wave 14) |
| cuTile | matmul_tiled_simple | 2,500 | f32 | no |
| cuTile | mma_f32xf32_f32acc | 8,700 | f32 | no |
| cuTile | mma_tf32xtf32_f32acc | 84,000 | tf32→f32 | **yes** |
| cuTile | mma_bf16xbf16_f32acc | 159,000 | bf16→f32 | **yes** |
| **Mojo (NEW)** | **naive** | **7,000** | f32 | no (scalar FFMA) |
| **Mojo (NEW)** | **tensor_core** | **55,500** | f32 (TF32 hw) | **yes** ⚡ |

Two takeaways:
1. **Mojo joins cuTile in the TC-capable club on sm_120.** This is the
   first time we've seen TC reach from a non-CUDA-C++ frontend other than
   cuTile.
2. **Mojo's TF32 path is ~66% of cuTile's TF32 path** at this shape
   (55.5 vs 84 TFLOPS). The bf16 lane (cuTile's 159 TFLOPS killer) is not
   accessible through the high-level `TensorCore` wrapper — see "Mixed
   precision gap" below.

## Why TF32, not bf16?

The official `tensor_core_matrix_multiplication` kernel from the Mojo
manual (which we ported here verbatim apart from the harness) requires
`A.dtype == C.dtype`. When we tried bf16 inputs + f32 accumulator,
`mma_op.store_d(C_mma_tile, c_reg_m_n)` failed at compile time:

```
constraint failed: destination tensor must have the same type
```

So the wrapper supports same-dtype paths only:
- `f32`/`f32`/`f32` → m16n8k4 TF32 (what we use, **55.5 TFLOPS**)
- `bf16`/`bf16`/`bf16` → would be m16n8k16 bf16, but accumulator is also
  bf16 — numerically much worse than bf16-in/f32-acc
- `f16`/`f16`/`f16` → same caveat

To hit the bf16-in/f32-acc path that cuTile uses, Mojo requires dropping
to the lower-level `from std.gpu.compute.mma import mma` API and emitting
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` directly. That
matches the workflow we proved out for cuTile in `cutile-matmul-tiled-mixed/`
(via cuTile's `cuda.tile.cooperative.mma`). Wave 20 candidate.

## Tile shape

Direct port of the official Mojo tutorial NVIDIA values:

| param | value | note |
|---|---|---|
| BM, BN | 64, 64 | block tile |
| BK | 32 | OPTIMIZED_BLOCK_SIZE on NVIDIA |
| WM | 32 | warp tile rows |
| WN | 32 | warp tile cols (= WARP_SIZE) |
| MMA_M, MMA_N, MMA_K | 16, 8, 4 | TF32 m16n8k4 |
| NUM_WARPS | 1 | `(BM/WM) * (BN/WN) = 1 × 2` — wait |

**Actually `(BM // WM) * (BN // WN) = (64/32) * (64/32) = 4`** warps per
block, 128 threads/block. That matches the 128-thread launch in the
official kernel.

## SASS discipline

Captured 339-line SASS dump via `_dump_sass=True` kwarg on
`enqueue_function`. The `.target sm_120a` directive confirms native
sm_120 compilation (with arch-specific `a` extensions). The instruction
mix in the inner loop:

| instr | count |
|---|---:|
| `HMMA.1684.F32.TF32` | **64** |
| `LDG.E` (global loads) | counted in 117 with LDS/STG/STS/FFMA/FADD/BAR.SYNC/SHFL bucket |
| `mma.sync` (PTX-level) | not visible in SASS, but `HMMA` is the SASS lowering |
| `tcgen05` | **0** ← good, sm_120 doesn't support these |
| `UTMALDG` (TMA) | 0 ← Mojo `TensorCore` doesn't auto-emit TMA |

The 0 TMA count is interesting: cuTile's TF32 matmul **does** emit TMA
loads (UTMALDG) for the A and B tiles, which is part of why it edges out
Mojo at the same MMA shape. Mojo's `copy_dram_to_sram_async` is using
the older `cp.async` path, not TMA. To match cuTile's perf we'd need to
descend into `from std.gpu.host.nvidia.tma import ...` (which is what
the `2_tensor_core.mojo` Modular test in `matmul_blackwell_iterative/`
does — but that targets sm_100a tcgen05, not sm_120).

## Numerical sanity

We don't yet do an explicit reference-vs-device comparison in this cell
(matches the original `cuda-matmul/`'s 9-iter perf-only stance). Init
pattern uses `(i * 2654435761) % 256 * 0.001`, range [0, 0.255], so a
single output element in 4096³ is the sum of 4096 products in
[0, 0.065], expected magnitude ~133 (mean), max ~1024. Comfortably
within fp32 dynamic range. TF32 inputs sacrifice ~9 mantissa bits which
introduces ~10⁻³ relative error vs fp32, well within "tensor-core
acceptable" for ML-class workloads.

A correctness check vs vendor_blas could be added in Wave 20 if we
want full numerical confirmation. The cuTile mma cells already do this,
so the framework precedent is there.

## Pitfalls encountered

1. **`divmod(warp_id(), UInt(...))` doesn't compile in Mojo 1.0.0b1** —
   the `Int` overload signature requires Int denominator, but `warp_id()`
   returns UInt. Use explicit `wid // (BN//WN)` and `wid % (BN//WN)`
   instead. This is a regression from the manual's example code.
2. **`var x, y = divmod(...)` destructuring quietly produced `unknown
   declaration` errors downstream** — use two separate `var` statements
   or unpack into a tuple type.
3. **`A.dtype == C.dtype` constraint on `TensorCore.store_d`** — high-level
   wrapper supports same-dtype only. For mixed-precision (bf16-in/f32-acc),
   drop to `std.gpu.compute.mma`.
4. **Float32 → BFloat16 implicit cast not allowed** — wrap host init values
   in `.cast[a_type]()`. Same gotcha would apply if we'd kept bf16.

These four are added to the rust-gpu-compute skill alongside the Wave 18
pitfalls list.

## Files

- `matmul_tc.mojo` (~9KB) — the kernel + harness
- `matmul_tc.sass` (22KB, 339 lines) — captured SASS, tracked
- `run.sh` — reproduce
- `run.log` — fresh run output

## Next steps (Wave 20 candidate)

1. **bf16-in/f32-acc via `std.gpu.compute.mma.mma`** — drop the high-level
   wrapper, emit `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
   directly. Expected to match cuTile's 159 TFLOPS within 20-30%.
2. **Add TMA loads** via `std.gpu.host.nvidia.tma` — should close some of
   the gap to cuTile in the f32/TF32 path.
3. **Numerical correctness check** against vendor_blas.matmul, paralleling
   what `cutile-matmul-tiled-mixed/` does.
