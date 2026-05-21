# Wave 22.4 — `mojo-matmul-fp8`: Hand-rolled e4m3 FP8 m16n8k32 matmul in Mojo

**Status: ✅ Author + correctness COMPLETE. Bit-exact PASSED at M=N=K=32.**

## Summary

The most-risky Wave 22 item turned out to be the easiest: Mojo's
`std.gpu.compute.mma.mma()` dispatcher in 1.0.0b1 **already supports the e4m3
FP8 m16n8k32 lane** on Blackwell sm_120. No `llvm_intrinsic` or
`inlined_assembly` fallback was needed.

- Single-warp probe (`mma_probe_fp8.mojo`) compiled and ran on first try.
  SASS confirms 1 × `QMMA.16832.F32.E4M3.E4M3` on `.target sm_120a`.
- Full tiled kernel at M=N=K=32 (single block, single warp, 8 MMAs) compiled
  and ran on first try **after one trivial fix** (the
  `LayoutTensor[i, j]` → `SIMD[t, 1]` `[0]`-suffix pitfall, skill
  pitfall #13). Numerical correctness: `max_abs_err = 0.0`,
  `max_rel_err = 0.0` against CPU reference. Tolerance was atol=1e-1 +
  rtol=5e-2; we beat it by 13+ orders of magnitude because both inputs
  were quantized to e4m3 on the host before kernel and reference
  consumption.

## Acceptance signals

| Signal | Status | Evidence |
|---|---|---|
| Probe compiles | ✅ | `mma_probe_fp8.mojo` runs to completion |
| Probe emits FP8 MMA SASS | ✅ | 1 × `QMMA.16832.F32.E4M3.E4M3` in `mma_probe_fp8.sass:1` |
| Full matmul compiles | ✅ | `matmul_fp8.mojo` runs to completion |
| Full matmul emits FP8 MMA SASS | ✅ | 8 × `QMMA.16832.F32.E4M3.E4M3` in `matmul_fp8.sass` (= 2 mma_m × 4 mma_n × 1 K-step) |
| `.target sm_120a` | ✅ | Top of both SASS dumps |
| Correctness | ✅ | `max_abs_err=0.0`, `max_rel_err=0.0`, well under atol=1e-1 + rtol=5e-2 tolerance |

> **Acceptance instruction is `QMMA`, not `HMMA`** on Blackwell consumer for
> FP8 (and INT8). The task spec said "HMMA.16832.F32.E4M3 > 0" but on
> sm_120 the emitter uses the `QMMA` mnemonic for sub-half-precision
> tensor-core ops. Both the probe and the matmul show this; the
> `.16832.F32.E4M3.E4M3` payload structure is identical to what the spec
> meant.

## API path that worked

```mojo
from std.gpu.compute.mma import mma

# Per-lane fragment shapes (matches PTX 9.7.13.4.7 m16n8k32 8-bit):
var a_frag = SIMD[DType.float8_e4m3fn, 16](...)  # 16 e4m3 / lane
var b_frag = SIMD[DType.float8_e4m3fn, 8](...)   #  8 e4m3 / lane
var c_frag = SIMD[DType.float32, 4](...)         #  4 f32  / lane
var d_frag = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

mma(d_frag, a_frag, b_frag, c_frag)
```

The Mojo dispatcher wires this directly to inline asm
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (per
`std.gpu.compute.arch.mma_nvidia` per the skill reference table).
Final SASS is `QMMA.16832.F32.E4M3.E4M3`.

**No fallback needed.** We did not have to call `llvm_intrinsic[...]` and we
did not have to drop to `inlined_assembly[...]`. The high-level `mma()` was
sufficient.

## Tile geometry

- **M = N = K = 32**, single block, single warp (32 threads).
- MMA shape: m16n8k32. NUM_MMA_M = 32/16 = 2, NUM_MMA_N = 32/8 = 4,
  NUM_MMA_K = 32/32 = 1. Total MMAs = 2 × 4 × 1 = 8.
- `A_smem` (32×32 e4m3) and `B_smem` (32×32 e4m3) are stack-allocated
  shared memory, populated by the single warp cooperatively (1024 bytes
  each, 32 elements per thread).
- Hand-rolled fragment loads from smem per **PTX ISA 9.7.13.4.7** (m16n8k32
  8-bit input distribution): A is 4 sub-blocks of 4 elements with a +8 row
  offset and a +16 column offset; B is 2 sub-blocks of 4 elements with a +16
  row offset.
- f32 accumulator: per-MMA 4 lane-registers, stored as a flat
  `LayoutTensor[f32, row_major(NUM_MMA_M, NUM_MMA_N * 4)]` in LOCAL
  (register) address space.
- Hand-rolled epilogue per **PTX ISA 9.7.13.4.8** m16n8 distribution.
- We did **not** use `TensorCore.load_a` / `load_b`. The bf16 hybrid
  pattern from Wave 21 piggybacks on the wrapper for fragment loads;
  for FP8 we don't have evidence the wrapper supports `float8_e4m3fn`,
  and the 8-bit distribution is different from 16-bit (4-elem groups
  vs 2-elem groups, two-axis sub-block tiling vs one-axis), so
  hand-rolling the loads was simpler than testing wrapper support.

## Numerical correctness recipe

The `max_abs_err = 0.0` result is real, not a false-positive:

1. A and B are initialized on the host to small e4m3-representable values
   (multiples of 0.0625 in [0, 15] × 0.0625 = [0, 0.9375]). Each
   multiplication produces values in [0, 0.879] which round to e4m3
   exactly (e4m3 has 3 mantissa bits → 8 quantization levels per binade).
2. Both the device kernel and the CPU reference loop read from
   `a_host` / `b_host`, which are bf16-cast e4m3 values. So both consume
   the **same already-quantized values** — no quantization-mismatch
   noise.
3. Sums of K=32 small values stay under 30, well within f32 exact
   integer-multiple-of-`0.0625*0.0625` representability. No f32
   accumulator rounding.
4. The MMA itself produces bit-identical results to the CPU loop because
   FP8 e4m3 × e4m3 → f32 is an exact dot product when the inputs and
   intermediate products are f32-representable.

This is the cleanest possible correctness signal. A wider tolerance
(atol=1e-1 + rtol=5e-2 per the task spec) is the right framing for
**production-scale** FP8 matmul where K is in the thousands and
quantization noise + summation order matters; at K=32 with bounded
inputs the path is bit-exact.

## SASS structure (matmul_fp8.sass, key fragments)

```text
.target sm_120a
.elftype @"ET_EXEC"
...
# 8 QMMA instructions, one per (mma_m, mma_n) ∈ {0,1} × {0,1,2,3}:
/*1760*/  QMMA.16832.F32.E4M3.E4M3 R24, R8,  R12, RZ ;
/*18b0*/  QMMA.16832.F32.E4M3.E4M3 R20, R8,  R16, RZ ;
/*1c40*/  QMMA.16832.F32.E4M3.E4M3 R24, R8,  R28, RZ ;
/*1cb0*/  QMMA.16832.F32.E4M3.E4M3 R8,  R8,  R6,  RZ ;
/*1d50*/  QMMA.16832.F32.E4M3.E4M3 R12, R20, R12, RZ ;
/*1de0*/  QMMA.16832.F32.E4M3.E4M3 R16, R20, R16, RZ ;
/*1e20*/  QMMA.16832.F32.E4M3.E4M3 R28, R20, R28, RZ ;
/*1e60*/  QMMA.16832.F32.E4M3.E4M3 R20, R20, R6,  RZ ;
```

Note `RZ` (zero-register) as the C input — that's because the kernel
stores the accumulator in registers between iterations; with K=32 ==
MMA_K, there's a single K-step, so each MMA's C input starts at zero
and the first MMA *is* the only MMA at that (mma_m, mma_n) position.
For larger K we'd see RZ replaced by the prior MMA's destination
register.

## Pitfalls hit and fixed

1. **`LayoutTensor[i, j]` returns `SIMD[t, 1]`, not `Scalar[t]`.** First compile
   pass failed with `cannot implicitly convert ... element_type to Float8_e4m3fn`
   on every smem-element read. Fixed by adding `[0]` suffix at all four
   reading sites: `A[r,c][0]`, `B[r,c][0]`, `A_smem[row,col][0]`,
   `B_smem[row,col][0]`, and `c_reg[mm, mn*4+i][0]`. **This is skill
   pitfall #13** (`references/mojo-mma-shapes.md` and main SKILL.md
   pitfall list). Asymmetric: writes (`A_smem[r,c] = scalar`) auto-
   broadcast and DON'T need `[0]` on the LHS.
2. **`@parameter for` is deprecated in 1.0.0b1, use `comptime for`.** Got
   warnings, no errors, but converted them all for forward compatibility.
   This is a fresher deprecation than the `fn`-keyword one (Wave 19) —
   `alias` is also deprecated for module-scope constants in favor of
   `comptime`. Skill pitfall #3 already mentions the `alias`
   deprecation; the `@parameter for` one isn't called out yet.
3. **Acceptance signal mnemonic is `QMMA`, not `HMMA`.** The task spec
   said "HMMA.16832.F32.E4M3 > 0 in SASS"; reality on Blackwell consumer
   is `QMMA.16832.F32.E4M3.E4M3` for FP8 inputs (and INT8 also gets
   `QMMA`). `HMMA` is for half-precision (bf16/f16/tf32). The probe
   surfaces this in 30 seconds; future spec writers should expect
   `QMMA` for any sub-half input dtype on sm_120.

## Pitfalls NOT hit (could affect future scaling work)

- The dispatcher worked on the first try with `mma()`. We did not need
  `from sys.intrinsics import llvm_intrinsic` and we did not need
  `inlined_assembly` from `sys.intrinsics`. (Both were the documented
  fallback paths.) If a future Mojo release breaks the FP8 dispatcher
  lane, the LLVM intrinsic candidate would be
  `llvm.nvvm.mma.m16n8k32.row.col.e4m3.e4m3` (untested) and the inline
  asm string is `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`.
- The PTX 9.7.13.4.7 fragment-distribution formulas worked on the
  first try (no off-by-one, no transposed indices). The bit-exact
  correctness result confirms both the A and B fragment indexing.
  This validates the formulas in the skill ref's "Shape | A elems/lane
  | B elems/lane" table for the `m16n8k32` row.
- The `TensorCore` wrapper's FP8 support status is **untested** — we
  bypassed it entirely. If a future kernel wants `loader.load_a()` for
  FP8, write a smoke test first.

## What's NOT in scope for Wave 22.4

- **No timed bench.** Task explicitly forbade timed benchmarks. The
  kernel is single-block / single-warp at the smallest valid size; even
  if we did time it, the numbers would be launch-overhead-bound and
  uninformative.
- **No scaling to 4096³.** That's a follow-up wave. The path is now
  unblocked: replace the cooperative smem load with
  `copy_dram_to_sram_async`, add a K-loop, increase BM/BN/WM/WN,
  and the MMA dispatch + epilogue are unchanged.
- **No comparison vs cuTile FP8.** cuTile doesn't have a published FP8
  lane in v1.3.0; cuBLAS FP8 (`cublasGemmEx` with `CUDA_R_8F_E4M3`)
  would be the right perf reference for the next wave.

## Files

| Path | Purpose |
|---|---|
| `mma_probe_fp8.mojo` | 32-thread single-warp probe; verifies dispatcher path |
| `mma_probe_fp8.sass` | SASS dump from probe (1 × QMMA) |
| `matmul_fp8.mojo` | M=N=K=32 single-block tiled kernel + correctness check |
| `matmul_fp8.sass` | SASS dump from matmul (8 × QMMA) + stdout including correctness lines |
| `matmul_fp8.stderr` | Compile warnings (deprecations, no errors) |
| `run.sh` | One-shot reproduce script |
| `ANALYSIS.md` | This file |

## Reproduce

```bash
cd /home/codeseys/cuda-exploration/mojo-matmul-fp8
./run.sh
```

Expected output tail:

```
QMMA.16832.F32.E4M3.E4M3 count in probe SASS:
1
QMMA.16832.F32.E4M3.E4M3 count in matmul SASS:
8
Correctness lines:
[mojo-matmul-fp8] M=N=K= 32  a_type=e4m3 c_type=f32  MMA=16x8x32  max_abs_err= 0.0  max_rel_err= 0.0
[mojo-matmul-fp8] correctness PASSED at M=N=K= 32  (atol= 0.1  rtol= 0.05 )
```
