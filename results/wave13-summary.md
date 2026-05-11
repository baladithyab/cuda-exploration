# Wave 13 — cuTile dtype falsification + SASS root-cause

**Date:** 2026-05-11. Continuation of Wave 12. Two parallel subagents fanned
out for `cutile-matmul-tiled-mixed` (Wave 13.1) and `analysis/wave13-sass`
(Wave 13.2). Orchestrator independently verified each subagent's
critical claims by re-grepping SASS counts.

## Headline reframing — Wave 12.4 finding overturned at the dtype level

**Wave 12.4 (overstated):** "cuTile's `ct.mma` is 5× behind nvcc on matmul.
Doesn't engage tensor cores."

**Wave 13.1 (corrected):** "cuTile's `ct.mma` engages Blackwell tensor cores
correctly at f16, bf16, and tf32 — reaching **172.5 TFLOPS at f16** on N=4096,
which is 22.8× faster than the f32 path. The Wave 12.4 result was correct
but unrepresentative: Blackwell consumer GPUs (sm_120) have NO f32 MMA
hardware, so calling `ct.mma` on f32 inputs falls back to a CUDA-core
broadcast-and-sum path that also lacks FFMA fusion (a separate, smaller
upstream issue). Use `ct.mma` with f16/bf16/tf32 inputs and an f32
accumulator for compute-bound matmul work."

## SASS-verified evidence (RTX 5090 sm_120, cuobjdump CUDA 13.2)

### cuTile matmul, all 4 dtypes (`ct.mma` lowering)

| variant | HMMA shape | HMMA count | FFMA | FMUL | FADD | TFLOPS @ N=4096 |
|---|---|---:|---:|---:|---:|---:|
| `mma_f16xf16_f32acc`   | `HMMA.16816.F32`        | 64  | 0 | 2    | 0    | **172.5** |
| `mma_bf16xbf16_f32acc` | `HMMA.16816.F32.BF16`   | 64  | 0 | 2    | 0    |   159.8  |
| `mma_tf32xtf32_f32acc` | `HMMA.1688.F32.TF32`    | 128 | 0 | 3    | 0    |    84.0  |
| `mma_f32xf32_f32acc`   | (none — no f32 MMA HW)  | 0   | 0 | 2051 | 2176 |     8.7  |

The instruction-count ratios match the perf ratios. tf32's 2× HMMA count
(due to k=8 vs k=16 shape) corresponds to ~half throughput — exactly what
the bench shows (84 vs 172 TF).

### Reduction win explained: TMA bulk loads (cuTile only)

cuTile reduce_sum hits 1696 GB/s vs oxide/nvcc's 1519/1522 GB/s. SASS reveals
**why**: cuTile emits `UTMALDG.1D` (unified TMA bulk-copy global→shared)
instructions — 7 of them per kernel — that nvcc and oxide do not emit at all.

| metric | cuTile | oxide | nvcc |
|---|---:|---:|---:|
| `UTMALDG.1D` (TMA bulk) | **7** | 0 | 0 |
| `LDG.E` per-thread global | 0 | 0 | 1 |
| `LD.E` (untyped) | 0 | 3 | 0 |
| `LDG.E.CONSTANT` | 0 | 0 | 1 |
| `SYNCS.*` (async mbar) | **10** | 0 | 0 |
| `BAR.SYNC` | 8 | 1 | 1 |
| `FADD` (reduction tree) | 18 | 12 | 9 |
| `SHFL.BFLY` (warp reduce) | 7 | 8 | 8 |
| total SASS lines | 489 | 217 | 169 |

cuTile is **not leaner** in ALU work (more FADDs, more BAR.SYNCs, 2× the
SASS lines). It wins because it pulls each tile via a single TMA bulk
instruction directly into shared memory, while nvcc/oxide load element-
by-element through the per-thread LDG path. Same hardware FP throughput,
much lower per-byte LSU pressure on the load side.

This is a **real architectural advantage** of the cuTile programming model
— `ct.load(buffer, index, shape)` lowers naturally to TMA on Blackwell,
whereas a hand-written CUDA C++ kernel needs explicit `cuda::memcpy_async`
or pre-recorded `cuTensorMapEncodeTiled` descriptors to reach the same path.

### Why cuTile's f32 matmul is so slow (8.7 TF on a 38.4-TF algorithm)

The Wave 12.4 finding holds: cuTile's `matmul_tiled` SASS at f32 has:

- 2049 FMUL + 2176 FADD with **0 FFMA** — no FMA contraction
- 0 HMMA — confirmed no tensor cores (which is correct: no f32 MMA in HW)
- Compared to nvcc-tiled: **256 FFMAs**, no spills, register-tight inner loop
- Compared to oxide-tiled-microtile: **192 FFMAs** with `.reuse` register hints

So even discounting the no-tensor-core ceiling, cuTile's f32 path is
~8-12× lighter on FP-issue ops than it should be. This is a separate
upstream issue worth filing alongside the dtype-engagement question:

> **Upstream issue draft:** "`ct.mma` at f32×f32 falls back to scalar
> FMUL+FADD pairs without FFMA contraction. Expected: hardware FMA
> instructions or contracted FFMA from libNVVM's pattern-match contractor.
> Observed: 2049 FMUL + 2176 FADD over a 4096-iter K-loop. nvcc compiling
> the same shared-mem-tiled algorithm emits 256 FFMAs."

## Methodology check — what got verified

Subagent 1 (`cutile-matmul-tiled-mixed`) hit max_iterations (50 tool calls)
before writing ANALYSIS.md. Orchestrator independently verified the
killer claim by re-grepping SASS:

```bash
$ grep -c HMMA mma_f16xf16_f32acc.sass            # → 64 ✓
$ grep -c 'HMMA.16816.F32' mma_f16xf16_f32acc.sass  # → 64 ✓
$ grep -c 'HMMA.1688.F32' mma_tf32xtf32_f32acc.sass # → 128 ✓
$ grep -c HMMA mma_f32xf32_f32acc.sass            # → 0 ✓
```

Subagent's "STL register spills" claim did NOT reproduce on re-grep —
the f32 cubin has 0 STL instructions, not "233". The kernel doesn't
spill; it's just unfused. Caveat documented in `cutile-matmul-tiled-mixed/ANALYSIS.md`.

Subagent 2 (`analysis/wave13-sass`) extracted cubins and disassembled all
6 kernels successfully. TMA bulk-load finding verified:

```bash
$ grep -c UTMALDG cutile_reduction.sass  # → 7 ✓
$ grep -c UTMALDG cuda_reduction.sass    # → 0 ✓
$ grep -c UTMALDG oxide_reduction.sass   # → 0 ✓
```

## Honest caveat for the f16 win

**172.5 TFLOPS for cuTile f16×f16→f32 is not apples-to-apples with cuBLAS sgemm
at 73.6 TFLOPS** — different dtypes. The right reference is cuBLAS hgemm
(half-precision GEMM), which on RTX 5090 reaches several hundred TFLOPS for
a hand-tuned implementation. cuTile at 172 TF is a real tensor-core engagement
but a reasonable cuBLAS hgemm baseline would likely beat it by 2-3×.

Adding cuBLAS hgemm + bgemm benchmarks to this repo is the right next step
(Wave 13.3 candidate) to make the comparison table self-contained.

## Updated user-facing read (replaces Wave 12 SUMMARY take)

If you're choosing a Python-first or Rust-first GPU compute frontend on
Blackwell consumer hardware **today** (May 2026):

1. **For matmul-shaped compute-bound work**: use cuTile with `ct.mma` at
   f16/bf16/tf32 inputs + f32 accumulator. Reach competitive tensor-core
   throughput (172 TF f16) from Python with one decorator. **Do not call
   `ct.mma` on f32 inputs** — Blackwell has no f32 MMA, and cuTile's
   fallback path is 4× slower than even hand-written CUDA C++ at f32.

2. **For memory-bound reduction-pattern work**: use cuTile.
   `ct.sum`/`ct.reduce` lowers to TMA bulk loads on Blackwell, beating
   hand-written warp-shuffle paths by 11% on the same hardware.

3. **For memory-bound vec-add work**: pick whatever you prefer. All three
   stacks (cuTile, cuda-oxide, nvcc) are within 1%.

4. **For naive matmul without using `ct.mma`**: cuTile is significantly
   *worse* than cuda-oxide / nvcc, because the broadcast-and-sum form is
   redundant work. Prefer cuda-oxide for hand-written non-TC matmul.

5. **For peak f32 matmul performance**: cuBLAS sgemm (73 TF) wins. Use
   it if your dtypes can stay f32.

## Files added in Wave 13

- `cutile-matmul-tiled-mixed/` — 4 dtype variants, results.csv, 4 SASS dumps,
  ANALYSIS.md (orchestrator-written from subagent data)
- `analysis/wave13-sass/` — 6 cubins + 6 SASS dumps for cross-stack
  comparison, REDUCTION_SASS_DIFF.md, MATMUL_SASS_DIFF.md, instruction_counts.csv
- `results/wave13-summary.md` — this document

## Wave 14 candidates (lowest-hanging fruit)

- **W14.1: cuBLAS hgemm baseline** for fair comparison vs cuTile f16
- **W14.2: Filed upstream issue draft** for the cuTile f32 no-FFMA-fusion path
- **W14.3: cuTile reduction at smaller N** — does the TMA win persist when
  the bulk-load setup overhead can't amortize?
- **W14.4: oxide register-microtile + tf32 inputs** — can oxide use TF32 too?
  (rust-cuda probably doesn't have a tensor-core API today; verify.)
- **W14.5: 3DGS rasterizer port to cuTile** for completeness alongside oxide and nvcc.
