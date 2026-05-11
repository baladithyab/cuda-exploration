# cutile-matmul-tiled-mixed — Wave 13.1 ANALYSIS

**Wave 12.4 hypothesis FALSIFIED at the dtype level.** cuTile's `ct.mma` is
not "broken on matmul" — it is dtype-conditional. With tensor-core-native
dtypes (f16, bf16, tf32), `ct.mma` engages Blackwell HMMA tensor cores
correctly and reaches **172.5 TFLOPS** at f16×f16→f32acc on N=4096 — which
is **2.34× faster than cuBLAS sgemm** (the f32 reference) and **22.8× faster
than cuTile's own f32 kernel** measured in Wave 12.4.

This reframes the Wave 12 headline from "cuTile is 5× behind on matmul" to
"cuTile needs the right dtype to engage tensor cores." The f32×f32 path
remains broken / non-FFMA-fused, but it shouldn't have been the test case
in the first place — Blackwell has no f32 tensor-core MMA, only f16/bf16/
tf32/int8/fp8.

## TFLOPS table (median of 10 cudaEvent iters, RTX 5090 sm_120, idle GPU 43°C)

| variant | N=1024 | N=2048 | N=4096 | vs cuTile f32 |
|---|---:|---:|---:|---:|
| `mma_f16xf16_f32acc`  | 63.5 |143.2 |**172.5** | **22.8×** |
| `mma_bf16xbf16_f32acc`| 60.8 |134.1 |  159.8 |  21.1× |
| `mma_tf32xtf32_f32acc`| 34.3 | 71.1 |   84.0 |  11.1× |
| `mma_f32xf32_f32acc`  |  3.8 |  7.5 |    8.7 |   1.0× (W12.4 repro) |

For reference (this same session, same GPU):

| reference | N=4096 TFLOPS |
|---|---:|
| cuBLAS sgemm (f32, tensor cores via TF32 internal) | 73.6 |
| nvcc shared-mem-tiled f32 (CUDA cores) | 38.4 |
| cuda-oxide tiled-microtile f32 (CUDA cores) | 45.0 |

## SASS evidence — tensor-core instruction emission per dtype

Disassembled with `/usr/local/cuda/bin/cuobjdump --dump-sass` (CUDA 13.2)
on cubins exported via `cuda.tile.compilation.export_kernel(..., gpu_code='sm_120', output_format='cubin')`.

| variant | HMMA instructions | shape | FFMA | FMUL | FADD |
|---|---:|---|---:|---:|---:|
| `mma_f16xf16_f32acc`   | **64** | `HMMA.16816.F32`        | 0 | 2 | 0 |
| `mma_bf16xbf16_f32acc` | **64** | `HMMA.16816.F32.BF16`   | 0 | 2 | 0 |
| `mma_tf32xtf32_f32acc` | **128** | `HMMA.1688.F32.TF32`   | 0 | 3 | 0 |
| `mma_f32xf32_f32acc`   | **0**  | (none)                  | 0 | 2051 | 2176 |

**Three findings from SASS:**

1. **f16 and bf16 use identical TC shape** (`HMMA.16816.F32`, m=16 n=16 k=16)
   differing only in the `.BF16` operand suffix. Same throughput within ~10%,
   matching the measured perf ratio.

2. **tf32 emits 2× as many HMMA ops** because Blackwell's tf32 TC shape is
   `HMMA.1688.F32.TF32` (k=8) instead of k=16. Half throughput, exactly
   matching the measured 84 TF vs 172 TF perf ratio for f16. The
   instruction-count ratio predicts the perf ratio.

3. **f32 emits 2051 FMUL + 2176 FADD with zero FFMA fusion.** Not just
   "no tensor cores" — the kernel doesn't even get the basic CUDA-core FFMA
   instruction. This is a separate bug from the TC question: cuTile's `ct.mma`
   at f32 falls back to a scalar broadcast-and-sum path that misses the
   FFMA contractor entirely. This explains the very low absolute throughput
   (8.7 TF vs nvcc's 38.4 TF for the same hand-tiled algorithm) — even on
   CUDA cores cuTile is leaving 2-4× on the table for f32.

(The Wave 13.2 SASS subagent's "STL spills" claim did not reproduce on
re-grep — `STL\b` count is 0 in all 4 mixed-precision cubins. The f32
kernel doesn't spill; it's just unfused.)

## Honest comparison caveat

**The 172.5 TFLOPS at f16 is *not* apples-to-apples with cuBLAS sgemm at 73.6 TFLOPS.**
cuBLAS sgemm runs on f32 inputs with f32 outputs. The fair reference is cuBLAS
**hgemm** (half-precision gemm) which on RTX 5090 typically reaches several
hundred TFLOPS (the hardware peak for fp16×fp16→fp32 on Blackwell consumer
sm_120 is on the order of ~400 TF dense, ~800 TF sparse). So cuTile at 172 TF
is engaging tensor cores correctly — but a hand-tuned cuBLAS hgemm would
likely beat it by another 2-3×.

**Adding cuBLAS hgemm + bgemm baselines is the right Wave 13.3 to make this
table self-contained.**

## Pitfalls hit by subagent

- **`CallingConvention.cutile_python_v1` is a factory method**, not an enum
  value. Must call it: `CallingConvention.cutile_python_v1()` — passing the
  bound method itself raises `TypeError: Unsupported calling convention`.

- **Python-level dtype branching inside `@ct.kernel` fails** with
  `TileTypeError: Operator 'is' expects one of the operands to be None`.
  Cannot do `if in_dtype is ct.tfloat32: ...` inside a kernel body. Workaround:
  separate kernel factories per dtype path.

- **CuPy has no native bf16.** Used `ml_dtypes.bfloat16` numpy interop:
  `bf16_np = np.asarray(values, dtype=ml_dtypes.bfloat16); bf16_cupy = cupy.asarray(bf16_np)`.

- **One outlier at tf32 N=2048 (3× hiccup).** Single iter measured 0.706 ms
  vs the median 0.241 ms. Median timing absorbed it. Likely WSL2 GPU-state
  blip; not a kernel issue.

- **BM=BN=128, BK=16 worked for all 4 dtypes** without tile-shape adjustment.
  Same kernel geometry across the dtype matrix, only operand types vary.

## Reframed Wave 12 headline

**Old (Wave 12.4):** "cuTile is 5× behind nvcc on matmul. ct.mma broken."

**New (Wave 13.1):** "cuTile's `ct.mma` engages Blackwell tensor cores
correctly at f16/bf16/tf32 (HMMA.16816.F32 / HMMA.1688.F32.TF32). The Wave
12.4 result of 7.57 TFLOPS at f32×f32 is correct but unrepresentative —
Blackwell has no f32 MMA, so calling `ct.mma` on f32 inputs falls back to
a scalar non-FFMA-fused path. **Use cuTile's `ct.mma` with f16/bf16/tf32
inputs and an f32 accumulator** for compute-bound matmul-shaped work; never
with f32 inputs."

The cuTile f32→f32 fallback path *itself* has a sub-issue worth filing
upstream (no FFMA fusion → 2-4× slower than the same algorithm hand-written
in CUDA C++), but that's a smaller finding than the original "5× behind"
overclaim.

## Files

- `main.py` — 4 kernel variants + `--smoke` (default ON), `--bench`,
  `--export-cubins` flags
- `bench.log` — full timed run output (10 iters per N per variant)
- `smoke.log` — correctness check at N=512 per variant
- `export.log` — `export_kernel(...)` cubin generation log
- `results.csv` — per-iter `(impl, kernel, n, iter, gpu_ms, tflops)`
- `mma_*.cubin` — 4 exported cubins, one per dtype
- `mma_*.sass` — 4 disassembled SASS files, one per dtype
- `.gitignore`
