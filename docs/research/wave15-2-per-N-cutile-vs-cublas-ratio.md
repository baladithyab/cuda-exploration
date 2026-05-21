# Wave 15.2 — Per-N cuTile vs cuBLAS ratios

## Executive summary

The cuTile-vs-cuBLAS efficiency ratio is **strongly N-dependent** on RTX 5090
(sm_120). At the headline size N=4096 the bf16 ratio sits around 73-74%, the f16
ratio at 81%, and the tf32 ratio at 81%. At N=2048 the picture is *better* —
cuTile pulls within 80-92% of cuBLAS — but at N=1024 every variant collapses to
**44-54%** of cuBLAS, indicating a small-N regime where cuBLAS's specialized
launch path and tile-shape autoselection dominate. The 79% figure quoted for
N=4096 is therefore a single point on a non-monotonic curve, not a stable
characterization.

## Data sources

- `cutile-matmul-tiled-mixed/results.csv` — 121 rows, four kernels
  (`mma_{f16,bf16,tf32,f32}xX_f32acc`), N ∈ {1024, 2048, 4096}, 10 iters each.
- `cublas-half-precision/results.csv` — 91 rows, three kernels (`hgemm`,
  `bgemm`, `sgemm_tf32`), same N grid, 10 iters each.

Both files were collected on the same RTX 5090 / CUDA 13.2 / sm_120 box per
this repo's standing methodology (see `cuda-exploration/AGENTS.md` Wave 1 +
Wave 4-6 notes). All TFLOPS values below are **medians of 10 iters** (avg of
the two middle values after sorting). Iter-0 outliers and occasional
WSL-WDDM-induced 30-50% spikes (e.g. cutile bf16 N=4096 iter 9 = 124.7 TFLOPS
vs neighbors ~160 TFLOPS) are diluted by the median.

### Kernel pairings used for ratios

| cuTile kernel                   | cuBLAS kernel | Tensor-Core class |
|---------------------------------|---------------|-------------------|
| `mma_f16xf16_f32acc`            | `hgemm`       | FP16 in, FP32 acc |
| `mma_bf16xbf16_f32acc`          | `bgemm`       | BF16 in, FP32 acc |
| `mma_tf32xtf32_f32acc`          | `sgemm_tf32`  | TF32 in, FP32 acc |

The fp32-in/fp32-acc cuTile kernel (`mma_f32xf32_f32acc`, ~3.7-8.7 TFLOPS) has
no apples-to-apples cuBLAS counterpart in the half-precision CSV and is
omitted; cuBLAS plain-fp32 sgemm would be the comparison and is not collected.

## Median TFLOPS by (impl, kernel, N)

Computed manually from the CSVs. Cited row ranges in the CSV are inclusive
(1-indexed, header = row 1).

### cuTile (`cutile-matmul-tiled-mixed/results.csv`)

| Kernel                  | N=1024 | N=2048 | N=4096 | CSV rows (1024 / 2048 / 4096) |
|-------------------------|--------|--------|--------|------------------------------|
| `mma_f16xf16_f32acc`    | 63.46  | 143.20 | 172.22 | 2-11 / 42-51 / 82-91         |
| `mma_bf16xbf16_f32acc`  | 60.68  | 134.08 | 159.77 | 12-21 / 52-61 / 92-101       |
| `mma_tf32xtf32_f32acc`  | 34.28  |  70.91 |  83.98 | 22-31 / 62-71 / 102-111      |

### cuBLAS (`cublas-half-precision/results.csv`)

| Kernel       | N=1024 | N=2048 | N=4096 | CSV rows (1024 / 2048 / 4096) |
|--------------|--------|--------|--------|------------------------------|
| `hgemm`      | 117.95 | 156.02 | 212.50 | 2-11 / 12-21 / 22-31         |
| `bgemm`      | 127.00 | 164.27 | 217.37 | 32-41 / 42-51 / 52-61        |
| `sgemm_tf32` |  76.96 |  89.12 | 104.08 | 62-71 / 72-81 / 82-91        |

## Per-variant ratios (cuTile / cuBLAS)

| Variant       | N=1024 | N=2048 | N=4096 |
|---------------|-------:|-------:|-------:|
| f16   / hgemm  | 0.538  | 0.918  | 0.810  |
| bf16  / bgemm  | 0.478  | 0.816  | 0.735  |
| tf32  / sgemm_tf32 | 0.445  | 0.796  | 0.807  |

## Per-N pivot (same numbers, transposed)

| N    | f16/hgemm | bf16/bgemm | tf32/sgemm_tf32 | Mean of 3 |
|------|----------:|-----------:|----------------:|----------:|
| 1024 |   0.538   |   0.478    |     0.445       |   0.487   |
| 2048 |   0.918   |   0.816    |     0.796       |   0.843   |
| 4096 |   0.810   |   0.735    |     0.807       |   0.784   |

## Answers to the W15.2 questions

**(1) Does the 79% bf16 ratio at N=4096 hold at smaller N?**
No. The bf16/bgemm ratio is 73.5% at N=4096 (the doc-of-record's "79%" appears
to round generously or used a slightly different summary statistic — *I report
median-of-10*; mean-of-10 lifts bf16 cuTile at N=4096 from 159.77 to ~155.6
TFLOPS due to the iter-9 outlier of 124.7, and bgemm mean is ~215.3, giving
mean-ratio ~0.72). The ratio is materially different at smaller N: 81.6% at
N=2048 and only 47.8% at N=1024.

**(2) If not, does it improve or degrade at smaller N?**
**Non-monotonic.** From N=4096 → N=2048 it *improves* (73.5% → 81.6%). From
N=2048 → N=1024 it *degrades hard* (81.6% → 47.8%). Same shape for f16 and
tf32 — the N=2048 sweet spot and the N=1024 collapse are universal in this
data set.

**(3) Does the trend differ between bf16/f16/tf32?**
The *shape* is the same for all three (sweet spot at N=2048, collapse at
N=1024). The *magnitudes* differ:

- **f16/hgemm** is the closest match across the board (0.92 / 0.81 / 0.54).
- **bf16/bgemm** is consistently 8-10 percentage points behind f16/hgemm at
  every N. cuBLAS's bgemm seems to have a slightly stronger small-N kernel
  selection than its hgemm (bgemm = 127 vs hgemm = 118 TFLOPS at N=1024),
  which widens the gap.
- **tf32/sgemm_tf32** is the worst at N=1024 (0.45) but *catches up to f16*
  by N=4096 (0.81). cuBLAS's sgemm_tf32 scales relatively poorly with N
  (76.96 → 89.12 → 104.08, only a 1.35× speedup from N=1024 to N=4096),
  while cuTile's tf32 scales 2.45× over the same range. So the tf32
  ratio improves with N largely because cuBLAS-tf32 *underdelivers* at
  large N, not because cuTile tf32 is exceptional.

**(4) sgemm_tf32 specifically:** the cuTile tf32 / cuBLAS sgemm_tf32 ratio
ends up *roughly the same as the f16 ratio* at N=4096 (0.807 vs 0.810).
Read as: cuTile tf32 is no worse off relative to cuBLAS than cuTile f16 is.
The TF32 path is not a special weakness for cuTile on Blackwell.

## "Why" — hypotheses for the small-N collapse

The N=1024 cliff is the most interesting feature. Several non-exclusive
explanations:

1. **Kernel launch + cuBLAS handle/init amortization.** Even with handle
   reuse, cuBLAS dispatches into hand-tuned heuristics that pick smaller,
   higher-occupancy tile shapes for small problems. cuTile's static tile
   shape (chosen for N=4096-class workloads) is wrong for N=1024, leaving
   SMs underused. At GEMM N=1024 the total work is ~2.1 GFLOP — at ~60
   TFLOPS that's 35 µs of device time, on the same order as a kernel launch
   (~10 µs) and any per-launch SM scheduler ramp. cuBLAS at 127 TFLOPS does
   the same work in 16.5 µs.

2. **Tile-shape mismatch / register pressure.** The cuTile kernels appear
   to be authored with a single tile geometry. At N=1024, an M=N=K=128 tile
   yields only 8×8=64 CTAs, well below saturation on a 170-SM Blackwell
   GPU. cuBLAS likely picks a smaller tile (e.g. 64×64) at this size to
   unlock more SMs. This isn't a register-pressure problem per se — it's a
   *tile-too-large-for-the-problem* problem. (Register pressure would show
   up at *large* N as occupancy degradation, not at small N.)

3. **Tail effect on the K loop.** At N=1024 with a typical 128-K tile, the
   K loop has only 8 iterations. cuBLAS's pipelined async-copy schedules
   amortize prologue/epilogue better than cuTile's pre-MMAv5-ish tile
   abstraction can over only 8 iterations. The 92% / 82% / 80% ratios at
   N=2048 (16 K-iters) and N=4096 (32 K-iters) are consistent with the
   prologue/epilogue overhead being roughly constant per kernel launch and
   thus a smaller fraction of total time at larger K.

4. **Variance, not trend.** Iter-0 cuBLAS measurements at N=1024 are warm-up
   (e.g. hgemm iter-0 = 73.26 TFLOPS vs steady ~118), but the median already
   excludes those. The ratio gap is real and reproducible across all three
   variants, not a measurement artifact.

The likeliest *primary* cause is (2)+(3): a single static tile shape geared
for N=4096 that loses parallelism (too few CTAs) and prologue-amortization
(too few K-iters) at N=1024. The fact that f16 and bf16 collapse together at
N=1024 supports this — they share the cuTile tile-shape selection and only
differ in the MMA datatype.

## Caveats

- **WSL-WDDM jitter.** Iter-9 cuTile runs at N=4096 show 25-30% drops on
  bf16 and f16; this matches the Wave-1/3 thermal/scheduling notes in
  `cuda-exploration/AGENTS.md`. The median absorbs these but the mean would
  not.
- **Single tile shape per cuTile kernel.** Conclusions about "cuTile vs
  cuBLAS" are really "this particular cuTile tile shape vs cuBLAS's
  autotuned selection". A retune of cuTile at each N could close most of
  the small-N gap.
- **Iter-0 cold start.** All cuBLAS kernels have a warm-up iter-0 (e.g.
  bgemm N=1024 iter-0 = 64.65 TFLOPS, row 32, vs steady ~125 TFLOPS in
  iters 1-9). Using any single-iter or mean-with-iter-0 statistic would
  understate cuBLAS at small N and inflate the cuTile ratio. This doc uses
  median, which is robust.
- **No fp32 baseline.** The cuTile `mma_f32xf32_f32acc` kernel has no cuBLAS
  pairing in this CSV pair. Add `cublas-fp32` results to extend the table.
- **Headline "79%" from prior write-ups.** My median-based bf16/bgemm at
  N=4096 is 73.5%; depending on which iters and which summary statistic
  prior docs used, the "79%" is in the same neighborhood but not exact.
  Whichever summary one prefers, the takeaway is the same: it's not a
  stable cross-N number.

## Reproducing the medians

The numbers above are derivable in <30 lines of pandas:

```python
import pandas as pd
ct = pd.read_csv("cutile-matmul-tiled-mixed/results.csv")
cb = pd.read_csv("cublas-half-precision/results.csv").rename(columns={"N":"n"})
ct_med = ct.groupby(["kernel","n"])["tflops"].median().unstack()
cb_med = cb.groupby(["kernel","n"])["tflops"].median().unstack()
print("cutile medians:\n", ct_med)
print("cublas medians:\n", cb_med)
# ratios
pairings = [("mma_f16xf16_f32acc",  "hgemm"),
            ("mma_bf16xbf16_f32acc","bgemm"),
            ("mma_tf32xtf32_f32acc","sgemm_tf32")]
for ck, bk in pairings:
    print(ck, "/", bk, ":", (ct_med.loc[ck] / cb_med.loc[bk]).to_dict())
```
