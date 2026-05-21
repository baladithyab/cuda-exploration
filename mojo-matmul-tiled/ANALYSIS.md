# mojo-matmul-tiled — classical FFMA-tiled matmul in Mojo

**Wave C1.4** of the Rosetta Stone matmul-tiled cross-frontend column.

## What this cell is

The **classical FFMA tile** in Mojo: `BM=BN=BK=16`, one output element per
thread (256 threads/block, no register microtile), shared-memory staging via
`copy_dram_to_sram_async`, plain `*+` FFMA inner loop. **No `mma()`. No
TensorCore. No register microtile.**

This rounds out the matmul-tiled cross-frontend column alongside:

| Cell                       | Algorithm                              | TFLOPS @ 4096³ |
|----------------------------|----------------------------------------|----------------|
| cuda-matmul-tiled          | BM=BN=32, BK=16, TM=TN=4 reg microtile | **38.35**      |
| oxide-matmul-tiled         | (Rust port of cuda-tiled)              | (per cell)     |
| cutile-matmul-tiled        | (cuTile DSL port)                      | (per cell)     |
| wgpu-matmul-tiled          | (WebGPU port; concurrent task)         | (per cell)     |
| **mojo-matmul-tiled (here)** | **BM=BN=BK=16, 1 out/thread, FFMA**  | **7.69**       |

The Mojo lineage already has `mojo-matmul-bf16` (Wave 21) at **79.85 TF** doing
hand-rolled bf16 MMA — i.e. Mojo *does* have a tiled matmul, but that one is
MMA-based. This cell is the FFMA-only datapoint.

## Result @ 4096³

```
[mojo-matmul-tiled] M=N=K= 4096  dtype=f32 FFMA-only  BM= 16  BN= 16  BK= 16
  min_ms= 17.592735  median_ms= 17.866432  max_ms= 17.903743
  TFLOPS_median= 7.69  TFLOPS_best= 7.81
[mojo-matmul-tiled] correctness: max_abs_err= 0.0  max_rel_err= 0.0
[mojo-matmul-tiled] correctness PASSED at M=N=K= 4096
```

**1024-cell sampled correctness PASSED with max_abs_err = 0.0** — bit-exact
against the CPU reference because both kernels accumulate `f32 += f32 * f32`
in the same K-major order with no FFMA contraction differences.

## Cross-frontend ratios

| Comparison                               | This cell / Other |
|------------------------------------------|-------------------|
| **vs cuda-matmul-tiled (38.35 TF)**      | **0.201x** (≈5.0x slower) |
| **vs mojo-matmul-bf16 (79.85 TF MMA)**   | **0.096x** (≈10.4x slower) |

Both gaps are expected and informative:

- **5x vs cuda-matmul-tiled** is almost exactly the register-microtile factor:
  cuda-tiled does 4x4=16 outputs/thread, this does 1. The cuda-tiled kernel
  amortizes shared-memory loads across 16 FMAs/k-step, producing a denser
  arithmetic intensity at the register level.
- **10.4x vs mojo-matmul-bf16** is the **FFMA-vs-MMA delta**, *which is the
  point of this cell*. RTX 5090 sm_120 has dedicated bf16 tensor-core HMMA
  pipes that issue 2*128*16=4096 ops per cycle/SM, vs scalar FFMA at 128
  ops/cycle/SM. The expected ratio is ~16x at peak; we observe 10x because
  bf16 MMA is bandwidth-limited at this tile shape and the FFMA kernel is
  not at peak FFMA throughput either.

## Where the 7.7 TF goes (vs naive FFMA)

The naive Mojo matmul (`mojo-matmul`) at N=4096 reports **6.4 TF** in the
project headline tables. This cell adds:
- Shared-memory staging via `copy_dram_to_sram_async` (cp.async.bulk path)
- Per-K-block barrier
- comptime-unrolled inner loop

…for a +20% gain to **7.7 TF**. Without register microtiling, the smem path
trades fewer global loads for warp-level latency in the inner FFMA. The fact
that the gain is "only" 20% — instead of the 4-6x cuda-tiled gets vs
cuda-matmul — confirms the missing register tile is the dominant lever, not
the smem staging.

## Pitfalls encountered

1. **2D `block_dim=(BN,BM)` produces wrong answer.** First attempt used
   `block_dim=(16,16)` with `ty=thread_idx.y` / `tx=thread_idx.x`. The kernel
   compiled and ran but produced `got = ref/4` (1/4 of the expected sum),
   suggesting `copy_dram_to_sram_async` dispatched only 64 of 256 threads
   and loaded ¼ of the smem tile per K-pass. **Fix:** use flat
   `block_dim=(BLOCK_THREADS,)=(256,)` with `tid=thread_idx.x` and
   reconstruct `ty=tid//BN`, `tx=tid%BN`. The rest of the cuda-exploration
   Mojo lineage uses 1D block_dim consistently — this is the canonical form.

2. **`thread_layout=Layout.row_major(R, C)` constraint with BK=16.** Per the
   `mojo-gpu-kernels` skill: for `vectorize[1,4]`, inner-axis-of-vectors =
   `BK/4 = 4`. C must divide 4. With 256 block threads and the SKILL's
   `R*C ≈ block_threads/4 = 64` heuristic, the only viable option is
   `(16,4)`. That works.

3. **Bit-exact correctness.** With both A,B initialized via the same
   deterministic LCG pattern as `mojo-matmul`, the GPU and CPU references
   accumulate the K=4096 sum in the same order with the same rounding,
   giving max_abs_err = 0.0. Don't be alarmed by perfect zero — it's the
   correct outcome for K-major FFMA without contraction.

## Files

- `matmul_tiled.mojo` — kernel + main + bench harness
- `run.sh` — pixi launcher; calls `pixi run mojo matmul_tiled.mojo` from
  the workspace
- `run.log` — captured output of the most recent run

## Reproduce

```bash
cd /home/codeseys/cuda-exploration/mojo-matmul-tiled
bash run.sh 2>&1 | tee run.log
```
