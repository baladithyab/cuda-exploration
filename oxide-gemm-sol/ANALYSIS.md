# Wave 6 W6A — cuda-oxide `gemm_sol` example on RTX 5090 (sm_120)

## What the example is

Upstream cuda-oxide ships a "speed-of-light" GEMM example targeting **datacenter
Blackwell (sm_100, B200/H200)**. It contains **eight kernel variants** in a
single 6,400-line `src/main.rs`, progressively layering Blackwell-only features:

| Phase | Name                                     | Headline feature                                      |
|-------|------------------------------------------|-------------------------------------------------------|
| 1     | `gemm_sol_tiled`                         | K-loop + grid tiling, 8 tiled TMA copies / K-tile     |
| 1.5   | `gemm_sol_swizzled`                      | `SWIZZLE_128B`, single TMA copy per matrix            |
| 2     | `gemm_sol_pipelined`                     | Double-buffered SMEM, TMA/MMA overlap                 |
| 3     | `gemm_sol_warp_spec`                     | Warp-specialized producer/consumer (6 warps)          |
| 4A    | `gemm_sol_persistent`                    | Persistent CTAs + 2-stage TMEM accum pipeline         |
| 4B    | `gemm_sol_clc`                           | CLC hardware tile-scheduling (replaces atomic)        |
| 4C    | `gemm_sol_clc_multicast`                 | CLC + TMA multicast for B                             |
| 4D    | `gemm_sol_clc_multicast_4_stage_pipeline`| `cta_group::2` pair-UMMA + 4-stage SMEM pipeline      |

All variants use f16 inputs, bf16 C output, f32 TMEM accumulation. All depend
on `tcgen05` (5th-gen tensor cores), `cp_async_bulk_tensor` (TMA), mbarriers,
clusters, CLC (cluster launch control), and pair-UMMA — a tight stack of
sm_100-specific instructions. On B200 upstream reports **868 TFLOPS at 4K**
(Phase 4D, 57.8% of cublasLtMatmul SoL).

## Build status — ✅ succeeded

`cargo oxide build oxide-gemm-sol` completes cleanly in ~69s with 140 warnings
(all of them the benign "`SharedArray` lowered to per-block shared memory"
note). `oxide_gemm_sol.ptx` is produced (216 KB). We symlinked it to
`gemm_sol.ptx` because `main.rs` hard-codes that filename via
`env!("CARGO_MANIFEST_DIR")`.

Toolchain fix needed during setup: copied `rust-toolchain.toml`
(`nightly-2026-04-03`) from `oxide-matmul`; the upstream example omitted it
because it lived in the cuda-oxide workspace.

## Run status — ❌ cannot execute on sm_120

```
GPU: sm_120
Loading PTX: .../gemm_sol.ptx

⚠️  tcgen05 (5th gen tensor cores) requires sm_100 (datacenter Blackwell only).
   Your GPU is sm_120 (consumer Blackwell has no tcgen05).
   PTX was generated successfully; run on sm_100 to execute kernels.
```

`ctx.load_module_from_file` returns `CUDA_ERROR_INVALID_PTX`. The example
correctly detects this and falls through to `verify_ptx_only()`. **No kernel
from any of the eight phases runs.** This is not a cuda-oxide bug and not
a sm_120 codegen bug — it is an architectural fact: consumer Blackwell
(GeForce RTX 50-series / sm_120) ships without `tcgen05` units and without
tensor memory (TMEM), so the PTX cannot be JIT-compiled at module load.
The RTX 5090's tensor cores are 4th-gen (Hopper-class `wgmma`/`mma`), not
5th-gen `tcgen05`.

## Comparison to our other benches

| Bench                              | N=4096 TFLOPS on RTX 5090 | Precision             |
|------------------------------------|---------------------------|-----------------------|
| oxide naive (Wave 1)               | ~3 TFLOPS                 | f32                   |
| oxide tiled (Wave 2)               | ~8 TFLOPS                 | f32                   |
| cuBLAS sgemm                       | ~60 TFLOPS                | f32                   |
| **oxide gemm_sol (this run)**      | **N/A (sm_100 required)** | f16 in / bf16 out     |
| cuBLAS hgemm (not benched)         | ~180–240 TFLOPS (est.)    | f16                   |
| Upstream B200 Phase 4D             | 868 TFLOPS                | f16 / bf16            |

## Caveat on TFLOPS comparability

`gemm_sol` is **f16 → bf16 with f32 accumulation**, a fundamentally different
precision class than our f32 matmul benches. Even if it had run, direct
TFLOPS comparison to `cuBLAS sgemm` or our naive/tiled kernels would be
unfair — f16 tensor-core throughput on RTX 5090 is ~3–4× f32 cuBLAS peak.
The apples-to-apples comparison is **`gemm_sol` vs `cublasLtMatmul` with
f16 inputs + f32 compute**, which we have not benched here. Upstream's
868 TFLOPS / 57.8% SoL is the right reference frame.

## Takeaway

The negative result is still informative: **cuda-oxide's flagship GEMM
example targets data-center Blackwell only**, and the "sophisticated TMA +
WGMMA + CLC" story in the README is really a **`tcgen05` + CLC story that
requires sm_100**. For consumer Blackwell (sm_120, RTX 50-series) the
peak-perf ceiling we can realistically reach via cuda-oxide is gated by
(a) whether upstream adds a Hopper/consumer-Blackwell `wgmma` or
sm_120-compatible MMA path, or (b) whether we port these kernels to the
`mma.sync` instruction family ourselves. The PTX generation half of the
pipeline works — build, codegen, and module packaging are sound — it's
the hardware primitives that are missing on our device.
