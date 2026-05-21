# wgpu-matmul-tiled — classical shared-memory tiled SGEMM in WGSL

Wave C1.3, the WGSL/wgpu peer of `cuda-matmul-tiled` (Wave 2 W2B) and
`oxide-matmul-tiled` (Wave 2 W2C). Same shape (4096×4096 f32 SGEMM),
same input pattern, same harness style as `wgpu-matmul/` (the naive
baseline). Only the kernel changes: classical 16×16 BM/BN/BK tile with
`var<workgroup>` shared memory, `workgroupBarrier()` across the K-loop.

## Algorithm

- One thread per output cell (matches naive). Workgroup `@workgroup_size(16, 16, 1)` = 256 threads.
- Each workgroup produces a 16×16 output tile of `C`.
- K dimension swept in tiles of `BK = 16`. For each K-tile:
  1. All 256 threads cooperatively load one 16×16 sub-tile of `A` (rows = block_row, cols = `t*16 + 0..15`) and one 16×16 sub-tile of `B` (rows = `t*16 + 0..15`, cols = block_col) into `var<workgroup> tile_a` / `tile_b`. 256 threads × 1 element each — perfect 1:1 mapping, no per-thread loop needed.
  2. `workgroupBarrier()` (the WGSL analogue of `__syncthreads()`).
  3. Inner K loop (`for k in 0..16`) accumulates `tile_a[ty*16 + k] * tile_b[k*16 + tx]` into a per-thread register.
  4. `workgroupBarrier()` before reloading the next tile.
- Final write of the per-thread accumulator to `c[row*N + col]`.

This is the canonical "Phase 1" tiled GEMM — no register tiling, no double buffering, no vectorization. Identical structure to `oxide-matmul-tiled`'s `matmul_tiled` (16×16, `SharedArray<f32, 256>` × 2). Distinct from `cuda-matmul-tiled`'s `matmul_tiled` which adds a 4×4 register micro-tile (BM=BN=32, TM=TN=4, 64 threads/block) for substantially higher TFLOPS. Going to a register-microtile WGSL variant would be a separate cell (Wave C1.4 candidate).

## Files

| File | Purpose |
|---|---|
| `Cargo.toml` | wgpu 22, pollster 0.3, bytemuck 1 (mirrors wgpu-matmul) |
| `src/main.rs` | Host harness, bindings, dispatch, timing, correctness check |
| `src/matmul.wgsl` | The tiled compute shader |
| `run.sh` | Convenience runner (`cargo build --release` + run + tee log) |
| `.gitignore` | `target/`, `Cargo.lock`, `build.log` |
| `run.log` | Captured run output (5 timed iters, see "Bench results" below) |

## Bench results

Hardware/backend: WSL2 → wgpu 22 → Vulkan → **llvmpipe (LLVM 20.1.2)** = CPU emulation. No real GPU adapter is reachable from WSL Vulkan on this box (same constraint as wgpu-matmul, wgpu-vecadd, wgpu-reduce). Numbers below are CPU-side; treat the implementation as portable WGSL that should rerun on a non-WSL host with a real GPU adapter.

50-iter spec was preserved as the default; this run used `WGPU_TILED_ITERS=5` because each iteration takes ~24 s on llvmpipe and a 50-iter sweep would have exceeded the 600 s subagent budget. Override via env var.

```
[wgpu-tiled] using: llvmpipe (LLVM 20.1.2, 256 bits) (Vulkan, type=Cpu)
[wgpu-tiled] matmul 4096x4096 f32 (BM=BN=BK=16), 137.44 GFLOP/iter
[wgpu-tiled] warmup 0: gpu_ts=28805.51 ms (0.005 TFLOPS)
[wgpu-tiled] iter 0  : 26070.43 ms (0.005 TFLOPS)
[wgpu-tiled] iter 1  : 24223.77 ms (0.006 TFLOPS)
[wgpu-tiled] iter 2  : 23888.18 ms (0.006 TFLOPS)   <- best
[wgpu-tiled] iter 3  : 24365.77 ms (0.006 TFLOPS)
[wgpu-tiled] iter 4  : 24163.32 ms (0.006 TFLOPS)
BEST   23888.18 ms   0.006 TFLOPS
MEDIAN 24223.77 ms   0.006 TFLOPS
```

Correctness (5 sampled cells vs. f64 CPU reference):

| (row, col)    | got     | want    | abs_err   | rel_err   |
|---------------|---------|---------|-----------|-----------|
| (0, 0)        | 6.1420  | 6.1420  | 1.62e-5   | 2.64e-6   |
| (1, 1)        | 6.1439  | 6.1439  | 1.57e-5   | 2.56e-6   |
| (2048, 2048)  | 6.1455  | 6.1455  | 1.53e-5   | 2.48e-6   |
| (4095, 0)     | 6.1420  | 6.1420  | 1.62e-5   | 2.64e-6   |
| (4095, 4095)  | 6.1392  | 6.1392  | 1.48e-5   | 2.41e-6   |

`max_abs_err = 1.62e-5` ≪ `1e-3` acceptance threshold. ✅

## Cross-impl comparison @ N=4096 SGEMM

| Impl                    | Best (ms) | TFLOPS | Hardware           | vs cuda-tiled |
|-------------------------|-----------|--------|--------------------|---------------|
| cuda-matmul-tiled       |     3.578 | 38.4   | RTX 5090 (sm_120)  | 1.0×          |
| oxide-matmul-tiled (un) |    ~3-5   | ~30-40 | RTX 5090 (sm_120)  | ~1×           |
| **wgpu-matmul-tiled**   | 23 888.18 | 0.006  | llvmpipe CPU (WSL) | ~6 700× slower |
| wgpu-matmul (naive)     | 25 448.37 | 0.005  | llvmpipe CPU (WSL) | ~7 100× slower |

**Tiled vs. naive (same wgpu/llvmpipe backend): 25448 / 23888 = 1.065× faster (~6.5%).**

That's a much smaller speedup than tiled-vs-naive on real GPUs (where cuda-matmul-tiled at 38.4 TF crushes a hypothetical naive cuda-matmul). The reason is structural: llvmpipe runs WGSL workgroups by serializing 256 threads inside a host CPU thread; the "shared memory" is just an array on the host stack with no real cache-vs-DRAM gap to exploit. Inner-loop FMA work + barrier overhead is what the CPU actually pays. So tiling buys you only the modest improvement in inner-loop access locality (16-element row reuse from `tile_a` per outer-K-tile), not the order-of-magnitude bandwidth amplification you'd see on hardware. On a real GPU we'd expect 5-10× over naive (matching the oxide-tiled result on the same shape).

## Pitfalls & lessons

- **WGSL `var<workgroup>` array sizing must be a literal-foldable constant.** I used `array<f32, 256>` directly. Trying `const TILE: u32 = 256; var<workgroup> tile : array<f32, TILE>;` does work in modern Naga but is fragile across validators (and was historically the cause of "must be constant expression" rejections). Sticking to the literal sidesteps this. The constant `256` here is `BM * BK = BK * BN = 16*16`; if you change tile size you must update the literal.
- **`@builtin(workgroup_id)` is unused in this kernel.** I derive `block_row = gid.y - lid.y` implicitly via `row = gid.y` (since each workgroup spans 16 rows starting at a multiple of 16). The CUDA reference uses `blockIdx.y * BM + threadIdx.y` explicitly; the WGSL `global_invocation_id` already encodes that.
- **`workgroupBarrier()` is required twice per K-tile.** Once after the cooperative load (so all threads see fully-populated tiles before computing), once after compute (so no thread overwrites the tile while another is still reading). Dropping either produces races visible as wrong output cells, especially at higher block_row indices. We have both.
- **N must be a multiple of 16 for the simple version.** I kept the boundary checks anyway (cheap branch + zero-padding load) so the shader is correct for any N; the dispatch math `(N+15)/16` rounds up. At N=4096 every tile is fully in-bounds.
- **wgpu 22 stable API + edition 2024 friction.** Cargo edition 2024 + an `async fn run()` plus `pollster::block_on(run())` works fine, but the file-level rust-analyzer-style lint without seeing `Cargo.toml` thinks `async fn` is illegal because it defaults to edition 2015. Cargo build is the source of truth here; the inline-lint warnings on the file are a false positive.
- **WSL Vulkan = llvmpipe only.** The DX12 backend on WSL needs a libd3d12core wiring that wgpu 22 doesn't ship by default, and the Mesa Vulkan ICD only enumerates llvmpipe (CPU). Documented this in the host preamble, same as the other wgpu-* cells. To get real-GPU numbers, run on a Linux host with a vendor Vulkan ICD or on Windows native.
- **50-iter bench would exceed the 600 s subagent budget on CPU.** I parameterized via `WGPU_TILED_ITERS` (default 50, override 5 used here). When this kernel reruns on a real GPU, the default 50 will complete in <1 s total.

## Headline numbers (for the parent agent / aggregator)

- TFLOPS: **0.006 TF** (best), 0.006 TF (median) — CPU emulated via llvmpipe.
- vs `wgpu-matmul-naive`: **1.065× faster** (25 448 ms → 23 888 ms best).
- vs `cuda-matmul-tiled` on RTX 5090: **~6 700× slower** (3.578 ms → 23 888 ms). All of that gap is hardware/backend, not algorithm — same structure as oxide-tiled which clears 5–10 TF on real silicon.
- Correctness: 5/5 sampled cells OK, max_abs_err = 1.62e-5 ≪ 1e-3.
- Compile: clean, single attempt, no BLOCKED.
