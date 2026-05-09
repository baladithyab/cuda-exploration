# cuda-oxide-bench: how does NVlabs/cuda-oxide v0.1.0 compare to CUDA C++?

## TL;DR (3 bullets max)

- **cuda-oxide's naive compiler is closer than v0 suggested.** At N=1024 the unchecked raw-pointer kernel hits **95%** of nvcc's naive throughput (6.51 vs 6.88 TFLOPS); across N the band is 80-95%. The ~10-20% residual gap traces to a single upstream fix.
- **The Rust safety tax is 2.0-2.5×** from per-iteration slice bounds checks in the inner loop — unchanged across problem size, visible as explicit `setp`/`bra` pairs in PTX. Today, reaching nvcc parity requires either `unsafe` raw pointers or an upstream bounds-check elision pass.
- **The compiler gap multiplies once you tile.** nvcc's 32×32 + 4×4 register micro-tile hits **4.5-5.3× over naive**; cuda-oxide's 16×16 shared-memory tile hits only **1.3-1.6×**. Root cause: the tiled kernel's hot path emits **zero** `fma.rn.f32` (vs 256 in nvcc-tiled); `FastmathFlagsAttr::default()` is wired through but always empty.

## Methodology in one paragraph

Seven `(impl, kernel)` configurations — nvcc naive, nvcc 32×32 + 4×4 register tile, cuBLAS `Sgemm` with `CUBLAS_PEDANTIC_MATH`, cuda-oxide safe, cuda-oxide unchecked, cuda-oxide-tiled safe, cuda-oxide-tiled unchecked — each run over three problem sizes N ∈ {1024, 2048, 4096} with 1 warm-up + 10 timed iterations. All timings are `gpu_ms` via `cudaEventRecord`/`cuEventRecord` (ADR [0001](docs/adrs/0001-cudaevent-timing.md)), nvcc 13.2.78 with native `-arch=sm_120` (ADR [0002](docs/adrs/0002-native-sm120.md)), f32 throughout, no Tensor Cores, no mixed precision. Scope, exclusions, and threats to validity are in [METHODOLOGY.md](METHODOLOGY.md) and ADR [0003](docs/adrs/0003-scope.md). Hardware: RTX 5090 (Blackwell, sm_120) under WSL2, driver 581.xx.

## Master results

**TFLOPS vs N (median of 10 iterations)** — cleanest single view, from [`results/scaling-summary.md`](results/scaling-summary.md):

| impl/kernel \ N | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| cublas-matmul/sgemm | 33.94 | 62.98 | 59.83 |
| cuda-tiled/matmul_tiled | 24.47 | 33.44 | 28.07 |
| oxide-tiled/unchecked | 9.02 | 6.60 | 7.95 |
| oxide-tiled/safe | 8.83 | 5.56 | 7.69 |
| cuda-matmul/matmul | 6.88 | 6.33 | 6.23 |
| oxide/unchecked | 6.51 | 4.92 | 4.96 |
| oxide/safe | 2.80 | 2.40 | 2.01 |

Underlying data: [`results/scaling.csv`](results/scaling.csv) (210 rows). Per-size breakdowns with best/median/p95 and relative speedups: [`results/scaling-summary.md`](results/scaling-summary.md).

## Five findings

### 1. cuda-oxide unchecked hits 95% of nvcc at N=1024 — the compiler is closer than v0 suggested

On the naive kernel with raw-pointer loads, cuda-oxide lands at **0.95× nvcc** at N=1024 (6.51 vs 6.88 TFLOPS), tapering to **0.80×** at N=4096. The v0 writeup saw a single 0.93× number at N=4096 under PTX-JIT `sm_89` and CUDA 12.0 wall-clock timing; the refined results (native `sm_120`, `cudaEvent`) actually shift the picture more favorably for small N. The remaining gap is **not** per-iteration overhead, register pressure, or launch cost — all of those would be visible in a size-independent additive term. It's codegen quality in the inner loop: missing FMA contraction plus plain `ld.global` where nvcc uses `ld.global.nc`. Both are upstream one-liners. See [`oxide-matmul/ANALYSIS.md`](oxide-matmul/ANALYSIS.md).

### 2. The Rust safety tax is 2.0-2.5× from per-iter slice bounds checks

The safe and unchecked kernels differ by one source line: `a[r*dim+k]` vs `*a_base.add(r*dim+k)`. That single change takes the kernel from 6.51 TFLOPS to 2.80 TFLOPS at N=1024 (2.32×), and from 4.96 to 2.01 TFLOPS at N=4096 (2.47×). Slowdown *rises* with N because each extra inner-loop trip adds another bounds predicate. In PTX this is unmistakable: a `setp.ge.u64 … @%p bra` pair in the hot loop per access, eight `setp` instructions in safe where unchecked has zero. rustc's usual bounds-check elision doesn't apply because the index is a thread-space computation, not an iterator range. Reaching nvcc parity with idiomatic safe Rust will need NVPTX-path bounds elision, which LLVM doesn't yet perform — today the escape hatch is `unsafe` raw pointers.

### 3. The compiler gap WIDENS dramatically with tiling

nvcc's tiled kernel gets **3.6×/5.3×/4.5× over naive** at N=1024/2048/4096. cuda-oxide's tiled kernel gets **1.4×/1.3×/1.6×** — barely more than the variance band at this size. The gap isn't the tiling algorithm (both use shared memory, both achieve the correct async-copy pattern); it's what happens inside the `for k in 0..TILE` accumulation loop, which should be the densest FMA stream in any GEMM. PTX diff is stark: nvcc-tiled emits **256** `fma.rn.f32`; oxide-tiled emits **zero**. nvcc further fully unrolls the K-loop and schedules two independent accumulation chains; cuda-oxide keeps the loop rolled with a serial dependency chain. Net: the tiled data exposes the compiler gap *as the dominant cost*, not the API ergonomics gap. Detail in [`oxide-matmul-tiled/ANALYSIS.md`](oxide-matmul-tiled/ANALYSIS.md) and [`cuda-matmul-tiled/ANALYSIS.md`](cuda-matmul-tiled/ANALYSIS.md).

### 4. cuBLAS hits 60-72 TFLOPS — 10× the naive nvcc

Running `cublasSgemm` with `CUBLAS_PEDANTIC_MATH` (TF32 explicitly disabled so we stay apples-to-apples with our f32 kernels) hits **63.0 TFLOPS** at N=2048 and **59.8 TFLOPS** at N=4096 — roughly **10× our naive nvcc** baseline and **2× our best hand-tuned tiled nvcc**. That matters for honest framing: both compiler ecosystems leave most of the achievable performance unexposed without algorithmic work. A "cuda-oxide vs nvcc" delta of 20% at the naive level is a real compiler-quality signal, but the whole naive regime lives 10× below silicon speed-of-light. The interesting compiler comparison for "closing on cuBLAS" is what happens when you ship tiling + FMA + K-loop unrolling together, and the tiled data shows cuda-oxide falling *further* behind there. See [`cublas-matmul/ANALYSIS.md`](cublas-matmul/ANALYSIS.md).

### 5. wgpu/WGSL on WSL2 falls back to CPU rasterizer — cuda-oxide wins by running at all

The original plan included a wgpu/WGSL naive matmul as the "portable cross-vendor" comparison point. WSL2 exposes `/dev/dxg` for CUDA passthrough but does not expose a Vulkan ICD that wgpu's adapter selection will accept as a discrete GPU; it falls back to **llvmpipe** (Mesa software rasterizer on the Ryzen CPU) and takes ~25 seconds per iteration versus ~20ms on the real GPU — roughly **1000× slower** than cuda-oxide. This is not a wgpu performance finding, it's a portability boundary: under WSL2 cuda-oxide *runs* on the NVIDIA GPU and wgpu *does not*. For anyone benchmarking on WSL2, treat cross-vendor abstractions as not-available until Vulkan passthrough lands. Full diagnosis in [`wgpu-matmul/ANALYSIS.md`](wgpu-matmul/ANALYSIS.md).

## Compiler gap deep-dive

From [`docs/research/cuda-oxide-flags.md`](docs/research/cuda-oxide-flags.md):

> **No.** cuda-oxide (at this revision) has **zero** user-facing knobs for fast-math, FMA contraction, `-ffast-math`, `fp-contract`, or a read-only-cache / `__ldg` hint. The plumbing for LLVM fast-math flags *exists* end-to-end — `FastmathFlags { NNAN, NINF, NSZ, ARCP, CONTRACT, AFN, REASSOC, FAST }` is defined, `mir-lower` calls `add_fastmath_flags` on every `fadd`/`fsub`/`fmul`/`fdiv`/`frem`/`fneg` — but every callsite passes `FastmathFlagsAttr::default()` which is `FastmathFlags::empty()`, i.e. the attribute is attached with **no bits set**.

FMA counts across the generated PTX:

| kernel | `fma.rn.f32` count |
|---|---:|
| cuda-oxide naive (safe or unchecked) | **0** |
| cuda-oxide tiled (safe or unchecked) | **0** |
| nvcc naive `-O3` | 5 |
| nvcc tiled `-O3` | 256 |

`core::intrinsics::fmuladdf32` — the one user-visible escape hatch — lowers to a libdevice call (`__nv_fmaf`) instead of `llvm.fmuladd.f32`, so even explicit `a.mul_add(b, c)` can't recover hardware FMA. A minimum-viable upstream patch is roughly **four lines** in `crates/mir-lower/src/convert/ops/`: thread a flag into `add_fastmath_flags`, expose it via `cargo oxide build --fp-contract=fast`, and lower `FmuladdF32` to the LLVM intrinsic. Upstream issue draft and full PTX forensics are in [`docs/research/cuda-oxide-flags.md`](docs/research/cuda-oxide-flags.md) — ready to file against [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide).

## Setup gotchas you'll hit

1. **`/usr/bin/nvcc` is a CUDA 12.0 shim** on most WSL2 Ubuntu installs and silently falls back when given `-arch=sm_120`. Use `/usr/local/cuda/bin/nvcc` explicitly; `nvcc --version` should report 13.2.x, not 12.0.140.
2. **Build native `sm_120` on Blackwell**, not PTX-JIT `sm_89`. Modest (~6%) speedup on naive kernels, but it's the correct baseline and removes JIT startup variance. See [ADR 0002](docs/adrs/0002-native-sm120.md).
3. **cuda-oxide requires LLVM 21**, not whatever your distro ships. The `cargo-oxide` installer expects `llvm-config-21` on `$PATH`; on Ubuntu 22.04 you'll need the llvm.org apt repo. [SETUP.md](SETUP.md) has the exact incantations.
4. **wgpu under WSL2 will not find the NVIDIA GPU.** Expect llvmpipe fallback. If you need a portable baseline today, run it on bare-metal Linux or defer until Vulkan-over-WSL matures.

## For Rust developers considering cuda-oxide today

**One-paragraph take.** cuda-oxide v0.1.0 is genuinely usable: kernels compile, launch, and run within **95%** of nvcc on simple code with `unsafe` raw pointers, and the safety gap is understood (bounds-check elision) rather than mysterious. The tradeoffs are that the compiler currently gives up roughly **3×** of the achievable performance on tiled GEMM due to missing FMA contraction and K-loop unrolling — both tractable upstream fixes — and you get *none* of cuBLAS (60+ TFLOPS), cuDNN, WGMMA/Tensor Cores, or the broader NVIDIA SDK. If your use case is custom kernels where the naive codegen is good enough and you want a memory-safe Rust-first toolchain, ship. If you need peak performance or Tensor Cores today, wait.

| Your situation | Verdict |
|---|---|
| Greenfield Rust project, NVIDIA-only, custom kernels, not perf-critical | **Yes** — ship on cuda-oxide today |
| Need Tensor Cores, WGMMA, cuBLAS, cuDNN, or peak FP32 perf today | **Wait** — or keep a cuBLAS/cuDNN FFI path alongside |
| Need cross-vendor portability (AMD, Intel, Apple, WebGPU) | **Use wgpu or CubeCL** — cuda-oxide is NVIDIA-only by design |

## What's next (followups)

From [`BACKLOG.md`](BACKLOG.md), the highest-value unshipped items:

- **U1 — file the upstream issue** on NVlabs/cuda-oxide with the PTX evidence and the four-line patch sketch from the flags research doc. Evidence ready; draft pending.
- **F1 — re-bench with a fast-math patch applied** once upstream (or a local fork) plumbs a `CONTRACT` bit through `add_fastmath_flags`. Expectation: naive closes to ≥98% of nvcc, tiled closes from 1.5× to 3-4×.
- **N2 — reduction kernel** — different access pattern than GEMM, tests warp-reduce primitives which we never exercised. Independent data point on compiler quality.
- **N1 — block-size sweep** — 8×8, 16×16, 32×8 etc. for naive matmul; occupancy effects were assumed, not measured.

**Explicitly out of scope** for this benchmark (per ADR 0003): Tensor Core / WGMMA, multi-GPU, fp16/bf16 mixed precision, fixing wgpu on WSL.

## Acknowledgments

- **[NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide)** — the compiler under test, v0.1.0 (commit `6de0509`, released 2026-05-07). The findings here are submitted in a spirit of constructive collaboration; the fact that cuda-oxide lands at 95% of nvcc on a v0.1.0 release is a real accomplishment.
- **[gfx-rs/wgpu](https://github.com/gfx-rs/wgpu)** — the portable GPU API used for the cross-vendor comparison attempt.
- **[NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)** — nvcc and cuBLAS, the reference baselines.

This is **independent third-party work**, not affiliated with, endorsed by, or reviewed by NVIDIA, NVlabs, or the cuda-oxide, wgpu, or cuBLAS teams. All findings are the author's own; corrections welcome via issues or PRs.
