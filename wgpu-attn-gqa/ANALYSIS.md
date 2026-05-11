# wgpu-attn-gqa — Analysis

**Wave 16.2 — wgpu/WGSL data point for the Grouped-Query Attention cross-frontend
comparison.** Honest up-front: on this WSL2 host the wgpu→NVIDIA path is still
dead (same as Waves 1-11 and the existing `wgpu-matmul/` cell). The WGSL kernel
itself is a correct, portable artifact; the TFLOPS numbers below are llvmpipe
running on 24 CPU cores and carry **zero GPU-compute signal**.

## 1. Headline

**Adapter detected: `llvmpipe (LLVM 20.1.2, 256 bits)`, type = `Cpu`, backend = Vulkan (also GL).**
Exactly matches the Wave 1-11 finding. There is no usable NVIDIA Vulkan ICD in
WSL2, the DX12 backend does not enumerate the GPU without Microsoft's Dozen ICD,
and the GL backend also reports the CPU llvmpipe adapter. Full details in
`../wgpu-matmul/ANALYSIS.md §3` and `~/.hermes/skills/mlops/rust-gpu-compute/SKILL.md`
("wgpu/WGSL on WSL2 — known dead end").

Correctness (small shape, against PyTorch SDPA reference at f32):

```
[correctness] B=1 Nq=4 Nkv=2 S=128 D=64 groups=2
[correctness] correctness: max_abs_err=1.341e-7  rel_err=6.202e-7  atol=1e-5  OK
```

~75× margin under the f32 tolerance. The WGSL kernel is algorithmically correct.

Llvmpipe throughput (CPU, **not GPU**, included for completeness only):

| shape       | total | QKᵀ    | softmax | PV     | "TFLOPS" |
|-------------|------:|-------:|--------:|-------:|---------:|
| correctness | 6.4 ms| 1.7 ms (26%) | 2.8 ms (43%) | 1.7 ms (25%) | 0.0026 |
| medium_512  | 82.8 ms | 31.3 ms (37%) | 18.5 ms (21%) | 32.6 ms (41%) | 0.0065 |
| llama3_8b   | — SKIP — scores buffer 512 MiB exceeds llvmpipe's 128 MiB storage-binding cap |

## 2. Methodology

- wgpu 22.1, Vulkan+GL+DX12 backends enabled (`features = ["dx12"]`), Naga
  translating WGSL to the backend's native IR.
- 3-pass naive GQA, f32 throughout:
  1. `gqa_qkt`: `scores[b,h_q,i,j] = scale · Σ_d Q[b,h_q,i,d]·K[b,h_kv,j,d]`
  2. `softmax`: numerically-stable row-wise (subtract rowmax, exp, divide by rowsum)
  3. `gqa_pv`:  `out[b,h_q,i,d] = Σ_j probs[b,h_q,i,j]·V[b,h_kv,j,d]`
- GQA broadcasting done inline in WGSL: `h_kv = h_q / groups` where
  `groups = n_q / n_kv`. No tensor expansion, no memory copies — just index math.
- Workgroup sizes: 16×16 for the two matmul passes (tile the (i,j) or (i,d)
  output), 256 for softmax (one workgroup per row, tree-reduced rowmax and
  rowsum through `workgroupBarrier`).
- `TIMESTAMP_QUERY` feature was available on llvmpipe — per-pass ns-resolution
  timings from `wgpu::ComputePassTimestampWrites`, not CPU wall-clock. A CPU
  wall-clock number is printed side-by-side for sanity (consistently within
  ~1 ms of the GPU-ts total).
- Inputs loaded from `../analysis/wave15-attention-architecture/inputs/*.npy`
  via the `npyz` crate (f32 C-contiguous). Output compared element-wise to
  `gqa_<shape>_expected_f32.npy` with the shared f32 tolerance (atol=rtol=1e-5
  from `reference/tolerances.py`).
- 1 warmup + N timed iters (2 for correctness, 3 for medium_512).

## 3. Why llama3_8b skipped: wgpu storage-binding cap

The canonical wave15 bench shape (B=1, S=2048, Nq=32, Nkv=8, D=128) needs a
`scores` tensor of shape (1, 32, 2048, 2048) = **512 MiB at f32**. llvmpipe's
adapter advertises `max_storage_buffer_binding_size = 128 MiB`, and wgpu rejects
a device request that exceeds the adapter's own cap. Requesting more ICEs as
`LimitsExceeded(FailedLimit { name: "max_storage_buffer_binding_size", requested: 1073741824, allowed: 134217728 })`.

This is a **wgpu/llvmpipe portability constraint**, not an attention-specific one.
Two ways to lift it on a real run:

a. **A real GPU adapter**: NVIDIA Vulkan typically reports 2 GiB+ for this limit,
   at which point the bench shape fits trivially.
b. **Per-head scores tiling** (or FlashAttention-style fusion, where the entire
   (seq×seq) tile never exists in DRAM at all). The wave15 `cutile-attn-gqa`
   cell's 165 TFLOPS *from* this exact property — keep scores in registers,
   skip the HBM round-trip entirely.

We added an intermediate `medium_512` shape (B=1, Nq=8, S=512, D=64) with
8 MiB scores to at least exercise the kernel at a non-toy size. It's perf-only
(synthetic inputs, no golden reference); the correctness pass on S=128 covers
the "does the kernel compute the right thing" question.

## 4. Where the time goes (llvmpipe only)

The per-stage split is itself interesting even on CPU:

```
correctness: qkt 26%  sm 43%  pv 25%   (softmax dominates — seq=128, D=64, compute-light)
medium_512:  qkt 37%  sm 21%  pv 41%   (matmuls dominate — seq=512 scales Sᵈ faster than S in softmax)
```

On a real GPU the profile would look very different. Wave 15's cuBLAS-3-kernel
cell at llama3_8b ran 31% / 31% / 38% (QKᵀ / softmax / PV) — *softmax took
as much wall-clock as either GEMM*, because it's HBM-bandwidth-bound at
S=2048, hitting ~70% of HBM peak. That's the FlashAttention argument: on a
real GPU the 3-kernel pipeline spends ~30% of its time moving the seq×seq
matrix through HBM that a fused kernel keeps in registers. None of this
translates on llvmpipe, where bandwidth *is* compute and there's no HBM.

## 5. Comparison to Wave 15 cells (headline only, llama3_8b shape)

| frontend                         | backend           | TFLOPS  | ratio vs cuBLAS hgemm (218) |
|----------------------------------|-------------------|--------:|---------------------------:|
| `cutile-attn-gqa` (fused FA-2)   | NVIDIA PTX/cuTile | 165.14  | 76%                        |
| `cublas-attn-gqa` (3-kernel)     | cuBLAS + custom   |  46.38  | 21%                        |
| `cuda-attn-gqa` (nvcc WMMA)      | CUDA C++          |  23.47  | 11%                        |
| **`wgpu-attn-gqa` (this, WSL2)** | llvmpipe / CPU    | **not comparable** — 0.0065 TFLOPS on CPU, not GPU |

The right comparison is "what does cuTile's fused kernel buy over a portable
cross-vendor stack?" — and on this host the answer is "we can't say; wgpu has
no GPU here". On a bare-metal Linux box with `libnvidia-vulkan`, or on Windows
with wgpu DX12, this cell would produce a real number. Published wgpu/Vulkan-
compute benchmarks on naive kernels typically trail vendor-compute APIs by
5–10×; for GEMM-heavy attention with no Tensor-Core path in WGSL (f32 only,
naive loops) you would expect a larger gap still — maybe 50–100× off cuTile's
165 TF, putting it in the 1–3 TF range. That is a guess, not a measurement.

## 6. The WGSL kernel itself (portable artifact)

`src/attn.wgsl` is the main deliverable. It:

- Uses only WGSL core features — no extensions, no SHADER_F16. Runs on any
  WebGPU-compliant backend: Vulkan, DX12, Metal, WebGPU-in-browser.
- Encodes GQA broadcasting in plain index arithmetic (`h_kv = h_q / groups`),
  no repeat/expand. Mirrors what the `cuda-attn-gqa` and `cublas-attn-gqa`
  cells do in CUDA C++.
- Does a numerically-stable row-wise softmax with explicit `workgroupBarrier()`
  between reduction phases. Tree-reduce over `WG_SOFTMAX = 256` threads.
- f32 throughout. WGSL f16 (`enable f16;`) is gated behind `wgpu::Features::
  SHADER_F16`, which is available on some backends but absent on llvmpipe and
  patchy on mobile. For a portability artifact, f32 is the safer call.

The Rust host (`src/main.rs`) reuses the `wgpu-matmul` pattern: enumerate all
adapters, prefer non-CPU, loud-log when we fall back to llvmpipe, request
TIMESTAMP_QUERY only if the adapter offers it (wgpu 22 will error out otherwise).
The only wgpu-22-specific quirk vs the skill's note is `Instance::new(desc)` —
in 22.1 it takes the descriptor **by value**, not by reference (contradicting
one of the skill's notes; value-form matches the existing `wgpu-matmul/`).

## 7. Pitfalls captured (new, in addition to the `wgpu-matmul` set)

- **wgpu 22.1 `Instance::new` takes `InstanceDescriptor` by value, not `&`.**
  Contradicts the skill's note; verified against `wgpu-22.1.0/src/lib.rs:2336`.
- **Adapter-reported limits are a hard ceiling** — `adapter.request_device` will
  reject any `required_limits` above the adapter's advertised cap. Raising
  `max_storage_buffer_binding_size` above the adapter's stated value *does not*
  work, even as a hint. Llvmpipe's 128 MiB cap is fatal for llama3_8b attention.
- **llvmpipe does support `TIMESTAMP_QUERY`** — convenient for per-pass splits
  even on CPU. Unlike some drivers that silently clamp the resolution, llvmpipe
  returns honest ns timings; the GPU-ts total vs CPU-wall total agree to ~1 ms.
- **`npyz` crate** is the lightest `.npy` reader for pure-Rust hosts — ~30 LOC
  of setup for this use case. Handles both C- and F-contiguous, but we only
  support C (wave15 inputs are C-contiguous by construction).
- **Zink EGL warnings on startup are harmless noise on WSL2** — Mesa's Zink
  (OpenGL-on-Vulkan) probes `/dev/dri/*` for a Vulkan ICD, fails, and logs. This
  does not affect wgpu's adapter enumeration.

## 8. Reproducing

```bash
cd /home/codeseys/cuda-exploration/wgpu-attn-gqa
cargo run --release 2>&1 | tee run.log
```

Expect ~30 seconds total on this WSL2 host (most of it in llvmpipe's
medium_512 pass). On a real GPU the same binary will pick the hardware
adapter, skip the "CPU-only" warnings, and produce a real number — probably
in the 1–10 TFLOPS range for this naive 3-pass implementation on consumer
Blackwell, *without* tensor cores. FlashAttention-style fusion would need
either wgpu's SHADER_F16 + workgroup shared memory tiling (an hour of WGSL
work) or a different frontend (the `cutile-attn-gqa` cell, which hits 165 TF
via the NVIDIA-native path and is the point of comparison).
