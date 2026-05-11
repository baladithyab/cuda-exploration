# Wave 16 — Multi-platform attention: 2 mechanisms × 4 frontends

**Date:** 2026-05-11. Continuation of Wave 15 (GQA × 3 frontends MVP).
Wave 16 fans across **all the previously-tested platforms** (added cuda-oxide
and wgpu) and **adds the next two attention mechanisms** (MLA and GDN-decode)
recommended by the Wave 15 research. Four parallel implementation subagents.
Orchestrator independently re-ran each on idle GPU in same session and verified
SASS-level claims before commit.

## Headline @ canonical bench shapes (RTX 5090 sm_120, idle GPU)

### GQA — Llama-3-8B (B=1, S=2048, n_q=32, n_kv=8, d=128) — 4 frontends

| impl | best ms | metric | vs cuBLAS hgemm 218 TF | HMMA |
|---|---:|---:|---:|---:|
| `cutile-attn-gqa` (cuTile fused) | 0.416 | **165.14 TF** | **75.7%** | 256 |
| `cublas-attn-gqa` (cuBLAS-3-kernel) | 1.482 | 46.38 TF | 21.3% | (cuBLAS) |
| **`oxide-attn-gqa` (cuda-oxide f32, no-TC)** | **2.846** | **24.15 TF** | 11.1% | **0** |
| `cuda-attn-gqa` (nvcc 3-kernel WMMA) | 2.928 | 23.47 TF | 10.8% | 20 |
| `wgpu-attn-gqa` (WSL2 → llvmpipe CPU) | 6391 | 0.0026 TF | 0.001% | (CPU) |

### MLA — DeepSeek-V3 (B=1, S=2048, n_h=128, qk=192-pad-256, d_v=128) — cuTile

| impl | best ms | TFLOPS | vs cuBLAS hgemm | vs cuTile-GQA 165 TF | HMMA |
|---|---:|---:|---:|---:|---:|
| `cutile-attn-mla` (cuTile fused) | 3.057 | **112.4** | 51.6% | 68.1% | 384 |

### GDN-decode — Qwen3-Next (B=1, H=16, d_k=d_v=256) — cuTile

| impl | best µs | best GB/s | % HBM peak (1792 GB/s) | HMMA |
|---|---:|---:|---:|---:|
| `cutile-attn-gdn` (cuTile fused decode) | **13.79** | **610.6** | **34.1%** | **0** |

## Three findings that reframe the Wave 16 story

### 1. **Surprise**: cuda-oxide ties nvcc-WMMA at f32 GQA

`oxide-attn-gqa` (24.15 TF, **HMMA=0**, FFMA=157) **ties** `cuda-attn-gqa`
(23.47 TF, HMMA=20). Despite no tensor cores, cuda-oxide's 4×4 register
microtile pattern (Wave 7's `oxide-matmul-tiled-microtile`, 45 TF f32)
exactly offsets the WMMA cell's tiny 16×16 tiles with no shared-mem tiling.

Per-stage: oxide qkt=1.34ms (42%) sm=0.75ms (23%) pv=1.12ms (35%). The 0.7-ms
softmax HBM round-trip dominates the gap to fused (cuTile fused has 0
softmax-stage HBM cost and reaches 165 TF, the same 6.8× ratio observed
in Wave 15.1).

This is a real demonstration of the **FlashAttention thesis** — the gap
between unfused 3-kernel and fused single-kernel has nothing to do with
TC engagement and everything to do with the (seq²) attention-matrix HBM
round-trip. Both nvcc-WMMA and oxide-no-TC hit the same ~24 TF ceiling
for unfused attention because softmax bandwidth, not matmul throughput,
is the bottleneck.

### 2. **wgpu confirmed dead-end on WSL2** (not a regression — Wave 1-11 finding holds)

`wgpu-attn-gqa` enumerates only `llvmpipe (LLVM 20.1.2)` device_type=Cpu.
No NVIDIA Vulkan ICD passthrough; DX12 backend doesn't enumerate without
Dozen wiring. Same conclusion as Wave 1-11 wgpu-matmul.

The WGSL kernel itself is **a portable correctness artifact**: passes
PyTorch reference at correctness shape with max_abs=1.34e-7, 75× margin
under the f32 tolerance. The 0.0026 TFLOPS number on WSL2 carries zero
GPU-compute signal — this cell exists for completeness and as a wgpu/WGSL
implementation reference for non-WSL2 systems.

Bonus pitfall: `llama3_8b` shape skipped because llvmpipe caps
`max_storage_buffer_binding_size` at 128 MiB (the (seq, seq) scores buffer
for n_q=32 needs 512 MiB). On real NVIDIA Vulkan the cap is typically 2 GiB
so the bench would run; this is a wgpu-portable-limit issue specific to
software adapters.

### 3. **cuTile generalizes to non-attention regimes**

cuTile's attention story extends from softmax attention (GQA, MLA) to
**linear-attention recurrent decode** (GDN). The kernel uses none of cuTile's
matmul-shaped primitives — outer products, gated state updates, scalar
FMA — and **no primitive fell over**. 485 LOC for a fused single-kernel
GDN-decode that hits 34% of HBM peak.

This is the strongest evidence yet that cuTile is a **general-purpose tile
DSL**, not just a "fused attention DSL". The Wave 13 mixed-precision finding
extended to fused attention (Wave 15); now Wave 16 extends it to recurrent
state machines.

## Independent verification (orchestrator re-ran each cell on idle GPU)

All four cells reproduced their headline numbers within bench noise:

```
$ ./oxide-attn-gqa/target/release/oxide-attn-gqa  → best 24.15 TF ✓
$ python cutile-attn-mla/main.py --bench           → best 112.4 TF ✓
$ python cutile-attn-gdn/main.py --bench           → best 610 GB/s ✓
$ wgpu-attn-gqa/target/release/wgpu-attn-gqa       → llvmpipe CPU, 0.0026 TF ✓
```

SASS-level TC-engagement checks:

```
$ /usr/local/cuda/bin/cuobjdump --dump-sass oxide-attn-gqa/oxide_attn_gqa.cubin | grep -c HMMA   → 0  ✓
$ grep -c HMMA cutile-attn-mla/mla_fwd_fused.sass                                                 → 384 ✓
$ grep -c HMMA cutile-attn-gdn/gdn_decode_fused.sass                                              → 0  ✓ (expected for decode)
```

## Correctness vs PyTorch reference (all cells)

- **GQA correctness shape** (b=1, s=128, n_q=4, n_kv=2, d=64):
  - cuda: max_abs=1.559e-04 OK
  - cublas: max_abs=1.559e-04 OK
  - cutile: max_abs=1.300e-04 OK
  - **oxide: max_abs=1.192e-07 OK** (f32 throughout, much tighter than f16 cells)
  - **wgpu: max_abs=1.341e-07 OK** (f32 + correct WGSL, llvmpipe CPU)

- **MLA correctness** (deepseek-v3 small): cutile max_abs=1.5e-04 OK
- **GDN correctness**: cutile max_abs=3.05e-05 (output), 2.98e-08 (state), OK

## Pitfalls captured in Wave 16

- **`cargo oxide build --release` doesn't exist** — the subcommand has its own flag
  set. Use `cargo oxide build --arch sm_120` instead.
- **cuda-oxide helper-fn `&mut DeviceBuffer<f32>` params** need an inner
  rebind (`let mut scores_dev = scores_dev;`) before `slice_mut()` inside
  `cuda_launch!` — the macro reborrows twice via deref coercion.
- **wgpu 22.1 `Instance::new` takes `InstanceDescriptor` by value**, not `&` —
  contradicts an earlier note in the rust-gpu-compute skill; matches existing
  wgpu-matmul usage. Skill updated.
- **Adapter-reported limits are a hard ceiling** in wgpu — `request_device`
  fails with `LimitsExceeded` if `required_limits` exceed adapter cap. No
  "request higher as a hint" path.
- **GDN-decode bench needs 50+ iters** (vs 10 for matmul-shape benches) —
  per-iter is 14 µs so cudaEvent variance is high; report median + IQR.
- **MLA's 192-padded-to-256 QK dim** wastes 25% of the QK^T flops vs an
  optimal split-and-combine implementation. Documented; Wave 16.5+ candidate.
- **MLA softmax scale must use TRUE qk_head_dim=192**, not QK_PAD=256.
  Caught by tight smoke-shape correctness check.
- **f32 attention in cuda-oxide hits relative-error >1e-2 on near-zero outputs**
  even when atol passes at 1e-7. Use atol-OR-rtol logic (numpy.allclose
  default), not atol-AND-rtol. Documented in oxide-attn-gqa/ANALYSIS.md.

## Files added in Wave 16

```
analysis/wave15-attention-architecture/reference/   (extended)
  shapes_mla.py, flops_mla.py, pytorch_reference_mla.py
  shapes_gdn.py, flops_gdn.py, pytorch_reference_gdn.py
  README.md (updated with MLA + GDN sections)

oxide-attn-gqa/
  src/main.rs (~870 LOC, 3-kernel naive f32 register-microtile)
  Cargo.toml, Cargo.lock, rust-toolchain.toml
  results.csv, run.log
  ANALYSIS.md (~180 lines)

wgpu-attn-gqa/
  src/main.rs (573 LOC) + src/attn.wgsl (205 LOC)
  Cargo.toml, Cargo.lock
  ANALYSIS.md (170 lines, the WSL2 dead-end honest writeup)
  run.log

cutile-attn-mla/
  main.py (~500 LOC, fused FA-2 over MLA shape, BLOCK=64×64, QK_PAD=256)
  run.sh, .gitignore
  results.csv, run.log, smoke.log, bench.log
  mla_fwd_fused.sass (1.9 MB, evidence for HMMA=384)
  ANALYSIS.md (~100 lines)

cutile-attn-gdn/
  main.py (~485 LOC, fused 1-timestep recurrent decode)
  run.sh, .gitignore
  results.csv, run_smoke.log, run_bench.log
  gdn_decode_fused.sass (1.3 MB, evidence for HMMA=0)
  ANALYSIS.md (~150 lines, BW-bound regime writeup)

results/wave16-summary.md (this document)
```

## Cross-wave attention-comparison table (the master view)

After Wave 15 and Wave 16, here's what the repo demonstrates across attention
mechanisms × frontends:

| | nvcc | cuda-oxide | cuTile | cuBLAS | wgpu |
|---|---|---|---|---|---|
| **GQA** (Llama-3 shape) | 23.5 TF | **24.2 TF** | **165.1 TF** | 46.4 TF | (CPU only) |
| **MLA** (DeepSeek-V3) | — | — | **112.4 TF** | — | — |
| **GDN-decode** (Qwen3-Next) | — | — | **610 GB/s** | — | — |

(All cuTile fused single-kernel; nvcc/cuda-oxide/cuBLAS are 3-kernel naive.)

cuTile dominates because it can *fuse*; the other frontends would need
hand-rolled FlashAttention ports (~1000 LOC nvcc, currently scoped as
W16.7+ for the multi-mechanism story). cuda-oxide additionally lacks any
TC API on consumer Blackwell (Wave 14.4 finding).

## Wave 17+ candidates (in priority order)

- **W17.1 cuda-oxide MLA + GDN** — the f32 no-TC ceiling for non-GQA mechanisms
- **W17.2 cuTile KDA-decode** — vector-gate variant of GDN (~30-LOC delta over
  existing GDN-decode kernel; near-free given GDN works)
- **W17.3 cuTile GQA bigger tiles** — close 24% gap to cuBLAS hgemm peak
- **W17.4 nvcc fused FlashAttention reference** — only the fused entry would
  let nvcc compete with cuTile on attention; ~800 LOC
- **W17.5 cuBLAS-3-kernel MLA + GDN** — the 3-kernel ceiling for non-GQA
- **W17.6 GDN prefill (chunkwise)** — 1000+ LOC, the WY/triangular-solve version
- **W17.7 DSA / CSA / HCA** — DeepSeek sparse mechanisms; deferred per Wave 15
  research because they need FP4 + on-device top-k + sparse-gather (likely
  nvcc-only first)
