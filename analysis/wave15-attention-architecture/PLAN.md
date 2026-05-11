# Wave 15 — Attention kernels architecture plan

**Status:** architecture / scoping proposal (pre-implementation).
**Scope goal:** 6 mechanisms × 4 frontends = 24 theoretical cells. Realistic ship for a single session = 2–3 cells (the **MVP**). Expansion plan below covers the remaining cells in priority order across subsequent waves.
**Frontends under test:** `cuda-oxide` (Rust → PTX, v0.1.0), `cutile` (Python tile DSL, v1.3.0), `nvcc` CUDA C++ (13.2), `cublas` (13.4.0).
**Mechanisms:** MLA (Multi-head Latent Attention, DeepSeek), GQA (Grouped-Query, Llama-2/3), DSA (DeepSeek Sparse Attention), HCA (Hierarchical Context Attention), GDN (Gated DeltaNet), KDA (Kimi Delta Attention).

## 0. Conventions being followed

Existing repo uses **flat top-level folders** of the form `<frontend>-<workload>[-<variant>]/`, each with its own `ANALYSIS.md`, `run.log`, and `results.csv` (see `cutile-matmul-tiled-mixed/`, `cublas-half-precision/`, `oxide-matmul-tiled-microtile/`). Wave-level rollup lives at `results/waveNN-summary.md`. Cross-cell investigations live at `analysis/waveNN-<topic>/`. This plan keeps that layout.

## 1. Folder layout

```
# Per-cell implementation folders (flat, matches existing convention)
cutile-attn-gqa/           # Wave 15 MVP cell 1
cuda-attn-gqa/             # Wave 15 MVP cell 2  (nvcc C++ reference port)
cublas-attn-gqa/           # Wave 15 MVP cell 3  (3-kernel: cuBLAS QKᵀ + custom softmax + cuBLAS PV)
oxide-attn-gqa/            # Wave 15.5 (expansion): tests the no-TC ceiling in Rust
# ... future cells follow the same <frontend>-attn-<mechanism>/ pattern

# Shared infrastructure (lives under analysis/, reused by every cell)
analysis/wave15-attention-architecture/
    PLAN.md                      # this file
    reference/
        pytorch_reference.py     # canonical numpy/torch reference for every mechanism
        shapes.py                # canonical shape constants + generators
        tensors.py               # deterministic Q/K/V generator (seeded), saves .npy
        tolerances.py            # per-dtype correctness thresholds
    harness/
        aggregate.py             # walks <frontend>-attn-<mech>/results.csv → results/wave15-attention.csv
        plot.py                  # optional: TFLOPS vs seq_len curves
    inputs/                      # generated once, committed as .npy (or .gitignored, regen-script)
        qkv_correctness_s128_h4_d64.npy
        qkv_bench_s2048_h32_d128.npy
results/
    wave15-summary.md            # wave-level narrative + headline table (added at wave close)
```

**Why flat per-cell, not `attention/<mech>/<frontend>/`?** Three reasons: (a) keeps the existing `grep` / `ls` navigation story intact — a user who knows `cutile-matmul-tiled-mixed/` exists immediately understands `cutile-attn-gqa/`; (b) each cell is independently buildable/runnable with its own `Cargo.toml` / `Makefile` / `run.sh` just like existing cells; (c) `attention/` nesting would fork the repo's convention for one wave and force readers to learn a second layout. The extra cost — 24 folder names at the top level if we ever fill the matrix — is real but explicit.

## 2. Frontend × mechanism capability matrix

For each (frontend, mechanism), what's realistically implementable on RTX 5090 sm_120 in May 2026, given the Wave 13/14 findings:

|            | MLA                          | GQA                          | DSA                         | HCA                        | GDN                        | KDA                        |
|------------|------------------------------|------------------------------|-----------------------------|----------------------------|----------------------------|----------------------------|
| cuda-oxide | naive 3-kernel f32 only (**~5 TF ceiling**) | naive 3-kernel f32 (**MVP expansion candidate**) | naive 3-kernel f32; top-k hand-rolled | naive 3-kernel f32 | naive 3-kernel f32; DN state = extra kernel | as GDN |
| **cutile** | **fused** via `ct.mma` f16/bf16 (latent down-proj → small attn) | **fused** f16 (**MVP**) | fused f16 + mask; top-k is the hard bit | fused-per-level f16 | split-state kernel; `ct.mma` for matmul parts | as GDN |
| **nvcc**   | naive 3-kernel or FlashMLA port (reference) | **naive 3-kernel f16→f32acc (MVP reference)** + optional FlashAttn-2 port | FlashAttn + index-gather softmax | per-level FlashAttn | hand-written delta-rule kernel + GEMM | hand-written delta-rule kernel |
| **cublas** | QKᵀ via `cublasGemmEx` + custom softmax + PV `cublasGemmEx`; latent = extra GEMM in front | **QKᵀ + custom softmax + PV via cublasGemmEx (MVP)** | cuBLAS can't express sparsity; not a natural fit | per-level cuBLAS GEMMs | delta-rule lacks a cuBLAS primitive; not a natural fit | as GDN |

### Calibration notes (carried over from Waves 12–14)

- **cuda-oxide** has no usable TC API on consumer Blackwell (Wave 14.4). So oxide attention is f32 via CUDA cores only: ceiling ≈ the 45 TF microtile ceiling × the attention flops/gemm ratio, i.e. **~5–10 TF effective**. Worth running anyway as the "no-TC naive ceiling" data point, but not competitive.
- **cutile** `ct.mma` works at f16/bf16/tf32 with f32 accumulator (Wave 13.1). Attention is expressible tile-wise; softmax fits in `ct.sum`/`ct.max` + arithmetic; fused Flash-style within a single `@ct.kernel` is the right target.
- **nvcc** can host any reference port including a faithful FlashAttention-2 inner loop if we want one. Writing two flavors per mechanism (naive 3-kernel + optional fused) is reasonable for mechanisms where a fused reference exists.
- **cuBLAS** is matmul-only. "cuBLAS attention" is always exactly 3 kernels: `cublasGemmEx(QKᵀ)` + hand-written `softmax_kernel` (hosted alongside, in the same `.cu`) + `cublasGemmEx(PV)`. The custom softmax kernel is the same code path for every mechanism (except DSA, where the mask/index is different), so the cuBLAS column mostly benchmarks how close 3 unfused GEMM+softmax kernels get to a fused reference.

## 3. Per-cell LOC estimate

Rough numbers, based on existing cells (`cutile-matmul-tiled-mixed/` ≈ 300 LOC Python, `cublas-half-precision/matmul.cu` ≈ 250 LOC C++, `oxide-matmul-tiled-microtile/src/main.rs` ≈ 500 LOC Rust):

| implementation style            | LOC (core) | LOC (+ANALYSIS.md) | notes                                                  |
|---------------------------------|-----------:|-------------------:|--------------------------------------------------------|
| naive 3-kernel (any frontend)   |       ~200 |               ~400 | QKᵀ matmul + softmax + PV matmul, separate launches    |
| cuBLAS 3-kernel (QKᵀ + softmax + PV) | ~250 |               ~450 | the softmax kernel is the only nontrivial piece        |
| cuTile fused (Flash-style inner loop) | ~400 |             ~700 | tiled loop over KV blocks with online softmax          |
| nvcc FlashAttention-2 port      |   ~600–800 |          ~900–1100 | full online-softmax + block-sparse masking             |
| cuda-oxide 3-kernel f32         |       ~500 |               ~750 | SharedArray + explicit block+microtile structure       |

Mechanism-specific adders (on top of a base attention cell):

- **GQA**: ~20 LOC (index remap on K/V heads). Simplest.
- **MLA**: ~80–120 LOC (down-projection + latent compression + up-projection). Adds one matmul on each side.
- **DSA**: ~150 LOC (top-k index kernel or lightning-indexer) — top-k is the hard sub-kernel, not the attention itself.
- **HCA**: ~100–150 LOC (tree/chunk iteration + per-level attention calls).
- **GDN / KDA**: ~200 LOC (delta-rule state update) — different kernel structure from standard attention, closer to a recurrent cell.

## 4. MVP (this session / first deliverable)

**Mechanism:** **GQA** (simplest; well-documented; Llama-2/3-in-production; index remap on top of standard scaled-dot-product attention).
**Frontends:** **cuTile + nvcc + cuBLAS** — the three where half-precision TC is available, so the comparison is meaningful at competitive TFLOPS.
**Cells shipped:**

1. `cutile-attn-gqa/` — fused `@ct.kernel` attention with `ct.mma` f16 → f32 acc, online-softmax inner loop. Target: ≥ 50% of cuBLAS-3-kernel TFLOPS at canonical shapes.
2. `cuda-attn-gqa/` — nvcc naive 3-kernel f16 → f32acc reference. Target: ~60–90 TF at seq=2048,h=32,d=128. Correctness oracle for the wave.
3. `cublas-attn-gqa/` — `cublasGemmEx` QKᵀ + custom f16 online-softmax kernel + `cublasGemmEx` PV. Target: ~130–180 TF at canonical shape (softmax will bottleneck below hgemm peak).

### MVP file list (what gets written)

- `analysis/wave15-attention-architecture/PLAN.md` *(this file)*
- `analysis/wave15-attention-architecture/reference/pytorch_reference.py` — canonical GQA reference (`torch.nn.functional.scaled_dot_product_attention` with `enable_gqa=True`) + hand-written loop for correctness witness; dumps expected output.
- `analysis/wave15-attention-architecture/reference/shapes.py` — the two canonical shape sets below.
- `analysis/wave15-attention-architecture/reference/tensors.py` — seeded Q/K/V generation; writes `inputs/qkv_*.npy`.
- `analysis/wave15-attention-architecture/reference/tolerances.py` — per-dtype thresholds (f16: atol=5e-3 rtol=5e-3; bf16: atol=1e-2 rtol=1e-2; f32: atol=1e-5 rtol=1e-5).
- `analysis/wave15-attention-architecture/harness/aggregate.py` — reads every `<frontend>-attn-<mech>/results.csv`, writes `results/wave15-attention.csv` and a markdown snippet for the wave summary.
- `cutile-attn-gqa/{main.py, run.sh, results.csv, run.log, ANALYSIS.md}`
- `cuda-attn-gqa/{attn_gqa.cu, Makefile, results.csv, run.log, ANALYSIS.md}`
- `cublas-attn-gqa/{attn_gqa.cu, softmax_kernel.cu, Makefile, results.csv, run.log, ANALYSIS.md}`
- `results/wave15-summary.md` — wave close.

### MVP commands

```bash
# 0. Generate reference tensors + expected outputs (run once, commit .npy OR .gitignore)
python analysis/wave15-attention-architecture/reference/tensors.py

# 1. cuTile GQA
cd cutile-attn-gqa && ./run.sh | tee run.log

# 2. nvcc GQA reference
cd cuda-attn-gqa && make && ./attn_gqa | tee run.log

# 3. cuBLAS-3-kernel GQA
cd cublas-attn-gqa && make && ./attn_gqa | tee run.log

# 4. Aggregate
python analysis/wave15-attention-architecture/harness/aggregate.py
```

## 5. Canonical shapes

Two sets, both fixed for the entire wave so cross-frontend comparisons are apples-to-apples.

| name          | batch | seq  | n_heads (Q) | n_kv_heads | d_head | dtype       | purpose                          |
|---------------|------:|-----:|------------:|-----------:|-------:|-------------|----------------------------------|
| **correctness** | 1 |  128 |           4 |          2 |     64 | f16 / f32   | fast numerical check vs PyTorch  |
| **bench**       | 1 | 2048 |          32 |          8 |    128 | f16 → f32acc | Llama-3 8B-style standard shape  |

Rationale: the **bench** shape is the industry-standard SDPA benchmark (Llama-3 8B) and is what every FlashAttention paper reports; the **correctness** shape runs in <100ms even through the slowest backend and stays small enough that a dense numerical diff (~32k attention weights) fits a single assert block.

For mechanisms with a latent/compression dim (MLA), add `d_latent=512`. For mechanisms with multi-level structure (HCA), add `levels=3, chunk=256`. For DSA, add `top_k=64`. These are mechanism-local extensions to the shape record.

## 6. Shared infrastructure (build once, reuse for every cell)

**PyTorch reference** (`reference/pytorch_reference.py`):
- One class per mechanism (`GQAReference`, `MLAReference`, …) with a `.forward(q, k, v, **kwargs) -> out` method.
- For GQA, delegates to `F.scaled_dot_product_attention(..., enable_gqa=True)` and double-checks against a hand-written einsum loop — second source inside the reference itself.
- Emits an `expected_out.npy` alongside each `qkv_*.npy` input file, so every frontend can byte-compare without needing a PyTorch-in-the-loop.

**Deterministic tensor generator** (`reference/tensors.py`):
- `torch.manual_seed(42)` + normal init (standard attention init range).
- Writes `.npy` at float32 (canonical precision) and `.f16.npy` / `.bf16.npy` views. Every frontend reads the precision it wants.
- Commits to repo OR provides a one-liner regenerator; size budget ~60 MB for the bench shape, so probably `.gitignore` + a regen script (matches `cuda-3dgs-real/` style).

**Tolerance table** (`reference/tolerances.py`): dtype → (atol, rtol) used by every cell's correctness check. Centralized so "why did test X pass and test Y fail at the same threshold" doesn't devolve into per-cell bike-shedding.

**Result aggregation** (`harness/aggregate.py`):
- Walks `<frontend>-attn-<mech>/results.csv` for every (frontend, mech) cell present on disk.
- Each cell's CSV schema: `impl, mechanism, shape_name, dtype, iter, ms, tflops, correctness_pass`. (Consistent with the schema already used by `cutile-matmul-tiled-mixed/results.csv` + the Wave 14 cuBLAS cell.)
- Writes `results/wave15-attention.csv` (long format, every iter) + `results/wave15-summary.md` table (median/best per cell).
- Runs idempotently; re-runs on any single cell just update that slice.

**Flops model** (put in `harness/flops.py`): one function `attention_flops(mechanism, batch, seq, n_heads, d_head, **kwargs) -> int`. Avoid per-cell copy-paste errors that silently scale TFLOPS wrong across frontends (the #1 way to ship a false headline).

## 7. Expansion plan (post-MVP, in priority order)

Each item is one additional cell, sized at roughly half a wave each:

1. **Wave 15.5 — `oxide-attn-gqa/`.** Tests the "no-TC, pure CUDA-core f32" ceiling for attention. Predicted ~5–10 TFLOPS effective. Useful *because* it quantifies the loss cuda-oxide users must accept today; keeps the cross-frontend story consistent with Waves 12–14.
2. **Wave 16.1 — `cutile-attn-mla/` + `cuda-attn-mla/`.** MLA is the most important new mechanism (DeepSeek-V3-class). Tests cuTile's ability to express the latent down-projection + compressed attention + up-projection as one fused pipeline. cuda-attn-mla exists as the correctness reference.
3. **Wave 16.2 — `cublas-attn-mla/`.** Adds the 5-kernel cuBLAS path (down-proj GEMM, QKᵀ, softmax, PV, up-proj GEMM). Direct test of how much fusion matters for latent attention.
4. **Wave 16.3 — `cutile-attn-dsa/` + `cuda-attn-dsa/`.** DSA's top-k indexer is the hardest sub-kernel in the whole plan. Tests cuTile primitives on non-dense patterns (expected pain point; useful characterization).
5. **Wave 17.1 — `cutile-attn-gdn/` + `cuda-attn-gdn/`.** GDN's delta-rule state is structurally different (recurrent-style). Tests whether cuTile's kernel abstraction fits state-carrying kernels at all.
6. **Wave 17.2 — `cutile-attn-kda/` + `cuda-attn-kda/`.** Kimi Delta. Very close to GDN structurally; inherits the GDN infrastructure.
7. **Wave 17.3 — `cutile-attn-hca/` + `cuda-attn-hca/`.** HCA is conceptually straightforward (nested Flash at each level) but multi-level scheduling makes it the largest single cell by LOC.
8. **Fill-in cells as bandwidth allows:** `cublas-attn-gdn/`, `cublas-attn-kda/`, and `oxide-attn-<mech>/` for each new mechanism. These are low-priority because the cuBLAS column adds little signal beyond GQA/MLA (no sparsity/recurrence primitives), and the oxide column will consistently show the same "no-TC ceiling" story.

Expected state after Wave 17: **12 of 24 cells filled** (all of cuTile + nvcc, GQA across the 4-frontend matrix, MLA through 3 frontends). That's the realistic horizon. The cuBLAS and cuda-oxide rows for GDN/KDA/HCA/DSA are **deliberately unshipped** — they'd repeat an already-established finding without new signal.

## 8. Risks and mitigations

- **Token budget on a single wave.** The MVP is 3 cells + infra — close to the upper end of what one wave can ship. Mitigation: the shared `reference/` and `harness/` modules are written *first* and touched by only one subagent; the three per-frontend cells are file-disjoint and can run as parallel subagents (same pattern as Waves 1–2).
- **Correctness drift at f16.** Softmax on f16 is notoriously lossy; f16 accumulator would fail our tolerance. Mitigation: enforce `f32 accumulator` across every cell (matches cuBLAS `CUBLAS_COMPUTE_32F`), document in tolerance table, assert in every cell.
- **Flops-model error.** Easy to get GQA flops wrong (QKᵀ is `n_heads·seq²·d`, not `n_kv_heads·seq²·d`). Mitigation: single `harness/flops.py`, unit-tested against a known-good reference number for canonical shape.
- **cuTile `ct.mma` at non-matmul-shape tiles.** Attention's QKᵀ has shape `(Bq, d) × (d, Bk)` which is a standard matmul tile and should be fine; PV similarly. The risk is the online-softmax rescale step inside the tile loop — `ct.max` + `ct.exp` + `ct.sum` may or may not fuse cleanly. Mitigation: if fused fails, fall back to 3-kernel cuTile for the MVP cell and document.
- **RTX 5090 f16 TC peak is ~318 TF dense.** cuTile attention is unlikely to exceed cuBLAS-3-kernel (which already pays a softmax-launch tax); realistic cuTile ceiling for fused attention on this hardware is probably **130–170 TF**. Frame the MVP expectation around "within 2× of cuBLAS-3-kernel at canonical shape", not "beat FlashAttention".

## 9. Out of scope for Wave 15

- Backward pass for any mechanism.
- SM_100-only instructions (WGMMA, TCGEN05) — already established in Wave 6/14 that these don't run on consumer Blackwell.
- wgpu/WGSL attention — Wave 1 already showed WSL2 can't reach the GPU via Vulkan.
- Training-loop integration. Every cell is a standalone forward-only microbench.
- Paged/KV-cache attention variants (vLLM PagedAttention etc.). Orthogonal axis; separate future wave if it ever makes sense.

## 10. Definition of done for Wave 15 (MVP)

- Three cells build, run, and pass correctness against `expected_out.npy` at both **correctness** and **bench** shapes.
- Each cell's `ANALYSIS.md` reports: best/median TFLOPS at bench shape, correctness pass/fail, one paragraph on what bottleneck the cell hit (softmax-launch tax / TC engagement / f32 fallback etc.).
- `results/wave15-summary.md` contains the 3-row headline table (cuTile / nvcc / cuBLAS × GQA) and a one-paragraph verdict in the same voice as `results/wave14-summary.md`.
- `BACKLOG.md` Wave 15 block updated with ✅ for MVP items and open boxes for the Wave 15.5 / 16.x expansion items defined in §7.
