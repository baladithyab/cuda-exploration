# cuda-oxide-bench backlog

Vision: ship a comprehensive, third-party-citeable evaluation of NVlabs/cuda-oxide v0.1.0 vs CUDA C++ baselines on Blackwell. Quantify the cost of Rust safety, identify compiler gaps, document setup pitfalls. Independent third-party work, not affiliated.

## Status

- ✅ v0 shipped (initial commit `5065f3e`): naive 4096×4096 f32 matmul, three backends, per-folder ANALYSIS.md.
- ✅ Wave 1-3 shipped: cudaEvent timing, scaling sweep, cuBLAS + tiled, FMA escape hatch, libNVVM corrigendum.
- ✅ Wave 4 shipped: reduction (W4A), bandwidth bench (W4B), libNVVM causal isolation (W4C — inconclusive).
- ✅ Wave 5 shipped: SASS-level analysis identifying `LDG.E.CONSTANT` vs `LDG.E` as the residual-gap root cause.
- ✅ Wave 6 shipped: ran cuda-oxide's `gemm_sol`, `tma_copy`, `tcgen05_matmul` examples — characterizing consumer (sm_120) vs datacenter (sm_100) Blackwell support.
- Open question: how does the LDG.E.CONSTANT delta translate to actual perf delta on a tiled kernel? Wave 7 candidate.

## Items

### P0 — methodology rigor (blocks any new claim)

- [ ] **M1: cudaEvent timing for cuda-oxide.** v0 used wall-clock + `stream.synchronize()` for cuda-oxide; nvcc uses `cudaEventRecord`. Fix to apples-to-apples. Eliminates ~5-50µs/iter sync overhead from the comparison.
  - Files: `oxide-matmul/src/main.rs`. Owner: wave 1 subagent A.
  - Acceptance: oxide bench reports `gpu_ms` (event-based) and `cpu_wall_ms` separately; results table shows `gpu_ms`.

- [ ] **M2: Size-scaling sweep.** Run all benches at N ∈ {1024, 2048, 4096, 8192}. One data point isn't a curve. Look for: does the safe-vs-unchecked gap scale with N (it should — more inner-loop iterations = more bounds checks)? Does cuda-oxide vs nvcc gap stay constant?
  - Files: introduce `--size N` flag in each binary. Owner: wave 1 subagent B (parallel with M1; different binaries don't conflict).
  - Acceptance: `results/scaling.csv` with columns `(impl, N, best_ms, median_ms, tflops)`.

### P1 — close the obvious gaps in cuda-oxide PTX

- [ ] **F1: Fast-math / FMA contraction flag.** PTX shows zero `fma.rn.f32`. Investigate: does cuda-oxide expose a `#[fast_math]` kernel attribute, a global `RUSTFLAGS` switch, or a `cargo oxide build --release-fast` mode? If yes, re-bench. If no, that's the upstream issue.
  - Files: `oxide-matmul/src/main.rs` adds a third kernel `matmul_fastmath` if the toolchain supports it; analysis writes results.
  - Acceptance: either we find a switch (and TFLOPS jumps) or we have a definitive "no, here are the issues we tried" paragraph for the upstream report.

- [ ] **F2: __restrict__ equivalent for `ld.global.nc`.** nvcc uses read-only cache via `__restrict__`. cuda-oxide takes `&[T]` references which should imply non-aliasing already. Investigate why PTX doesn't emit `ld.global.nc` and whether marking with raw `*const f32` + a hint changes it.
  - Owner: same wave as F1.

### P1 — broader comparison axes

- [ ] **C1: cuBLAS reference baseline.** Add `cublas-matmul/` that calls `cublasSgemm`. ~80-90 TFLOPS on RTX 5090. Quantifies how much of the gap is "naive algo fundamentally bad" vs "compiler gap." Without this, we don't know if our 6 TFLOPS naive baseline is even a meaningful comparison point.
  - Files: new folder `cublas-matmul/`. Owner: wave 2 subagent A.
  - Acceptance: `cublas-matmul/{matmul.cu,matmul,run.log,ANALYSIS.md}` with a TFLOPS number.

- [ ] **C2: Tiled shared-memory matmul.** Both cuda-oxide and nvcc, with 16×16 or 32×32 tiles. ~30-50 TFLOPS expected. Tests cuda-oxide's `SharedArray` API which is the next-most-important feature after kernels. Real apples-to-apples beyond the naive case.
  - Files: new folder `oxide-matmul-tiled/` and `cuda-matmul-tiled/`. Owner: wave 2 subagent B.
  - Acceptance: both binaries hit ≥20 TFLOPS, results filed.

### P2 — write-up + upstream

- [ ] **U1: NVlabs/cuda-oxide upstream issue.** With our PTX evidence, file an issue on `NVlabs/cuda-oxide` flagging the FMA + ld.global.nc gap. Be precise: link to commit, include PTX excerpts, propose `#[fast_math]` annotation if not already supported.
  - Files: draft in `docs/upstream-issue-fma.md`; user submits.
  - Acceptance: draft text reviewed by Phase 8.

- [ ] **W1: SUMMARY.md / final writeup.** A standalone results writeup with all scaling curves + the "if you're a Rust dev considering cuda-oxide today, what should you know?" guidance. Less of a README, more of a blog-post-shaped artifact.
  - Files: `SUMMARY.md`. Owner: wave 3.
  - Acceptance: cross-family-review-clean.

### P2 — lower priority / nice-to-have

- [ ] **N1: Multiple block sizes.** 8×8, 16×16, 32×8 etc. for the naive matmul. Occupancy effects.
- [x] **N2: Reduction kernel.** Different access pattern than matmul; tests warp-reduce primitives. Done in Wave 4 W4A: cuda-oxide hits 96% of nvcc on 1-GB sum-reduction (1451 vs 1517 GB/s).
- [ ] **N3: GitHub Actions CI.** Hard without a GPU runner. Document as not-applicable until self-hosted runner available, or use a lightweight "build-only" CI that catches Rust compile errors.

### P3 — opened by Wave 4-6

- [ ] **N4: Memory-bandwidth bench at varying N.** Done in Wave 4 W4B: cuda-oxide and nvcc within 0.1% at N=64M (~90% of HBM peak).
- [ ] **N5: SASS-level disassembly of remaining gap.** Done in Wave 5: `LDG.E.CONSTANT` vs `LDG.E` is the residual-gap root cause; both compilers unroll 8x.
- [ ] **N6: Drop-in cuda-oxide `gemm_sol` / `tcgen05_matmul`.** Done in Wave 6: PTX builds, runtime fails on consumer sm_120 Blackwell — datacenter-only (sm_100/sm_100a). TMA works on sm_120.
- [ ] **U2: Upstream issue for `LDG.E.CONSTANT` / read-only cache hint.** Wave 5 SASS evidence supports drafting a second upstream issue alongside the FMA one. Suggested wording: "rustc-codegen-cuda doesn't emit `LDG.E.CONSTANT` for `&[T]` reads where the slice is shared and immutable; equivalent CUDA C++ with `const __restrict__` does." Patch space: NVPTX lowering of slice-deref to add `nontemporal!`/`invariant.load` hints, or expose a `#[restrict]` attribute on slice params.
- [ ] **W2: Wave 7 candidate — quantify LDG.E.CONSTANT impact in a tiled kernel.** Patch oxide-matmul-tiled to use raw pointers + manual cache-hint inline asm via `core::arch::asm!`, see if the tiled gap closes.
- [ ] **W3: Wave 8 candidate — try cuda-oxide on a Hopper SXM5 (H100) cloud instance** to run `gemm_sol` end-to-end and compare to upstream's claimed 868 TFLOPS on B200.

## Out of scope (for now)

- Tensor Core / WGMMA (requires the cuda-oxide `wgmma` API, ~50% of effort for marginal cross-comparison value vs cuBLAS GEMV/GEMM)
- Multi-GPU
- Mixed precision (fp16, bf16) — single-precision is the cleanest cross-stack comparison
- Fixing wgpu on WSL — well-documented elsewhere; we already showed the limitation

## Wave plan

- **Wave 1 (parallel, 2 subagents):** M1 + M2 — methodology fixes. Independent file ownership. Runs first because all later results depend on the new timing baseline.
- **Wave 2 (parallel, 2 subagents):** F1+F2 (compiler gaps) + C1+C2 (broader axes). Some cross-talk on results format but file-disjoint.
- **Wave 3 (parallel, 2 subagents):** U1 (upstream draft) + W1 (final writeup). Both consume wave-1 + wave-2 outputs.

## Budget

- 3 waves × ~3 parallel subagents/wave + Phase 8 (3-way review) = ~12 subagent calls. Token budget: ~150-300k summary tokens to orchestrator.
- Wall-clock: empirical runs themselves (<5 min each, batched) are smaller than the subagent reasoning time. Whole loop ~30-60 min realistic.

## Cross-cutting risks

- **WSL2 thermals / variability:** v0 saw 5-10% noise in median. With multiple sizes the cv may rise. Mitigation: 10+ iters, report median + IQR, not mean.
- **CUDA 12.0 PTX-JIT to sm_120:** every result is JIT'd, not native. Could mask Blackwell-specific behavior. Document as known limitation; don't try to fix in this loop (would require CUDA 13 install).
- **cuBLAS version skew:** ships with the toolkit. Whatever 12.0 has, that's the baseline. Document the version.
- **Subagent claims that they "ran" something but didn't:** common failure mode. Each wave's commit message must include the exact `./target/release/<bin>` invocation that produced the new run.log; reviewer confirms by reading the log header (which has timestamps) before signing off.
