# Wave V020 — NVlabs cuda-oxide v0.2.0 verification

**Date:** 2026-06-07 · **HW:** RTX 5090 (sm_120), CUDA 13.2, LLVM 21.1.8, driver 596.21
**Toolchain:** `cargo-oxide` v0.2.0 (NVlabs commit `faea3959`, v0.2.0 tag), nightly-2026-04-03.

## Provenance audit (why this wave exists)

`web_search "cuda-oxide latest version"` surfaces **crates.io `cuda-oxide` v0.4.0** — an UNRELATED,
abandoned **2021 GPL-3.0** host-side CUDA driver wrapper by `Protryon`. The project this repo benchmarks
is **NVlabs/cuda-oxide** (a `rustc` backend, Apache-2.0, git-only, latest tag **v0.2.0**).

Audit of every `oxide-*` cell in this repo: **all** depend on NVlabs via
`cuda-device|cuda-host|cuda-core = { git = "https://github.com/NVlabs/cuda-oxide.git" }`. No crates.io
`cuda-oxide` dep, no `[patch]` redirect anywhere. Prior work used the **correct distribution**. (Bare
`git=` without a `rev`/`tag` did let cells lock to slightly different v0.1.0-era commits — May 8–9:
`6de0509`, `9822c76`, `44abb07`. Harmless; this wave pins `tag = "v0.2.0"` for reproducibility.)

## Three verification questions

### Q1 — does v0.2.0's typed launch obsolete the v0.1.0 `cuda_launch!` gotchas? PARTLY.
The `#[cuda_module]` + `module.<kernel>(&stream, cfg, args…)` surface sidesteps `cuda_launch!`, so for
code using it, gotchas #1 (artifact-name `NoArtifact`), #3 (Arc-by-value), #4 (`&mut` rebind) don't
apply. They're **avoidable, not deleted** — the raw `cuda_launch!` macro path (used by upstream
`sharedmem`/TMA examples) still hits them. Verified: typed `saxpy(f32, &[f32], &[f32], out)` launch ✓.

**New v0.2.0 API breaks that void v0.1.0 source:**
- `DisjointSlice` is now generic: `DisjointSlice<'a, T, IndexSpace = Index1D>` → write `DisjointSlice<f32>`.
- 2D indexing: `index_2d_row()`, `index_2d_col()`, `unsafe { index_2d_runtime(n) } -> Option<…>`
  (replaces v0.1.0 `index_2d(stride)`). Canonical: upstream `examples/gemm`.
- `cargo oxide new` now scaffolds **edition 2024**.

### Q2 — does v0.2.0 close the compute-bound gaps (`LDG.E.CONSTANT`, FMA)? NO. Both still open.
Naive f32 matmul, immutable `&[f32]` operands, runtime-bounded `while i<n` loop.
PTX: 0 × `ld.global.nc`. SASS (`ptxas -arch=sm_120 -O3` → `cuobjdump --dump-sass`):

| metric | oxide v0.2.0 `&[f32]` | nvcc control `const __restrict__` |
|---|---|---|
| read-only cache | `LDG.E`×34, **0 `LDG.E.CONSTANT`** | **`LDG.E.CONSTANT`×30** |
| FMA | `FMUL`×17 + `FADD`×17, **0 `FFMA`** | **`FFMA`×15** |

Same algo/arch/tool → oxide codegen gap, not a ptxas artifact. Findings (a) FMA-on-runtime-loop and
(b) read-only-cache-hint **persist unchanged**. v0.2.0 was ergonomics/features, not codegen-quality.

### Q3 — is `#[constant]` the read-only-cache mechanism? NO. It's `.const` (AS4) only.
`#[constant] ConstantMemory<[f32;4]>` → PTX `.visible .const .align 4 .b8 …COEFFS[16];`, reads via
`ld.const.b32` — the 64KB broadcast `.const` bank, NOT the read-only *data* cache, and not viable for
large (64MB) matmul operands. `UNINIT` is zero-init only; host must `module.set_<name>()` before launch
(verified round-trip + re-set-between-launches). `#[readonly]` is a **device-FFI-function** NVVM attr
(LLVM `readonly` fn attribute for LTOIR linking), NOT a slice/param hint — wrong tool for finding (b).

## Wall-clock perf: v0.1.0 vs v0.2.0 — no meaningful difference (measured)

Backend-isolation A/B: same `oxide-matmul` source (old library lock, so ONLY the codegen backend `.so`
varies), `CUDA_OXIDE_BACKEND` env override swaps OLD=`2a03dfd` (v0.1.0-era) vs NEW=`faea3959` (v0.2.0),
`rm -rf target *.ptx *.ll` between runs. RTX 5090 idle ~44–45°C, N∈{1024,2048,4096}, safe/unchecked/fmuladd.

- **best latency** (noise-resistant): median **−0.31%**, range [−4.5%, +1.1%] → dead even; NEW edges ahead
  at N=4096 (19.4–19.7 ms vs OLD 20.2–20.4 ms).
- **median throughput**: median **−2.3%**, range [−11.2%, +0.1%] → dominated by run-to-run variance (the
  −11.2% `safe@2048` cell is a jitter artifact; WSL2 CV is 5–15% with no clock-lock).
- SASS instruction mix identical. **Verdict: v0.2.0 changed what's expressible/ergonomic, not how fast
  emitted code runs.** Discipline: report BEST latency, not median, on noisy WSL2 GPU runs.

## Self-contained binaries: `.oxart` embedding (no sidecar `.ptx` to ship)

`#[cuda_module]` + `oxide-artifacts` bake the PTX into an ELF `.oxart` section (magic `OXIDEART`, v1,
per-target Ptx/NvvmIr/Ltoir/Cubin payloads). `kernels::load(&ctx)` reads from that section at runtime —
no filename lookup, no loose `.ptx`. Zero opt-in (no extra dep / `build.rs`; `cargo oxide` injects it).
**Proven:** copied `oxide-v020-verify` binary alone to an empty `/tmp` dir (no `.ptx`/`.ll`) → ran correctly.
This obsoletes the v0.1.0 `NoArtifact` filename gotcha for `#[cuda_module]` cells. NB: the ~280 MB codegen
backend `.so` is a BUILD-TIME tool (the rustc→PTX compiler), never shipped — it can't and needn't be
eliminated; only the runtime `.ptx` is now self-contained.

## Bottom line
- v0.1.0 compute-bound perf characterization **unchanged** by v0.2.0; wall-clock A/B confirms no perf delta.
- v0.2.0 value = ergonomics + new capabilities (`#[constant]`, typed `#[cuda_module]` launch + `.oxart`
  self-contained binaries, `gpu_printf!`, `launch_bounds`, math fns) + breaking API shape changes.
- Closing finding (b) still needs an upstream codegen patch (emit `invariant.load`/non-temporal metadata
  or `ld.global.nc` for immutable `&[T]`). `#[constant]`/`#[readonly]` are not it.

## Cells
- `oxide-v020-verify/` — Q1 typed `saxpy` + Q3 `#[constant]` poly (PTX `.const` confirmation).
- `oxide-v020-matmul/` — Q2 naive matmul + SASS audit vs nvcc control.
