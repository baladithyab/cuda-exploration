# Upstream issue drafts — NVlabs/cuda-oxide

Drafts prepared for submission to <https://github.com/NVlabs/cuda-oxide/issues>.
All findings are from **Wave 5 SASS-level analysis** (with cross-kernel
confirmation from Waves 7, 8, 11, and 13) of cuda-exploration, an independent
third-party benchmark comparing cuda-oxide, CUDA C++ (nvcc), cuTile, and Mojo
on RTX 5090 (Blackwell consumer, `sm_120`) with cuda-oxide v0.1.0 (commit
`6de0509` of `master`).

Each issue is evidence-first: every headline claim has a corresponding SASS
dump, per-iter CSV, or reproducer source committed in this repo, and every
fix-sketch cites the specific source file and line range in the
`NVlabs/cuda-oxide` tree where the change would land.

## Drafts

| # | file | title | status |
|---|---|---|---|
| 1 | [`01-fastmath-contract-runtime-bounded-loops.md`](01-fastmath-contract-runtime-bounded-loops.md) | `[bug]` `FastmathFlagsAttr::default()` blocks FFMA contraction in runtime-bounded inner loops (default `*+` codegen, RTX 5090 sm_120, v0.1.0) | **ready to submit** |
| 2 | [`02-ldg-e-constant-readonly-cache-hint.md`](02-ldg-e-constant-readonly-cache-hint.md) | `[performance]` `LDG.E.CONSTANT` (read-only cache hint) not emitted for `&[T]` reads where slice is shared and immutable; equivalent CUDA C++ with `const __restrict__` does (RTX 5090 sm_120, v0.1.0) | **ready to submit** |

## Evidence chain at a glance

**Issue 1 — FFMA contraction missing on runtime-bounded inner loops:**

- `oxide-matmul/src/main.rs` — three-kernel reproducer (`matmul`, `matmul_unchecked`, `matmul_fmuladd`); inner loop `while k < dim_us` is runtime-bounded.
- `cuda-matmul/matmul.cu` — identical algorithm in CUDA C++ (`__restrict__` + `const float*`).
- `docs/experiments/sass-analysis.md` (Wave 5) — table: nvcc 8 FFMA per unrolled body, oxide_unchecked 0 FFMA + 8 FMUL + 8 FADD, oxide_fmuladd 8 FFMA per body. Both compilers unroll 8×.
- `docs/experiments/fma-toggle.md` (Wave 3) — empirical proof that `core::intrinsics::fmuladdf32` resolves to hardware `fma.rn.f32` post-libdevice-link (working escape hatch).
- `docs/research/cuda-oxide-flags.md` — full plumbing audit: every callsite of `add_fastmath_flags` in `crates/mir-lower/src/convert/ops/arithmetic.rs` passes `FastmathFlagsAttr::default()` (= empty); the `CONTRACT` bit is defined at `crates/dialect-llvm/src/attributes.rs:89-100` but never set.
- **Counter-evidence the bug is loop-shape-specific** (`oxide-matmul-tiled-microtile/`, `oxide-3dgs-mini/`): fully-unrolled inner loops produce FFMA from plain `*+` because libNVVM's pattern-match contractor fires on linear IR. So this is precisely about runtime-bounded inner loops.
- **Suggested fix anchor**: flip `FastmathFlagsAttr::default()` at `crates/dialect-llvm/src/attributes.rs:121-124` to set `CONTRACT`, OR thread `LoweringOptions` through the existing callsite at `crates/mir-lower/src/convert/ops/arithmetic.rs:97-102`. Companion change: `crates/mir-lower/src/convert/ops/call.rs:269-270` so `FmuladdF32`/`FmuladdF64` lower to `llvm.fmuladd.*` rather than libdevice `__nv_fmaf`/`__nv_fma`.

**Issue 2 — `LDG.E.CONSTANT` (read-only cache hint) not emitted:**

- Same naive-matmul reproducer as issue 1 (matmul kernel reads through `&[f32]` slice arguments — shared, immutable for the kernel lifetime).
- `docs/experiments/sass-analysis.md` (Wave 5) — table: nvcc emits `LDG.E.CONSTANT` (16 hoisted loads in the unrolled hot block, all promoted); cuda-oxide emits plain `LDG.E` for all variants (`matmul`, `matmul_unchecked`, `matmul_fmuladd`). The fmuladd variant reaches FFMA parity but the LDG variant doesn't change — isolating this as an independent issue.
- **Cross-kernel confirmation** (Wave 11): a 12-arg 2D-Gaussian-Splatting forward rasterizer ported line-by-line between cuda-oxide and CUDA C++ (`oxide-3dgs-mini/` vs `cuda-3dgs-real/`) produces **byte-identical pixel outputs** (md5 verified) with the only SASS-level difference being 0/9 vs 9/0 `LDG.E.CONSTANT` vs `LDG.E`. Confirms this is a codegen-quality issue, not a semantics issue, and reproduces outside matmul.
- **Perf consequence**: ~8% gap at N=4096 naive matmul once FMA-contraction is matched (oxide_fmuladd at 5.70 TF vs nvcc at 6.23 TF) — consistent with read-only-cache promotion being the next-largest lever.
- **Suggested fix anchor**: cuda-oxide's NVPTX lowering of `&[T]` slice reads should attach `!invariant.load !{}` metadata or call out to the `llvm.nvvm.ldg.global.*` intrinsic. The Rust slice type already carries the immutability guarantee for the kernel lifetime — promotion can be unconditional for `&[T]` (vs `&mut [T]` / `DisjointSlice<T>`). Natural location: a new module next to the existing `arithmetic.rs` / `call.rs` in `crates/mir-lower/src/convert/ops/`.

## Context for maintainers

**cuda-exploration** (<https://github.com/baladithyab/cuda-exploration>) is a
public benchmark repository evaluating GPU programming frontends for Blackwell
consumer hardware. Wave 5 disassembled the cubins from Waves 1-3 (naive
matmul, tiled matmul) with `cuobjdump --dump-sass` (CUDA 13.2) for SASS-level
cross-stack comparison. Subsequent waves (7 register-microtile, 8 3DGS
rasterizer, 11 cross-stack 3DGS port, 13 mixed-precision MMA) re-tested the
findings against fully-unrolled and large-kernel cases — these two issues are
what survived that audit with cross-kernel evidence that the behavior is a
cuda-oxide codegen-quality issue (i.e. not a hardware ceiling, not an
environment misconfiguration, not single-benchmark noise).

The repo's overall findings for cuda-oxide are positive:

- **Memory-bound parity** with nvcc: vec-add 1573 vs 1568 GB/s, warp-shuffle reduction 1519 vs 1522 GB/s — both at ~85-90% of HBM peak, within 0.1-0.5%.
- **Byte-identical pixels** in a side-by-side cuda-oxide ↔ CUDA C++ port of a 2D Gaussian Splatting forward rasterizer (Wave 11; md5 hashes match on cameras A/C, 3-pixel sub-ULP diff on cam D from clang-vs-rustc reordering).
- **Complex 12-arg kernels** with `expf` via libdevice and per-pixel state machines compile cleanly without API friction.
- **Algorithm geometry beats compiler tricks** — the 4×4 register-microtile cuda-oxide matmul (Wave 7) reaches nvcc-tiled parity at N=1024 (27-28 vs 24.5 TFLOPS) using only safe slice indexing + plain `*+`, validating the toolchain end-to-end.

So these two filings are narrowly scoped: the residual N=4096-naive-matmul
gap and equivalent compute-bound shapes. Patching either would close part of
the gap; patching both should bring cuda-oxide on `&[T]`-typed slice
arguments through default `*+` codegen to within `nvcc -O3` parity on
runtime-bounded inner loops, which is the most user-facing case.

The full repo is at <https://github.com/baladithyab/cuda-exploration>.
Issues are submitted to <https://github.com/NVlabs/cuda-oxide/issues>. Happy
to send PRs for either MVP fix-sketch (single-line `Default` change for
issue 1; `!invariant.load` metadata on slice-read lowering for issue 2)
if that would be welcome.
