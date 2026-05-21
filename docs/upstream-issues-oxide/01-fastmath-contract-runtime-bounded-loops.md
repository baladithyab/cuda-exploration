# [bug] `FastmathFlagsAttr::default()` blocks FFMA contraction in runtime-bounded inner loops (default `*+` codegen, RTX 5090 sm_120, v0.1.0)

## Summary

In cuda-oxide v0.1.0, every floating-point op lowered in
`crates/mir-lower/src/convert/ops/arithmetic.rs` attaches
`FastmathFlagsAttr::default()` (= `FastmathFlags::empty()`), so the emitted
NVVM IR carries no fast-math bits — in particular no `contract` flag. **For
fully-unrolled inner loops, libNVVM's pattern-match contractor still fires
on plain `fmul` + `fadd` chains** and the emitted SASS contains hardware
`FFMA` (verified in our 4×4 register-microtile matmul and a 2D-Gaussian-
Splatting forward rasterizer — see "Independently-verifiable evidence"
below). **For runtime-bounded inner loops** (e.g. `for k in 0..dim` where
`dim` is a kernel parameter), the contractor declines to fuse, and the
hot loop ends up with `FMUL` + `FADD` pairs at SASS — doubling FP-issue
pressure vs the same kernel through `nvcc`.

In other words, the libNVVM contractor handles the unrolled-loop case
itself today; the unfixed case is the runtime-bounded inner loop, which
is exactly the shape of the canonical naive-matmul kernel — so this is
the most user-visible manifestation. The two-line patch (set the
`CONTRACT` bit at the existing callsites in `arithmetic.rs:97-102`)
unblocks contraction in *both* shapes uniformly and matches what `nvcc`'s
default `-fp-contract=fast` does for CUDA C++.

`core::intrinsics::fmuladdf32` (i.e. `f32::mul_add`) is a working escape
hatch today, because `mir-lower/src/convert/ops/call.rs:269-270` lowers
it to a libdevice `__nv_fmaf` call, whose body itself contains
`fma.rn.f32` — nvJitLink resolves at module load and the post-link SASS
has hardware FMA. So the issue scope is precisely "default `*+` codegen
on runtime-bounded inner loops," not "no FMA reachable at all."

## Environment

| item | value |
|---|---|
| cuda-oxide | **v0.1.0** (commit `6de0509` of `master`) |
| Rust toolchain | nightly (per crate `rust-toolchain.toml`) |
| LLVM | 21 (`/usr/lib/llvm-21/bin`) |
| CUDA toolkit | 13.2 (`/usr/local/cuda/bin/nvcc` → `release 13.2`) |
| libNVVM | `/usr/local/cuda/nvvm/lib64/libnvvm.so` (libNVVM 22.0.0) |
| GPU | NVIDIA GeForce RTX 5090 |
| compute capability | `sm_120` (Blackwell consumer) |
| OS | Windows 11 host, WSL2 Ubuntu 24.04 |
| nvcc reference | `nvcc -ccbin clang-14 -O3 -arch=sm_120 -cubin` |
| cuobjdump | `/usr/local/cuda/bin/cuobjdump` (CUDA 13.2) |

## Minimal reproducer

The naive matmul kernel below is the canonical runtime-bounded inner-loop
shape. It compiles, runs, and produces numerically correct results — the
bug is purely in the SASS quality of the generated hot loop.

```rust
#![feature(core_intrinsics)]
#![allow(internal_features)]
use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn matmul_unchecked(a: &[f32], b: &[f32],
                        mut c: DisjointSlice<f32>, dim: u32) {
    let row = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();
    let col = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    if row >= dim || col >= dim { return; }
    let dim_us = dim as usize;
    let r = row as usize;
    let c_idx = col as usize;
    let a_base = a.as_ptr();
    let b_base = b.as_ptr();
    let mut acc: f32 = 0.0;
    let mut k: usize = 0;
    while k < dim_us {                                  // runtime-bounded
        unsafe {
            let av = *a_base.add(r * dim_us + k);
            let bv = *b_base.add(k * dim_us + c_idx);
            acc += av * bv;                             // plain *+
        }
        k += 1;
    }
    unsafe { *c.as_mut_ptr().add(r * dim_us + c_idx) = acc; }
}
```

Build and disassemble:

```bash
export CUDA_HOME=/usr/local/cuda
export LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so
export PATH=/usr/lib/llvm-21/bin:$PATH

cargo oxide build --arch sm_120
/usr/local/cuda/bin/cuobjdump --dump-sass oxide_matmul.cubin > oxide_matmul.sass

grep -c '^\s*/\*[0-9a-f]\+\*/\s*FFMA' oxide_matmul.sass     # → 0
grep -c '^\s*/\*[0-9a-f]\+\*/\s*FMUL' oxide_matmul.sass     # → 17
grep -c '^\s*/\*[0-9a-f]\+\*/\s*FADD' oxide_matmul.sass     # → 17
```

The full repro lives in the cuda-exploration repo:

- <https://github.com/baladithyab/cuda-exploration/blob/master/oxide-matmul/src/main.rs>

A side-by-side `nvcc` reference (`__restrict__` `const float*` arguments,
identical algorithm) is at:

- <https://github.com/baladithyab/cuda-exploration/blob/master/cuda-matmul/matmul.cu>

## Expected behavior

`cargo oxide build` should produce, by default, FFMA-contracted SASS
matching what `nvcc -O3 -arch=sm_120` produces from the same algorithm in
CUDA C++. Concretely, on the unrolled K-loop body of the kernel above,
both compilers should emit **8 `FFMA`** per unrolled iteration (the loop
unrolls 8× both in nvcc and in cuda-oxide → libNVVM today — see
"Observed behavior" for the SASS dumps confirming the unroll factor).

This is the same default `nvcc` ships with: `-fmad=true` /
`-fp-contract=fast` is on by default, and the only way to *disable*
contraction in CUDA C++ is to pass `-fmad=false`. Mirror the default.

## Observed behavior — SASS instruction counts

From the same naive matmul above, on the **unrolled hot loop body**
(per Wave 5 SASS audit, `docs/experiments/sass-analysis.md` in this repo):

| Metric | nvcc `matmul` | oxide `matmul_unchecked` | oxide `matmul_fmuladd` |
|---|---|---|---|
| FFMA per unrolled body | **8** | **0** | **8** |
| FMUL | 0 | **8** | 0 |
| FADD | 0 | **8** | 0 |
| LDG count | 16 | 16 | 16 |
| LDG cache variant | `LDG.E.CONSTANT` | `LDG.E` | `LDG.E` |
| Total insns in hot block | 49 | 44 | 36 |
| K-loop unroll factor | **8×** | **8×** | **8×** |

Whole-kernel totals: nvcc 15 `FFMA` / 30 `LDG`; oxide_unchecked 17 `FMUL`
+ 17 `FADD` / 34 `LDG`; oxide_fmuladd 17 `FFMA` / 34 `LDG`. The unroll
factor is identical — this is *not* a missing-unroller issue (originally
hypothesized, then rejected at SASS level).

**Hot-loop SASS excerpts** (full dumps in
`analysis/wave13-sass/cuda_matmul_tiled.sass` for the nvcc-tiled
reference and `oxide-matmul/oxide_matmul.cubin` → `oxide_matmul.sass`
for the oxide variants):

```
# nvcc (one unrolled body)
/*0490*/ FFMA R12, R13, R12, R0    ;
/*04a0*/ FFMA R12, R15, R14, R12   ;
/*04b0*/ FFMA R17, R17, R16, R12   ;
/*04c0*/ FFMA R17, R24, R20, R17   ;
/*04d0*/ FFMA R17, R26, R22, R17   ;
/*04e0*/ FFMA R17, R28, R18, R17   ;
/*04f0*/ FFMA R8,  R30, R8,  R17   ;
/*0500*/ FFMA R0,  R21, R10, R8    ;
```

```
# oxide matmul_unchecked (one unrolled body) — split into FMUL + FADD pairs
/*0440*/ FMUL R31, R25, R24                ;
/*0470*/ FADD R18, R31, R28                ;
/*0490*/ FMUL R35, R29, R26                ;
/*04f0*/ FADD R35, R18, R35                ;
/*0500*/ FMUL R32, R33, R32                ;
/*0530*/ FADD R32, R35, R32                ;
/*0570*/ FMUL R23, R28, R36                ;
/*0580*/ FADD R23, R32, R23                ;
# … 4 more FMUL+FADD pairs
```

**Perf consequence** (RTX 5090 sm_120 native, `cudaEvent` timing, 10
iters median, naive matmul same algorithm in both stacks):

| Implementation | TFLOPS @ N=1024 | TFLOPS @ N=4096 | vs nvcc @ N=4096 |
|---|---:|---:|---:|
| nvcc CUDA C++ (`-arch=sm_120`) | 6.88 | 6.23 | 1.00× |
| cuda-oxide unchecked raw-ptr | 6.95 | 5.13 | **0.82×** |
| cuda-oxide safe slice indexing | 6.94 | 4.84 | 0.78× |
| cuda-oxide `core::intrinsics::fmuladdf32` | 6.92 | 5.70 | **0.92×** |

The fmuladd path (which goes via libdevice `__nv_fmaf`, see issue
summary) closes most of the gap. The remaining ~8% delta at N=4096 is
the separate `LDG.E.CONSTANT` issue (filed as #02 in this directory) —
not part of this issue.

**Critical context: when the contractor DOES fire on plain `*+`.**
Wave 7 (4×4 register microtile, K-loop fully unrolled at compile time)
and Wave 8 (2D Gaussian Splatting forward rasterizer, fully-unrolled
per-pixel kernel) both produced FFMA at SASS level on `safe` kernels
using only plain `acc = acc + a*b` — no `fmuladd`, no fast-math flag.
The libNVVM pattern-match contractor handles those shapes correctly
*without* the `contract` bit. So this issue's scope is precisely the
runtime-bounded inner-loop shape (the canonical naive-matmul, the
GEMV-style memory-bound matmul, anything with a kernel-parameter
trip count). The patch unblocks the case the contractor doesn't
handle today.

## Independently-verifiable evidence

All in the cuda-exploration repo
(<https://github.com/baladithyab/cuda-exploration>):

**This issue (FFMA contraction missing on runtime-bounded loops):**

- `oxide-matmul/src/main.rs` — three-kernel reproducer (`matmul`, `matmul_unchecked`, `matmul_fmuladd`)
- `oxide-matmul/oxide_matmul.cubin` — the problem cubin (all three kernels)
- `cuda-matmul/matmul.cu` — nvcc baseline (identical algorithm, `__restrict__` + `const float*`)
- `cuda-matmul/matmul.cubin` — nvcc reference cubin
- `docs/experiments/sass-analysis.md` — Wave 5 SASS audit (table + excerpts above)
- `docs/experiments/fma-toggle.md` — Wave 3 W3A: empirical proof that `fmuladdf32` → `__nv_fmaf` → `fma.rn.f32` resolves to hardware FMA post-libdevice-link
- `docs/research/cuda-oxide-flags.md` — full investigation of cuda-oxide's fast-math plumbing (CLI surface, `#[kernel]` attributes, every callsite of `add_fastmath_flags`)

**Counter-evidence (cases where libNVVM's contractor fires on plain `*+`):**

- `oxide-matmul-tiled-microtile/src/main.rs` — 4×4 register-microtile tiled matmul; `_safe` kernel with plain `*+` produces 128 `FFMA` at SASS
- `analysis/wave13-sass/oxide_matmul_tiled_microtile.sass` — the SASS proving it
- `oxide-3dgs-mini/oxide_3dgs_mini.sass` — 2D-Gaussian-Splatting forward rasterizer; per-pixel kernel produces 9 `FFMA` from plain `*+`

Grep commands that reproduce the headline counts:

```bash
# Runtime-bounded inner loop — contractor does NOT fire:
grep -c FFMA oxide-matmul/oxide_matmul.cubin.sass             # → 0  on matmul_unchecked
grep -c FMUL oxide-matmul/oxide_matmul.cubin.sass             # → 17 (whole kernel)
grep -c FFMA cuda-matmul/matmul.sass                          # → 15 (nvcc reference)

# Fully-unrolled inner loop — contractor DOES fire (counter-evidence):
grep -c FFMA analysis/wave13-sass/oxide_matmul_tiled_microtile.sass  # → 192
grep -c FFMA oxide-3dgs-mini/oxide_3dgs_mini.sass                    # → 9
```

## Suggested fix

Smallest useful patch: thread a `LoweringOptions { fast_math: FastmathFlags }`
through `Context` so the existing `add_fastmath_flags` callsite in
`crates/mir-lower/src/convert/ops/arithmetic.rs:97-102` emits a non-empty
attribute. The bit set is already defined in
`crates/dialect-llvm/src/attributes.rs:89-100`:

```rust
// crates/dialect-llvm/src/attributes.rs (existing, already there):
bitflags::bitflags! {
    pub struct FastmathFlags: u32 {
        const NNAN     = 1 << 1;
        const NINF     = 1 << 2;
        const NSZ      = 1 << 3;
        const ARCP     = 1 << 4;
        const CONTRACT = 1 << 5;     // ← the only bit needed for FFMA fusion
        const AFN      = 1 << 6;
        const REASSOC  = 1 << 7;
        const FAST     = NNAN|NINF|NSZ|ARCP|CONTRACT|AFN|REASSOC;
    }
}

// crates/dialect-llvm/src/attributes.rs:117-124 (existing):
#[pliron_attr(name = "llvm.fast_math_flags", verifier = "succ")]
pub struct FastmathFlagsAttr(pub FastmathFlags);
impl Default for FastmathFlagsAttr {
    fn default() -> Self {
        FastmathFlagsAttr(FastmathFlags::empty())   // ← becomes CONTRACT
    }
}
```

Two paths, in increasing order of scope:

1. **MVP — flip the default to `CONTRACT` only.** Single-line change at
   `crates/dialect-llvm/src/attributes.rs:122-124`, returning
   `FastmathFlagsAttr(FastmathFlags::CONTRACT)` instead of `empty()`.
   Preserves IEEE-strict semantics for everything except the
   FMA-fusion-vs-rounding case, which is what `nvcc`'s default
   `-fp-contract=fast` already does. Should make `fma.rn.f32` appear
   in cuda-oxide's PTX for both the runtime-bounded and unrolled cases
   uniformly, with no impact on the cases the contractor handles today.

2. **Full plumbing — `--fast-math` / `--fp-contract` CLI flag.** Thread a
   `LoweringOptions` through `crates/mir-lower/src/convert/ops/arithmetic.rs`
   (callsites at `arithmetic.rs:122, 148, 174, 200, 227, 542` and
   `cast.rs:453` all currently call `add_fastmath_flags` with the default
   attr). Expose via `--fp-contract={off|on|fast}` in
   `crates/cargo-oxide/src/main.rs` (CLI surface enumerated in
   `docs/research/cuda-oxide-flags.md`) plus a `CUDA_OXIDE_FAST_MATH` env
   var forwarded in `commands.rs:803-812` next to the existing
   `RUSTFLAGS` builder.

Optional follow-up that pairs cleanly with this patch: change
`crates/mir-lower/src/convert/ops/call.rs:269-270` so `FmuladdF32`/
`FmuladdF64` lower to `llvm.fmuladd.f32`/`llvm.fmuladd.f64` instead of
to a libdevice `__nv_fmaf`/`__nv_fma` call. NVPTX lowers `llvm.fmuladd`
to `fma.rn.*` regardless of fast-math flags, giving per-call opt-in via
`a.mul_add(b, c)` without a global flag — and avoids the
nvJitLink-time round trip the current path takes. Independent of
this issue's main fix, but clean to ship together.

## Workarounds known to the filing repo

- **Use `core::intrinsics::fmuladdf32`** (or `f32::mul_add` once stable
  on the cuda-oxide nightly) on the runtime-bounded inner-loop hot path.
  Verified to lower to libdevice `__nv_fmaf` → `fma.rn.f32` post-link.
  This is the canonical workaround in the cuda-exploration matmul cells
  (`oxide-matmul/src/main.rs` `matmul_fmuladd` kernel; ~5-10% lift on the
  N=4096 naive matmul).
- **Move to a fully-unrolled inner loop** when the geometry allows.
  Wave 7's 4×4 register microtile has a compile-time-known inner-loop
  trip count (the microtile dimensions), and libNVVM's contractor
  handles plain `*+` correctly there. Algorithm-level patch, not a
  workaround for the compiler issue, but documented because it's the
  shape that already works.

## Context for maintainers

This finding comes from **cuda-exploration**
(<https://github.com/baladithyab/cuda-exploration>), an independent
third-party benchmark comparing cuda-oxide, CUDA C++ (nvcc), cuTile, and
Mojo as GPU programming frontends on RTX 5090 Blackwell consumer
hardware. The repo's overall findings for cuda-oxide are positive:
**memory-bound parity** with nvcc (vec-add 1573 vs 1568 GB/s, reduction
1519 vs 1522 GB/s — both at ~85-90% of HBM peak, within 0.1-0.5%);
**byte-identical pixels** in a side-by-side 2D Gaussian Splatting
rasterizer port; and **complex 12-arg kernels** (with `expf` via
libdevice and per-pixel state machines) compile cleanly without API
friction.

The two upstream-patchable items we identified at SASS level are this
one (FFMA contraction in runtime-bounded loops) and a separate read-only
cache hint issue (`LDG.E.CONSTANT` not emitted for `&[T]` reads where
the slice is shared and immutable, filed as `02-ldg-e-constant-readonly-cache-hint.md`
in this directory). Both are narrow, well-scoped, and have working
escape hatches today; both would close the residual N=4096 naive matmul
gap to within `nvcc` parity if patched together.

The full repo is at <https://github.com/baladithyab/cuda-exploration>.
Happy to provide additional SASS dumps, NVVM IR dumps, rerun the
experiment with any suggested flag changes, or send a PR for the MVP
single-line `Default` change if that would be welcome.
