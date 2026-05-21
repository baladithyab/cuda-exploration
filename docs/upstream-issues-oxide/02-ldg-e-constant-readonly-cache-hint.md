# [performance] `LDG.E.CONSTANT` (read-only cache hint) not emitted for `&[T]` reads where slice is shared and immutable; equivalent CUDA C++ with `const __restrict__` does (RTX 5090 sm_120, v0.1.0)

## Summary

In cuda-oxide v0.1.0, kernel reads through `&[T]` slice arguments lower
to plain `LDG.E` SASS instructions on Blackwell consumer (`sm_120`) —
**never** the `LDG.E.CONSTANT` cache-hint variant. Equivalent CUDA C++
with `const float* __restrict__` arguments emits `LDG.E.CONSTANT` for
the same kernel, routing the load through the read-only / uniform
cache path. This costs ~10-15% on memory-traffic-heavy compute-bound
kernels (verified: the residual gap on the FMA-parity oxide naive
matmul, where `matmul_fmuladd` reaches FFMA-count parity with nvcc but
still trails by ~8-15% across N=2048-4096).

The Rust signature `&[T]` already encodes "shared and immutable for the
duration of the kernel" — so there is no aliasing issue and no API
change needed at the user surface; the lowering can promote
unconditionally for `&[T]` (vs `&mut [T]` / `DisjointSlice<T>`). This is
narrowly an NVPTX-lowering / IR-metadata gap.

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

The naive matmul kernel below is the canonical case. `a` and `b` are
`&[f32]` (shared, immutable for the kernel lifetime); `c` is
`DisjointSlice<f32>` (the per-thread mutable view) — only the reads of
`a` and `b` are candidates for read-only cache promotion.

```rust
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
    while k < dim_us {
        unsafe {
            let av = *a_base.add(r * dim_us + k);   // ← should be LDG.E.CONSTANT
            let bv = *b_base.add(k * dim_us + c_idx); // ← should be LDG.E.CONSTANT
            acc = core::intrinsics::fmuladdf32(av, bv, acc);
        }
        k += 1;
    }
    unsafe { *c.as_mut_ptr().add(r * dim_us + c_idx) = acc; }
}
```

The CUDA C++ reference (identical algorithm, same launch shape) is:

```cuda
__global__ void matmul(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim || col >= dim) return;
    float acc = 0.0f;
    for (int k = 0; k < dim; ++k) {
        acc += A[row * dim + k] * B[k * dim + col];   // ← becomes LDG.E.CONSTANT
    }
    C[row * dim + col] = acc;
}
```

Build and disassemble both:

```bash
export CUDA_HOME=/usr/local/cuda
export LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so
export PATH=/usr/lib/llvm-21/bin:$PATH

# cuda-oxide build
cargo oxide build --arch sm_120
/usr/local/cuda/bin/cuobjdump --dump-sass oxide_matmul.cubin > oxide_matmul.sass

# nvcc reference
/usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -cubin -o matmul.cubin matmul.cu
/usr/local/cuda/bin/cuobjdump --dump-sass matmul.cubin > matmul.sass

# Headline counts:
grep -c 'LDG.E.CONSTANT' matmul.sass         # → 30  (nvcc, all global loads promoted)
grep -c 'LDG.E.CONSTANT' oxide_matmul.sass   # → 0   (cuda-oxide, none promoted)
grep -c '\bLDG.E\b'      oxide_matmul.sass   # → 34  (all plain LDG.E)
```

The full repro lives in the cuda-exploration repo:

- <https://github.com/baladithyab/cuda-exploration/blob/master/oxide-matmul/src/main.rs>
- <https://github.com/baladithyab/cuda-exploration/blob/master/cuda-matmul/matmul.cu>

A second independent reproducer — a 12-arg 2D-Gaussian-Splatting forward
rasterizer ported line-by-line between Rust (cuda-oxide) and CUDA C++ —
shows the same exact `LDG.E.CONSTANT vs LDG.E` split (Wave 11):

- <https://github.com/baladithyab/cuda-exploration/tree/master/oxide-3dgs-mini>
- <https://github.com/baladithyab/cuda-exploration/tree/master/cuda-3dgs-real>
- 0 / 9 `LDG.E.CONSTANT` (cuda-oxide vs nvcc) on byte-identical-output
  pipelines — the only SASS-level difference between the two ports.

## Expected behavior

`cargo oxide build` should emit `LDG.E.CONSTANT` (or equivalently the
PTX `ld.global.nc` / `__ldg`) for reads through `&[T]` slice arguments.
The Rust slice type already carries the "shared and immutable for the
kernel lifetime" guarantee, so the lowering can promote without any new
user-facing attribute or aliasing escape hatch.

Concretely — at minimum — for the naive matmul above, the `a` and `b`
reads (the 16 LDGs hoisted out of the 8×-unrolled inner loop) should
each become `LDG.E.CONSTANT` instructions. The `c` write through
`DisjointSlice<T>::as_mut_ptr()` is correctly *not* promoted.

For comparison, `nvcc -O3 -arch=sm_120` on the equivalent CUDA C++ with
`const __restrict__` annotations emits **30 `LDG.E.CONSTANT` and 0
plain `LDG.E`** (whole kernel) — every global read promoted.

## Observed behavior — SASS instruction counts

From the same kernel pair above, post-libdevice-link, on the unrolled
hot loop body (per Wave 5 SASS audit, `docs/experiments/sass-analysis.md`):

| Metric | nvcc `matmul` | oxide `matmul_unchecked` | oxide `matmul_fmuladd` |
|---|---|---|---|
| LDG count (hot block) | 16 | 16 | 16 |
| **LDG cache variant (hot block)** | **`LDG.E.CONSTANT`** | `LDG.E` | `LDG.E` |
| LDG count (whole kernel) | 30 | 34 | 34 |
| **`LDG.E.CONSTANT` (whole kernel)** | **30** | **0** | **0** |
| K-loop unroll factor | 8× | 8× | 8× |
| FFMA per unrolled body | 8 | 0 | 8 |

(The FFMA column is the separate FMA-contraction issue, filed as
`01-fastmath-contract-runtime-bounded-loops.md` in this directory. Note
that even with FFMA parity in `matmul_fmuladd`, the LDG variant doesn't
change — that's how we isolated this as an independent issue.)

**Hot-loop SASS excerpts** (full dumps in `oxide-matmul/oxide_matmul.cubin`
and `cuda-matmul/matmul.cubin` in the cuda-exploration repo):

```
# nvcc — 16 hoisted loads, all .CONSTANT
/*0240*/ LDG.E.CONSTANT R13, desc[UR16][R4.64+-0x10]   ;
/*0290*/ LDG.E.CONSTANT R15, desc[UR16][R4.64+-0xc]    ;
/*0XXX*/ LDG.E.CONSTANT … (14 more)                   ;
```

```
# oxide matmul_fmuladd — same 16 hoisted loads, plain LDG.E
/*0460*/ LDG.E R19, desc[UR4][R16.64+0x18]             ;   // no .CONSTANT suffix
/*0XXX*/ LDG.E … (15 more)                            ;
```

**Cross-stack comparison on a second, independent kernel** (Wave 11
2D-Gaussian-Splatting forward rasterizer port — Rust↔CUDA-C++
line-by-line same algorithm, byte-identical PPM outputs verified by
md5):

| Metric | cuda-oxide oxide-3dgs-mini | nvcc cuda-3dgs-real |
|---|---:|---:|
| `LDG.E.CONSTANT` count | **0** | **9** |
| plain `LDG.E` count | **9** | **0** |
| FFMA per kernel | 9 | 9 (parity here — fully-unrolled inner loop, FMA contractor fires) |
| Pixel output md5 | matches nvcc | matches oxide |

Pixel outputs are byte-identical; SASS is arithmetically identical; the
`LDG.E.CONSTANT` vs `LDG.E` flip is the only delta. Confirms this is a
codegen-quality issue, not a semantics issue.

**Perf consequence** (RTX 5090 sm_120, `cudaEvent` timing, 10 iters
median, naive matmul same algorithm, FMA-parity variant only — i.e.
isolating the LDG cache-variant effect):

| Implementation | TFLOPS @ N=2048 | TFLOPS @ N=4096 | vs nvcc @ N=4096 |
|---|---:|---:|---:|
| nvcc CUDA C++ (`const __restrict__`, has `LDG.E.CONSTANT`) | 6.71 | 6.23 | 1.00× |
| cuda-oxide `matmul_fmuladd` (FFMA parity, plain `LDG.E`) | 6.18 | 5.70 | **0.92×** |

The ~8% residual once FMA parity is reached is consistent with the
read-only-cache promotion being the next-largest lever.

## Independently-verifiable evidence

All in the cuda-exploration repo
(<https://github.com/baladithyab/cuda-exploration>):

**This issue (LDG.E.CONSTANT not emitted):**

- `oxide-matmul/src/main.rs` — three-kernel reproducer
- `oxide-matmul/oxide_matmul.cubin` — the problem cubin (all three kernels)
- `cuda-matmul/matmul.cu` — nvcc baseline (identical algorithm,
  `const float* __restrict__`)
- `cuda-matmul/matmul.cubin` — nvcc reference cubin
- `docs/experiments/sass-analysis.md` — Wave 5 SASS audit (`LDG.E.CONSTANT`
  vs `LDG.E` table + excerpts above)
- `analysis/wave13-sass/cuda_matmul_tiled.sass` — nvcc tiled matmul SASS
  (also all `LDG.E.CONSTANT`)
- `analysis/wave13-sass/oxide_matmul_tiled_microtile.sass` — cuda-oxide
  4×4-microtile SASS (all plain `LDG.E`, FFMA contraction fires here so
  it's an isolated LDG-only delta)

**Cross-kernel evidence (3D Gaussian Splatting port, byte-identical-output):**

- `oxide-3dgs-mini/src/main.rs` — 12-arg cuda-oxide forward rasterizer
- `oxide-3dgs-mini/oxide_3dgs_mini.sass` — 0 `LDG.E.CONSTANT`, 9 plain `LDG.E`
- `cuda-3dgs-real/rasterize_kernel.cu` — line-by-line CUDA C++ port
- `cuda-3dgs-real/rasterize_kernel.sass` — 9 `LDG.E.CONSTANT`, 0 plain `LDG.E`
- `cuda-3dgs-real/ANALYSIS.md` — md5 verification + perf table

Grep commands that reproduce the headline counts:

```bash
# Naive matmul
grep -c 'LDG.E.CONSTANT' cuda-matmul/matmul.sass                              # → 30
grep -c 'LDG.E.CONSTANT' oxide-matmul/oxide_matmul.sass                       # → 0
grep -c '\bLDG.E\b'      oxide-matmul/oxide_matmul.sass                       # → 34

# Tiled matmul (FMA-parity case — isolates the LDG-only effect)
grep -c 'LDG.E.CONSTANT' analysis/wave13-sass/cuda_matmul_tiled.sass          # nonzero
grep -c 'LDG.E.CONSTANT' analysis/wave13-sass/oxide_matmul_tiled_microtile.sass  # → 0

# 3DGS rasterizer (byte-identical output)
grep -c 'LDG.E.CONSTANT' cuda-3dgs-real/rasterize_kernel.sass                 # → 9
grep -c 'LDG.E.CONSTANT' oxide-3dgs-mini/oxide_3dgs_mini.sass                 # → 0
```

## Suggested fix

The PTX-level mechanism is well-known: NVPTX lowers `llvm.nvvm.ldg.global.*`
intrinsics (or any load carrying `!invariant.load` metadata) to
`ld.global.nc`, which `ptxas` then schedules as `LDG.E.CONSTANT`. The
gap is in cuda-oxide's NVPTX lowering of `&[T]` reads to LLVM IR — the
generated `load` instructions don't carry that metadata or use that
intrinsic.

Two paths, in increasing order of scope:

1. **MVP — promote `&[T]` slice reads unconditionally.** Where slice-deref
   reads lower to LLVM `load` instructions (in `crates/mir-lower/src/`,
   the slice/pointer-deref lowering paths), attach `!invariant.load !{}`
   metadata, *or* call out to `llvm.nvvm.ldg.global.f32` /
   `llvm.nvvm.ldg.global.i32` / etc. for the f32/i32/etc. concrete
   element types. The `&[T]` Rust signature carries the immutability
   guarantee for the kernel lifetime, so this is sound without any
   user-facing attribute. `&mut [T]` and `DisjointSlice<T>` should
   continue to lower as plain loads.

   The narrow fix probably touches only the slice-read lowering site
   (and possibly the raw-pointer-read lowering when the originating
   slice is `&[T]` — which would require a small bit of provenance
   tracking in mir-lower, or a heuristic of "`*const T` reads from
   slice arguments typed `&[T]`"). Pointing at exact line numbers is
   harder than for issue #1 because the lowering is spread across
   multiple files; the natural neighbour is
   `crates/mir-lower/src/convert/ops/` next to the existing
   `arithmetic.rs` / `call.rs` modules.

2. **Optional follow-up — explicit `ReadOnlyPtr<T>` / `#[readonly]` on
   slice arguments.** Mirror the existing `#[readonly]` proc-macro
   (which today applies to FFI declarations and maps to LLVM
   `readonly` function attributes per
   `cuda-macros` + `device_codegen.rs:302`) to slice parameters,
   surfacing an explicit user-facing knob for the case where a
   `*const T` is known-immutable but lives behind a raw pointer with
   no slice provenance. Less important than (1) — most user code
   uses `&[T]` already.

The `cuda-macros` crate already defines `#[convergent]`, `#[pure]`,
`#[readonly]` per `docs/research/cuda-oxide-flags.md` (which surveyed
the full proc-macro surface) — so the attribute machinery for (2) is
mostly in place, the question is wiring it through to the load
metadata at the slice-argument level.

## Workarounds known to the filing repo

- **None at the user surface.** Inline `asm!` is not available for
  user kernels in cuda-oxide v0.1.0
  (`cuda-oxide-book/appendix/supported-features.md:175` lists it as
  *Planned*; the existing `inline_asm_convergent` is an internal
  mir-lower helper used by tcgen05/wgmma/mbarrier/tma intrinsics,
  with no stable user-facing path). So the `__ldg` intrinsic / `LDG.E.CONSTANT`
  cache hint is not reachable from kernel code today.
- The fmuladd workaround for the separate FMA-contraction issue
  (issue #01 in this directory) reaches FFMA parity but does not
  change the LDG variant — verified in the same SASS dumps.

## Context for maintainers

This finding comes from **cuda-exploration**
(<https://github.com/baladithyab/cuda-exploration>), an independent
third-party benchmark comparing cuda-oxide, CUDA C++ (nvcc), cuTile, and
Mojo as GPU programming frontends on RTX 5090 Blackwell consumer
hardware. The repo's overall findings for cuda-oxide are positive:
**memory-bound parity** with nvcc on vec-add (1573 vs 1568 GB/s) and
warp-shuffle reduction (1519 vs 1522 GB/s) — both at ~85-90% of HBM
peak, within 0.1-0.5%. The compiler-level deltas show up only on
compute-bound kernels with sustained global-memory traffic in the inner
loop (matmul, the per-pixel rasterizer's gaussian-fetch path), where
the read-only cache hint becomes load-bandwidth-relevant.

The two upstream-patchable items we identified at SASS level are this
one (`LDG.E.CONSTANT` not emitted) and a separate FFMA-contraction
issue on runtime-bounded inner loops (filed as
`01-fastmath-contract-runtime-bounded-loops.md` in this directory).
Both are narrow, well-scoped, and have working escape hatches today
(the FMA one does — `f32::mul_add`); together they account for the
residual ~10-20% N=4096 naive-matmul gap once libNVVM is
correctly configured. We were specifically careful in
Wave 11 (the 2D-Gaussian-Splatting cross-stack port) to confirm the
behavior reproduces on a non-matmul kernel with byte-identical
output — so this isn't a single-benchmark generalization.

The full repo is at <https://github.com/baladithyab/cuda-exploration>.
Happy to provide additional SASS dumps, the exact LLVM IR pre/post the
slice-read lowering, rerun with any candidate patches, or send a PR for
the MVP path (slice-read `!invariant.load` metadata) if that would be
welcome.
