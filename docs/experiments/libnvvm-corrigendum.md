# Wave 3 corrigendum: the libNVVM mystery

**Date:** 2026-05-08, late in Wave 3
**Severity:** Critical — invalidates the v0 + Wave 1+2 "Rust safety tax = 2.5×" headline.

## What we discovered

While testing the new `matmul_fmuladd` kernel (W3A), `cargo oxide run` failed with:

```
load: Nvvm(Call { operation: "nvvmCompileProgram", code: 7,
  log: Some("libnvvm : error: -arch=compute_120 is an unsupported option") })
```

The libNVVM `cuda-oxide` was loading at runtime didn't recognize the Blackwell `compute_120` arch.

## Root cause

`/usr/lib/x86_64-linux-gnu/libnvvm.so.4` exists at the system loader's path. `ls -la` shows:
```
/usr/lib/x86_64-linux-gnu/libnvvm.so.4 -> libnvvm.so.4.0.0
/usr/lib/x86_64-linux-gnu/libnvvm.so.4.0.0  (27 MB, dated Jan 28 2023)
```

`strings` confirms this is libNVVM 7.0.1 from CUDA 12.0:
```
$ strings /usr/lib/x86_64-linux-gnu/libnvvm.so.4 | grep arch=compute_
-arch=compute_50, _52, _53, _60, _61, _62, _70, _72, _75, _80, _86, _87, _89, _90, _90a
```

The CUDA 13.2 toolkit at `/usr/local/cuda` has the modern libNVVM 22.0.0:
```
$ strings /usr/local/cuda/nvvm/lib64/libnvvm.so | grep arch=compute_
... _100, _100a, _100f, _103, _103a, _103f, _110, _110a, _110f, _120, _120a, _120f, _121, _121a, _121f
```

cuda-oxide's `libnvvm-sys` crate (`crates/libnvvm-sys/src/lib.rs:385`) tries `libnvvm.so.4`, `libnvvm.so.3`, `libnvvm.so` against the system loader **before** falling back to `CUDA_HOME/nvvm/lib64/libnvvm.so`. The system loader resolves `libnvvm.so.4` to the older 2023 binary, so cuda-oxide silently uses an outdated libNVVM that:

1. **Doesn't recognize `compute_120`** — failed our Wave 3 builds outright once we used the fmuladd kernel (which needs libNVVM, not just llc).
2. **Generates lower-quality code for the bounds-checked slice indexing** — explains why Wave 1's `oxide/safe` was 2.5× slower than `oxide/unchecked`. With the modern libNVVM, the difference largely vanishes.

This kind of dual-install / silent-shadow pattern is a recurring class of bug in CUDA setups.

## The fix

Set `CUDA_HOME=/usr/local/cuda` in the environment before any `cargo oxide` invocation, OR set `LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so` directly:

```bash
export CUDA_HOME=/usr/local/cuda
export LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so
```

This forces cuda-oxide to use libNVVM 22.0.0 from CUDA 13.2 — which knows `compute_120` and emits modern code.

## Result delta

After applying the fix and re-running the full Wave 1/3 oxide-matmul bench:

| kernel  | N=1024 (med TF) | N=4096 (med TF) |  Δ vs old |
|---------|----------------:|----------------:|---------:|
| safe       | 6.94 (old: 2.80) | 4.84 (old: 2.01) | **+2.4–2.7×** |
| unchecked  | 6.95 (old: 6.51) | 5.13 (old: 4.96) | small / noise |
| fmuladd    | 6.92            | 5.70            | new |

**The "Rust safety tax" essentially disappears.** At N=1024, all three kernels hit 6.92-6.95 TFLOPS — within 1% of each other and 1% of nvcc's 6.88. The 2.5× tax we reported in v0 was a libNVVM artifact, not a property of cuda-oxide's design.

The `oxide/fmuladd` kernel ALSO works as the W3A subagent predicted — `core::intrinsics::fmuladdf32` lowers to `__nv_fmaf` libdevice call, which nvJitLink resolves to `fma.rn.f32` SASS at module load time. So the upstream issue draft was partially wrong in its framing: **users CAN access FMA today via `core::intrinsics::fmuladdf32`** (or any other path that lowers to `__nv_fmaf`). The `--fast-math` issue the upstream draft flagged is still a real defect in the default `*+` codegen path, but a working escape hatch exists today.

## Action items for the upstream issue

The earlier upstream-issue-fma.md draft needs revision:
1. Soften the "no escape hatch" framing — `core::intrinsics::fmuladdf32` is one.
2. Keep the FastmathFlags::default() finding — fixing it would let plain `*+` contract too, which is the more general win.
3. Add a NEW bug: `libnvvm-sys` should preferentially honor `CUDA_HOME/nvvm/lib64/libnvvm.so` over the system soname, OR doctor should detect the version mismatch.

## Lesson for users

If you're on a system where `/usr/local/cuda` has a newer toolkit than what was originally installed via apt, **always set CUDA_HOME explicitly** when working with cuda-oxide or any tool that uses libNVVM via dlopen. Doctor's "libNVVM 2.0" output is misleading — it's reporting the IR version, not the toolkit version, so you can't tell from doctor alone which libNVVM you got.
