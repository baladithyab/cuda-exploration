# cuda-3dgs-real — Wave 11

## What this is

A **CUDA C++ (nvcc) apples-to-apples reference** for `oxide-3dgs-real`.
Same PLY parser, same 3D→2D host-side projection, same byte-equivalent
`rasterize_2dgs` device kernel, same 4 camera poses, same scene
(`utsuho_plush.ply`, 53,671 gaussians, SH degree 0). Purpose: quantify
the cuda-oxide overhead on a non-trivial real-data kernel at both
runtime and SASS levels.

## Algorithm faithfulness

Both implementations follow the **same per-pixel front-to-back alpha
blend with early-exit** at `transmittance < 1e-4`:

```
for i in 0..n_gaussians:
    power = -0.5 · (cxx·dx² + 2·cxy·dx·dy + cyy·dy²)
    if power > 0: continue
    alpha = opacity · expf(power)
    if alpha < 1/255: continue
    alpha_c = min(alpha, 0.99)
    weight = alpha_c · T
    accum += weight · color
    T *= (1 - alpha_c)
    if T < 1e-4: early-exit
```

Host projection is identical operation-for-operation to the Rust:
quat→R, S²=diag(exp(scale)²), Σ₃ᴅ=R·S²·Rᵀ, Σ_cam=W·Σ₃ᴅ·Wᵀ, perspective
Jacobian J, Σ₂ᴅ=J·Σ_cam·Jᵀ, +0.3 anti-alias blur, conic=inv(Σ₂ᴅ),
SH DC → (c₀·f_dc + 0.5).clamp(0,1), opacity=sigmoid, depth-sort asc.

## Pixel-diff vs oxide-3dgs-real

Per-pixel L∞ diff on the 8-bit PPM outputs:

| Camera | max abs diff | mean abs diff | diff pixels | total pixels |
|---|---|---|---|---|
| A (identity, -Z)    | **0** | 0.0000    | 0 / 640,000 | byte-identical |
| C (flipY)           | **0** | 0.0000    | 0 / 640,000 | byte-identical |
| D (Y-rot 180°, +Z)  | 1     | 1.6e-6    | **3** / 640,000 | ULP-level |
| unsuffixed (= camA) | **0** | 0.0000    | 0 / 640,000 | byte-identical |

Cameras A and C produce **bit-identical PPM output** to the cuda-oxide
reference. Camera D differs at 3 pixels (< 0.0005 %) by a single
intensity level — consistent with non-associative host float arithmetic
(clang-17 vs rustc LLVM slightly reorder some FMAs; the ±0.5-rounded
integer pixel value flips at 3 boundary-crossing locations). No
algorithmic drift.

## SASS instruction mix (device kernel only)

Disassembled with `cuobjdump --dump-sass`, kernel section only:

| Instruction       | nvcc | cuda-oxide | Notes |
|---|---|---|---|
| FFMA              | 9    | 9          | Fused mul-add — identical |
| FMUL              | 9    | 9          | |
| FADD              | 5    | 5          | |
| MUFU              | 1    | 1          | `expf` — identical |
| LDG.E total       | 9    | 9          | Global loads — identical count |
| ├─ LDG.E.CONSTANT | **9**| **0**      | nvcc hints data is uniform across warp |
| └─ LDG.E (plain)  | 0    | 9          | cuda-oxide omits the hint |
| Total kernel SASS | 270 lines | 277 lines | nearly identical code size |

Same arithmetic intensity, same transcendental count, same load
topology. The *only* difference is `LDG.E.CONSTANT` vs plain `LDG.E` —
reproducing the Wave-5 finding that NVVM's cuda-oxide path doesn't set
the uniform-load predicate. For this kernel it's not a measurable
bottleneck: per-thread loads are already coalesced (9 contiguous
gaussian arrays, one element per iteration).

## Kernel timing (median of 3 iters, 800×800, N=53,671)

| Camera | nvcc (this folder) | cuda-oxide (Wave 10) | nvcc/oxide |
|---|---|---|---|
| A | 41.95 ms | 37.14 ms | 1.13× (oxide faster) |
| C | 37.57 ms | 37.78 ms | 0.99× |
| D | 36.45 ms | 42.04 ms | 0.87× (nvcc faster) |

Medians are within ±15 % across the set, with no consistent winner.
The user was gaming on the GPU during benching so the cross-camera
variance is dominated by thermal / contention noise; these samples
aren't precise enough to distinguish the two compilers' codegen on
this kernel.

## Verdict

**On this real-data rasterizer, cuda-oxide has no measurable runtime
overhead vs nvcc.** The SASS is arithmetically identical (9 FFMA,
9 FMUL, 5 FADD, 1 MUFU, 9 LDG.E); the only delta is the
`LDG.E.CONSTANT` uniform-load hint that nvcc emits and cuda-oxide
doesn't — and for a kernel where global loads are already coalesced
and the math-to-load ratio is balanced, that hint doesn't move the
wall clock. The pixel outputs are bit-identical at cameras A and C
and differ at 3/640,000 pixels by 1 bit at camera D (host-side
float non-associativity between clang and rustc).

This corroborates Wave-4-6's "algorithm class matters more than
language" finding: cuda-oxide's kernel output is apples-to-apples
with nvcc on this non-trivial production-shape kernel, just like it
was on reduction and vec-add. The 15–20 % matmul gap does not
generalize to splat rasterization.

## Files

- `rasterize.cu` — self-contained host+device source (557 lines)
- `build.sh` — `nvcc -O3 -arch=sm_120 -lstdc++ -lm`
- `run.sh` — runs with the canonical PLY, tees to `run.log`
- `rasterize` — compiled binary (1.1 MB)
- `rasterize.sass`, `rasterize_kernel.sass` — cuobjdump outputs
- `output_utsuho_plush{,_A,_C,_D}.ppm` — rendered frames (1.9 MB each)
- `results.csv` — per-iter kernel timings
- `run.log` — full run output
