# oxide-3dgs-mini: Rudimentary 2D Gaussian Splatting forward rasterizer in cuda-oxide

## What 3DGS is (one paragraph)

3D Gaussian Splatting represents a scene as a soup of N anisotropic 3D
Gaussians, each with mean `μ ∈ ℝ³`, covariance `Σ ∈ ℝ^{3×3}`, opacity `α`, and
view-dependent color (typically encoded as spherical harmonics).
Forward rasterization projects every Gaussian to 2D screen space (yielding a
2D mean and a 2D covariance), then for each pixel alpha-blends the Gaussians
that overlap it in front-to-back depth order. Production implementations
(gsplat, the official 3DGS CUDA code) tile-bin the projected Gaussians into
16×16 screen tiles, sort per-tile by depth, and use a tiled kernel where
each block cooperatively rasterizes one tile. Backward pass differentiates
through the alpha blend to learn means/covariances/colors.

## Simplifications taken

This implementation is a forward-only proof of life designed to exercise
cuda-oxide's expressiveness, not a renderer:

- **2D inputs directly.** Means and covariances are already 2D; no 3D→2D
  projection.
- **Diagonal covariance only.** Σ on host is `diag(σ_x², σ_y²)` (off-diagonal
  zero). Conic = inv(Σ) is precomputed on host and passed as three flat
  float arrays (`conic_xx, conic_xy, conic_yy`).
- **No tile binning.** The kernel is per-pixel, brute-force: every pixel
  iterates over every Gaussian. O(N·W·H) work — fine for N=512, W=H=256.
- **No spherical harmonics.** Each Gaussian carries direct RGB.
- **No on-GPU sort.** Host sorts by `depth = my + 0.1·mx` so input order is
  front-to-back.
- **No backward pass / no training.**

## Kernel structure

One thread per pixel, grid `(W/16, H/16, 1)`, block `(16, 16, 1)`. Per-thread
state: `accum_{r,g,b}` and `transmittance`. The body is a single while-loop
over all gaussians with three early-skip / early-exit branches:

1. `power > 0` → outside of the Gaussian's positive lobe → skip.
2. `alpha < 1/255` → contribution below pixel quantization → skip.
3. `transmittance < 1e-4` → pixel saturated → write & return.

`expf` uses `core::intrinsics::expf32`. cuda-oxide's mir-importer recognizes
that intrinsic (`float_math.rs:118`) and mir-lower routes it to libdevice
`__nv_expf` (`call.rs:259`).

## PTX / SASS inspection

Two artifacts are produced: the full LLVM IR (`oxide_3dgs_mini.ll`, NVVM
dialect emitted by cuda-oxide) and a runtime cubin
(`oxide_3dgs_mini.cubin`) compiled via libNVVM at `load_kernel_module`
time. PTX shown here is from `llc-21` re-translation of the IR to give a
human-readable view; the cubin SASS is what actually executes.

LLC-emitted PTX summary:

```
lines        : 240
bra          : 18      (control flow not flattened)
fma          : 0       (llc-21 default doesn't contract)
call         : 3       (one __nv_expf call per branch path)
__nv_expf    : 3
```

Cubin SASS summary (sm_120, RTX 5090):

```
lines        : 281
FFMA         :   9     (libNVVM did contract muls+adds)
BRA/BSYNC    :  10     (loop back-edge + early-exit predication)
MUFU         :   1     (libNVVM inlined __nv_expf to MUFU.EX2)
LDG.E        :   9     (one per gaussian-array load: mx, my, cxx, cxy,
                        cyy, op, cr, cg, cb — exactly nine streams)
STG.E        :   6     (3 outputs × 2 write-paths: early-exit + fall-through)
```

The loop is **not unrolled**: a single LDG per array stream and a single
FFMA pack execute per iteration, then BRA back. Predicated control flow
appears as @P0/BSSY/BSYNC pairs around the early-exit; the
`transmittance < 1e-4` exit is a real branch, not predication-folded. This
is what you'd hand-write — N is unknown at compile time, so unrolling
isn't possible without a versioned/specialized kernel, and the hot
branch is too divergent to predicate cleanly.

`__nv_expf` did **not** survive as a function call in the cubin —
libNVVM inlined it to a single `MUFU.EX2` (hardware exp2 unit) plus a
multiplicative ln(2) scale, exactly as nvcc CUDA C++ would.

## Performance

256×256 image, 512 gaussians, 5 timed iters on RTX 5090, sm_120 native
build. Background note: workstation is gaming-loaded — variance is real.

```
iter 0: gpu_ms=0.069
iter 1: gpu_ms=0.078
iter 2: gpu_ms=0.080
iter 3: gpu_ms=0.075
iter 4: gpu_ms=0.075
median = 0.075 ms,  best = 0.069 ms
```

Naive flop estimate per iter: `W·H·N·~20 flops ≈ 670 MFLOP`. At 0.075 ms
that's `~8.9 TFLOP/s`. The early-exit is doing real work — most pixels
hit `transmittance < 1e-4` long before iterating all 512 gaussians once
the planted "big red" gaussian and a handful of others have stacked. The
sub-100µs frame time is plausible given N is small and most threads exit
early.

The 8.9 TF/s number should be read as *compute throughput on the work
that actually executed*, not as fraction-of-peak — the kernel is heavily
memory-bound on the 9 stream loads when the early-exit doesn't fire,
and the gaussian arrays fit comfortably in L1 at N=512 so there's
essentially no cache miss cost.

## Correctness sanity

We plant a known dominant Gaussian at `(128, 128)` with `σ=30, α=1.0,
RGB=(1.0, 0.05, 0.05)`. Output at that pixel:

```
pixel(128,128) = (0.928, 0.129, 0.126)   PASS
```

R is overwhelmingly dominant; G and B come from background gaussians
behind it that haven't been fully alpha-killed. This matches expectation.

## Verdict

**cuda-oxide handled this kernel cleanly.** No API friction, no
type-system hassles, no missing intrinsic. The notable findings:

- `core::intrinsics::expf32` "just works" → libdevice `__nv_expf` →
  inlined to `MUFU.EX2` SASS by libNVVM.
- Multi-array slice arguments (we pass nine `&[f32]` and three
  `DisjointSlice<f32>`) compose without trouble; `cuda_launch!` accepts
  them positionally.
- Control flow with multiple `continue`/early-`return` paths inside a
  single while-loop is lowered to clean predicated branches in SASS;
  the early-exit is a real BRA, not a predicate.
- libNVVM did contract `*+` into FFMA in this kernel (9 FFMAs in SASS,
  zero in llc-21 PTX). This is interesting next to the matmul finding
  where naive `acc += a * b` *didn't* yield FFMA at the SASS level —
  worth a follow-up to characterize when contraction fires.

No build issues, no runtime issues, output looks plausible visually
(see `output.ppm`).

## Scene quality (Wave 8.5)

The Wave 8 test scene was 512 gaussians with uniformly random position, size,
color, and depth — this produced a convincing but uninterpretable pastel-noise
image. It validated the pipeline but didn't show that the rasterizer can
actually render a structured image. Wave 8.5 replaces the random generator
with two procedural scenes that make the output unambiguously recognizable:

- **`output_rings.ppm`** — 4096 gaussians arranged on 16 concentric rings
  around (128, 128), radius 8–120, hue varying with radius (rainbow). σ=4–6,
  opacity 0.7–0.9, depth = −radius so innermost rings render frontmost under
  the kernel's front-to-back / transmittance-early-exit blend (first gaussian
  in the array = frontmost).

- **`output_smiley.ppm`** — ~6025 gaussians: sky-blue background on a 55×55
  jittered grid (σ=12, opacity 0.3, depth −1 = backmost), a yellow face disk
  of 2000 gaussians inside radius 80 (σ=6, opacity 0.85, depth −100), two
  black eye disks at (104, 108) and (152, 108) (250 gaussians each), and a
  500-gaussian black mouth on the lower half of a radius-30 arc. Eyes and
  mouth use depth −1000 so they occlude the face.

Depth sort is ASCENDING (smallest/most-negative first), matching the kernel's
front-to-back convention: the gaussian at index 0 is drawn first into full
transmittance; once transmittance drops below 1e-4 the loop early-exits.
Swapping the sort direction produces a visibly broken smiley (background
occluding the face), so this is a genuine functional check on the renderer's
alpha-compositing order as well as a visual demo.

No kernel changes; only the host-side gaussian array construction changed.
Kernel timings with N=4096 (rings) are ~1.0 ms best / 1.7 ms median, and
N=6025 (smiley) ~0.8 ms best / 2.1 ms median on the RTX 5090 box, similar
to the Wave 8 N=512 numbers once you account for the ~10× more gaussians
— i.e. the per-gaussian inner loop is cheap and the transmittance early
exit is doing real work on the smiley (saturated-opacity central face +
eyes short-circuit most of the inner loop for opaque pixels).
