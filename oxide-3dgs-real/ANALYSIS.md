# oxide-3dgs-real — Wave 9

## What this is

Renders a **real, public 3D Gaussian Splatting scene** through the existing
`oxide-3dgs-mini` kernel (a 2D forward rasterizer in cuda-oxide). The kernel
itself is *unchanged*. All 3D→2D work (quaternion-to-rotation, covariance
construction, perspective Jacobian, conic inversion, SH-DC-to-RGB, sigmoid
opacity, depth sort) happens host-side in Rust before the kernel launch.

## Scene

- **URL**: `https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/luigi/luigi.ply`
- **File size**: 988,183 bytes (≈ 966 KB)
- **Gaussian count**: 14,526
- **Properties**: `x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3` (17 floats/vertex = **SH degree 0 only**, no `f_rest_*`)
- **Scene bbox**: `min=(-0.55,-0.66,-0.27)  max=(0.54,0.65,0.27)`, diag ≈ 1.79
- **Centroid**: `(-0.002, -0.005, -0.017)` — already normalised to origin

This is an object-scale asset (not a room scan), which is ideal: no floor,
no distractors, tight frustum.

## Camera pose used (winning render: `camC_flipY`)

- Intrinsics: `fx = fy = 800`, `cx = cy = 400` (for 800×800 render)
- World→camera rotation `W = diag(1, -1, 1)` (flip Y axis — PLY appears to
  be y-down while our projection expects y-up in image coords)
- Translation `t = (-cx_centroid, +cy_centroid, -(cz_centroid - 1.5·diag))`
  ⇒ camera placed at ≈ 2.7 world units from centroid along world -Z.
- Depth range in camera frame: **`[2.43, 2.97]`** — entire scene at a nearly
  constant depth, confirming the camera is far enough that the object is
  fully enclosed in a narrow depth slab.
- Projected 2D means range: `x∈[233, 564]`, `y∈[211, 602]` — well inside the
  800×800 image, centred-ish, with plenty of headroom on all sides.

## Sanity checks (all passed)

- Gaussians total: 14,526; projected: **14,526** (0 culled). Expected for an
  object scene entirely in front of the camera.
- Conic scale median ≈ 53 → implies a typical σ² of ≈ 1/53 ≈ 0.019 in pixel
  units squared, i.e. splats of radius ≈ 0.14 px before the 0.3-anti-alias
  blur we add. The anti-alias blur dominates visible footprint — this is
  normal for a sub-megapixel distilled splat.
- Non-zero-pixel fraction: **9.6 %** (61,431 / 640,000). A figurine occupies
  a compact bounding box; this is in the expected range.
- Bounding box of non-zero output: x∈[213, 582], y∈[193, 619] — the rendered
  subject is ~370×420 px centered in the image. Consistent with projected
  2D-means range above.
- No NaNs, no all-black, no all-one-color: SH +0.5 shift and sigmoid-opacity
  are both visibly correct (colour cast matches a green/overalls-red Luigi).

## Timing

Measured with `cuEventRecord` / `elapsed_ms` on RTX 5090 (sm_120), one timed
launch per camera after a warmup launch, 14,526 gaussians, 800×800 grid:

| Camera               | gpu_ms | cpu_wall_ms |
|---------------------:|-------:|------------:|
| A (identity, -Z)     | 9.60   | 10.31       |
| C (flipY, -Z)        | 11.11  | 11.43       |
| D (Y-rot 180°, +Z)   | 9.58   | 9.87        |

Render time is dominated by the kernel's O(pixels × N) inner loop. 800×800
pixels × 14,526 gaussians ≈ 9.3 G inner iterations; 10 ms ⇒ ~930 Giter/s,
which is in the right ballpark for this naive untiled rasterizer.

## Final visual verdict

**Recognizable.** An ASCII-downsampled view of the output shows a clearly
humanoid silhouette: a vertical "cap + head" structure at top (two blobs
side-by-side: the hat crown + a small cranial highlight), spread-out arms
across the middle, torso, and two legs with feet at the bottom. Camera A
rendered Luigi upside-down (confirming a y-axis convention mismatch);
camera C with the Y-flip produced a right-side-up figurine.

## Failure modes observed and diagnosed

- **Camera B (`+Z no flip`)**: all 14,526 gaussians were culled as
  "behind camera" — expected, because it places the camera on the opposite
  side of the object; object ends up with negative camera-z. Not a bug,
  the cull worked.
- No conic-degeneracy failures (`culled_bad_cov=0`) — the 0.3 anti-alias
  blur is doing its job keeping Σ_2d positive-definite.

## Files

- `scenes/luigi.ply` — *gitignored* (988 KB is small but the policy is to
  not commit binary scene data; document the URL instead).
- `src/main.rs` — parser + projection + render driver (420 lines).
- `output_real_{A,C,D}.ppm` — raw PPM outputs (ignored).
- `/tmp/real_3dgs.png` — canonical PNG for handoff = `output_real_C.ppm`.

## Unchanged pieces (validates kernel reuse)

The kernel `rasterize_2dgs` is byte-identical to the one in
`oxide-3dgs-mini` — same function body, same signature (2D means, 3-float
conic, opacity, per-gaussian RGB). Only the host marshalling changed. This
confirms the Wave-8 kernel is general enough to render real projected
3DGS data with no kernel-side modification.

## Wave 10: canonical Utsuho-plush render

Extended the Wave-9 pipeline to render a second, richer scene with
**no kernel changes** and only a one-line scene-path swap in `main.rs`.

- **Scene**: `utsuho_plush.ply` — 13 MB, 53,671 gaussians, SH degree 3
  (62 float props incl. `f_rest_0..44`).
- **Source**: `solaaaa/sample-gaussian-splats` on HuggingFace
  (`datasets/solaaaa/sample-gaussian-splats/resolve/main/Utsuho%20Plush/utsuho_plush.ply`).
  Canonical 3DGS-format splat of a Touhou plush figurine — a clearly
  recognizable real-world object scan.
- **Bbox**: x∈[-0.97, 1.47], y∈[-2.15, 2.45], z∈[-2.59, 3.20], diag=7.79.
  Centroid ≈ (0.18, 0.12, 0.18). Tall, upright figurine.
- **Camera A (chosen)**: identity rotation, origin at
  `(cx, cy, cz - diag*1.5) = (0.18, 0.12, -11.5)`, looking down +Z in
  COLMAP convention. All 53,671 gaussians project, none culled,
  depth range [8.92, 14.71].
- **Render time (3-iter, cams A/C/D)**: 37.14 / 37.78 / 42.04 ms
  (median 37.78 ms at 800×800, N=53,671). Cam B (+Z side) culls
  everything, confirming COLMAP sign convention.
- **Visual verdict**: recognizable — tall plush figurine at
  approx. 384×203 px centered in frame, 21,759 distinct foreground
  colors, warm-brown mean (102, 93, 68) consistent with Utsuho's
  reddish-black plush coloring. SH degree 0 only (f_dc) used — the
  `f_rest_0..44` SH bands are present in the PLY but not evaluated
  yet. Basic render quality was already clearly recognizable so the
  optional SH-3 stretch goal was skipped.
- **Outputs**: `output_utsuho_plush.ppm` (= cam A), plus
  `output_utsuho_plush_{A,C,D}.ppm` per-camera. PNG at
  `/tmp/utsuho_plush.png`.

## Wave 11: nvcc apples-to-apples comparison

An nvcc CUDA-C++ reference (`cuda-3dgs-real/`) was built as an
algorithm-equivalent port of this folder's Rust-kernel pipeline: same
PLY parser, same per-gaussian projection math, same 4 cameras, same
scene, same kernel body. Purpose: quantify cuda-oxide overhead on a
non-trivial real-data kernel.

**Pixel-level:** camera A and camera C produce **byte-identical
PPM outputs** between cuda-oxide and nvcc. Camera D differs at
3 pixels out of 640,000 by 1 intensity level — consistent with
host-side float non-associativity (clang-17 vs rustc LLVM reorder
some FMAs so the final u8-quantized value crosses a boundary at 3
pixels). No algorithmic drift.

**SASS (device kernel only):** identical arithmetic mix —
FFMA=9 FMUL=9 FADD=5 MUFU=1, LDG.E total=9. **The only difference
is `LDG.E.CONSTANT` (nvcc, 9×) vs plain `LDG.E` (cuda-oxide, 9×)**,
reproducing the Wave-5 uniform-load-hint finding. Kernel section:
270 SASS lines (nvcc) vs 277 (cuda-oxide).

**Kernel timing (median of 3 iters, 800×800, N=53,671, user gaming
on GPU during run so noise is elevated):**

| Camera | oxide | nvcc | nvcc/oxide |
|---|---|---|---|
| A | 37.14 ms | 41.95 ms | 1.13× |
| C | 37.78 ms | 37.57 ms | 0.99× |
| D | 42.04 ms | 36.45 ms | 0.87× |

Within ±15 %, no consistent winner. The `LDG.E.CONSTANT` hint
nvcc emits does not translate to a measurable wall-clock delta on
this kernel — loads are already coalesced, the math-to-load ratio
is balanced, and the L1/TEX path absorbs the "not-flagged-uniform"
penalty without a visible cost at this N.

**Verdict:** on this 3DGS rasterize kernel, cuda-oxide is effectively
tied with nvcc at both the SASS and runtime level. The matmul-scale
codegen gap does not generalize to front-to-back alpha-blend splat
rasterization, consistent with Wave-4-6's "algorithm class matters
more than language" finding.
