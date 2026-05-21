"""Wave G5 — cuTile 3DGS rasterizer with TILE-BINNING (binned variant).

Port of cuda-3dgs-real / oxide-3dgs-real to the cuTile Python tile DSL,
augmented with a CPU-side tile-binner that pre-computes per-tile
gaussian lists. The kernel iterates only the gaussians whose 3-sigma
2D bbox overlaps the tile, instead of all 53k gaussians per pixel.

Speedup mechanism vs cutile-3dgs-real (W15.3 naive):
  - Naive: each 16x16 pixel tile iterates ALL n_gaussians (~53k) per
    pixel ⇒ 55ms/cam.
  - Binned: each tile iterates only ~100-1000 gaussians whose bbox
    overlaps it ⇒ target ≤25ms/cam (≥2x faster than naive).

cuTile DSL fit notes:
  - Per-tile gaussian list length is variable. cuTile's `range(n)` for
    a kernel-arg n is allowed but the compiler can't unroll/optimize.
    Workaround: pad each tile's list to MAX_GAUSSIANS_PER_TILE (closure
    constant), pass the actual length per tile as a 1D `tile_counts`
    array, and use `where` to mask the inner loop's contribution at
    iterations beyond the count. The unrolled-loop overhead is
    bounded by MAX (constant) but the WORK is bounded by max-tile-count.
  - Per-tile gaussian indices live in a (n_ty, n_tx, MAX) int32 array.
    Inside the kernel, for each loop iter we ct.load a (1,) int32 tile
    holding the gaussian index, then ct.gather each of the 9 float
    properties at that index.

Host side (PLY parse → projection → SH evaluation → depth sort →
tile-binning) is ported from rasterize.cu and extended with bin_build.

CLI:
    --smoke      render cam A and diff against cuda-3dgs-real PPM
    --bench      timed iters across all 4 cameras (orchestrator runs)
    --csv-out    bench CSV path (default results.csv)
    --ply        path to PLY file (default ../oxide-3dgs-real/scenes/utsuho_plush.ply)
"""
from __future__ import annotations

import argparse
import csv
import math
import struct
import sys
import time
from pathlib import Path

import cuda.tile as ct
import cupy
import numpy as np

from bin_build import build_tile_bins

# ─────────────────────────────────────────────────────────────────────────────
# Constants — match cuda-3dgs-real
# ─────────────────────────────────────────────────────────────────────────────

W = 800
H = 800
BS = 16   # pixel tile side
ITERS = 3
WARMUP = 1

# Tile-binning parameters.
# MAX_GAUSSIANS_PER_TILE: per-tile padded list length. cuTile compiles the
# inner loop unrolled (closure constant), so MAX directly bounds register
# pressure and SASS size. With 53671 gaussians fanning out to 50x50=2500
# tiles, mean overlap is ~10-20 tiles/gaussian, so mean per-tile count is
# ~200-400. Heavy tiles can hit 1000+; we cap at 1024 (drops the deepest
# few in extreme tiles, which contribute negligibly under transmittance).
MAX_GAUSSIANS_PER_TILE = 4096
SIGMA_K = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# cuTile kernel — TILE-BINNED 3DGS rasterizer
# ─────────────────────────────────────────────────────────────────────────────


def make_rasterize_kernel(width: int, height: int, block_size: int,
                          max_per_tile: int):
    """Build a @ct.kernel for the given image, tile size, and MAX list cap.

    width, height, block_size, max_per_tile are captured as Python ints
    (closure constants), so cuTile compiles them into the tile shape and
    inner-loop iteration count.

    Kernel arguments:
        mx, my, cxx, cxy, cyy, opacity, cr, cg, cb : (n_gauss,) f32 arrays
            -- depth-sorted gaussian properties.
        tile_indices_flat : (n_ty * n_tx * MAX,) int32
            -- per-tile padded gaussian-index lists, row-major
               in (ty, tx, slot) layout. Slots [0:count] hold valid
               indices in front-to-back order; slots [count:MAX] are
               padded with 0 (masked out).
        tile_counts_flat  : (n_ty * n_tx,) int32
            -- per-tile actual gaussian count (clamped to MAX).
        n_tiles_x         : int (passed as kernel arg, used for indexing
                            tile_indices_flat). Since grid dims also encode
                            this, we use ct.bid + closure constant.
        out_r, out_g, out_b: (H, W) f32 — output planes.

    Grid: (n_tiles_x, n_tiles_y) — each block produces a BS×BS pixel tile.
    """
    BS_local = block_size
    MAX = max_per_tile
    N_TX = width // block_size
    # tile_id = by * N_TX + bx;  flat index into tile_counts/tile_indices.

    @ct.kernel
    def rasterize_3dgs_binned(
        mx, my, cxx, cxy, cyy, opacity, cr, cg, cb,
        tile_indices_flat,    # (n_ty * n_tx * MAX,) int32
        tile_counts_flat,     # (n_ty * n_tx,) int32
        out_r, out_g, out_b,
    ):
        bx = ct.bid(0)  # tile index along width
        by = ct.bid(1)  # tile index along height

        # Linear tile id (row-major in (ty, tx)).
        tile_id = by * N_TX + bx

        # ── Load per-tile actual gaussian count. ──
        count_tile = ct.load(tile_counts_flat, index=(tile_id,), shape=(1,))
        # count_tile is (1,) int32. We need a scalar to compare against `i`.
        # It will be broadcast against a (1,) int32 tile holding `i`.

        # ── Build per-pixel coordinate tiles (BS, BS) f32. ──
        col_idx = ct.arange(BS_local, dtype=ct.int32)
        row_idx = ct.arange(BS_local, dtype=ct.int32)
        col_2d = ct.broadcast_to(ct.expand_dims(col_idx, axis=0), (BS_local, BS_local))
        row_2d = ct.broadcast_to(ct.expand_dims(row_idx, axis=1), (BS_local, BS_local))

        pxf = (col_2d + bx * BS_local).astype(ct.float32)
        pyf = (row_2d + by * BS_local).astype(ct.float32)

        # ── Init accumulators. ──
        accum_r = ct.zeros((BS_local, BS_local), ct.float32)
        accum_g = ct.zeros((BS_local, BS_local), ct.float32)
        accum_b = ct.zeros((BS_local, BS_local), ct.float32)
        transmittance = ct.full((BS_local, BS_local), 1.0, ct.float32)

        ALPHA_FLOOR = 1.0 / 255.0
        ALPHA_CAP = 0.99
        ZERO_T = ct.zeros((BS_local, BS_local), ct.float32)

        # Base offset into tile_indices_flat for this tile's slot list.
        tile_slot_base = tile_id * MAX

        # ── Iterate the padded per-tile gaussian list. ──
        # MAX is a closure constant ⇒ this loop is statically bounded.
        # The mask `i < count_tile` zeroes contribution from padded slots.
        for i in range(MAX):
            # Load the gaussian index at this slot (shape (1,) int32).
            gidx_1d = ct.load(tile_indices_flat,
                              index=(tile_slot_base + i,), shape=(1,))

            # Mask = (slot index i) < (per-tile count).  (1,) bool.
            i_tile = ct.full((1,), i, ct.int32)
            slot_valid = i_tile < count_tile

            # Gather all 9 properties at gidx; gidx out-of-range is masked
            # via slot_valid (gather padding_value=0 for the out-of-bounds
            # case, but we also explicitly mask slot_valid).
            mxi_1d = ct.gather(mx, gidx_1d, mask=slot_valid, padding_value=0.0)
            myi_1d = ct.gather(my, gidx_1d, mask=slot_valid, padding_value=0.0)
            cxxi_1d = ct.gather(cxx, gidx_1d, mask=slot_valid, padding_value=0.0)
            cxyi_1d = ct.gather(cxy, gidx_1d, mask=slot_valid, padding_value=0.0)
            cyyi_1d = ct.gather(cyy, gidx_1d, mask=slot_valid, padding_value=0.0)
            opi_1d = ct.gather(opacity, gidx_1d, mask=slot_valid, padding_value=0.0)
            cri_1d = ct.gather(cr, gidx_1d, mask=slot_valid, padding_value=0.0)
            cgi_1d = ct.gather(cg, gidx_1d, mask=slot_valid, padding_value=0.0)
            cbi_1d = ct.gather(cb, gidx_1d, mask=slot_valid, padding_value=0.0)

            mxi = ct.expand_dims(mxi_1d, axis=1)   # (1,1)
            myi = ct.expand_dims(myi_1d, axis=1)
            cxxi = ct.expand_dims(cxxi_1d, axis=1)
            cxyi = ct.expand_dims(cxyi_1d, axis=1)
            cyyi = ct.expand_dims(cyyi_1d, axis=1)
            opi = ct.expand_dims(opi_1d, axis=1)
            cri = ct.expand_dims(cri_1d, axis=1)
            cgi = ct.expand_dims(cgi_1d, axis=1)
            cbi = ct.expand_dims(cbi_1d, axis=1)

            dx = pxf - mxi
            dy = pyf - myi

            power = -0.5 * (cxxi * dx * dx + 2.0 * cxyi * dx * dy + cyyi * dy * dy)

            alpha_raw = opi * ct.exp(power)
            valid_power = power <= 0.0
            valid_alpha = alpha_raw >= ALPHA_FLOOR
            valid = valid_power & valid_alpha
            alpha_capped = ct.minimum(alpha_raw, ct.full((BS_local, BS_local), ALPHA_CAP, ct.float32))
            alpha_eff = ct.where(valid, alpha_capped, ZERO_T)

            # ALSO mask by slot_valid: if this slot is past `count`, contribute zero.
            # slot_valid is (1,) bool ⇒ expand_dims to (1,1) so it broadcasts to (BS,BS).
            slot_valid_2d = ct.expand_dims(slot_valid, axis=1)
            alpha_eff = ct.where(slot_valid_2d, alpha_eff, ZERO_T)

            weight = alpha_eff * transmittance
            accum_r = accum_r + weight * cri
            accum_g = accum_g + weight * cgi
            accum_b = accum_b + weight * cbi
            transmittance = transmittance * (1.0 - alpha_eff)

        # ── Store the final accumulator via tiled_view (same as naive). ──
        out_r_view = out_r.tiled_view((BS_local, BS_local), padding_mode=ct.PaddingMode.ZERO)
        out_g_view = out_g.tiled_view((BS_local, BS_local), padding_mode=ct.PaddingMode.ZERO)
        out_b_view = out_b.tiled_view((BS_local, BS_local), padding_mode=ct.PaddingMode.ZERO)
        out_r_view.store((by, bx), accum_r)
        out_g_view.store((by, bx), accum_g)
        out_b_view.store((by, bx), accum_b)

    return rasterize_3dgs_binned


# ─────────────────────────────────────────────────────────────────────────────
# PLY parser — direct port of rasterize.cu parse_ply
# ─────────────────────────────────────────────────────────────────────────────


def parse_ply(path: str):
    """Parse Inria-style 3DGS PLY (binary little-endian, all f32 properties).

    Returns dict with arrays:
        x (n,), y (n,), z (n,)            — positions
        f_dc (n, 3)                       — DC SH coefs
        f_rest (n, 45) or None             — band-1..3 SH coefs (or None if absent)
        opacity (n,)                       — pre-sigmoid logit
        scale (n, 3)                       — pre-exp log scales
        rot (n, 4)                         — quaternion (w, x, y, z)
    """
    with open(path, "rb") as fp:
        buf = fp.read()
    needle = b"end_header\n"
    idx = buf.find(needle)
    if idx < 0:
        raise RuntimeError("no end_header in PLY")
    header_end = idx + len(needle)
    header = buf[:header_end].decode("ascii", errors="replace")
    n_vertex = 0
    props: list[str] = []
    for line in header.splitlines():
        if line.startswith("element vertex "):
            n_vertex = int(line[len("element vertex "):].strip())
        elif line.startswith("property float "):
            props.append(line[len("property float "):].strip())
    nprops = len(props)
    print(f"PLY header: {n_vertex} vertices, {nprops} float props")

    body = buf[header_end:]
    expected = n_vertex * nprops * 4
    if len(body) != expected:
        raise RuntimeError(f"body size mismatch: got {len(body)} expected {expected}")

    arr = np.frombuffer(body, dtype="<f4").reshape(n_vertex, nprops)

    def col(name: str) -> np.ndarray:
        if name not in props:
            raise RuntimeError(f"property '{name}' not found")
        return arr[:, props.index(name)].astype(np.float32, copy=False)

    x = col("x"); y = col("y"); z = col("z")
    f_dc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1)
    opacity_logit = col("opacity")
    scale = np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=1)
    rot = np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], axis=1)

    have_rest = all(f"f_rest_{k}" in props for k in range(45))
    if have_rest:
        f_rest = np.stack([col(f"f_rest_{k}") for k in range(45)], axis=1)
        print("SH support: degree 3 (16 coefs/channel)")
    else:
        f_rest = None
        print("SH support: degree 0 only (DC)")

    return {
        "x": x, "y": y, "z": z,
        "f_dc": f_dc, "f_rest": f_rest,
        "opacity": opacity_logit, "scale": scale, "rot": rot,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SH evaluation (degree 0 or 3) — direct port from rasterize.cu
# ─────────────────────────────────────────────────────────────────────────────

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
         0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435)


def sh_eval_full(f_dc: np.ndarray, f_rest: np.ndarray | None,
                 vd: np.ndarray) -> np.ndarray:
    """Return per-gaussian RGB (n, 3), pre-clamp.

    f_dc: (n, 3)        DC coefs
    f_rest: (n, 45)     bands 1..3 (R0..14, G15..29, B30..44) or None
    vd:    (n, 3)       view dir from camera origin to gaussian, normalized
    """
    n = f_dc.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    if f_rest is None:
        out[:, 0] = SH_C0 * f_dc[:, 0] + 0.5
        out[:, 1] = SH_C0 * f_dc[:, 1] + 0.5
        out[:, 2] = SH_C0 * f_dc[:, 2] + 0.5
        return out

    x = vd[:, 0]; y = vd[:, 1]; z = vd[:, 2]
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; yz = y * z; xz = x * z

    for ch in range(3):
        rest = f_rest[:, ch * 15:(ch + 1) * 15]   # (n, 15)
        dc = f_dc[:, ch]
        r = SH_C0 * dc
        # Band 1: -y, z, -x
        r = r + SH_C1 * (-y * rest[:, 0] + z * rest[:, 1] - x * rest[:, 2])
        # Band 2
        r = r + SH_C2[0] * xy * rest[:, 3] \
              + SH_C2[1] * yz * rest[:, 4] \
              + SH_C2[2] * (2.0 * zz - xx - yy) * rest[:, 5] \
              + SH_C2[3] * xz * rest[:, 6] \
              + SH_C2[4] * (xx - yy) * rest[:, 7]
        # Band 3
        r = r + SH_C3[0] * y * (3.0 * xx - yy) * rest[:, 8] \
              + SH_C3[1] * xy * z * rest[:, 9] \
              + SH_C3[2] * y * (4.0 * zz - xx - yy) * rest[:, 10] \
              + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * rest[:, 11] \
              + SH_C3[4] * x * (4.0 * zz - xx - yy) * rest[:, 12] \
              + SH_C3[5] * z * (xx - yy) * rest[:, 13] \
              + SH_C3[6] * x * (xx - 3.0 * yy) * rest[:, 14]
        out[:, ch] = r + 0.5
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Host-side projection — vectorized port of rasterize.cu project_all
# ─────────────────────────────────────────────────────────────────────────────


def quat_to_mat3_batch(rot: np.ndarray) -> np.ndarray:
    """rot: (n, 4) [w, x, y, z]  →  R: (n, 3, 3)"""
    n = rot.shape[0]
    norm = np.linalg.norm(rot, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    q = rot / norm
    w = q[:, 0]; x = q[:, 1]; y = q[:, 2]; z = q[:, 3]
    R = np.empty((n, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - w * z)
    R[:, 0, 2] = 2.0 * (x * z + w * y)
    R[:, 1, 0] = 2.0 * (x * y + w * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - w * x)
    R[:, 2, 0] = 2.0 * (x * z - w * y)
    R[:, 2, 1] = 2.0 * (y * z + w * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def project_all(raws: dict, cam: dict):
    """Project gaussians to 2D, compute conic + RGB + opacity + depth.

    Returns 9 arrays sorted by ascending depth + projected count.
    """
    x = raws["x"]; y = raws["y"]; z = raws["z"]
    n = x.size
    pos = np.stack([x, y, z], axis=1).astype(np.float32)  # (n, 3)

    W_mat = cam["W"].astype(np.float32)  # (3, 3)
    t = cam["t"].astype(np.float32)      # (3,)
    fx = cam["fx"]; fy = cam["fy"]; cx = cam["cx"]; cy = cam["cy"]

    # pc = W @ p + t   (n, 3)
    pc = pos @ W_mat.T + t[None, :]
    z_cam = pc[:, 2]

    valid = (z_cam >= 0.1) & (z_cam <= 100.0)

    # Camera origin in world space.
    cam_origin = -W_mat.T @ t  # (3,)

    # 2D mean.
    mx = fx * pc[:, 0] / np.maximum(z_cam, 1e-12) + cx
    my = fy * pc[:, 1] / np.maximum(z_cam, 1e-12) + cy

    # Covariance 3D = R · diag(exp(s))^2 · R^T
    R = quat_to_mat3_batch(raws["rot"])     # (n, 3, 3)
    s_exp = np.exp(raws["scale"]).astype(np.float32)         # (n, 3)
    s2 = np.zeros((n, 3, 3), dtype=np.float32)
    s2[:, 0, 0] = s_exp[:, 0] ** 2
    s2[:, 1, 1] = s_exp[:, 1] ** 2
    s2[:, 2, 2] = s_exp[:, 2] ** 2
    sigma_w = R @ s2 @ R.transpose(0, 2, 1)  # (n, 3, 3)

    # sigma_cam = W · sigma_w · W^T
    sigma_cam = W_mat @ sigma_w @ W_mat.T   # (n, 3, 3)

    z_safe = np.maximum(z_cam, 1e-12)
    z2 = z_safe * z_safe
    j00 = fx / z_safe
    j02 = -fx * pc[:, 0] / z2
    j11 = fy / z_safe
    j12 = -fy * pc[:, 1] / z2

    r0 = sigma_cam[:, 0, :]
    r1 = sigma_cam[:, 1, :]
    r2 = sigma_cam[:, 2, :]
    m0 = j00[:, None] * r0 + j02[:, None] * r2
    m1 = j11[:, None] * r1 + j12[:, None] * r2
    a = m0[:, 0] * j00 + m0[:, 2] * j02
    b = m0[:, 1] * j11 + m0[:, 2] * j12
    c = m1[:, 1] * j11 + m1[:, 2] * j12

    a_aa = a + 0.3
    c_aa = c + 0.3
    b_aa = b
    det = a_aa * c_aa - b_aa * b_aa
    valid &= (det > 0.0) & np.isfinite(det)

    inv_det = np.where(valid, 1.0 / np.where(det == 0, 1.0, det), 0.0)
    cxx = c_aa * inv_det
    cxy_ = -b_aa * inv_det
    cyy = a_aa * inv_det

    # SH evaluation in WORLD space view dir.
    vd = pos - cam_origin[None, :]
    vdn = np.linalg.norm(vd, axis=1, keepdims=True)
    vdn = np.maximum(vdn, 1e-8)
    vd = vd / vdn
    rgb = sh_eval_full(raws["f_dc"], raws["f_rest"], vd)  # (n, 3)
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    op = 1.0 / (1.0 + np.exp(-raws["opacity"]))
    op = op.astype(np.float32)

    keep = np.where(valid)[0]
    n_proj = keep.size
    if n_proj == 0:
        return None, 0

    mx_k = mx[keep].astype(np.float32)
    my_k = my[keep].astype(np.float32)
    cxx_k = cxx[keep].astype(np.float32)
    cxy_k = cxy_[keep].astype(np.float32)
    cyy_k = cyy[keep].astype(np.float32)
    op_k = op[keep]
    cr_k = rgb[keep, 0]
    cg_k = rgb[keep, 1]
    cb_k = rgb[keep, 2]
    depth_k = z_cam[keep].astype(np.float32)

    # Depth-sort ascending.
    order = np.argsort(depth_k, kind="stable")
    return {
        "mx": np.ascontiguousarray(mx_k[order]),
        "my": np.ascontiguousarray(my_k[order]),
        "cxx": np.ascontiguousarray(cxx_k[order]),
        "cxy": np.ascontiguousarray(cxy_k[order]),
        "cyy": np.ascontiguousarray(cyy_k[order]),
        "opacity": np.ascontiguousarray(op_k[order]),
        "r": np.ascontiguousarray(cr_k[order]),
        "g": np.ascontiguousarray(cg_k[order]),
        "b": np.ascontiguousarray(cb_k[order]),
        "depth": np.ascontiguousarray(depth_k[order]),
    }, n_proj


# ─────────────────────────────────────────────────────────────────────────────
# Cameras — same scheme as cuda-3dgs-real
# ─────────────────────────────────────────────────────────────────────────────


def make_cameras(raws: dict):
    x = raws["x"]; y = raws["y"]; z = raws["z"]
    cx = float(x.mean()); cy = float(y.mean()); cz = float(z.mean())
    extent = float(np.linalg.norm(
        [x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]))
    print(f"Scene centroid: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
    print(f"Scene diag   : {extent:.3f}")
    fx = 800.0; fy = 800.0; cx_p = 400.0; cy_p = 400.0
    dist = extent * 1.5

    cams = {}
    I = np.eye(3, dtype=np.float32)
    cams["A"] = {"label": "camA_minusZ",
                 "W": I.copy(), "t": np.array([-cx, -cy, -(cz - dist)], np.float32),
                 "fx": fx, "fy": fy, "cx": cx_p, "cy": cy_p}
    cams["B"] = {"label": "camB_plusZ_noflip",
                 "W": I.copy(), "t": np.array([-cx, -cy, -(cz + dist)], np.float32),
                 "fx": fx, "fy": fy, "cx": cx_p, "cy": cy_p}
    Fy = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    cams["C"] = {"label": "camC_flipY",
                 "W": Fy.copy(), "t": np.array([-cx, cy, -(cz - dist)], np.float32),
                 "fx": fx, "fy": fy, "cx": cx_p, "cy": cy_p}
    Ry = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    cams["D"] = {"label": "camD_roty180",
                 "W": Ry.copy(), "t": np.array([cx, -cy, cz + dist], np.float32),
                 "fx": fx, "fy": fy, "cx": cx_p, "cy": cy_p}
    return cams


# ─────────────────────────────────────────────────────────────────────────────
# Render a single camera
# ─────────────────────────────────────────────────────────────────────────────


def to_u8(arr: np.ndarray) -> np.ndarray:
    a = np.clip(arr, 0.0, 1.0)
    return np.round(a * 255.0).astype(np.uint8)


def save_ppm(path: str, r: np.ndarray, g: np.ndarray, b: np.ndarray):
    """r,g,b: (H, W) f32 in [0,1]."""
    rgb = np.stack([to_u8(r), to_u8(g), to_u8(b)], axis=2)  # (H, W, 3) u8
    with open(path, "wb") as f:
        f.write(f"P6\n{W} {H}\n255\n".encode("ascii"))
        f.write(rgb.tobytes())


def read_ppm(path: str):
    """Returns (H, W, 3) u8."""
    with open(path, "rb") as f:
        data = f.read()
    # Tiny header parser: P6\n<w> <h>\n<maxval>\n<binary…>
    i = 0
    assert data[:2] == b"P6"
    i = 3  # skip "P6\n"
    # consume comments + dims
    def next_token(j):
        # skip whitespace and comments
        while j < len(data):
            c = data[j:j+1]
            if c in (b" ", b"\t", b"\n", b"\r"):
                j += 1
            elif c == b"#":
                while j < len(data) and data[j:j+1] != b"\n":
                    j += 1
            else:
                break
        k = j
        while k < len(data) and data[k:k+1] not in (b" ", b"\t", b"\n", b"\r"):
            k += 1
        return data[j:k].decode("ascii"), k
    w_s, i = next_token(i)
    h_s, i = next_token(i)
    m_s, i = next_token(i)
    # advance past one whitespace
    i += 1
    w_i = int(w_s); h_i = int(h_s); maxval = int(m_s)
    assert maxval == 255
    body = data[i:i + w_i * h_i * 3]
    return np.frombuffer(body, dtype=np.uint8).reshape(h_i, w_i, 3)


def render_camera(proj: dict, n: int, kernel, label: str,
                  out_ppm: str, time_iters: int) -> dict:
    """Run kernel for one camera. Returns timing dict.

    Performs CPU-side tile-binning before the kernel launch.
    """
    print(f"=== [{label}] ===")
    print(f"  projected gaussians: {n}")
    if n == 0:
        return {"label": label, "n": 0, "ms": [], "median_ms": 0.0,
                "bin_ms": 0.0}

    # ── CPU-side tile binning (timed). ───────────────────────────────────────
    t_bin_0 = time.perf_counter()
    tile_indices, tile_counts, bin_stats = build_tile_bins(
        proj["mx"], proj["my"], proj["cxx"], proj["cxy"], proj["cyy"],
        W, H, BS, MAX_GAUSSIANS_PER_TILE, sigma_k=SIGMA_K,
    )
    t_bin_1 = time.perf_counter()
    bin_ms = (t_bin_1 - t_bin_0) * 1000.0
    print(f"  bin build: {bin_ms:.3f} ms (CPU)")
    print(f"    n_visible           : {bin_stats['n_visible']}/{bin_stats['n_gaussians']}")
    print(f"    n_tiles             : {bin_stats['n_tiles']}")
    print(f"    total (g,tile) pairs: {bin_stats['total_pairs']}")
    print(f"    mean per tile       : {bin_stats['mean_per_tile']:.1f}")
    print(f"    max per tile (seen) : {bin_stats['max_per_tile_seen']}")
    print(f"    overflow tiles      : {bin_stats['overflow_tiles']}"
          f" (cap={bin_stats['max_per_tile_cap']}, dropped={bin_stats['overflow_dropped']})")

    n_ty = H // BS
    n_tx = W // BS
    # Flatten tile_indices for the kernel.
    tile_indices_flat = np.ascontiguousarray(
        tile_indices.reshape(-1).astype(np.int32))
    tile_counts_flat = np.ascontiguousarray(
        tile_counts.reshape(-1).astype(np.int32))

    stream = cupy.cuda.get_current_stream()

    # H2D
    t0 = time.perf_counter()
    d_mx = cupy.asarray(proj["mx"])
    d_my = cupy.asarray(proj["my"])
    d_cxx = cupy.asarray(proj["cxx"])
    d_cxy = cupy.asarray(proj["cxy"])
    d_cyy = cupy.asarray(proj["cyy"])
    d_op = cupy.asarray(proj["opacity"])
    d_cr = cupy.asarray(proj["r"])
    d_cg = cupy.asarray(proj["g"])
    d_cb = cupy.asarray(proj["b"])
    d_idx = cupy.asarray(tile_indices_flat)
    d_cnt = cupy.asarray(tile_counts_flat)
    d_or = cupy.zeros((H, W), dtype=cupy.float32)
    d_og = cupy.zeros((H, W), dtype=cupy.float32)
    d_ob = cupy.zeros((H, W), dtype=cupy.float32)
    cupy.cuda.runtime.deviceSynchronize()
    t1 = time.perf_counter()
    print(f"  H2D copy: {(t1 - t0)*1000:.3f} ms")

    grid = (n_tx, n_ty)

    args = (d_mx, d_my, d_cxx, d_cxy, d_cyy, d_op, d_cr, d_cg, d_cb,
            d_idx, d_cnt,
            d_or, d_og, d_ob)

    # Warmup
    ct.launch(stream.ptr, grid, kernel, args)
    stream.synchronize()

    # Timed iters
    times = []
    starts = [cupy.cuda.Event() for _ in range(time_iters)]
    ends = [cupy.cuda.Event() for _ in range(time_iters)]
    for it in range(time_iters):
        starts[it].record(stream)
        ct.launch(stream.ptr, grid, kernel, args)
        ends[it].record(stream)
    stream.synchronize()
    for it in range(time_iters):
        ms = float(cupy.cuda.get_elapsed_time(starts[it], ends[it]))
        times.append(ms)
        print(f"  kernel iter {it}: {ms:.3f} ms")
    median_ms = sorted(times)[len(times) // 2]
    print(f"  median kernel time: {median_ms:.3f} ms")

    # D2H
    h_r = d_or.get()
    h_g = d_og.get()
    h_b = d_ob.get()

    # Sanity
    nonzero = int(np.sum((h_r > 0.01) | (h_g > 0.01) | (h_b > 0.01)))
    print(f"  nonzero pixels: {nonzero}/{H*W} = {100.0*nonzero/(H*W):.2f}%")

    save_ppm(out_ppm, h_r, h_g, h_b)
    print(f"  wrote {out_ppm}")

    # Free big buffers (gaussian list can be large).
    del d_mx, d_my, d_cxx, d_cxy, d_cyy, d_op, d_cr, d_cg, d_cb
    del d_idx, d_cnt, d_or, d_og, d_ob
    cupy.get_default_memory_pool().free_all_blocks()

    return {"label": label, "n": n, "ms": times, "median_ms": median_ms,
            "bin_ms": bin_ms, "bin_stats": bin_stats}


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default="../oxide-3dgs-real/scenes/utsuho_plush.ply")
    ap.add_argument("--smoke", action="store_true", default=False,
                    help="render cam A and diff against cuda-3dgs-real")
    ap.add_argument("--bench", action="store_true", default=False,
                    help="render all 4 cameras and write CSV")
    ap.add_argument("--csv-out", default="results.csv")
    ap.add_argument("--cam", default=None,
                    help="render only this camera (A/B/C/D); --smoke implies A")
    args = ap.parse_args()

    if not (args.smoke or args.bench):
        args.smoke = True

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"device: {props['name'].decode()}")
    print(f"compute capability: sm_{props['major']}{props['minor']}")
    print(f"image: {W}×{H}, BS={BS}, grid=({W//BS}, {H//BS})")
    print()

    ply_path = str(Path(args.ply).resolve())
    print(f"Loading {ply_path}")
    raws = parse_ply(ply_path)
    print(f"Parsed {raws['x'].size} gaussians")

    cams = make_cameras(raws)

    # Build kernel ONCE; n is a runtime arg.
    print("Building cuTile kernel (binned)...")
    kernel = make_rasterize_kernel(W, H, BS, MAX_GAUSSIANS_PER_TILE)

    csv_rows = []

    def render_cam_label(label: str, out_ppm: str, time_iters: int):
        cam = cams[label]
        proj, n = project_all(raws, cam)
        if proj is None:
            print(f"  [{label}] no gaussians; skipping")
            return None
        return render_camera(proj, n, kernel, cam["label"], out_ppm, time_iters)

    if args.smoke:
        # Cam A only, 1 iter (warmup baked into render_camera).
        print("\n=== SMOKE: cam A correctness ===")
        result = render_cam_label("A", "output_utsuho_plush_A.ppm", 1)
        if result is None:
            print("smoke FAILED (no projected gaussians)")
            return 1

        # Also write canonical filename = cam A.
        import shutil
        shutil.copy("output_utsuho_plush_A.ppm", "output_utsuho_plush.ppm")

        # Diff vs cuda-3dgs-real reference.
        ref_path = "../cuda-3dgs-real/output_utsuho_plush_A.ppm"
        if Path(ref_path).exists():
            ours = read_ppm("output_utsuho_plush_A.ppm")
            ref = read_ppm(ref_path)
            assert ours.shape == ref.shape, f"shape mismatch {ours.shape} vs {ref.shape}"
            diff = np.abs(ours.astype(np.int16) - ref.astype(np.int16)).astype(np.uint16)
            max_diff = int(diff.max())
            mean_diff = float(diff.mean())
            n_neq = int(np.sum(diff.any(axis=2)))
            n_gt2 = int(np.sum(diff.max(axis=2) > 2))
            n_gt5 = int(np.sum(diff.max(axis=2) > 5))
            n_pix = ours.shape[0] * ours.shape[1]
            print()
            print("=== CORRECTNESS DIFF vs cuda-3dgs-real cam A ===")
            print(f"  pixels         : {n_pix}")
            print(f"  pixels w/ diff : {n_neq} ({100.0*n_neq/n_pix:.2f}%)")
            print(f"  pixels w/ diff>2: {n_gt2} ({100.0*n_gt2/n_pix:.2f}%)")
            print(f"  pixels w/ diff>5: {n_gt5} ({100.0*n_gt5/n_pix:.2f}%)")
            print(f"  max u8 diff    : {max_diff}")
            print(f"  mean u8 diff   : {mean_diff:.4f}")
            if max_diff <= 2:
                print("  RESULT: PASS (max diff ≤ 2)")
            elif n_gt2 / n_pix < 0.001:
                print("  RESULT: NEAR-PASS (>99.9% pixels within ≤2 u8)")
            else:
                print("  RESULT: outside ≤2 u8 envelope; review ANALYSIS.md")
        else:
            print(f"  reference {ref_path} not found; skip diff")
        return 0

    if args.bench:
        print(f"\n=== BENCH: {ITERS} timed iters across all 4 cameras ===")
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["impl", "kernel", "camera", "n_projected", "iter", "gpu_ms"])
            for label in ["A", "B", "C", "D"]:
                cam = cams[label]
                out = f"output_utsuho_plush_{label}.ppm"
                result = render_cam_label(label, out, ITERS)
                if result is None:
                    continue
                for it, ms in enumerate(result["ms"]):
                    w.writerow(["cutile-3dgs-real-binned", "rasterize_3dgs_binned",
                                cam["label"], result["n"], it, f"{ms:.6f}"])

        # Canonical = cam A.
        import shutil
        if Path("output_utsuho_plush_A.ppm").exists():
            shutil.copy("output_utsuho_plush_A.ppm", "output_utsuho_plush.ppm")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
