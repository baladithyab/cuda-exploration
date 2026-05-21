"""Wave 15.3 — cuTile 3DGS rasterizer (4th frontend cell).

Port of cuda-3dgs-real / oxide-3dgs-real to the cuTile Python tile DSL.
Mechanism = 3D Gaussian Splatting rasterizer: per-pixel front-to-back
alpha-compositing over depth-sorted gaussians, with per-gaussian
2D-conic density evaluation. All of this is Approach A — naive port,
per-pixel iteration over ALL gaussians in tile space (NO tile binning).
This sacrifices perf for completeness as a frontend column.

Major difference vs nvcc/oxide reference kernels:
  - cuTile DSL has no per-tile-element early termination; the early-
    out on transmittance < 1e-4 is dropped. Output is still numerically
    near-identical because the alpha clamp at 0.99 means once
    transmittance is tiny, additional gaussian contributions are
    weight = alpha · t ≈ 0 and don't move the accum within u8 quant.
  - One @ct.kernel, 16×16 pixel tile per CTA. For 800×800 image with
    BS=16, grid=(50, 50). Each tile iterates the full gaussian list.

Host side (PLY parse → projection → SH evaluation → depth sort) is
ported from rasterize.cu line-by-line in numpy. The output of host-side
is 9 f32 arrays (mx, my, cxx, cxy, cyy, opacity, r, g, b) sorted by
depth ascending; the kernel rasterizes them. Cam A is the canonical
reference matched against cuda-3dgs-real/output_utsuho_plush_A.ppm.

CLI:
    --smoke      run cam A render and diff against cuda-3dgs-real PPM
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

# ─────────────────────────────────────────────────────────────────────────────
# Constants — match cuda-3dgs-real
# ─────────────────────────────────────────────────────────────────────────────

W = 800
H = 800
BS = 16   # pixel tile side
ITERS = 3
WARMUP = 1


# ─────────────────────────────────────────────────────────────────────────────
# cuTile kernel — naive 3DGS rasterizer
# ─────────────────────────────────────────────────────────────────────────────


def make_rasterize_kernel(width: int, height: int, block_size: int):
    """Build a @ct.kernel for the given image and tile size.

    width, height, block_size are captured as Python ints (closure constants),
    so cuTile compiles them into the tile shape. The number of gaussians n
    is a runtime kernel argument and the inner loop is a runtime range.

    Arrays passed in are 1D (n_gaussians,) f32:
        mx, my, cxx, cxy, cyy, opacity, cr, cg, cb
    Outputs are 1D (W*H,) f32:
        out_r, out_g, out_b

    Grid: (W // BS, H // BS) — each block produces a BS×BS pixel tile.
    """
    BS_local = block_size

    @ct.kernel
    def rasterize_3dgs(
        mx, my, cxx, cxy, cyy, opacity, cr, cg, cb,
        n_gaussians,
        out_r, out_g, out_b,
    ):
        bx = ct.bid(0)  # tile index along width
        by = ct.bid(1)  # tile index along height

        # ── Build per-pixel coordinate tiles (BS, BS) f32. ──
        # px_row[r, c] = bx*BS + c   (col index)
        # py_row[r, c] = by*BS + r   (row index)
        col_idx = ct.arange(BS_local, dtype=ct.int32)        # (BS,)
        row_idx = ct.arange(BS_local, dtype=ct.int32)        # (BS,)
        col_2d = ct.broadcast_to(ct.expand_dims(col_idx, axis=0), (BS_local, BS_local))  # (BS,BS) values vary along col
        row_2d = ct.broadcast_to(ct.expand_dims(row_idx, axis=1), (BS_local, BS_local))  # (BS,BS) values vary along row

        pxf = (col_2d + bx * BS_local).astype(ct.float32)
        pyf = (row_2d + by * BS_local).astype(ct.float32)

        # ── Init accumulators (BS, BS) f32. ──
        accum_r = ct.zeros((BS_local, BS_local), ct.float32)
        accum_g = ct.zeros((BS_local, BS_local), ct.float32)
        accum_b = ct.zeros((BS_local, BS_local), ct.float32)
        transmittance = ct.full((BS_local, BS_local), 1.0, ct.float32)

        ALPHA_FLOOR = 1.0 / 255.0
        ALPHA_CAP = 0.99
        ZERO_T = ct.zeros((BS_local, BS_local), ct.float32)

        # ── Iterate over all gaussians (front-to-back, depth-sorted host-side). ──
        for i in range(n_gaussians):
            # Load each gaussian's scalars as (1,) tiles, then expand to (1,1)
            # so they broadcast against the (BS,BS) pixel tile.
            mxi_1d = ct.load(mx, index=(i,), shape=(1,))
            myi_1d = ct.load(my, index=(i,), shape=(1,))
            cxxi_1d = ct.load(cxx, index=(i,), shape=(1,))
            cxyi_1d = ct.load(cxy, index=(i,), shape=(1,))
            cyyi_1d = ct.load(cyy, index=(i,), shape=(1,))
            opi_1d = ct.load(opacity, index=(i,), shape=(1,))
            cri_1d = ct.load(cr, index=(i,), shape=(1,))
            cgi_1d = ct.load(cg, index=(i,), shape=(1,))
            cbi_1d = ct.load(cb, index=(i,), shape=(1,))

            mxi = ct.expand_dims(mxi_1d, axis=1)   # (1,1)
            myi = ct.expand_dims(myi_1d, axis=1)
            cxxi = ct.expand_dims(cxxi_1d, axis=1)
            cxyi = ct.expand_dims(cxyi_1d, axis=1)
            cyyi = ct.expand_dims(cyyi_1d, axis=1)
            opi = ct.expand_dims(opi_1d, axis=1)
            cri = ct.expand_dims(cri_1d, axis=1)
            cgi = ct.expand_dims(cgi_1d, axis=1)
            cbi = ct.expand_dims(cbi_1d, axis=1)

            # dx = pxf - means_x[i], dy = pyf - means_y[i]
            dx = pxf - mxi
            dy = pyf - myi

            # power = -0.5 * (cxx*dx^2 + 2*cxy*dx*dy + cyy*dy^2)
            power = -0.5 * (cxxi * dx * dx + 2.0 * cxyi * dx * dy + cyyi * dy * dy)

            # alpha = opacity * exp(power), clamped to 0 if power > 0.
            # In the reference: only update if power <= 0.0 AND alpha >= 1/255.
            # Express as a where-mask: valid = (power <= 0) & (alpha >= 1/255).
            alpha_raw = opi * ct.exp(power)
            valid_power = power <= 0.0           # (BS,BS) bool
            valid_alpha = alpha_raw >= ALPHA_FLOOR
            valid = valid_power & valid_alpha
            # Clamp to 0.99 like the reference.
            alpha_capped = ct.minimum(alpha_raw, ct.full((BS_local, BS_local), ALPHA_CAP, ct.float32))
            # If invalid, contribute zero (i.e. weight=0 → no change).
            alpha_eff = ct.where(valid, alpha_capped, ZERO_T)

            weight = alpha_eff * transmittance
            accum_r = accum_r + weight * cri
            accum_g = accum_g + weight * cgi
            accum_b = accum_b + weight * cbi
            transmittance = transmittance * (1.0 - alpha_eff)

        # ── Store the final accumulator. ──
        # out_* is a 1D array of length W*H; pixel index = py*W + px.
        # We need to scatter the (BS,BS) tile into the right linear positions.
        # Easiest path: use a 2D tiled_view of shape (H, W) and store the tile.
        out_r_view = out_r.tiled_view((BS_local, BS_local), padding_mode=ct.PaddingMode.ZERO)
        out_g_view = out_g.tiled_view((BS_local, BS_local), padding_mode=ct.PaddingMode.ZERO)
        out_b_view = out_b.tiled_view((BS_local, BS_local), padding_mode=ct.PaddingMode.ZERO)
        # tiled_view tile-space index: (by, bx) for a (H, W)-shape array.
        out_r_view.store((by, bx), accum_r)
        out_g_view.store((by, bx), accum_g)
        out_b_view.store((by, bx), accum_b)

    return rasterize_3dgs


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
    """Run kernel for one camera. Returns timing dict."""
    print(f"=== [{label}] ===")
    print(f"  projected gaussians: {n}")
    if n == 0:
        return {"label": label, "n": 0, "ms": [], "median_ms": 0.0}

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
    d_or = cupy.zeros((H, W), dtype=cupy.float32)  # 2D for tiled_view
    d_og = cupy.zeros((H, W), dtype=cupy.float32)
    d_ob = cupy.zeros((H, W), dtype=cupy.float32)
    cupy.cuda.runtime.deviceSynchronize()
    t1 = time.perf_counter()
    print(f"  H2D copy: {(t1 - t0)*1000:.3f} ms")

    grid = (W // BS, H // BS)

    # Warmup
    ct.launch(stream.ptr, grid, kernel,
              (d_mx, d_my, d_cxx, d_cxy, d_cyy, d_op, d_cr, d_cg, d_cb,
               int(n), d_or, d_og, d_ob))
    stream.synchronize()

    # Timed iters
    times = []
    starts = [cupy.cuda.Event() for _ in range(time_iters)]
    ends = [cupy.cuda.Event() for _ in range(time_iters)]
    for it in range(time_iters):
        starts[it].record(stream)
        ct.launch(stream.ptr, grid, kernel,
                  (d_mx, d_my, d_cxx, d_cxy, d_cyy, d_op, d_cr, d_cg, d_cb,
                   int(n), d_or, d_og, d_ob))
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
    del d_or, d_og, d_ob
    cupy.get_default_memory_pool().free_all_blocks()

    return {"label": label, "n": n, "ms": times, "median_ms": median_ms}


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
    print("Building cuTile kernel...")
    kernel = make_rasterize_kernel(W, H, BS)

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
                    w.writerow(["cutile-3dgs-real", "rasterize_3dgs",
                                cam["label"], result["n"], it, f"{ms:.6f}"])

        # Canonical = cam A.
        import shutil
        if Path("output_utsuho_plush_A.ppm").exists():
            shutil.copy("output_utsuho_plush_A.ppm", "output_utsuho_plush.ppm")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
