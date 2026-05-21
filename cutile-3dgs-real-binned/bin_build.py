"""CPU-side tile-binning for 3DGS rasterizer (Wave G5).

Given depth-sorted projected gaussians + per-gaussian conic, build a per-tile
list of gaussian indices whose 3-sigma 2D bounding box overlaps the tile.

Output:
    tile_indices: (n_tiles_y, n_tiles_x, MAX) int32   — gaussian indices
                  per tile, padded with 0 (mask via tile_counts).
    tile_counts:  (n_tiles_y, n_tiles_x)      int32   — actual count per tile
                  (clamped to MAX). Indices [0:count] are valid, in front-to-back
                  depth order.

Algorithm:
  1. For each gaussian compute a 2D bbox: mean_xy ± k*sigma_xy where
     sigma_xy = sqrt(diag of inverse-conic), clamped to [0, W) × [0, H).
     We use the conic in projected space:
         conic = [[cxx, cxy], [cxy, cyy]]
         sigma2_x = cyy / (cxx*cyy - cxy^2)   # element [0,0] of inverse
         sigma2_y = cxx / (cxx*cyy - cxy^2)
     and bbox half-width = k * sqrt(sigma2_x), half-height = k * sqrt(sigma2_y).
  2. Convert bbox to tile-space: tx_lo = bbox_x_lo // BS, etc.
  3. Each gaussian g iterates tx in [tx_lo..tx_hi], ty in [ty_lo..ty_hi]
     and pushes index g into the (ty, tx) tile's bucket.
  4. Buckets are populated in depth order (we iterate gaussians in front-to-back
     order), so the per-tile list is already depth-sorted.
  5. We pad to MAX by truncating any tile bucket that exceeds MAX (extreme
     overflow tiles drop the deepest gaussians, which contribute least due to
     transmittance saturation).

Performance: the loops are NOT vectorized as one numpy call (we'd need a
ragged scatter), but the per-gaussian tile span is small (3-sigma is typically
2-8 tiles each axis), so this is O(n_gaussians * mean_tiles_per_gaussian),
which is ~50k * ~20 = ~1M iterations. We use a numba-style hot loop here
implemented in pure numpy (boolean fan-out) for simplicity. The expected
end-to-end CPU bin-build time is ~10ms.
"""
from __future__ import annotations

import numpy as np


def build_tile_bins(
    mx: np.ndarray,
    my: np.ndarray,
    cxx: np.ndarray,
    cxy: np.ndarray,
    cyy: np.ndarray,
    width: int,
    height: int,
    block_size: int,
    max_per_tile: int,
    sigma_k: float = 3.0,
):
    """Returns (tile_indices, tile_counts, stats_dict)."""
    n = mx.size
    n_tx = width // block_size
    n_ty = height // block_size

    # ── 1. Compute 2D bbox per gaussian. ─────────────────────────────────────
    # conic = [[cxx, cxy], [cxy, cyy]]; this is the INVERSE 2D covariance.
    # The 2D covariance Σ = inverse(conic) has:
    #     det_c = cxx*cyy - cxy^2
    #     Σ[0,0] = cyy / det_c     (sigma2_x along screen x)
    #     Σ[1,1] = cxx / det_c     (sigma2_y along screen y)
    det_c = cxx * cyy - cxy * cxy
    # det_c > 0 was already enforced host-side (project_all valid mask).
    inv_det = 1.0 / np.maximum(det_c, 1e-30)
    sigma_x = np.sqrt(np.maximum(cyy * inv_det, 0.0))
    sigma_y = np.sqrt(np.maximum(cxx * inv_det, 0.0))

    half_w = sigma_k * sigma_x
    half_h = sigma_k * sigma_y

    bx_lo = mx - half_w
    bx_hi = mx + half_w
    by_lo = my - half_h
    by_hi = my + half_h

    # Clamp to viewport (anything entirely outside contributes nothing).
    bx_lo_c = np.maximum(bx_lo, 0.0)
    bx_hi_c = np.minimum(bx_hi, float(width - 1))
    by_lo_c = np.maximum(by_lo, 0.0)
    by_hi_c = np.minimum(by_hi, float(height - 1))

    visible = (bx_lo_c <= bx_hi_c) & (by_lo_c <= by_hi_c)

    # Tile-space bounds (inclusive).
    tx_lo = np.floor(bx_lo_c / block_size).astype(np.int32)
    tx_hi = np.floor(bx_hi_c / block_size).astype(np.int32)
    ty_lo = np.floor(by_lo_c / block_size).astype(np.int32)
    ty_hi = np.floor(by_hi_c / block_size).astype(np.int32)
    tx_lo = np.clip(tx_lo, 0, n_tx - 1)
    tx_hi = np.clip(tx_hi, 0, n_tx - 1)
    ty_lo = np.clip(ty_lo, 0, n_ty - 1)
    ty_hi = np.clip(ty_hi, 0, n_ty - 1)

    # ── 2. Fan out: for each gaussian, push to each (ty, tx) tile it touches. ─
    # Build via "duplicate-and-scatter": create one (gauss_idx, tile_id) pair
    # for every overlapping (gauss, tile) pair. Then bucket-sort by tile_id.
    n_per_g = np.where(visible, (tx_hi - tx_lo + 1) * (ty_hi - ty_lo + 1), 0).astype(np.int64)
    total_pairs = int(n_per_g.sum())

    # Pre-allocate flat arrays.
    gauss_idx_flat = np.empty(total_pairs, dtype=np.int32)
    tile_id_flat = np.empty(total_pairs, dtype=np.int32)

    # Fill via per-gaussian Python loop (over visible only). This is the hot
    # CPU loop. It's ~50k iterations of small (1-100) tile-spans = ~1M total.
    cursor = 0
    vis_idx = np.where(visible)[0]
    for g in vis_idx:
        gx_lo = int(tx_lo[g]); gx_hi = int(tx_hi[g])
        gy_lo = int(ty_lo[g]); gy_hi = int(ty_hi[g])
        nx = gx_hi - gx_lo + 1
        ny = gy_hi - gy_lo + 1
        n_tiles_g = nx * ny
        if n_tiles_g <= 0:
            continue
        # Outer-product-style enumeration.
        ys = np.arange(gy_lo, gy_hi + 1, dtype=np.int32)
        xs = np.arange(gx_lo, gx_hi + 1, dtype=np.int32)
        # tile_id = ty * n_tx + tx (row-major in tile space).
        tids = (ys[:, None] * n_tx + xs[None, :]).reshape(-1)
        gauss_idx_flat[cursor:cursor + n_tiles_g] = g
        tile_id_flat[cursor:cursor + n_tiles_g] = tids
        cursor += n_tiles_g
    assert cursor == total_pairs, f"cursor {cursor} vs total {total_pairs}"

    # ── 3. Bucket by tile_id, preserving depth order (gauss_idx is already
    #       depth-sorted ascending front-to-back). Stable sort by tile_id. ──
    # We want tiles that appear earlier in gaussian order (smaller g) to be
    # earlier in their bucket. argsort(stable) on tile_id gives this naturally.
    order = np.argsort(tile_id_flat, kind="stable")
    sorted_tids = tile_id_flat[order]
    sorted_gauss = gauss_idx_flat[order]

    # Find run starts via diff.
    n_tiles = n_tx * n_ty
    counts_full = np.bincount(sorted_tids, minlength=n_tiles).astype(np.int32)
    # Position offsets: cumulative sum.
    offsets = np.concatenate([[0], np.cumsum(counts_full[:-1])]).astype(np.int64)

    # ── 4. Pack into (n_ty, n_tx, MAX) padded indices. ───────────────────────
    tile_indices = np.zeros((n_ty, n_tx, max_per_tile), dtype=np.int32)
    tile_counts = np.zeros((n_ty, n_tx), dtype=np.int32)

    overflow_tiles = 0
    overflow_dropped = 0
    max_seen = 0
    for tid in range(n_tiles):
        n_in = int(counts_full[tid])
        max_seen = max(max_seen, n_in)
        ty = tid // n_tx
        tx = tid % n_tx
        if n_in == 0:
            continue
        n_keep = min(n_in, max_per_tile)
        start = int(offsets[tid])
        tile_indices[ty, tx, :n_keep] = sorted_gauss[start:start + n_keep]
        tile_counts[ty, tx] = n_keep
        if n_in > max_per_tile:
            overflow_tiles += 1
            overflow_dropped += n_in - max_per_tile

    stats = {
        "n_gaussians": int(n),
        "n_visible": int(visible.sum()),
        "n_tiles": n_tiles,
        "total_pairs": total_pairs,
        "mean_per_tile": float(total_pairs / n_tiles),
        "max_per_tile_seen": int(max_seen),
        "overflow_tiles": overflow_tiles,
        "overflow_dropped": overflow_dropped,
        "max_per_tile_cap": int(max_per_tile),
    }
    return tile_indices, tile_counts, stats
