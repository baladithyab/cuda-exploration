"""Wave 17.W1e — cuTile fused KDA (Kimi Delta Attention) single-timestep decode.

KDA is the **per-channel-gate** variant of GDN (Gated DeltaNet).  This file
is a **semantic-fenced fork** of `cutile-attn-gdn/main.py` per ADR-0006:
the diff against that file contains ONLY the four allowed change classes
(a–d): shape-constant changes, the `g[B,S,H]` → `g[B,S,H,d_k]` gate-tensor
shape change, the `s_tile * α_f32` → `s_tile * exp_g.reshape(-1, 1)`
broadcast-rescale change, and the bench-shape change (Qwen3-Next →
Kimi-Linear-48B-A3B).  The inner-loop ordering is byte-identical to GDN.

Recurrence (verbatim from `docs/research/wave17-kda-spec.md`, §1):

    S_t = ( I − β_t k_t k_tᵀ ) · Diag(α_t) · S_{t−1}  +  β_t k_t v_tᵀ
    o_t = S_tᵀ q_t                                        with α_t ∈ ℝ^{d_k}

The only mathematical change vs GDN is line 1 of the inner body: the
state is rescaled by the per-channel decay vector exp(g) along the d_k
axis instead of by a scalar α.  Everything downstream commutes through
the rescale because both `k·S` (K-axis reduction) and `k ⊗ residual`
(K-axis outer product) operate on the SAME K-axis the decay decorrelates.

Grid: (batch * n_heads, d_v / BLOCK_V).  No cross-block reductions —
each block owns its disjoint slice of d_v columns.

Per-head state tile memory:
  - d_k=64, BV=32:   64 · 32 · 4 = 8 KB   (correctness shape)
  - d_k=128, BV=64:  128 · 64 · 4 = 32 KB (Kimi-Linear-48B decode shape)

CLI:
    --smoke     correctness at SHAPE_CORRECTNESS
    --bench     timed at SHAPE_KIMI_LINEAR_DECODE (1 warmup + 50 iters)
    --csv-out FILE    bench CSV
    --export-cubin    write cubin for SASS inspection
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import cuda.tile as ct
import cupy
import numpy as np
from cuda.tile.compilation import (
    ArrayConstraint,
    CallingConvention,
    KernelSignature,
    export_kernel,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Shape constants — ADR-0006 §2(a) shape-constant changes vs GDN's
# `from shapes_gdn import GDNShape, SHAPE_CORRECTNESS, SHAPE_QWEN3_NEXT_DECODE`.
# Inlined because the W1e file-ownership policy keeps writes inside
# cutile-attn-kda/.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KDAShape:
    """Kimi-Delta-Attention decode shape parameters (single timestep)."""

    name: str
    batch: int
    n_heads: int
    d_k: int
    d_v: int


# ADR-0006 §2(d): bench shape changes from GDN's Qwen3-Next (H=16, d_k=d_v=256)
# to KDA's Kimi-Linear-48B-A3B-Instruct (H=32, d_k=d_v=128).
SHAPE_CORRECTNESS = KDAShape(
    name="correctness", batch=1, n_heads=2, d_k=64, d_v=64,
)
SHAPE_KIMI_LINEAR_DECODE = KDAShape(
    name="kimi_linear_decode", batch=1, n_heads=32, d_k=128, d_v=128,
)


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs / bytes / bandwidth — ADR-0006 §2(a) shape-constant helpers,
# inlined replacements for the GDN `from flops_gdn import …` block.
# ─────────────────────────────────────────────────────────────────────────────


def kda_decode_flops(shape: KDAShape) -> int:
    """6 · d_k · d_v per (batch, head) — three matmul-shaped passes."""
    return 6 * shape.batch * shape.n_heads * shape.d_k * shape.d_v


def kda_decode_bytes(shape: KDAShape, state_dtype_bytes: int = 4,
                     io_dtype_bytes: int = 2) -> int:
    """HBM bytes per decode step.  State (R+W) dominates; I/O is negligible.

    KDA gate is `g[B*H, d_k]` (vector, vs GDN's scalar) so the gate term is
    `d_k · 2` bytes per (batch, head) instead of `2`.
    """
    bh = shape.batch * shape.n_heads
    state = 2 * shape.d_k * shape.d_v * state_dtype_bytes
    # q, k, g at d_k f16; v, o at d_v f16; beta scalar.
    io = (3 * shape.d_k + 2 * shape.d_v + 1) * io_dtype_bytes
    return bh * (state + io)


def kda_decode_gbps(shape: KDAShape, gpu_ms: float,
                    state_dtype_bytes: int = 4, io_dtype_bytes: int = 2) -> float:
    bytes_moved = kda_decode_bytes(shape, state_dtype_bytes, io_dtype_bytes)
    return bytes_moved / (gpu_ms * 1e-3) / 1e9


def kda_decode_tflops(shape: KDAShape, gpu_ms: float) -> float:
    return kda_decode_flops(shape) / (gpu_ms * 1e-3) / 1e12


# ─────────────────────────────────────────────────────────────────────────────
# Reference oracle — numpy port of fla/ops/kda/naive.py::naive_recurrent_kda
# Single timestep, no GVA (n_v_heads == n_heads), float64 accumulation.
# ─────────────────────────────────────────────────────────────────────────────


def naive_recurrent_kda_step(q, k, v, g, beta, S_in, scale=1.0):
    """One-step KDA recurrence in float64.

    Args (numpy):
        q, k:   (B, H, d_k)        f32
        v:      (B, H, d_v)        f32
        g:      (B, H, d_k)        f32   — log-gate; decay = exp(g)
        beta:   (B, H)             f32
        S_in:   (B, H, d_k, d_v)   f32

    NOTE: FLA's naive_recurrent_kda applies `scale = K**-0.5` to q internally.
    Our kernel does NOT apply that scale (matching GDN's reference convention
    which pre-scales q OUTSIDE the kernel).  We default `scale=1.0` so the
    oracle output matches what the kernel computes.

    Returns:
        o:      (B, H, d_v)        f64
        S_out:  (B, H, d_k, d_v)   f64
    """
    q = q.astype(np.float64); k = k.astype(np.float64); v = v.astype(np.float64)
    g = g.astype(np.float64); beta = beta.astype(np.float64)
    S = S_in.astype(np.float64)
    q = q * scale
    # Decay: S = exp(g)[..., None] * S   (broadcast (B,H,K,1) over (B,H,K,V))
    S = S * np.exp(g)[..., None]
    # Residual u = (k * S).sum(K) — shape (B,H,V)  (k post-broadcast over V)
    u = (k[..., None] * S).sum(axis=-2)
    residual = v - u
    # Outer product β · k ⊗ residual → (B,H,K,V)
    S = S + (beta[..., None] * k)[..., None] * residual[..., None, :]
    # Output o = S^T q  →  (B,H,V)
    o = np.einsum("bhk,bhkv->bhv", q, S)
    return o, S


# ─────────────────────────────────────────────────────────────────────────────
# Kernel factory
# ─────────────────────────────────────────────────────────────────────────────


def make_kda_decode_kernel(D_K: int, D_V: int, BLOCK_V: int):
    """Build a @ct.kernel specialized to these KDA shape constants.

    Grid layout (caller picks):
        bid0 ∈ [0, B·H)         — flattened (batch, head)
        bid1 ∈ [0, D_V // BV)   — d_v-block index

    Array layouts:
        Q       (B*H, D_K)        f16   — tiled_view((1, D_K))
        K       (B*H, D_K)        f16   — tiled_view((1, D_K))
        V       (B*H, D_V)        f16   — tiled_view((1, BLOCK_V))
        G       (B*H, D_K)        f16   — tiled_view((1, D_K))   ADR-0006 §2(b)
        beta    (B*H, 1)          f16   — tiled_view((1, 1))
        S_in    (B*H*D_K, D_V)    f32   — tiled_view((D_K, BLOCK_V))
        S_out   (B*H*D_K, D_V)    f32
        O       (B*H, D_V)        f16
    """
    D_V_BLOCKS = D_V // BLOCK_V

    @ct.kernel
    def kda_decode(Q, K, V, G, Beta, S_in, S_out, O):
        bh = ct.bid(0)
        bv = ct.bid(1)

        q_view = Q.tiled_view((1, D_K), padding_mode=ct.PaddingMode.ZERO)
        k_view = K.tiled_view((1, D_K), padding_mode=ct.PaddingMode.ZERO)
        v_view = V.tiled_view((1, BLOCK_V), padding_mode=ct.PaddingMode.ZERO)
        o_view = O.tiled_view((1, BLOCK_V), padding_mode=ct.PaddingMode.ZERO)
        # ADR-0006 §2(b): gate-tensor shape change.
        # GDN had Alpha (B*H, 1) — scalar per head.
        # KDA has G     (B*H, D_K) — d_k-vector per head, log-space.
        g_view = G.tiled_view((1, D_K), padding_mode=ct.PaddingMode.ZERO)
        b_view = Beta.tiled_view((1, 1), padding_mode=ct.PaddingMode.ZERO)
        s_view = S_in.tiled_view((D_K, BLOCK_V), padding_mode=ct.PaddingMode.ZERO)
        s_out_view = S_out.tiled_view(
            (D_K, BLOCK_V), padding_mode=ct.PaddingMode.ZERO
        )

        # ── Load inputs ──
        q_tile = q_view.load((bh, 0))   # (1, D_K) f16
        k_tile = k_view.load((bh, 0))   # (1, D_K) f16
        v_tile = v_view.load((bh, bv))  # (1, BLOCK_V) f16
        g_tile = g_view.load((bh, 0))   # (1, D_K) f16  — ADR-0006 §2(b)
        b_tile = b_view.load((bh, 0))   # (1, 1) f16

        s_tile = s_view.load((bh, bv))  # (D_K, BLOCK_V) f32

        k_f32 = k_tile.astype(ct.float32)    # (1, D_K)
        q_f32 = q_tile.astype(ct.float32)    # (1, D_K)
        v_f32 = v_tile.astype(ct.float32)    # (1, BLOCK_V)
        g_f32 = g_tile.astype(ct.float32)    # (1, D_K)  — ADR-0006 §2(b)
        beta_f32 = b_tile.astype(ct.float32) # (1, 1)
        # KDA: per-channel decay = exp(g).  GDN: scalar α loaded directly.
        exp_g = ct.exp(g_f32)                # (1, D_K)

        # ── (1) S_scaled = Diag(exp(g)) · S_in   — ADR-0006 §2(c) ──
        # GDN had:  s_scaled = s_tile * alpha_f32              (scalar broadcast)
        # KDA:      s_scaled = s_tile * exp_g.reshape(-1, 1)   ((D_K, 1) broadcast)
        # ct.transpose lifts the (1, D_K) tile to (D_K, 1) which broadcasts
        # over the (D_K, BLOCK_V) state tile, scaling each row by its own decay.
        s_scaled = s_tile * ct.transpose(exp_g)  # (D_K, 1) × (D_K, BV) broadcast

        # ── (2) u = k^T · S_scaled   (shape (1, BLOCK_V))  ──
        u_acc = ct.zeros((1, BLOCK_V), ct.float32)
        u_acc = ct.mma(k_f32, s_scaled, u_acc)

        # ── (3) residual = v - u  ──
        residual = v_f32 - u_acc  # (1, BLOCK_V)

        # ── (4) S_out = S_scaled + beta · (k ⊗ residual)  ──
        k_col = ct.transpose(k_f32)  # (D_K, 1)
        outer_acc = ct.zeros((D_K, BLOCK_V), ct.float32)
        outer_acc = ct.mma(k_col, residual, outer_acc)

        s_out = s_scaled + beta_f32 * outer_acc

        # ── (5) o = q^T · S_out   (shape (1, BLOCK_V))  ──
        o_acc = ct.zeros((1, BLOCK_V), ct.float32)
        o_acc = ct.mma(q_f32, s_out, o_acc)

        # ── Store ──
        s_out_view.store((bh, bv), s_out)              # f32
        o_view.store((bh, bv), o_acc.astype(O.dtype))  # f16

    return kda_decode


# ─────────────────────────────────────────────────────────────────────────────
# Input plumbing — ADR-0006 §2(a) shape-constant change vs GDN's disk-loaded
# `gdn_<name>_*.npy` block.  KDA inputs are generated in-process via the
# numpy oracle `naive_recurrent_kda_step`; W1e file-ownership policy bars
# writing to the shared analysis/wave15-attention-architecture/inputs/ tree.
# ─────────────────────────────────────────────────────────────────────────────


def load_inputs(shape: KDAShape, seed: int = 17):
    rng = np.random.default_rng(seed)
    B, H, d_k, d_v = shape.batch, shape.n_heads, shape.d_k, shape.d_v
    # Pre-scale q and k by 1/sqrt(d_k) (matches GDN's reference convention,
    # cf. analysis/wave15-attention-architecture/reference/pytorch_reference_gdn.py:140-142).
    scale_k = 1.0 / np.sqrt(d_k)
    q = (rng.standard_normal((B, H, d_k)) * scale_k).astype(np.float16)
    k = (rng.standard_normal((B, H, d_k)) * scale_k).astype(np.float16)
    v = rng.standard_normal((B, H, d_v)).astype(np.float16)
    # Log-gate: small negative magnitudes → exp(g) ∈ (~0.6, 1].
    g = (-np.abs(rng.standard_normal((B, H, d_k))) * 0.5).astype(np.float16)
    beta = (rng.uniform(0.1, 0.9, size=(B, H))).astype(np.float16)
    S_in = (rng.standard_normal((B, H, d_k, d_v)) * 0.1).astype(np.float32)

    o_expected, S_out_expected = naive_recurrent_kda_step(
        q.astype(np.float32), k.astype(np.float32), v.astype(np.float32),
        g.astype(np.float32), beta.astype(np.float32), S_in,
        scale=1.0,
    )
    return {
        "q": q, "k": k, "v": v, "g": g, "beta": beta, "S_in": S_in,
        "o_expected": o_expected.astype(np.float16),
        "S_out_expected": S_out_expected.astype(np.float32),
    }


def prepare_device(shape: KDAShape, inp: dict):
    B, H, d_k, d_v = shape.batch, shape.n_heads, shape.d_k, shape.d_v
    BH = B * H

    q_d = cupy.asarray(inp["q"].reshape(BH, d_k), dtype=cupy.float16)
    k_d = cupy.asarray(inp["k"].reshape(BH, d_k), dtype=cupy.float16)
    v_d = cupy.asarray(inp["v"].reshape(BH, d_v), dtype=cupy.float16)
    # ADR-0006 §2(b): gate is (BH, d_k) instead of (BH, 1).
    g_d = cupy.asarray(inp["g"].reshape(BH, d_k), dtype=cupy.float16)
    beta_d = cupy.asarray(inp["beta"].reshape(BH, 1), dtype=cupy.float16)
    s_in_d = cupy.asarray(inp["S_in"].reshape(BH * d_k, d_v), dtype=cupy.float32)
    s_out_d = cupy.zeros((BH * d_k, d_v), dtype=cupy.float32)
    o_d = cupy.zeros((BH, d_v), dtype=cupy.float16)
    return q_d, k_d, v_d, g_d, beta_d, s_in_d, s_out_d, o_d


# ─────────────────────────────────────────────────────────────────────────────
# Block-size picker
# ─────────────────────────────────────────────────────────────────────────────


def pick_block_v(shape: KDAShape) -> int:
    """For d_k=64: BV=32 → 8 KB tile; for d_k=128: BV=64 → 32 KB tile."""
    max_elems = 16384
    for bv in (128, 64, 32, 16, 8):
        if shape.d_v % bv == 0 and shape.d_k * bv <= max_elems:
            return bv
    raise ValueError(f"no feasible BLOCK_V for d_k={shape.d_k} d_v={shape.d_v}")


# ─────────────────────────────────────────────────────────────────────────────
# Correctness smoke
# ─────────────────────────────────────────────────────────────────────────────


def run_smoke(shape: KDAShape) -> bool:
    print(
        f"[smoke] shape={shape.name} B={shape.batch} H={shape.n_heads} "
        f"d_k={shape.d_k} d_v={shape.d_v}"
    )
    inp = load_inputs(shape)
    q_d, k_d, v_d, g_d, b_d, s_in_d, s_out_d, o_d = prepare_device(shape, inp)

    bv = pick_block_v(shape)
    print(f"[smoke] BLOCK_V={bv}")
    kernel = make_kda_decode_kernel(shape.d_k, shape.d_v, bv)
    grid = (shape.batch * shape.n_heads, shape.d_v // bv)
    stream = cupy.cuda.get_current_stream()

    ct.launch(
        stream.ptr, grid, kernel,
        (q_d, k_d, v_d, g_d, b_d, s_in_d, s_out_d, o_d),
    )
    cupy.cuda.runtime.deviceSynchronize()

    B, H, d_k, d_v = shape.batch, shape.n_heads, shape.d_k, shape.d_v
    o_np = o_d.get().reshape(B, H, d_v)
    s_out_np = s_out_d.get().reshape(B, H, d_k, d_v)

    o_exp = inp["o_expected"]
    s_exp = inp["S_out_expected"]

    abs_err_o = np.abs(o_np.astype(np.float32) - o_exp.astype(np.float32))
    max_o = float(abs_err_o.max())
    rel_o = max_o / (float(np.abs(o_exp).max()) + 1e-30)
    # ADR-0006 acceptance: max_abs ≤ 1e-3 (or rel ≤ 1e-3 — f16 round-off slack).
    ok_o = max_o <= 1e-3 or rel_o <= 1e-3
    print(f"[smoke] o    max_abs={max_o:.3e} rel={rel_o:.3e}  {'OK' if ok_o else 'FAIL'}")

    abs_err_s = np.abs(s_out_np - s_exp)
    max_s = float(abs_err_s.max())
    rel_s = max_s / (float(np.abs(s_exp).max()) + 1e-30)
    ok_s = max_s <= 1e-3 or rel_s <= 1e-3
    print(f"[smoke] S_out max_abs={max_s:.3e} rel={rel_s:.3e}  {'OK' if ok_s else 'FAIL'}")

    if not (ok_o and ok_s):
        if not ok_o:
            w = np.unravel_index(abs_err_o.argmax(), abs_err_o.shape)
            print(
                f"        worst o offender at {w}: "
                f"got={o_np[w]:.4f} expected={o_exp[w]:.4f}"
            )
        if not ok_s:
            w = np.unravel_index(abs_err_s.argmax(), abs_err_s.shape)
            print(
                f"        worst S offender at {w}: "
                f"got={s_out_np[w]:.6f} expected={s_exp[w]:.6f}"
            )
    return ok_o and ok_s


# ─────────────────────────────────────────────────────────────────────────────
# Bench
# ─────────────────────────────────────────────────────────────────────────────

WARMUP = 2
ITERS = 50


def run_bench(shape: KDAShape, csv_path: str) -> None:
    print(
        f"[bench] shape={shape.name} B={shape.batch} H={shape.n_heads} "
        f"d_k={shape.d_k} d_v={shape.d_v}"
    )
    inp = load_inputs(shape)
    q_d, k_d, v_d, g_d, b_d, s_in_d, s_out_d, o_d = prepare_device(shape, inp)

    bv = pick_block_v(shape)
    print(f"[bench] BLOCK_V={bv}")
    kernel = make_kda_decode_kernel(shape.d_k, shape.d_v, bv)
    grid = (shape.batch * shape.n_heads, shape.d_v // bv)
    print(f"[bench] grid={grid}  n_blocks={grid[0]*grid[1]}")
    stream = cupy.cuda.get_current_stream()

    flops = kda_decode_flops(shape)
    bytes_ = kda_decode_bytes(shape)
    print(f"[bench] flops/iter = {flops / 1e6:.3f} MFLOPS")
    print(f"[bench] bytes/iter = {bytes_ / 1024:.1f} KB")

    for _ in range(WARMUP):
        ct.launch(
            stream.ptr, grid, kernel,
            (q_d, k_d, v_d, g_d, b_d, s_in_d, s_out_d, o_d),
        )
    cupy.cuda.runtime.deviceSynchronize()

    starts = [cupy.cuda.Event() for _ in range(ITERS)]
    ends = [cupy.cuda.Event() for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record(stream)
        ct.launch(
            stream.ptr, grid, kernel,
            (q_d, k_d, v_d, g_d, b_d, s_in_d, s_out_d, o_d),
        )
        ends[i].record(stream)
    stream.synchronize()

    ms_list = []
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "impl", "kernel", "batch", "n_heads", "d_k", "d_v",
            "block_v", "iter", "gpu_us", "gbps", "tflops",
        ])
        for i in range(ITERS):
            gpu_ms = cupy.cuda.get_elapsed_time(starts[i], ends[i])
            ms_list.append(gpu_ms)
            gbps = kda_decode_gbps(shape, gpu_ms)
            tf = kda_decode_tflops(shape, gpu_ms)
            if i < 3 or i == ITERS - 1:
                print(
                    f"[bench] iter={i:2d} gpu_us={gpu_ms*1000:.2f} "
                    f"gbps={gbps:.1f} tflops={tf:.3f}"
                )
            w.writerow([
                "cutile", "kda_decode_fused", shape.batch, shape.n_heads,
                shape.d_k, shape.d_v, bv, i,
                f"{gpu_ms*1000:.3f}", f"{gbps:.3f}", f"{tf:.6f}",
            ])

    ms_sorted = sorted(ms_list)
    med = statistics.median(ms_list)
    q1 = ms_sorted[ITERS // 4]
    q3 = ms_sorted[(3 * ITERS) // 4]
    best = ms_sorted[0]
    med_gbps = kda_decode_gbps(shape, med)
    best_gbps = kda_decode_gbps(shape, best)
    best_tf = kda_decode_tflops(shape, best)
    peak_gbps = 1792.0

    print()
    print("=" * 72)
    print(f" BENCH SUMMARY — {shape.name} (B={shape.batch} H={shape.n_heads} "
          f"d_k={shape.d_k} d_v={shape.d_v})")
    print(f"   BLOCK_V={bv}  grid={grid}")
    print(f"   iters={ITERS}  warmup={WARMUP}")
    print(f"   median: {med*1000:.2f} us   {med_gbps:.1f} GB/s   "
          f"({med_gbps/peak_gbps:.1%} of 5090 HBM peak)")
    print(f"   IQR   : [{q1*1000:.2f}, {q3*1000:.2f}] us")
    print(f"   best  : {best*1000:.2f} us   {best_gbps:.1f} GB/s   "
          f"({best_gbps/peak_gbps:.1%} of peak)   {best_tf:.3f} TFLOPS")
    print(f"   headline: GB/s ({best_gbps:.1f}) — KDA decode is memory-bound")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Cubin export
# ─────────────────────────────────────────────────────────────────────────────


def _ac(dt):
    return ArrayConstraint(
        dtype=dt, ndim=2, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=(), may_alias_internally=False,
        stride_constant=(None, 1), stride_divisible_by=1,
        shape_divisible_by=1, base_addr_divisible_by=1,
    )


def export_cubin(out_path: str, shape: KDAShape) -> str | None:
    bv = pick_block_v(shape)
    kernel = make_kda_decode_kernel(shape.d_k, shape.d_v, bv)
    sig = KernelSignature(
        parameters=[
            _ac(ct.float16),  # Q
            _ac(ct.float16),  # K
            _ac(ct.float16),  # V
            _ac(ct.float16),  # G       — ADR-0006 §2(b): vector instead of scalar Alpha
            _ac(ct.float16),  # Beta
            _ac(ct.float32),  # S_in
            _ac(ct.float32),  # S_out
            _ac(ct.float16),  # O
        ],
        calling_convention=CallingConvention.cutile_python_v1(),
    )
    try:
        export_kernel(
            kernel, [sig], out_path,
            gpu_code="sm_120", output_format="cubin",
        )
        import os
        size = os.path.getsize(out_path)
        print(f"  wrote {out_path}  ({size} bytes)")
        return out_path
    except Exception as e:
        print(
            f"  FAILED cubin export: {type(e).__name__}: {str(e)[:600]}",
            file=sys.stderr,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", default=False)
    ap.add_argument("--bench", action="store_true", default=False)
    ap.add_argument("--export-cubin", action="store_true", default=False)
    ap.add_argument("--csv-out", default="results.csv")
    ap.add_argument("--cubin-out", default="kda_decode_fused.cubin")
    args = ap.parse_args()

    if not (args.smoke or args.bench or args.export_cubin):
        args.smoke = True

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"device: {props['name'].decode()}")
    print(f"compute capability: sm_{props['major']}{props['minor']}")
    print()

    rc = 0
    if args.smoke:
        ok = run_smoke(SHAPE_CORRECTNESS)
        if not ok:
            rc = 1
        print()

    if args.bench:
        run_bench(SHAPE_KIMI_LINEAR_DECODE, args.csv_out)
        print()

    if args.export_cubin:
        print("Exporting cubin at bench shape for SASS inspection…")
        export_cubin(args.cubin_out, SHAPE_KIMI_LINEAR_DECODE)

    return rc


if __name__ == "__main__":
    sys.exit(main())
