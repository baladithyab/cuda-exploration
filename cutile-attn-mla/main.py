"""Wave 16.3 — cuTile fused MLA (Multi-Head Latent Attention, DeepSeek-V3).

Single @ct.kernel implementing FlashAttention-2-style forward attention over
MLA-shaped inputs. Pedagogical MVP: the kernel consumes already-up-projected
Q, K, V — no weight absorption, no separate latent cache, no on-the-fly up-
projection. That way the attention core itself is just "MHA with qk_head_dim
= 192 and v_head_dim = 128", and the weight-absorption-trick / latent-cache
memory win becomes an upstream optimization we can layer in a later wave.

MVP padding decision. qk_head_dim = 192 = 128 + 64. 192 is not a power of
two and cuTile's `ct.mma` prefers power-of-two inner dims. We **pad to 256**
in the host arrays (zeros in the trailing 64 columns) so both Q and K use a
256-wide tile. Extra compute cost: (256 - 192) / 192 = 33% wasted QK^T FLOPS,
plus 33% wasted HBM traffic for Q/K. Reported TFLOPS is against the TRUE
FLOPS count (using qk_head_dim = 192), so the padding shows up as a
~25% ceiling haircut on observed throughput vs what a 192-native kernel
would hit. Real FlashMLA dodges this via a split-and-combine (d_h=128
subtile + d_rope=64 subtile separately); that's a Wave 16.5+ refinement.

V and O are 128-wide (power of two already) — no padding needed there.

Layout trick (mirrors cutile-attn-gqa). All arrays are flattened over
(batch, head, seq) so the tiled_view stays 2D with static shape. For each
(bid0, bid1) = (flattened head = batch*n_h+h, q_block), we compute the row
block for Q and K.

Pitfalls carried forward:
  - ct.launch(stream.ptr, grid, kernel, args) — not kernel[grid](args).
  - Tile shape constants must be Python ints captured via closure; ct.Constant
    doesn't work for tile shapes in cuda-tile 1.3.0.
  - ct.mma(a, b, acc) returns new tile; always re-assign.
  - p_f16 = p.astype(ct.float16) before second ct.mma (both operands must
    be f16).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
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

# Hook the wave-15 shared infra (MLA shapes, FLOPS, tolerances).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(
    0, str(REPO_ROOT / "analysis" / "wave15-attention-architecture" / "reference")
)
from shapes_mla import MLAShape, SHAPE_CORRECTNESS_MLA, SHAPE_BENCH_MLA  # noqa: E402
from flops_mla import mla_attention_flops  # noqa: E402
from tolerances import get as get_tol  # noqa: E402

INPUTS_DIR = REPO_ROOT / "analysis" / "wave15-attention-architecture" / "inputs"

WARMUP = 2
ITERS = 10

# MVP: pad qk_head_dim up to next power of two. For DeepSeek-V3 (qk = 192)
# this gives 256. For correctness (qk = 96) this gives 128. Anything above
# 64 and ≤ 256 rounds to the next power of two.
def _pad_to_pow2_ge(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Kernel factory
# ─────────────────────────────────────────────────────────────────────────────

def make_mla_kernel(
    BLOCK_M: int, BLOCK_N: int,
    QK_PAD: int, D_V: int, SEQ: int, N_H: int,
    qk_head_dim_true: int,
):
    """Build a @ct.kernel specialized to these shape constants.

    Grid:
        bid0 ∈ [0, B * N_H)       flattened (batch, head)
        bid1 ∈ [0, SEQ // BLOCK_M) query block index

    Array layouts (pre-padded on host):
        Q : (B * N_H * SEQ, QK_PAD)  f16, trailing (QK_PAD - qk_head_dim_true)
                                     cols are zero
        K : (B * N_H * SEQ, QK_PAD)  f16, same padding
        V : (B * N_H * SEQ, D_V)     f16
        O : (B * N_H * SEQ, D_V)     f16

    Softmax scale uses the TRUE qk_head_dim, not the padded QK_PAD — padded
    columns are all zero so their contribution to QK^T is zero regardless
    of scale, but the user-visible attention must match the reference.
    """
    SEQ_TILES_M = SEQ // BLOCK_M
    SEQ_TILES_N = SEQ // BLOCK_N
    scale = 1.0 / math.sqrt(qk_head_dim_true)
    NEG_INF = -1.0e30

    @ct.kernel
    def mla_fwd(Q, K, V, O):
        bid0 = ct.bid(0)  # flattened (batch, head)
        bid1 = ct.bid(1)  # query block

        # Row base within each flattened 2D array. Each (batch, head) pair
        # occupies SEQ_TILES_M BLOCK_M-sized row tiles contiguously.
        q_tile_row = bid0 * SEQ_TILES_M + bid1
        kv_tile_row_base = bid0 * SEQ_TILES_N

        # Tiled views with zero-padding on the edge (should be no-op since
        # shapes divide evenly, but defensive).
        q_view = Q.tiled_view((BLOCK_M, QK_PAD), padding_mode=ct.PaddingMode.ZERO)
        k_view = K.tiled_view((BLOCK_N, QK_PAD), padding_mode=ct.PaddingMode.ZERO)
        v_view = V.tiled_view((BLOCK_N, D_V), padding_mode=ct.PaddingMode.ZERO)
        o_view = O.tiled_view((BLOCK_M, D_V), padding_mode=ct.PaddingMode.ZERO)

        # Load Q tile once — persists across the K/V loop.
        q_tile = q_view.load((q_tile_row, 0))  # (BLOCK_M, QK_PAD) f16

        # Online-softmax state (all f32, in registers).
        m_i = ct.full((BLOCK_M, 1), NEG_INF, ct.float32)
        l_i = ct.zeros((BLOCK_M, 1), ct.float32)
        o_acc = ct.zeros((BLOCK_M, D_V), ct.float32)

        for kb in range(SEQ_TILES_N):
            k_tile = k_view.load((kv_tile_row_base + kb, 0))  # (BLOCK_N, QK_PAD) f16
            v_tile = v_view.load((kv_tile_row_base + kb, 0))  # (BLOCK_N, D_V) f16

            # QK^T: (BLOCK_M, QK_PAD) × (QK_PAD, BLOCK_N) → (BLOCK_M, BLOCK_N) f32
            k_t = ct.transpose(k_tile)  # (QK_PAD, BLOCK_N) f16
            s_acc = ct.zeros((BLOCK_M, BLOCK_N), ct.float32)
            s_acc = ct.mma(q_tile, k_t, s_acc)

            s_scaled = s_acc * scale

            m_row = ct.max(s_scaled, axis=1, keepdims=True)       # (BLOCK_M, 1) f32
            m_new = ct.maximum(m_i, m_row)                        # (BLOCK_M, 1) f32
            alpha = ct.exp(m_i - m_new)                           # (BLOCK_M, 1) f32

            p = ct.exp(s_scaled - m_new)                          # (BLOCK_M, BLOCK_N) f32
            p_row_sum = ct.sum(p, axis=1, keepdims=True)          # (BLOCK_M, 1) f32

            l_i = alpha * l_i + p_row_sum
            o_acc = o_acc * alpha                                 # broadcast over D_V

            p_f16 = p.astype(ct.float16)
            # PV: (BLOCK_M, BLOCK_N) × (BLOCK_N, D_V) → (BLOCK_M, D_V) f32
            o_acc = ct.mma(p_f16, v_tile, o_acc)

            m_i = m_new

        o_final = o_acc / l_i  # (BLOCK_M, D_V) f32
        o_view.store((q_tile_row, 0), o_final.astype(O.dtype))

    return mla_fwd


# ─────────────────────────────────────────────────────────────────────────────
# Input I/O + padding
# ─────────────────────────────────────────────────────────────────────────────

def load_inputs(shape: MLAShape):
    prefix = INPUTS_DIR / f"mla_{shape.name}"
    q_np = np.load(f"{prefix}_q_f16.npy")
    k_np = np.load(f"{prefix}_k_f16.npy")
    v_np = np.load(f"{prefix}_v_f16.npy")
    exp_np = np.load(f"{prefix}_expected_f32.npy")
    return q_np, k_np, v_np, exp_np


def prepare_device(q_np, k_np, v_np, shape: MLAShape, qk_pad: int):
    """Reshape to 2D, pad Q/K's last dim from qk_head_dim → qk_pad, upload.

    Q,K input shape: (B, n_h, S, qk_head_dim) → (B*n_h*S, qk_pad) with
    trailing columns zero. V: (B, n_h, S, d_v) → (B*n_h*S, d_v).
    """
    B, S, N = shape.batch, shape.seq, shape.n_h
    qk = shape.qk_head_dim
    dv = shape.d_v
    assert q_np.shape == (B, N, S, qk)
    assert k_np.shape == (B, N, S, qk)
    assert v_np.shape == (B, N, S, dv)

    if qk_pad == qk:
        q_2d = q_np.reshape(B * N * S, qk)
        k_2d = k_np.reshape(B * N * S, qk)
    else:
        # Zero-pad trailing (qk_pad - qk) columns.
        q_2d = np.zeros((B * N * S, qk_pad), dtype=np.float16)
        k_2d = np.zeros((B * N * S, qk_pad), dtype=np.float16)
        q_2d[:, :qk] = q_np.reshape(B * N * S, qk)
        k_2d[:, :qk] = k_np.reshape(B * N * S, qk)
    v_2d = v_np.reshape(B * N * S, dv)
    return (
        cupy.asarray(q_2d, dtype=cupy.float16),
        cupy.asarray(k_2d, dtype=cupy.float16),
        cupy.asarray(v_2d, dtype=cupy.float16),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Block-size picker
# ─────────────────────────────────────────────────────────────────────────────

def pick_blocks(shape: MLAShape) -> tuple[int, int]:
    """Default 64×64 per Wave 15.1 finding (128×128 falls off register cliff
    even at d=128; MLA's 256-wide Q tile is tighter on registers still).

    Small correctness shape uses 32×32 so we get multiple Q and K blocks.
    """
    if shape.seq >= 512:
        return 64, 64
    bm = min(32, shape.seq)
    bn = min(32, shape.seq)
    assert shape.seq % bm == 0 and shape.seq % bn == 0
    return bm, bn


# ─────────────────────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke(shape: MLAShape) -> bool:
    print(
        f"[smoke] shape={shape.name} B={shape.batch} S={shape.seq} "
        f"n_h={shape.n_h} qk={shape.qk_head_dim} d_v={shape.d_v}"
    )
    q_np, k_np, v_np, expected = load_inputs(shape)
    qk_pad = _pad_to_pow2_ge(shape.qk_head_dim)
    print(f"[smoke] qk_head_dim={shape.qk_head_dim} → qk_pad={qk_pad} "
          f"(wasted_cols={qk_pad - shape.qk_head_dim})")
    q_d, k_d, v_d = prepare_device(q_np, k_np, v_np, shape, qk_pad)
    o_d = cupy.zeros((q_d.shape[0], shape.d_v), dtype=cupy.float16)

    bm, bn = pick_blocks(shape)
    print(f"[smoke] BLOCK_M={bm} BLOCK_N={bn}")
    kernel = make_mla_kernel(
        bm, bn, qk_pad, shape.d_v, shape.seq, shape.n_h,
        qk_head_dim_true=shape.qk_head_dim,
    )

    grid = (shape.batch * shape.n_h, shape.seq // bm)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream.ptr, grid, kernel, (q_d, k_d, v_d, o_d))
    cupy.cuda.runtime.deviceSynchronize()

    # Reshape back and compare in f32.
    o_np = o_d.get().reshape(shape.batch, shape.n_h, shape.seq, shape.d_v)
    o_f32 = o_np.astype(np.float32)

    tol = get_tol("f16")
    abs_err = np.abs(o_f32 - expected)
    ref_mag = np.abs(expected).max() + 1e-30
    max_abs = float(abs_err.max())
    rel_err = max_abs / ref_mag
    ok = np.allclose(o_f32, expected, atol=tol.atol, rtol=tol.rtol)
    status = "OK" if ok else "FAIL"
    print(
        f"[smoke] max_abs={max_abs:.3e} rel={rel_err:.3e}  "
        f"atol={tol.atol} rtol={tol.rtol}  {status}"
    )
    if not ok:
        worst = np.unravel_index(abs_err.argmax(), abs_err.shape)
        print(
            f"        worst at {worst}: got={o_f32[worst]:.4f} "
            f"expected={expected[worst]:.4f}"
        )
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Bench
# ─────────────────────────────────────────────────────────────────────────────

def run_bench(shape: MLAShape, csv_path: str) -> None:
    print(
        f"[bench] shape={shape.name} B={shape.batch} S={shape.seq} "
        f"n_h={shape.n_h} qk={shape.qk_head_dim} d_v={shape.d_v}"
    )
    q_np, k_np, v_np, _expected = load_inputs(shape)
    qk_pad = _pad_to_pow2_ge(shape.qk_head_dim)
    print(f"[bench] qk_head_dim={shape.qk_head_dim} → qk_pad={qk_pad}")
    q_d, k_d, v_d = prepare_device(q_np, k_np, v_np, shape, qk_pad)
    o_d = cupy.zeros((q_d.shape[0], shape.d_v), dtype=cupy.float16)

    bm, bn = pick_blocks(shape)
    print(f"[bench] BLOCK_M={bm} BLOCK_N={bn}")
    kernel = make_mla_kernel(
        bm, bn, qk_pad, shape.d_v, shape.seq, shape.n_h,
        qk_head_dim_true=shape.qk_head_dim,
    )

    grid = (shape.batch * shape.n_h, shape.seq // bm)
    stream = cupy.cuda.get_current_stream()
    flops = mla_attention_flops(shape)

    for _ in range(WARMUP):
        ct.launch(stream.ptr, grid, kernel, (q_d, k_d, v_d, o_d))
    cupy.cuda.runtime.deviceSynchronize()

    starts = [cupy.cuda.Event() for _ in range(ITERS)]
    ends = [cupy.cuda.Event() for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record(stream)
        ct.launch(stream.ptr, grid, kernel, (q_d, k_d, v_d, o_d))
        ends[i].record(stream)
    stream.synchronize()

    rows = []
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "impl", "kernel", "batch", "seq", "n_h", "qk_head_dim", "qk_pad",
            "d_v", "block_m", "block_n", "iter", "gpu_ms", "tflops",
        ])
        for i in range(ITERS):
            gpu_ms = cupy.cuda.get_elapsed_time(starts[i], ends[i])
            tflops = flops / (gpu_ms * 1e-3) / 1e12
            print(f"[bench] iter={i} gpu_ms={gpu_ms:.3f} tflops={tflops:.3f}")
            w.writerow([
                "cutile", "mla_fwd_fused", shape.batch, shape.seq, shape.n_h,
                shape.qk_head_dim, qk_pad, shape.d_v, bm, bn, i,
                f"{gpu_ms:.6f}", f"{tflops:.6f}",
            ])
            rows.append((gpu_ms, tflops))

    ms_sorted = sorted(r[0] for r in rows)
    tf_sorted = sorted(r[1] for r in rows)
    median_ms = ms_sorted[ITERS // 2]
    median_tf = tf_sorted[ITERS // 2]
    best_ms = ms_sorted[0]
    best_tf = tf_sorted[-1]
    print()
    print("=" * 64)
    print(
        f" BENCH SUMMARY — {shape.name} (B={shape.batch} S={shape.seq} "
        f"n_h={shape.n_h} qk={shape.qk_head_dim} d_v={shape.d_v})"
    )
    print(f"   BLOCK_M={bm} BLOCK_N={bn} QK_PAD={qk_pad} grid={grid}")
    print(f"   median : {median_ms:.3f} ms   {median_tf:.3f} TFLOPS")
    print(f"   best   : {best_ms:.3f} ms   {best_tf:.3f} TFLOPS")
    print(f"   cuBLAS hgemm peak (Wave 14.1) : 218 TFLOPS  → ratio {best_tf/218:.2%}")
    print(f"   cutile-attn-gqa (Wave 15.1)   : 165 TFLOPS  → ratio {best_tf/165:.2%}")
    print("=" * 64)


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


def export_cubin(out_path: str, shape: MLAShape) -> str | None:
    bm, bn = pick_blocks(shape)
    qk_pad = _pad_to_pow2_ge(shape.qk_head_dim)
    kernel = make_mla_kernel(
        bm, bn, qk_pad, shape.d_v, shape.seq, shape.n_h,
        qk_head_dim_true=shape.qk_head_dim,
    )
    sig = KernelSignature(
        parameters=[_ac(ct.float16), _ac(ct.float16), _ac(ct.float16), _ac(ct.float16)],
        calling_convention=CallingConvention.cutile_python_v1(),
    )
    try:
        export_kernel(kernel, [sig], out_path, gpu_code="sm_120", output_format="cubin")
        import os
        size = os.path.getsize(out_path)
        print(f"  wrote {out_path}  ({size} bytes)")
        return out_path
    except Exception as e:
        print(f"  FAILED cubin export: {type(e).__name__}: {str(e)[:400]}", file=sys.stderr)
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
    ap.add_argument("--cubin-out", default="mla_fwd_fused.cubin")
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
        ok = run_smoke(SHAPE_CORRECTNESS_MLA)
        if not ok:
            rc = 1
        print()

    if args.bench:
        run_bench(SHAPE_BENCH_MLA, args.csv_out)
        print()

    if args.export_cubin:
        print("Exporting cubin at bench shape for SASS inspection…")
        export_cubin(args.cubin_out, SHAPE_BENCH_MLA)

    return rc


if __name__ == "__main__":
    sys.exit(main())
