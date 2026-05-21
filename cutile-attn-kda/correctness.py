"""Wave 17.W1e — KDA correctness test (separate harness for the orchestrator).

Runs the kernel at SHAPE_CORRECTNESS and ALSO at SHAPE_KIMI_LINEAR_DECODE,
comparing kernel output against `naive_recurrent_kda_step` (the f64 numpy
oracle inlined in main.py).  The acceptance bar from ADR-0006 / wave-17.md
is `max_abs_err ≤ 1e-3` vs the oracle.

Usage:
    python correctness.py
    python correctness.py --shape kimi_linear_decode

This file is not on the ADR-0006 §2 fork-fence (the fence applies to
main.py only).  It exists per the wave-17 plan-row deliverables.
"""
from __future__ import annotations

import argparse
import sys

import cupy
import numpy as np

import cuda.tile as ct

# Re-export the helpers from main.py so the harness has zero duplicated logic.
from main import (
    SHAPE_CORRECTNESS,
    SHAPE_KIMI_LINEAR_DECODE,
    KDAShape,
    load_inputs,
    make_kda_decode_kernel,
    naive_recurrent_kda_step,
    pick_block_v,
    prepare_device,
)

THRESHOLD = 1e-3  # ADR-0006 acceptance bar
SHAPES = {
    "correctness": SHAPE_CORRECTNESS,
    "kimi_linear_decode": SHAPE_KIMI_LINEAR_DECODE,
}


def check(shape: KDAShape) -> tuple[bool, float, float]:
    """Run the kernel at `shape` and return (ok, max_abs_o, max_abs_S).

    Comparison is against the **f64 oracle** (no f16 round-trip on the
    expected side) so we get a tight bound on the kernel's f32-arith
    correctness.  `ok` is `max_abs_o ≤ 1e-3 AND max_abs_S ≤ 1e-3`.
    """
    print(
        f"\n[check] shape={shape.name} B={shape.batch} H={shape.n_heads} "
        f"d_k={shape.d_k} d_v={shape.d_v}"
    )
    inp = load_inputs(shape)
    q_d, k_d, v_d, g_d, b_d, s_in_d, s_out_d, o_d = prepare_device(shape, inp)

    bv = pick_block_v(shape)
    print(f"[check] BLOCK_V={bv}")
    kernel = make_kda_decode_kernel(shape.d_k, shape.d_v, bv)
    grid = (shape.batch * shape.n_heads, shape.d_v // bv)
    stream = cupy.cuda.get_current_stream()

    ct.launch(
        stream.ptr, grid, kernel,
        (q_d, k_d, v_d, g_d, b_d, s_in_d, s_out_d, o_d),
    )
    cupy.cuda.runtime.deviceSynchronize()

    B, H, dk, dv = shape.batch, shape.n_heads, shape.d_k, shape.d_v
    o_kernel = o_d.get().astype(np.float32).reshape(B, H, dv)
    s_kernel = s_out_d.get().astype(np.float32).reshape(B, H, dk, dv)

    # f64 oracle (no f16 cast on the reference side).
    o_exp, s_exp = naive_recurrent_kda_step(
        inp["q"].astype(np.float32),
        inp["k"].astype(np.float32),
        inp["v"].astype(np.float32),
        inp["g"].astype(np.float32),
        inp["beta"].astype(np.float32),
        inp["S_in"],
        scale=1.0,
    )

    err_o = np.abs(o_kernel - o_exp.astype(np.float32))
    err_s = np.abs(s_kernel - s_exp.astype(np.float32))
    max_o = float(err_o.max())
    max_s = float(err_s.max())
    rel_o = max_o / (float(np.abs(o_exp).max()) + 1e-30)
    rel_s = max_s / (float(np.abs(s_exp).max()) + 1e-30)

    ok_o = max_o <= THRESHOLD
    ok_s = max_s <= THRESHOLD
    print(
        f"[check] o    max_abs={max_o:.3e}  rel={rel_o:.3e}  "
        f"{'OK' if ok_o else 'FAIL'} (≤ {THRESHOLD:.0e})"
    )
    print(
        f"[check] S    max_abs={max_s:.3e}  rel={rel_s:.3e}  "
        f"{'OK' if ok_s else 'FAIL'} (≤ {THRESHOLD:.0e})"
    )

    if not ok_o:
        w = np.unravel_index(err_o.argmax(), err_o.shape)
        print(
            f"        worst o offender at {w}: "
            f"got={o_kernel[w]:.6f} expected={o_exp[w]:.6f}"
        )
    if not ok_s:
        w = np.unravel_index(err_s.argmax(), err_s.shape)
        print(
            f"        worst S offender at {w}: "
            f"got={s_kernel[w]:.6f} expected={s_exp[w]:.6f}"
        )

    return (ok_o and ok_s), max_o, max_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shape",
        choices=("correctness", "kimi_linear_decode", "all"),
        default="all",
        help="Which shape(s) to verify (default: all).",
    )
    args = ap.parse_args()

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"device: {props['name'].decode()}")
    print(f"compute capability: sm_{props['major']}{props['minor']}")
    print(f"acceptance threshold: max_abs_err ≤ {THRESHOLD:.0e} (ADR-0006)")

    todo = (
        ["correctness", "kimi_linear_decode"]
        if args.shape == "all"
        else [args.shape]
    )

    summary = []
    rc = 0
    for name in todo:
        ok, mo, ms = check(SHAPES[name])
        summary.append((name, ok, mo, ms))
        if not ok:
            rc = 1

    print()
    print("=" * 68)
    print(" CORRECTNESS SUMMARY (ADR-0006 W1e acceptance)")
    print("=" * 68)
    for name, ok, mo, ms in summary:
        verdict = "PASS" if ok else "FAIL"
        print(
            f"  {name:<22}  o:{mo:.3e}  S:{ms:.3e}  -> {verdict}"
        )
    print("=" * 68)
    return rc


if __name__ == "__main__":
    sys.exit(main())
