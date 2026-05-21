"""Wave 17 W2d — BLOCK_M sweep for cutile fused GQA attention.

Reuses the kernel factory + run_smoke + run_bench logic in main.py and
varies BLOCK_M ∈ {64, 128, 256} while holding BLOCK_N = 64 fixed (the
known-good ct.mma tile width from Wave 13.1).

The plan claim under test:
    BLOCK_M=64  → 165 TF (existing baseline, 75.9% of cuBLAS hgemm peak)
    BLOCK_M=128 → ?       (might close the 24% gap to 218 TF)
    BLOCK_M=256 → register-cliff suspected (cuTile pitfall #10:
                  silent local-mem spill, 3-4× perf drop)

What this script does (author / orchestrator split):
  - per BLOCK_M: pick a shape that satisfies (seq % BLOCK_M == 0):
      * BLOCK_M ∈ {64, 128} → run correctness at SHAPE_CORRECTNESS (seq=128)
      * BLOCK_M = 256       → seq=128 too small; only the bench shape
                              (seq=2048) divides evenly. Run a compile-only
                              kernel build via export_kernel + cubin write
                              (sm_120) to confirm the kernel codegens. No
                              live correctness check at bench shape because
                              we have no precomputed expected_f32 fast path
                              there and timed runs are orchestrator-only.
  - emits sweep_block_m_results.csv with one row per swept value:
        block_m, block_n, correctness_status, compile_status, notes

Author scope (this script): NO timed benches, NO jj, NO commits. The
orchestrator runs the timed sweep separately (`python main.py --bench
--block-m 64`, `--block-m 128`, `--block-m 256`) and merges the per-tile
results.csv files into the headline.

Run:
    /home/codeseys/cuda-exploration/cutile-vecadd-bench/.venv/bin/python \
        sweep_block_m.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Reuse main.py's primitives. Both files live in the same cell.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import main as _main  # noqa: E402
from main import (  # noqa: E402
    SHAPE_BENCH,
    SHAPE_CORRECTNESS,
    export_cubin,
    make_gqa_kernel,
    run_smoke,
)

# BLOCK_M values to sweep. BLOCK_N stays 64 throughout (Wave 13.1 sweet spot).
BLOCK_M_SWEEP = [64, 128, 256]
BLOCK_N_FIXED = 64


def correctness_phase(block_m: int) -> tuple[str, str]:
    """Returns (correctness_status, notes).

    Statuses:
        PASS — kernel ran at SHAPE_CORRECTNESS and matched expected within tol
        FAIL — kernel ran but numerical mismatch
        SKIP — BLOCK_M doesn't divide seq=128 (or BLOCK_N doesn't either)
        ERROR — exception during compile/launch
    """
    bm = block_m
    bn = BLOCK_N_FIXED
    seq = SHAPE_CORRECTNESS.seq  # 128

    if seq % bm != 0:
        return ("SKIP", f"seq={seq} not divisible by BLOCK_M={bm}; "
                        "compile-only at bench shape")
    if seq % bn != 0:
        return ("SKIP", f"seq={seq} not divisible by BLOCK_N={bn}")

    try:
        ok = run_smoke(SHAPE_CORRECTNESS, block_m=bm, block_n=bn)
    except Exception as e:  # noqa: BLE001 — broad catch is fine in sweep
        return ("ERROR", f"{type(e).__name__}: {str(e)[:200]}")
    return ("PASS" if ok else "FAIL", "")


def compile_phase(block_m: int, cubin_dir: Path) -> tuple[str, str]:
    """Confirm the kernel codegens at SHAPE_BENCH for this BLOCK_M.

    Used both as a corroborating signal for BLOCK_M ∈ {64,128} and as the
    primary 'kernel works' signal for BLOCK_M=256 (since seq=128 can't
    host a 256-wide query tile).

    Returns (compile_status, notes). Status is OK / ERROR.
    """
    bm = block_m
    bn = BLOCK_N_FIXED
    if SHAPE_BENCH.seq % bm != 0 or SHAPE_BENCH.seq % bn != 0:
        return ("SKIP",
                f"bench seq={SHAPE_BENCH.seq} not divisible by ({bm},{bn})")

    cubin_path = cubin_dir / f"gqa_fwd_fused_bm{bm}.cubin"
    try:
        # First, just instantiate the kernel factory to catch shape errors.
        _ = make_gqa_kernel(bm, bn, SHAPE_BENCH.d_head,
                            SHAPE_BENCH.seq, SHAPE_BENCH.n_q, SHAPE_BENCH.n_kv)
        # Then export — exercises the full nvvm/ptxas path and catches
        # register-pressure local-memory spills via stderr/log messages.
        out = export_cubin(str(cubin_path), SHAPE_BENCH,
                           block_m=bm, block_n=bn)
        if out is None:
            return ("ERROR", "export_kernel returned None")
    except Exception as e:  # noqa: BLE001
        return ("ERROR", f"{type(e).__name__}: {str(e)[:200]}")

    size = cubin_path.stat().st_size if cubin_path.exists() else 0
    return ("OK", f"cubin {cubin_path.name} ({size} bytes)")


def lmem_check(cubin_path: Path) -> str:
    """Best-effort SASS scan for local-memory spill markers.

    A register-cliff at large BLOCK_M shows up in SASS as LDL/STL
    (load-local / store-local) instructions in the inner loop. Returns a
    short summary string used in the sweep CSV's notes column.
    """
    if not cubin_path.exists():
        return "no cubin"
    try:
        import subprocess
        sass_path = cubin_path.with_suffix(".sass")
        # Best-effort; tolerate missing cuobjdump.
        for tool in ("/usr/local/cuda/bin/cuobjdump", "cuobjdump"):
            try:
                with open(sass_path, "w") as f:
                    subprocess.run(
                        [tool, "--dump-sass", str(cubin_path)],
                        stdout=f, stderr=subprocess.DEVNULL,
                        check=True, timeout=30,
                    )
                break
            except (FileNotFoundError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired):
                continue
        else:
            return "cuobjdump unavailable"

        if not sass_path.exists() or sass_path.stat().st_size == 0:
            return "sass dump empty"
        text = sass_path.read_text(errors="ignore")
        ldl = sum(1 for ln in text.splitlines() if " LDL" in ln)
        stl = sum(1 for ln in text.splitlines() if " STL" in ln)
        hmma = sum(1 for ln in text.splitlines() if "HMMA" in ln)
        spill = "spill" if (ldl + stl) > 0 else "no-spill"
        return f"HMMA={hmma} LDL={ldl} STL={stl} ({spill})"
    except Exception as e:  # noqa: BLE001
        return f"sass-scan-error: {type(e).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-out", default="sweep_block_m_results.csv",
                    help="Sweep summary CSV (default: sweep_block_m_results.csv)")
    ap.add_argument("--cubin-dir", default=".",
                    help="Where to write per-BLOCK_M cubins")
    ap.add_argument("--values", type=int, nargs="+", default=BLOCK_M_SWEEP,
                    help=f"BLOCK_M values to sweep (default: {BLOCK_M_SWEEP})")
    args = ap.parse_args()

    cubin_dir = Path(args.cubin_dir).resolve()
    cubin_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_out)

    print(f"=== Wave 17 W2d : BLOCK_M sweep — values={args.values} "
          f"BLOCK_N={BLOCK_N_FIXED} ===")
    print(f"   cuTile module : {_main.ct.__version__}")
    print(f"   csv-out       : {csv_path}")
    print(f"   cubin-dir     : {cubin_dir}")
    print()

    rows: list[dict] = []
    for bm in args.values:
        print(f"--- BLOCK_M = {bm} -----------------------------------------")

        corr_status, corr_note = correctness_phase(bm)
        print(f"  correctness : {corr_status}  {corr_note}")

        comp_status, comp_note = compile_phase(bm, cubin_dir)
        print(f"  compile@bench: {comp_status}  {comp_note}")

        cubin_path = cubin_dir / f"gqa_fwd_fused_bm{bm}.cubin"
        sass_note = lmem_check(cubin_path)
        print(f"  sass        : {sass_note}")

        notes = " | ".join(filter(None, [corr_note, comp_note, sass_note]))
        rows.append({
            "block_m": bm,
            "block_n": BLOCK_N_FIXED,
            "correctness_status": corr_status,
            "compile_status": comp_status,
            "notes": notes,
        })
        print()

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["block_m", "block_n", "correctness_status",
                        "compile_status", "notes"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"=== sweep summary written to {csv_path} ===")
    for r in rows:
        print(f"  BLOCK_M={r['block_m']:<4} corr={r['correctness_status']:<6} "
              f"compile={r['compile_status']:<6}  {r['notes']}")

    # Exit non-zero only if every value errored (the sweep itself "worked"
    # otherwise, even when it found a register-cliff).
    bad = [r for r in rows if r["correctness_status"] in ("ERROR",)
                            and r["compile_status"] in ("ERROR",)]
    return 1 if bad and len(bad) == len(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
