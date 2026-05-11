"""Wave 13.1 — cuTile mixed-precision tiled matmul.

Falsifies the Wave 12 hypothesis that "cuTile is bad at matmul" by testing
whether `ct.mma` engages tensor cores at f16/bf16/tf32 input dtypes even
though it doesn't at f32 × f32. Four variants:

    mma_f16xf16_f32acc    — input f16, accumulator f32, output f16 (cast at store)
    mma_bf16xbf16_f32acc  — input bf16, accumulator f32, output bf16
    mma_tf32xtf32_f32acc  — input storage f32, cast to tfloat32 for mma, f32 out
    mma_f32xf32_f32acc    — baseline f32 (re-runs Wave 12.4 for this session's A/B)

All use BM=BN=128, BK=16. Shape-divisibility confirmed to compile on
cuda-tile 1.3.0 for all four dtypes.

CLI:
    --smoke                 (default): correctness check at N=512
    --bench                  timed sweep [1024, 2048, 4096], 1 warmup + 10 iters
    --export-cubins          compile each variant's cubin via export_kernel()
    --csv-out FILE           bench CSV output path

Notes on cuTile 1.3.0 pitfalls (carried from cutile-vecadd-bench / cutile-matmul-tiled):

  - `ct.launch(stream.ptr, grid, kernel, args_tuple)` — NOT `kernel[grid](args)`.
  - Use Python-closure factories for tile shapes: `ct.Constant[int]` launch-args
    fail with `TileTypeError: Invalid argument tile_shape`.
  - For tf32 compute with f32 storage: cast tiles with `.astype(ct.tfloat32)`
    before `ct.mma`. The dtype is a compute hint; storage stays f32.
  - `ct.mma(a, b, acc)` returns a new tile — the assignment `acc = ct.mma(...)`
    is required.
  - Branching on Python-level dtype identity (`if dtype is ct.tfloat32:`)
    inside a kernel fails type-check — use separate kernel factories.
"""

from __future__ import annotations

import argparse
import csv
import sys

import cuda.tile as ct
import cupy
import ml_dtypes
import numpy as np
from cuda.tile.compilation import (
    ArrayConstraint,
    CallingConvention,
    KernelSignature,
    export_kernel,
)

# Default tile sizes.
BM = 128
BN = 128
BK = 16

# Correctness N.
CORRECTNESS_N = 512

# Bench sizes.
BENCH_SIZES = [1024, 2048, 4096]
WARMUP = 1
ITERS = 10

# Relative-error tolerances — f16/bf16 are noisier than tf32/f32.
TOL_F16 = 1e-2
TOL_BF16 = 3e-2  # bf16 has only 8 mantissa bits; 1e-2 is borderline at N=512
TOL_TF32 = 1e-3
TOL_F32 = 1e-3


# ────────────────────────────────────────────────────────────────────
# Kernel factories
# ────────────────────────────────────────────────────────────────────


def make_mm_plain(in_dtype, acc_dtype, bm: int, bn: int, bk: int):
    """Straight-through MMA: input tiles cast directly into ct.mma."""

    @ct.kernel
    def matmul_plain(A, B, C):
        i, j = ct.bid(0), ct.bid(1)
        a_view = A.tiled_view((bm, bk), padding_mode=ct.PaddingMode.ZERO)
        b_view = B.tiled_view((bk, bn), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.zeros((bm, bn), acc_dtype)
        for k in range(a_view.num_tiles(1)):
            tx = a_view.load((i, k))
            ty = b_view.load((k, j))
            acc = ct.mma(tx, ty, acc)
        ct.store(C, (i, j), acc.astype(C.dtype))

    return matmul_plain


def make_mm_tf32(bm: int, bn: int, bk: int):
    """TF32 path: f32 storage, explicit cast to ct.tfloat32 before ct.mma."""

    @ct.kernel
    def matmul_tf32(A, B, C):
        i, j = ct.bid(0), ct.bid(1)
        a_view = A.tiled_view((bm, bk), padding_mode=ct.PaddingMode.ZERO)
        b_view = B.tiled_view((bk, bn), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.zeros((bm, bn), ct.float32)
        for k in range(a_view.num_tiles(1)):
            tx = a_view.load((i, k)).astype(ct.tfloat32)
            ty = b_view.load((k, j)).astype(ct.tfloat32)
            acc = ct.mma(tx, ty, acc)
        ct.store(C, (i, j), acc.astype(C.dtype))

    return matmul_tf32


# Instantiate the four variants (tile shapes identical across dtypes).
mma_f16xf16_f32acc = make_mm_plain(ct.float16, ct.float32, BM, BN, BK)
mma_bf16xbf16_f32acc = make_mm_plain(ct.bfloat16, ct.float32, BM, BN, BK)
mma_tf32xtf32_f32acc = make_mm_tf32(BM, BN, BK)
mma_f32xf32_f32acc = make_mm_plain(ct.float32, ct.float32, BM, BN, BK)


# Dtype registry for the --smoke / --bench / --export code paths.
# Each entry: (variant_name, cupy_dtype, cuda_tile_dtype, kernel, rel_tol)
VARIANTS = [
    ("mma_f16xf16_f32acc", cupy.float16, ct.float16,
     mma_f16xf16_f32acc, TOL_F16),
    ("mma_bf16xbf16_f32acc", ml_dtypes.bfloat16, ct.bfloat16,
     mma_bf16xbf16_f32acc, TOL_BF16),
    ("mma_tf32xtf32_f32acc", cupy.float32, ct.float32,  # storage dtype f32
     mma_tf32xtf32_f32acc, TOL_TF32),
    ("mma_f32xf32_f32acc", cupy.float32, ct.float32,
     mma_f32xf32_f32acc, TOL_F32),
]


# ────────────────────────────────────────────────────────────────────
# Correctness
# ────────────────────────────────────────────────────────────────────


def make_inputs(n: int, np_dtype):
    """Build two n×n input arrays + cupy reference in the same dtype.
    For bf16 we seed through cupy f32 then cast, since random.default_rng
    doesn't dispatch on ml_dtypes dtype directly."""
    rng_np = np.random.default_rng(0xC0FFEE)
    a_np = rng_np.random((n, n), dtype=np.float32).astype(np_dtype)
    b_np = rng_np.random((n, n), dtype=np.float32).astype(np_dtype)
    a = cupy.asarray(a_np)
    b = cupy.asarray(b_np)
    # Reference: cupy.matmul handles bf16 via ml_dtypes through numpy fallback.
    # For tf32 path the reference is plain f32 matmul (the tile-side mma_tf32
    # variant takes f32 storage too, but truncates mantissa at the tensor-core).
    expected = cupy.matmul(a, b)
    return a, b, expected


def run_correctness(n: int) -> dict[str, tuple[bool, float, str]]:
    """For each variant, verify output vs cupy.matmul reference at size n.
    Returns {variant_name: (pass, rel_err, error_msg_if_any)}."""
    stream = cupy.cuda.get_current_stream()
    results: dict[str, tuple[bool, float, str]] = {}

    for name, np_dtype, _ct_dtype, kernel, tol in VARIANTS:
        try:
            a, b, expected = make_inputs(n, np_dtype)
            # Output dtype = input dtype for f16/bf16, f32 for tf32 and f32 paths.
            if name == "mma_f16xf16_f32acc":
                out_dtype = cupy.float16
            elif name == "mma_bf16xbf16_f32acc":
                out_dtype = ml_dtypes.bfloat16
            else:
                out_dtype = cupy.float32

            out = cupy.zeros((n, n), dtype=out_dtype)
            assert n % BM == 0 and n % BN == 0 and n % BK == 0
            grid = (n // BM, n // BN)

            ct.launch(stream.ptr, grid, kernel, (a, b, out))
            cupy.cuda.runtime.deviceSynchronize()

            # Compare in f32 space.
            out_f32 = out.astype(cupy.float32)
            exp_f32 = expected.astype(cupy.float32)
            max_err = float(cupy.max(cupy.abs(out_f32 - exp_f32)))
            ref_mag = float(cupy.max(cupy.abs(exp_f32)))
            rel = max_err / max(ref_mag, 1e-6)
            ok = rel < tol
            results[name] = (ok, rel, "")
            status = "OK" if ok else "FAIL"
            print(f"[{name}] N={n} tol={tol:.0e}  "
                  f"max_abs={max_err:.3e} rel={rel:.3e}  {status}")

            del a, b, expected, out, out_f32, exp_f32
            cupy.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            results[name] = (False, float("nan"), err)
            print(f"[{name}] ERROR {err}", file=sys.stderr)

    return results


# ────────────────────────────────────────────────────────────────────
# Bench — tight block; run all iters per variant back-to-back
# ────────────────────────────────────────────────────────────────────


def run_bench(csv_path: str) -> None:
    stream = cupy.cuda.get_current_stream()

    rows = []  # collect for final summary print

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["impl", "kernel", "n", "iter", "gpu_ms", "tflops"])

        for n in BENCH_SIZES:
            total_flops = 2.0 * (n ** 3)
            for name, np_dtype, _ct_dtype, kernel, _tol in VARIANTS:
                try:
                    rng_np = np.random.default_rng(0xC0FFEE)
                    a_np = rng_np.random((n, n), dtype=np.float32).astype(np_dtype)
                    b_np = rng_np.random((n, n), dtype=np.float32).astype(np_dtype)
                    a = cupy.asarray(a_np)
                    b = cupy.asarray(b_np)
                    del a_np, b_np

                    if name == "mma_f16xf16_f32acc":
                        out_dtype = cupy.float16
                    elif name == "mma_bf16xbf16_f32acc":
                        out_dtype = ml_dtypes.bfloat16
                    else:
                        out_dtype = cupy.float32
                    out = cupy.zeros((n, n), dtype=out_dtype)

                    grid = (n // BM, n // BN)

                    # Warmup (drops JIT latency + first-launch cost).
                    for _ in range(WARMUP):
                        ct.launch(stream.ptr, grid, kernel, (a, b, out))
                    cupy.cuda.runtime.deviceSynchronize()

                    # Timed iters — tight block.
                    starts = [cupy.cuda.Event() for _ in range(ITERS)]
                    ends = [cupy.cuda.Event() for _ in range(ITERS)]
                    for i in range(ITERS):
                        starts[i].record(stream)
                        ct.launch(stream.ptr, grid, kernel, (a, b, out))
                        ends[i].record(stream)
                    stream.synchronize()

                    variant_rows = []
                    for i in range(ITERS):
                        gpu_ms = cupy.cuda.get_elapsed_time(starts[i], ends[i])
                        tflops = (total_flops / 1e12) / (gpu_ms / 1000.0)
                        print(f"[cutile-{name}] N={n} iter={i} "
                              f"gpu_ms={gpu_ms:.3f} tflops={tflops:.3f}")
                        writer.writerow(["cutile", name, n, i,
                                         f"{gpu_ms:.6f}", f"{tflops:.6f}"])
                        variant_rows.append((gpu_ms, tflops))
                    # median across ITERS
                    med_ms = sorted(r[0] for r in variant_rows)[ITERS // 2]
                    med_tf = sorted(r[1] for r in variant_rows)[ITERS // 2]
                    rows.append((n, name, med_ms, med_tf))

                    del a, b, out, starts, ends
                    cupy.get_default_memory_pool().free_all_blocks()
                except Exception as e:
                    print(f"[{name}] N={n} BENCH ERROR "
                          f"{type(e).__name__}: {str(e)[:200]}", file=sys.stderr)

    # Summary print.
    print()
    print("=" * 72)
    print(" BENCH SUMMARY — median of", ITERS, "iters")
    print("=" * 72)
    print(f"  {'N':>5}  {'variant':<28}  {'ms':>9}  {'TFLOPS':>8}")
    for n, name, ms, tf in rows:
        print(f"  {n:>5}  {name:<28}  {ms:>9.3f}  {tf:>8.3f}")
    print()
    # Reference numbers from Wave 12 for visual A/B.
    print("  Reference (RTX 5090, sm_120, N=4096):")
    print("    cuBLAS sgemm            ~  8.906 ms   73.59 TFLOPS")
    print("    nvcc shared-tiled f32   ~ 17.071 ms   38.41 TFLOPS")
    print("    cuTile f32 (W12.4)      ~ 86.07  ms    7.57 TFLOPS")
    print()


# ────────────────────────────────────────────────────────────────────
# Cubin export
# ────────────────────────────────────────────────────────────────────


def _ac(dt):
    return ArrayConstraint(
        dtype=dt, ndim=2, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=(), may_alias_internally=False,
        stride_constant=(None, 1), stride_divisible_by=1,
        shape_divisible_by=1, base_addr_divisible_by=1,
    )


def export_cubins(out_dir: str = ".") -> list[str]:
    """Compile each variant to a cubin on sm_120 and write to out_dir.
    Returns list of filenames written."""
    specs = [
        ("mma_f16xf16_f32acc.cubin", mma_f16xf16_f32acc,
         ct.float16, ct.float16),
        ("mma_bf16xbf16_f32acc.cubin", mma_bf16xbf16_f32acc,
         ct.bfloat16, ct.bfloat16),
        ("mma_tf32xtf32_f32acc.cubin", mma_tf32xtf32_f32acc,
         ct.float32, ct.float32),
        ("mma_f32xf32_f32acc.cubin", mma_f32xf32_f32acc,
         ct.float32, ct.float32),
    ]
    import os
    written = []
    for fname, kern, in_dt, out_dt in specs:
        path = os.path.join(out_dir, fname)
        sig = KernelSignature(
            parameters=[_ac(in_dt), _ac(in_dt), _ac(out_dt)],
            calling_convention=CallingConvention.cutile_python_v1(),
        )
        try:
            export_kernel(kern, [sig], path,
                          gpu_code="sm_120", output_format="cubin")
            size = os.path.getsize(path)
            print(f"  wrote {path}  ({size} bytes)")
            written.append(path)
        except Exception as e:
            print(f"  FAILED {fname}: {type(e).__name__}: {str(e)[:200]}",
                  file=sys.stderr)
    return written


# ────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", default=True,
                        help="correctness check only (default)")
    parser.add_argument("--bench", action="store_true", default=False,
                        help="run timed sweep")
    parser.add_argument("--export-cubins", action="store_true", default=False,
                        help="compile each variant to a .cubin on sm_120")
    parser.add_argument("--n", type=int, default=CORRECTNESS_N,
                        help="size for correctness check")
    parser.add_argument("--csv-out", default="results.csv")
    parser.add_argument("--cubin-dir", default=".")
    args = parser.parse_args()

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"device: {props['name'].decode()}")
    print(f"compute capability: sm_{props['major']}{props['minor']}")
    print(f"BM={BM} BN={BN} BK={BK}")
    print()

    if args.export_cubins:
        print("Exporting cubins…")
        export_cubins(args.cubin_dir)
        print()

    if args.bench:
        print(f"Running correctness first at N={args.n}…")
        run_correctness(args.n)
        print()
        print(f"Running bench sweep {BENCH_SIZES} ({WARMUP} warm + {ITERS} iters)…")
        run_bench(args.csv_out)
        return 0

    # smoke path (default)
    results = run_correctness(args.n)
    print()
    print("=" * 48)
    all_ok = all(r[0] for r in results.values())
    any_ok = any(r[0] for r in results.values())
    for k, (ok, rel, err) in results.items():
        label = "PASS" if ok else "FAIL"
        tail = f"  rel={rel:.3e}" if err == "" else f"  {err}"
        print(f"  {k}: {label}{tail}")
    print("=" * 48)
    if all_ok:
        print("SMOKE TEST OK")
        return 0
    elif any_ok:
        print("SMOKE TEST PARTIAL — some variant(s) failed")
        return 1
    else:
        print("SMOKE TEST FAILED — all variants failed")
        return 2


if __name__ == "__main__":
    sys.exit(main())
