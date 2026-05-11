"""Wave 12 W12.1 — cuTile vecadd bench, apples-to-apples with oxide-vecadd-bench.

Mirrors oxide-vecadd-bench/src/main.rs:
  - Same N sweep: 1M / 16M / 64M / 256M f32 elements
  - Same iter counts: 1 warmup, 10 timed
  - Same algorithm: c[i] = a[i] + b[i]  (3 buffers, 12 bytes/elem traffic)
  - cudaEvent timing → gpu_ms + GB/s
  - Block size matches oxide-vecadd-bench's `LaunchConfig::for_num_elems(n)` which
    uses block 256.

Two cuTile variants:
  - tile256:   TILE_SIZE=256 — matches the cuda-oxide block-256 launch shape
  - tile1024:  TILE_SIZE=1024 — bigger tile to amortize launch overhead

cuTile's "tile" is roughly the unit of work per CTA. With TILE_SIZE=N elements, each
block does N adds, which on Blackwell vectorizes to ~N/4 ld.global.v4 + ld.global.v4 +
st.global.v4 instructions. Bigger tiles → fewer launches → less launch overhead, but
also more pressure on per-thread register count and may hurt occupancy.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time

import cuda.tile as ct
import cupy
import numpy as np

SIZES = [1 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024]
WARMUP = 1
ITERS = 10

# ────────────────────────────────────────────────────────────────────
# Two kernel definitions — one per TILE_SIZE so the constant is folded
# into the compiled tile shape.
# ────────────────────────────────────────────────────────────────────


def make_kernel(tile_size: int):
    @ct.kernel
    def vecadd(a, b, out):
        bid = ct.bid(0)
        a_t = ct.load(a, index=(bid,), shape=(tile_size,))
        b_t = ct.load(b, index=(bid,), shape=(tile_size,))
        ct.store(out, index=(bid,), tile=a_t + b_t)

    return vecadd


# ────────────────────────────────────────────────────────────────────
# Bench driver
# ────────────────────────────────────────────────────────────────────


def run_kernel_sweep(name: str, tile_size: int, kernel, csv_writer) -> None:
    stream = cupy.cuda.get_current_stream()

    for n in SIZES:
        if n % tile_size != 0:
            print(f"[cutile-{name}] N={n} not divisible by tile_size={tile_size}, skipping")
            continue
        grid = (n // tile_size,)

        # Allocate
        rng = cupy.random.default_rng(0xC0FFEE)
        a = rng.random(n, dtype=cupy.float32)
        b = rng.random(n, dtype=cupy.float32)
        out = cupy.empty(n, dtype=cupy.float32)

        # Warmup (also triggers JIT compile of this (kernel, dtype, shape) instance)
        for _ in range(WARMUP):
            ct.launch(stream.ptr, grid, kernel, (a, b, out))
        cupy.cuda.runtime.deviceSynchronize()

        # Correctness check on first iter
        expected = a + b
        if not bool(cupy.allclose(out, expected, atol=1e-6, rtol=1e-6)):
            max_err = float(cupy.max(cupy.abs(out - expected)))
            print(f"[cutile-{name}] N={n} CORRECTNESS FAIL max_err={max_err:.2e}", file=sys.stderr)
            sys.exit(1)

        # Timed iterations with cudaEvent
        traffic_bytes = 3.0 * n * 4.0
        starts = [cupy.cuda.Event() for _ in range(ITERS)]
        ends = [cupy.cuda.Event() for _ in range(ITERS)]
        wall_times_ms: list[float] = []
        for i in range(ITERS):
            t0 = time.perf_counter()
            starts[i].record(stream)
            ct.launch(stream.ptr, grid, kernel, (a, b, out))
            ends[i].record(stream)
            stream.synchronize()
            wall_times_ms.append((time.perf_counter() - t0) * 1000.0)

        gpu_times_ms = [
            cupy.cuda.get_elapsed_time(starts[i], ends[i]) for i in range(ITERS)
        ]
        for i, (gpu_ms, wall_ms) in enumerate(zip(gpu_times_ms, wall_times_ms)):
            gbps = (traffic_bytes / 1e9) / (gpu_ms / 1000.0)
            print(f"[cutile-{name}] N={n} iter={i} gpu_ms={gpu_ms:.4f} GB/s={gbps:.2f}")
            csv_writer.writerow(["cutile", name, n, i, f"{gpu_ms:.6f}", f"{gbps:.6f}"])

        gpu_times_ms_sorted = sorted(gpu_times_ms)
        best = gpu_times_ms_sorted[0]
        median = statistics.median(gpu_times_ms_sorted)
        gbps_med = (traffic_bytes / 1e9) / (median / 1000.0)
        print(
            f"[cutile-{name}] N={n} correctness OK  best={best:.4f}ms med={median:.4f}ms "
            f"({gbps_med:.2f} GB/s median)"
        )

        # Free for next size to keep memory usage bounded
        del a, b, out, expected
        cupy.get_default_memory_pool().free_all_blocks()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-out",
        default="results.csv",
        help="path to write per-iter CSV results",
    )
    args = parser.parse_args()

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    print(f"device: {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    cc_major = cupy.cuda.runtime.getDeviceProperties(0)["major"]
    cc_minor = cupy.cuda.runtime.getDeviceProperties(0)["minor"]
    print(f"compute capability: sm_{cc_major}{cc_minor}")
    print(f"sizes: {SIZES}")
    print(f"warmup: {WARMUP}, iters: {ITERS}")
    print()

    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["impl", "kernel", "n", "iter", "gpu_ms", "gbps"])

        kernel_256 = make_kernel(256)
        run_kernel_sweep("tile256", 256, kernel_256, writer)
        print()
        kernel_1024 = make_kernel(1024)
        run_kernel_sweep("tile1024", 1024, kernel_1024, writer)

    print(f"\nResults written to {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
