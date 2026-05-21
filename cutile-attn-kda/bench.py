"""Wave 17.W1e + Wave 22.7 — KDA decode bench harness.

W1e: original wraps `main.run_bench` at SHAPE_KIMI_LINEAR_DECODE.
W22.7: --shape selects from SHAPE_REGISTRY for the larger-shape sweep.

Usage:
    python bench.py [--shape NAME] [--csv-out FILE]
"""
from __future__ import annotations

import argparse
import sys

import cupy
import cuda.tile as ct

from main import SHAPE_KIMI_LINEAR_DECODE, SHAPE_REGISTRY, run_bench


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shape",
        default=SHAPE_KIMI_LINEAR_DECODE.name,
        choices=sorted(SHAPE_REGISTRY.keys()),
        help="Bench shape (default: kimi_linear_decode)",
    )
    ap.add_argument("--csv-out", default="results.csv")
    args = ap.parse_args()

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"device: {props['name'].decode()}")
    print(f"compute capability: sm_{props['major']}{props['minor']}")
    print()

    shape = SHAPE_REGISTRY[args.shape]
    run_bench(shape, args.csv_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
