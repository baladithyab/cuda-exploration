"""Wave 17.W1e — KDA decode bench harness (separate entrypoint per W1e plan-row).

Wraps `main.run_bench` at SHAPE_KIMI_LINEAR_DECODE (B=1, H=32, d_k=d_v=128).
Per the wave-17 plan, the orchestrator runs this; this file exists so the
correctness harness and bench harness are independent CLI entry-points.

Usage:
    python bench.py [--csv-out FILE]
"""
from __future__ import annotations

import argparse
import sys

import cupy
import cuda.tile as ct

from main import SHAPE_KIMI_LINEAR_DECODE, run_bench


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-out", default="results.csv")
    args = ap.parse_args()

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"device: {props['name'].decode()}")
    print(f"compute capability: sm_{props['major']}{props['minor']}")
    print()

    run_bench(SHAPE_KIMI_LINEAR_DECODE, args.csv_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
