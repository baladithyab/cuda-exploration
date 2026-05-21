#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/codeseys/cuda-exploration/cutile-vecadd-bench/.venv/bin/python

echo "=== smoke (cam A, correctness vs cuda-3dgs-real) ==="
$PY rasterize.py --smoke 2>&1 | tee run.log
