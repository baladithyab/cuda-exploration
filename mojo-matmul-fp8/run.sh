#!/usr/bin/env bash
# Wave 22.4 -- mojo-matmul-fp8 reproduce script
# Hand-rolled e4m3 FP8 m16n8k32 tiled matmul via raw mma() on Blackwell sm_120.
set -e
cd "$(dirname "$0")"
HERE="$(pwd)"

export PATH="$HOME/.pixi/bin:$PATH"
WORKSPACE=/home/codeseys/cuda-exploration/mojo-workspace

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,temperature.gpu,power.draw --format=csv

echo ""
echo "=== mojo --version ==="
(cd "$WORKSPACE" && pixi run mojo --version)

echo ""
echo "=== Run mma_probe_fp8 (single-warp probe) ==="
(cd "$WORKSPACE" && pixi run mojo "$HERE/mma_probe_fp8.mojo") > "$HERE/mma_probe_fp8.sass" 2> "$HERE/mma_probe_fp8.stderr"
echo "QMMA.16832.F32.E4M3.E4M3 count in probe SASS:"
grep -cE "QMMA\.16832" "$HERE/mma_probe_fp8.sass" || true

echo ""
echo "=== Run matmul_fp8 (full M=N=K=32 tiled kernel) ==="
(cd "$WORKSPACE" && pixi run mojo "$HERE/matmul_fp8.mojo") > "$HERE/matmul_fp8.sass" 2> "$HERE/matmul_fp8.stderr"
echo "QMMA.16832.F32.E4M3.E4M3 count in matmul SASS:"
grep -cE "QMMA\.16832" "$HERE/matmul_fp8.sass" || true
echo ""
echo "Correctness lines:"
grep -E "max_abs_err|PASSED|FAIL" "$HERE/matmul_fp8.sass" || true
