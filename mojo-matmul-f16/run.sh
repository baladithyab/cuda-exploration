#!/usr/bin/env bash
# Wave 22.2 -- mojo-matmul-f16 reproduce script
# Hand-rolled f16-in/f32-acc tiled matmul (parity scaffold to bf16, swaps dtype).
# Authoring + correctness only at M=N=K=64; orchestrator runs the full bench serially.
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
echo "=== Run mojo-matmul-f16 ==="
(cd "$WORKSPACE" && pixi run mojo "$HERE/matmul_f16.mojo")
