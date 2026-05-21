#!/usr/bin/env bash
# Wave C1.4 -- mojo-matmul-tiled reproduce script
# Classical FFMA-tiled matmul in Mojo (rounds out matmul-tiled cross-frontend column).
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
echo "=== Run mojo-matmul-tiled ==="
(cd "$WORKSPACE" && pixi run mojo "$HERE/matmul_tiled.mojo")
