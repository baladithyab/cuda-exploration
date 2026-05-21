#!/usr/bin/env bash
# Wave 19 Phase C1 -- mojo-matmul reproduce script
set -e
cd "$(dirname "$0")"

export PATH="$HOME/.pixi/bin:$PATH"
WORKSPACE=/home/codeseys/cuda-exploration/mojo-workspace

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,temperature.gpu,power.draw --format=csv

echo ""
echo "=== mojo --version ==="
(cd "$WORKSPACE" && pixi run mojo --version)

echo ""
echo "=== Run mojo-matmul ==="
(cd "$WORKSPACE" && pixi run mojo "$(pwd)/matmul.mojo")
