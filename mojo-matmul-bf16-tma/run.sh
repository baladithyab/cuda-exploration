#!/usr/bin/env bash
# Wave 22.1 -- mojo-matmul-bf16-tma reproduce script
# TMA-loaded variant of Wave 21's bf16-in/f32-acc tiled matmul. Goal: trigger UTMALDG.
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
echo "=== Run mojo-matmul-bf16-tma (correctness @ M=64) ==="
(cd "$WORKSPACE" && pixi run mojo "$HERE/matmul_bf16_tma.mojo")
