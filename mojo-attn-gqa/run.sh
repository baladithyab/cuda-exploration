#!/usr/bin/env bash
# Wave C1.5 -- mojo-attn-gqa reproduce script (GQA llama3_8b shape, timed bench).
# 3-kernel GQA attention (Q@K^T + softmax + P@V) with bf16 matmul stages.
# Builds on Wave 21 mojo-matmul-bf16 + Wave 22.5b mojo-attn-bf16 patterns.
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
echo "=== Run mojo-attn-gqa (llama3_8b shape, timed bench + 1024-sample correctness) ==="
(cd "$WORKSPACE" && pixi run mojo "$HERE/attn_gqa.mojo")
