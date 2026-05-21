#!/usr/bin/env bash
# Wave C1.2 wgpu-reduction: 6th frontend port of the reduction kernel.
# Reproducible: from repo root, `bash wgpu-reduction/run.sh | tee wgpu-reduction/run.log`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/wgpu-reduction"

echo "=== nvidia-smi (informational; on WSL wgpu only sees Vulkan llvmpipe) ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,temperature.gpu,power.draw \
    --format=csv 2>&1 || echo "(nvidia-smi unavailable)"

echo ""
echo "=== rustc / cargo versions ==="
rustc --version
cargo --version

echo ""
echo "=== cargo run --release ==="
cargo run --release
