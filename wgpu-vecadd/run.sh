#!/usr/bin/env bash
# Build + run the wgpu-vecadd bench, mirroring the wgpu-matmul + cuda/oxide-vecadd
# cells. Output is captured into run.log so we have an artifact even if a future
# run lands a different adapter.
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release 2>&1 | tee build.log
./target/release/wgpu-vecadd 2>&1 | tee run.log
