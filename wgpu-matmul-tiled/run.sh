#!/usr/bin/env bash
# wgpu-matmul-tiled: 4096x4096 f32 SGEMM with classical 16x16 shared-memory tiling
# in WGSL. 1 warmup + 50 timed iterations. Output captured for analysis.
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release 2>&1 | tee build.log
./target/release/wgpu-matmul-tiled 2>&1 | tee run.log
