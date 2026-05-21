#!/usr/bin/env bash
# Wave C1.6 — wgpu-attn-mla driver script.
#
# Builds and runs the Rust+WGSL MLA attention port. Per AGENTS.md, always
# pipe through `tee run.log`.

set -e
set -o pipefail
cd "$(dirname "$0")"

echo "[run.sh] === build ==="
cargo build --release 2>&1 | tee build.log

echo "[run.sh] === run (correctness + bench) ==="
cargo run --release 2>&1 | tee run.log
