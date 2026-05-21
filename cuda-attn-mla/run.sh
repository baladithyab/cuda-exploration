#!/usr/bin/env bash
# Wave 17 W1a — cuda-attn-mla driver script.
#
# Runs the build, correctness, SASS dump, and (optionally) timed bench.
# Usage:
#   ./run.sh             — build + correctness + SASS sanity (default; what subagent runs)
#   ./run.sh bench       — additionally run timed bench iters (orchestrator-serial-only)
#
# Per AGENTS.md run-discipline, ALWAYS pipe through `tee run.log` so we have
# a captured artifact even if the binary crashes mid-run.

set -e
set -o pipefail

cd "$(dirname "$0")"

echo "[run.sh] === build ==="
make 2>&1 | tee build.log

echo "[run.sh] === correctness + bench setup banner ==="
if [ "${1:-}" = "bench" ]; then
    ./attn_mla --bench-now 2>&1 | tee run.log
else
    ./attn_mla 2>&1 | tee run.log
fi

echo "[run.sh] === SASS dump (HMMA / FFMA / LDG sanity) ==="
/usr/local/cuda/bin/cuobjdump --dump-sass attn_mla > attn_mla.sass

echo "[run.sh] HMMA count: $(grep -c HMMA attn_mla.sass)"
echo "[run.sh] FFMA count: $(grep -c FFMA attn_mla.sass)"
echo "[run.sh] MUFU count: $(grep -c MUFU attn_mla.sass)"
echo "[run.sh] LDG.E count: $(grep -c 'LDG.E' attn_mla.sass)"
echo "[run.sh] LDG.E.128 count: $(grep -c 'LDG.E.128' attn_mla.sass)"
