#!/bin/bash
# Wave 17 W1c — cuda-attn-gdn build + correctness driver.
# Per task: COMPILE + CORRECTNESS only (no timed bench in W1c authoring).
# Orchestrator runs `./bench` separately on idle GPU.
set -euo pipefail
cd "$(dirname "$0")"

NVCC=/usr/local/cuda/bin/nvcc
CXX=clang-14
CUOBJ=/usr/local/cuda/bin/cuobjdump

echo "=== build attn_gdn (correctness binary) ==="
make clean >/dev/null
make attn_gdn 2>&1 | tee build.log

echo
echo "=== build bench (smoke-only at this stage) ==="
make bench 2>&1 | tee -a build.log

echo
echo "=== correctness run ==="
./attn_gdn 2>&1 | tee run.log

echo
echo "=== SASS dump (HMMA/FFMA/LDG.E.128 sanity per ADR-0004) ==="
$CUOBJ --dump-sass attn_gdn > attn_gdn.sass 2>&1 || true
HMMA=$(grep -c "HMMA" attn_gdn.sass || true)
FFMA=$(grep -c "FFMA" attn_gdn.sass || true)
LDGE=$(grep -c "LDG.E"  attn_gdn.sass || true)
LDG128=$(grep -c "LDG.E.128" attn_gdn.sass || true)
LDG64=$(grep -c "LDG.E.64"  attn_gdn.sass || true)
STG128=$(grep -c "STG.E.128" attn_gdn.sass || true)
MUFU=$(grep -c "MUFU" attn_gdn.sass || true)
TOTAL=$(wc -l < attn_gdn.sass)
echo "SASS lines : $TOTAL"
echo "HMMA       : $HMMA      (expect 0; GDN is memory-bound, no TC)"
echo "FFMA       : $FFMA      (expect > 0)"
echo "LDG.E      : $LDGE      (any width)"
echo "LDG.E.128  : $LDG128    (expect > 0; the 'win over cuTile' point)"
echo "LDG.E.64   : $LDG64"
echo "STG.E.128  : $STG128    (expect > 0; vectorized state writes)"
echo "MUFU       : $MUFU      (expect 0; GDN has no exp/softmax)"
