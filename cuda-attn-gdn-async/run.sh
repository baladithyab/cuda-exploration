#!/bin/bash
# Wave 22.9 — cuda-attn-gdn-async build + correctness driver.
# Per task: COMPILE + CORRECTNESS only (no timed bench in W22.9 authoring).
# Orchestrator runs `./bench` separately on idle GPU.
set -euo pipefail
cd "$(dirname "$0")"

NVCC=/usr/local/cuda/bin/nvcc
CXX=clang-14
CUOBJ=/usr/local/cuda/bin/cuobjdump

echo "=== build attn_gdn_async (correctness binary) ==="
make clean >/dev/null
make attn_gdn_async 2>&1 | tee build.log

echo
echo "=== build bench (smoke-only at this stage) ==="
make bench 2>&1 | tee -a build.log

echo
echo "=== correctness run ==="
./attn_gdn_async 2>&1 | tee run.log

echo
echo "=== SASS dump (cp.async / mbarrier / instruction mix per ADR-0004) ==="
$CUOBJ --dump-sass attn_gdn_async > attn_gdn_async.sass 2>&1 || true
HMMA=$(grep -c "HMMA"      attn_gdn_async.sass || true)
FFMA=$(grep -c "FFMA"      attn_gdn_async.sass || true)
LDGE=$(grep -c "LDG.E"     attn_gdn_async.sass || true)
LDG128=$(grep -c "LDG.E.128" attn_gdn_async.sass || true)
LDG64=$(grep -c "LDG.E.64"  attn_gdn_async.sass || true)
STG128=$(grep -c "STG.E.128" attn_gdn_async.sass || true)
MUFU=$(grep -c "MUFU"      attn_gdn_async.sass || true)
LDGSTS=$(grep -c "LDGSTS"  attn_gdn_async.sass || true)
MBAR=$(grep -cE "ARRIVES|BAR.ARV|MBAR" attn_gdn_async.sass || true)
SYNCS=$(grep -c "SYNCS"    attn_gdn_async.sass || true)
TOTAL=$(wc -l < attn_gdn_async.sass)
echo "SASS lines : $TOTAL"
echo "HMMA       : $HMMA      (expect 0; GDN is memory-bound, no TC)"
echo "FFMA       : $FFMA      (expect > 0)"
echo "LDG.E      : $LDGE      (any width — likely DOWN vs W1c if cp.async takes over)"
echo "LDG.E.128  : $LDG128    (W1c had 16; expect lower if memcpy_async lowers to LDGSTS)"
echo "LDG.E.64   : $LDG64"
echo "STG.E.128  : $STG128    (state writes — should remain present)"
echo "MUFU       : $MUFU      (expect 0; GDN has no exp/softmax)"
echo "LDGSTS     : $LDGSTS    (cp.async — *the* W22.9 acceptance signal; expect > 0)"
echo "MBAR.*     : $MBAR      (mbarrier.* — async-barrier infrastructure)"
echo "SYNCS      : $SYNCS     (Blackwell async-tx barriers — cuTile parity signal)"
