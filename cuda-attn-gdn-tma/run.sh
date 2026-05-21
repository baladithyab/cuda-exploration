#!/bin/bash
# Wave 22.10 — cuda-attn-gdn-tma build + correctness driver.
# Per task: COMPILE + CORRECTNESS only (no timed bench in author cell).
# Orchestrator runs `./bench` separately on idle GPU.
set -euo pipefail
cd "$(dirname "$0")"

NVCC=/usr/local/cuda/bin/nvcc
CXX=clang-14
CUOBJ=/usr/local/cuda/bin/cuobjdump

echo "=== build attn_gdn_tma (correctness binary) ==="
make clean >/dev/null
make attn_gdn_tma 2>&1 | tee build.log

echo
echo "=== build bench (smoke-only at this stage) ==="
make bench 2>&1 | tee -a build.log

echo
echo "=== correctness run ==="
./attn_gdn_tma 2>&1 | tee run.log

echo
echo "=== SASS dump (W22.10 target signal: UTMALDG > 0) ==="
$CUOBJ --dump-sass attn_gdn_tma > attn_gdn_tma.sass 2>&1 || true
HMMA=$(grep -c "HMMA" attn_gdn_tma.sass || true)
FFMA=$(grep -c "FFMA" attn_gdn_tma.sass || true)
LDGE=$(grep -c "LDG.E"  attn_gdn_tma.sass || true)
LDG128=$(grep -c "LDG.E.128" attn_gdn_tma.sass || true)
LDG64=$(grep -c "LDG.E.64"  attn_gdn_tma.sass || true)
STG128=$(grep -c "STG.E.128" attn_gdn_tma.sass || true)
LDS128=$(grep -c "LDS.128" attn_gdn_tma.sass || true)
STS128=$(grep -c "STS.128" attn_gdn_tma.sass || true)
MUFU=$(grep -c "MUFU" attn_gdn_tma.sass || true)
UTMALDG=$(grep -c "UTMALDG" attn_gdn_tma.sass || true)
UTMASTG=$(grep -c "UTMASTG" attn_gdn_tma.sass || true)
LDGSTS=$(grep -c "LDGSTS" attn_gdn_tma.sass || true)
ARRIVES=$(grep -cE "ARRIVES|MBAR" attn_gdn_tma.sass || true)
SYNCS=$(grep -c "SYNCS" attn_gdn_tma.sass || true)
TOTAL=$(wc -l < attn_gdn_tma.sass)
echo "SASS lines : $TOTAL"
echo "HMMA       : $HMMA"
echo "FFMA       : $FFMA"
echo "LDG.E      : $LDGE"
echo "LDG.E.128  : $LDG128    (W1c had 16; expect ~0 here)"
echo "LDG.E.64   : $LDG64"
echo "STG.E.128  : $STG128"
echo "LDS.128    : $LDS128"
echo "STS.128    : $STS128"
echo "MUFU       : $MUFU"
echo "UTMALDG    : $UTMALDG    (W22.10 target signal: W1c had 0; expect > 0)"
echo "UTMASTG    : $UTMASTG"
echo "LDGSTS     : $LDGSTS    (cp.async legacy — should be 0; we use bulk.tensor)"
echo "ARRIVES    : $ARRIVES    (mbarrier ops)"
echo "SYNCS      : $SYNCS"
