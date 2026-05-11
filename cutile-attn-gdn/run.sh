#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/codeseys/cuda-exploration/cutile-vecadd-bench/.venv/bin/python

echo "=== smoke test (correctness shape) ==="
$PY main.py --smoke

echo
echo "=== bench (Qwen3-Next decode shape) ==="
$PY main.py --bench --csv-out results.csv

echo
echo "=== cubin export for SASS inspection ==="
$PY main.py --export-cubin --cubin-out gdn_decode_fused.cubin

if [ -x /usr/local/cuda/bin/cuobjdump ]; then
  CUOBJ=/usr/local/cuda/bin/cuobjdump
elif command -v cuobjdump >/dev/null 2>&1; then
  CUOBJ=cuobjdump
else
  echo "cuobjdump not found; skipping SASS dump"
  CUOBJ=""
fi
if [ -n "$CUOBJ" ] && [ -f gdn_decode_fused.cubin ]; then
  $CUOBJ --dump-sass gdn_decode_fused.cubin > gdn_decode_fused.sass 2>&1 || true
  HMMA=$(grep -c "HMMA" gdn_decode_fused.sass || true)
  FFMA=$(grep -c "FFMA" gdn_decode_fused.sass || true)
  LDG=$(grep -c "LDG.E" gdn_decode_fused.sass || true)
  STG=$(grep -c "STG.E" gdn_decode_fused.sass || true)
  echo "SASS lines : $(wc -l < gdn_decode_fused.sass)"
  echo "HMMA insts : ${HMMA}"
  echo "FFMA insts : ${FFMA}"
  echo "LDG insts  : ${LDG}"
  echo "STG insts  : ${STG}"
fi
