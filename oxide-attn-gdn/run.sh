#!/bin/bash
# Wave 17 W1d — build + run cuda-oxide GDN decode kernel.
# Compile + correctness only; NO timed benchmarks per task spec.

set -euo pipefail
cd "$(dirname "$0")"

export CUDA_HOME=/usr/local/cuda
export LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so
export PATH=/usr/lib/llvm-21/bin:$PATH:$HOME/.cargo/bin

echo "=== build ==="
cargo oxide build --arch sm_120 2>&1 | tee build.log

echo
echo "=== run ==="
cargo oxide run --arch sm_120 2>&1 | tee run.log

echo
echo "=== SASS analysis ==="
if [ -x /usr/local/cuda/bin/cuobjdump ]; then
  CUOBJ=/usr/local/cuda/bin/cuobjdump
else
  echo "ERROR: /usr/local/cuda/bin/cuobjdump not found; cannot analyze SASS"
  exit 1
fi
# cuda-oxide emits cubin next to the crate output as <crate>.cubin if present;
# otherwise it lives under target/. Find it.
CUBIN=""
for c in oxide_attn_gdn.cubin target/oxide_attn_gdn.cubin \
         target/sm_120/release/deps/oxide_attn_gdn.cubin; do
  if [ -f "$c" ]; then CUBIN="$c"; break; fi
done
if [ -z "$CUBIN" ]; then
  CUBIN=$(find . -name 'oxide_attn_gdn*.cubin' 2>/dev/null | head -n1 || true)
fi
if [ -z "$CUBIN" ] || [ ! -f "$CUBIN" ]; then
  echo "WARN: no cubin found; SASS analysis skipped"
else
  echo "cubin: $CUBIN"
  $CUOBJ --dump-sass "$CUBIN" > oxide_attn_gdn.sass 2>&1 || true
  HMMA=$(grep -c "HMMA" oxide_attn_gdn.sass || true)
  FFMA=$(grep -c "FFMA" oxide_attn_gdn.sass || true)
  FMUL=$(grep -c "FMUL" oxide_attn_gdn.sass || true)
  FADD=$(grep -c "FADD" oxide_attn_gdn.sass || true)
  LDG=$(grep -c "LDG.E" oxide_attn_gdn.sass || true)
  STG=$(grep -c "STG.E" oxide_attn_gdn.sass || true)
  echo "SASS lines : $(wc -l < oxide_attn_gdn.sass)"
  echo "HMMA insts : ${HMMA}"
  echo "FFMA insts : ${FFMA}"
  echo "FMUL insts : ${FMUL}"
  echo "FADD insts : ${FADD}"
  echo "LDG insts  : ${LDG}"
  echo "STG insts  : ${STG}"
fi
