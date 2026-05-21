#!/usr/bin/env bash
# Wave 17 W1b — build + run helper for oxide-attn-mla.
#
# Per cuda-oxide skill: must export CUDA_HOME + LIBNVVM_PATH + LLVM 21 on PATH
# before any cargo oxide command, otherwise the libNVVM shadow bug triggers
# and either fails to target sm_120 or produces silently-bad codegen.
#
# Build via `cargo oxide build --arch sm_120` (NOT `cargo build` — that gives a
# host-only binary with no PTX). Run via `cargo oxide run`.

set -euo pipefail

export CUDA_HOME=/usr/local/cuda
export LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so
export PATH=/usr/lib/llvm-21/bin:$PATH:$HOME/.cargo/bin

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

mode="${1:-run}"

case "$mode" in
  build)
    echo "[oxide-attn-mla] cargo oxide build --arch sm_120"
    cargo oxide build --arch sm_120 2>&1 | tee build.log
    ;;
  run)
    echo "[oxide-attn-mla] cargo oxide run --arch sm_120 (correctness + warm bench)"
    cargo oxide run --arch sm_120 2>&1 | tee run.log
    ;;
  sass)
    # Disassemble cubin via cuobjdump (USE THE CUDA-13.2 ABSOLUTE PATH —
    # /usr/bin/cuobjdump silently produces empty SASS on sm_120 cubins).
    cubin="oxide_attn_mla.cubin"
    if [[ ! -f "$cubin" ]]; then
      echo "build first: $cubin not found; run './run.sh build'" >&2
      exit 1
    fi
    /usr/local/cuda/bin/cuobjdump --dump-sass "$cubin" > oxide_attn_mla.sass
    echo "[oxide-attn-mla] SASS written to oxide_attn_mla.sass"
    echo "  HMMA = $(grep -c '\bHMMA\.' oxide_attn_mla.sass || true)"
    echo "  FFMA = $(grep -c '\bFFMA\b'  oxide_attn_mla.sass || true)"
    echo "  MUFU = $(grep -c '\bMUFU\.'  oxide_attn_mla.sass || true)"
    ;;
  *)
    echo "usage: $0 {build|run|sass}" >&2
    exit 2
    ;;
esac
