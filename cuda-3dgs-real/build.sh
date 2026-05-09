#!/bin/bash
set -e
export CUDA=/usr/local/cuda
cd "$(dirname "$0")"
$CUDA/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lstdc++ -lm \
    -o rasterize rasterize.cu
echo "Build OK"
