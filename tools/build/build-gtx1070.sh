#!/usr/bin/env bash
# Build this repo's CUDA targets for a GTX 1070 (Pascal, compute capability 6.1) on Linux (verified: Ubuntu, GCC 11/12, CUDA 12.8).
#   tools/build/build-gtx1070.sh                  # llama-server llama-completion llama-bench
#   tools/build/build-gtx1070.sh llama-server     # other targets, as arguments
# Build directory: build/llama-cuda (the path the tools/dense scripts read). BUILD_DIR=... overrides it.
# CMAKE_CUDA_ARCHITECTURES must stay 61: CMake's default list targets Blackwell, which this card cannot run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BUILD_DIR="${BUILD_DIR:-build/llama-cuda}"
ARCH=61
if [ "$#" -gt 0 ]; then TARGETS=("$@"); else TARGETS=(llama-server llama-completion llama-bench); fi

command -v cmake > /dev/null || { echo "build-gtx1070: cmake not found" >&2; exit 1; }
command -v nvcc > /dev/null || { echo "build-gtx1070: nvcc not found; add CUDA's bin directory (e.g. /usr/local/cuda/bin) to PATH" >&2; exit 1; }

NVCC_REL=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
echo "build-gtx1070: nvcc $NVCC_REL, $(cmake --version | head -1)"
case "$NVCC_REL" in
  12.*) ;;
  *) echo "build-gtx1070: warning: CUDA 12.x expected (verified on 12.8), found $NVCC_REL" >&2 ;;
esac

if command -v nvidia-smi > /dev/null; then
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2> /dev/null || true)
  if [ -n "$CAP" ] && [ "$CAP" != "6.1" ]; then
    echo "build-gtx1070: warning: GPU 0 reports compute capability $CAP; this script builds for sm_$ARCH (6.1)" >&2
  fi
fi

cmake -S third_party/llama.cpp -B "$BUILD_DIR" \
  -DGGML_CUDA=ON \
  -DLLAMA_LAYER_STREAMING=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release --parallel "$(nproc)" --target "${TARGETS[@]}"

for t in "${TARGETS[@]}"; do
  exe="$BUILD_DIR/bin/$t"
  if [ -x "$exe" ]; then
    echo "built $exe"
  else
    echo "build-gtx1070: $t was not produced at $exe" >&2
    exit 1
  fi
done
