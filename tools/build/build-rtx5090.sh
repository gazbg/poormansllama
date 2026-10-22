#!/usr/bin/env bash
# Build this repo's CUDA targets for an RTX 5090 (Blackwell, compute capability 12.0) on Windows, run from Git Bash.
#   tools/build/build-rtx5090.sh                  # llama-server llama-completion llama-bench llama-cli
#   tools/build/build-rtx5090.sh llama-server     # other targets, as arguments
# Build directory: build/llama-cuda (the path the tools/dense and tools/*.ps1 scripts read). BUILD_DIR=... overrides it.
# CMake's default Visual Studio generator is used; set CMAKE_GENERATOR to choose another, e.g. "Visual Studio 17 2022".
# CUDA_PATH is set here when the shell has none: the MSBuild CUDA step reads it at BUILD time, so a shell without it
# fails in the build step even though configure succeeded (docs/AGENT_ENVIRONMENT.md, "CUDA_PATH").
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BUILD_DIR="${BUILD_DIR:-build/llama-cuda}"
ARCH=120
DEFAULT_CUDA="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"
if [ "$#" -gt 0 ]; then TARGETS=("$@"); else TARGETS=(llama-server llama-completion llama-bench llama-cli); fi

if [ -z "${CUDA_PATH:-}" ]; then export CUDA_PATH="$DEFAULT_CUDA"; fi
if [ -z "${CUDA_PATH_V13_2:-}" ]; then export CUDA_PATH_V13_2="$CUDA_PATH"; fi

NVCC="$CUDA_PATH/bin/nvcc.exe"
if [ ! -f "$NVCC" ]; then
  echo "build-rtx5090: nvcc not found at $NVCC; set CUDA_PATH to the CUDA 13.2 directory" >&2
  exit 1
fi
command -v cmake > /dev/null || { echo "build-rtx5090: cmake not found on PATH" >&2; exit 1; }

NVCC_REL=$("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
echo "build-rtx5090: nvcc $NVCC_REL from $CUDA_PATH, $(cmake --version | head -1)"
case "$NVCC_REL" in
  12.8|12.9|13.*) ;;
  *) echo "build-rtx5090: warning: CUDA 13.2 expected (sm_120 needs 12.8 or later), found $NVCC_REL" >&2 ;;
esac

if command -v nvidia-smi > /dev/null; then
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2> /dev/null || true)
  if [ -n "$CAP" ] && [ "$CAP" != "12.0" ]; then
    echo "build-rtx5090: warning: GPU 0 reports compute capability $CAP; this script builds for sm_$ARCH (12.0)" >&2
  fi
fi

# Opt-in AVX-512 VNNI for the x86 CPU kernels that run the host-resident layers (Q8/Q6 dot products, CPU repack).
# OFF by default, which is what the build has always produced. The MSVC native detection (ggml-cpu/cmake/FindSIMD.cmake)
# never sets it, so it must be asked for: GGML_AVX512_VNNI=1 tools/build/build-rtx5090.sh. The value is always passed,
# so switching back to OFF in an existing build directory works too.
VNNI="${GGML_AVX512_VNNI:-0}"
case "$VNNI" in
  0) VNNI=OFF ;;
  1) VNNI=ON ;;
  *) echo "build-rtx5090: GGML_AVX512_VNNI must be 0 or 1, got '$VNNI'" >&2; exit 1 ;;
esac

# Opt-in CUDA flash attention for EVERY K/V cache type pair. OFF by default: then only a K and a V cache of the SAME type
# (f16, q4_0, q8_0, bf16) have a GPU kernel, and any other pair (for example -ctk q8_0 -ctv q4_0, or -ctvd q4_0 with the
# default f16 draft K) runs its attention on the CPU: the GPU idles, prompt and decode crawl. ON compiles many more kernel
# instances (a much longer build, a bigger DLL): GGML_CUDA_FA_ALL_QUANTS=1 BUILD_DIR=build/llama-cuda-faq tools/build/build-rtx5090.sh
FAQ="${GGML_CUDA_FA_ALL_QUANTS:-0}"
case "$FAQ" in
  0) FAQ=OFF ;;
  1) FAQ=ON ;;
  *) echo "build-rtx5090: GGML_CUDA_FA_ALL_QUANTS must be 0 or 1, got '$FAQ'" >&2; exit 1 ;;
esac

cmake -S third_party/llama.cpp -B "$BUILD_DIR" \
  -DGGML_CUDA=ON \
  -DLLAMA_LAYER_STREAMING=ON \
  -DGGML_AVX512_VNNI="$VNNI" \
  -DGGML_CUDA_FA_ALL_QUANTS="$FAQ" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release --parallel "${NUMBER_OF_PROCESSORS:-8}" --target "${TARGETS[@]}"

for t in "${TARGETS[@]}"; do
  exe="$BUILD_DIR/bin/Release/$t.exe"
  if [ -f "$exe" ]; then
    echo "built $exe"
  else
    echo "build-rtx5090: $t.exe was not produced at $exe" >&2
    exit 1
  fi
done
