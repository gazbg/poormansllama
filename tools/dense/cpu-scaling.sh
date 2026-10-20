#!/usr/bin/env bash
# Where does the decode headroom sit on the host side?  CPU-only llama-bench of the whole dense model (no GPU involved):
#   A: threads 4 / 8 / 4 pinned one per core, batch 1 2 4 8  (does a host layer scale with threads, with the batch size?)
#   B: the same 1-token pass while a host->GPU copy loop runs on GPU 0 (does the ring's copy traffic slow the CPU layers?)
# Log: build/dense/cpu-scaling.log
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
MODEL=~/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf
BENCH=build/llama-cuda/bin/llama-bench
nvcc -O2 -arch=sm_61 -o build/dense/h2d-loop tools/dense/h2d-loop.cu || exit 1
run() { CUDA_VISIBLE_DEVICES="" $BENCH -m $MODEL -ngl 0 -n 0 -r 3 "$@" 2>/dev/null | grep -E "^\|" | grep -v -- "---" | sed 's/  */ /g'; }
echo "== A1: -t 4, batch 1 2 4 8";            run -t 4 -p 1,2,4,8
echo "== A2: -t 8, batch 1 2 4 8";            run -t 8 -p 1,2,4,8
echo "== A3: -t 4 pinned to cores 0-3";       run -t 4 -C 0xF --cpu-strict 1 -p 1,2,4,8
echo "== A4: -t 3, batch 1";                  run -t 3 -p 1
echo "== B0: -t 4, batch 1, no copy load";    run -t 4 -p 1
echo "== B1: -t 4, batch 1 WITH copy load (h2d-loop on GPU 0 in the background)"
CUDA_VISIBLE_DEVICES=0 build/dense/h2d-loop 120 32 > build/dense/h2d-loop.out 2>&1 &
LOOP=$!
sleep 3
run -t 4 -p 1
wait $LOOP; cat build/dense/h2d-loop.out
echo "CPU SCALING DONE"
