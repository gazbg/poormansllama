#!/bin/bash
# MP-2b A/B: flint, fixture agentic-A, sequential arms under the GPU mutex (each arm takes it inside with-gpu-lock)
cd /c/Development/AI/poormansllama
LOG=build/llama-cuda/mp-queue5.log
run() { tag=$1; mkdir -p build/llama-cuda/$tag; rm -f build/llama-cuda/verdicts/$tag.*;
  pwsh -NoProfile -File tools/with-gpu-lock.ps1 -AllowUnprovenanced -VerdictTag $tag -TimeoutSeconds 3400 -Command "python tools/server_mt.py run --config tools/server-configs/$tag.json --out build/llama-cuda/$tag --fixture tools/server-fixtures/agentic-A.json --fixture2 tools/server-fixtures/agentic-B.json --scenarios agentic --reprocess-allow 2112 --think 3" > build/llama-cuda/$tag/wrapper.txt 2>&1
  echo "done $tag $(date +%H:%M:%S)" >> $LOG; }
run mp210-r1-default-rep
run mp211-r1-free
run mp212-r2-default
run mp213-r2-free
run mp214-r2-free-async
echo "queue5 finished" >> $LOG
