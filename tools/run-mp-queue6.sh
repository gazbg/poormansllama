#!/bin/bash
# MP-1-SWA: flint, -fitt 8192 (the margin of the pasted log), fixture agentic-A, sequential arms under the GPU mutex
cd /c/Development/AI/poormansllama
LOG=build/llama-cuda/mp-queue6.log
run() { tag=$1; mkdir -p build/llama-cuda/$tag; rm -f build/llama-cuda/verdicts/$tag.*;
  pwsh -NoProfile -File tools/with-gpu-lock.ps1 -AllowUnprovenanced -VerdictTag $tag -TimeoutSeconds 3400 -Command "python tools/server_mt.py run --config tools/server-configs/$tag.json --out build/llama-cuda/$tag --fixture tools/server-fixtures/agentic-A.json --fixture2 tools/server-fixtures/agentic-B.json --scenarios agentic --reprocess-allow 2112 --think 3" > build/llama-cuda/$tag/wrapper.txt 2>&1
  echo "done $tag $(date +%H:%M:%S)" >> $LOG; }
run mp220-swa-off
run mp221-swa-always
run mp222-swa-always-free
echo "queue6 finished" >> $LOG
