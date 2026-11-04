#!/bin/bash
# MP-1f root check: the equivalence suite of the prepared build with the preparation switched off (identity arm of the same binary).
cd /c/Development/AI/poormansllama
LOG=build/llama-cuda/mp-queue4.log
bash tools/run-g42-arm.sh mp140-new-suite-off mp1 prefix,agentic,branch,alternate,sampling,equiv --ho "--think 3" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg off > build/llama-cuda/mp140-new-suite-off-driver.out 2>&1
echo "done mp140 $(date +%H:%M:%S)" >> $LOG
