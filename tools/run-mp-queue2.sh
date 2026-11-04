#!/bin/bash
# MP-1 queue 2: flag surface, full harness suite, async-regrow env arm, OFF replicate, small-turn fixture S. One GPU workload at a time (each takes the mutex itself).
cd /c/Development/AI/poormansllama
LOG=build/llama-cuda/mp-queue2.log
note() { echo "$1 $(date +%H:%M:%S)" >> $LOG; }
run() { tag=$1; build=$2; scen=$3; shift 3; bash tools/run-g42-arm.sh "$tag" "$build" "$scen" --ho "--think 3" "$@" > build/llama-cuda/$tag-driver.out 2>&1; note "done $tag"; }

mkdir -p build/llama-cuda/mp120-flags
rm -f build/llama-cuda/verdicts/mp120-flags.*
pwsh -NoProfile -File tools/with-gpu-lock.ps1 -AllowUnprovenanced -VerdictTag mp120-flags -TimeoutSeconds 3400 -Command "python tools/check-mp-flags.py --exe build/llama-cuda/bin/mp1/llama-server.exe" > build/llama-cuda/mp120-flags/wrapper.txt 2>&1
note "done mp120-flags"

run mp110-new-suite  mp1     prefix,agentic,branch,alternate,sampling,equiv --arg --lm-prefill-fuse
run mp111-base-suite mp-base prefix,agentic,branch,alternate,sampling,equiv --arg --lm-prefill-fuse
run mp112-new-async  mp1     agentic --arg --lm-prefill-fuse --env LLAMA_MOE_HYBRID_SHRINK_KEEP_ONLY=1 --env LLAMA_MOE_HYBRID_REGROW_ASYNC=1
run mp113-new-off2   mp1     agentic --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg off
run mp114-base-S     mp-base agentic --fix agentic-S --arg --lm-prefill-fuse
run mp115-new-S      mp1     agentic --fix agentic-S --arg --lm-prefill-fuse
run mp116-base-S2    mp-base agentic --fix agentic-S --arg --lm-prefill-fuse
run mp117-new-S2     mp1     agentic --fix agentic-S --arg --lm-prefill-fuse
note "queue2 finished"
