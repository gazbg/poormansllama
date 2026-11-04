#!/bin/bash
# MP-0 / MP-1 speed series, fixture agentic-A, 3 s think time, np 1. Sequential arms under the GPU mutex (each arm takes it inside run-g42-arm.sh).
cd /c/Development/AI/poormansllama
THINK="--think 3"
run() { tag=$1; build=$2; shift 2; bash tools/run-g42-arm.sh "$tag" "$build" agentic --ho "$THINK" "$@" > build/llama-cuda/$tag-driver.out 2>&1; echo "done $tag $(date +%H:%M:%S)" >> build/llama-cuda/mp-queue1.log; }
# second halves of A B B A: BASE (prearm off), BASE + always (the G42 reference on this tree), NEW with the preparation switched off (identity), then BASE and NEW again
run mp102-base-fuse        mp-base --arg --lm-prefill-fuse
run mp103-base-fuse-always mp-base --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
run mp104-new-fuse-off     mp1     --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg off
run mp105-new-fuse         mp1     --arg --lm-prefill-fuse
run mp106-base-fuse        mp-base --arg --lm-prefill-fuse
echo "queue1 finished" >> build/llama-cuda/mp-queue1.log
