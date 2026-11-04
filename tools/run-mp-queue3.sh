#!/bin/bash
# MP-1f identity: fresh-run greedy continuations (xcheck scenario, cache off) of the prepared build vs the base build, fused flow.
cd /c/Development/AI/poormansllama
LOG=build/llama-cuda/mp-queue3.log
note() { echo "$1 $(date +%H:%M:%S)" >> $LOG; }
run() { tag=$1; build=$2; scen=$3; shift 3; bash tools/run-g42-arm.sh "$tag" "$build" "$scen" --ho "--think 3" "$@" > build/llama-cuda/$tag-driver.out 2>&1; note "done $tag"; }
run mp130-xc-base mp-base xcheck --arg --lm-prefill-fuse
run mp131-xc-new  mp1     xcheck --arg --lm-prefill-fuse
note "queue3 finished"
