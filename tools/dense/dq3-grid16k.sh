#!/usr/bin/env bash
# DQ-3 at 15.7k tokens: stock candidates against LM candidates, one run each (then an A-B-B-A of the best pair is run by hand)
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
C="-c 20480 -ctk q8_0 -ctv q8_0"
export NPRED=2
one() { label=$1; shift; echo "### $label"; sleep 40; tools/dense/bench.sh "$label" "$@" 46000 2>&1 | grep -E "prefill|peak VRAM|declines|died"; }
LMB="--lm-prefill --lm-prefill-fuse --lm-prefill-bf16"
one g-st36-ub2048  "$C -ngl 36 -ub 2048 -b 2048"
one g-st30-ub4096  "$C -ngl 30 -ub 4096 -b 4096"
one g-st24-ub4096  "$C -ngl 24 -ub 4096 -b 4096"
one g-lm38-ub2048  "$C -ub 512 --lm-prefill-host-layers 38 $LMB --lm-prefill-segment 4096 --lm-prefill-ubatch 2048"
one g-lm50-ub4096  "$C -ub 512 --lm-prefill-host-layers 50 $LMB --lm-prefill-segment 8192 --lm-prefill-ubatch 4096"
export LLAMA_FATTN_TILE_Q8QK=1
one g-lm38-ub2048-q8 "$C -ub 512 --lm-prefill-host-layers 38 $LMB --lm-prefill-segment 4096 --lm-prefill-ubatch 2048"
one g-lm50-ub4096-q8 "$C -ub 512 --lm-prefill-host-layers 50 $LMB --lm-prefill-segment 8192 --lm-prefill-ubatch 4096"
echo G16 DONE
