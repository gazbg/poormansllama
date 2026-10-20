#!/usr/bin/env bash
# DQ-3 at 15.7k tokens, A-B-B-A to cancel the GPU clock drift: STOCK -ngl 36 -ub 2048, LM + int8 QK attention with the f32 slab (the validated configuration), LM exact (f32 slab)
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
C="-c 20480 -ctk q8_0 -ctv q8_0"
LM="--lm-prefill --lm-prefill-fuse --lm-prefill-segment 4096 --lm-prefill-ubatch 2048 --lm-prefill-host-layers 38 -ub 512"
export NPRED=2
one() { label=$1; shift; echo "### $label"; sleep 45; tools/dense/bench.sh "$label" "$@" 46000 2>&1 | grep -E "prefill|peak VRAM|declines|died"; }
one ab-stock1  "$C -ngl 36 -ub 2048 -b 2048"
one ab-lmq8-1  "$C $LM --lm-prefill-attn-int8"
one ab-lmq8-2  "$C $LM --lm-prefill-attn-int8"
one ab-stock2  "$C -ngl 36 -ub 2048 -b 2048"
one ab-lmex-1  "$C $LM"
one ab-stock3  "$C -ngl 36 -ub 2048 -b 2048"
one ab-lmex-2  "$C $LM"
one ab-stock4  "$C -ngl 36 -ub 2048 -b 2048"
echo ABBA DONE
