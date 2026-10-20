#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
export LLAMA_FATTN_TILE_Q8QK=1 NPRED=8
C="-c 20480 -ctk q8_0 -ctv q8_0"
LM="--lm-prefill --lm-prefill-fuse --lm-prefill-segment 4096 --lm-prefill-bf16 --lm-prefill-ubatch 2048"
echo "### LM + int8 QK attention"
tools/dense/bench.sh lm38-q8qk "$C -ub 512 --lm-prefill-host-layers 38 $LM" 12000 52000 2>&1 | grep -v "^lm-prefill: pass"
echo "### STOCK -ngl 36 -ub 2048 + int8 QK attention"
tools/dense/bench.sh st36-q8qk "$C -ngl 36 -ub 2048 -b 2048" 12000 52000 2>&1 | grep -v "^lm-prefill: pass"
echo Q8QK DONE
