#!/usr/bin/env bash
# DQ-3 at ~16k tokens exactly: one pass of ~16k tokens (no second pass behind the 16,384-token logical batch)
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
C="-c 20480 -ctk q8_0 -ctv q8_0"
LM="--lm-prefill --lm-prefill-fuse --lm-prefill-segment 4096 --lm-prefill-bf16 --lm-prefill-ubatch 2048"
export NPRED=4
echo "### STOCK -ngl 36 -ub 2048"
tools/dense/bench.sh d16-stock "$C -ngl 36 -ub 2048 -b 2048" 46000 2>&1 | grep -v "^lm-prefill: pass"
echo "### LM default kernels"
tools/dense/bench.sh d16-lm "$C -ub 512 --lm-prefill-host-layers 38 $LM" 46000 2>&1 | grep -v "^lm-prefill: pass [12]:"
echo "### LM int8 QK attention"
LLAMA_FATTN_TILE_Q8QK=1 tools/dense/bench.sh d16-lmq8 "$C -ub 512 --lm-prefill-host-layers 38 $LM" 46000 2>&1 | grep -v "^lm-prefill: pass [12]:"
echo D16 DONE
