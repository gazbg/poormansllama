#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
C="-c 20480 -ctk q8_0 -ctv q8_0"
LM="--lm-prefill --lm-prefill-fuse --lm-prefill-segment 4096 --lm-prefill-bf16"
export NPRED=8
run() { label=$1; shift; echo "### $label"; tools/dense/bench.sh "$label" "$@" 12000 30000 52000 2>&1 | grep -v "^lm-prefill: pass 1:"; }
run lm38-ub2048 "$C -ub 512 --lm-prefill-host-layers 38 $LM --lm-prefill-ubatch 2048"
run lm28-ub1024 "$C -ub 512 --lm-prefill-host-layers 28 $LM --lm-prefill-ubatch 1024"
run st36-ub2048 "$C -ngl 36 -ub 2048 -b 2048"
run st36-ub1024 "$C -ngl 36 -ub 1024 -b 1024"
run st40-ub1024 "$C -ngl 40 -ub 1024 -b 1024"
echo GRID DONE
