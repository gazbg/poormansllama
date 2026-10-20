#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
C="-c 20480 -ctk q8_0 -ctv q8_0 -ub 512"
LM="--lm-prefill --lm-prefill-fuse --lm-prefill-segment 4096 --lm-prefill-bf16 --lm-prefill-ubatch 2048"
export NPRED=8
echo "### lm48 (16 resident layers)"
tools/dense/bench.sh lm48 "$C --lm-prefill-host-layers 48 $LM" 12000 52000 2>&1 | grep -v "^lm-prefill: pass [12]:"
echo "### lm58 (6 resident layers)"
tools/dense/bench.sh lm58 "$C --lm-prefill-host-layers 58 $LM" 12000 2>&1 | grep -v "^lm-prefill: pass [12]:"
echo "### lm28 (36 resident layers, the guard)"
tools/dense/bench.sh lm28 "$C --lm-prefill-host-layers 28 $LM" 12000 2>&1 | grep -v "^lm-prefill: pass [12]:"
grep -a "takes the ordinary path" build/dense/srv-lm28.log | head -2 | cut -c1-330
echo DQ4 DONE
