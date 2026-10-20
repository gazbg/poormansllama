#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
tools/dense/capture2.sh lm18k 52000 "-c 20480 -ctk q8_0 -ctv q8_0 -ub 512 --lm-prefill-host-layers 38 --lm-prefill --lm-prefill-fuse --lm-prefill-segment 4096 --lm-prefill-bf16 --lm-prefill-ubatch 2048"
tools/dense/capture2.sh st18k 52000 "-c 20480 -ctk q8_0 -ctv q8_0 -ngl 36 -ub 2048 -b 2048"
echo PAIR DONE
