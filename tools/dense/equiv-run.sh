#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
COMMON="-c 16384 -ctk q8_0 -ctv q8_0 -ub 512 --ctx-checkpoints 0 --lm-prefill-host-layers 38"
echo "== reference: ordinary path, same placement"; tools/dense/collect.sh ref "$COMMON" 
echo "== lm pass, chunk 512"; tools/dense/collect.sh lm512 "$COMMON --lm-prefill --lm-prefill-ubatch 512 --lm-prefill-segment 4096 --lm-prefill-min 1024"
python3 -I tools/dense_equiv.py compare build/dense/eq-ref.json build/dense/eq-lm512.json
echo EQUIV DONE
