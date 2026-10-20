#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
COMMON="-c 16384 -ctk q8_0 -ctv q8_0 -ub 512 --ctx-checkpoints 0 --lm-prefill-host-layers 38"
LLAMA_FATTN_TILE_Q8QK=1 tools/dense/collect.sh lmq8 "$COMMON --lm-prefill --lm-prefill-ubatch 512 --lm-prefill-segment 4096 --lm-prefill-min 1024"
echo "== int8 QK pass against the exact pass (both chunk 512)"
python3 -I tools/dense_equiv.py compare build/dense/eq-lm512.json build/dense/eq-lmq8.json
echo Q8EQ DONE
