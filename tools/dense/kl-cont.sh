#!/usr/bin/env bash
# the 8-prompt continuation check (DQ-2 rule) for the two pass variants the benchmarks use, against the ordinary path (build/dense/eq-ref.json from equiv-run.sh)
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
COMMON="-c 8192 -ctk q8_0 -ctv q8_0 -ub 512 --ctx-checkpoints 0 --lm-prefill-host-layers 38"
PASS="--lm-prefill --lm-prefill-ubatch 512 --lm-prefill-segment 4096 --lm-prefill-min 1024"
NPRED=24 NPROMPTS=8 SET=std tools/dense/collect.sh d2bf16 "$COMMON $PASS --lm-prefill-bf16"
NPRED=24 NPROMPTS=8 SET=std tools/dense/collect.sh d2q8 "$COMMON $PASS --lm-prefill-bf16 --lm-prefill-attn-int8"
for v in d2bf16 d2q8; do echo "== $v against the ordinary path (eq-ref.json)"; python3 -I tools/dense_equiv.py compare build/dense/eq-ref.json build/dense/eq-$v.json; done
echo CONT DONE
