#!/usr/bin/env bash
# First-token KL of the layer-major pass variants against the ordinary path at an equal chunk shape (512), 24 prompts of 1.5-4.3k tokens,
# plus the 8-prompt continuation check (DQ-2 rule) for the two variants the benchmarks use (bf16 slab; bf16 slab + int8 QK attention)
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
COMMON="-c 8192 -ctk q8_0 -ctv q8_0 -ub 512 --ctx-checkpoints 0 --lm-prefill-host-layers 38"
PASS="--lm-prefill --lm-prefill-ubatch 512 --lm-prefill-segment 4096 --lm-prefill-min 1024"
run() { # label flags env...
  label=$1; flags=$2; shift 2
  env "$@" PORT=8099 NPRED=1 NPROMPTS=24 SET=big tools/dense/collect.sh "$label" "$flags"
}
export EQSET_ARGS="--set big"
run klref   "$COMMON"                          A=1
run klf32   "$COMMON $PASS"                    A=1
run klbf16  "$COMMON $PASS --lm-prefill-bf16"  A=1
run klq8    "$COMMON $PASS --lm-prefill-bf16 --lm-prefill-attn-int8"  A=1
for v in klf32 klbf16 klq8; do echo "== first-token KL: $v against the ordinary path"; python3 -I tools/dense_equiv.py kl build/dense/eq-klref.json build/dense/eq-$v.json; done
echo "== 8-prompt continuation check (DQ-2 rule), 24 tokens"
NPRED=24 NPROMPTS=8 SET=std tools/dense/collect.sh d2bf16 "$COMMON $PASS --lm-prefill-bf16"
NPRED=24 NPROMPTS=8 SET=std tools/dense/collect.sh d2q8 "$COMMON $PASS --lm-prefill-bf16 --lm-prefill-attn-int8"
for v in d2bf16 d2q8; do echo "== $v against the ordinary path (eq-ref.json)"; python3 -I tools/dense_equiv.py compare build/dense/eq-ref.json build/dense/eq-$v.json | tail -9; done
echo KL DONE
