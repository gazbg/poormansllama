#!/bin/bash
# Plan H102 M3.2: the layer interleave against the one-GPU pass, A-B-B-A over a ladder of prompt sizes (cold prefill, greedy, 32 tokens, top-5 logprobs).
#   bash tools/pascal/ilv_abba.sh [SIZES] [OUTDIR]
# A0 = the owner's recipe (--lm-prefill-min default), A1 = the same with --lm-prefill-min 256, B = A1 + the second GPU (interleave below 4096 tokens, split above).
# Each arm is tools/pascal/split_check.py; its JSON is copied to OUTDIR/<arm>.json. Order: A0 A1 B1 B2 A1b A0b. Writes OUTDIR/queue.done at the end.
cd "$(dirname "$0")/../.." || exit 1
SIZES=${1:-1500,2048,2500,3000,4000,4096,4500,6000,16384}
OUT=${2:-build/pascal/ilv}
mkdir -p "$OUT"
rm -f "$OUT/queue.done"
run() {   # label arm extra-args...
    local label=$1 arm=$2
    shift 2
    echo "ilv_abba: $label ($arm) $*" >> "$OUT/queue.log"
    echo "ilv_abba.sh arm $label (my queue), remove after the job" > build/pascal/GPU.lock
    python3 -I tools/pascal/split_check.py --arm "$arm" --sizes "$SIZES" "$@" >> "$OUT/queue.log" 2>&1
    rm -f build/pascal/GPU.lock
    cp "build/pascal/split/$arm.json" "$OUT/$label.json"
    cp "build/pascal/split/$arm.server.log" "$OUT/$label.server.log"
}
run A0 ref
run A1 ref --add "--lm-prefill-min 256"
run B1 peer --add "--lm-prefill-min 256"
run B2 peer --add "--lm-prefill-min 256"
run A1b ref --add "--lm-prefill-min 256"
run A0b ref
touch "$OUT/queue.done"
