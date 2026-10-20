#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
S="2,5,8,11,14,17,20,23,26,29"
BASE="-c 6144 -ctk q8_0 -ctv q8_0 -ub 512 --ctx-checkpoints 0"
echo "== P: host 0-37, layers $S streamed"
tools/dense/collect.sh exP "$BASE --lm-prefill-host-layers 0-37 --lm-decode-stream $S --lm-decode-slots 2"
echo "== P': the same layers resident on the device (host = the other 28)"
tools/dense/collect.sh exQ "$BASE --lm-prefill-host-layers 0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30-37"
python3 -I - <<'PY'
import json
a=json.load(open('build/dense/eq-exP.json')); b=json.load(open('build/dense/eq-exQ.json'))
ok=0
for x,y in zip(a,b):
    ids_x=[t['id'] for t in x['tokens']]; ids_y=[t['id'] for t in y['tokens']]
    lp_x=[t['lp'] for t in x['tokens']]; lp_y=[t['lp'] for t in y['tokens']]
    same_ids = ids_x==ids_y; same_lp = lp_x==lp_y
    ok += same_ids and same_lp
    print(f"prompt {x['i']} ({x['prompt_n']} tok): ids {'identical' if same_ids else 'DIFFER'} ({len(ids_x)} vs {len(ids_y)}), logprobs {'bit-identical' if same_lp else 'differ (max %.2e)' % max(abs(p-q) for p,q in zip(lp_x,lp_y))}")
print(f"{ok}/{len(a)} prompts identical")
PY
echo EXACT DONE
