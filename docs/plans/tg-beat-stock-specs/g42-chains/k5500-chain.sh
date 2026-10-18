#!/bin/bash
# G42 FINAL binary s53 (= s47 + g43..g49): speed (fixture A and the small-turn fixture S, 3 s think time, A B B A), canonical speed, identity of the default mode, first token against BASE.
# Timing sessions: run NO host-side analysis, build or test while this chain runs.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
rm -f build/llama-cuda/k5500-chain.done
F=tools/server-fixtures
P=build/llama-cuda/gplr07-qa-prompts
run() { bash tools/run-g42-arm.sh "$@" > /dev/null 2>&1; }
TH="--think 3"
# --- fixture A, default mode: base / fuse / fuse + always / fuse + auto, A B B A
run k5501-A-base    s47 agentic --ho "$TH"
run k5502-A-fuse    s53 agentic --ho "$TH" --arg --lm-prefill-fuse
run k5503-A-fuseal  s53 agentic --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
run k5504-A-fuseau  s53 agentic --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg auto
run k5505-A-fuseau  s53 agentic --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg auto
run k5506-A-fuseal  s53 agentic --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
run k5507-A-fuse    s53 agentic --ho "$TH" --arg --lm-prefill-fuse
run k5508-A-base    s47 agentic --ho "$TH"
# --- small-turn fixture S
run k5511-S-base    s47 agentic --fix agentic-S --ho "$TH"
run k5512-S-fuse    s53 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse
run k5513-S-fuseal  s53 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
run k5514-S-fuseau  s53 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg auto
run k5515-S-fuseau  s53 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg auto
run k5516-S-fuseal  s53 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
run k5517-S-fuse    s53 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse
run k5518-S-base    s47 agentic --fix agentic-S --ho "$TH"
# --- canonical mode, fixture A: base / fuse / fuse + auto, A B B A
run k5521-C-base    s47 agentic --ho "$TH" --arg --lm-prefill-canonical
run k5522-C-fuse    s53 agentic --ho "$TH" --arg --lm-prefill-canonical --arg --lm-prefill-fuse
run k5523-C-fuseau  s53 agentic --ho "$TH" --arg --lm-prefill-canonical --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg auto
run k5524-C-fuseau  s53 agentic --ho "$TH" --arg --lm-prefill-canonical --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg auto
run k5525-C-fuse    s53 agentic --ho "$TH" --arg --lm-prefill-canonical --arg --lm-prefill-fuse
run k5526-C-base    s47 agentic --ho "$TH" --arg --lm-prefill-canonical
echo done > build/llama-cuda/k5500-part1.done
# --- identity of the default mode: smoke, s47 against s53 without the new flags, A B B A
SMK=tools/server-scenarios/smoke.json
LMS="--ub 0 --batch 0 --no-hybrid --no-override --env LLAMA_HYBRID_FREE_WATCH_MS=20 --env LLAMA_LM_LOG=1 --arg=--hybrid-moe --arg=--lm-prefill --arg=-fitt --arg=16000"
bash tools/run-server-probe.sh k5531-id-A1 $SMK --exe=build/llama-cuda/bin/s47/llama-server.exe $LMS > build/llama-cuda/k5531.out 2>&1
bash tools/run-server-probe.sh k5532-id-B1 $SMK --exe=build/llama-cuda/bin/s53/llama-server.exe $LMS > build/llama-cuda/k5532.out 2>&1
bash tools/run-server-probe.sh k5533-id-B2 $SMK --exe=build/llama-cuda/bin/s53/llama-server.exe $LMS > build/llama-cuda/k5533.out 2>&1
bash tools/run-server-probe.sh k5534-id-A2 $SMK --exe=build/llama-cuda/bin/s47/llama-server.exe $LMS > build/llama-cuda/k5534.out 2>&1
python tools/server_smoke_cmp.py k5531-id-A1 k5532-id-B1 k5533-id-B2 k5534-id-A2 > build/llama-cuda/k5530-identity.txt 2>&1
# --- first token against BASE (the four prompts of LM-3), checkpoints on
PO="--prompt $P/mtpcode-4096.txt --prompt $P/mtpprose-4096.txt --prompt $P/natdeep-24000.txt --prompt $P/natdeep-50000.txt --reps 3"
D47="--exe=build/llama-cuda/bin/s47/llama-server.exe --ub 0 --batch 0 --no-hybrid --no-override --env LLAMA_LM_LOG=1 --arg=--hybrid-moe --arg=--lm-prefill --arg=-fitt --arg=16000"
D53="--exe=build/llama-cuda/bin/s53/llama-server.exe --ub 0 --batch 0 --no-hybrid --no-override --env LLAMA_LM_LOG=1 --arg=--hybrid-moe --arg=--lm-prefill --arg=-fitt --arg=16000"
bash tools/run-server-firsttok.sh k5541-ft-base "$PO" -- $D47 > build/llama-cuda/k5541.out 2>&1
bash tools/run-server-firsttok.sh k5542-ft-fuse "$PO" -- $D53 --arg=--lm-prefill-fuse > build/llama-cuda/k5542.out 2>&1
bash tools/run-server-firsttok.sh k5543-ft-fuse-canon "$PO" -- $D53 --arg=--lm-prefill-fuse --arg=--lm-prefill-canonical > build/llama-cuda/k5543.out 2>&1
python tools/server_firsttok_cmp.py build/llama-cuda/k5541-ft-base/firsttok.json build/llama-cuda/k5542-ft-fuse/firsttok.json > build/llama-cuda/k5540-firsttok.txt 2>&1
echo done > build/llama-cuda/k5500-chain.done
