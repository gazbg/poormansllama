#!/bin/bash
# G42 one closing verification on the final binary s53 (kept short on purpose): canonical + fuse + idle preparation, stock decode swapped suite (bit-identical expected), and the canonical checkpoint hashes with idle preparation
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
until [ -f build/llama-cuda/k5600-chain.done ]; do sleep 20; done
sleep 25
rm -f build/llama-cuda/k5800-chain.done
F=tools/server-fixtures
B=s53
SC="prefix,agentic,branch,alternate,sampling,equiv,xcheck,noise,stream"
HOS="--fixture $F/agentic-B.json --fixture2 $F/agentic-A.json --scenarios $SC --reprocess-allow 2112"
N37="--no-hybrid --no-override --arg=-ngl --arg=99 --arg=-ncmoe --arg=37 --arg=-fit --arg=off"
X="--exe=build/llama-cuda/bin/$B/llama-server.exe --batch 0 --env LLAMA_LM_LOG=1 $N37 --arg=--lm-prefill --arg=--lm-prefill-canonical --arg=--lm-prefill-fuse --arg=--lm-prefill-prearm --arg=always"
bash tools/run-server-mt.sh k5801-cfa-lmstock-mt-swap run "$HOS" -- $X > build/llama-cuda/k5801.out 2>&1
bash tools/run-g42-arm.sh k5802-hash-canon-fuseal s53 agentic --env LLAMA_LM_HASH=1 --arg --lm-prefill-canonical --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always --ho "--think 3" > /dev/null 2>&1
echo done > build/llama-cuda/k5800-chain.done
