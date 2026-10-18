#!/bin/bash
# G42 first functional run: fixture A agentic, default mode, --lm-prefill-fuse (build s48)
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
F=tools/server-fixtures
B=s48
LMH="--exe=build/llama-cuda/bin/$B/llama-server.exe --ub 0 --batch 0 --no-hybrid --no-override --env LLAMA_HYBRID_FREE_WATCH_MS=20 --env LLAMA_LM_LOG=1 --env LLAMA_SRV_PHASE_LOG=1 --arg=--hybrid-moe --arg=--lm-prefill --arg=-fitt --arg=16000"
HO="--fixture $F/agentic-A.json --fixture2 $F/agentic-B.json --scenarios agentic --reprocess-allow 2112"
rm -f build/llama-cuda/k5100-chain.done
bash tools/run-server-mt.sh k5101-fuse-def-agentic run "$HO" -- $LMH --arg=--lm-prefill-fuse > build/llama-cuda/k5101.out 2>&1
echo done > build/llama-cuda/k5100-chain.done
