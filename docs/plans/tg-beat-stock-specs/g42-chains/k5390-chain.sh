#!/bin/bash
# G42 GA-4 / GA-6: checkpoint positions (default mode, trace log) and checkpoint hashes (canonical mode), fixture A agentic, build s51 against s47
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
until [ -f build/llama-cuda/k5350-chain.done ]; do sleep 20; done
sleep 25
rm -f build/llama-cuda/k5390-chain.done
run() { bash tools/run-g42-arm.sh "$@" > /dev/null 2>&1; }
run k5391-trace-def-base   s47 agentic --arg -lv --arg 4
run k5392-trace-def-fuse   s51 agentic --arg -lv --arg 4 --arg --lm-prefill-fuse
run k5393-hash-canon-base  s51 agentic --env LLAMA_LM_HASH=1 --arg --lm-prefill-canonical
run k5394-hash-canon-fuse  s51 agentic --env LLAMA_LM_HASH=1 --arg --lm-prefill-canonical --arg --lm-prefill-fuse
echo done > build/llama-cuda/k5390-chain.done
