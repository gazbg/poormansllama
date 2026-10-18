#!/bin/bash
# G42 fault injection with --lm-prefill-fuse (build s50): the pass declines every batch / fails once after the first layer / refuses its resources
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
rm -f build/llama-cuda/k5310-chain.done
run() { bash tools/run-g42-arm.sh "$@" > /dev/null 2>&1; }
run k5311-fi-run1-fuse  s50 agentic --ho "--max-turns 5" --env LLAMA_LM_FAIL_RUN=1 --env LLAMA_LM_FAIL_AT=0 --arg --lm-prefill-fuse
run k5312-fi-run2-fuse  s50 agentic --ho "--max-turns 5" --env LLAMA_LM_FAIL_RUN=2 --env LLAMA_LM_FAIL_AT=3 --arg --lm-prefill-fuse
run k5313-fi-acq-fuse   s50 agentic --ho "--max-turns 5" --env LLAMA_LM_FAIL_ACQUIRE=1 --arg --lm-prefill-fuse
echo done > build/llama-cuda/k5310-chain.done
