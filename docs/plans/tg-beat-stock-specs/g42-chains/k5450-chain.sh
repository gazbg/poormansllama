#!/bin/bash
# G42 diagnostic (s52): the restore of a checkpoint made inside the pass and what the staged banks do to it
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
run() { bash tools/run-g42-arm.sh "$@" > /dev/null 2>&1; }
run k5451-restore-def     s52 agentic --ho "--max-turns 6 --think 3"
run k5452-restore-fuse    s52 agentic --ho "--max-turns 6 --think 3" --arg --lm-prefill-fuse
run k5453-restore-fusear  s52 agentic --ho "--max-turns 6 --think 3" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
run k5454-restore-fusearS s52 agentic --ho "--max-turns 6 --think 3" --env LLAMA_LM_ARM_SYNC=1 --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
echo done > build/llama-cuda/k5450-chain.done
