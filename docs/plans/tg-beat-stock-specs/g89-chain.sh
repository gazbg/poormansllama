#!/bin/bash
# G26 chain: waits for the G25 chain, then runs the mmap probe (build g45).
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
until [ -f build/llama-cuda/g88-chain.finished ]; do sleep 60; done
rm -f build/llama-cuda/g89-chain.finished
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g45 g89-mmap
echo finished > build/llama-cuda/g89-chain.finished
