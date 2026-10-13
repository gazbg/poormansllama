#!/bin/bash
# G25 chain: waits for the G24 chain to finish (its marker), then runs the batch-major ubatch sweep sessions one at a time under the GPU mutex (build g45).
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
until [ -f build/llama-cuda/g87-chain.finished ]; do sleep 60; done
rm -f build/llama-cuda/g88-chain.finished
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g45 g88-bm50k g88-bm100k g88-bm200k g88-bm250k
echo finished > build/llama-cuda/g88-chain.finished
