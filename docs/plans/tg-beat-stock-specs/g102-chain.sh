#!/bin/bash
# G30 / M3 round 2 supplements on build g52: the 500-token code cell with 8 reps, then the deep code cells (100k, 200k).
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g52 g102-sc500 g102-sgc100k g102-sgc200k
echo finished > build/llama-cuda/g102-chain.finished
