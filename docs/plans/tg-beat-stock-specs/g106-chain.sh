#!/bin/bash
# G30-4 repair, round 4: the remaining cells (prose 1k / 24k / 50k, code 100k / 200k) with the band recipes that hold the bars, on build g55.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g55 g106-ip1k g106-lp24k g106-sgp50k g106-sgc100k g106-sgc200k
echo finished > build/llama-cuda/g106-chain.finished
