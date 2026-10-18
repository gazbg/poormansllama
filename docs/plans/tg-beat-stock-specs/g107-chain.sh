#!/bin/bash
# G30-4 repair, final: every cell whose recipe changed, with the band recipes that hold the bars, on build g56 (g55 + the 128-row start-up warm-up).
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g56 g107-lc24k g107-sgc50k g107-ic1k g107-ip1k g107-lp24k g107-sgp50k g107-sgc100k g107-sgc200k
echo finished > build/llama-cuda/g107-chain.finished
