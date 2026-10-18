#!/bin/bash
# G30 / M3 round 2: the final matrix on build g52 (B0 against the final MTP recipe MF, plus MFt / MT at 24k and 50k), code and prose per length.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g52 g99-sc24 g99-sp24 g99-sc500 g99-sp500 g99-ic1k g99-ip1k g99-ic2k g99-ip2k g99-lc24k g99-lp24k g99-sgc50k g99-sgp50k
echo finished > build/llama-cuda/g99-chain.finished
