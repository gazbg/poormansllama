#!/bin/bash
# G30 final evidence on build g56: answer stability with the late recipe (G30-5), default-off identity (G30-1), and the cells whose recipe did not change, rerun on the final build.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g56 g109-qa24000 g109-qa50000 g110-sc24 g110-lc4k g110-sgc50k g108-sc24 g108-sp24 g108-sc500 g108-sp500 g108-ic2k g108-ip2k g108-lc4k g108-lp4k
echo finished > build/llama-cuda/g108-chain.finished
