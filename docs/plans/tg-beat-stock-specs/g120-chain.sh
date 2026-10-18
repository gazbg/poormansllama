#!/bin/bash
# MTP draft depth study on build g58: depth 1/2/4/6/8 with and without --mtp-draft-p-min 0.1, 4k code and prose.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g58 g120-lc4k g120-lp4k
echo finished > build/llama-cuda/g120-chain.finished
