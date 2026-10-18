#!/bin/bash
# G30-4 repair, round 2: the late draft bring-up and the 3-slot interleave variant on build g54.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g54 g104-ic1k g104-lc24k g104-sgc50k
echo finished > build/llama-cuda/g104-chain.finished
