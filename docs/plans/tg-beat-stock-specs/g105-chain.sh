#!/bin/bash
# G30-4 repair, round 3: the late bring-up with its start-up warm-up, and the 1,024-row variant of the interleave cell, on build g55.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g55 g105-ic1k g105-lc24k g105-sgc50k
echo finished > build/llama-cuda/g105-chain.finished
