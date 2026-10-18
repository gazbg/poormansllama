#!/bin/bash
# G30-5 (answer stability with MTP on) and G30-1 (default-off identity) on build g52.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g52 g100-qa24000 g100-qa50000 g101-sc24 g101-lc4k g101-sgc50k
echo finished > build/llama-cuda/g100-chain.finished
