#!/bin/bash
# G30 final evidence on build g57 (g56 + the graph-build accounting of the early release): answer stability of the late recipe, its timing, and the default-off identity.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g57 g112-qa24000 g112-qa50000 g113-lc24k g113-sgc50k g114-sc24 g114-lc4k g114-sgc50k
echo finished > build/llama-cuda/g112-chain.finished
