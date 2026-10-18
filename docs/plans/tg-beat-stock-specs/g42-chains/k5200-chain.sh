#!/bin/bash
# G42 speed arms with 3 s of think time between turns (the idle time the preparation is for): fixture A agentic, default and canonical mode. Order A B B A (round 1 forward, round 2 backward).
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
rm -f build/llama-cuda/k5200-chain.done
T="--ho"
TH="--think 3"
run() { bash tools/run-g42-arm.sh "$@" > /dev/null 2>&1; }
# round 1
run k5201-base-def     s47 agentic $T "$TH"
run k5202-fuse-def     s50 agentic $T "$TH" --arg --lm-prefill-fuse
run k5203-fuseal-def   s50 agentic $T "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
run k5204-fuseau-def   s50 agentic $T "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg auto
# round 2 (reverse)
run k5205-fuseau-def   s50 agentic $T "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg auto
run k5206-fuseal-def   s50 agentic $T "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
run k5207-fuse-def     s50 agentic $T "$TH" --arg --lm-prefill-fuse
run k5208-base-def     s47 agentic $T "$TH"
echo done > build/llama-cuda/k5200-chain.done
