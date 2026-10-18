#!/bin/bash
# G42 small-turn fixture S (turns of ~350-850 new tokens), 3 s think time: the break-even of --lm-prefill-fuse-min and the cost of the idle preparation when most turns are small. A B C D E F F E D C B A not needed: one forward round, one backward.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
rm -f build/llama-cuda/k5330-chain.done
run() { bash tools/run-g42-arm.sh "$@" > /dev/null 2>&1; }
S="--fix agentic-S --ho --think=3"
TH="--think 3"
run k5331-S-base      s47 agentic --fix agentic-S --ho "$TH"
run k5332-S-fuse300   s51 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-fuse-min --arg 300
run k5333-S-fuse700   s51 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse
run k5334-S-fuse1100  s51 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-fuse-min --arg 1100
run k5335-S-fuse700al s51 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
run k5336-S-fuse700au s51 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg auto
run k5337-S-fuse700au s51 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg auto
run k5338-S-fuse700al s51 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
run k5339-S-fuse1100  s51 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-fuse-min --arg 1100
run k5340-S-fuse700   s51 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse
run k5341-S-fuse300   s51 agentic --fix agentic-S --ho "$TH" --arg --lm-prefill-fuse --arg --lm-prefill-fuse-min --arg 300
run k5342-S-base      s47 agentic --fix agentic-S --ho "$TH"
echo done > build/llama-cuda/k5330-chain.done
