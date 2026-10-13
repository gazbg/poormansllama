#!/bin/bash
# G24 v2 chain: after the G26 chain, rerun the two natural-prompt sessions whose v1 prompt was read differently by the two engines (backslash escape processing), build g45.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
until [ -f build/llama-cuda/g89-chain.finished ] && [ -f build/llama-cuda/g87-natdeep-v2.done ]; do sleep 60; done
rm -f build/llama-cuda/g87b-chain.finished
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g45 g87-n24kv2 g87-n50kv2
echo finished > build/llama-cuda/g87b-chain.finished
