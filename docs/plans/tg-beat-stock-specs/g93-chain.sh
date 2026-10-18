#!/bin/bash
# G30 / M1 chain on build g47 (no source edit): the clean decode effect of MTP (HB against HB + MTP, same batch-major prefill, GQA8 on in both arms).
# g93-c512 is run first by hand (smoke); this chain waits for its .done marker, then runs the other cells one session at a time.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
while [ ! -f build/llama-cuda/g93-c512.done ]; do sleep 20; done
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g47 g93-p512 g93-c4k g93-p4k g93-c24k g93-p24k g93-c50k g93-p50k
echo finished > build/llama-cuda/g93-chain.finished
