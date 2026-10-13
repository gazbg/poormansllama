#!/bin/bash
# G24 chain: the owner's llama-server baseline reproduction, then the run-tg-arms sessions one at a time under the GPU mutex (build g45).
# The server is an external binary, so it cannot be attested: the wrapper is called with -AllowUnprovenanced (recorded in provenance-waivers.log).
# Every other session goes through run-session-queue.sh, which attests build/llama-cuda/bin/g45/llama-layer-streaming.exe (named LAST on the wrapped command).
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
rm -f build/llama-cuda/g87-doc.done
powershell -NoProfile -File tools/with-gpu-lock.ps1 -TimeoutSeconds 7200 -VerdictTag g87-doc -AllowUnprovenanced -Command "python docs/plans/tg-beat-stock-specs/g87_docserver.py nat-B" > build/llama-cuda/g87-doc.wrapper.log 2>&1
echo "exit=$?" > build/llama-cuda/g87-doc.done
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g45 g87-qab24000 g87-qab50000 g87-short-B g87-short-A g87-n24k g87-n50k g87-qab100000 g87-n100k g87-n200k g87-n250k
echo finished > build/llama-cuda/g87-chain.finished
