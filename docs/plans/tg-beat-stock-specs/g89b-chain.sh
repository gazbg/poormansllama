#!/bin/bash
# G26b chain: waits for the G24 v2 chain, then runs the owner's-binary probe (an external binary: wrapper waiver, recorded).
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
until [ -f build/llama-cuda/g87b-chain.finished ]; do sleep 60; done
rm -f build/llama-cuda/g89b-ownerbin.done
powershell -NoProfile -File tools/with-gpu-lock.ps1 -TimeoutSeconds 7200 -VerdictTag g89b-ownerbin -AllowUnprovenanced -Command "python tools/run-tg-arms.py --spec build/llama-cuda/specs/g89b-ownerbin.json" > build/llama-cuda/g89b-ownerbin.wrapper.log 2>&1
echo "exit=$?" > build/llama-cuda/g89b-ownerbin.done
echo finished > build/llama-cuda/g89b-chain.finished
