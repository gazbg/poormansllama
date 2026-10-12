#!/bin/bash
# certified re-runs of the two mains speed sessions: the g19 / g20 side-copies predate the width-W edits of ggml.h, so the provenance gate refuses to attest them
# (g19-fs2 / g20-scan2 are kept as UNCERTIFIED measurements); build g27 postdates every source edit
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
rm -f build/llama-cuda/g27-fs2.done build/llama-cuda/g27-scan2.done
powershell -NoProfile -File tools/with-gpu-lock.ps1 -TimeoutSeconds 7200 -VerdictTag g27-fs2 -Command "python tools\run-tg-arms.py --spec build/llama-cuda/specs/g27-fs2.json --attest build\llama-cuda\bin\g27\llama-layer-streaming.exe" > build/llama-cuda/g27-fs2.wrapper.log 2>&1
echo "exit=$?" > build/llama-cuda/g27-fs2.done
powershell -NoProfile -File tools/with-gpu-lock.ps1 -TimeoutSeconds 7200 -VerdictTag g27-scan2 -Command "python tools\run-tg-arms.py --spec build/llama-cuda/specs/g27-scan2.json --attest build\llama-cuda\bin\g27\llama-layer-streaming.exe" > build/llama-cuda/g27-scan2.wrapper.log 2>&1
echo "exit=$?" > build/llama-cuda/g27-scan2.done
