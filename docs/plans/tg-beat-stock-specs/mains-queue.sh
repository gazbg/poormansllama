#!/bin/bash
# the speed sessions that waited for mains power: G10-2 (fast sampler, build g19) then G11-4 (async scan, build g20); counterbalanced, 6 pairs each
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
rm -f build/llama-cuda/g19-fs2.done build/llama-cuda/g20-scan2.done
powershell -NoProfile -File tools/with-gpu-lock.ps1 -TimeoutSeconds 7200 -VerdictTag g19-fs2 -Command "python tools\run-tg-arms.py --spec build/llama-cuda/specs/g19-fs2.json --attest build\llama-cuda\bin\g19\llama-layer-streaming.exe" > build/llama-cuda/g19-fs2.wrapper.log 2>&1
echo "exit=$?" > build/llama-cuda/g19-fs2.done
powershell -NoProfile -File tools/with-gpu-lock.ps1 -TimeoutSeconds 7200 -VerdictTag g20-scan2 -Command "python tools\run-tg-arms.py --spec build/llama-cuda/specs/g20-scan2.json --attest build\llama-cuda\bin\g20\llama-layer-streaming.exe" > build/llama-cuda/g20-scan2.wrapper.log 2>&1
echo "exit=$?" > build/llama-cuda/g20-scan2.done
