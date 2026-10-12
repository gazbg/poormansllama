#!/bin/bash
# G12 correctness sessions (battery allowed: exactness / acceptance only): code prompt 4 reps, prose prompt 2 reps, build g22, ctx 8192 without lazy KV (batch-major PP reads the whole window)
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
for tag in g22-mtp-code g22-mtp-prose; do
  rm -f build/llama-cuda/$tag.done
  powershell -NoProfile -File tools/with-gpu-lock.ps1 -TimeoutSeconds 7200 -VerdictTag $tag -Command "python tools\run-tg-arms.py --spec build/llama-cuda/specs/$tag.json --attest build\llama-cuda\bin\g22\llama-layer-streaming.exe" > build/llama-cuda/$tag.wrapper.log 2>&1
  echo "exit=$?" > build/llama-cuda/$tag.done
done
