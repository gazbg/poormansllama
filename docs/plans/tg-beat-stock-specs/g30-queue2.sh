#!/bin/bash
# G14 lean host leg, the sessions after the first self-check (g30-chk-code ran from g30-queue.sh, whose grep for the clean-check line was broken by a NUL byte in the script).
# One session at a time under the GPU mutex; each ends with a .done marker. A self-check session that is not clean stops the queue: the speed numbers would mean nothing.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
for s in "$@"; do
  rm -f build/llama-cuda/$s.done
  powershell -NoProfile -File tools/with-gpu-lock.ps1 -TimeoutSeconds 7200 -VerdictTag $s -Command "python tools\run-tg-arms.py --spec build/llama-cuda/specs/$s.json --attest build\llama-cuda\bin\g30\llama-layer-streaming.exe" > build/llama-cuda/$s.wrapper.log 2>&1
  echo "exit=$?" > build/llama-cuda/$s.done
  case "$s" in
    g30-chk-*)
      if ! cat build/llama-cuda/$s/*LEAN2*.log 2>/dev/null | tr -d '\0' | grep -a -q "graph: 0 DIFFER"; then
        echo "self-check not clean in $s, queue stopped" > build/llama-cuda/g30-queue.stopped
        exit 1
      fi
      ;;
  esac
done
echo finished > build/llama-cuda/g30-queue.finished
