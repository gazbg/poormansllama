#!/bin/bash
# usage: run-session-queue.sh <build dir name under build/llama-cuda/bin, e.g. g31> <session> [<session> ...]
# One session at a time under the GPU mutex; each ends with a .done marker next to its spec's session directory. A session whose name contains "-chk-" is a self-check
# (the lean leg and the graph compared on every request): it must report "graph: 0 DIFFER" or the queue stops, because the speed numbers after it would mean nothing.
# The attested exe path is built with tr, not typed with backslashes: a typed "\$B" reached the wrapper as the literal text $B and every g31 session was refused as UNCERTIFIED.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
B="$1"; shift
EXE=$(echo "build/llama-cuda/bin/$B/llama-layer-streaming.exe" | tr '/' '\')
echo "attesting $EXE"
for s in "$@"; do
  rm -f build/llama-cuda/$s.done
  powershell -NoProfile -File tools/with-gpu-lock.ps1 -TimeoutSeconds 7200 -VerdictTag $s -Command "python tools\run-tg-arms.py --spec build/llama-cuda/specs/$s.json --attest $EXE" > build/llama-cuda/$s.wrapper.log 2>&1
  echo "exit=$?" > build/llama-cuda/$s.done
  case "$s" in
    *-chk-*)
      if ! cat build/llama-cuda/$s/*.log 2>/dev/null | tr -d '\0' | grep -a -q "graph: 0 DIFFER"; then
        echo "self-check not clean in $s, queue stopped" > build/llama-cuda/$B-queue.stopped
        exit 1
      fi
      ;;
  esac
done
echo finished > build/llama-cuda/$B-queue.finished
