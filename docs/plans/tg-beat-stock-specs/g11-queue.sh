#!/bin/bash
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
run() { # tag attest...
  tag=$1; shift
  rm -f build/llama-cuda/g11-$tag.done
  powershell -NoProfile -File tools/with-gpu-lock.ps1 -TimeoutSeconds 7200 -VerdictTag g11-$tag -Command "python tools\run-tg-arms.py --spec build\llama-cuda\specs\g11-$tag.json --attest $*" > build/llama-cuda/g11-$tag.wrapper.log 2>&1
  echo done > build/llama-cuda/g11-$tag.done
}
run auto "build\llama-cuda\bin\g11\llama-layer-streaming.exe"
for tag in anchor code prose; do
  run $tag "build\llama-cuda\bin\g2\llama-layer-streaming.exe build\llama-cuda\bin\g11\llama-completion.exe build\llama-cuda\bin\g11\llama-layer-streaming.exe"
done
run rel "build\llama-cuda\bin\g11\llama-layer-streaming.exe"
rm -f build/llama-cuda/g11-nsys.done
powershell -NoProfile -File tools/with-gpu-lock.ps1 -TimeoutSeconds 7200 -VerdictTag g11-nsys -Command "python tools\run-nsys-arm.py --verdict build\llama-cuda\g11-code\g11-code-verdict.json --arm HYB42R#1 HYB42RB#1 --tg-tokens 64 --out build\llama-cuda\g11-nsys\prof --attest build\llama-cuda\bin\g11\llama-layer-streaming.exe" > build/llama-cuda/g11-nsys.wrapper.log 2>&1
echo done > build/llama-cuda/g11-nsys.done
