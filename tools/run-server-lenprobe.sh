#!/bin/bash
# usage: tools/run-server-lenprobe.sh <tag> "<server_len_probe.py options>" -- [server-config-gen.py options...]
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
tag=$1; hopts=$2; shift 3
python tools/server-config-gen.py tools/server-configs/$tag.json "$@" > /dev/null
mkdir -p build/llama-cuda/$tag
pwsh -NoProfile -File tools/with-gpu-lock.ps1 -AllowUnprovenanced -VerdictTag $tag -TimeoutSeconds 3400 -Command "python tools/server_len_probe.py --config tools/server-configs/$tag.json --out build/llama-cuda/$tag $hopts" > build/llama-cuda/$tag/wrapper.txt 2>&1
echo "== $tag"
grep -E "^len " build/llama-cuda/$tag/wrapper.txt | cut -c1-200
