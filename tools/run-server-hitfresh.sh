#!/bin/bash
# usage: tools/run-server-hitfresh.sh <tag> "<server_hitfresh.py run options>" -- [server-config-gen.py options...]   (one server under the GPU mutex; results in build/llama-cuda/<tag>/)
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
tag=$1; hopts=$2; shift 3
python tools/server-config-gen.py tools/server-configs/$tag.json "$@" > /dev/null
mkdir -p build/llama-cuda/$tag
pwsh -NoProfile -File tools/with-gpu-lock.ps1 -AllowUnprovenanced -VerdictTag $tag -TimeoutSeconds 5400 -Command "python tools/server_hitfresh.py run --config tools/server-configs/$tag.json --out build/llama-cuda/$tag $hopts" > build/llama-cuda/$tag/wrapper.txt 2>&1
echo "== $tag"
grep -a -E "Traceback|Error|skip" build/llama-cuda/$tag/wrapper.txt | cut -c1-260
