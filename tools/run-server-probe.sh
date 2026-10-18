#!/bin/bash
# usage: tools/run-server-probe.sh <tag> <scenario.json> [server-config-gen.py options...]   (one server run under the GPU mutex; results in build/llama-cuda/<tag>/)
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
tag=$1; scen=$2; shift 2
python tools/server-config-gen.py tools/server-configs/$tag.json "$@" > /dev/null
mkdir -p build/llama-cuda/$tag
pwsh -NoProfile -File tools/with-gpu-lock.ps1 -AllowUnprovenanced -VerdictTag $tag -TimeoutSeconds 3400 -Command "python tools/server-probe.py --config tools/server-configs/$tag.json --out build/llama-cuda/$tag --scenario $scen" > build/llama-cuda/$tag/wrapper.txt 2>&1
echo "== $tag"
grep -E "server ready|^[a-z0-9-]+ +http|exited early|not ready|Traceback" build/llama-cuda/$tag/wrapper.txt | cut -c1-210
tr -d '\000' < build/llama-cuda/$tag/server.log 2>/dev/null | grep -a -E "CUDA error|in function|GGML_ASSERT" | head -2 | cut -c1-200
