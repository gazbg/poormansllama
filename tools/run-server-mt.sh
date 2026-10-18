#!/bin/bash
# usage: tools/run-server-mt.sh <tag> <build-fixture|run> <harness options after --> -- [server-config-gen.py options...]
#   e.g. tools/run-server-mt.sh g220-fixA build-fixture "--fixture tools/server-fixtures/agentic-A.json --variant A" -- --exe=build/llama-cuda/bin/s0/llama-server.exe --no-hybrid --no-override --arg=-fitt --arg=16000
# one server run under the GPU mutex; results in build/llama-cuda/<tag>/ (report.json, server.log, free_mib.txt)
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
tag=$1; cmd=$2; hopts=$3; shift 4
python tools/server-config-gen.py tools/server-configs/$tag.json "$@" > /dev/null
mkdir -p build/llama-cuda/$tag
pwsh -NoProfile -File tools/with-gpu-lock.ps1 -AllowUnprovenanced -VerdictTag $tag -TimeoutSeconds 3400 -Command "python tools/server_mt.py $cmd --config tools/server-configs/$tag.json --out build/llama-cuda/$tag $hopts" > build/llama-cuda/$tag/wrapper.txt 2>&1
echo "== $tag"
grep -v -E "^(Started|Waiting|Acquired|Released|GPU pre|PROVENANCE|  REFUSE|  WAIVER)" build/llama-cuda/$tag/wrapper.txt | cut -c1-230
tr -d '\000' < build/llama-cuda/$tag/server.log 2>/dev/null | grep -a -E "CUDA error|in function|GGML_ASSERT|free-watch: the engine" | head -3 | cut -c1-200
