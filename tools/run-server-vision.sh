#!/bin/bash
# usage: tools/run-server-vision.sh <tag> "<server_vision.py run options>" -- <server_vision.py mkcfg options: --arm base|novis|cpuv|od --exe EXE [--free-all] [--env K=V] [--extra-arg A]>
#   e.g. tools/run-server-vision.sh vod0-base "--phase p0 --reps 2" -- --arm base --exe build/llama-cuda/bin/vod-base/llama-server.exe --free-all
# one server run under the GPU mutex; results in build/llama-cuda/<tag>/ (report.json, server.log, free_mib.txt, wrapper.txt). Plan: docs/plans/vision-on-demand-plan.md
cd "$(dirname "$0")/.."
tag=$1; hopts=$2; shift 3
mkdir -p tools/server-configs build/llama-cuda/$tag
python tools/server_vision.py mkcfg --out tools/server-configs/$tag.json "$@" > /dev/null || exit 2
pwsh -NoProfile -File tools/with-gpu-lock.ps1 -AllowUnprovenanced -VerdictTag $tag -TimeoutSeconds 3400 -Command "python tools/server_vision.py run --config tools/server-configs/$tag.json --out build/llama-cuda/$tag $hopts" > build/llama-cuda/$tag/wrapper.txt 2>&1
echo "== $tag"
grep -v -E "^(Started|Waiting|Acquired|Released|GPU pre|PROVENANCE|  REFUSE|  WAIVER)" build/llama-cuda/$tag/wrapper.txt | cut -c1-230
grep -a -E "GPU pre|GPU post|PROVENANCE" build/llama-cuda/$tag/wrapper.txt | head -4 | cut -c1-200
tr -d '\000' < build/llama-cuda/$tag/server.log 2>/dev/null | grep -a -E "CUDA error|in function|GGML_ASSERT|free-watch: the engine" | head -3 | cut -c1-200
