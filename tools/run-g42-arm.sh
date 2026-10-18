#!/bin/bash
# G42: one server arm of the idle-preparation / fused-prompt work under the GPU mutex.
# usage: tools/run-g42-arm.sh <tag> <build> <scenarios> [--ho "extra harness options"] [--env K=V]... [--arg SERVERARG]...
#   e.g. tools/run-g42-arm.sh k5201-fuse s49 agentic --arg --lm-prefill-fuse
# Server flags common to every arm: --hybrid-moe --lm-prefill -fitt 16000, -np 1 (server-config-gen), LLAMA_LM_LOG=1, LLAMA_SRV_PHASE_LOG=1, free-memory watcher 20 ms.
# Results: build/llama-cuda/<tag>/ (report.json, server.log, wrapper.txt); the per-request timeline: python tools/server_phase_timeline.py build/llama-cuda/<tag>/server.log --brief
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
tag=$1; build=$2; scen=$3; shift 3
F=tools/server-fixtures
FIX=agentic-A
EXTRA=""
HOX=""
while [ $# -gt 0 ]; do
  case "$1" in
    --fix) FIX=$2; shift 2;;
    --ho) HOX="$HOX $2"; shift 2;;
    --env) EXTRA="$EXTRA --env $2"; shift 2;;
    --arg) EXTRA="$EXTRA --arg=$2"; shift 2;;
    *) echo "bad option $1"; exit 2;;
  esac
done
HO="--fixture $F/$FIX.json --fixture2 $F/agentic-B.json --scenarios $scen --reprocess-allow 2112$HOX"
LMH="--exe=build/llama-cuda/bin/$build/llama-server.exe --ub 0 --batch 0 --no-hybrid --no-override --env LLAMA_HYBRID_FREE_WATCH_MS=20 --env LLAMA_LM_LOG=1 --env LLAMA_SRV_PHASE_LOG=1 --arg=--hybrid-moe --arg=--lm-prefill --arg=-fitt --arg=16000"
bash tools/run-server-mt.sh $tag run "$HO" -- $LMH $EXTRA > build/llama-cuda/$tag.out 2>&1
cat build/llama-cuda/$tag.out | tail -20
