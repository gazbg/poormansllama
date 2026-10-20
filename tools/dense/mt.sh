#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
D=build/dense; PORT=8099
export CUDA_VISIBLE_DEVICES=0 LLAMA_LM_LOG=1
build/llama-cuda/bin/llama-server -m $HOME/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf --host 127.0.0.1 --port $PORT -dev CUDA0 -sm none -mg 0 -fa on -np 1 -t 4 --no-webui "$@" > $D/srv-mt.log 2>&1 &
SERVER=$!
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -5 $D/srv-mt.log | cut -c1-250; exit 1; }; sleep 2; done
python3 -I tools/dense_multiturn.py --port $PORT
kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
grep -a -E "lm-prefill: pass|lm-fuse:|dense-stream:|checkpoint|erased|restor" $D/srv-mt.log | sed -E 's/^[0-9.]+ [WI] //' | cut -c1-210 | head -24
