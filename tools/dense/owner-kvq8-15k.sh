#!/usr/bin/env bash
# Owner's stock command with -ctk q8_0 -ctv q8_0 (and -ub 256 -b 1024 as the owner's line), at 15,700 tokens of prefill.
cd "$(dirname "$0")/../.." || exit 1
PORT=8090
CUDA_VISIBLE_DEVICES=0 build/llama-cuda/bin/llama-server --model "$HOME/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf" --device CUDA0 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --chat-template-kwargs '{"reasoning_effort":"xhigh"}' \
  -fa on -b 1024 -ub 256 -lv 4 --port $PORT --host 127.0.0.1 -ctk q8_0 -ctv q8_0 -np 1 --load-mode none -c 262144 > build/dense/srv-kvq8-15k.log 2>&1 &
SERVER=$!
for i in $(seq 1 400); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "server died"; exit 1; }; sleep 2; done
python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 4 --sampled --label warm > /dev/null
python3 -I tools/dense_probe.py --port $PORT --chars 46000 --n-predict 4 --ignore-eos --sampled --label "kvq8 PP15700" --out build/dense/out-kvq8-15k.json | head -1
kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
echo KVQ8 DONE
