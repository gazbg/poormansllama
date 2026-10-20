#!/usr/bin/env bash
# Owner's stock command, ONE change per arm (the flags that differ from the default placement of build/dense/srv-maxctx-*):
#   A: -ub 512 -b 2048 instead of -ub 256 -b 1024      B: -ctk q8_0 -ctv q8_0 instead of -ctk q5_1 -ctv q4_0
# TG: 64 tokens after a 670-token prompt; PP: 3,659 tokens (4 generated), sampled with the owner's settings.
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
PORT=8090
MODEL="$HOME/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf"
arm() {
  label=$1; shift
  CUDA_VISIBLE_DEVICES=0 build/llama-cuda/bin/llama-server --model "$MODEL" --device CUDA0 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --chat-template-kwargs '{"reasoning_effort":"xhigh"}' \
    -fa on -lv 4 --port $PORT --host 127.0.0.1 -np 1 --load-mode none -c 262144 "$@" > build/dense/srv-abl-$label.log 2>&1 &
  SERVER=$!
  for i in $(seq 1 400); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "$label: server died"; return; }; sleep 2; done
  grep -a -E "load_tensors: offloaded [0-9]+/66|KV buffer size" build/dense/srv-abl-$label.log | tail -3 | cut -c1-150
  python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 4 --sampled --label warm > /dev/null
  python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 64 --ignore-eos --sampled --label "$label TG670" | head -1
  python3 -I tools/dense_probe.py --port $PORT --chars 12000 --n-predict 4 --ignore-eos --sampled --label "$label PP3659" | head -1
  kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
  sleep 5
}
arm A-ub512-b2048 -ub 512 -b 2048 -ctk q5_1 -ctv q4_0
arm B-kv-q8       -ub 256 -b 1024 -ctk q8_0 -ctv q8_0
echo ABL DONE
