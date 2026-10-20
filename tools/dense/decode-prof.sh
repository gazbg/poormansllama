#!/usr/bin/env bash
# Decode profile of the decode-stream command (every second layer streamed, 2 slots, all KV on GPU q4_0, 262144 ctx).
# A warm-up request first; then nsys around one cold 670-token request that generates 16 tokens. Output: build/dense/decode-prof.*
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
PORT=8090
export CUDA_VISIBLE_DEVICES=0 LLAMA_LM_LOG=1
SESSION=decprof
timeout 1500 nsys launch --session-new=$SESSION -t cuda build/llama-cuda/bin/llama-server \
  --model ~/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf --device CUDA0  --port $PORT --host 127.0.0.1 \
  -fa on -np 1 -c 262144 -ctk q4_0 -ctv q4_0 -ub 256 -t 4 \
  --lm-prefill-host-layers 64 --lm-prefill --lm-prefill-fuse --lm-prefill-min 2 \
  --lm-prefill-prearm always \
  --lm-prefill-slots 2 --lm-prefill-ubatch 1024 --lm-prefill-segment 2048 \
  --lm-decode-stream "$(seq -s, 0 2 62)" --lm-decode-slots 2 > build/dense/srv-decprof.log 2>&1 &
for i in $(seq 1 400); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; sleep 2; done
echo healthy
python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 2 --label warm --ignore-eos | head -1
nsys start --session=$SESSION -o build/dense/decode-prof --force-overwrite=true > /dev/null 2>&1
python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 16 --ignore-eos --label capture | head -1
nsys stop --session=$SESSION > /dev/null 2>&1
fuser -k $PORT/tcp > /dev/null 2>&1
sleep 5
echo "DECPROF DONE"
