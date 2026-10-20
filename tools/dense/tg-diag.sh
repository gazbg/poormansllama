#!/usr/bin/env bash
# Token-generation diagnosis of the owner's sanity-check command (-c 162144, q4_0 KV, decode stream every second layer).
# Warm-up request, then nsys around one 670-token request that generates 48 tokens. Logs: build/dense/tg-diag.*
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
PORT=8090
export CUDA_VISIBLE_DEVICES=0 LLAMA_LM_LOG=1
SESSION=tgdiag
timeout 1500 nsys launch --session-new=$SESSION -t cuda build/llama-cuda/bin/llama-server \
  --model ~/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf --device CUDA0 --port $PORT --host 127.0.0.1 \
  -fa on -np 1 -c 162144 -ctk q4_0 -ctv q4_0 -ub 256 -t 4 --lm-prefill-host-layers 64 --lm-prefill --lm-prefill-fuse --lm-prefill-min 2 \
  --lm-prefill-prearm always --lm-prefill-slots 2 --lm-prefill-ubatch 1024 --lm-prefill-segment 2048 \
  --lm-decode-stream "$(seq -s, 0 2 62)" --lm-decode-slots 2 > build/dense/srv-tgdiag.log 2>&1 &
for i in $(seq 1 400); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; sleep 2; done
( while curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1; sleep 1; done ) > build/dense/vram-tgdiag.csv &
echo healthy
python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 2 --label warm --ignore-eos | head -1
nsys start --session=$SESSION -o build/dense/tg-diag --force-overwrite=true > /dev/null 2>&1
python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 48 --ignore-eos --label capture | tee build/dense/tg-diag-probe.txt | head -1
nsys stop --session=$SESSION > /dev/null 2>&1
fuser -k $PORT/tcp > /dev/null 2>&1
sleep 5
echo "TGDIAG DONE"
