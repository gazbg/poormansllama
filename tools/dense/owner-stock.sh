#!/usr/bin/env bash
# The owner's stock command (2026-10-20), one GPU, at the native maximum context 262144.
# Changes from the owner's line: the binary path of this tree, CUDA_VISIBLE_DEVICES=0 (one-GPU rule), --host 127.0.0.1 (local probes only).
# TG: 64 tokens after a 670-token prompt, sampled with the owner's settings, 3 repeats; PP: 3,659 and 15,700 tokens, 64 tokens generated.
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
PORT=8090
export CUDA_VISIBLE_DEVICES=0
LOG=build/dense/srv-owner-stock.log
CUDA_VISIBLE_DEVICES=0 build/llama-cuda/bin/llama-server --model ~/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf --device CUDA0 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --chat-template-kwargs '{"reasoning_effort":"xhigh"}' \
  -fa on -b 1024 -ub 256 -lv 4 --port $PORT --host 127.0.0.1 -ctk q5_1 -ctv q4_0 -np 1 --load-mode none -c 262144 > $LOG 2>&1 &
SERVER=$!
for i in $(seq 1 400); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -5 $LOG | cut -c1-250; exit 1; }; sleep 2; done
( while kill -0 $SERVER 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1; sleep 1; done ) > build/dense/vram-owner-stock.csv &
python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 4 --sampled --label warm > /dev/null
for r in 1 2 3; do python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 64 --ignore-eos --sampled --label "owner TG670 r$r" | head -1; done
python3 -I tools/dense_probe.py --port $PORT --chars 12000 --n-predict 64 --ignore-eos --sampled --label "owner PP3659" | head -1
python3 -I tools/dense_probe.py --port $PORT --chars 46000 --n-predict 64 --ignore-eos --sampled --label "owner PP15700" | head -1
kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
awk '{ if ($1 > m) m = $1 } END { printf "   peak VRAM %d MiB\n", m }' build/dense/vram-owner-stock.csv
echo OWNER DONE
