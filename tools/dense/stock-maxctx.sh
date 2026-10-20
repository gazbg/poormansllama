#!/usr/bin/env bash
# STOCK, default placement (no -ngl, no overrides: llama.cpp's own fit), at the model's maximum context (262144) and at 16384 for comparison.
# TG: 64 tokens after a 670-token prompt, 3 repeats. PP: 3,659 and 15,700 tokens (one cold request each, 64 tokens generated).
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
PORT=8099
export CUDA_VISIBLE_DEVICES=0
run() {
  label=$1; ctx=$2
  build/llama-cuda/bin/llama-server -m $HOME/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf --host 127.0.0.1 --port $PORT \
     -dev CUDA0 -sm none -mg 0 -fa on -np 1 -t 4 --no-webui -c $ctx -ctk q8_0 -ctv q8_0 -ub 512 -v > build/dense/srv-maxctx-$label.log 2>&1 &
  SERVER=$!
  for i in $(seq 1 400); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "$label: server died"; tail -5 build/dense/srv-maxctx-$label.log | cut -c1-250; return; }; sleep 2; done
  ( while kill -0 $SERVER 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1; sleep 1; done ) > build/dense/vram-maxctx-$label.csv &
  python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 4 --label warm > /dev/null
  for r in 1 2 3; do python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 64 --ignore-eos --label "$label TG670 r$r" | head -1; done
  python3 -I tools/dense_probe.py --port $PORT --chars 12000 --n-predict 64 --ignore-eos --label "$label PP3659" | head -1
  python3 -I tools/dense_probe.py --port $PORT --chars 46000 --n-predict 64 --ignore-eos --label "$label PP15700" | head -1
  kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
  awk '{ if ($1 > m) m = $1 } END { printf "   peak VRAM %d MiB\n", m }' build/dense/vram-maxctx-$label.csv
}
run c262144 262144
run c16384 16384
echo MAXCTX DONE
