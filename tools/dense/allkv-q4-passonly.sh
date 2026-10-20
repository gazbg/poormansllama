#!/usr/bin/env bash
# All KV on the GPU (q4_0, 262144 ctx), 64 layers streamed through 2 slots, and ONLY the layer-major path for prompts:
# --lm-prefill-min 2 sends every prompt batch of 2 or more tokens through the pass; single-token decode steps stay on the ordinary path.
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
PORT=8092
export CUDA_VISIBLE_DEVICES=0 LLAMA_LM_LOG=1
LOG=build/dense/srv-allkv-q4-passonly.log
build/llama-cuda/bin/llama-server --model "$HOME/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf" --device CUDA0 \
  -fa on -np 1 -c 262144 -ctk q4_0 -ctv q4_0 -ub 256 -t 4 --host 127.0.0.1 --port $PORT --no-webui \
  --lm-prefill-host-layers 64 --lm-prefill --lm-prefill-fuse --lm-prefill-min 2 \
  --lm-prefill-slots 2 --lm-prefill-ubatch 1024 --lm-prefill-segment 2048 > $LOG 2>&1 &
SERVER=$!
for i in $(seq 1 400); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "server died"; exit 1; }; sleep 2; done
( while kill -0 $SERVER 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1; sleep 1; done ) > build/dense/vram-allkv-q4-passonly.csv &
python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 4 --label warm > /dev/null
python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 32 --ignore-eos --cache --label "cold 670 (pass only)" | head -1
python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 32 --ignore-eos --cache --label "repeat 670, cache hit" | head -1
python3 -I tools/dense_probe.py --port $PORT --chars 12000 --n-predict 16 --ignore-eos --cache --label "cold 3659 (pass only)" | head -1
kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
echo "== pass batches (lm-prefill: pass lines):"; grep -a -c "lm-prefill: pass [0-9]*:" $LOG
echo "== batches that did NOT take the pass (reasons):"; grep -a -E "takes the ordinary path" $LOG | sed -E 's/^[0-9.]+ [WIE] //' | cut -c1-200 | head -8
echo "== pass lines:"; grep -a -E "lm-prefill: pass [0-9]+:" $LOG | sed -E 's/^[0-9.]+ [WIE] //' | cut -c1-120
awk '{ if ($1 > m) m = $1 } END { printf "   peak VRAM %d MiB\n", m }' build/dense/vram-allkv-q4-passonly.csv
echo PASSONLY DONE
