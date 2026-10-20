#!/usr/bin/env bash
# Layer-major prefill with ALL KV cache on the GPU at the native maximum context (262144) with q4_0 K and V (4.5 GiB),
# all 64 decoder layers streamed from the host through 2 slots (--lm-prefill-host-layers 64 = every layer's weights on the host, KV on the GPU).
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
PORT=8091
export CUDA_VISIBLE_DEVICES=0 LLAMA_LM_LOG=1
LOG=build/dense/srv-allkv-q4.log
build/llama-cuda/bin/llama-server --model "$HOME/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf" --device CUDA0 \
  -fa on -np 1 -c 262144 -ctk q4_0 -ctv q4_0 -ub 256 -t 4 --host 127.0.0.1 --port $PORT --no-webui \
  --lm-prefill-host-layers 64 --lm-prefill --lm-prefill-fuse --lm-prefill-slots 2 --lm-prefill-ubatch 1024 --lm-prefill-segment 2048 > $LOG 2>&1 &
SERVER=$!
for i in $(seq 1 400); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "server died"; grep -a -E " E |error|failed" $LOG | head -5 | cut -c1-250; exit 1; }; sleep 2; done
echo "== placement"; grep -a -E "KV buffer size|llama_kv_cache: size|lm-prefill: [0-9]+ decoder|compute buffer size" $LOG | head -6 | cut -c1-200
( while kill -0 $SERVER 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1; sleep 1; done ) > build/dense/vram-allkv-q4.csv &
python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 4 --label warm > /dev/null
python3 -I tools/dense_probe.py --port $PORT --chars 12000 --n-predict 16 --label "allKV-q4 PP3659" | head -2
python3 -I tools/dense_probe.py --port $PORT --chars 46000 --n-predict 16 --label "allKV-q4 PP15700" | head -2
python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 32 --ignore-eos --label "allKV-q4 TG670" | head -1
kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
echo "== pass lines"; grep -a -E "lm-prefill: pass [0-9]+:|dense|free|declin|ordinary path" $LOG | sed -E 's/^[0-9.]+ [WIE] //' | cut -c1-230 | head -12
awk '{ if ($1 > m) m = $1 } END { printf "   peak VRAM %d MiB\n", m }' build/dense/vram-allkv-q4.csv
echo ALLKV DONE
