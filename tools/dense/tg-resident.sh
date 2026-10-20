#!/usr/bin/env bash
# The owner's sanity-check command with the idle VRAM spent on resident layers (plans from tools/dense_plan.py):
#   R162k: -c 162144, host layers 0-45 (18 resident), 24 streamed   predicted 1.58 tok/s (today's all-host plan: 1.24 measured)
#   R65k : -c  65536, host layers 0-34 (29 resident), 20 streamed   predicted 1.98 tok/s
#   R162k15: -c 162144, host layers 0-48 (15 resident), 26 streamed (16 resident missed the guard by 2 MiB: 848 free, 850 wanted)
#   R162k16: -c 162144, host layers 0-47 (16 resident), 26 streamed  predicted 1.53 tok/s (the 18-resident plan left only 232 MiB beyond the ring: the pass cannot run)
# Per arm: warm-up, two 64-token decodes after a 670-token prompt, then a 3,659-token prompt (does the layer-major pass still fit?).
# Log: build/dense/srv-res-*.log, VRAM poller build/dense/vram-res-*.csv
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
PORT=8090
export CUDA_VISIBLE_DEVICES=0 LLAMA_LM_LOG=1
arm() {
  label=$1; ctx=$2; host=$3; stream=$4
  timeout 900 build/llama-cuda/bin/llama-server --model ~/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf --device CUDA0 --port $PORT --host 127.0.0.1 \
    -fa on -np 1 -c $ctx -ctk q4_0 -ctv q4_0 -ub 256 -t 4 --lm-prefill-host-layers "$host" --lm-prefill --lm-prefill-fuse --lm-prefill-min 2 \
    --lm-prefill-prearm always --lm-prefill-slots 2 --lm-prefill-ubatch 1024 --lm-prefill-segment 2048 \
    --lm-decode-stream "$stream" --lm-decode-slots 2 > build/dense/srv-res-$label.log 2>&1 &
  SERVER=$!
  for i in $(seq 1 400); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "$label: server died"; tail -3 build/dense/srv-res-$label.log | cut -c1-200; return; }; sleep 2; done
  ( while kill -0 $SERVER 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1; sleep 1; done ) > build/dense/vram-res-$label.csv &
  echo "== $label: -c $ctx host $host stream $stream"
  python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 2 --label warm --ignore-eos > /dev/null
  python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 64 --ignore-eos --label "$label TG" | head -1
  python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 64 --ignore-eos --label "$label TG (2)" | head -1
  python3 -I tools/dense_probe.py --port $PORT --chars 12000 --n-predict 4 --ignore-eos --label "$label PP3659" | head -1
  echo "   peak VRAM (MiB): $(sort -n build/dense/vram-res-$label.csv | tail -1)"
  echo "   guard declines: $(grep -a -c 'ordinary path' build/dense/srv-res-$label.log)   ring recoveries/suspensions: $(grep -a -c -i 'suspend' build/dense/srv-res-$label.log)"
  kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
  sleep 5
}
arm R162k15 162144 0-48 0,2,4,6,8,10,12,14,16-17,19,21,23,25,27,29,31-32,34,36,38,40,42,44,46,48
echo "RESIDENT DONE"
