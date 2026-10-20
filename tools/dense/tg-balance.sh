#!/usr/bin/env bash
# Decode balance test of the owner's sanity-check command (-c 162144, q4_0 KV, host layers streamed through 2 slots):
# the same server with K streamed layers (evenly spread over the 64 layers, tools/dense_plan.spread) for K in "32 28 24".
# Prediction of the nsys capture (build/dense/tg-diag.*): at 32 the copy engine is 99.8% busy (806 ms/token), every layer moved
# from the stream to the CPU saves 25 ms of copy and adds ~19 ms of CPU, so the optimum is near 27-28 layers.
# Per arm: warm-up request, then 64 greedy tokens after a 670-token prompt.  Log: build/dense/srv-bal-K*.log
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
PORT=8090
export CUDA_VISIBLE_DEVICES=0
KS=${KS:-"32 28 24"}
for K in $KS; do
  SPEC=$(python3 -I -c "
import sys; sys.path.insert(0, 'tools')
from dense_plan import spread, fmt_list
print(','.join(map(str, spread(list(range(64)), $K))))")
  timeout 900 build/llama-cuda/bin/llama-server \
    --model ~/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf --device CUDA0 --port $PORT --host 127.0.0.1 \
    -fa on -np 1 -c 162144 -ctk q4_0 -ctv q4_0 -ub 256 -t 4 --lm-prefill-host-layers 64 --lm-prefill --lm-prefill-fuse --lm-prefill-min 2 \
    --lm-prefill-prearm always --lm-prefill-slots 2 --lm-prefill-ubatch 1024 --lm-prefill-segment 2048 \
    --lm-decode-stream "$SPEC" --lm-decode-slots 2 > build/dense/srv-bal-K$K.log 2>&1 &
  SERVER=$!
  for i in $(seq 1 400); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "K=$K: server died"; continue 2; }; sleep 2; done
  python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 2 --label warm --ignore-eos > /dev/null
  echo "K=$K streamed [$SPEC]"
  python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 64 --ignore-eos --label "K=$K TG" | head -1
  python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 64 --ignore-eos --label "K=$K TG (2)" | head -1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1 | sed 's/^/  VRAM now (MiB): /'
  grep -a -c "ordinary path" build/dense/srv-bal-K$K.log | sed 's/^/  guard declines: /'
  kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
  sleep 5
done
echo "BALANCE DONE"
