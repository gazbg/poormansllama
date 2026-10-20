#!/usr/bin/env bash
# usage: quick.sh LABEL "<server flags>" PROMPT_CHARS NPRED  -- start, one probe, stop; prints the probe line and the stats
set -u
LABEL=$1; FLAGS=$2; CHARS=${3:-2000}; NP=${4:-16}
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
D=build/dense; PORT=${PORT:-8099}
export CUDA_VISIBLE_DEVICES=0 LLAMA_LM_LOG=1
build/llama-cuda/bin/llama-server -m ${MODEL:-$HOME/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf} --host 127.0.0.1 --port $PORT \
   -dev CUDA0 -sm none -mg 0 -fa on -np 1 -t 4 --no-webui $FLAGS > $D/srv-$LABEL.log 2>&1 &
SERVER=$!
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "$LABEL: server died"; tail -5 $D/srv-$LABEL.log | cut -c1-300; exit 1; }; sleep 2; done
python3 -I tools/dense_probe.py --port $PORT --chars $CHARS --n-predict $NP --label "$LABEL" --out $D/out-$LABEL.json
kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
grep -a -E "dense-stream|lm-decode-stream|error|ERROR|assert|Assert" $D/srv-$LABEL.log | cut -c1-300 | head -12
