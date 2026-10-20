#!/usr/bin/env bash
# usage: collect.sh LABEL "<server flags>"  -- starts a server (GPU 0), runs tools/dense_equiv.py collect, stops it
set -u
LABEL=$1; FLAGS=$2
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
D=build/dense; PORT=${PORT:-8099}
export CUDA_VISIBLE_DEVICES=0 LLAMA_LM_LOG=1
build/llama-cuda/bin/llama-server -m ${MODEL:-$HOME/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf} --host 127.0.0.1 --port $PORT \
   -dev CUDA0 -sm none -mg 0 -fa on -np 1 -t 4 --no-webui $FLAGS > $D/srv-$LABEL.log 2>&1 &
SERVER=$!
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "$LABEL: server died"; tail -5 $D/srv-$LABEL.log | cut -c1-250; exit 1; }; sleep 2; done
python3 -I tools/dense_equiv.py collect --port $PORT --out $D/eq-$LABEL.json --n-predict ${NPRED:-24} --n-prompts ${NPROMPTS:-8} --set ${SET:-std}
kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
echo "passes: $(grep -a -c 'lm-prefill: pass [0-9]*:' $D/srv-$LABEL.log)"
