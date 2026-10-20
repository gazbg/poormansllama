#!/usr/bin/env bash
# usage: bench.sh LABEL "<server flags>" CHARS [CHARS...]   -- starts a server (GPU 0 only), warms it, runs one cold probe per CHARS, stops it
set -u
LABEL=$1; FLAGS=$2; shift 2
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
D=build/dense
PORT=${PORT:-8099}
export CUDA_VISIBLE_DEVICES=0 LLAMA_LM_LOG=1
build/llama-cuda/bin/llama-server -m ${MODEL:-$HOME/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf} --host 127.0.0.1 --port $PORT \
   -dev CUDA0 -sm none -mg 0 -fa on -np 1 -t 4 --no-webui $FLAGS > $D/srv-$LABEL.log 2>&1 &
SERVER=$!
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "$LABEL: server died"; tail -5 $D/srv-$LABEL.log | cut -c1-250; exit 1; }; sleep 2; done
( while kill -0 $SERVER 2>/dev/null; do nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv,noheader,nounits -i 0 | head -1; sleep 1; done ) > $D/vram-$LABEL.csv &
python3 -I tools/dense_probe.py --port $PORT --chars 6000 --n-predict 2 --label warmup > /dev/null
for n in "$@"; do
  python3 -I tools/dense_probe.py --port $PORT --chars $n --n-predict ${NPRED:-16} --ignore-eos --label "$LABEL/$n" --out $D/out-$LABEL-$n.json
done
kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
awk -F', ' '{ if ($1 > m) m = $1; s += $2; n++ } END { printf "   peak VRAM %d MiB, mean SM clock %d MHz\n", m, s/n }' $D/vram-$LABEL.csv
grep -a -E 'lm-prefill: pass [0-9]+:' $D/srv-$LABEL.log | tail -3 | sed -E 's/^[0-9.]+ W //' | cut -c1-160
grep -a -c -E 'no room|takes the ordinary path|declin' $D/srv-$LABEL.log | sed 's/^/   declines logged: /'
