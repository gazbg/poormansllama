#!/usr/bin/env bash
# usage: capture2.sh LABEL N_CHARS "<server flags>"  -- nsys capture around one cold prompt (a warm-up prompt first)
set -u
LABEL=$1; NCHARS=$2; FLAGS=$3
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
D=build/dense; PORT=8050
export CUDA_VISIBLE_DEVICES=0 LLAMA_LM_LOG=1
timeout 2400 nsys launch --session-new=dn_$LABEL -t cuda build/llama-cuda/bin/llama-server \
    -m $HOME/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf --host 127.0.0.1 --port $PORT \
    -dev CUDA0 -sm none -mg 0 -fa on -np 1 -t 4 --no-webui $FLAGS > $D/server-$LABEL.log 2>&1 &
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; sleep 2; done
python3 -I tools/dense_probe.py --port $PORT --chars 6000 --n-predict 1 --label warm > /dev/null
nsys start --session=dn_$LABEL -o $D/$LABEL --force-overwrite=true > /dev/null 2>&1
python3 -I tools/dense_probe.py --port $PORT --chars $NCHARS --n-predict 1 --label $LABEL
nsys stop --session=dn_$LABEL > /dev/null 2>&1
fuser -k $PORT/tcp > /dev/null 2>&1
sleep 5
echo "CAPTURE DONE $LABEL"
