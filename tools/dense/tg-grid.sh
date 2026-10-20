#!/usr/bin/env bash
# decode plans: one server per plan, 3 probes of 64 tokens on a ~670-token prompt
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
D=build/dense; PORT=8099
export CUDA_VISIBLE_DEVICES=0 LLAMA_LM_LOG=1
BASE="-c 16384 -ctk q8_0 -ctv q8_0 -ub 512"
run() {
  label=$1; shift
  build/llama-cuda/bin/llama-server -m $HOME/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf --host 127.0.0.1 --port $PORT -dev CUDA0 -sm none -mg 0 -fa on -np 1 -t 4 --no-webui $BASE "$@" > $D/srv-tg-$label.log 2>&1 &
  SERVER=$!
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SERVER 2>/dev/null || { echo "$label: server died"; tail -4 $D/srv-tg-$label.log | cut -c1-250; return; }; sleep 2; done
  ( while kill -0 $SERVER 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1; sleep 1; done ) > $D/vram-tg-$label.csv &
  python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 8 --label warm > /dev/null
  for r in 1 2 3; do python3 -I tools/dense_probe.py --port $PORT --chars 2000 --n-predict 64 --ignore-eos --label "$label/r$r" | head -1; done
  kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
  awk '{ if ($1 > m) m = $1 } END { printf "   peak VRAM %d MiB\n", m }' $D/vram-tg-$label.csv
  grep -a "dense-stream:" $D/srv-tg-$label.log | cut -c1-200
}
S20="0,2,4,6,8,10,12-13,15,17,19,21,23-24,26,28,30,32,34,36"
S18B="1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35"
S16="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30"
S18C="0,2,4,6,8,10-11,13,15,17,19,21-22,24,26,28,30,32"
ATT16="3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63"
ATT8="3,7,11,15,19,23,27,31"
run A-nostream-h35  --lm-prefill-host-layers 0-34
run E-nostream-h38  --lm-prefill-host-layers 0-37
run B-h37-s20-n2    --lm-prefill-host-layers 0-36 --lm-decode-stream $S20 --lm-decode-slots 2
run C-h37-s18-n3    --lm-prefill-host-layers 0-36 --lm-decode-stream $S18B --lm-decode-slots 3
run F-h31-s16-n2    --lm-prefill-host-layers 0-30 --lm-decode-stream $S16 --lm-decode-slots 2
run G-h33-s18-n2    --lm-prefill-host-layers 0-32 --lm-decode-stream $S18C --lm-decode-slots 2
run H-h38-att8-n2   --lm-prefill-host-layers 0-37 --lm-decode-stream $ATT8 --lm-decode-slots 2
run D-owner-att16   --lm-prefill-host-layers 0-63 --lm-decode-stream $ATT16 --lm-decode-slots 2
echo TG GRID DONE
