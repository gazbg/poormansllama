#!/bin/bash
# Waits until no compute process is on the GPUs, then runs the --gpu-instances validation arms one after another.
# usage: bash tools/pascal/queue-instances.sh   (from the repo root; log in build/pascal/queue-instances.log)
cd "$(dirname "$0")/../.." || exit 1
LOG=build/pascal/queue-instances.log
exec >>"$LOG" 2>&1
echo "queue started $(date)"
flock /tmp/gpu-2x1070.lock true            # waits for a build that holds the lock
mkdir -p build/pascal/final-bin
cp -a build/llama-cuda/bin/*.so* build/llama-cuda/bin/llama-server build/pascal/final-bin/
B="--bin build/pascal/final-bin"
busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
until ! busy; do sleep 60; done
echo "GPUs free $(date)"
run() { # label np flags extra...
  local l=$1 np=$2 f=$3; shift 3
  until ! busy; do sleep 60; done
  echo "== $l $(date)"
  bash tools/pascal/r7.sh "$l" "$np" "$f" $B "$@" > "build/pascal/$l.out" 2>&1
  tail -14 "build/pascal/$l.out" | cut -c1-300
}
run f-c0-np1   1 "--device CUDA0"                 --np-list 1 --deep 16384        # default behaviour unchanged (A4)
run f-inst2-np1 1 "--gpu-instances 2"             --np-list 1,2 --deep 16384
run f-inst2-np2 2 "--gpu-instances 2"             --np-list 1,2,4
run f-inst2-fill 1 "--gpu-instances 2"            --np-list 1 --pp "" --tg 96 --fill 200000,200000
run f-inst2-frozen 1 "--gpu-instances 2 --hybrid-moe-frozen" --np-list 1,2 --check-solo
echo "queue done $(date)"
