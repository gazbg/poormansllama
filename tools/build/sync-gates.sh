#!/usr/bin/env bash
# Gates of one upstream-sync waypoint (docs/plans/upstream-sync-plan.md, T-SYNC-04), run under the GPU mutex:
#   pwsh -NoProfile -File tools/with-gpu-lock.ps1 -AllowUnprovenanced -VerdictTag t-sync-<wp>-gates -Command "bash tools/build/sync-gates.sh <wp> <bin dir of the synced build>"
# Arms: synced (this waypoint), control (the fork before the sync, side-copied), upstream (pure ggml-org at the target tag).
# Output: build/llama-cuda/sync-gates/<wp>/ ; compare the test-backend-ops logs with tools/sync-tbo-compare.py.
# Defaults can be overridden through the environment.
cd "$(dirname "$0")/../.." || exit 1
WP=$1
S=$2
C=${SYNC_CONTROL:-build/llama-cuda/bin/sync-control}
U=${SYNC_UPSTREAM_BIN:-/c/Development/AI/llama-up-b12004/build-cuda/bin/Release}
MODEL=${SYNC_MODEL:-C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf}
OUT=build/llama-cuda/sync-gates/$WP
mkdir -p "$OUT"

# Greedy text is NOT a cross-build gate on this model (near-tied logits flip the argmax); perplexity over a few thousand tokens is.
echo "== perplexity, 3 builds  $(date +%T)"
for arm in synced control upstream; do
  case $arm in synced) d=$S;; control) d=$C;; upstream) d=$U;; esac
  "$d/llama-perplexity.exe" -m "$MODEL" -ngl 99 -ncmoe 12 -c 512 -b 512 -ub 512 -f tools/sync-fixtures/ppl-text.txt --chunks 8 > "$OUT/ppl-$arm.txt" 2> "$OUT/ppl-$arm.err"
  echo "$arm exit=$?  $(tr -d '\000' < "$OUT/ppl-$arm.err" | grep -a 'Final estimate' | tail -1 | sed 's/.*Final estimate/Final estimate/')"
done

echo "== test-moe-hybrid-dup-ids (synced)"
"$S/test-moe-hybrid-dup-ids.exe" > "$OUT/dupids.log" 2>&1; echo "exit=$?" >> "$OUT/dupids.log"; tail -2 "$OUT/dupids.log"

echo "== FLASH_ATTN_EXT q8_0 head 256 GQA8 (synced)"
"$S/test-backend-ops.exe" test -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=2,nr23=\[8,1\].*type_K=q8_0' > "$OUT/fa-gqa8.log" 2>&1; echo "exit=$?" >> "$OUT/fa-gqa8.log"; tail -2 "$OUT/fa-gqa8.log"

if [ -z "${SYNC_SKIP_TBO:-}" ]; then   # SYNC_SKIP_TBO=1: intermediate waypoints skip the 5-10 minute full run
  echo "== test-backend-ops full (synced)  $(date +%T)"
  "$S/test-backend-ops.exe" test -b CUDA0 > "$OUT/tbo.log" 2>&1; echo "exit=$?" >> "$OUT/tbo.log"; tail -3 "$OUT/tbo.log"
fi
echo "== done $(date +%T)"
