#!/usr/bin/env bash
# usage: [ARGS_A="..."] [ARGS_B="..."] tools/pascal/abba.sh TAG BIN_A BIN_B [moe_bench.py options...]
# Runs A, B, B, A (one server each, one at a time) and prints the decode and prefill lines of every arm.
# ARGS_A / ARGS_B: extra moe_bench.py options of that arm only (e.g. ARGS_B="--env GGML_CUDA_GRAPHS_PRE_VOLTA=1").
set -u
TAG=$1; A=$2; B=$3; shift 3
cd "$(dirname "$0")/../.." || exit 1
for arm in A1 B1 B2 A2; do
    case $arm in A*) bin=$A; extra=${ARGS_A:-} ;; *) bin=$B; extra=${ARGS_B:-} ;; esac
    eval "python3 -I tools/pascal/moe_bench.py --label $TAG-$arm --bin $bin $extra \"\$@\"" > "build/pascal/$TAG-$arm.txt" 2>&1 || { echo "$TAG-$arm failed"; tail -3 "build/pascal/$TAG-$arm.txt"; exit 1; }
done
python3 -I tools/pascal/abba_table.py "$TAG"
