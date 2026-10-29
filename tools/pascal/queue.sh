#!/usr/bin/env bash
# usage: tools/pascal/queue.sh FILE
# One moe_bench.py arm per line of FILE ("LABEL | options"), run one after the other; prints each arm's result lines.
set -u
cd "$(dirname "$0")/../.." || exit 1
while IFS='|' read -r label opts; do
    label=$(echo "$label" | xargs)
    [ -z "$label" ] && continue
    case "$label" in \#*) continue ;; esac
    eval "python3 -I tools/pascal/moe_bench.py --label $label $opts" > "build/pascal/$label.txt" 2>&1 || { echo "$label failed"; tail -3 "build/pascal/$label.txt"; continue; }
    grep -vE '"(load|end)"' "build/pascal/$label.txt"
done < "$1"
