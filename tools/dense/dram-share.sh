#!/usr/bin/env bash
# DRAM bandwidth alone, host-to-device copy alone, and both together (Windows, Git Bash). Run under tools/with-gpu-lock.ps1.
cd "$(dirname "$0")/../.." || exit 1
B=build/dense
for n in 8 16 24 32; do $B/dram-bw.exe $n 4; done
$B/h2d-loop.exe 4 1
for n in 8 16; do
  echo "--- both: dram-bw $n threads + h2d-loop"
  $B/h2d-loop.exe 6 1 > build/dense/win/h2d.out &
  sleep 1
  $B/dram-bw.exe $n 4
  wait
  cat build/dense/win/h2d.out
done
