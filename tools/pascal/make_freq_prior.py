#!/usr/bin/env python3
"""Expert frequency prior for `--hybrid-moe-freq` (LLAMA_MOE_HYBRID_FREQ) from routing traces of the hybrid engine.

  LLAMA_MOE_HYBRID_TRACE=trace.bin llama-server --hybrid-moe ...   # record while serving the workload
  python3 tools/pascal/make_freq_prior.py trace.bin [more.bin ...] --out freq.txt [--skip-rows N] [--max-rows M]

The file has one line per layer with one integer count per expert (how often it was selected in the rows used). The engine uses it to choose the experts
of the initial cache and to break ties before it has statistics of its own (the first shrink of the cache after a start keeps the popular ones).
Pure Python (no numpy).
"""
import argparse
import struct
import sys


def read_trace(path):
    with open(path, "rb") as fh:
        if fh.read(8) != b"MOETRC01":
            sys.exit(f"{path}: not a MOETRC01 trace")
        n_layer, n_used, n_expert, max_used = struct.unpack("<4i", fh.read(16))
        raw = fh.read()
    rec = 1 + n_layer * max_used
    arr = struct.unpack("<%di" % (len(raw) // 4), raw[: len(raw) // 4 * 4])
    for r in range(len(arr) // rec):
        base = r * rec + 1
        yield [[arr[base + l * max_used + k] for k in range(n_used)] for l in range(n_layer)], n_layer, n_used, n_expert


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-rows", type=int, default=0, help="leave out the first N rows of every trace (for example the chats kept for testing)")
    ap.add_argument("--max-rows", type=int, default=0, help="use at most M rows of every trace after the skipped ones (0: all)")
    a = ap.parse_args()
    counts = None
    used = 0
    for path in a.traces:
        for i, (row, n_layer, n_used, n_expert) in enumerate(read_trace(path)):
            if i < a.skip_rows or (a.max_rows and i >= a.skip_rows + a.max_rows):
                continue
            if counts is None:
                counts = [[0] * n_expert for _ in range(n_layer)]
            for l in range(n_layer):
                for e in row[l]:
                    if e >= 0:
                        counts[l][e] += 1
            used += 1
    if counts is None:
        sys.exit("no rows used")
    with open(a.out, "w") as fh:
        for l in range(len(counts)):
            fh.write(" ".join(str(c) for c in counts[l]) + "\n")
    print(f"{used} rows -> {a.out} ({len(counts)} layers x {len(counts[0])} experts)")


if __name__ == "__main__":
    main()
