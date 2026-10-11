#!/usr/bin/env python3
"""Print the hybrid engine's own per-token timeline (LLAMA_MOE_HYBRID_LOG) for one or more runs.

  python tools/hybrid-timeline.py build/llama-cuda/g1-tl/hybrid_*_r1.jsonl

Fields of the `timeline` record (microseconds per token unless noted):
  span   GPU-side span of one decoded token (first publish to last wait-add)
  stall  part of the span the GPU spent waiting for the host miss engine
  gap    host time BETWEEN the end of one token's span and the start of the next (sampling, launch, input upload)
  miss_layers  mean number of layers per token that had at least one cache miss (40 = every layer waits on the CPU)
  hserve host-engine time serving misses; token time ~ span + gap
Host-only, stdlib-only. It reports; it claims nothing.
"""
import json
import os
import sys


def main(paths):
    print("%-26s %8s %8s %8s %8s %11s %9s %9s" % ("run", "tokens", "span", "stall", "gap", "span+gap", "tok/s", "missLyrs"))
    for p in paths:
        name = os.path.basename(p)
        for pre in ("hybrid_",):
            if name.startswith(pre):
                name = name[len(pre):]
        if name.endswith(".jsonl"):
            name = name[:-6]
        rec = None
        with open(p, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if '"event":"timeline"' in ln:
                    rec = json.loads(ln)
        if rec is None:
            print("%-26s (no timeline record)" % name)
            continue
        tot = rec["span_us"] + max(rec["gap_us"], 0.0)
        print("%-26s %8d %8.0f %8.0f %8.0f %11.0f %9.2f %9.2f" % (
            name, rec["tokens"], rec["span_us"], rec["stall_us"], rec["gap_us"], tot, 1e6 / tot, rec["miss_layers"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) if len(sys.argv) > 1 else print(__doc__) or 2)
