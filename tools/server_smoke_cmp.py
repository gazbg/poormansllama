#!/usr/bin/env python3
"""G32 LM-1: compare four smoke-scenario sessions (A B B A) of tools/server-scenarios/smoke.json: texts, prompt_n / cache_n / predicted_n, PP and TG per step.
  python tools/server_smoke_cmp.py A1 B1 B2 A2        (session tags under build/llama-cuda/)"""
import json
import sys

tags = sys.argv[1:5]
rows = {}
for t in tags:
    for l in open("build/llama-cuda/%s/steps.jsonl" % t, encoding="utf-8"):
        r = json.loads(l)
        rows.setdefault(r.get("name"), {})[t] = r
for name, d in rows.items():
    out = []
    for t in tags:
        r = d.get(t)
        if not r:
            out.append("-")
            continue
        tm = r.get("timings") or {}
        out.append("%.1f/%.1f (n %s/%s/%s)" % (tm.get("prompt_per_second", 0), tm.get("predicted_per_second", 0), tm.get("prompt_n"), tm.get("cache_n"), tm.get("predicted_n")))
    same = len({(d[t].get("text") or d[t].get("content") or "") for t in tags if t in d}) == 1
    cnt = len({tuple((d[t].get("timings") or {}).get(k) for k in ("prompt_n", "cache_n", "predicted_n")) for t in tags if t in d}) == 1
    print("%-10s texts same: %-5s counts same: %-5s | PP/TG %s" % (name, same, cnt, " | ".join(out)))
