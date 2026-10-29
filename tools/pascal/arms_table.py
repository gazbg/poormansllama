#!/usr/bin/env python3
"""Side-by-side table of single moe_bench.py arms: tok/s, ms per verify round and draft acceptance per measurement, prefill rates, hit rate and cache slots.
  python3 -I tools/pascal/arms_table.py LABEL [LABEL ...]"""
import json, os, sys

root = os.path.join(os.path.dirname(__file__), "..", "..", "build", "pascal")
labels = sys.argv[1:]
res = {}
for lb in labels:
    p = os.path.join(root, lb, "results.jsonl")
    res[lb] = [json.loads(l) for l in open(p)] if os.path.exists(p) else []

def key(r):
    return (r["what"], r.get("chat", r.get("n")))

keys = []
for lb in labels:
    for r in res[lb]:
        if r["what"] in ("pp", "tg", "deep") and key(r) not in keys:
            keys.append(key(r))
w = max(10, max(len(l) for l in labels))
print(f"{'measurement':20s} " + " ".join(f"{l:>{w}s}" for l in labels))
for k in keys:
    rr = [next((r for r in res[l] if key(r) == k), None) for l in labels]
    print(f"{k[0] + ' ' + str(k[1]):20s} " + " ".join(f"{r['tok_s']:>{w}.2f}" if r else f"{'-':>{w}s}" for r in rr))
    if k[0] in ("tg", "deep"):
        mr = [1000.0 * r["n"] / r["tok_s"] / max(1, r["n"] - r["draft_acc"]) if r and r.get("draft_acc") is not None else None for r in rr]
        print(f"{'  ms/round':20s} " + " ".join(f"{v:>{w}.2f}" if v else f"{'-':>{w}s}" for v in mr))
        ac = [r["draft_acc"] / max(1, r["draft_n"]) if r and r.get("draft_n") else None for r in rr]
        print(f"{'  acceptance':20s} " + " ".join(f"{v:>{w}.3f}" if v is not None else f"{'-':>{w}s}" for v in ac))
    if k[0] == "deep":
        print(f"{'  (its prefill)':20s} " + " ".join(f"{r['pp_tok_s']:>{w}.1f}" if r else f"{'-':>{w}s}" for r in rr))
hits, slots, peaks = [], [], []
for l in labels:
    h = s = "-"
    log = os.path.join(root, l, "server.log")
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            if "moe_hybrid: tokens" in line and "hit rate" in line:
                h = line.split("hit rate ")[1].split(" ")[0]
            if "slots per layer (speculative context ready)" in line:
                s = line.split("expert cache ")[1].split(" slots")[0].split("-> ")[-1]
    hits.append(h); slots.append(s)
    en = [r for r in res[l] if r["what"] == "end"]
    peaks.append(str(en[0]["peak_vram_mib"]) if en else "-")
print(f"{'hit rate':20s} " + " ".join(f"{v:>{w}s}" for v in hits))
print(f"{'cache slots':20s} " + " ".join(f"{v:>{w}s}" for v in slots))
print(f"{'peak VRAM MiB':20s} " + " ".join(f"{v:>{w}s}" for v in peaks))
