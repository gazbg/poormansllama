#!/usr/bin/env python3
"""Table of an A-B-B-A session written by tools/pascal/abba.sh: decode and prefill rates per arm, the A and B means, and whether the generated texts agree."""
import json, os, sys

tag = sys.argv[1]
root = os.path.join(os.path.dirname(__file__), "..", "..", "build", "pascal")
arms = ["A1", "B1", "B2", "A2"]
res = {}
for a in arms:
    p = os.path.join(root, f"{tag}-{a}", "results.jsonl")
    res[a] = [json.loads(l) for l in open(p)] if os.path.exists(p) else []

def key(r):
    return (r["what"], r.get("chat", r.get("n")), r.get("rep", 0))

rows = {}
for a in arms:
    for r in res[a]:
        if r["what"] in ("tg", "deep", "pp"):
            rows.setdefault(key(r), {})[a] = r
print(f"{'measurement':24s} " + " ".join(f"{a:>8s}" for a in arms) + "   mean A   mean B   B/A   text")
for k, d in rows.items():
    val = (lambda r: r["tok_s"])
    name = f"{k[0]} {k[1]}" + (f" r{k[2]}" if k[0] == "tg" else "")
    vs = [d[a]["tok_s"] if a in d else float("nan") for a in arms]
    ma = (vs[0] + vs[3]) / 2
    mb = (vs[1] + vs[2]) / 2
    texts = {d[a].get("text") for a in arms if a in d and d[a].get("text")}
    same = "-" if not texts else ("same" if len(texts) == 1 else f"{len(texts)} differ")
    print(f"{name:24s} " + " ".join(f"{v:8.2f}" for v in vs) + f" {ma:8.2f} {mb:8.2f} {mb/ma:6.3f}   {same}")
    if k[0] in ("tg", "deep") and all(a in d and d[a].get("draft_acc") is not None for a in arms):
        # ms per verify round (rounds = tokens - accepted drafts): the step cost, apart from how many drafts the text lets through
        mr = [1000.0 * d[a]["n"] / d[a]["tok_s"] / max(1, d[a]["n"] - d[a]["draft_acc"]) for a in arms]
        acc = [d[a]["draft_acc"] / max(1, d[a]["draft_n"]) for a in arms]
        print(f"{'  ms/round':24s} " + " ".join(f"{v:8.2f}" for v in mr) + f" {(mr[0]+mr[3])/2:8.2f} {(mr[1]+mr[2])/2:8.2f} {(mr[1]+mr[2])/(mr[0]+mr[3]):6.3f}")
        print(f"{'  draft acceptance':24s} " + " ".join(f"{v:8.3f}" for v in acc))
    if k[0] == "deep":
        pv = [d[a]["pp_tok_s"] if a in d else float("nan") for a in arms]
        print(f"{'  (its prefill)':24s} " + " ".join(f"{v:8.1f}" for v in pv) + f" {(pv[0]+pv[3])/2:8.1f} {(pv[1]+pv[2])/2:8.1f} {((pv[1]+pv[2])/(pv[0]+pv[3])):6.3f}")
for a in arms:
    ld = [r for r in res[a] if r["what"] == "load"]
    en = [r for r in res[a] if r["what"] == "end"]
    if ld:
        log = os.path.join(root, f"{tag}-{a}", "server.log")
        hit = slots = ""
        if os.path.exists(log):
            for l in open(log, errors="replace"):
                if "moe_hybrid: tokens" in l and "hit rate" in l:
                    hit = l.split("hit rate ")[1].split(" ")[0]
                if "slots per layer (speculative context ready)" in l:
                    slots = l.split("expert cache ")[1].split(" slots")[0]
        print(f"{a}: hit rate {hit}, cache {slots} slots, peak VRAM {en[0]['peak_vram_mib'] if en else '?'} MiB, libs {[os.path.basename(os.path.dirname(p)) for p in ld[0].get('libs', [])][:1]}")
