#!/usr/bin/env python3
"""Equivalence of two prefill paths on a running llama-server (dense Qwen3.8 work, contract DQ-2).

  python3 -I tools/dense_equiv.py collect --port 8099 --out build/dense/eq-ref.json     # server A (reference path)
  python3 -I tools/dense_equiv.py collect --port 8099 --out build/dense/eq-lm.json      # server B (path under test)
  python3 -I tools/dense_equiv.py compare build/dense/eq-ref.json build/dense/eq-lm.json

The prompt set is fixed (repo text with a needle at several depths, 1.2-4.3k tokens).  Every request is cold
(cache_prompt false), greedy, n_probs 2.  The comparison rule of contract DQ-2: the first generated token is the same
token, its logprob differs by <= 0.07 nats, and the greedy continuation is identical or its first difference sits where
the reference's top-2 gap is <= 0.25 nats.  Only meaningful for two paths at an EQUAL chunk shape.
"""
import argparse
import json
import sys
import time
import os
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_probe import read_doc  # noqa: E402  (the texts are read from a pinned commit)

SRC = ["docs/plans/generic-prefill-layer-residency-plan.md", "docs/tg-expert-path-reconciliation.md", "docs/currentstate.md", "docs/layer-residency-sdd.md"]
NEEDLE = "\n\nNOTE FOR THE READER: the access code for the vault is 7391-QX. Remember it.\n\n"
QUESTION = "\n\n=== QUESTION ===\nWhat is the access code for the vault mentioned in the note above? Answer with the code only.\nAnswer:"
# (source file index, start offset in chars, length in chars, needle depth)
PROMPTS = [(0, 0, 5200, 0.5), (1, 3000, 6500, 0.2), (2, 10000, 7600, 0.8), (3, 0, 9000, 0.5), (0, 20000, 10500, 0.35), (1, 50000, 12000, 0.65), (2, 40000, 13500, 0.1), (3, 8000, 14500, 0.9)]


def big_prompts(n=24):
    """24 more prompts of 1.5-4.3k tokens for the first-token KL study (deterministic: sources 0-2 cycle, offsets spread over the file)"""
    out = []
    for i in range(n):
        src = i % 3
        ln = 5000 + 400 * (i % 24)
        total = len(read_doc(SRC[src]))
        off = (i * 17321 + 1000) % max(1, total - ln)
        out.append((src, off, ln, [0.1, 0.3, 0.5, 0.7, 0.9][i % 5]))
    return out


# the KL study asks a question whose first token is a real choice (the needle question starts with "\n\n" at probability ~1, which tells nothing about a numerical change)
QUESTION_OPEN = "\n\n=== TASK ===\nIn one sentence, what kind of text is this and what is it mostly about?\nAnswer:\n\n"


def build(i, prompts=None):
    src, off, ln, depth = (prompts or PROMPTS)[i]
    text = read_doc(SRC[src])[off:off + ln]
    cut = int(len(text) * depth)
    return text[:cut] + NEEDLE + text[cut:] + (QUESTION_OPEN if prompts is not None else QUESTION)


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3000) as r:
        return json.load(r)


def collect(a):
    out = []
    prompts = big_prompts() if a.set == "big" else None      # None: the standard 8 prompts with the needle question
    for i in range(min(a.n_prompts, len(prompts or PROMPTS))):
        t0 = time.time()
        d = post(a.port, {"prompt": build(i, prompts), "n_predict": a.n_predict, "temperature": 0, "cache_prompt": False, "n_probs": 20 if a.set == "big" else 2})
        toks = []
        for c in d.get("completion_probabilities", []):
            top = c.get("top_logprobs") or []
            toks.append({"id": c.get("id"), "tok": c.get("token"), "lp": c.get("logprob"),
                         "lp2": top[1]["logprob"] if len(top) > 1 else None, "id2": top[1].get("id") if len(top) > 1 else None})
        t = d["timings"]
        first_top = [(x.get("id"), x.get("logprob")) for x in ((d.get("completion_probabilities") or [{}])[0].get("top_logprobs") or [])]
        out.append({"i": i, "prompt_n": t["prompt_n"], "pp_tps": t["prompt_per_second"], "text": d["content"], "tokens": toks, "first_top": first_top})
        print(f"prompt {i}: {t['prompt_n']:5d} tokens, prefill {t['prompt_ms'] / 1000:6.1f}s ({t['prompt_per_second']:6.1f} tok/s), first tokens {d['content'][:40]!r}  [{time.time() - t0:.0f}s]", flush=True)
    json.dump(out, open(a.out, "w"))


def compare(a):
    ref, tst = json.load(open(a.ref)), json.load(open(a.test))
    bad = 0
    worst_first = 0.0
    for r, t in zip(ref, tst):
        if r["prompt_n"] != t["prompt_n"]:
            print(f"prompt {r['i']}: prompt_n differs ({r['prompt_n']} vs {t['prompt_n']})")
            bad += 1
            continue
        rt, tt = r["tokens"], t["tokens"]
        ok_first = rt[0]["id"] == tt[0]["id"] and abs(rt[0]["lp"] - tt[0]["lp"]) <= 0.07
        d_first = abs(rt[0]["lp"] - tt[0]["lp"]) if rt[0]["id"] == tt[0]["id"] else float("nan")
        worst_first = max(worst_first, d_first) if d_first == d_first else worst_first
        pos = next((k for k in range(min(len(rt), len(tt))) if rt[k]["id"] != tt[k]["id"]), None)
        if pos is None:
            cont = f"continuation identical ({len(rt)} tokens)"
            ok_cont = True
        else:
            gap = rt[pos]["lp"] - rt[pos]["lp2"] if rt[pos]["lp2"] is not None else float("inf")
            ok_cont = gap <= 0.25
            cont = f"first difference at token {pos}: reference gap {gap:.3f} nats -> {'near-tie' if ok_cont else 'FAIL'}"
        flag = "ok " if ok_first and ok_cont else "BAD"
        bad += 0 if ok_first and ok_cont else 1
        print(f"{flag} prompt {r['i']} ({r['prompt_n']:5d} tokens): first token {'same' if rt[0]['id'] == tt[0]['id'] else 'DIFFERENT'}, |dlogprob| {d_first:.4f}; {cont}")
    print(f"{len(ref) - bad}/{len(ref)} prompts meet DQ-2; worst first-token |dlogprob| {worst_first:.4f}")
    return 1 if bad else 0


def kl(a):
    """first-token KL(reference || test) over the reference's top-20 tokens (a test token missing from its top-20 gets the test's smallest top-20 probability), and top-1 agreement"""
    import math
    ref, tst = json.load(open(a.ref)), json.load(open(a.test))
    kls, agree, dl = [], 0, []
    for r, t in zip(ref, tst):
        pr = {i: math.exp(lp) for i, lp in r["first_top"]}
        pt = {i: math.exp(lp) for i, lp in t["first_top"]}
        floor = min(pt.values()) if pt else 1e-9
        z = sum(pr.values())
        k = sum((pr[i] / z) * math.log((pr[i] / z) / (pt.get(i, floor) / z)) for i in pr)
        kls.append(max(k, 0.0))
        agree += r["tokens"][0]["id"] == t["tokens"][0]["id"]
        dl.append(abs(r["tokens"][0]["lp"] - t["tokens"][0]["lp"]) if r["tokens"][0]["id"] == t["tokens"][0]["id"] else float("nan"))
    kls_sorted = sorted(kls)
    print(f"{len(kls)} prompts: mean KL {sum(kls) / len(kls):.5f}, median {kls_sorted[len(kls) // 2]:.5f}, max {max(kls):.5f}; top-1 agrees {agree}/{len(kls)}; max |dlogprob| of the top-1 {max(x for x in dl if x == x):.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--port", type=int, default=8099)
    c.add_argument("--out", required=True)
    c.add_argument("--n-predict", type=int, default=24)
    c.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    c.add_argument("--set", choices=["std", "big"], default="std", help="big: 24 more prompts, first-token top-20 for the KL study")
    p = sub.add_parser("compare")
    p.add_argument("ref")
    p.add_argument("test")
    k = sub.add_parser("kl")
    k.add_argument("ref")
    k.add_argument("test")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a)
        return 0
    if a.cmd == "kl":
        kl(a)
        return 0
    return compare(a)


if __name__ == "__main__":
    sys.exit(main())
