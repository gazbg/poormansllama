#!/usr/bin/env python3
"""Strict identity of two tools/dense_equiv.py collections (host only): same prompt count, same token ids, bit-identical logprobs of every generated token and of the top-2 alternatives.

  python tools/flint-identical.py build/flint/eq-id-ctl-std.json build/flint/eq-id-new-std.json

Exit 0 only if every prompt is identical.  Used by FL-4 (default-off identity of the stock flint path against the pre-change build).
"""
import json
import sys


def main():
    a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    bad = 0
    if len(a) != len(b):
        print(f"prompt counts differ: {len(a)} vs {len(b)}")
        return 1
    for r, t in zip(a, b):
        same = r["prompt_n"] == t["prompt_n"] and r["text"] == t["text"] and [(x["id"], x["lp"], x["lp2"], x["id2"]) for x in r["tokens"]] == [(x["id"], x["lp"], x["lp2"], x["id2"]) for x in t["tokens"]]
        if not same:
            bad += 1
            n = next((i for i, (x, y) in enumerate(zip(r["tokens"], t["tokens"])) if (x["id"], x["lp"]) != (y["id"], y["lp"])), None)
            print(f"prompt {r['i']}: DIFFERENT (prompt_n {r['prompt_n']} vs {t['prompt_n']}, first differing token {n})")
        else:
            print(f"prompt {r['i']}: identical ({r['prompt_n']} prompt tokens, {len(r['tokens'])} generated, first token lp {r['tokens'][0]['lp']:.6f})")
    print(f"{len(a) - bad}/{len(a)} prompts bit-identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
