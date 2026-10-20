#!/usr/bin/env python3
"""One-request probe of a running llama-server (dense Qwen3.8 work, 2026-10-19).

Builds a prompt of ~N characters of repo text with a needle in it, sends it to /completion with greedy
decoding and prints the server's own timings.  No GPU use of its own.

  python3 -I tools/dense_probe.py --port 8099 --chars 12000 --n-predict 24
  python3 -I tools/dense_probe.py --port 8099 --chars 12000 --out build/dense/a.json   # keep the raw answer

The needle ("the access code for the vault is 7391-QX") sits at --depth of the text; the question asks for it.
"""
import argparse
import json
import sys
import time
import urllib.request

SRC = ["docs/plans/generic-prefill-layer-residency-plan.md", "docs/tg-expert-path-reconciliation.md"]
PIN = "408a1f7"      # the texts are read from this commit (git show), so that editing the docs never changes a prompt of an earlier run


def read_doc(rel):
    """the text of a repo file as of the pinned commit (falls back to the working tree outside a git checkout)"""
    import subprocess
    try:
        return subprocess.run(["git", "show", f"{PIN}:{rel}"], check=True, capture_output=True).stdout.decode("utf-8", errors="replace")
    except Exception:
        return open(rel, encoding="utf-8", errors="replace").read()

NEEDLE = "\n\nNOTE FOR THE READER: the access code for the vault is 7391-QX. Remember it.\n\n"
QUESTION = "\n\n=== QUESTION ===\nWhat is the access code for the vault mentioned in the note above? Answer with the code only.\nAnswer:"


def build_prompt(chars, depth):
    per = chars // len(SRC)
    first, second = (read_doc(p)[:per] for p in SRC)
    text = first + second
    cut = int(len(text) * depth)
    return text[:cut] + NEEDLE + text[cut:] + QUESTION


def post(port, body, timeout=3000):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    d["_wall"] = time.time() - t
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--chars", type=int, default=12000)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--n-predict", type=int, default=24)
    ap.add_argument("--n-probs", type=int, default=0, help="top-N token probabilities for every generated token")
    ap.add_argument("--cache", action="store_true", help="let the server reuse a cached prefix (default: cold)")
    ap.add_argument("--ignore-eos", action="store_true", help="generate n-predict tokens whatever the model says (decode speed)")
    ap.add_argument("--sampled", action="store_true", help="the owner's sampling settings (temp 1.0, top-p 0.95, top-k 20, min-p 0) instead of greedy")
    ap.add_argument("--prompt-file", help="use this text instead of the repo text (the question is NOT added)")
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    prompt = open(a.prompt_file, encoding="utf-8").read() if a.prompt_file else build_prompt(a.chars, a.depth)
    body = {"prompt": prompt, "n_predict": a.n_predict, "temperature": 0, "cache_prompt": a.cache, "ignore_eos": a.ignore_eos}
    if a.sampled:
        body.update({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0})
    if a.n_probs:
        body["n_probs"] = a.n_probs
    d = post(a.port, body)
    t = d["timings"]
    ok = "7391" in d["content"]
    print(f"{a.label:24s} wall {d['_wall']:7.2f}s | prompt_n {t['prompt_n']:6d} cache_n {t.get('cache_n', 0):6d} "
          f"prefill {t['prompt_ms'] / 1000:8.2f}s ({t['prompt_per_second']:7.2f} tok/s) | "
          f"decode {t['predicted_n']:3d} tok @ {t['predicted_per_second']:6.3f} tok/s | needle {'OK' if ok else '--'}")
    print("   answer:", repr(d["content"].strip()[:100]))
    if a.out:
        json.dump(d, open(a.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
