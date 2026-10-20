#!/usr/bin/env python3
"""Multi-turn smoke of a running llama-server with the dense Qwen3.8 layer-residency options (contract DQ-1, prompt cache + layer-major pass behind a cached prefix).

  python3 -I tools/dense_multiturn.py --port 8099

Turn 1: ~6k characters of repo text with a needle, then the question.  Turn 2: the SAME text, ~15k more characters, the question again (the
server reuses the cached prefix / restores a context checkpoint and processes the rest, which is big enough for the layer-major pass to take it
at a start position > 0).  Turn 3: turn 2 again (a full cache hit).  Every answer must contain the needle; the cache numbers are printed.
"""
import argparse
import json
import sys
import time
import urllib.request

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from dense_probe import SRC, NEEDLE, QUESTION, read_doc  # noqa: E402


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3000) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    src0 = read_doc(SRC[0])
    src1 = read_doc(SRC[1])
    a_text = src0[:3000] + NEEDLE + src0[3000:6000]
    extra = src1[2000:17000]
    turns = [("turn 1 (cold)", a_text + QUESTION), ("turn 2 (prefix + 15k chars)", a_text + extra + QUESTION), ("turn 3 (repeat of turn 2)", a_text + extra + QUESTION)]
    bad = 0
    for name, prompt in turns:
        t0 = time.time()
        d = post(a.port, {"prompt": prompt, "n_predict": 12, "temperature": 0, "cache_prompt": True})
        t = d["timings"]
        ok = "7391" in d["content"]
        bad += 0 if ok else 1
        print(f"{name:28s} {time.time() - t0:6.1f}s | prompt_n {t['prompt_n']:5d} cache_n {t.get('cache_n', 0):5d} prefill {t['prompt_ms'] / 1000:6.2f}s | needle {'OK' if ok else 'MISSING'} | {d['content'].strip()[:40]!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
