#!/usr/bin/env python3
"""Upstream-sync G7-lite: prompt and decode speed of one llama-server configuration, from the server's own timings.

usage: sync-server-timings.py --config CFG.json --out DIR [--prompt FILE:LEN] [--n-predict 128] [--reps 3]

For each rep: one greedy /completion (cache_prompt false) of a prompt of LEN tokens of FILE, n_predict tokens; prints the
server's prompt_per_second and predicted_per_second and writes DIR/timings.json. Run it for the control and the synced
build in the SAME GPU-mutex hold, A B B A, mains power, nothing else running (docs/plans/upstream-sync-plan.md, G7).
stdlib only; imports tools/server_mt.py (Server, load_cfg).
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_mt import Server, load_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="tools/sync-fixtures/ppl-text.txt:1600")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--port", type=int, default=18082)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path, _, ln = a.prompt.rpartition(":")
    srv = Server(load_cfg(a.config), a.out, a.port, [])
    res = []
    try:
        srv.start()
        text = open(path, "r", encoding="utf-8", errors="replace").read()
        st, js = srv.api("/tokenize", {"content": text})
        toks = js["tokens"][:int(ln)]
        for r in range(a.reps):
            st, js = srv.api("/completion", {"prompt": toks, "n_predict": a.n_predict, "temperature": 0,
                                             "cache_prompt": False, "seed": 1, "ignore_eos": True})
            tm = js.get("timings", {}) if isinstance(js, dict) else {}
            res.append({"status": st, "prompt_n": tm.get("prompt_n"), "prompt_ps": tm.get("prompt_per_second"),
                        "predicted_n": tm.get("predicted_n"), "predicted_ps": tm.get("predicted_per_second")})
            print("rep %d: prompt %s tokens %.1f t/s, decode %s tokens %.2f t/s" % (
                r + 1, res[-1]["prompt_n"], res[-1]["prompt_ps"] or 0, res[-1]["predicted_n"], res[-1]["predicted_ps"] or 0))
    finally:
        srv.stop()
    json.dump(res, open(os.path.join(a.out, "timings.json"), "w"), indent=1)
    ok = [x for x in res if x["predicted_ps"]]
    if ok:
        print("median: prompt %.1f t/s, decode %.2f t/s" % (
            statistics.median(x["prompt_ps"] for x in ok), statistics.median(x["predicted_ps"] for x in ok)))


if __name__ == "__main__":
    main()
