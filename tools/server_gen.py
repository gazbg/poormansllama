#!/usr/bin/env python3
r"""G32 LM-3: greedy continuations of N tokens (default 64) of cold prompts, with the top logprobs of every generated token, from one server; and the comparison of two such runs.

  python tools/server_gen.py run --config CFG.json --out DIR --prompt FILE[:LEN] [--prompt ...] [--n-predict 64] [--n-probs 5] [--extra-arg ...]
  python tools/server_gen.py compare REF.json CAND.json [--near-tie 0.25]

run: for every prompt (a prefix of LEN tokens of FILE, sent as token ids, cache_prompt false, temperature 0) it records the generated tokens and, per position, the (token, logprob) list. Writes DIR/gen.json.
compare: per prompt, over the common length, either "identical over N tokens" or the first position where the two continuations differ, with the margin = the reference's logprob of its own token minus its
logprob of the candidate's token (looked up in the reference's top list; ">top-n" when the candidate's token is not in it). A first difference with a margin below --near-tie is a near-tie flip (ok); above is a
FAIL. Exit 1 on a FAIL. stdlib only; imports tools/server_mt.py (Server)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def top_lists(js):
    out = []
    for c in (js.get("completion_probabilities") or []):
        lst = c.get("top_logprobs") or c.get("top_probs") or []
        out.append({"token": c.get("token"), "logprob": c.get("logprob"), "top": [(t.get("token"), t.get("logprob")) for t in lst]})
    return out


def run(a):
    from server_mt import Server, load_cfg
    srv = Server(load_cfg(a.config), a.out, a.port, a.extra_arg)
    res = {}
    rc = 0
    try:
        srv.start()
        for spec in a.prompt:
            path, ln = spec, ""
            head, sep, tail = spec.rpartition(":")
            if sep and tail.isdigit():
                path, ln = head, tail
            text = open(path, "r", encoding="utf-8", errors="replace").read()
            st, js = srv.api("/tokenize", {"content": text})
            toks = js["tokens"]
            if ln:
                toks = toks[:int(ln)]
            t0 = time.time()
            st, js = srv.api("/completion", {"prompt": toks, "n_predict": a.n_predict, "temperature": 0, "cache_prompt": False, "n_probs": a.n_probs, "seed": 1})
            tm = js.get("timings", {}) if isinstance(js, dict) else {}
            key = "%s:%d" % (os.path.basename(path), len(toks))
            res[key] = {"status": st, "prompt_n": tm.get("prompt_n"), "pp_tps": tm.get("prompt_per_second"), "text": js.get("content") if isinstance(js, dict) else None,
                        "steps": top_lists(js) if isinstance(js, dict) else [], "wall": round(time.time() - t0, 2)}
            print("%-26s status %s prompt_n %s pp %.0f t/s, %d tokens generated" % (key, st, tm.get("prompt_n"), tm.get("prompt_per_second") or 0, len(res[key]["steps"])))
            if st != 200 or not res[key]["steps"]:
                rc = 1
        json.dump(res, open(os.path.join(a.out, "gen.json"), "w"), indent=1)
    finally:
        time.sleep(1.0)
        alive = srv.alive()
        srv.stop()
        if not alive:
            rc = 1
    return rc


def compare(a):
    R, C = json.load(open(a.ref)), json.load(open(a.cand))
    rc = 0
    for k in R:
        if k not in C:
            continue
        rs, cs = R[k]["steps"], C[k]["steps"]
        n = min(len(rs), len(cs))
        first = next((i for i in range(n) if rs[i]["token"] != cs[i]["token"]), None)
        if first is None:
            print("%-26s ok   identical over %d tokens" % (k, n))
            continue
        lp = dict(rs[first]["top"])
        own = rs[first]["logprob"]
        other = lp.get(cs[first]["token"])
        if own is None or other is None:
            margin, mtxt = float("inf"), ">top-n (candidate's token is not in the reference's top list)"
        else:
            margin, mtxt = abs(own - other), "%.3f nats" % abs(own - other)
        ok = margin < a.near_tie
        rc |= 0 if ok else 1
        print("%-26s %s %s: first difference at token %d (%r vs %r), margin %s" % (k, "ok  " if ok else "FAIL", "near-tie flip" if ok else "NOT a near tie", first + 1, rs[first]["token"], cs[first]["token"], mtxt))
    return rc


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--prompt", action="append", required=True)
    r.add_argument("--n-predict", type=int, default=64)
    r.add_argument("--n-probs", type=int, default=5)
    r.add_argument("--extra-arg", action="append", default=[])
    r.add_argument("--port", type=int, default=18080)
    r.set_defaults(func=run)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("cand")
    c.add_argument("--near-tie", type=float, default=0.25)
    c.set_defaults(func=compare)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
