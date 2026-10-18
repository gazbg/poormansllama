#!/usr/bin/env python3
r"""G32 LM-3/LM-4 round 3: replicated hit-versus-fresh and fresh-versus-fresh pairs, so that "the dynamic configuration's hit-versus-fresh expectations fail" can be measured instead of argued.

  python tools/server_hitfresh.py run --config CFG.json --out DIR --prompt FILE[:LEN] [--prompt ...] [--k 2200] [--n-predict 48]
  python tools/server_hitfresh.py analyze LABEL=DIR/hitfresh.json [LABEL=...] [--near-tie 0.25]

run: for every prompt T (token ids, temperature 0, greedy) one server session records
    F1 = fresh (cache_prompt false); P1 = T[:N-k] primes the slot (cache_prompt true, 1 token); H1 = T with cache_prompt true (a cache HIT: the last k tokens are processed on top of the cached prefix);
    F2 = fresh; P2, H2 as above; F3 = fresh.
  It keeps the generated tokens with the top-10 logprobs of every position, and the prompt_n / cache_n of every request (a hit must show cache_n > 0).
analyze: pair classes FF (F1~F2, F1~F3, F2~F3: the SAME request twice in one session = the server's own noise), HF (H1/H2 against F1/F2/F3: the cache-hit continuation against a fresh one: the expectation of the
  multi-turn suites) and HH (H1~H2). Per class: pairs, the first-token top-1 |dlogprob| (median / p90 / max), pairs identical over all tokens, pairs whose first difference is a near tie (< --near-tie nats) and pairs above it
  (a missing candidate token counts as above), and, per prompt, whether ANY pair of the class is above. With two or more labels it prints Fisher exact p-values for "same rate of pairs above the rule" between labels.
  stdlib only; imports tools/server_mt.py (Server)."""
import argparse
import json
import math
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

        def req(toks, cache, n):
            t0 = time.time()
            st, js = srv.api("/completion", {"prompt": toks, "n_predict": n, "temperature": 0, "cache_prompt": cache, "n_probs": a.n_probs, "seed": 1})
            tm = js.get("timings", {}) if isinstance(js, dict) else {}
            return {"status": st, "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"), "pp_tps": tm.get("prompt_per_second"),
                    "steps": top_lists(js) if isinstance(js, dict) else [], "wall": round(time.time() - t0, 2)}

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
            n = len(toks)
            k = a.k
            if a.align_mod:
                # the hit resumes at q = n - k (the end of the primer): choose q = rem (mod align_mod), the largest one with k >= --k
                q = n - a.k
                q -= (q - a.align_rem) % a.align_mod
                k = n - q
            if n <= k + 256 or n - k <= 0:
                print("skip %s: %d tokens is too short for k = %d" % (path, n, k))
                continue
            key = "%s:%d" % (os.path.basename(path), n)
            r = {}
            r["F1"] = req(toks, False, a.n_predict)
            req(toks[:n - k], True, 1)
            r["H1"] = req(toks, True, a.n_predict)
            r["F2"] = req(toks, False, a.n_predict)
            req(toks[:n - k], True, 1)
            r["H2"] = req(toks, True, a.n_predict)
            r["H1"]["resume_q"] = n - k
            r["F3"] = req(toks, False, a.n_predict)
            res[key] = r
            print("%-26s %s" % (key, "  ".join("%s %s/%s" % (k, v["prompt_n"], v["cache_n"]) for k, v in r.items())))
            if any(v["status"] != 200 or not v["steps"] for v in r.values()) or not (r["H1"]["cache_n"] or 0) > 0:
                rc = 1
        json.dump(res, open(os.path.join(a.out, "hitfresh.json"), "w"), indent=1)
    finally:
        time.sleep(1.0)
        alive = srv.alive()
        srv.stop()
        if not alive:
            rc = 1
    return rc


def first_token(a, b):
    """top-1 |dlogprob| of the first generated token (9.99 when A's top-1 token is missing from B's top list)"""
    if not a or not b:
        return None
    ta, tb = a[0]["top"], dict(b[0]["top"])
    if not ta:
        return None
    tok, lp = ta[0]
    return abs(lp - tb[tok]) if tok in tb else 9.99


def first_flip(a, b):
    n = min(len(a), len(b))
    i = next((j for j in range(n) if a[j]["token"] != b[j]["token"]), None)
    if i is None:
        return None
    other = dict(a[i]["top"]).get(b[i]["token"])
    own = a[i]["logprob"]
    return (i + 1, float("inf") if own is None or other is None else abs(own - other))


CLASSES = {"FF": [("F1", "F2"), ("F1", "F3"), ("F2", "F3")],
           "HF": [("F1", "H1"), ("F2", "H1"), ("F3", "H1"), ("F1", "H2"), ("F2", "H2"), ("F3", "H2")],
           "HH": [("H1", "H2")]}


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def fisher(a, b, c, d):
    """two-sided Fisher exact p for [[a, b], [c, d]]"""
    n1, n2, k = a + b, c + d, a + c
    tot = n1 + n2

    def pr(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(tot, k)
    p0 = pr(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9)))


def analyze(a):
    data = {}
    for spec in a.runs:
        label, _, path = spec.partition("=")
        data[label] = json.load(open(path))
    summ = {}
    for label, D in data.items():
        print("== %s: %d prompts" % (label, len(D)))
        hits = [r["H1"]["cache_n"] for r in D.values()] + [r["H2"]["cache_n"] for r in D.values()]
        print("   hit requests with cache_n > 0: %d of %d" % (sum(1 for h in hits if h), len(hits)))
        for cls, prs in CLASSES.items():
            t1s, ident, near, above, per_prompt_above = [], 0, 0, 0, 0
            n = 0
            for key, r in D.items():
                any_above = False
                for x, y in prs:
                    sa, sb = r[x]["steps"], r[y]["steps"]
                    t = first_token(sa, sb)
                    if t is not None:
                        t1s.append(t)
                    f = first_flip(sa, sb)
                    n += 1
                    if f is None:
                        ident += 1
                    elif f[1] < a.near_tie:
                        near += 1
                    else:
                        above += 1
                        any_above = True
                per_prompt_above += 1 if any_above else 0
            summ[(label, cls)] = (n, above, len(D), per_prompt_above)
            print("   %s: %3d pairs | first token top-1 median %.4f p90 %.4f max %.4f | over %d tokens: identical %d, near-tie flip %d, ABOVE %.2f nats %d | prompts with any pair above: %d of %d" % (
                cls, n, sorted(t1s)[len(t1s) // 2], pct(t1s, 0.9), max(t1s), a.n_predict_hint, ident, near, a.near_tie, above, per_prompt_above, len(D)))
    labels = list(data)
    if len(labels) > 1:
        print("== Fisher exact p-values (two-sided) for 'the same share of pairs above the rule', pair class by pair class")
        for cls in CLASSES:
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    n1, a1, _, _ = summ[(labels[i], cls)]
                    n2, a2, _, _ = summ[(labels[j], cls)]
                    print("   %s %-8s vs %-8s: %d/%d against %d/%d  p = %.3f" % (cls, labels[i], labels[j], a1, n1, a2, n2, fisher(a1, n1 - a1, a2, n2 - a2)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--prompt", action="append", required=True)
    r.add_argument("--k", type=int, default=2200)
    r.add_argument("--align-mod", type=int, default=0, help="choose the resume position q = n - k with q = --align-rem (mod this); --k is then the minimum k")
    r.add_argument("--align-rem", type=int, default=0)
    r.add_argument("--n-predict", type=int, default=48)
    r.add_argument("--n-probs", type=int, default=10)
    r.add_argument("--extra-arg", action="append", default=[])
    r.add_argument("--port", type=int, default=18080)
    r.set_defaults(func=run)
    c = sub.add_parser("analyze")
    c.add_argument("runs", nargs="+")
    c.add_argument("--near-tie", type=float, default=0.25)
    c.add_argument("--n-predict-hint", type=int, default=48)
    c.set_defaults(func=analyze)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
