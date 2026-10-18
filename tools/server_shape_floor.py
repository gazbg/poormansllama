#!/usr/bin/env python3
r"""G32 round 5: the batch-shape floor of the hit-versus-fresh prompts.

The suite's hit-versus-fresh checks (tools/server_mt.py: branch/rollback-vs-fresh, alternate/restore-vs-first, equiv/same-turn*) compare a cache-hit continuation with a from-scratch run of the SAME prompt.
Whatever a server does, the two runs process the prompt in different batch partitions (the cached prefix was computed inside an earlier request's batches), and a server's output moves with its batch shape.
This tool measures how far a server's own FRESH output moves with the batch shape on exactly the prompts of those checks, so a hit-versus-fresh margin can be read against it.

  python tools/server_shape_floor.py run --config CFG.json --out DIR --fixture A.json --fixture2 B.json [--n-predict 24] [--port 18080]
  python tools/server_shape_floor.py analyze LABEL=DIR/floor.json [LABEL=...] [--pair A,B ...] [--group A,B,C]

run: starts the server of CFG, sends the 10 prompts (5 per fixture: the rollback branch, the alternate conversation, equiv turns 1, 3 and the last) FRESH (cache_prompt false, greedy, logprobs, top 5), writes DIR/floor.json.
analyze: for every prompt and every pair prints the first difference (token index and the margin in nats, as the suite scores it) and, with --group, the largest margin inside the group (the floor)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_mt import Server, load_cfg, conv, compare_greedy, first_token_delta, NEAR_TIE  # noqa: E402


def prompts(fx, tag):
    n = len(fx["turns"])
    out = {}
    j = n // 2
    m = conv(fx, j) + [{"role": "assistant", "content": fx["turns"][j]["assistant"]}, {"role": "user", "content": "Result of the last tool call:\nTimeout: the command did not finish."}]
    out["%s/rollback" % tag] = m
    out["%s/alternate" % tag] = conv(fx, min(4, n))
    for k in (1, 3, n):
        k = min(k, n)
        out["%s/equiv-turn%02d" % (tag, k)] = conv(fx, k)
    return out


def run(a):
    cfg = load_cfg(a.config)
    os.makedirs(a.out, exist_ok=True)
    srv = Server(cfg, a.out, a.port, a.extra_arg, dict(kv.split("=", 1) for kv in a.extra_env))
    rc = 0
    res = {}
    try:
        t = srv.start()
        print("server ready after %.1f s" % t)
        fa = json.load(open(a.fixture, "r", encoding="utf-8"))
        fb = json.load(open(a.fixture2, "r", encoding="utf-8"))
        allp = {}
        allp.update(prompts(fa, "A"))
        allp.update(prompts(fb, "B"))
        for name, m in allp.items():
            r = srv.chat(m, a.n_predict, logprobs=True, cache_prompt=False)
            if r["status"] != 200:
                print("%-22s HTTP %s" % (name, r["status"]))
                rc = 1
                continue
            res[name] = {"prompt_n": r["timings"].get("prompt_n"), "tokens": r["tokens"], "content": r["content"][:200]}
            print("%-22s prompt_n %6s first %r" % (name, res[name]["prompt_n"], r["tokens"][0][0] if r["tokens"] else None))
        json.dump(res, open(os.path.join(a.out, "floor.json"), "w"), indent=1)
    finally:
        time.sleep(1.0)
        srv.stop()
    return rc


def cell(ref, cand):
    ident, first, margin, ok = compare_greedy(ref["tokens"], cand["tokens"])
    fd, top1 = first_token_delta(ref["tokens"], cand["tokens"])
    if ident:
        return "=", 0.0, fd
    mm = 9.99 if margin is None else margin
    return "t%d m%.3f" % (first, mm), mm, fd


def analyze(a):
    runs = {}
    for spec in a.runs:
        lab, _, p = spec.partition("=")
        runs[lab] = json.load(open(p))
    labels = list(runs)
    pairs = [tuple(x.split(",")) for x in a.pair] if a.pair else [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    names = list(runs[labels[0]])
    print("first difference between two FRESH runs of the same prompt, as the suite scores it (token index, margin in nats; '=' identical over %d tokens; margin 9.99 = candidate not in the reference's top 5)" % len(runs[labels[0]][names[0]]["tokens"]))
    print("%-22s %7s | %s" % ("prompt", "tokens", " | ".join("%s~%s" % p for p in pairs)))
    for n in names:
        row = []
        for x, y in pairs:
            if n in runs[x] and n in runs[y]:
                row.append(cell(runs[x][n], runs[y][n])[0])
            else:
                row.append("-")
        print("%-22s %7s | %s" % (n, runs[labels[0]][n]["prompt_n"], " | ".join(row)))
    if a.group:
        g = a.group.split(",")
        print("\nfloor inside the group %s: the largest first-difference margin over all pairs of the group, per prompt (rule line %.2f)" % (g, NEAR_TIE))
        for n in names:
            ms = []
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if n in runs[g[i]] and n in runs[g[j]]:
                        ms.append(cell(runs[g[i]][n], runs[g[j]][n])[1])
                        ms.append(cell(runs[g[j]][n], runs[g[i]][n])[1])
            print("%-22s max margin %.3f   pairs at or above the rule: %d of %d" % (n, max(ms) if ms else 0.0, sum(1 for x in ms if x >= NEAR_TIE), len(ms)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--fixture", required=True)
    r.add_argument("--fixture2", required=True)
    r.add_argument("--n-predict", type=int, default=24)
    r.add_argument("--port", type=int, default=18080)
    r.add_argument("--extra-arg", action="append", default=[])
    r.add_argument("--extra-env", action="append", default=[])
    r.set_defaults(func=run)
    c = sub.add_parser("analyze")
    c.add_argument("runs", nargs="+")
    c.add_argument("--pair", action="append", default=[])
    c.add_argument("--group")
    c.set_defaults(func=analyze)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
