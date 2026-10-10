#!/usr/bin/env python3
"""Explain WHERE each demanded expert ran during token generation.

Answers, from the trace alone:
  1. CPU or GPU execution, per layer (ffn_dev / cpu_us).
  2. What happened to experts that were not already resident (staged?).
  3. What the in-flight / wait story was (host_submit_us, transfer_us,
     ready_latency_us).
  4. What happened to a demanded expert that was not mapped anywhere.

Usage: tg-expert-routing.py path.jsonl
"""
import collections
import json
import os
import statistics
import sys


def load(path):
    if not os.path.isfile(path):
        print("MISSING trace %s" % path)
        raise SystemExit(1)
    evs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                evs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return evs


def q(vals, p):
    """p in [0,1]."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


def ms(vals):
    return [v / 1000.0 for v in vals]


def stat(name, vals, scale=1000.0):
    if not vals:
        print("  %-26s (none)" % name)
        return
    v = [x / scale for x in vals]
    print("  %-26s n=%-6d median=%9.3f  mean=%9.3f  p90=%9.3f  max=%9.3f  min=%9.3f"
          % (name, len(v), statistics.median(v), statistics.mean(v),
             q(v, 0.90), max(v), min(v)))


def main(argv):
    if len(argv) != 1:
        print(__doc__)
        return 2
    path = argv[0]
    evs = load(path)

    tg_layer = [e for e in evs if e.get("event") == "layer_compute"
                and e.get("phase") == 1]
    tg_fetch = [e for e in evs if e.get("event") == "rowset_fetch"
                and e.get("phase") == 1]
    tg_round = [e for e in evs if e.get("event") == "rowset_round"]

    if not tg_layer:
        print("NO token-generation layer_compute records -- nothing to explain")
        return 1

    print("=== 1. WHERE EACH LAYER'S EXPERT FFN RAN ===")
    ffn = collections.Counter(e.get("ffn_dev") for e in tg_layer)
    attn = collections.Counter(e.get("attn_dev") for e in tg_layer)
    print("  layer_compute records (TG): %d" % len(tg_layer))
    print("  ffn_dev  : %s   (0=host, 1=device)" % dict(ffn))
    print("  attn_dev : %s" % dict(attn))
    cpu = [e.get("cpu_us", -1.0) for e in tg_layer]
    host = [c for c in cpu if c is not None and c > 0]
    print("  cpu_us>0 (layers that ran host arithmetic): %d of %d" % (len(host), len(cpu)))
    if host:
        stat("  host us/layer", host)
    else:
        print("  -> NO layer executed any expert arithmetic on the CPU.")

    print("")
    print("=== 2. WHAT THE ROUTER ASKED FOR, AND WHAT WAS ALREADY THERE ===")
    fetched = [e["n_rows_fetched"] for e in tg_fetch]
    resident = [e["n_rows_resident"] for e in tg_fetch]
    retained = [e["n_rows_retained"] for e in tg_fetch]
    retired = [e["n_rows_retired"] for e in tg_fetch]
    # n_rows_reused is a per-ROUND field; it is absent on rowset_fetch.
    reused = [e.get("n_rows_reused", 0) for e in tg_fetch]
    topk = collections.Counter(e.get("top_k") for e in tg_fetch)
    union = [e.get("route_union") for e in tg_fetch]
    print("  top_k values: %s ; route_union median=%.1f max=%s"
          % (dict(topk), statistics.median(union), max(union)))
    print("  rows per layer demanded (route_union) totals over %d layers: %d"
          % (len(tg_fetch), sum(union)))
    print("  n_rows_fetched   sum=%-8d median=%.0f max=%d" % (sum(fetched), statistics.median(fetched), max(fetched)))
    print("  n_rows_resident  sum=%-8d max=%d   <- already device-resident"
          % (sum(resident), max(resident) if resident else 0))
    print("  n_rows_retained  sum=%-8d max=%d   <- served from a resident row cache"
          % (sum(retained), max(retained) if retained else 0))
    print("  n_rows_retired   sum=%-8d max=%d" % (sum(retired), max(retired) if retired else 0))
    print("  n_rows_reused    sum=%-8d median=%.0f max=%d   <- reused WITHOUT refetch"
          % (sum(reused), statistics.median(reused), max(reused)))

    print("")
    print("=== 3. THE IN-FLIGHT / WAIT STORY PER LAYER ===")
    stat("host_submit_us", [e["host_submit_us"] for e in tg_fetch])
    stat("transfer_us", [e["transfer_us"] for e in tg_fetch])
    stat("ready_latency_us", [e["ready_latency_us"] for e in tg_fetch])
    stat("bytes_staged MiB", [e["bytes_staged"] for e in tg_fetch], 1048576.0)
    ready_false = sum(1 for e in tg_fetch if e.get("ready_at_consumption") is False)
    ready_present = sum(1 for e in tg_fetch if "ready_at_consumption" in e)
    print("  layers NOT ready at consumption: %d of %d" % (ready_false, len(tg_fetch)))
    if ready_present == 0:
        print("  !! ready_at_consumption is ABSENT from all %d TG rowset_fetch records."
              % len(tg_fetch))
        print("     It lives only on layer_ready_wait, which is emitted for PP only.")
        print("     TG readiness-at-consumption is UNMEASURED -- do not report it.")
    else:
        print("  layers carrying the field: %d of %d" % (ready_present, len(tg_fetch)))

    pp_wait = [e for e in evs if e.get("event") == "layer_ready_wait"
               and e.get("phase") == 0]
    if pp_wait:
        notready = sum(1 for e in pp_wait if e.get("ready_at_consumption") is False)
        print("  [PP, for contrast] layer_ready_wait n=%d, not-ready-at-consumption %d"
              % (len(pp_wait), notready))
        stat("[PP] wait_us", [e["wait_us"] for e in pp_wait])

    if tg_round:
        print("")
        print("=== 4. PER TOKEN (rowset_round) ===")
        f = [e["n_rows_fetched"] for e in tg_round]
        r = [e["n_rows_reused"] for e in tg_round]
        tot = sum(f) + sum(r)
        print("  tokens=%d  rows fetched=%d  reused=%d  reuse rate=%.1f%%"
              % (len(tg_round), sum(f), sum(r), 100.0 * sum(r) / max(1, tot)))
        print("  first round (cold)  : fetched=%d reused=%d"
              % (tg_round[0]["n_rows_fetched"], tg_round[0]["n_rows_reused"]))
        steady_f = f[1:]
        steady_r = r[1:]
        st = sum(steady_f) + sum(steady_r)
        print("  rounds 2..%d (steady): fetched=%d reused=%d reuse rate=%.1f%%"
              % (len(f), sum(steady_f), sum(steady_r),
                 100.0 * sum(steady_r) / max(1, st)))
        stat("bytes_staged/round MiB", [e["bytes_staged"] for e in tg_round], 1048576.0)
        gaps = [tg_round[i + 1]["t_us"] - tg_round[i]["t_us"] for i in range(len(tg_round) - 1)]
        stat("round-to-round ms", gaps)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))