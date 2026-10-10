#!/usr/bin/env python3
"""Summarise a layer-streaming telemetry trace: phase summaries + TG per-token stats.

The phase `tokens_per_s` is a MEAN and is contaminated by any one slow token
(e.g. a KV view rebuild). The median below is the figure to compare arms on.

Usage: trace-summary.py TAG=path.jsonl [TAG=path.jsonl ...]
"""
import json
import os
import statistics
import sys


def events(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def token_costs(evs):
    """Per-TG-token cost = sum over its 40 layers of (compute_us + setup_us).

    `phase` 1 marks token generation, 0 prefill. `cpu_us` < 0 means the layer
    ran NO host arithmetic (T-GPLR-79's zero-host-compute result).
    """
    per_token = {}
    host_us = []
    ffn_dev = {}
    for e in evs:
        if e.get("event") != "layer_compute" or e.get("phase") != 1:
            continue
        key = e["ubatch"]
        per_token[key] = per_token.get(key, 0.0) + e["compute_us"] + e["setup_us"]
        host_us.append(e.get("cpu_us", -1.0))
        d = e.get("ffn_dev", -1)
        ffn_dev[d] = ffn_dev.get(d, 0) + 1
    return per_token, host_us, ffn_dev


def main(argv):
    if not argv:
        print(__doc__)
        return 2

    rc = 0
    for spec in argv:
        tag, _, path = spec.partition("=")
        if not path:
            path, tag = tag, os.path.basename(tag)
        if not os.path.isfile(path):
            print("%-16s MISSING trace %s" % (tag, path))
            rc = 1
            continue

        evs = list(events(path))
        phases = [e for e in evs if e.get("event") == "phase_summary"]
        per_token, host_us, ffn_dev = token_costs(evs)

        print("=== %s ===" % tag)
        print("  trace : %s" % path)
        if not phases:
            print("  NO phase_summary events -- run died early")
            rc = 1
            continue

        for p in phases:
            wall_s = p["wall_us"] / 1e6
            summed_s = (p["total_transfer_us"] + p["total_wait_us"]
                        + p["total_compute_us"]) / 1e6
            print("  phase %-3s tok=%-6d wall=%9.3fs  mean=%8.3f tok/s" % (
                p["phase"], p["tokens"], wall_s, p["tokens_per_s"]))
            print("        transfer=%8.3fs  wait=%8.3fs  compute=%8.3fs" % (
                p["total_transfer_us"] / 1e6, p["total_wait_us"] / 1e6,
                p["total_compute_us"] / 1e6))
            if p["phase"] == "pp":
                print("        overlap(summed/wall)=%.3fx  evict=%d  feed=%.3f GB/s"
                      % (summed_s / wall_s, p["evictions"],
                         p["bytes_transferred"] / p["total_transfer_us"] / 1e3))
            if p["phase"] == "tg":
                print("        graph_builds=%d  view %d->%d  rebuilds=%d" % (
                    p["graph_builds"], p["kv_view_built"], p["kv_view_live"],
                    p["view_rebuilds_total"]))

        if per_token:
            vals = sorted(per_token.values())
            med = statistics.median(vals) / 1000.0
            n_host = sum(1 for c in host_us if c is not None and c > 0)
            print("  TG per-token n=%d  median=%.3f ms (%.3f tok/s)  mean=%.3f ms  min=%.3f  max=%.3f"
                  % (len(vals), med, 1000.0 / med,
                     statistics.mean(vals) / 1000.0,
                     vals[0] / 1000.0, vals[-1] / 1000.0))
            print("  host-arithmetic layers=%d/%d (cpu_us>0)   ffn_dev=%s"
                  % (n_host, len(host_us), ffn_dev))
        print("")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))