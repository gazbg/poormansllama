#!/usr/bin/env python3
"""Compare matched --kv-graph-pad arms from layer-streaming JSONL telemetry.

The phase_summary `tokens_per_s` field is a MEAN over the TG window, and the TG
window includes the token at which the pad-induced graph rebuild happens. One
measured rebuild token cost 419 ms against a ~52 ms median, which moves the mean
by ~30% and made two runs of the SAME configuration read 12.568 and 18.600
tok/s. A pad-delta claim built on that number is not a measurement of the pad.

This tool therefore reports, per arm:
  * the raw mean (reproducible against phase_summary, so the reader can see what
    the engine itself printed),
  * the MEDIAN per-token latency and its reciprocal, which the rebuild token
    cannot move,
  * a trimmed mean over the fastest 115 of 128 tokens (drops the top 10%),
  * the max token, so contamination is visible rather than hidden,
and the pad-vs-control delta for each statistic.

It also fits the single-parameter model implied by the mechanism -- attention
compute proportional to the live KV view width, everything else width-invariant
-- and reports the implied full-width attention cost implied by each shape. Two
shapes agreeing on that implied constant is a falsifiable cross-check that the
delta really is the attention width and not a session artifact.

Usage:
    python tools/pad-delta.py TAG=path.jsonl [TAG=path.jsonl ...] [--shape 50176]

Exit status is 0 whenever the traces parsed. This is analysis, not a gate: a
non-zero status means the INPUT was unusable, never that the pad helped.
"""

import argparse
import json
import os
import statistics
import sys


def load(path):
    """Return (phase_summaries, run_end, tg_token_us, tg_ids, rebuilds)."""
    phases = {}
    run_end = None
    rounds = []
    ids = None
    rebuilds = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ev = d.get("event")
            if ev == "phase_summary":
                phases[d["phase"]] = d
            elif ev == "run_end":
                run_end = d
            elif ev == "tg_view_rebuild":
                rebuilds.append(d)
            elif ev == "tg_tokens":
                ids = d.get("ids")
            elif ev == "rowset_round" and d.get("n_tokens") == 1:
                rounds.append(d["t_us"])
    latencies = None
    if len(rounds) >= 2:
        # Consecutive single-token rounds are one generation step apart. The TG
        # phase is the tail of the trace, so take the last N intervals.
        deltas = [(rounds[i + 1] - rounds[i]) / 1000.0 for i in range(len(rounds) - 1)]
        latencies = deltas
    return phases, run_end, latencies, ids, rebuilds


def trimmed_mean(xs, keep):
    return sum(sorted(xs)[:keep]) / float(keep)


def attention_model(c_nopad, c_pad, live, full):
    """Split compute into width-proportional attention + width-invariant rest.

    total(live) = A * (live / full) + R, with A and R shared by both arms of a
    pair because only the view width differs.
    """
    frac = live / float(full)
    if frac >= 1.0:
        return None
    a = (c_nopad - c_pad) / (1.0 - frac)
    return a, c_nopad - a


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", metavar="TAG=path.jsonl")
    ap.add_argument("--shape", type=int, default=None,
                    help="live KV width of the pad arm (e.g. 50176). Inferred from telemetry when omitted.")
    ap.add_argument("--full", type=int, default=262144,
                    help="live KV width of the no-pad arm (default: ctx-size).")
    ap.add_argument("--tokens", type=int, default=128, help="TG window length (default 128).")
    ap.add_argument("--control", default=None,
                    help="TAG whose phase_summary is the no-pad control for the model fit.")
    args = ap.parse_args(argv)

    arms = {}
    for spec in args.arms:
        if "=" not in spec:
            ap.error("each arm must be TAG=path.jsonl, got %r" % spec)
        tag, path = spec.split("=", 1)
        if not os.path.exists(path):
            sys.stderr.write(
                "pad-delta.py: error: no such trace for arm %r: %s%s"
                % (tag, path, chr(10)))
            return 1
        arms[tag] = load(path)

    hdr = "%-16s %9s %9s %9s %10s %10s %9s %8s %6s"
    print(hdr % ("arm", "mean_ms", "median_ms", "trim_ms", "tok/s_mean",
                 "tok/s_med", "tok/s_trim", "max_ms", "builds"))
    stats = {}
    for tag in sorted(arms):
        phases, run_end, lat, ids, rebuilds = arms[tag]
        if lat is None:
            print("%-16s  <no single-token rounds>" % tag)
            continue
        tg = lat[-args.tokens:]
        med = statistics.median(tg)
        trim = trimmed_mean(tg, len(tg) - len(tg) // 10)
        stats[tag] = dict(mean=statistics.mean(tg), median=med, trim=trim,
                          max=max(tg), lat=tg, phases=phases, run_end=run_end,
                          ids=ids, rebuilds=rebuilds)
        builds = phases["tg"]["graph_builds"] if "tg" in phases else "?"
        print(hdr % (tag, "%.2f" % stats[tag]["mean"], "%.2f" % med, "%.2f" % trim,
                     "%.2f" % (1000.0 / stats[tag]["mean"]),
                     "%.2f" % (1000.0 / med), "%.2f" % (1000.0 / trim),
                     "%.2f" % stats[tag]["max"], builds))

    print()
    for tag in sorted(stats):
        s = stats[tag]
        for ph in ("pp", "tg"):
            if ph in s["phases"]:
                p = s["phases"][ph]
                print("  %-16s %-3s built=%-7s live=%-7s builds=%-2s rebuilds=%-2s "
                      "compute_ms=%.0f transfer_ms=%.0f wait_ms=%.0f"
                      % (tag, ph, p["kv_view_built"], p["kv_view_live"],
                         p["graph_builds"], p["view_rebuilds_total"],
                         p["total_compute_us"] / 1000.0,
                         p["total_transfer_us"] / 1000.0,
                         p["total_wait_us"] / 1000.0))
        if s["run_end"]:
            print("  %-16s peak_vram=%s exit=%s tokens=%s"
                  % (tag, s["run_end"].get("peak_vram_bytes"),
                     s["run_end"].get("exit_status"),
                     s["run_end"].get("n_tokens_generated")))
        for r in s["rebuilds"]:
            print("  %-16s rebuild built=%s required=%s mechanism=%s"
                  % (tag, r.get("built_n_kv"), r.get("required_n_kv"),
                     r.get("mechanism")))

    # Output identity, when two arms produced token streams.
    if len(stats) > 1:
        print()
        names = sorted(stats)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = stats[names[i]]["ids"], stats[names[j]]["ids"]
                if a is None or b is None:
                    continue
                n = min(len(a), len(b))
                div = next((k for k in range(n) if a[k] != b[k]), None)
                print("  tokens %s vs %s: identical=%s first_divergence=%s (%d/%d equal)"
                      % (names[i], names[j], a == b, div,
                         sum(1 for k in range(n) if a[k] == b[k]), n))

    ctrl = args.control or next((t for t in sorted(stats)
                                 if stats[t]["rebuilds"] == []), None)
    if ctrl is None:
        print()
        print("model: no control arm (one with zero rebuilds) -- skipped")
        return 0
    pads = [t for t in sorted(stats) if t != ctrl]
    print()
    for pad in pads:
        if pad not in stats or ctrl not in stats:
            continue
        live = args.shape
        if live is None:
            live = stats[pad]["phases"].get("tg", {}).get("kv_view_live")
        c_pad = stats[pad]["phases"]["tg"]["total_compute_us"] / 1000.0
        c_ctl = stats[ctrl]["phases"]["tg"]["total_compute_us"] / 1000.0
        if live is None or c_pad is None or c_ctl is None:
            continue
        res = attention_model(c_ctl, c_pad, live, args.full)
        if res is None:
            print("model %s vs %s: live=%d is not narrower than full=%d -- skipped"
                  % (pad, ctrl, live, args.full))
            continue
        a, r = res
        print("model %s vs %s: live=%d (%.2f%% of %d)"
              % (pad, ctrl, live, 100.0 * live / args.full, args.full))
        print("      compute %.0f -> %.0f ms (%.0f ms, %.1f%%)"
              % (c_ctl, c_pad, c_ctl - c_pad, 100.0 * (c_ctl - c_pad) / c_ctl))
        print("      implied full-width attention A=%.0f ms (%.1f%% of no-pad compute),"
              " width-invariant R=%.0f ms" % (a, 100.0 * a / c_ctl, r))
        for k in ("median", "trim", "mean"):
            d = 100.0 * (stats[ctrl][k] / stats[pad][k] - 1.0)
            print("      %-7s latency %+.1f%%  (%s slower than %s)"
                  % (k, d, ctrl, pad))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))