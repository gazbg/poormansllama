#!/usr/bin/env python3
"""Per-layer expert routing skew from rowset_fetch telemetry.

Answers one question: if the N most-frequently-routed experts of every layer
were pinned to VRAM statically, what fraction of decode demand would they
cover?  Compares against the uniform-routing null (N/n_expert), which is what
a load-balancing training loss drives the model toward.

The held-out mode is the decisive one: rank experts on trace A, score coverage
on trace B.  In-sample coverage always beats uniform a little just from finite
sampling noise; only held-out coverage tells you whether the hot set is a
property of the model or a property of the prompt.

Reads the `union_ids` field of `rowset_fetch` events.  TG events only
(n_tokens == 1); prefill chunks route nearly the whole bank and are excluded.
"""

import argparse
import json
import sys
from collections import Counter

N_EXPERT = 256
DEFAULT_NS = (8, 16, 32, 56, 64, 96, 128)


def load_events(path):
    """-> (list of (order, layer, [ids]), run_ids seen).

    `order` is the event's position in the file, which for TG is token order:
    the telemetry is emitted synchronously as each layer of each token runs.
    """
    events = []
    run_ids = set()
    resident = 0
    fetched = 0
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or '"rowset_fetch"' not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") != "rowset_fetch":
                continue
            if ev.get("n_tokens") != 1:  # TG only
                continue
            ids = ev.get("union_ids")
            if not ids:
                continue
            events.append((i, ev.get("layer"),
                           [int(t) for t in ids.split(",") if t.strip()]))
            resident += ev.get("n_rows_resident", 0)
            fetched += ev.get("n_rows_fetched", 0)
            if "run_id" in ev:
                run_ids.add(ev["run_id"])
    load_events.last_residency = (
        resident / (resident + fetched) if (resident + fetched) else None
    )
    return events, run_ids


def tally(events):
    """-> per-layer Counter of expert id."""
    layers = {}
    for _, layer, ids in events:
        c = layers.setdefault(layer, Counter())
        for e in ids:
            c[e] += 1
    return layers


def load_counts(path):
    """-> (per-layer Counter, n_events, run_ids)."""
    events, run_ids = load_events(path)
    return tally(events), len(events), run_ids


def measured_residency(path):
    """Hit rate the run's own cache actually achieved, from the telemetry.

    Cross-check for the coverage numbers: a policy's simulated coverage and the
    real cache's n_rows_resident rate must be the same quantity, or the ladder
    is comparing two different things.
    """
    load_events(path)
    return load_events.last_residency


def coverage(rank_layers, score_layers, n, allocation="uniform"):
    """Fraction of score_layers demand landing in the pinned set of rank_layers.

    Aggregated over layers by total demand, which is the quantity that maps to
    wall time: every covered demand event is one expert FFN that runs on the
    device instead of the host.

    allocation="uniform" pins the top `n` of every layer -- the arrangement a
    fixed per-layer row cache implements.  allocation="global" spends the same
    total budget (n * n_layers rows) on the globally hottest (layer, expert)
    pairs, which is what a shared pool would do if per-layer capacity were free
    to vary.  The gap between them is the value of varying it.
    """
    pinned = pin_sets(rank_layers, n, allocation)
    hit = 0
    total = 0
    for layer, sc in score_layers.items():
        top = pinned.get(layer, frozenset())
        for e, cnt in sc.items():
            total += cnt
            if e in top:
                hit += cnt
    return (hit / total) if total else 0.0


def pin_sets(rank_layers, n, allocation="uniform"):
    """-> {layer: frozenset(expert ids)} under a budget of n rows per layer."""
    if allocation == "uniform":
        # deterministic tie-break on expert id so repeated runs agree
        return {
            layer: frozenset(
                e for e, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
            )
            for layer, c in rank_layers.items()
        }
    if allocation == "global":
        budget = n * len(rank_layers)
        pairs = [
            (cnt, layer, e)
            for layer, c in rank_layers.items()
            for e, cnt in c.items()
        ]
        pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
        out = {}
        for _, layer, e in pairs[:budget]:
            out.setdefault(layer, set()).add(e)
        return {k: frozenset(v) for k, v in out.items()}
    raise ValueError(f"unknown allocation {allocation!r}")


def gini(counter):
    """Concentration of one layer's demand. 0 = perfectly uniform."""
    vals = sorted(counter.get(e, 0) for e in range(N_EXPERT))
    n = len(vals)
    s = sum(vals)
    if s == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(vals))
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def main(argv):
    ap = argparse.ArgumentParser(
        prog="expert-skew",
        description="per-layer expert routing skew and static-placement coverage",
    )
    ap.add_argument("traces", nargs="+", help="rowset telemetry .jsonl paths")
    ap.add_argument(
        "--ns",
        default=",".join(str(n) for n in DEFAULT_NS),
        help="comma-separated experts-per-layer budgets to report",
    )
    ap.add_argument("--json", dest="json_out", default=None, help="machine-readable output path")
    ap.add_argument(
        "--warmups",
        default="16,32,64,128",
        help="comma-separated warm-up token counts for the within-session mode",
    )
    args = ap.parse_args(argv)

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    args.warmups = [int(x) for x in args.warmups.split(",") if x.strip()]

    loaded = []
    for path in args.traces:
        layers, n_events, run_ids = load_counts(path)
        if not layers:
            print(f"!! {path}: no TG rowset_fetch events with union_ids", file=sys.stderr)
            continue
        loaded.append((path, layers, n_events, run_ids))

    if not loaded:
        return 1

    report = {"traces": [], "held_out": []}

    for path, layers, n_events, run_ids in loaded:
        demand = sum(sum(c.values()) for c in layers.values())
        ginis = [gini(c) for c in layers.values()]
        print(f"\n== {path}")
        print(f"   run_id(s): {sorted(run_ids)}   TG events: {n_events}   "
              f"layers: {len(layers)}   demand: {demand}")
        print(f"   gini over layers: min {min(ginis):.4f} / mean "
              f"{sum(ginis)/len(ginis):.4f} / max {max(ginis):.4f}   (0 = uniform)")
        res = measured_residency(path)
        if res is not None:
            print(f"   this run's own cache achieved h = {res:.4f} "
                  f"(n_rows_resident / demand)")
        print("   in-sample static top-N coverage vs uniform null:")
        print("      N     VRAM/40L   per-layer   global   uniform   lift")
        cov_rows = []
        for n in ns:
            cov = coverage(layers, layers, n)
            covg = coverage(layers, layers, n, allocation="global")
            unif = n / N_EXPERT
            mib = 40 * n * 1769472 / (1024 * 1024)
            print(f"   {n:4d}   {mib:8.0f} MiB  {cov:8.4f}  {covg:8.4f}  "
                  f"{unif:8.4f}   {cov/unif:5.2f}x")
            cov_rows.append({"n": n, "coverage": cov, "coverage_global": covg,
                             "uniform": unif, "mib": mib})
        report["traces"].append({
            "path": path,
            "run_ids": sorted(run_ids),
            "n_events": n_events,
            "demand": demand,
            "gini_mean": sum(ginis) / len(ginis),
            "in_sample": cov_rows,
        })

    # held-out: rank on one trace, score on another
    if len(loaded) >= 2:
        print("\n== HELD-OUT static placement (rank on A, score on B)")
        print("   this is the decisive number: in-sample coverage is inflated by")
        print("   finite-sample noise, held-out is not.")
        for i, (pa, la, _, _) in enumerate(loaded):
            for j, (pb, lb, _, _) in enumerate(loaded):
                if i == j:
                    continue
                print(f"\n   rank: {pa}\n   score: {pb}")
                print("      N     per-layer   global   uniform   lift")
                rows = []
                for n in ns:
                    cov = coverage(la, lb, n)
                    covg = coverage(la, lb, n, allocation="global")
                    unif = n / N_EXPERT
                    print(f"   {n:4d}   {cov:9.4f}  {covg:8.4f}  {unif:8.4f}   "
                          f"{cov/unif:5.2f}x")
                    rows.append({"n": n, "coverage": cov,
                                 "coverage_global": covg, "uniform": unif})
                report["held_out"].append({"rank": pa, "score": pb, "rows": rows})

    # within-session: rank on the first `warmup` tokens, score on the rest.
    # This is the cheap engineering middle ground -- one placement decision per
    # session instead of per-token eviction machinery.
    print("\n== WITHIN-SESSION warm-then-pin (rank on first W tokens, score on rest)")
    print("   rank set is chosen once per session, then frozen.")
    for path in args.traces:
        events, _ = load_events(path)
        if not events:
            continue
        n_layers = len(set(e[1] for e in events))
        if not n_layers:
            continue
        print(f"\n   {path}")
        header = "      W    " + "".join(f"  N={n:<7d}" for n in ns)
        print(header)
        rows = []
        for w in args.warmups:
            cut = w * n_layers
            if cut >= len(events):
                continue
            rank = tally(events[:cut])
            score = tally(events[cut:])
            covs = [coverage(rank, score, n) for n in ns]
            covg = [coverage(rank, score, n, allocation="global") for n in ns]
            print(f"   {w:4d}    " + "".join(f"  {c:7.4f}  " for c in covs)
                  + "   global:" + "".join(f"  {c:7.4f}" for c in covg))
            rows.append({"warmup_tokens": w,
                         "coverage": dict(zip((str(n) for n in ns), covs)),
                         "coverage_global": dict(zip((str(n) for n in ns), covg))})
        report.setdefault("within_session", []).append({"path": path, "rows": rows})
    print(f"\n   uniform null:  " + "".join(f"  {n/N_EXPERT:7.4f}  " for n in ns))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
