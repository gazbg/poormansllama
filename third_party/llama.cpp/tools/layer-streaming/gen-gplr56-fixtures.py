#!/usr/bin/env python3
"""T-GPLR-56 A3 fixtures, host-only, no GPU and no model.

The paced mode's telemetry (`span_pace_init`, `span_pace`, `span_pace_stage`,
`span_pace_end`) is emitted only by a real GPU arm, so until one runs the shape of a
PACED trace is proved the way this repo proves every telemetry shape: by deriving it from
a RECORDED trace, keeping every existing event byte-for-byte, and checking it with the
readers that will later judge the arm.

  python tools/layer-streaming/gen-gplr56-fixtures.py --real <recorded-arm.jsonl> --out <paced.jsonl>
      [--groups 40] [--ahead 1] [--unfinished N]

What it does to the recorded stream: the one-shot `mmap_register_init` event is replaced
by `span_pace_init` (that call does NOT happen in paced mode), one `span_pace` +
`span_pace_stage` pair is inserted per group with its issue slot spread evenly over the
events inside the pass-1 window (so the fixture's issue order and coverage are placed
where the mode would place them), and `span_pace_end` closes the commit after the last
event inside that window. Every other line is copied unchanged, so the runner's
`--check-telemetry` still reads a real trace.

  --unfinished N   stop after N groups issued and emit NO `span_pace_end`, i.e. the
                   refusal fixture ("the paced commit is UNFINISHED"): the mode's contract
                   is that no layer is staged before its span is locked, so a partial
                   commit must be refused rather than scored, and that path needs
                   real-shaped data too, not only hand-made fixtures.
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description="T-GPLR-56 paced-trace fixtures from a recorded trace")
    ap.add_argument("--real", required=True, help="recorded arm trace JSONL (carries the markers)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", type=int, default=40)
    ap.add_argument("--ahead", type=int, default=1)
    ap.add_argument("--unfinished", type=int, default=0,
                    help="issue only N groups and omit span_pace_end (0 = a complete commit)")
    a = ap.parse_args()

    events = []
    with open(a.real, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
    if not events:
        print(f"REFUSED: {a.real} has no JSONL events")
        return 2
    reg = [i for i, e in enumerate(events) if e.get("event") == "mmap_register_init"]
    pb = [e for e in events if e.get("event") == "host_mem" and e.get("label") == "pass_begin" and e.get("pass") == 1]
    pe = [e for e in events if e.get("event") == "host_mem" and e.get("label") == "pass_end" and e.get("pass") == 1]
    if not pb or not pe:
        print("REFUSED: the trace has no pass-1 host_mem bracket (run the arm with --pp-recon 1)")
        return 2
    pb, pe = pb[0], pe[0]
    n_groups = a.groups if a.unfinished <= 0 else a.unfinished
    group_bytes = 457322496          # layer 0's measured span (A2); the fixture scales nothing
    run_id = events[0].get("run_id", 1)

    inside = [i for i, e in enumerate(events)
              if e.get("t_us") is not None and pb["t_us"] < e["t_us"] < pe["t_us"]]
    if len(inside) < n_groups:
        print(f"REFUSED: only {len(inside)} events inside the pass-1 window for {n_groups} groups")
        return 2
    slots = {}
    for n in range(n_groups):
        slots[inside[int(n * len(inside) / n_groups)]] = n

    out = []
    for i, e in enumerate(events):
        if i in reg:
            out.append({"event": "span_pace_init", "run_id": run_id, "mode": 1, "ahead": a.ahead,
                        "groups": a.groups, "bytes_total": a.groups * group_bytes,
                        "bytes_issued": group_bytes, "us": 1620.0, "t_us": e["t_us"]})
            continue
        out.append(e)
        if i in slots:
            k = slots[i]
            out.append({"event": "span_pace", "run_id": run_id, "gi": k, "layer": k,
                        "bytes": group_bytes, "members": 3, "merged": 0,
                        "base": 8 * 1024 * 1024 * (k + 1), "issued": k + 1, "total": a.groups,
                        "bytes_unissued": max(0, a.groups - k - 1) * group_bytes, "t_us": e["t_us"]})
            out.append({"event": "span_pace_stage", "run_id": run_id, "layer": k, "covered": 1,
                        "groups_issued": min(a.groups, k + 1 + a.ahead), "groups_total": a.groups,
                        "bytes_unissued": max(0, a.groups - k - 1 - a.ahead) * group_bytes,
                        "t_us": e["t_us"] + 1})
    if a.unfinished <= 0:
        out.append({"event": "span_pace_end", "run_id": run_id, "groups": a.groups,
                    "bytes": a.groups * group_bytes, "t_us": pe["t_us"] + 1})

    with open(a.out, "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")
    print(f"wrote {a.out}: {len(out)} events, {n_groups} groups issued, "
          f"span_pace_end {'ABSENT (unfinished)' if a.unfinished > 0 else 'present'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
