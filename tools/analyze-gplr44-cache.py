#!/usr/bin/env python3
"""T-GPLR-44 step 1: read the `patch_cache` composition dump and report whether
the per-launch patch really walks a whole graph SET while consuming one GRAPH.

Input: a telemetry JSONL produced with --pp-recon 1 (the event is emitted once per
graph set at patch-cache build time).

What the numbers mean:

  set_input_bytes   what one (layer, chunk) launch restores TODAY: every
                    INPUT-flagged tensor of the set.
  layer bytes_in    what a scoped launch for that layer graph would restore.
  redundancy_input  set_input_bytes / mean(layer graph bytes). This is the
                    CEILING on the lever, not a wall-clock prediction: bytes are
                    copied faster than they are attributed, and the FA patch loop
                    (redundancy_fa) is a separate term with a different cost.
  union_matches     closure. Every tensor in the set-level cache must be
                    referenced by at least one graph; if this is 0 the per-graph
                    attribution is incomplete and this dump CANNOT size the
                    lever. Exit code is non-zero in that case, so the falsifier
                    fires mechanically rather than by reading.

T-GPLR-45 fields (absent in pre-T-GPLR-45 traces, which keep the behavior
above): the cache now EXCLUDES zero-consumer members from the restore, and
the dump reports the excluded set so the filter is auditable:

  excluded_bytes / walked_bytes
                    byte identity: kept (set_input_bytes) + excluded must
                    equal the unfiltered candidate walk. A mismatch means the
                    filter created or lost bytes -> exit non-zero.
  excluded_members  each dropped member with graphs=0 BY CONSTRUCTION. An
                    excluded member reporting consumers is the corrupting
                    direction (a restore something needed was dropped) ->
                    exit non-zero. This is the negative control.

Usage:  python tools/analyze-gplr44-cache.py <trace.jsonl>
        python tools/analyze-gplr44-cache.py --selftest
"""

import json
import os
import sys
import tempfile

# Pre-registered in the task contract BEFORE the arm ran, from call structure
# alone: caches are per-set, the launch is per-graph, n_layer layer graphs in a
# set. The band is deliberately wide -- the 8x floor exists because the 10
# full-attention layers' FA patching is not layer-redundant and may dominate
# once the input restore is scoped.
PREDICTED_FLOOR = 8.0
PREDICTED_CEIL = 40.0


def load_events(path):
    events = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    return events


def report(events):
    dumps = [e for e in events if e.get("event") == "patch_cache"]
    print("=" * 78)
    print("T-GPLR-44 step 1: patch-cache composition")
    print("=" * 78)
    if not dumps:
        print("NO patch_cache EVENT in the trace.")
        print("Either the run was not --pp-recon 1, or the dump never fired")
        print("(patch-cache build never ran => prepare_chunk never ran).")
        return 1

    worst_union = 1
    worst_partition = 1
    worst_filter = 1  # T-GPLR-45 byte identity + two-sided closure (1 = ok/absent)
    orphan_names = []
    for d in dumps:
        members = d.get("graph_members") or []
        ng = d.get("n_graphs", 0)
        set_t = d.get("set_input_tensors", 0)
        set_b = d.get("set_input_bytes", 0)
        set_f = d.get("set_fa_nodes", 0)
        union = d.get("union_input_tensors", 0)
        matches = d.get("union_matches", 0)
        worst_union = min(worst_union, 1 if matches else 0)

        print()
        print("set %s  graphs=%s  (0=embed, last=output, %s layer graphs)"
              % (d.get("set"), ng, d.get("layer_graphs")))
        print("  set-level cache      : %s input tensors, %s B, %s FA nodes"
              % (set_t, set_b, set_f))
        print("  closure              : union %s/%s -> union_matches=%s"
              % (union, set_t, matches))
        if "ctx_tensors" in d:
            print("  set ctx              : %s tensors total" % d.get("ctx_tensors"))

        # Name every cached member and how many graphs consume it. The closure
        # failure is only actionable if it says WHICH tensor is unowned.
        # NOTE: distinct from `members` below, which is graph_members. Reusing the
        # name here silently blanked the graph table (caught by --selftest).
        set_members = d.get("set_members") or []
        if set_members:
            orphans = [m for m in set_members if not m.get("graphs")]
            print("  cached members       : %d" % len(set_members))
            for m in sorted(set_members, key=lambda m: -m.get("bytes", 0)):
                flag = "  <-- ORPHAN (no graph consumes this)" if not m.get("graphs") else ""
                print("    %-28s %-9s B  graphs=%s%s"
                      % (m.get("name"), m.get("bytes"), m.get("graphs"), flag))
            if orphans:
                print("  CLOSURE BROKEN by %d member(s): %s"
                      % (len(orphans),
                         ", ".join(m.get("name") for m in orphans)))
                print("  The per-graph numbers below are incomplete: a scoped patch")
                print("  built on them could drop a restore that something else needs.")
            else:
                print("  CLOSURE BROKEN: union %s/%s but no member reports zero refs --"
                      % (union, set_t))
                print("  the attribution counts disagree with the member table.")
        # T-GPLR-45: byte identity (kept + excluded == walked) and the
        # two-sided closure on the EXCLUDED side: an excluded member with
        # consumers means a restore something needed was dropped. Absent
        # fields (pre-T-GPLR-45 traces) leave this check inert.
        if "excluded_bytes" in d or "walked_bytes" in d:
            excl_b = int(d.get("excluded_bytes", 0))
            walked = int(d.get("walked_bytes", 0))
            ident_ok = (int(set_b) + excl_b == walked)
            worst_filter = min(worst_filter, 1 if ident_ok else 0)
            print("  filter identity      : kept %s + excluded %s = %s vs walked %s -> %s"
                  % (set_b, excl_b, int(set_b) + excl_b, walked,
                     "OK" if ident_ok else "BROKEN"))
            if not ident_ok:
                print("  FILTER BROKEN: the restore changed the byte total; the")
                print("  filter created or lost bytes versus the candidate walk.")
            consumed_excluded = [m for m in (d.get("excluded_members") or [])
                                 if m.get("graphs")]
            if consumed_excluded:
                worst_filter = 0
                print("  CORRUPTING EXCLUSION: %d excluded member(s) report consumers: %s"
                      % (len(consumed_excluded),
                         ", ".join(m.get("name") for m in consumed_excluded)))
                print("  A consumed input must never be excluded -- this is the")
                print("  corrupting direction and the trace must not be trusted.")

        print("  layer graph inputs   : min %s  mean %.0f  max %s  B"
              % (d.get("layer_input_bytes_min"), d.get("layer_input_bytes_mean", 0.0),
                 d.get("layer_input_bytes_max")))
        print("  layer graph FA nodes : min %s  mean %.2f  max %s"
              % (d.get("layer_fa_min"), d.get("layer_fa_mean", 0.0), d.get("layer_fa_max")))
        print("  REDUNDANCY (ceiling) : input %.3fx   FA %.3fx"
              % (d.get("redundancy_input", 0.0), d.get("redundancy_fa", 0.0)))

        # The non-layer members are the reason the ceiling is not the prediction:
        # they are single graphs, so a scoped launch cannot cut them by n_layer.
        if members:
            top = sorted(members, key=lambda m: -m.get("bytes_in", 0))[:4]
            print("  largest graphs by bytes_in:")
            for m in top:
                print("    gi=%-3s in=%-4s bytes=%-10s fa=%s"
                      % (m.get("gi"), m.get("layers_in"), m.get("bytes_in"), m.get("fa_nodes")))
            first, last = members[0], members[-1]
            print("  non-layer members    : embed gi=0 (%s B, %s fa) | output gi=%s (%s B, %s fa)"
                  % (first.get("bytes_in"), first.get("fa_nodes"),
                     last.get("gi"), last.get("bytes_in"), last.get("fa_nodes")))
            fa_layers = sum(1 for m in members if m.get("fa_nodes", 0) > 0)
            print("  graphs carrying FA   : %s of %s" % (fa_layers, ng))

            # SPLIT INVARIANT, checkable from the trace with no GPU: the per-graph
            # FA lists must PARTITION the set-level FA cache over the layer graphs
            # (the same range the cache build walks). A split that drops or
            # duplicates a node breaks this equality, and a scoped patch built on
            # it would under- or over-patch attention.
            fa_layer_sum = sum(m.get("fa_nodes", 0) for m in members[1:-1])
            partition_ok = (fa_layer_sum == d.get("set_fa_nodes"))
            worst_partition = min(worst_partition, 1 if partition_ok else 0)
            print("  FA partition         : layer graphs sum to %s vs set cache %s -> %s"
                  % (fa_layer_sum, d.get("set_fa_nodes"), "OK" if partition_ok else "MISMATCH"))
            if not partition_ok:
                print("  SPLIT BROKEN: the per-graph FA lists do not add up to the set")
                print("  cache, so a scoped patch would drop or double-patch nodes.")

    print()
    print("-" * 78)
    r = dumps[-1].get("redundancy_input", 0.0)
    if not worst_union:
        print("VERDICT: closure broken -> the lever cannot be sized from this dump.")
        print("         Task closes as falsified-instrumentation, not as a patch.")
    elif not worst_partition:
        print("VERDICT: split invariant FAILED -> the per-graph FA lists do not")
        print("         partition the set cache. Do not trust the patch: report it.")
    elif not worst_filter:
        print("VERDICT: T-GPLR-45 filter invariant FAILED (byte identity or a")
        print("         consumed-but-excluded member). Do not trust the restore: report it.")
    elif PREDICTED_FLOOR <= r <= PREDICTED_CEIL:
        print("VERDICT: composition CONFIRMS the lever's premise "
              "(input redundancy %.1fx inside the pre-registered %.0f-%.0fx band)."
              % (r, PREDICTED_FLOOR, PREDICTED_CEIL))
    elif r > PREDICTED_CEIL:
        print("VERDICT: redundancy %.1fx is ABOVE the pre-registered band -- the"
              % r)
        print("         set holds more than one graph's worth per layer; report as")
        print("         measured, do not round it into the band.")
    else:
        print("VERDICT: redundancy %.1fx is BELOW the pre-registered %.0fx floor."
              % (r, PREDICTED_FLOOR))
        print("         Either the set is not as shared as the call structure")
        print("         suggested, or the caches were already scoped. The lever is")
        print("         small or absent: report and do not implement.")
    print("-" * 78)
    return 0 if (worst_union and worst_partition and worst_filter) else 1


def selftest():
    """A parse/report bug must not be discovered after an arm has been spent, so
    the analyzer is exercised on synthetic dumps: one well-formed with closure,
    one with closure BROKEN (which must exit non-zero), and one empty."""
    ok = 0
    tmpdir = tempfile.mkdtemp(prefix="gplr44-selftest-")

    def write(name, events):
        p = os.path.join(tmpdir, name)
        with open(p, "w", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")
        return p

    # 3 graphs (embed / one layer / output) so the FA partition invariant is
    # exercisable: the layer graph's FA count must equal the set cache's.
    good = {
        "event": "patch_cache", "set": 0, "n_graphs": 3, "layer_graphs": 1,
        "ctx_tensors": 90000,
        "set_members": [
            {"name": "inp_tokens", "bytes": 2048, "graphs": 42},
            {"name": "KQ_mask", "bytes": 4000000, "graphs": 42},
            {"name": "inp_pos", "bytes": 2048, "graphs": 1},
        ],
        "set_input_tensors": 12, "set_input_bytes": 4000000, "set_fa_nodes": 200,
        "union_input_tensors": 12, "union_matches": 1,
        # T-GPLR-45: the zero-consumer filter's audit fields. The kept bytes
        # plus the excluded bytes must equal the unfiltered walk, and every
        # excluded member must report graphs=0 BY CONSTRUCTION.
        "excluded_count": 1, "excluded_bytes": 4194304, "walked_bytes": 8194304,
        "excluded_members": [{"name": "inp_embd", "bytes": 4194304, "graphs": 0}],
        "layer_input_bytes_min": 90000, "layer_input_bytes_max": 110000,
        "layer_input_bytes_mean": 100000.0,
        "layer_fa_min": 4, "layer_fa_max": 6, "layer_fa_mean": 5.0,
        "redundancy_input": 40.0, "redundancy_fa": 40.0,
        "graph_members": [
            {"gi": 0, "layers_in": 3, "bytes_in": 4096, "fa_nodes": 0},
            {"gi": 1, "layers_in": 12, "bytes_in": 100000, "fa_nodes": 200},
            {"gi": 2, "layers_in": 2, "bytes_in": 8192, "fa_nodes": 0},
        ],
    }
    # The closure-broken case must exercise the NAMING path too: an orphan member
    # is the actionable form of the failure, so the fixture carries one.
    broken_members = [dict(m) for m in good["set_members"]]
    broken_members[1]["graphs"] = 0
    broken = dict(good, union_input_tensors=9, union_matches=0, set_members=broken_members)
    # A split that drops an FA node must also be caught: same well-formed dump,
    # one node missing from the only layer graph.
    dropped_members = [dict(m) for m in good["graph_members"]]
    dropped_members[1]["fa_nodes"] = 199
    split_broken = dict(good, graph_members=dropped_members)
    low = dict(good, redundancy_input=2.0, union_matches=1)
    # T-GPLR-45 negative controls. (a) Byte identity: kept + excluded must
    # equal walked -- a mismatch means the filter created or lost bytes.
    ident_broken = dict(good, walked_bytes=8194305)
    # (b) The corrupting direction: an EXCLUDED member that reports consumers
    # means a restore something needed was dropped. The selftest must FAIL on
    # it -- this is the control that would have caught a direct-src walk
    # silently excluding a view-consumed input (the T-GPLR-44 closure bug).
    corrupt = dict(good, excluded_members=[{"name": "inp_embd", "bytes": 4194304, "graphs": 2}])

    cases = [
        ("closure intact", [good], 0),
        ("closure broken", [broken], 1),
        ("split dropped FA", [split_broken], 1),
        ("no dump at all", [{"event": "host_ledger"}], 1),
        ("filter identity", [ident_broken], 1),
        ("corrupt exclusion", [corrupt], 1),
    ]
    for label, events, want in cases:
        path = write(label.replace(" ", "-") + ".jsonl", events)
        got = report(load_events(path))
        verdict = "OK" if got == want else "BAD"
        if got == want:
            ok += 1
        print("%-16s exit=%d want=%d  %s" % (label, got, want, verdict))
        print()

    # Redundancy below the band must still exit 0 (it is a real measurement, not
    # a broken instrument) but must say so.
    path = write("low.jsonl", [low])
    got = report(load_events(path))
    if got == 0:
        ok += 1
    print("%-16s exit=%d want=0  %s" % ("below band", got, "OK" if got == 0 else "BAD"))

    print()
    total = len(cases) + 1  # + the below-band case
    print("selftest: %d/%d" % (ok, total))
    return 0 if ok == total else 1


def main(argv):
    if len(argv) >= 2 and argv[1] == "--selftest":
        return selftest()
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    if not os.path.exists(path):
        print("MISSING TRACE: %s" % path)
        return 2
    return report(load_events(path))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
