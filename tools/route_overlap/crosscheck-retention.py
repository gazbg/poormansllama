#!/usr/bin/env python3
"""Cross-check the runtime retention rule against the analyzer's simulation.

Host-only, no GPU. This is the anti-drift check for T-GPLR-19's per-layer row
retention: two independent implementations of the same rule exist --

  A. the RUNTIME + C++ checker:
       third_party/llama.cpp/src/llama-layer-streaming.cpp  load_row_set()
       third_party/llama.cpp/tools/layer-streaming/main.cpp ls_retention_errors()

  B. the OFFLINE analyzer:
       tools/route_overlap/analyze.py  simulate_retention(policy=kwindow-union)

and nothing previously forced them to agree. This script replays ONE shared
recorded stream through both and fails loudly on any disagreement.

Both are the same rule, transcribed once here from the C++ and once in the
analyzer. POLICY RESOLVED 2026-09-13: the rule is a ROW-CAPACITY LRU, which is
the analyzer's default policy and the one SDD gate 9's statistic (b) measures:

  keep the cap = min(n_expert, K*W*top_k) most recently ROUTED rows per layer.
  Before placing a window, evict the least recently routed row that this
  window does not need, until the fetched complement fits; then stamp recency
  across the whole union in ascending expert-id order.

The alternative -- a window-count membership set, "keep exactly the rows routed
in the last K windows" -- is still available as the analyzer's kwindow-union
policy and is printed for comparison. It has the SAME footprint (cap rows per
layer) and weakly LOSES to the LRU: a row routed in the last K windows has at
most cap-1 rows routed after it, so it is always among the cap most recently
routed. The two agree exactly at K=1 and diverge only at K>=2.

Exit status is driven only by the equivalence that MUST hold: the runtime rule
against the analyzer policy named by --require-policy, at every K, plus the
window sequence identity that makes the comparison meaningful at all.

Usage:
    python3 tools/route_overlap/crosscheck-retention.py TRACE.jsonl [TRACE...]
    python3 tools/route_overlap/crosscheck-retention.py --retention-k 1,2,4 \
        --checker build/llama-cuda/bin/Release/llama-layer-streaming.exe TRACE
"""
import argparse
import io
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze  # noqa: E402

PROG = "crosscheck-retention.py"
EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_USAGE = 2

# The preserved 256-token batch-major probe: the only recorded stream with a
# full 10,200 pair-per-layer set (ladder rungs carry 2 pairs per layer, which is
# cross-validation, never the verdict).
DEFAULT_TRACE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "build" / "llama-cuda" / "gplr15-phaseA-probe" / "probe-tg256.jsonl"
)


def load_windows(path):
    """Per-layer [(ubatch, chunk, ids)] in file order, plus the trace's init."""
    by_layer = {}
    init = None
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("event") == "rowset_init":
            init = ev
        elif ev.get("event") == "rowset_fetch":
            by_layer.setdefault(ev["layer"], []).append(
                (ev["ubatch"], ev["chunk"], analyze._ids_of(ev))
            )
    return by_layer, (init or {})


def runtime_rule(windows, cap):
    """Replay the C++ rule. Returns per-window (ru, retained, retired).

    Transcribed from llama-layer-streaming.cpp load_row_set: classify the union
    against the cache, evict the least recently routed non-union position until
    the fetched complement fits, then stamp recency across the whole union in
    ascending expert-id order. `cap` is row_set_cap = min(n_expert, K*top_k).
    """
    resident = {}   # expert -> recency stamp
    tick = 0
    out = []
    for ubatch, _chunk, ids in windows:
        uniq = sorted(set(ids))          # the C++ uniq is sorted ascending
        uset = set(uniq)
        retained = sum(1 for x in uniq if x in resident)
        need = len(uniq) - retained
        retired = 0
        while len(resident) + need > cap:
            victims = [x for x in resident if x not in uset]
            if not victims:
                raise AssertionError(
                    f"runtime rule cannot fit the union at ubatch {ubatch}: "
                    f"{len(uniq)} rows > cap {cap}"
                )
            del resident[min(victims, key=resident.get)]
            retired += 1
        for x in uniq:
            tick += 1
            resident[x] = tick
        if len(resident) > cap:
            raise AssertionError(
                f"runtime rule exceeded capacity at ubatch {ubatch}: "
                f"{len(resident)} > {cap}"
            )
        out.append((len(uniq), retained, retired))
    return out


def check_plumbing(by_layer, series):
    """The comparison is only meaningful if BOTH index windows the same way.

    The analyzer scores windows in `chunk` order and treats position 0 as the
    cold window; the runtime uses epoch = ubatch - 1. If chunk != ubatch - 1, or
    chunks are not consecutive per layer, the two are not describing the same
    window sequence and every number below would be comparing different things.
    """
    errors = []
    if sorted(by_layer) != sorted(series):
        errors.append("layer sets differ between the trace and the analyzer series")
    for layer, wins in sorted(by_layer.items()):
        for pos, (ubatch, chunk, ids) in enumerate(wins):
            if chunk != ubatch - 1:
                errors.append(
                    f"layer {layer} position {pos}: chunk {chunk} != ubatch-1 "
                    f"({ubatch - 1}) -- the runtime and analyzer index windows "
                    f"differently"
                )
        chunks = [c for _u, c, _i in wins]
        if chunks != sorted(chunks) or chunks != list(
            range(chunks[0], chunks[0] + len(chunks))
        ):
            errors.append(f"layer {layer}: chunks are not consecutive ({chunks[:6]}...)")
        ser = series.get(layer, [])
        if len(ser) != len(wins):
            errors.append(f"layer {layer}: window count {len(wins)} vs series {len(ser)}")
            continue
        for pos, ((_u, c, ids), (sc, sids, _h)) in enumerate(zip(wins, ser)):
            if c != sc or ids != sids:
                errors.append(f"layer {layer} position {pos}: window identity differs")
                break
    return errors


def emit_fixture(path, src_lines, retain_k, slabs, cap):
    """Write the C++ checker's view of this stream under the runtime rule.

    Only rowset_init's retain_k and each fetch's retention fields are rewritten;
    everything else is byte-preserved so the checker validates the rule and not
    an artifact of fixture generation.
    """
    # Keyed by (layer, expert), exactly as the checker's rowset_cache is keyed
    # by layer<<32|expert. A single flat expert key would let one layer's rows
    # satisfy an earlier layer's first window, which the checker rejects --
    # correctly, since residency is per logical layer.
    resident = {}
    tick = 0
    out = io.StringIO()
    for line in src_lines:
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("event") == "rowset_init":
            ev["retain_k"] = retain_k
        elif ev.get("event") == "rowset_fetch":
            layer = ev["layer"]
            uniq = sorted(set(analyze._ids_of(ev)))
            uset = set(uniq)
            retained = sum(1 for x in uniq if (layer, x) in resident)
            need = len(uniq) - retained
            retired = 0
            mine = [k for k in resident if k[0] == layer]
            while len(mine) + need > cap:
                victims = [k for k in mine if k[1] not in uset]
                if not victims:
                    raise AssertionError(
                        f"emitter cannot fit the union at ubatch {ev['ubatch']}"
                    )
                victim = min(victims, key=resident.get)
                del resident[victim]
                mine.remove(victim)
                retired += 1
            for x in uniq:
                tick += 1
                resident[(layer, x)] = tick
            s = uset
            ev["n_rows_retained"] = retained
            ev["n_rows_retired"] = retired
            ev["n_rows_resident"] = retained
            ev["n_rows_fetched"] = len(s) - retained
            ev["bytes_staged"] = (len(s) - retained) * slabs[ev["layer"]]
        out.write(json.dumps(ev, separators=(",", ":")) + "\r\n")
    pathlib.Path(path).write_text(out.getvalue(), encoding="utf-8", newline="")
    return path


def slab_sizes(lines):
    """Per-layer expert slab bytes from the trace (they are NOT uniform)."""
    slabs = {}
    for line in lines:
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("event") == "rowset_fetch" and ev["n_rows_fetched"]:
            n = ev["bytes_staged"] // ev["n_rows_fetched"]
            slabs.setdefault(ev["layer"], set()).add(n)
    for layer, ns in slabs.items():
        if len(ns) != 1:
            raise AssertionError(f"layer {layer} has {len(ns)} slab sizes: {sorted(ns)}")
    return {k: v.pop() for k, v in slabs.items()}


def crosscheck(path, ks, checker=None, fixture_dir=None,
               require_policy=analyze.RETENTION_POLICY_UNION):
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines(True)
    by_layer, init = load_windows(path)
    n_expert = init.get("n_expert") or 256
    top_k = init.get("top_k") or 8
    window_tokens = init.get("window_tokens") or 1
    slot_bytes = init.get("slot_bytes") or 0
    row_bytes = int(slot_bytes // n_expert) if (n_expert and slot_bytes) else 0
    fetch = []
    for line in lines:
        if line.strip():
            ev = json.loads(line)
            if ev.get("event") == "rowset_fetch":
                fetch.append(ev)
    series = analyze.rowset_series(fetch)

    print(f"== {pathlib.Path(path).name}: {len(fetch)} fetches, "
          f"{len(series)} layers, {sorted({len(w) for w in series.values()})} "
          f"windows/layer, top_k={top_k}, W={window_tokens}")

    errors = check_plumbing(by_layer, series)
    for e in errors:
        print(f"  PLUMBING  {e}")
    if not errors:
        print("  plumbing  OK: analyzer window sequence == runtime epoch sequence")

    if not series:
        print("  no rowset_fetch windows: nothing to compare")
        return False
    has_predecessor = len(fetch) > len(series)
    if not has_predecessor:
        print("  every layer has a single window: no window with a predecessor")

    ok = not errors
    print(f"  asserting equivalence against policy: {require_policy}")
    for k in ks:
        cap = k * window_tokens * top_k
        if n_expert:
            cap = min(n_expert, cap)
        # The analyzer scores windows-with-a-predecessor per layer; the runtime
        # scores the cold window too (0 hits, and it is the window that places
        # the first rows). Drop the first window per layer before comparing.
        rt_scored = []
        for _layer, wins in sorted(by_layer.items()):
            rt_scored.extend(runtime_rule(wins, cap)[1:])
        rt_hits = sum(r for _ru, r, _rt in rt_scored)
        rt_req = sum(ru for ru, _r, _rt in rt_scored)
        rt_value = rt_hits / rt_req if rt_req else 0.0

        def curve(policy):
            return analyze.resident_containment(
                series, (k,), n_expert, top_k, window_tokens, row_bytes, policy
            )[k]

        union = curve(analyze.RETENTION_POLICY_UNION)
        lru = curve(analyze.RETENTION_POLICY_LRU)
        target = union if require_policy == analyze.RETENTION_POLICY_UNION else lru

        agree = (rt_hits == target["hits_total"]
                 and rt_req == target["required_total"])
        ok = ok and agree
        print(
            f"  K={k} cap={cap} rows/layer "
            f"| runtime rule {rt_hits}/{rt_req} = {rt_value:.6f}"
            f" | analyzer {require_policy} {target['hits_total']}/"
            f"{target['required_total']} = {target['value']:.6f}"
            f" -> {'EQUAL' if agree else 'DRIFT'}"
        )
        other = lru if require_policy == analyze.RETENTION_POLICY_UNION else union
        print(
            f"      other policy {'row-capacity-lru' if require_policy == analyze.RETENTION_POLICY_UNION else 'kwindow-union'}"
            f" = {other['value']:.6f} ({other['hits_total']}/{other['required_total']})"
            f", spread vs the asserted policy {other['value'] - target['value']:+.6f}"
            + (" [row-capacity-lru is both the implemented and the published"
               " policy]" if require_policy == analyze.RETENTION_POLICY_LRU else
               " [kwindow-union is the superseded rule; this gap is by design]")
        )

    if checker:
        checker = str(pathlib.Path(checker).resolve())
        if not pathlib.Path(checker).exists():
            print(f"  checker binary not found: {checker}")
            return False
        fixture_dir = pathlib.Path(fixture_dir or "build/gplr19-checker")
        fixture_dir.mkdir(parents=True, exist_ok=True)
        slabs = slab_sizes(lines)
        for k in ks:
            fx = fixture_dir / f"crosscheck-k{k}.jsonl"
            cap = k * window_tokens * top_k
            if n_expert:
                cap = min(n_expert, cap)
            emit_fixture(fx, lines, k, slabs, cap)
            proc = subprocess.run(
                [checker, "--check-telemetry", str(fx.resolve())],
                capture_output=True, text=True,
            )
            accepted = proc.returncode == 0
            ok = ok and accepted
            print(
                f"  K={k} C++ checker on {fx.name}: exit {proc.returncode} "
                f"-> {'ACCEPTED' if accepted else 'REJECTED'}"
            )
            if not accepted:
                for l in proc.stderr.splitlines()[:3]:
                    print(f"        {l}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog=PROG,
        description="Cross-check the runtime retention rule against the "
        "analyzer's offline K-window simulation on one shared recorded stream.",
    )
    ap.add_argument("traces", nargs="*", default=None)
    ap.add_argument(
        "--retention-k",
        default="1,2",
        help="comma-separated K values to compare (default 1,2)",
    )
    ap.add_argument(
        "--checker",
        default=None,
        help="llama-layer-streaming binary: also validate the emitted fixture "
        "with the C++ checker (closes the third leg, host-only)",
    )
    ap.add_argument("--fixture-dir", default=None, help="where to write fixtures")
    ap.add_argument(
        "--require-policy",
        default=analyze.RETENTION_POLICY_LRU,
        choices=list(analyze.RETENTION_POLICIES),
        help="the analyzer policy the runtime rule must match. Default "
        "'row-capacity-lru' is the implemented and published policy; point it "
        "at 'kwindow-union' to demonstrate the check detects the superseded "
        "rule's K>=2 divergence instead of passing vacuously.",
    )
    args = ap.parse_args(argv)

    traces = args.traces or [str(DEFAULT_TRACE)]
    try:
        ks = tuple(int(x) for x in args.retention_k.split(",") if x.strip())
    except ValueError:
        print(f"{PROG}: --retention-k must be comma-separated integers", file=sys.stderr)
        return EXIT_USAGE
    if not ks or any(k < 1 for k in ks):
        print(f"{PROG}: --retention-k values must be >= 1", file=sys.stderr)
        return EXIT_USAGE

    all_ok = True
    for t in traces:
        if not pathlib.Path(t).exists():
            print(f"{PROG}: no such trace: {t}", file=sys.stderr)
            return EXIT_USAGE
        all_ok = crosscheck(
            t, ks, args.checker, args.fixture_dir, args.require_policy
        ) and all_ok

    print(
        "VERDICT: EQUIVALENCE HOLDS -- the runtime rule and the analyzer's "
        f"{args.require_policy} simulation agree at every K tested"
        if all_ok else
        f"VERDICT: DRIFT -- the runtime rule and the analyzer's "
        f"{args.require_policy} simulation disagree"
    )
    return EXIT_OK if all_ok else EXIT_DRIFT


if __name__ == "__main__":
    sys.exit(main())
