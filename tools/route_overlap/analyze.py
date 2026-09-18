#!/usr/bin/env python3
"""T-GPLR-14 route-overlap instrument (offline, host-only, deterministic).

Parses layer-streaming telemetry JSONL traces and computes, per trace:

  T1  per-layer route-union width (min/mean/max over PP chunks)
  T2  temporal window-equality P[hash(c) == hash(c+1)] per layer -- an EXACT
      lower bound on consecutive-window union overlap (equal FNV-1a route
      hashes imply identical expert-union sets; unequal hashes do not imply
      inequality, so this is a bound, never a probability of overlap)
  T3  spatial adjacent-layer equality P[hash(L) == hash(L+1)] -- same bound
      semantics, across layers within each chunk
  T4  overlap between consecutive chunks. Exact from full `union_ids`
      containment when the trace carries row-set windows, with the direction
      stated; otherwise the ADVISORY sampled-ID form (biased LOW: top_k
      sampled IDs per event only). Context only -- T4 is never the gate.
  T5  token-level temporal overlap verdict. MEASURED when the trace carries
      row-set windows (per-token `rowset_fetch`) or per-token route events
      (n_tokens == 1); sourced from `rowset_fetch` when both exist. Otherwise
      INSUFFICIENT_INPUT / NO_ROUTE_EVENTS -- reported loudly, never guessed.
      CONTEXT ONLY: never the gate verdict (see T6).
  T6  GATE 9 VERDICT STATISTIC (b), the realized predictor ceiling: the
      fraction of each window's routed rows a persisting per-layer cache
      would already hold under BOUNDED retention of K windows, simulated
      offline over the recorded union streams. Next window in the denominator
      (|resident n required| / |required|), capacity min(n_expert, K*W*top_k)
      rows per layer, W = rowset_init.window_tokens. The 0.6 threshold is
      read against this. SDD section 10 owns the rule; this tool computes it.
      Windows with a predecessor only (a cold cache holds nothing at the
      first window; including it would add structural zeros).
  T7  per-layer reappearance-distance histogram (offline from the union
      streams) -- the slot-count decision input, never a gate value.
  T8  ORACLE retention ceiling: statistic (b) under UNBOUNDED retention --
      the fraction of each window covered by the union of ALL prior windows
      for that layer. No bounded (or unbounded) cache policy can exceed it on
      the same stream, so it answers "could any policy reach the 0.6 gate
      here?". Caveat carried in the report: the oracle is MONOTONE in stream
      length, so a low oracle on a short stream bounds that stream only -- it
      is not a program-level falsification.

Gate threshold and retention band are constants here, read from the SDD
gate 9 decision rule; `--retention-k` overrides the band for a wider curve.

Row-set events are validated, not merely accepted: every `rowset_fetch` must
carry the canonical ascending unique comma-joined `union_ids`, whose recomputed
hash must equal the emitted `hash` (exit 2 on mismatch).

Failure contract: unsupported event lines, non-canonical or hash-mismatched
row-set events, and missing/unreadable files exit 2 naming file:line or the
path -- never a silent skip.

No third-party dependencies; no timestamps in output (byte-identical reruns
are part of validation). Read-only w.r.t. the traces it analyzes.

Artifacts validated against: build/llama-cuda/gplr14-route-overlap.{txt,json}
(the T-GPLR-14 report format; T6/T7 and the row-set blocks were added by
T-GPLR-15 Phase A, so those artifacts predate them and are not re-asserted).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile

PROG = "analyze.py"

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_LOUD = 2

TOKEN_VERDICT_MEASURED = "MEASURED"
TOKEN_VERDICT_INSUFFICIENT_INPUT = "INSUFFICIENT_INPUT"
TOKEN_VERDICT_NO_ROUTE_EVENTS = "NO_ROUTE_EVENTS"
TOKEN_VERDICT_NO_TG_PHASE = "NO_TG_PHASE"

# Gate 9 constants, owned by the SDD section 10 decision rule.
GATE9_THRESHOLD = 0.6
DEFAULT_RETENTION_KS = (1, 2)

# Retention policies the verdict statistic can be computed under. The SDD's
# section 4 (moe-row-cache) says capacity is a ROW count of K*W*top_k and that
# an overlapping stream lets a cache hold more than K windows' worth of rows;
# section 10 describes the statistic as "K-window cumulative-resident-set
# containment". Those two readings are NOT the same policy, so both are named
# explicitly. POLICY RESOLVED 2026-09-13 (owner-directed): the implementation
# and the gate statistic both use row-capacity-lru, so the published (b) curve
# is now a PREDICTION of the mechanism rather than an overstatement of it.
#
#   row-capacity-lru  the resident set is the cap = min(n_expert, K*W*top_k)
#                     most-recently-ROUTED rows. DEFAULT, and the policy the
#                     runtime (llama-layer-streaming.cpp load_row_set), the
#                     C++ checker (tools/layer-streaming/main.cpp
#                     ls_retention_errors) and the C++ acceptance gate
#                     (--residency-tg-retain K) all implement. Verified equal to
#                     the runtime by tools/route_overlap/crosscheck-retention.py
#                     at K=1,2,4,8 on the preserved probe stream.
#   kwindow-union     SUPERSEDED. the resident set is union(windows e-K+1..e)
#                     with an in-window row promoted. Same footprint, weakly
#                     weaker (see the cross-check), kept only so the superseded
#                     rule stays measurable and the 2026-09-13 decision
#                     remains falsifiable.
# The two coincide exactly at K=1 (cap == one window) and diverge at K>=2.
RETENTION_POLICY_LRU = "row-capacity-lru"
RETENTION_POLICY_UNION = "kwindow-union"
RETENTION_POLICIES = (RETENTION_POLICY_LRU, RETENTION_POLICY_UNION)

# Vendor hash parameters. NOTE: llama-layer-streaming.cpp's tel_rowset_fetch
# seeds with the literal 1469598103934665603, which is NOT the standard FNV-1a
# offset basis 0xCBF29CE484222325 (14695981039346656037) -- its comment claims
# equality with this tool's helper. Verified by reproducing the emitted hashes
# on the recorded row-set traces; the vendor constant is what both sides must
# use, and it is the reason this is not a textbook FNV-1a.
_FNV_SEED = 1469598103934665603
_FNV_PRIME = 0x100000001B3


def fail(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"[{PROG}] ERROR: {msg}", file=sys.stderr)
    sys.exit(EXIT_LOUD)


def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog=PROG,
        description="T-GPLR-14 route-overlap instrument "
        "(offline analysis of layer-streaming telemetry JSONL).",
    )
    ap.add_argument(
        "traces",
        nargs="*",
        default=None,
        help="telemetry .jsonl paths or glob patterns (required unless --self-test)",
    )
    ap.add_argument("--txt", default=None, help="human-readable report path (txt)")
    ap.add_argument(
        "--json",
        default=None,
        dest="json_out",
        help="machine-readable report path (json)",
    )
    ap.add_argument(
        "--retention-k",
        default=",".join(str(k) for k in DEFAULT_RETENTION_KS),
        help="comma-separated retention window counts (K) for the gate 9 "
        "statistic (b) curve; default is the SDD's reference band",
    )
    ap.add_argument(
        "--retention-policy",
        default=RETENTION_POLICY_LRU,
        choices=list(RETENTION_POLICIES),
        help="bounding rule for the statistic (b) curve: 'row-capacity-lru' "
        "(default; both the implemented and the published policy) keeps the "
        "cap most-recently-requested rows, 'kwindow-union' (superseded) keeps "
        "union(windows e-K+1..e) with promotion. They agree exactly at K=1.",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="run built-in self-test on temp-dir fixtures and exit",
    )
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------
# trace parsing
# ----------------------------------------------------------------------------


def load_trace(path: str):
    """Parse one telemetry JSONL file into (run_id, model, n_tg, routes, rowset).

    routes: list of route-event dicts in file order.
    rowset: {"init": event-or-None, "fetch": [events], "round": n,
             "refused": n} -- the moe-row-set TG windows, if any.
    Fails loudly (exit 2, file:line) on unsupported event types, missing
    route fields, malformed JSON, non-canonical or hash-mismatched row-set
    events, and missing/unreadable files.
    """
    try:
        f = open(path, "r", encoding="utf-8", errors="strict")
    except OSError as e:
        fail(f"cannot read trace file: {path}: {e}")

    run_id = None
    model = None
    n_tg = 0
    routes = []
    rowset = {"init": None, "fetch": [], "round": 0, "refused": 0}
    with f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError as e:
                fail(f"{path}:{lineno}: malformed JSON line ({e})")
            if not isinstance(ev, dict) or "event" not in ev:
                fail(f"{path}:{lineno}: line is not a telemetry event object")
            name = ev.get("event")
            if name == "run_start":
                run_id = ev.get("run_id")
                model = ev.get("model")
            elif name == "tg_tokens":
                n = ev.get("n_tokens_generated")
                if isinstance(n, int) and n > 0:
                    n_tg += n
            elif name == "route":
                missing = [
                    k
                    for k in ("layer", "chunk", "n_tokens", "route_union", "hash")
                    if k not in ev
                ]
                if missing:
                    fail(
                        f"{path}:{lineno}: route event missing field(s): "
                        + ", ".join(missing)
                    )
                routes.append(ev)
            elif name == "rowset_init":
                rowset["init"] = ev
            elif name == "rowset_round":
                rowset["round"] += 1
            elif name == "rowset_refused":
                rowset["refused"] += 1
            elif name == "rowset_fetch":
                missing = [
                    k
                    for k in (
                        "layer",
                        "chunk",
                        "n_tokens",
                        "top_k",
                        "route_union",
                        "union_ids",
                        "hash",
                    )
                    if k not in ev
                ]
                if missing:
                    fail(
                        f"{path}:{lineno}: rowset_fetch missing field(s): "
                        + ", ".join(missing)
                    )
                ids = _ids_of(ev)
                if list(ids) != sorted(set(ids)):
                    fail(
                        f"{path}:{lineno}: rowset_fetch union_ids is not the "
                        f"canonical ascending unique comma-joined list: "
                        f"{ev['union_ids']!r}"
                    )
                if len(ids) != ev["route_union"]:
                    fail(
                        f"{path}:{lineno}: rowset_fetch union_ids count "
                        f"{len(ids)} != route_union {ev['route_union']}"
                    )
                recomputed = _fnv1a64(ids)
                if recomputed != ev["hash"]:
                    fail(
                        f"{path}:{lineno}: rowset_fetch hash mismatch: emitted "
                        f"{ev['hash']}, recomputed {recomputed} from "
                        f"union_ids {ev['union_ids']!r}"
                    )
                rowset["fetch"].append(ev)
            elif name in (
                # Known non-route events (inventory from the recorded
                # 2026-09-09 gplr12-* traces): ignored here, validated
                # elsewhere by check-telemetry.
                "phase_summary",
                "run_end",
                "slot_copy",
                "slot_request",
                "slot_evict",
                "layer_ready_wait",
                "layer_compute",
                "copy_submit",
                "copy_complete",
                "act_bind",
                "pkg_member",
                "graph_footprint",
                "byte_account",
                "meta",
                # T-GPLR-25 async-feed PP telemetry (engine: tel_async_feed_init /
                # _submit / _ready). Added 2026-09-18: these are emitted on every
                # PP run of the current build, so their absence here made EVERY
                # modern trace unreadable ("unsupported event type:
                # 'async_feed_init'") and disabled the offline-vs-runtime
                # cross-check in tools/run-gplr19-rowcache-ab.ps1. They carry no
                # routing information and are ignored, as are the other non-route
                # events above; check-telemetry validates them.
                "async_feed_init",
                "async_feed_submit",
                "async_feed_ready",
                "prefetch_submit",
                "prefetch_complete",
                "prefetch_init",
                "migrate",
                "begin_handoff",
                "handoff",
                "release",
                "tg_text",
            ):
                pass
            else:
                fail(f"{path}:{lineno}: unsupported event type: {name!r}")
    return run_id, model, n_tg, routes, rowset


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------


def _fnv1a64(ids):
    """Reproduce the union hash emitted by tel_rowset_fetch, byte for byte.

    FNV-1a-style over the canonical ascending comma-joined ID bytes, using the
    VENDOR's seed literal (see _FNV_SEED) rather than the standard offset
    basis, so a recompute cross-check reproduces real traces instead of
    failing on all of them. Named for the vendor's comment, which refers to
    this helper by this name.
    """
    h = _FNV_SEED
    for b in ",".join(str(i) for i in ids).encode("ascii"):
        h ^= b
        h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def _ids_of(ev):
    """Canonical union list of a row-set event as a tuple of ints (ascending)."""
    raw = str(ev.get("union_ids", ""))
    return tuple(int(x) for x in raw.split(",") if x.strip() != "")


def _eq_rate(pairs):
    if not pairs:
        return None
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def union_width_rows(routes):
    """T1: per-layer (layer, min, mean, max) of route_union over chunks."""
    by_layer = {}
    for r in routes:
        by_layer.setdefault(r["layer"], []).append(r)
    rows = []
    for layer in sorted(by_layer):
        ws = [r["route_union"] for r in by_layer[layer]]
        rows.append(
            (layer, min(ws), sum(ws) / len(ws), max(ws))
        )
    return rows


def temporal_window_rows(routes):
    """T2: per-layer P[hash(c) == hash(c+1)] over consecutive chunks (bound)."""
    by_layer = {}
    for r in routes:
        by_layer.setdefault(r["layer"], []).append(r)
    rows = []
    for layer in sorted(by_layer):
        evs = sorted(by_layer[layer], key=lambda r: r["chunk"])
        pairs = [(a["hash"], b["hash"]) for a, b in zip(evs, evs[1:])]
        rate = _eq_rate(pairs)
        if rate is not None:
            rows.append((layer, rate))
    return rows


def spatial_rows(routes):
    """T3: per-chunk adjacent-layer hash-equality (exact lower bound)."""
    by_chunk = {}
    for r in routes:
        by_chunk.setdefault(r["chunk"], []).append(r)
    rates = []
    for chunk in sorted(by_chunk):
        evs = sorted(by_chunk[chunk], key=lambda r: r["layer"])
        pairs = [(a["hash"], b["hash"]) for a, b in zip(evs, evs[1:])]
        rate = _eq_rate(pairs)
        if rate is not None:
            rates.append(rate)
    if not rates:
        return None
    return sum(rates) / len(rates)


def advisory_sampled_overlap(routes):
    """T4: sampled-ID overlap between consecutive chunks.

    Metric (pinned by reproducing the recorded 2026-09-09 real-trace values,
    e.g. 0.28921875 on gplr12-100k-bf16): CONTAINMENT -- |A intersect B| / |A|
    where A is the earlier chunk's sampled_ids set. Fraction of the previous
    chunk's sampled IDs that recur in the next chunk.

    ADVISORY ONLY: sampled_ids carries only the top_k IDs per event, so this
    is biased low relative to full-union overlap. Never a go/no-go input.

    Pooling: per-layer mean, then mean of layers. With the uniform event
    counts of all recorded traces, pooled-pair and per-layer means are
    indistinguishable; recorded results are identical under either.
    """
    by_layer = {}
    for r in routes:
        by_layer.setdefault(r["layer"], []).append(r)
    pairs = []
    for layer in sorted(by_layer):
        evs = sorted(by_layer[layer], key=lambda r: r["chunk"])
        for a, b in zip(evs, evs[1:]):
            sa = {s.strip() for s in str(a.get("sampled_ids", "")).split(",") if s.strip()}
            sb = {s.strip() for s in str(b.get("sampled_ids", "")).split(",") if s.strip()}
            if not sa or not sb:
                continue
            pairs.append(len(sa & sb) / len(sa))
    if not pairs:
        return None
    return sum(pairs) / len(pairs)


def token_temporal(routes, n_tg, series=None):
    """T5: token-level temporal overlap verdict -- CONTEXT, never the gate.

    Returns (verdict, detail, hit_rate_or_None). Source precedence: row-set
    windows (`rowset_fetch`, per layer per token, W=1) when present, else
    per-token route events (n_tokens == 1). Chunked routes (n_tokens > 1)
    cannot yield the token-level statistic.
    """
    if series:
        pairs = 0
        hits = 0
        for _layer, wins in sorted(series.items()):
            for (_c0, _i0, h0), (_c1, _i1, h1) in zip(wins, wins[1:]):
                pairs += 1
                if h0 == h1:
                    hits += 1
        if pairs == 0:
            return TOKEN_VERDICT_INSUFFICIENT_INPUT, (
                "row-set windows present but fewer than two consecutive "
                "windows per layer; window-equality rate undefined"
            ), None
        return TOKEN_VERDICT_MEASURED, (
            "measured from rowset_fetch per-token windows (n_tokens==1); "
            "CONTEXT only -- the gate verdict is the T6 statistic"
        ), hits / pairs
    per_token = [r for r in routes if r["n_tokens"] == 1]
    if not per_token:
        if not routes:
            return TOKEN_VERDICT_NO_ROUTE_EVENTS, (
                "trace carries no route events; token-level temporal overlap "
                "cannot be computed"
            ), None
        if n_tg == 0:
            return TOKEN_VERDICT_NO_TG_PHASE, (
                "prefill-only trace; no TG tokens generated"
            ), None
        return TOKEN_VERDICT_INSUFFICIENT_INPUT, (
            "trace generates {} TG tokens but carries no per-token route "
            "events (routes exist only at PP chunk granularity, "
            "n_tokens>1). Exact token-level temporal overlap requires full "
            "per-token union ID lists -- provided by the row-set task's "
            "rowset_fetch telemetry (T-GPLR-13)."
        ).format(n_tg), None
    # MEASURED path: hit rate = fraction of consecutive per-token route pairs
    # (per layer, consecutive tokens) whose route unions are hash-identical.
    by_layer = {}
    for r in per_token:
        by_layer.setdefault(r["layer"], []).append(r)
    hits = 0
    total = 0
    for layer in sorted(by_layer):
        evs = sorted(by_layer[layer], key=lambda r: r["chunk"])
        for a, b in zip(evs, evs[1:]):
            total += 1
            if a["hash"] == b["hash"]:
                hits += 1
    if total == 0:
        return TOKEN_VERDICT_INSUFFICIENT_INPUT, (
            "per-token route events present but fewer than two consecutive "
            "token events per layer; hit rate undefined"
        ), None
    return TOKEN_VERDICT_MEASURED, (
        "measured from per-token route events (n_tokens==1); CONTEXT only -- "
        "the gate verdict is the T6 statistic"
    ), hits / total


def rowset_series(rowset_fetch):
    """Per-layer ordered windows from rowset_fetch: {layer: [(chunk, ids, hash)]}."""
    by_layer = {}
    for ev in rowset_fetch:
        by_layer.setdefault(ev["layer"], []).append(ev)
    return {
        layer: [
            (e["chunk"], _ids_of(e), e["hash"])
            for e in sorted(evs, key=lambda e: e["chunk"])
        ]
        for layer, evs in by_layer.items()
    }


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def exact_consecutive_containment(series):
    """T4 (row-set form): exact consecutive-window containment, both directions.

    prev_in_next = |A n B| / |B| (consecutive window in the denominator),
    next_in_prev = |A n B| / |A|. The direction is stated because the SDD
    requires it wherever a containment figure is reported. CONTEXT only.
    """
    prev_in_next = []
    next_in_prev = []
    for _layer, wins in sorted(series.items()):
        for (_c0, a, _h0), (_c1, b, _h1) in zip(wins, wins[1:]):
            if not a or not b:
                continue
            inter = len(set(a) & set(b))
            prev_in_next.append(inter / len(b))
            next_in_prev.append(inter / len(a))
    if not prev_in_next:
        return None
    return {
        "pairs": len(prev_in_next),
        "prev_in_next": _mean(prev_in_next),
        "next_in_prev": _mean(next_in_prev),
    }


def simulate_retention(wins, capacity, policy=RETENTION_POLICY_LRU, retain_k=None):
    """Retention per-window (required, hits) for one layer.

    capacity=None means UNBOUNDED retention (the T8 oracle); otherwise the
    resident set is bounded. `policy` selects the bounding rule (see the
    RETENTION_POLICY_* constants); `retain_k` is the K the union policy needs.

    row-capacity-lru (DEFAULT): the resident set is a row-count-bounded LRU
    over the most recent requests -- capacity rows, window-granular recency
    (all rows of one window share a recency), ascending-id tie-break. Capacity
    is a ROW count per SDD section 4, so when windows overlap a cache can hold
    more than K windows' worth of rows -- that is the realized ceiling, not a
    window-count abstraction.

    kwindow-union: the resident set is union(windows e-K+1..e). A row routed
    in the current window is PROMOTED to it, so a hot row stays part of the
    sliding union instead of ageing out while still in use; a row absent from
    the current window and older than K-1 windows is evicted. This mirrors the
    runtime's load_row_set retirement order exactly (age > K first, then
    promote an in-union row, then evict an out-of-union row of age > K-1), and
    it is what makes the cache bounded by K*W*top_k by construction rather
    than by capacity pressure.

    Windows with a predecessor only: a cold cache holds nothing at the first
    window, and counting it would add structural zeros the pilot never had.
    """
    if policy not in RETENTION_POLICIES:
        raise ValueError(f"unknown retention policy {policy!r}")
    if policy == RETENTION_POLICY_UNION:
        if retain_k is None:
            raise ValueError("kwindow-union needs retain_k")
        resident = {}          # row -> epoch it was last routed in
        out = []
        for idx, (_chunk, ids, _h) in enumerate(wins):
            s = set(ids)
            # Retirement runs BEFORE scoring, exactly as load_row_set step 1
            # runs before its step 2: a row routed in this window but older
            # than K is retired first and is therefore NOT a hit. Scoring
            # first would count a stale in-union row the runtime refuses.
            for x in list(resident):
                age = idx - resident[x]
                if age > retain_k:
                    del resident[x]
                elif x in s:
                    resident[x] = idx
                elif age > retain_k - 1:
                    del resident[x]
            if idx:
                out.append((len(s), len(s & set(resident))))
            for x in s:
                resident[x] = idx
            if capacity is not None and len(resident) > capacity:
                raise AssertionError(
                    f"kwindow-union exceeded its declared capacity at window "
                    f"{idx}: {len(resident)} > {capacity}"
                )
        return out
    resident = []
    out = []
    for idx, (_chunk, ids, _h) in enumerate(wins):
        if idx:
            s = set(ids)
            out.append((len(s), len(s.intersection(resident))))
        for i in sorted(set(ids)):
            if i in resident:
                resident.remove(i)
            resident.insert(0, i)
        if capacity is not None:
            del resident[capacity:]
    return out


def resident_containment(series, ks, n_expert, top_k, window_tokens, row_bytes=0,
                         policy=RETENTION_POLICY_LRU):
    """T6: SDD gate 9 statistic (b) -- the realized predictor ceiling, per K.

    value = mean over per-layer windows-with-a-predecessor of
    |resident n required| / |required| -- next window in the denominator.
    K=1 reduces exactly to consecutive-window containment, which is this
    tool's cross-validation anchor against the recorded pilot (0.358-0.389).
    Both K=1 anchors hold under either policy; they diverge at K>=2.
    """
    out = {}
    for k in ks:
        cap = k * window_tokens * top_k
        if n_expert:
            cap = min(n_expert, cap)
        cap = max(1, cap)
        pooled = []
        per_layer = []
        hits_total = 0
        required_total = 0
        for _layer, wins in sorted(series.items()):
            rows = simulate_retention(wins, cap, policy, retain_k=k)
            ratios = [h / req for req, h in rows if req]
            pooled.extend(ratios)
            hits_total += sum(h for _req, h in rows)
            required_total += sum(req for req, _h in rows)
            layer_mean = _mean(ratios)
            if layer_mean is not None:
                per_layer.append(layer_mean)
        out[k] = {
            "policy": policy,
            "capacity_rows_per_layer": cap,
            "capacity_bytes_total": (
                cap * row_bytes * len(series) if row_bytes else None
            ),
            "value": _mean(pooled),
            "per_layer_min": min(per_layer) if per_layer else None,
            "per_layer_mean": _mean(per_layer),
            "per_layer_max": max(per_layer) if per_layer else None,
            "layers": len(per_layer),
            "windows_with_predecessor": len(pooled),
            "hits_total": hits_total,
            "required_total": required_total,
        }
    return out


def oracle_containment(series):
    """T8: the unbounded-retention ceiling -- an upper bound on statistic (b).

    Every window with a predecessor is scored as |union(all prior windows) n
    window| / |window|, per layer. No cache policy can beat this on the same
    stream, so comparing it against the 0.6 threshold answers whether ANY
    policy could pass here -- and comparing it against the per-K curve shows
    how much of that headroom a small retention actually reaches.

    Reported with explicit totals so callers can detect EXACT saturation (a
    K whose pooled hits and required equal the oracle's) rather than inferring
    it from two rounded means.
    """
    pooled = []
    per_layer = []
    hits_total = 0
    required_total = 0
    for _layer, wins in sorted(series.items()):
        rows = simulate_retention(wins, None)
        ratios = [h / req for req, h in rows if req]
        pooled.extend(ratios)
        hits_total += sum(h for _req, h in rows)
        required_total += sum(req for req, _h in rows)
        layer_mean = _mean(ratios)
        if layer_mean is not None:
            per_layer.append(layer_mean)
    value = _mean(pooled)
    return {
        "value": value,
        "per_layer_min": min(per_layer) if per_layer else None,
        "per_layer_mean": _mean(per_layer),
        "per_layer_max": max(per_layer) if per_layer else None,
        "layers": len(per_layer),
        "windows_with_predecessor": len(pooled),
        "hits_total": hits_total,
        "required_total": required_total,
        "below_gate9_threshold": (
            None if value is None else value < GATE9_THRESHOLD
        ),
    }


def reappearance_histogram(series):
    """T7: per-layer reappearance distance, in windows, for each routed row.

    'unique_layer_rows' counts (layer, expert) pairs -- row identity is
    per-layer, matching the moe-row-cache residency class; the T-GPLR-15
    slot-count input is the coverage table, never a gate value.
    """
    gaps = {}
    unique_rows = 0
    rows_once = 0
    for _layer, wins in sorted(series.items()):
        pos = {}
        for idx, (_chunk, ids, _h) in enumerate(wins):
            for i in set(ids):
                pos.setdefault(i, []).append(idx)
        for i in sorted(pos):
            where = pos[i]
            unique_rows += 1
            if len(where) == 1:
                rows_once += 1
                continue
            for a, b in zip(where, where[1:]):
                gaps[b - a] = gaps.get(b - a, 0) + 1
    total = sum(gaps.values())
    coverage = {}
    for cap_windows in (1, 2, 4, 8):
        covered = sum(c for g, c in gaps.items() if g <= cap_windows)
        coverage[str(cap_windows)] = (covered / total) if total else None
    return {
        "unique_layer_rows": unique_rows,
        "rows_seen_once": rows_once,
        "reappearances": total,
        "gaps": {str(g): gaps[g] for g in sorted(gaps)},
        "coverage_by_retention_windows": coverage,
    }


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------


def fmt_float(x, nd=4):
    return f"{x:.{nd}f}"


def fmt_or_na(x, nd=4):
    return fmt_float(x, nd) if x is not None else "n/a"


def build_report(traces, retention_ks=DEFAULT_RETENTION_KS,
                 retention_policy=RETENTION_POLICY_LRU):
    report = []
    for path, (run_id, model, n_tg, routes, rowset) in traces:
        series = rowset_series(rowset["fetch"])
        widths = union_width_rows(routes)
        t2rows = temporal_window_rows(routes)
        t3 = spatial_rows(routes)
        t4 = advisory_sampled_overlap(routes)
        verdict, detail, hit = token_temporal(routes, n_tg, series)

        rowset_block = None
        if series:
            init = rowset["init"] or {}
            n_expert = init.get("n_expert") or 0
            top_k = init.get("top_k") or (rowset["fetch"][0].get("top_k") or 1)
            window_tokens = init.get("window_tokens") or 1
            slot_bytes = init.get("slot_bytes") or 0
            row_bytes = (
                int(slot_bytes // n_expert) if (n_expert and slot_bytes) else 0
            )
            b_curve = resident_containment(
                series, retention_ks, n_expert, top_k, window_tokens, row_bytes,
                retention_policy,
            )
            oracle = oracle_containment(series)
            # Exact saturation: a reported K whose pooled hits and required equal
            # the oracle's, i.e. bounded retention already holds everything the
            # unbounded cache could -- so more VRAM buys nothing on this stream.
            saturated_by = next(
                (
                    k
                    for k, v in sorted(b_curve.items())
                    if v["hits_total"] == oracle["hits_total"]
                    and v["required_total"] == oracle["required_total"]
                ),
                None,
            )
            rowset_block = {
                "n_fetch": len(rowset["fetch"]),
                "n_round": rowset["round"],
                "n_refused": rowset["refused"],
                "n_layers": len(series),
                "windows_per_layer": sorted({len(w) for w in series.values()}),
                "window_tokens": window_tokens,
                "top_k": top_k,
                "n_expert": n_expert,
                "retention_policy": retention_policy,
                "slot_bytes": slot_bytes,
                "row_bytes": row_bytes,
                "hash_recompute_ok": len(rowset["fetch"]),
                "t5_window_pairs": sum(len(w) - 1 for w in series.values()),
                "verdict_statistic_b": b_curve,
                "oracle_containment": oracle,
                "ceiling_saturated_by_k": saturated_by,
                "exact_consecutive": exact_consecutive_containment(series),
                "reappearance": reappearance_histogram(series),
            }

        report.append(
            {
                "path": path,
                "run_id": run_id,
                "model": model,
                "union_width_rows": [list(w) for w in widths],
                "temporal_window_rows": [
                    [l, r] for (l, r) in t2rows
                ],
                "temporal_window_agg": (
                    sum(r for _l, r in t2rows) / len(t2rows) if t2rows else None
                ),
                "spatial_agg": t3,
                "advisory_sampled_overlap": t4,
                "temporal_token_verdict": verdict,
                "temporal_token_detail": detail,
                "temporal_token_hit_rate": hit,
                "rowset": rowset_block,
            }
        )
    return report


def render_txt(report, json_basename=None):
    lines = []
    lines.append("T-GPLR-14 route-overlap report (deterministic; no timestamps)")
    lines.append(f"traces analyzed: {len(report)}")
    for rep in report:
        lines.append("")
        lines.append(f"== {rep['path']} (run_id={rep['run_id']}, model={rep['model']})")
        lines.append("   T1 route-union width per layer (min/mean/max over chunks):")
        for (layer, mn, mean, mx) in rep["union_width_rows"]:
            lines.append(
                f"     layer {layer:>3}: {mn:>4} / {mean:>8.2f} / {mx:>4}"
            )
        lines.append(
            "   T2 temporal window-equality (P[hash(c)==hash(c+1)], exact lower "
            "bound on consecutive-window overlap):"
        )
        if rep["temporal_window_rows"]:
            for (layer, rate) in rep["temporal_window_rows"]:
                lines.append(f"     layer {layer:>3}: {fmt_float(rate)}")
            lines.append(
                f"     aggregate: {fmt_float(rep['temporal_window_agg'])}"
            )
        else:
            lines.append("     aggregate: n/a")
        lines.append(
            "   T3 spatial adjacent-layer equality (P[hash(L)==hash(L+1)] per "
            "chunk, exact lower bound): "
            + (fmt_float(rep["spatial_agg"]) if rep["spatial_agg"] is not None else "n/a")
        )
        rb = rep.get("rowset")
        if rb and rb["exact_consecutive"]:
            ec = rb["exact_consecutive"]
            lines.append(
                "   T4 exact consecutive-window containment from full "
                f"union_ids ({ec['pairs']} pairs): |A n B|/|B| = "
                f"{fmt_float(ec['prev_in_next'])} (consecutive window in the "
                f"denominator), |A n B|/|A| = {fmt_float(ec['next_in_prev'])} "
                "-- CONTEXT only"
            )
        else:
            lines.append(
                "   T4 sampled-ID overlap, consecutive chunks (ADVISORY, biased "
                "low -- top_k samples per event, never used for go/no-go): "
                + fmt_or_na(rep["advisory_sampled_overlap"])
            )
        lines.append(
            f"   T5 token-level temporal overlap verdict: {rep['temporal_token_verdict']}"
            + (
                " (window-equality rate "
                f"{fmt_float(rep['temporal_token_hit_rate'])}"
                + (f", n_pairs {rb['t5_window_pairs']}" if rb else "")
                + ")"
                if rep.get("temporal_token_hit_rate") is not None
                else ""
            )
        )
        lines.append(f"     {rep['temporal_token_detail']}")
        if rb:
            lines.append(
                "   T6 GATE 9 statistic (b), realized predictor ceiling "
                f"[threshold {GATE9_THRESHOLD}; SDD section 10 owns the rule]:"
            )
            for k, v in sorted(rb["verdict_statistic_b"].items()):
                cap_note = f"capacity {v['capacity_rows_per_layer']} rows/layer"
                if v["capacity_bytes_total"]:
                    cap_note += (
                        f", {v['capacity_bytes_total'] / 1e6:.1f} MB over "
                        f"{v['layers']} layers"
                    )
                lines.append(
                    f"     K={k}: {fmt_or_na(v['value'])}  ({cap_note}; "
                    f"{v['windows_with_predecessor']} windows; per-layer "
                    f"min/mean/max {fmt_or_na(v['per_layer_min'])}/"
                    f"{fmt_or_na(v['per_layer_mean'])}/"
                    f"{fmt_or_na(v['per_layer_max'])}) -- W={rb['window_tokens']}"
                )
            orc = rb["oracle_containment"]
            sat = rb["ceiling_saturated_by_k"]
            lines.append(
                "   T8 oracle ceiling (unbounded retention; no policy can exceed "
                f"it on this stream): {fmt_or_na(orc['value'])}"
                + (
                    f" -- reported K={sat} ALREADY SATURATES it (more VRAM buys "
                    "nothing here)"
                    if sat is not None
                    else " -- NOT saturated by the reported K values (retention "
                    "is the limit, not the stream)"
                )
            )
            rap = rb["reappearance"]
            cov = rap["coverage_by_retention_windows"]
            lines.append(
                "   T7 reappearance distance in windows: "
                + (
                    ", ".join(f"{g}:{c}" for g, c in rap["gaps"].items())
                    if rap["gaps"]
                    else "none (no row reappears)"
                )
                + f"; rows seen once {rap['rows_seen_once']}/"
                f"{rap['unique_layer_rows']}; cumulative coverage at retention "
                + "/".join(f"{w}:{fmt_or_na(cov[w])}" for w in ("1", "2", "4", "8"))
                + " windows"
            )
    lines.append("")
    readings = [
        (
            rep["path"],
            rep["rowset"]["verdict_statistic_b"],
            rep["rowset"]["oracle_containment"],
            rep["rowset"]["ceiling_saturated_by_k"],
        )
        for rep in report
        if rep.get("rowset")
    ]
    rs_first = next((rep["rowset"] for rep in report if rep.get("rowset")), None)
    if readings:
        lines.append(
            f"T-GPLR-15 gate 9 reading (threshold {GATE9_THRESHOLD}; SDD section "
            "10 owns the rule) -- statistic (b), the realized predictor ceiling:"
        )
        for path, b, orc, sat in readings:
            cells = ", ".join(
                f"K={k}: {fmt_or_na(v['value'])}" for k, v in sorted(b.items())
            )
            lines.append(f"   {path}: {cells}")
            lines.append(
                "     ceiling (T8 oracle, unbounded retention): "
                + fmt_or_na(orc["value"])
                + (
                    f" -- reported K={sat} already saturates it"
                    if sat is not None
                    else " -- NOT saturated by the reported K values"
                )
                + (
                    f"; BELOW the {GATE9_THRESHOLD} threshold, so NO cache policy "
                    "of any size reaches the gate on this stream"
                    if orc["below_gate9_threshold"]
                    else f"; at or above the {GATE9_THRESHOLD} threshold"
                )
            )
        if all(orc["below_gate9_threshold"] for _, _, orc, _ in readings):
            lines.append(
                "   The oracle is MONOTONE in stream length (more history can only "
                "add coverage), so a below-threshold ceiling on these short samples "
                "falsifies the gate for THESE STREAMS only, never for the program. "
                "What separates 'no policy can pass' from 'retention is the limit' "
                "is a stream long enough for the T8 line to mean something, such as "
                "T-GPLR-15 Phase A's long single-prompt generation probe."
            )
        else:
            lines.append(
                "   The T8 oracle is an UPPER BOUND, not a verdict: it is statistic "
                "(b) under UNBOUNDED retention, i.e. the whole expert bank resident, "
                "which is the outcome the row-set policy exists to avoid. The gate "
                "is read against statistic (b) at the retention the adopting task "
                "DECLARES (SDD section 4 reference band: K=1-2, next window in the "
                "denominator), so a ceiling at or above the threshold does not pass "
                "the gate by itself -- it bounds what a larger declared retention "
                "could buy, at the VRAM cost printed at each K."
            )
        if rs_first is not None:
            lines.append(
                "   How to read K: K is the per-layer cache's retention DEPTH in "
                "token windows, not bytes and not a count of experts. A K-window "
                "cache holds min(n_expert, K*W*top_k) expert rows per layer (here "
                f"W={rs_first['window_tokens']}, top_k={rs_first['top_k']}, "
                f"n_expert={rs_first['n_expert']}), and one row is that expert's "
                f"gate+up+down slabs ({rs_first['row_bytes']:,} B here). The byte "
                "totals printed at each K therefore FOLLOW from K rather than "
                "being chosen: K=1 keeps only the previous window. Statistic (b) "
                "is defined AT a K, so a gate reading is complete only when the K "
                "it was read at is named -- the SDD section 4 reference band is "
                "K=1-2."
            )
            lines.append(
                "   Plot this curve as a standalone artifact (stdlib only, no "
                "network): python tools/route_overlap/plot-b-vs-k.py "
                "--input <json> --out <html>"
            )
    else:
        lines.append(
            "T-GPLR-15 go/no-go: UNDECIDABLE from these traces (no row-set "
            "windows and no per-token route events: token-level temporal hit "
            "rate INSUFFICIENT_INPUT, threshold "
            f"{GATE9_THRESHOLD} UNMEASURED). Gate stays tied to the row-set "
            "task's per-token rowset_fetch telemetry, per the escalation "
            "condition."
        )
    if json_basename:
        lines.append(f"json report: {json_basename}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

GOOD_JSONL = (
    '{"event": "run_start", "run_id": 7, "model": "M", "schedule": '
    '"layer-major", "residency": "moe-expert-bank"}\n'
    '{"event": "route", "layer": 0, "chunk": 0, "top_k": 8, "n_tokens": 512, '
    '"route_union": 254, "hash": 111, "sampled_ids": "1,2,3,4,5,6,7,8"}\n'
    '{"event": "route", "layer": 0, "chunk": 1, "top_k": 8, "n_tokens": 512, '
    '"route_union": 254, "hash": 111, "sampled_ids": "1,2,3,4,5,6,7,8"}\n'
    '{"event": "route", "layer": 0, "chunk": 2, "top_k": 8, "n_tokens": 512, '
    '"route_union": 100, "hash": 222, "sampled_ids": "9,10,11,12,13,14,15,16"}\n'
    '{"event": "route", "layer": 1, "chunk": 0, "top_k": 8, "n_tokens": 512, '
    '"route_union": 254, "hash": 111, "sampled_ids": "1,2,3,4,5,6,7,8"}\n'
    '{"event": "route", "layer": 1, "chunk": 1, "top_k": 8, "n_tokens": 512, '
    '"route_union": 250, "hash": 333, "sampled_ids": "21,22,23,24,25,26,27,28"}\n'
    '{"event": "route", "layer": 1, "chunk": 2, "top_k": 8, "n_tokens": 512, '
    '"route_union": 100, "hash": 222, "sampled_ids": "9,10,11,12,13,14,15,16"}\n'
    '{"event": "tg_tokens", "n_tokens_generated": 5, "ids": [1, 2, 3, 4, 5]}\n'
)

TOK_JSONL = (
    '{"event": "run_start", "run_id": 8, "model": "M", "schedule": '
    '"batch-major", "residency": "moe-expert-bank"}\n'
    '{"event": "route", "layer": 0, "chunk": 0, "top_k": 8, "n_tokens": 1, '
    '"route_union": 8, "hash": 1000, "sampled_ids": "1,2,3,4,5,6,7,8"}\n'
    '{"event": "route", "layer": 0, "chunk": 1, "top_k": 8, "n_tokens": 1, '
    '"route_union": 8, "hash": 1001, "sampled_ids": "1,2,3,4,5,6,7,8"}\n'
    '{"event": "route", "layer": 0, "chunk": 2, "top_k": 8, "n_tokens": 1, '
    '"route_union": 8, "hash": 1002, "sampled_ids": "9,10,11,12,13,14,15,16"}\n'
)

BAD_JSONL = (
    '{"event": "route", "layer": 0, "chunk": 0, "n_tokens": 512, '
    '"route_union": 254}\n'
)

BAD2_JSONL = "{not json}\n"

# Row-set fixture: 2 layers x 3 TG windows. Hashes are LITERAL vendor values
# (seed 1469598103934665603), not produced by the helper under test, so an
# accidental change to the hash function fails the self-test.
_ROWSET_WINDOWS = (
    (0, 1, "10,11,12,13,14,15,16,17", 7622554333669719173),
    (0, 2, "1,2,3,4,5,6,7,8", 9797314157462417237),
    (0, 3, "10,11,12,13,14,15,16,17", 7622554333669719173),
    (1, 1, "1,2,3,4,5,6,7,8", 9797314157462417237),
    (1, 2, "10,11,12,13,14,15,16,17", 7622554333669719173),
    (1, 3, "1,2,3,4,5,6,7,8", 9797314157462417237),
)


# Reuse distance LONGER than any 2-window capacity: window 4 re-requests
# window 1's rows after two intervening disjoint windows. K=2 (16 rows) has
# evicted them; the unbounded oracle still holds them. This is the fixture that
# separates "bounded retention is the limit" from "the stream is the limit".
_ORACLE_WINDOWS = (
    (0, 1, "1,2,3,4,5,6,7,8", 9797314157462417237),
    (0, 2, "10,11,12,13,14,15,16,17", 7622554333669719173),
    (0, 3, "18,19,20,21,22,23,24,25", 1405361139609267651),
    (0, 4, "1,2,3,4,5,6,7,8", 9797314157462417237),
)


# Each layer requires the OTHER window's rows one step later, so K=1 holds
# nothing (0.0) while K=2 holds everything (0.5 pooled) -- the fixture that
# proves the K curve is real and not a constant. Because its union never exceeds
# 16 rows, K=2 also exactly saturates the T8 oracle, which tests that path too.
def rowset_jsonl(mutation=None, windows=None):
    windows = _ROWSET_WINDOWS if windows is None else windows
    out = [
        '{"event": "rowset_init", "run_id": 9, "policy": "moe-row-set", '
        '"n_slots": 4, "slot_bytes": 452984832, "window_tokens": 1, '
        '"n_expert": 256, "top_k": 8, "t_us": 1}'
    ]
    for idx, (layer, chunk, ids, h) in enumerate(windows):
        if mutation == "hash" and idx == 0:
            h += 1
        if mutation == "order" and idx == 0:
            ids = "17,10,11,12,13,14,15,16"
        out.append(
            '{"event": "rowset_fetch", "run_id": 9, "ubatch": 1, '
            '"layer": %d, "slot": %d, "generation": 5, "chunk": %d, '
            '"phase": 1, "n_tokens": 1, "top_k": 8, "route_union": 8, '
            '"union_ids": "%s", "hash": %d, "n_rows_fetched": 8, '
            '"n_rows_resident": 0, "bytes_staged": 14155776, '
            '"bytes_full_bank": 452984832, "t_us": 2}'
            % (layer, layer % 4, chunk, ids, h)
        )
    out.append(
        '{"event": "rowset_round", "run_id": 9, "round": 1, "n_tokens": 1, '
        '"bytes_staged": 84934656, "bytes_full_bank": 452984832, '
        '"n_rows_fetched": 48, "n_rows_reused": 0, "t_us": 3}'
    )
    return "".join(line + "\n" for line in out)


def self_test() -> int:
    import contextlib
    import io

    tmp = tempfile.mkdtemp(prefix="route-overlap-selftest-")
    paths = {}
    for name, body in (
        ("good.jsonl", GOOD_JSONL),
        ("tok.jsonl", TOK_JSONL),
        ("bad.jsonl", BAD_JSONL),
        ("bad2.jsonl", BAD2_JSONL),
        ("rowset.jsonl", rowset_jsonl()),
        ("rowset_oracle.jsonl", rowset_jsonl(windows=_ORACLE_WINDOWS)),
        ("rowset_bad_hash.jsonl", rowset_jsonl("hash")),
        ("rowset_bad_order.jsonl", rowset_jsonl("order")),
    ):
        p = os.path.join(tmp, name)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        paths[name] = p

    failures = []
    checks = 0

    def check(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            failures.append(label)

    # -- good.jsonl: hand-computed expectations -----------------------------
    run_id, model, n_tg, routes, _rs = load_trace(paths["good.jsonl"])
    check(run_id == 7 and model == "M", "good: run_id/model")
    check(n_tg == 5, "good: tg count")

    widths = union_width_rows(routes)
    # layer 0 chunks: 254,254,100 -> min 100 mean 202.67 max 254
    # layer 1 chunks: 254,250,100 -> min 100 mean 201.33 max 254
    check(widths[0][:2] == (0, 100) and widths[0][3] == 254, "good: L0 width min/max")
    check(abs(widths[0][2] - 202.66666666666666) < 1e-9, "good: L0 width mean")
    check(widths[1][:2] == (1, 100) and widths[1][3] == 254, "good: L1 width min/max")

    t2 = temporal_window_rows(routes)
    # L0: 111==111 -> 1.0, 111!=222 -> 0.0 => 0.5 ; L1: 111!=333, 333!=222 => 0.0
    check(t2[0] == (0, 0.5), "good: L0 temporal 0.5")
    check(t2[1] == (1, 0.0), "good: L1 temporal 0.0")

    t3 = spatial_rows(routes)
    # per chunk: c0 L0(111) vs L1(111) eq ; c1 111 vs 333 ne ; c2 222 vs 222 eq -> 2/3
    check(abs(t3 - 2.0 / 3.0) < 1e-9, "good: spatial 2/3")

    t4 = advisory_sampled_overlap(routes)
    # containment |A n B| / |A|: L0: 1.0 (c0->c1 same set), 0.0 (c1->c2
    # disjoint) -> 0.5 ; L1: 0.0, 0.0 -> 0.0 ; overall (0.5 + 0)/2 = 0.25
    check(abs(t4 - 0.25) < 1e-9, "good: advisory containment 0.25")

    verdict, _detail, hit = token_temporal(routes, n_tg)
    check(
        verdict == TOKEN_VERDICT_INSUFFICIENT_INPUT and hit is None,
        "good: token verdict INSUFFICIENT_INPUT",
    )

    # -- tok.jsonl: per-token MEASURED path ----------------------------------
    _r, _m, n_tg2, routes2, _rs2 = load_trace(paths["tok.jsonl"])
    verdict2, _d2, hit2 = token_temporal(routes2, n_tg2)
    check(verdict2 == TOKEN_VERDICT_MEASURED, "tok: verdict MEASURED")
    # pairs: (1000,1001) ne, (1001,1002) ne -> 0.0
    check(hit2 == 0.0, "tok: hit rate 0.0")

    # -- loud failures --------------------------------------------------------
    def expect_loud(fn, label, needle):
        buf = io.StringIO()
        code = 0
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(io.StringIO()):
            try:
                fn()
            except SystemExit as e:
                code = int(e.code or 0)
        check(code == EXIT_LOUD, f"{label}: exit 2")
        check(needle in buf.getvalue(), f"{label}: message mentions {needle!r}")

    expect_loud(
        lambda: load_trace(paths["bad.jsonl"]),
        "bad: missing route fields",
        "route event missing field(s): hash",
    )
    expect_loud(
        lambda: load_trace(paths["bad2.jsonl"]),
        "bad2: malformed JSON",
        "malformed JSON",
    )
    expect_loud(
        lambda: load_trace(os.path.join(tmp, "nope.jsonl")),
        "missing file",
        "cannot read trace file",
    )

    # -- row-set fixture: T6/T7 hand-computed --------------------------------
    _r, _m, _n, _rt, _rowset = load_trace(paths["rowset.jsonl"])
    check(
        _rowset["round"] == 1 and _rowset["refused"] == 0,
        "rowset: round/refused counts",
    )
    check(_rowset["init"]["n_expert"] == 256, "rowset: rowset_init captured")
    series = rowset_series(_rowset["fetch"])
    check(
        len(series) == 2 and all(len(w) == 3 for w in series.values()),
        "rowset: 2 layers x 3 windows",
    )

    b = resident_containment(series, (1, 2), 256, 8, 1, 1769472)
    check(b[1]["capacity_rows_per_layer"] == 8, "rowset: K=1 capacity 8 rows")
    check(b[2]["capacity_rows_per_layer"] == 16, "rowset: K=2 capacity 16 rows")
    check(b[1]["value"] == 0.0, "rowset: (b) K=1 == 0.0")
    check(abs(b[2]["value"] - 0.5) < 1e-9, "rowset: (b) K=2 == 0.5")
    check(
        b[2]["windows_with_predecessor"] == 4,
        "rowset: 4 windows with a predecessor",
    )
    check(
        b[1]["capacity_bytes_total"] == 8 * 1769472 * 2,
        "rowset: K=1 capacity bytes over 2 layers",
    )

    ec = exact_consecutive_containment(series)
    check(
        ec["pairs"] == 4 and ec["prev_in_next"] == 0.0 and ec["next_in_prev"] == 0.0,
        "rowset: exact consecutive containment 0.0 both directions",
    )
    check(
        abs(b[1]["value"] - ec["prev_in_next"]) < 1e-9,
        "rowset: (b) at K=1 equals consecutive-window containment",
    )

    # Policy divergence fixture. A={1..8}, B={9..16}, windows [A, B, A, A, B].
    # At K=2 both rules see the same stream but bound it differently: the
    # row-capacity LRU still holds B at window 5 (its 16 slots filled at
    # window 2), while the K-window union evicted B at window 4 (age 2 > K-1),
    # so window 5 is 8 hits for the LRU and 0 for the union -- 0.75 against
    # 0.50 pooled. That spread IS the two readings of SDD gate 9 at K=2; the
    # self-test pins it so neither can be changed silently.
    _a = list(range(1, 9))
    _b = list(range(9, 17))
    _pser = {0: [(i, ids, 0) for i, ids in enumerate([_a, _b, _a, _a, _b])]}
    _bl = resident_containment(_pser, (1, 2), 256, 8, 1)
    _bu = resident_containment(
        _pser, (1, 2), 256, 8, 1, policy=RETENTION_POLICY_UNION
    )
    check(abs(_bl[2]["value"] - 0.75) < 1e-9, "rowset: policy LRU K=2 == 0.75")
    check(abs(_bu[2]["value"] - 0.5) < 1e-9, "rowset: policy UNION K=2 == 0.5")
    check(
        abs(_bl[1]["value"] - _bu[1]["value"]) < 1e-9
        and abs(_bl[1]["value"] - 0.25) < 1e-9,
        "rowset: the two policies agree exactly at K=1",
    )
    check(
        _bu[1]["policy"] == RETENTION_POLICY_UNION
        and _bl[1]["policy"] == RETENTION_POLICY_LRU,
        "rowset: policy recorded in the curve",
    )

    rap = reappearance_histogram(series)
    check(rap["gaps"] == {"2": 16}, "rowset: reappearance gaps {2: 16}")
    check(
        rap["rows_seen_once"] == 16 and rap["unique_layer_rows"] == 32,
        "rowset: rows seen once 16/32",
    )
    check(
        rap["coverage_by_retention_windows"]["1"] == 0.0
        and rap["coverage_by_retention_windows"]["2"] == 1.0,
        "rowset: coverage at 1/2 windows",
    )

    verdict_r, _dr, hit_r = token_temporal(_rt, _n, series)
    check(
        verdict_r == TOKEN_VERDICT_MEASURED and hit_r == 0.0,
        "rowset: T5 re-sourced (0/4 window-equality)",
    )

    rep_rowset = build_report([("rowset.jsonl", load_trace(paths["rowset.jsonl"]))])
    blk = rep_rowset[0]["rowset"]
    check(
        blk["n_fetch"] == 6 and blk["hash_recompute_ok"] == 6,
        "rowset: report fetch/recompute counts",
    )
    check(
        blk["verdict_statistic_b"][1]["value"] == 0.0
        and abs(blk["verdict_statistic_b"][2]["value"] - 0.5) < 1e-9,
        "rowset: report carries the K curve",
    )
    check(
        rep_rowset[0]["temporal_token_verdict"] == TOKEN_VERDICT_MEASURED,
        "rowset: report T5 MEASURED",
    )

    # -- T8 oracle ceiling: bounded retention vs unbounded -------------------
    _r3, _m3, _n3, _rt3, _ro3 = load_trace(paths["rowset_oracle.jsonl"])
    ser3 = rowset_series(_ro3["fetch"])
    b3 = resident_containment(ser3, (1, 2), 256, 8, 1)
    check(b3[1]["value"] == 0.0, "oracle fixture: K=1 holds nothing (0.0)")
    check(b3[2]["value"] == 0.0, "oracle fixture: K=2 evicts before the reuse")
    orc3 = oracle_containment(ser3)
    check(
        abs(orc3["value"] - 1.0 / 3.0) < 1e-9,
        "oracle fixture: unbounded ceiling is 1/3",
    )
    check(
        orc3["hits_total"] == 8 and orc3["required_total"] == 24,
        "oracle fixture: pooled hits/required 8/24",
    )
    check(
        orc3["below_gate9_threshold"] is True,
        "oracle fixture: ceiling below the gate threshold",
    )
    rep3 = build_report(
        [("rowset_oracle.jsonl", load_trace(paths["rowset_oracle.jsonl"]))]
    )
    check(
        rep3[0]["rowset"]["ceiling_saturated_by_k"] is None,
        "oracle fixture: reported K band does NOT saturate the ceiling",
    )
    # The 3-window fixture is the mirror image: its union never exceeds the
    # K=2 capacity, so the oracle is EXACTLY the K=2 value and must be detected
    # as saturated -- by totals, not by comparing rounded means.
    blk2 = rep_rowset[0]["rowset"]
    check(
        blk2["ceiling_saturated_by_k"] == 2,
        "3-window fixture: K=2 exactly saturates the oracle",
    )
    check(
        blk2["oracle_containment"]["value"]
        == blk2["verdict_statistic_b"][2]["value"],
        "3-window fixture: oracle equals K=2 (capacity 16 cannot evict)",
    )

    expect_loud(
        lambda: load_trace(paths["rowset_bad_hash.jsonl"]),
        "rowset: hash mismatch",
        "hash mismatch",
    )
    expect_loud(
        lambda: load_trace(paths["rowset_bad_order.jsonl"]),
        "rowset: non-canonical union_ids",
        "not the canonical",
    )
    expect_loud(
        lambda: main(
            ["--retention-k", "0", paths["rowset.jsonl"]]
        ),
        "rowset: retention-k < 1",
        "--retention-k",
    )

    # -- determinism of the renderer ------------------------------------------
    rep = build_report([(paths["good.jsonl"], load_trace(paths["good.jsonl"]))])
    txt1 = render_txt(rep)
    rep2 = build_report([(paths["good.jsonl"], load_trace(paths["good.jsonl"]))])
    txt2 = render_txt(rep2)
    check(txt1 == txt2, "render deterministic")

    if failures:
        for fl in failures:
            print(f"FAIL: {fl}", file=sys.stderr)
        print(f"self-test: {len(failures)} failure(s)", file=sys.stderr)
        return 1
    print(f"self-test: PASS ({checks} checks)")
    return 0


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main(argv):
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.traces:
        fail("no trace paths given (positional 'traces' is required)")

    try:
        retention_ks = tuple(
            int(x) for x in str(args.retention_k).split(",") if x.strip()
        )
    except ValueError:
        fail(
            "--retention-k must be comma-separated integers, got "
            f"{args.retention_k!r}"
        )
    if not retention_ks or any(k < 1 for k in retention_ks):
        fail(f"--retention-k values must be >= 1, got {args.retention_k!r}")

    # expand glob patterns, including nested directory components
    # (e.g. build/llama-cuda/gplr13-ladder/*/rowset-*.jsonl)
    trace_paths = []
    for pat in args.traces:
        try:
            matches = sorted(glob.glob(pat))
        except OSError as e:
            fail(f"trace pattern could not be expanded: {pat}: {e}")
        if not matches:
            fail(f"trace pattern matched no files: {pat}")
        for m in matches:
            if m not in trace_paths:
                trace_paths.append(m)

    parsed = []
    for p in trace_paths:
        parsed.append((os.path.basename(p), load_trace(p)))

    report = build_report(parsed, retention_ks, args.retention_policy)
    txt = render_txt(
        report,
        json_basename=os.path.basename(args.json_out) if args.json_out else None,
    )
    # JSON payload keeps the recorded 10-key schema per trace record (the
    # hit rate, when MEASURED, is surfaced in the txt T5 line only).
    payload_report = [
        {k: v for k, v in r.items() if k != "temporal_token_hit_rate"}
        for r in report
    ]
    payload = json.dumps(payload_report, indent=2, sort_keys=True) + "\n"

    if args.txt:
        # Default text mode (platform newline) matches the recorded 2026-09-09
        # artifacts byte-for-byte on Windows (CRLF).
        with open(args.txt, "w", encoding="utf-8") as f:
            f.write(txt)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(payload)
    if not args.txt and not args.json_out:
        sys.stdout.write(txt)

    underspecified = [
        r for r in report if r["temporal_token_verdict"] != TOKEN_VERDICT_MEASURED
    ]
    if underspecified:
        print(
            f"[{PROG}] NOTE: {len(underspecified)}/{len(report)} trace(s) could "
            "not yield the token-level temporal statistic (verdicts: "
            + ", ".join(sorted({r['temporal_token_verdict'] for r in underspecified}))
            + "); see report T5 rows",
            file=sys.stderr,
        )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
