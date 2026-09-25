"""T-GPLR-60 G0/G1 host-only learning-curve replay.

Scores prefix-trained predictors on a causal held-out suffix.  It uses no model,
CUDA, GPU, mutex, network, or third-party dependency.  The source is a recorded
Qwen3.6 calculator-prompt demand trace; predictor state is trained only on rows
before each prefix and is frozen for the held-out suffix.
"""
from __future__ import annotations

import argparse
import bisect
import datetime
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_PATH = ROOT / "tools" / "predictor-screen-gplr.py"
DEFAULT_TRACE = ROOT / "build" / "llama-cuda" / "gplr58-fresh-coact-k8-admission.jsonl"
PROMPT = ROOT / "build" / "calc-prompt.txt"
EXPECTED_PROMPT_SHA256 = "e29c2fcc0da0815554c45e49026d4f3e24df6401fa492fe1b26e6286b00011d9"
DEFAULT_OUT = ROOT / "build" / "llama-cuda" / "tg60-learning-curve.json"
PREFIXES = (32, 64, 128, 256)
KS = (1, 2, 4, 8)
POOL_BUDGET = 8


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


import tg_pool_core as analyzer

predictor = load_module("predictor_screen", PREDICTOR_PATH)


def fail(message):
    raise analyzer.Refusal(message)


def trace_rows(path):
    path = Path(path)
    if not path.is_file():
        fail("trace file not found: %s" % path)
    try:
        init, rows = predictor.load_cache_rows(str(path))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        fail("invalid trace %s: %s" % (path, exc))
    if int(init.get("top_k", 0)) != 8:
        fail("unexpected top_k=%s" % init.get("top_k"))
    return init, rows


def load_leg(path):
    path = Path(path)
    if not path.is_file():
        fail("trace file not found: %s" % path)
    try:
        return analyzer.load_trace(str(path), "demand-only")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        fail("invalid trace %s: %s" % (path, exc))


def validate_alignment(rows, stages):
    if not rows:
        fail("empty demand trace")
    if len(rows) != len(stages):
        fail("trace row/stage count mismatch: %d != %d" % (len(rows), len(stages)))
    for index, (row, stage) in enumerate(zip(rows, stages)):
        expected = (row["token"], row["layer"], row["ids"])
        actual = (stage["ubatch"], stage["layer"], stage["union"])
        if expected != actual:
            fail("trace row/stage order or ids mismatch at index %d" % index)


def first_run_start(path):
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if '"event":"run_start"' not in line and '"event": "run_start"' not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") == "run_start":
            return ev
    fail("no run_start in %s" % path)


def candidate_fuse(lists, budget=POOL_BUDGET):
    votes = Counter()
    scores = defaultdict(float)
    for candidates in lists:
        for rank, expert in enumerate(dict.fromkeys(candidates)):
            votes[expert] += 1
            scores[expert] += 0.9 ** rank
    return sorted(votes, key=lambda e: (-votes[e], -scores[e], e))[:budget]


class XorShift32:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state


def stable_arm_seed(name, prefix_tokens, row_index):
    """Stable across processes; Python's hash(name) is deliberately not used."""
    digest = hashlib.sha256(("tg60-null:%s:%d:%d" % (name, prefix_tokens, row_index)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def null_candidates(count, n_expert, seed):
    rng = XorShift32(seed)
    out = []
    seen = set()
    need = min(POOL_BUDGET, int(count), int(n_expert))
    while len(out) < need:
        expert = rng.next() % n_expert
        if expert not in seen:
            seen.add(expert)
            out.append(expert)
    return out


def row_bytes_by_layer(stages):
    values = defaultdict(set)
    for stage in stages:
        values[stage["layer"]].add(int(stage["row_bytes"]))
    for layer, sizes in values.items():
        if len(sizes) != 1:
            fail("inconsistent row_bytes for layer %d" % layer)
    return {str(layer): next(iter(sizes)) for layer, sizes in sorted(values.items())}


def transfer_accounting(pool, real_miss_h2d_bytes):
    pool_h2d = int(pool.get("h2d_bytes", 0))
    useful = int(pool.get("d2d_bytes", 0))
    if useful > pool_h2d:
        fail("useful transfer bytes exceed pool H2D bytes")
    return {
        "real_miss_h2d_bytes": int(real_miss_h2d_bytes),
        "pool_h2d_bytes": pool_h2d,
        "useful_h2d_bytes": useful,
        "wasted_h2d_bytes": pool_h2d - useful,
        "avoided_demand_h2d_bytes": useful,
        "net_modeled_h2d_bytes": int(real_miss_h2d_bytes) + pool_h2d - useful,
    }


def make_prefix_predictors(prefix_rows):
    objects = {
        "coact": predictor.Coact(),
        "bar09": predictor.Barred(predictor.Markov1(), 0.9),
        "fusion": predictor.VoteFusion(predictor.CANDIDATES),
    }
    decay_scores = defaultdict(Counter)
    prefix_by_layer = defaultdict(list)
    for row in prefix_rows:
        layer = row["layer"]
        ids = row["ids"]
        for obj in objects.values():
            obj.observe(layer, ids)
        # Exponential half-life score: each observation is weighted by 0.9 ** age.
        # This is prefix-only and deterministic; it is not a runtime claim.
        age = max(0, prefix_rows[-1]["token_pos"] - row["token_pos"])
        weight = 0.9 ** age
        for expert in ids:
            decay_scores[layer][expert] += weight
        prefix_by_layer[layer].append(set(ids))
    return objects, decay_scores, prefix_by_layer


def make_candidate_streams(rows, prefix_tokens):
    prefix_rows = [r for r in rows if r["token_pos"] < prefix_tokens]
    test_rows = [r for r in rows if r["token_pos"] >= prefix_tokens]
    objects, decay_scores, prefix_by_layer = make_prefix_predictors(prefix_rows)
    stream_names = ("coact", "bar09", "fusion", "super", "decay_frequency", "route_knn")
    streams = {name: {} for name in stream_names}
    nulls = {name: {} for name in stream_names}
    for row_index, row in enumerate(test_rows):
        token = row["token"]
        layer = row["layer"]
        key = (token, layer)
        arm_lists = {}
        for name, obj in objects.items():
            ranked = list(obj.predict(layer, POOL_BUDGET))
            arm_lists[name] = ranked
            streams[name][key] = ranked
        streams["super"][key] = candidate_fuse(
            [arm_lists["coact"], arm_lists["bar09"], arm_lists["fusion"]]
        )
        counts = decay_scores[layer]
        streams["decay_frequency"][key] = [
            e for e, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:POOL_BUDGET]
        ]
        # Frozen request-local context: the last prefix route is the query. No
        # held-out route is read, so this is not a future-leaking KNN predictor.
        query = prefix_by_layer[layer][-1] if prefix_by_layer[layer] else None
        if query is None:
            streams["route_knn"][key] = []
        else:
            best = max(
                prefix_by_layer[layer],
                key=lambda prior: (len(query & prior) / max(1, len(query | prior)), -sum(prior)),
            )
            streams["route_knn"][key] = sorted(best)
        for name in stream_names:
            count = len(streams[name][key])
            nulls[name][key] = null_candidates(count, 256, stable_arm_seed(name, prefix_tokens, row_index))
    return streams, nulls, len(prefix_rows), len(test_rows)


def pool_replay_tail(facts, start_index, candidate_map, capacity=POOL_BUDGET):
    totals, lead_hist = analyzer.replay_pool_stages(
        facts[start_index:],
        lambda stage: candidate_map.get(stage["key"], []),
        n_expert=256,
        capacity=capacity,
    )
    totals["lead_hist"] = dict(sorted(lead_hist.items()))
    return totals


def belady_hits(facts, capacity, scored_start=0):
    """Exact Belady ceiling over demand order, independently per layer.

    ``scored_start`` lets the prefix warm the oracle cache while hits are counted
    only on the held-out suffix. Future uses are used only to choose the classic
    Belady victim; no future demand is exposed to a predictor.
    """
    positions = defaultdict(lambda: defaultdict(list))
    for i, stage in enumerate(facts):
        for expert in set(stage["union"]):
            positions[stage["layer"]][expert].append(i)
    cache = defaultdict(set)
    hits = 0
    demand_ids = 0
    for i, stage in enumerate(facts):
        layer = stage["layer"]
        demanded = set(stage["union"])
        rows = cache[layer]
        if i >= scored_start:
            demand_ids += len(demanded)
            hits += sum(expert in rows for expert in demanded)
        # Belady evicts the resident row whose next use is furthest away. The
        # current demand is protected; capacity is in rows, not hit denominators.
        while len(rows) + len(demanded - rows) > capacity:
            def next_use(expert):
                seq = positions[layer][expert]
                j = bisect.bisect_right(seq, i)
                return seq[j] if j < len(seq) else 10**18
            victims = [expert for expert in rows if expert not in demanded]
            if not victims:
                fail("Belady cannot protect a demand that exceeds capacity")
            victim = max(victims, key=lambda expert: (next_use(expert), expert))
            rows.remove(victim)
        rows.update(demanded)
        if len(rows) > capacity:
            fail("Belady capacity invariant failed")
    if hits > demand_ids:
        fail("Belady hit count exceeds demand count")
    return hits


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_prompt(path=PROMPT, expected=EXPECTED_PROMPT_SHA256):
    path = Path(path)
    if not path.is_file():
        fail("prompt file not found: %s" % path)
    actual = sha256(path)
    if actual != expected:
        fail("prompt hash mismatch: expected %s, got %s" % (expected, actual))
    return actual


def selftest():
    checks = 0
    def check(value, name):
        nonlocal checks
        if not value:
            raise AssertionError(name)
        checks += 1
    check(candidate_fuse([[1, 2], [2, 3], [3, 1]]) == [1, 2, 3], "vote fusion dedup/order")
    check(len(set(null_candidates(8, 16, 12345))) == 8, "null uniqueness")
    check(stable_arm_seed("coact", 32, 7) == stable_arm_seed("coact", 32, 7), "null seed stability")
    check(stable_arm_seed("coact", 32, 7) != stable_arm_seed("bar09", 32, 7), "null arm separation")
    check(validate_prompt() == EXPECTED_PROMPT_SHA256, "fixed prompt identity")
    try:
        validate_prompt(PROMPT, "0" * 64)
    except analyzer.Refusal:
        checks += 1
    else:
        raise AssertionError("prompt hash mismatch accepted")
    try:
        validate_alignment([], [])
    except analyzer.Refusal:
        checks += 1
    else:
        raise AssertionError("empty trace accepted")
    toy = [{"key": (0, 0), "layer": 0, "union": [1], "generation": 0, "phase": 0, "row_bytes": 1, "apply_proposal": True, "real_rows_after": frozenset(), "miss_ids": [1], "occ": 0}, {"key": (1, 0), "layer": 0, "union": [2], "generation": 0, "phase": 0, "row_bytes": 1, "apply_proposal": True, "real_rows_after": frozenset(), "miss_ids": [2], "occ": 1}, {"key": (2, 0), "layer": 0, "union": [1], "generation": 0, "phase": 0, "row_bytes": 1, "apply_proposal": True, "real_rows_after": frozenset(), "miss_ids": [1], "occ": 2}]
    check(belady_hits(toy, 1) == 0, "Belady basic sequence")
    check(belady_hits(toy, 2) == 1, "Belady exact reuse")
    check(row_bytes_by_layer([
        {"layer": 0, "row_bytes": 10}, {"layer": 1, "row_bytes": 20},
    ]) == {"0": 10, "1": 20}, "per-layer row bytes")
    try:
        validate_alignment(
            [{"token": 2, "layer": 0, "ids": [1]}, {"token": 1, "layer": 0, "ids": [2]}],
            [{"ubatch": 1, "layer": 0, "union": [2]}, {"ubatch": 2, "layer": 0, "union": [1]}],
        )
    except analyzer.Refusal:
        checks += 1
    else:
        raise AssertionError("out-of-order rows accepted")
    check(transfer_accounting({"h2d_bytes": 30, "d2d_bytes": 20}, 40) == {
        "real_miss_h2d_bytes": 40, "pool_h2d_bytes": 30,
        "useful_h2d_bytes": 20, "wasted_h2d_bytes": 10,
        "avoided_demand_h2d_bytes": 20, "net_modeled_h2d_bytes": 50,
    }, "layer-aware transfer accounting")
    prefix = [{"token": 0, "token_pos": 0, "layer": 0, "ids": [1, 2]}]
    suffix_a = [{"token": 1, "token_pos": 1, "layer": 0, "ids": [3, 4]}]
    suffix_b = [{"token": 1, "token_pos": 1, "layer": 0, "ids": [9, 10]}]
    a, _, _, _ = make_candidate_streams(prefix + suffix_a, 1)
    b, _, _, _ = make_candidate_streams(prefix + suffix_b, 1)
    check(all(a[name][(1, 0)] == b[name][(1, 0)] for name in a), "prefix isolation/no future leak")
    facts = [{"key": (0, 0), "layer": 0, "union": [1], "generation": 0, "phase": 0, "row_bytes": 10, "apply_proposal": True, "real_rows_after": frozenset(), "miss_ids": [1], "occ": 0}]
    one = pool_replay_tail(facts, 0, {(0, 0): list(range(9))})
    check(one["admitted"] == 8 and one["refused_budget"] == 1, "per-stage proposal budget")
    lead_facts = [
        {"key": (0, 0), "layer": 0, "union": [1], "generation": 0, "phase": 0, "row_bytes": 1, "apply_proposal": True, "real_rows_after": frozenset(), "miss_ids": [1], "occ": 0},
        {"key": (1, 0), "layer": 0, "union": [2], "generation": 0, "phase": 0, "row_bytes": 1, "apply_proposal": True, "real_rows_after": frozenset(), "miss_ids": [2], "occ": 1},
        {"key": (2, 0), "layer": 0, "union": [1], "generation": 0, "phase": 0, "row_bytes": 1, "apply_proposal": True, "real_rows_after": frozenset(), "miss_ids": [1], "occ": 2},
    ]
    lead = pool_replay_tail(lead_facts, 0, {(0, 0): [1], (1, 0): [], (2, 0): []})
    check(lead["hits"] == 1 and lead["lead_hist"] == {2: 1}, "ideal-next-reap lead floor")
    print("selftest: %d checks, 0 failures" % checks)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(DEFAULT_TRACE))
    ap.add_argument("--json", default=str(DEFAULT_OUT))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    trace = Path(args.trace)
    prompt = PROMPT
    prompt_hash = validate_prompt(prompt)
    init, rows = trace_rows(trace)
    token_values = sorted(set(r["token"] for r in rows))
    token_rank = {token: i for i, token in enumerate(token_values)}
    for row in rows:
        row["token_pos"] = token_rank[row["token"]]
    leg = load_leg(trace)
    validate_alignment(rows, leg["stages"])
    row_sizes = row_bytes_by_layer(leg["stages"])
    # G0 provenance and equal-demand checks. The command's trace is part of
    # the checked set; the two recorded arms establish equal demand identity.
    source_paths = [trace, ROOT / "build" / "llama-cuda" / "gplr58-a4-bar09.jsonl", ROOT / "build" / "llama-cuda" / "gplr58-a4-fusion.jsonl"]
    signatures = {}
    for path in source_paths:
        if not path.is_file():
            fail("missing provenance source: %s" % path)
        other = leg if path == trace else load_leg(path)
        signatures[str(path)] = {
            "stream_signature": analyzer.stream_signature(other["stages"]),
            "demand_rows_signature": analyzer.demand_rows_signature(other["stages"]),
            "n_stages": len(other["stages"]),
            "demand_ids": sum(len(s["union"]) for s in other["stages"]),
        }
        if other["n_expert"] != leg["n_expert"] or other["top_k"] != leg["top_k"]:
            fail("source shape mismatch: %s" % path)
    if len(set(v["demand_rows_signature"] for v in signatures.values())) > 1:
        fail("source demand streams differ")
    run_start = first_run_start(trace)
    output = {
        "schema": "tg60-learning-curve/1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "g0": {
            "trace": str(trace),
            "trace_sha256": sha256(trace),
            "prompt_file": str(prompt),
            "prompt_sha256": prompt_hash,
            "model": leg["model"],
            "n_expert": leg["n_expert"],
            "top_k": leg["top_k"],
            "row_bytes_by_layer": row_sizes,
            "run_start": {k: run_start.get(k) for k in ("pp_tokens", "ctx_size", "tg_tokens", "row_spec_budget", "row_spec_mode", "row_spec_arm", "row_spec_null", "rep_penalty", "kv_cache_type", "n_slots") if k in run_start},
            "signatures": signatures,
            "online_faithfulness": "predictors train only on token < prefix and are frozen for held-out token >= prefix; no future rows are read",
        },
        "prefixes": [],
        "limits": [
            "H=256 has no held-out suffix in this 256-token trace and is UNVERIFIED_NO_HELDOUT, not zero",
            "decay_frequency is an exponential prefix score with age weight 0.9**age",
            "route_knn is a frozen nearest-prefix Jaccard match; it is not a learned or suffix-updated KNN",
            "equal-row retention is a separate K+1-window LRU replay; the predictor pool is a separate 8-position overlay, so the row-budget comparison is logical rather than a physical byte-layout result",
            "Belady is an oracle cache ceiling using future demand only to choose victims; it is not a deployable predictor",
            "no CUDA, transfer timing, or tok/s is measured",
        ],
    }
    for h in PREFIXES:
        streams, nulls, n_prefix, n_test = make_candidate_streams(rows, h)
        test_start = next((i for i, s in enumerate(leg["stages"]) if token_rank[s["ubatch"]] >= h), len(leg["stages"]))
        entry = {"prefix_tokens": h, "train_rows": n_prefix, "heldout_rows": n_test, "status": "scored" if n_test else "UNVERIFIED_NO_HELDOUT", "results": []}
        if n_test:
            demand_ids = sum(len(s["union"]) for s in leg["stages"][test_start:])
            for k in KS:
                facts = analyzer.replay_real_facts(
                    leg["stages"], min(leg["n_expert"], k * leg["top_k"]))
                real_hits = sum(len(s["real_hit_ids"]) for s in facts[test_start:])
                real_miss_ids = sum(len(s["miss_ids"]) for s in facts[test_start:])
                baseline_facts = analyzer.replay_real_facts(
                    leg["stages"], min(leg["n_expert"], (k + 1) * leg["top_k"]))
                baseline_hits = sum(len(s["real_hit_ids"]) for s in baseline_facts[test_start:])
                capacity_rows = min(leg["n_expert"], (k + 1) * leg["top_k"])
                oracle_hits = belady_hits(leg["stages"], capacity_rows, test_start)
                for arm in ("coact", "bar09", "fusion", "super", "decay_frequency", "route_knn"):
                    pool = pool_replay_tail(facts, test_start, streams[arm])
                    null_pool = pool_replay_tail(facts, test_start, nulls[arm])
                    real_miss_h2d = sum(len(s["miss_ids"]) * s["row_bytes"] for s in facts[test_start:])
                    transfer = transfer_accounting(pool, real_miss_h2d)
                    admitted = pool.get("admitted", 0)
                    pool_hits = pool.get("hits", 0)
                    if pool_hits > real_miss_ids or real_hits + pool_hits > demand_ids:
                        fail("pool hit accounting exceeded demand for %s K%d" % (arm, k))
                    if null_pool.get("hits", 0) > real_miss_ids:
                        fail("null hit accounting exceeded real misses")
                    entry["results"].append({
                        "k_real": k, "arm": arm, "demand_ids": demand_ids,
                        "real_cache_hits": real_hits,
                        "real_miss_ids": real_miss_ids,
                        "pool_hits": pool_hits,
                        "pending_fallbacks": pool.get("pending_fallbacks", 0),
                        "pool_admitted": admitted,
                        "pool_proposals": pool.get("proposals", 0),
                        "pool_fates": pool,
                        **transfer,
                        "lead_hist": pool.get("lead_hist", {}),
                        "total_hit_share": (real_hits + pool_hits) / demand_ids,
                        "marginal_pool_share": pool_hits / demand_ids,
                        "equal_row_baseline_k": k + 1,
                        "equal_row_baseline_capacity_rows": capacity_rows,
                        "equal_row_baseline_hits": baseline_hits,
                        "equal_row_baseline_share": baseline_hits / demand_ids,
                        "belady_hits": oracle_hits,
                        "belady_baseline_share": oracle_hits / demand_ids,
                        "null_pool_hits": null_pool.get("hits", 0),
                        "null_marginal_share": null_pool.get("hits", 0) / demand_ids,
                        "null_pool_fates": null_pool,
                    })
        output["prefixes"].append(entry)
    Path(args.json).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.json)
    for entry in output["prefixes"]:
        print("prefix", entry["prefix_tokens"], entry["status"], "heldout_rows", entry["heldout_rows"])
        if entry["results"]:
            for row in entry["results"]:
                if row["arm"] in ("coact", "bar09", "fusion", "super"):
                    print(" K%d %-14s total=%.2f%% baseline=%.2f%% pool=%d null=%s" % (row["k_real"], row["arm"], 100*row["total_hit_share"], 100*row["equal_row_baseline_share"], row["pool_hits"], row["null_pool_hits"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except analyzer.Refusal as exc:
        print("REFUSED (no measurement): %s" % exc, file=sys.stderr)
        raise SystemExit(2)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("REFUSED (no measurement): invalid input: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
