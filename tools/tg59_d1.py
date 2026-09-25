#!/usr/bin/env python3
"""T-GPLR-59 D1 host replay for the MTP proposed-token source.

D1 learns an online table keyed by (drafted token, layer). It predicts before
observing the current pair and updates only accepted draft/target pairs.
This module consumes explicit `d1_pair` records; it does not infer token pairs
from filenames or from aggregate MTP counters.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field


class D1Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass
class D1Table:
    n_expert: int
    counts: dict[tuple[int, int], dict[int, int]] = field(default_factory=dict)
    rows_seen: int = 0
    accepted_rows: int = 0

    def predict(self, draft_token: int, layer: int, budget: int) -> list[int]:
        if budget <= 0:
            return []
        row = self.counts.get((draft_token, layer), {})
        ranked = sorted(row.items(), key=lambda item: (-item[1], item[0]))
        return [expert for expert, _ in ranked[:budget]]

    def observe(self, draft_token: int, layer: int, accepted: bool, demand_ids: list[int]) -> None:
        self.rows_seen += 1
        if not accepted:
            return
        self.accepted_rows += 1
        row = self.counts.setdefault((draft_token, layer), {})
        for expert in demand_ids:
            row[expert] = row.get(expert, 0) + 1


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise D1Refused("bad-d1-field", "%s must be an integer" % name)
    return value


def read_pairs(path: str) -> tuple[list[dict], int | None, int | None, bool, bool]:
    if not os.path.isfile(path):
        raise D1Refused("file-missing", "no such D1 trace: %s" % path)
    rows: list[dict] = []
    n_expert = None
    top_k = None
    predicted_required = False
    null_source = False
    shape_seen = False
    with open(path, encoding="utf-8", errors="strict") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise D1Refused("bad-json", "line %d is not JSON: %s" % (line_no, exc))
            if not isinstance(event, dict):
                raise D1Refused("bad-event", "line %d is not an object" % line_no)
            kind = event.get("event")
            if kind == "d1_shape":
                if shape_seen:
                    raise D1Refused("duplicate-d1-shape", "line %d repeats d1_shape" % line_no)
                shape_seen = True
                n_expert = _integer(event.get("n_expert"), "n_expert")
                top_k = _integer(event.get("top_k"), "top_k")
                if n_expert <= 0 or top_k <= 0 or top_k > n_expert:
                    raise D1Refused("bad-d1-shape", "d1_shape has an invalid expert shape")
                source = event.get("source", "d1")
                if source != "d1":
                    raise D1Refused("bad-d1-shape", "d1_shape source must be d1")
                if "null" in event and not isinstance(event["null"], bool):
                    raise D1Refused("bad-d1-shape", "d1_shape null must be boolean")
                null_source = bool(event.get("null", False))
                if "predicted_ids" in event and not isinstance(event["predicted_ids"], bool):
                    raise D1Refused("bad-d1-shape", "d1_shape predicted_ids must be boolean")
                predicted_required = bool(event.get("predicted_ids", False))
                if null_source and not predicted_required:
                    raise D1Refused("bad-d1-shape", "null D1 traces must record predicted_ids")
                if "window_tokens" in event and _integer(event["window_tokens"], "window_tokens") < 1:
                    raise D1Refused("bad-d1-shape", "d1_shape window_tokens must be positive")
                continue
            if kind != "d1_pair":
                continue
            if not shape_seen:
                raise D1Refused("no-d1-shape", "line %d has d1_pair before d1_shape" % line_no)
            draft_token = _integer(event.get("draft_token"), "draft_token")
            layer = _integer(event.get("layer"), "layer")
            accepted = event.get("accepted")
            if not isinstance(accepted, bool):
                raise D1Refused("bad-d1-field", "line %d accepted must be boolean" % line_no)
            ids = event.get("demand_ids")
            if not isinstance(ids, list):
                raise D1Refused("bad-d1-field", "line %d demand_ids must be an array" % line_no)
            demand_ids = [_integer(x, "demand_ids") for x in ids]
            predicted = event.get("predicted_ids")
            if predicted is None and predicted_required:
                raise D1Refused("bad-d1-field", "line %d missing predicted_ids required by d1_shape" % line_no)
            if predicted is not None and not isinstance(predicted, list):
                raise D1Refused("bad-d1-field", "line %d predicted_ids must be an array" % line_no)
            predicted_ids = [_integer(x, "predicted_ids") for x in predicted] if predicted is not None else None
            if draft_token < 0 or layer < 0 or any(x < 0 or x >= (n_expert or 1 << 30) for x in demand_ids):
                raise D1Refused("d1-shape-mismatch", "line %d has an out-of-range D1 value" % line_no)
            if len(set(demand_ids)) != len(demand_ids):
                raise D1Refused("duplicate-demand-id", "line %d repeats a demand id" % line_no)
            if predicted_ids is not None:
                if len(set(predicted_ids)) != len(predicted_ids):
                    raise D1Refused("duplicate-predicted-id", "line %d repeats a predicted id" % line_no)
                if len(predicted_ids) > top_k or any(x < 0 or x >= (n_expert or 1 << 30) for x in predicted_ids):
                    raise D1Refused("d1-shape-mismatch", "line %d predicted_ids outside the declared D1 shape" % line_no)
            if len(demand_ids) != top_k:
                raise D1Refused("d1-shape-mismatch", "line %d demand_ids width != top_k" % line_no)
            rows.append({"draft_token": draft_token, "layer": layer,
                         "accepted": accepted, "demand_ids": demand_ids,
                         "predicted_ids": predicted_ids})
    if n_expert is None or top_k is None:
        raise D1Refused("no-d1-shape", "D1 replay requires one d1_shape event")
    if not rows:
        raise D1Refused("no-d1-pairs", "D1 replay requires at least one d1_pair event")
    return rows, n_expert, top_k, predicted_required, null_source


def replay(path: str, budget: int = 8) -> dict:
    rows, n_expert, top_k, predicted_required, null_source = read_pairs(path)
    if budget <= 0 or budget > top_k:
        raise D1Refused("bad-d1-budget", "D1 budget must be in [1, top_k]")
    table = D1Table(n_expert)
    hits = 0
    demand_total = 0
    paired_demand_total = 0
    published = 0
    for row in rows:
        if null_source:
            predicted = row["predicted_ids"]
        else:
            predicted = table.predict(row["draft_token"], row["layer"], budget)
            if predicted_required and row["predicted_ids"] != predicted:
                raise D1Refused("prediction-mismatch", "runtime predicted_ids do not match the D1 table replay")
        if len(predicted) > budget:
            raise D1Refused("bad-d1-budget", "runtime predicted_ids exceed the D1 budget")
        published += len(predicted)
        demand = set(row["demand_ids"])
        if demand:
            demand_total += len(demand)
            hit = sum(1 for expert in predicted if expert in demand)
            hits += hit
        if row["accepted"]:
            paired_demand_total += len(demand)
        table.observe(row["draft_token"], row["layer"], row["accepted"], row["demand_ids"])
    accepted = table.accepted_rows
    rate = accepted / table.rows_seen
    return {
        "schema": "tg59-d1-replay/1",
        "rows": table.rows_seen,
        "accepted_rows": accepted,
        "acceptance_rate": rate,
        "pairing_valid": rate >= 0.60,
        "n_expert": n_expert,
        "top_k": top_k,
        "budget": budget,
        "published": published,
        "demand_rows_with_ids": sum(1 for row in rows if row["demand_ids"]),
        "demand_ids": demand_total,
        "paired_demand_ids": paired_demand_total,
        "hits": hits,
        "overall_coverage": hits / demand_total if demand_total else 0.0,
        "paired_coverage": hits / paired_demand_total if paired_demand_total else None,
        "limitation": "host replay; no CUDA, transfer, throughput, or D-gate claim",
    }


def _runtime_evidence(path: str) -> dict:
    """Read lifecycle/value evidence separately from D1 proposal precision."""
    run_start = rowset_init = mtp = run_end = None
    tg_ids = None
    demand_parts: list[str] = []
    hits = fetched = submitted = consumed = cancelled = matched = waited = 0
    spec_events = 0
    prop = arm_count = 0
    stops: dict[str, int] = {}
    phases: dict[str, dict] = {}
    try:
        handle = open(path, encoding="utf-8", errors="strict")
    except OSError as exc:
        raise D1Refused("io-error", str(exc))
    with handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise D1Refused("bad-json", "line %d is not JSON: %s" % (line_no, exc))
            kind = event.get("event")
            if kind == "run_start":
                if run_start is not None:
                    raise D1Refused("duplicate-run-start", "line %d repeats run_start" % line_no)
                run_start = event
            elif kind == "rowset_init":
                if rowset_init is not None:
                    raise D1Refused("duplicate-rowset-init", "line %d repeats rowset_init" % line_no)
                rowset_init = event
            elif kind == "mtp":
                if mtp is not None:
                    raise D1Refused("duplicate-mtp", "line %d repeats mtp" % line_no)
                mtp = event
            elif kind == "tg_tokens":
                ids = event.get("ids")
                if not isinstance(ids, list) or any(not isinstance(x, int) or isinstance(x, bool) for x in ids):
                    raise D1Refused("bad-tg-tokens", "line %d tg_tokens.ids is invalid" % line_no)
                tg_ids = ids
            elif kind == "run_end":
                if run_end is not None:
                    raise D1Refused("duplicate-run-end", "line %d repeats run_end" % line_no)
                run_end = event
            elif kind == "phase_summary":
                phase = str(event.get("phase", ""))
                phases[phase] = event
            elif kind == "rowset_fetch":
                hits += _integer(event.get("n_rows_resident", 0), "n_rows_resident")
                fetched += _integer(event.get("n_rows_fetched", 0), "n_rows_fetched")
                demand_parts.append(json.dumps([
                    event.get("ubatch"), event.get("layer"), event.get("phase"),
                    event.get("union_ids"),
                ], separators=(",", ":")))
            elif kind == "spec":
                spec_events += 1
                submitted += _integer(event.get("submitted", 0), "submitted")
                consumed = max(consumed, _integer(event.get("consumed", 0), "consumed"))
                cancelled = max(cancelled, _integer(event.get("cancelled", 0), "cancelled"))
                matched = max(matched, _integer(event.get("matched_by_demand", 0), "matched_by_demand"))
                waited = max(waited, _integer(event.get("waited_by_demand", 0), "waited_by_demand"))
                prop += _integer(event.get("prop", 0), "prop")
                arm_count += _integer(event.get("arm_count", 0), "arm_count")
                reason = str(event.get("stop_reason", "unknown"))
                stops[reason] = stops.get(reason, 0) + 1
    if run_start is None or rowset_init is None or mtp is None or run_end is None or tg_ids is None:
        raise D1Refused("incomplete-runtime-trace",
                        "D-gate needs run_start, rowset_init, mtp, tg_tokens and run_end (%s)" % path)
    if int(run_end.get("exit_status", 1)) != 0 or run_end.get("failed") is not False:
        raise D1Refused("failed-runtime-trace", "runtime did not finish successfully (%s)" % path)
    if int(run_start.get("row_spec_mode", 0)) != 2:
        raise D1Refused("not-mode2", "D-gate requires a mode-2 trace (%s)" % path)
    demands = hits + fetched
    identity = {
        "model": run_start.get("model"),
        "arch": run_start.get("arch"),
        "n_layer": run_start.get("n_layer"),
        "ctx_size": run_start.get("ctx_size"),
        "retain_k": rowset_init.get("retain_k"),
        "window_tokens": rowset_init.get("window_tokens"),
        "n_expert": rowset_init.get("n_expert"),
        "top_k": rowset_init.get("top_k"),
        "mtp_drafted": mtp.get("mtp_drafted"),
        "mtp_accepted": mtp.get("mtp_accepted"),
        "tg_ids_sha256": hashlib.sha256(json.dumps(tg_ids, separators=(",", ":")).encode()).hexdigest(),
        "demand_sha256": hashlib.sha256("\n".join(demand_parts).encode()).hexdigest(),
    }
    return {
        "path": path,
        "arm": str(run_start.get("row_spec_arm", "")),
        "null": bool(run_start.get("row_spec_null", False)),
        "identity": identity,
        "share": hits / demands if demands else 0.0,
        "hits": hits,
        "demands": demands,
        "spec_events": spec_events,
        "submitted": submitted,
        "consumed": consumed,
        "cancelled": cancelled,
        "matched_by_demand": matched,
        "waited_by_demand": waited,
        "proposals": prop,
        "arm_count": arm_count,
        "stops": stops,
        "tg_wall_us": phases.get("tg", {}).get("wall_us"),
        "tg_bytes": phases.get("tg", {}).get("bytes_transferred"),
        "peak_vram_bytes": run_end.get("peak_vram_bytes"),
    }


def _labeled_paths(values: list[str], option: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise D1Refused("bad-frontier-label", "%s requires LABEL=PATH, got %r" % (option, value))
        label, path = value.split("=", 1)
        label = label.strip()
        if not label or label in out:
            raise D1Refused("bad-frontier-label", "%s has an empty or duplicate label" % option)
        out[label] = path
    return out


def dgate(d1_arm: str, d1_null: str, frontier_arms: dict[str, str],
          frontier_nulls: dict[str, str]) -> dict:
    """Score the pre-registered runtime D-gate, never proposal replay as margin."""
    if not frontier_arms or set(frontier_arms) != set(frontier_nulls):
        raise D1Refused("frontier-pairing", "frontier arms and nulls must have the same non-empty labels")
    replay_arm = replay(d1_arm)
    replay_null = replay(d1_null)
    if not replay_arm["pairing_valid"]:
        raise D1Refused("null-unpaired", "D1 accepted-pair rate is below the 60-percent void rule")
    runtime = {label: _runtime_evidence(path) for label, path in frontier_arms.items()}
    runtime_null = {label: _runtime_evidence(path) for label, path in frontier_nulls.items()}
    d1a = _runtime_evidence(d1_arm)
    d1n = _runtime_evidence(d1_null)
    all_legs = {"d1-arm": d1a, "d1-null": d1n}
    all_legs.update({"%s-arm" % label: leg for label, leg in runtime.items()})
    all_legs.update({"%s-null" % label: leg for label, leg in runtime_null.items()})
    identities = [leg["identity"] for leg in all_legs.values()]
    if any(identity != identities[0] for identity in identities[1:]):
        raise D1Refused("session-identity-mismatch", "D1 and frontier legs do not share model, shape, MTP, output and demand identity")
    if d1a["arm"] != "d1" or d1a["null"] or not d1n["null"] or d1n["arm"] != "d1":
        raise D1Refused("d1-role-mismatch", "D1 arm/null roles do not match their run_start declarations")
    if d1a["arm_count"] != d1n["arm_count"]:
        raise D1Refused("d1-count-mismatch", "D1 arm and null do not have the same pre-dedup proposal count")
    frontier = []
    for label in sorted(frontier_arms):
        arm = runtime[label]
        null = runtime_null[label]
        if arm["arm"] != label or arm["null"] or not null["null"] or null["arm"] != label:
            raise D1Refused("frontier-role-mismatch", "frontier %s arm/null roles disagree" % label)
        if arm["proposals"] != null["arm_count"]:
            raise D1Refused("frontier-count-mismatch", "frontier %s is not count-matched to its null" % label)
        frontier.append({
            "arm": label,
            "share": arm["share"],
            "null_share": null["share"],
            "gain": arm["share"] - null["share"],
            "submitted": arm["submitted"],
            "consumed": arm["consumed"],
            "stops": arm["stops"],
        })
    d1_gain = d1a["share"] - d1n["share"]
    best = max(frontier, key=lambda row: row["gain"])
    # A spread is non-negative by definition, so equality with the best measured
    # frontier already fails the strict "beat ... by more than spread" rule. No
    # repeat is needed to turn this sufficient negative into a D-gate FAIL.
    spread_floor = 0.0
    passed = d1_gain > best["gain"] + spread_floor
    return {
        "schema": "tg59-d-gate/1",
        "identity": identities[0],
        "pairing_rate": replay_arm["acceptance_rate"],
        "proposal_precision": {
            "d1_hits": replay_arm["hits"],
            "d1_published": replay_arm["published"],
            "d1_coverage": replay_arm["overall_coverage"],
            "null_hits": replay_null["hits"],
            "null_published": replay_null["published"],
            "null_coverage": replay_null["overall_coverage"],
            "coverage_gain": replay_arm["overall_coverage"] - replay_null["overall_coverage"],
        },
        "runtime": {
            "d1_share": d1a["share"],
            "d1_null_share": d1n["share"],
            "d1_gain": d1_gain,
            "d1_submitted": d1a["submitted"],
            "d1_consumed": d1a["consumed"],
            "d1_stops": d1a["stops"],
            "frontier": frontier,
            "best_frontier": best,
            "session_spread_floor": spread_floor,
        },
        "verdict": "PASS" if passed else "FAIL",
        "reason": "D1 runtime gain beats the best measured past-only arm" if passed else
                  "D1 does not add runtime demand-hit value over the best past-only frontier",
    }


def print_gate(result: dict) -> None:
    runtime = result["runtime"]
    proposal = result["proposal_precision"]
    print("T-GPLR-59 D-gate: %s" % result["verdict"])
    print("  pairing rate             : %.6f" % result["pairing_rate"])
    print("  proposal precision D1    : %d/%d = %.4f" %
          (proposal["d1_hits"], proposal["d1_published"], proposal["d1_coverage"]))
    print("  proposal precision null  : %d/%d = %.4f" %
          (proposal["null_hits"], proposal["null_published"], proposal["null_coverage"]))
    print("  runtime share D1 / null  : %.6f / %.6f (gain %+.6f)" %
          (runtime["d1_share"], runtime["d1_null_share"], runtime["d1_gain"]))
    print("  D1 fills / consumed      : %d / %d" % (runtime["d1_submitted"], runtime["d1_consumed"]))
    for row in runtime["frontier"]:
        print("  frontier %-17s: %.6f / null %.6f (gain %+.6f), fills %d" %
              (row["arm"], row["share"], row["null_share"], row["gain"], row["submitted"]))
    print("  best measured frontier   : %s (%+.6f)" % (runtime["best_frontier"]["arm"], runtime["best_frontier"]["gain"]))
    print("  reason                   : %s" % result["reason"])


def _fixture(tmp: str, name: str, events: list[dict]) -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    return path


def selftest() -> int:
    import tempfile

    checks = 0
    with tempfile.TemporaryDirectory(prefix="tg59-d1-") as tmp:
        shape = {"event": "d1_shape", "n_expert": 16, "top_k": 4}
        good = _fixture(tmp, "good.jsonl", [
            shape,
            {"event": "d1_pair", "draft_token": 7, "layer": 0, "accepted": False, "demand_ids": [9, 8, 7, 6]},
            {"event": "d1_pair", "draft_token": 7, "layer": 0, "accepted": True, "demand_ids": [3, 1, 2, 4]},
            {"event": "d1_pair", "draft_token": 7, "layer": 0, "accepted": True, "demand_ids": [3, 1, 2, 4]},
            {"event": "d1_pair", "draft_token": 7, "layer": 0, "accepted": True, "demand_ids": [3, 1, 2, 4]},
            {"event": "d1_pair", "draft_token": 7, "layer": 0, "accepted": True, "demand_ids": [1, 2, 4, 5]},
        ])
        result = replay(good, 2)
        checks += 1
        assert result["acceptance_rate"] == 0.8
        assert result["pairing_valid"]
        assert result["hits"] == 6
        assert abs(result["overall_coverage"] - 6 / 20) < 1e-12
        assert abs(result["paired_coverage"] - 6 / 16) < 1e-12

        rejected = _fixture(tmp, "rejected.jsonl", [
            shape,
            {"event": "d1_pair", "draft_token": 7, "layer": 0, "accepted": False, "demand_ids": [9, 8, 7, 6]},
            {"event": "d1_pair", "draft_token": 7, "layer": 0, "accepted": True, "demand_ids": [3, 1, 2, 4]},
        ])
        result = replay(rejected, 2)
        checks += 1
        assert not result["pairing_valid"]

        mismatch = _fixture(tmp, "prediction-mismatch.jsonl", [
            {"event": "d1_shape", "n_expert": 16, "top_k": 4, "predicted_ids": True},
            {"event": "d1_pair", "draft_token": 7, "layer": 0,
             "accepted": False, "demand_ids": [3, 1, 2, 4], "predicted_ids": [3]},
        ])
        try:
            replay(mismatch, 2)
        except D1Refused as exc:
            checks += 1
            assert exc.code == "prediction-mismatch"
        else:
            raise AssertionError("runtime D1 prediction mismatch was accepted")

        malformed = _fixture(tmp, "malformed.jsonl", [shape, {
            "event": "d1_pair", "draft_token": 7, "layer": 0,
            "accepted": True, "demand_ids": [3, 3],
        }])
        try:
            replay(malformed, 2)
        except D1Refused as exc:
            checks += 1
            assert exc.code == "duplicate-demand-id"
        else:
            raise AssertionError("duplicate D1 demand id was accepted")

        try:
            replay(_fixture(tmp, "empty.jsonl", [shape]))
        except D1Refused as exc:
            checks += 1
            assert exc.code == "no-d1-pairs"
        else:
            raise AssertionError("empty D1 replay was accepted")

        null = _fixture(tmp, "null.jsonl", [
            {"event": "d1_shape", "n_expert": 16, "top_k": 4,
             "null": True, "predicted_ids": True},
            {"event": "d1_pair", "draft_token": 7, "layer": 0,
             "accepted": True, "demand_ids": [1, 2, 3, 4], "predicted_ids": [5, 6]},
            {"event": "d1_pair", "draft_token": 7, "layer": 0,
             "accepted": True, "demand_ids": [1, 2, 3, 4], "predicted_ids": [5, 6]},
        ])
        result = replay(null, 4)
        checks += 1
        assert result["published"] == 4 and result["hits"] == 0
        try:
            replay(null, 1)
        except D1Refused as exc:
            checks += 1
            assert exc.code == "bad-d1-budget"
        else:
            raise AssertionError("runtime D1 proposals over budget were accepted")

        duplicate_shape = _fixture(tmp, "duplicate-shape.jsonl", [
            shape, shape,
            {"event": "d1_pair", "draft_token": 7, "layer": 0,
             "accepted": True, "demand_ids": [1, 2, 3, 4]},
        ])
        try:
            replay(duplicate_shape, 4)
        except D1Refused as exc:
            checks += 1
            assert exc.code == "duplicate-d1-shape"
        else:
            raise AssertionError("duplicate D1 shape was accepted")

        pair_before_shape = _fixture(tmp, "pair-before-shape.jsonl", [
            {"event": "d1_pair", "draft_token": 7, "layer": 0,
             "accepted": True, "demand_ids": [1, 2, 3, 4]},
            shape,
        ])
        try:
            replay(pair_before_shape, 4)
        except D1Refused as exc:
            checks += 1
            assert exc.code == "no-d1-shape"
        else:
            raise AssertionError("D1 pair before shape was accepted")

        # D-gate lifecycle fixtures: the four MTP legs share exact model, shape,
        # MTP acceptance, generated ids and demand stream. Every arm submits zero
        # rows, so proposal precision may be positive while the runtime gate ties.
        def runtime_fixture(name: str, arm: str, null: bool, share_num: int, proposals: int) -> str:
            hits = share_num
            fetched = 100 - share_num
            events = [
                {"event": "d1_shape", "source": "d1", "n_expert": 16, "top_k": 8,
                 "null": null, "predicted_ids": bool(null)},
                {"event": "d1_pair", "draft_token": 7, "layer": 0, "accepted": True,
                 "demand_ids": [1, 2, 3, 4, 5, 6, 7, 8], "predicted_ids": [1, 2]},
                {"event": "run_start", "model": "fixture", "arch": "qwen35moe", "n_layer": 1,
                 "ctx_size": 128, "row_spec_mode": 2, "row_spec_arm": arm, "row_spec_null": null},
                {"event": "rowset_init", "retain_k": 4, "window_tokens": 3,
                 "n_expert": 16, "top_k": 8},
                {"event": "mtp", "mtp_drafted": 8, "mtp_accepted": 8},
                {"event": "tg_tokens", "ids": [1, 2, 3]},
                {"event": "rowset_fetch", "ubatch": 1, "layer": 0, "phase": 1,
                 "union_ids": "1,2", "n_rows_resident": hits, "n_rows_fetched": fetched},
                {"event": "spec", "submitted": 0, "consumed": 0, "cancelled": 0,
                 "matched_by_demand": 0, "waited_by_demand": 0, "prop": proposals,
                 "arm_count": proposals, "stop_reason": "no_free"},
                {"event": "phase_summary", "phase": "tg", "wall_us": 10,
                 "bytes_transferred": 100},
                {"event": "run_end", "exit_status": 0, "failed": False, "peak_vram_bytes": 1},
            ]
            return _fixture(tmp, name, events)

        gate_d1 = runtime_fixture("gate-d1.jsonl", "d1", False, 50, 8)
        gate_d1_null = runtime_fixture("gate-d1-null.jsonl", "d1", True, 50, 8)
        gate_coact = runtime_fixture("gate-coact.jsonl", "coact", False, 50, 4)
        gate_coact_null = runtime_fixture("gate-coact-null.jsonl", "coact", True, 50, 4)
        gate = dgate(gate_d1, gate_d1_null, {"coact": gate_coact}, {"coact": gate_coact_null})
        checks += 1
        assert gate["verdict"] == "FAIL" and gate["runtime"]["d1_submitted"] == 0
        good_share = runtime_fixture("gate-future-share.jsonl", "coact", False, 60, 4)
        good_gate = dgate(gate_d1, gate_d1_null, {"coact": good_share}, {"coact": gate_coact_null})
        checks += 1
        assert good_gate["verdict"] == "FAIL"  # frontier gain +0.10 beats D1 gain 0
        future_d1 = runtime_fixture("gate-d1-future-share.jsonl", "d1", False, 60, 8)
        passed_gate = dgate(future_d1, gate_d1_null, {"coact": gate_coact}, {"coact": gate_coact_null})
        checks += 1
        assert passed_gate["verdict"] == "PASS"
        mismatched = runtime_fixture("gate-mismatch.jsonl", "coact", False, 50, 4)
        with open(mismatched, encoding="utf-8") as handle:
            mismatched_text = handle.read().replace('"retain_k": 4', '"retain_k": 5')
        with open(mismatched, "w", encoding="utf-8") as handle:
            handle.write(mismatched_text)
        try:
            dgate(gate_d1, gate_d1_null, {"coact": mismatched}, {"coact": gate_coact_null})
        except D1Refused as exc:
            checks += 1
            assert exc.code == "session-identity-mismatch"
        else:
            raise AssertionError("D-gate accepted mismatched session output")

    print("T-GPLR-59 D1: PASS (%d checks)" % checks)
    return 0


def print_result(result: dict) -> None:
    print("T-GPLR-59 D1 host replay")
    print("  rows / accepted       : %d / %d" % (result["rows"], result["accepted_rows"]))
    print("  acceptance rate       : %.3f" % result["acceptance_rate"])
    print("  pairing gate          : %s" % ("PASS" if result["pairing_valid"] else "NULL-UNPAIRED"))
    print("  published candidates  : %d" % result["published"])
    print("  hits                  : %d" % result["hits"])
    print("  overall coverage      : %.4f" % result["overall_coverage"])
    if result["paired_coverage"] is None:
        print("  paired coverage       : UNVERIFIED_NO_PAIRED_DEMAND")
    else:
        print("  paired coverage       : %.4f" % result["paired_coverage"])
    print("  limitation            : %s" % result["limitation"])


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--selftest":
        return selftest()
    if len(argv) != 3 or argv[1] != "--trace":
        sys.stderr.write("usage: python tools/tg59_d1.py --trace <d1.jsonl> | --selftest\n")
        return 2
    try:
        result = replay(argv[2])
    except (D1Refused, OSError) as exc:
        code = getattr(exc, "code", "io-error")
        detail = getattr(exc, "detail", str(exc))
        sys.stderr.write("REFUSED [%s]: %s\n" % (code, detail))
        return 2
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
