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
import math
import os
import re
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


def read_pairs(path: str) -> tuple[list[dict], int, int, int | None, bool, bool]:
    if not os.path.isfile(path):
        raise D1Refused("file-missing", "no such D1 trace: %s" % path)
    rows: list[dict] = []
    n_expert = None
    top_k = None
    n_layer = None
    shape_run_id = None
    predicted_required = False
    null_source = False
    shape_seen = False
    previous_order = None
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
                shape_run_id = _integer(event.get("run_id"), "run_id")
                if shape_run_id < 0:
                    raise D1Refused("bad-d1-shape", "d1_shape run_id must be non-negative")
                n_expert = _integer(event.get("n_expert"), "n_expert")
                top_k = _integer(event.get("top_k"), "top_k")
                if n_expert <= 0 or top_k <= 0 or top_k > n_expert:
                    raise D1Refused("bad-d1-shape", "d1_shape has an invalid expert shape")
                if "n_layer" in event:
                    n_layer = _integer(event.get("n_layer"), "n_layer")
                    if n_layer <= 0:
                        raise D1Refused("bad-d1-shape", "d1_shape n_layer must be positive")
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
            if _integer(event.get("run_id"), "run_id") != shape_run_id:
                raise D1Refused("run-id-mismatch", "line %d has a different run_id" % line_no)
            round_id = _integer(event.get("round"), "round")
            draft_index = _integer(event.get("draft_index"), "draft_index")
            layer = _integer(event.get("layer"), "layer")
            if round_id <= 0 or draft_index < 0 or layer < 0:
                raise D1Refused("bad-d1-field", "line %d has an invalid round, draft index, or layer" % line_no)
            if n_layer is not None and layer >= n_layer:
                raise D1Refused("d1-shape-mismatch", "line %d layer is outside n_layer" % line_no)
            order = (round_id, draft_index, layer)
            if previous_order is not None and order < previous_order:
                raise D1Refused("d1-order", "line %d is out of order" % line_no)
            previous_order = order
            draft_token = _integer(event.get("draft_token"), "draft_token")
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
            if draft_token < 0 or any(x < 0 or x >= (n_expert or 1 << 30) for x in demand_ids):
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
    return rows, n_expert, top_k, n_layer, predicted_required, null_source


def replay(path: str, budget: int = 8) -> dict:
    rows, n_expert, top_k, _n_layer, predicted_required, null_source = read_pairs(path)
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
    run_id = None
    tg_ids = None
    demand_parts: list[str] = []
    hits = fetched = submitted = consumed = cancelled = matched = waited = 0
    spec_events = 0
    prop = arm_count = 0
    stops: dict[str, int] = {}
    phases: dict[str, dict] = {}

    def check_run_id(event: dict, line_no: int) -> int:
        value = _integer(event.get("run_id"), "run_id")
        if value < 0:
            raise D1Refused("bad-run-id", "line %d run_id must be non-negative" % line_no)
        nonlocal run_id
        if run_id is None:
            run_id = value
        elif value != run_id:
            raise D1Refused("run-id-mismatch", "line %d changes run_id" % line_no)
        return value

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
            if not isinstance(event, dict):
                raise D1Refused("bad-event", "line %d is not an object" % line_no)
            kind = event.get("event")
            if run_end is not None and kind != "run_end":
                raise D1Refused("event-order", "line %d follows run_end" % line_no)
            if kind == "run_start":
                if run_start is not None:
                    raise D1Refused("duplicate-run-start", "line %d repeats run_start" % line_no)
                check_run_id(event, line_no)
                if not isinstance(event.get("model"), str) or not event["model"]:
                    raise D1Refused("bad-run-start", "line %d model is missing" % line_no)
                if not isinstance(event.get("arch"), str) or not event["arch"]:
                    raise D1Refused("bad-run-start", "line %d arch is missing" % line_no)
                for name in ("n_layer", "ctx_size"):
                    if _integer(event.get(name), name) <= 0:
                        raise D1Refused("bad-run-start", "line %d %s must be positive" % (line_no, name))
                if not isinstance(event.get("row_spec_arm"), str) or not event["row_spec_arm"]:
                    raise D1Refused("bad-run-start", "line %d row_spec_arm is missing" % line_no)
                if not isinstance(event.get("row_spec_null"), bool):
                    raise D1Refused("bad-run-start", "line %d row_spec_null must be boolean" % line_no)
                run_start = event
            elif kind == "rowset_init":
                if run_start is None:
                    raise D1Refused("event-order", "line %d rowset_init precedes run_start" % line_no)
                if rowset_init is not None:
                    raise D1Refused("duplicate-rowset-init", "line %d repeats rowset_init" % line_no)
                check_run_id(event, line_no)
                for name in ("retain_k", "window_tokens", "n_expert", "top_k"):
                    if _integer(event.get(name), name) <= 0:
                        raise D1Refused("bad-rowset-init", "line %d %s must be positive" % (line_no, name))
                rowset_init = event
            elif kind == "phase_summary":
                check_run_id(event, line_no)
                phase = str(event.get("phase", ""))
                if not phase or phase in phases:
                    raise D1Refused("bad-phase-summary", "line %d has an empty or duplicate phase" % line_no)
                phases[phase] = event
            elif kind == "rowset_fetch":
                if rowset_init is None:
                    raise D1Refused("event-order", "line %d rowset_fetch precedes rowset_init" % line_no)
                check_run_id(event, line_no)
                for name in ("ubatch", "layer"):
                    if _integer(event.get(name), name) < 0:
                        raise D1Refused("bad-rowset-fetch", "line %d %s must be non-negative" % (line_no, name))
                resident = _integer(event.get("n_rows_resident", 0), "n_rows_resident")
                fetched_rows = _integer(event.get("n_rows_fetched", 0), "n_rows_fetched")
                if resident < 0 or fetched_rows < 0:
                    raise D1Refused("bad-rowset-fetch", "line %d row counts must be non-negative" % line_no)
                hits += resident
                fetched += fetched_rows
                demand_parts.append(json.dumps([
                    event.get("ubatch"), event.get("layer"), event.get("phase"),
                    event.get("union_ids"),
                ], separators=(",", ":")))
            elif kind == "spec":
                if rowset_init is None:
                    raise D1Refused("event-order", "line %d spec precedes rowset_init" % line_no)
                check_run_id(event, line_no)
                spec_events += 1
                values = {name: _integer(event.get(name, 0), name) for name in (
                    "submitted", "consumed", "cancelled", "matched_by_demand",
                    "waited_by_demand", "prop", "arm_count")}
                if any(value < 0 for value in values.values()):
                    raise D1Refused("bad-spec", "line %d has a negative counter" % line_no)
                submitted += values["submitted"]
                consumed = max(consumed, values["consumed"])
                cancelled = max(cancelled, values["cancelled"])
                matched = max(matched, values["matched_by_demand"])
                waited = max(waited, values["waited_by_demand"])
                prop += values["prop"]
                arm_count += values["arm_count"]
                reason = str(event.get("stop_reason", "unknown"))
                stops[reason] = stops.get(reason, 0) + 1
            elif kind == "mtp":
                if "tg" not in phases:
                    raise D1Refused("event-order", "line %d mtp precedes tg phase_summary" % line_no)
                if mtp is not None:
                    raise D1Refused("duplicate-mtp", "line %d repeats mtp" % line_no)
                check_run_id(event, line_no)
                for name in ("mtp_drafted", "mtp_accepted"):
                    if _integer(event.get(name), name) < 0:
                        raise D1Refused("bad-mtp", "line %d %s must be non-negative" % (line_no, name))
                mtp = event
            elif kind == "tg_tokens":
                if mtp is None:
                    raise D1Refused("event-order", "line %d tg_tokens precedes mtp" % line_no)
                if tg_ids is not None:
                    raise D1Refused("duplicate-tg-tokens", "line %d repeats tg_tokens" % line_no)
                check_run_id(event, line_no)
                ids = event.get("ids")
                if not isinstance(ids, list) or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in ids):
                    raise D1Refused("bad-tg-tokens", "line %d tg_tokens.ids is invalid" % line_no)
                tg_ids = ids
            elif kind == "run_end":
                if run_end is not None:
                    raise D1Refused("duplicate-run-end", "line %d repeats run_end" % line_no)
                if tg_ids is None:
                    raise D1Refused("event-order", "line %d run_end precedes tg_tokens" % line_no)
                check_run_id(event, line_no)
                if not isinstance(event.get("failed"), bool):
                    raise D1Refused("bad-run-end", "line %d failed must be boolean" % line_no)
                if _integer(event.get("exit_status"), "exit_status") != 0:
                    raise D1Refused("failed-runtime-trace", "runtime did not finish successfully (%s)" % path)
                if _integer(event.get("peak_vram_bytes"), "peak_vram_bytes") < 0:
                    raise D1Refused("bad-run-end", "line %d peak_vram_bytes must be non-negative" % line_no)
                run_end = event
    if run_start is None or rowset_init is None or mtp is None or run_end is None or tg_ids is None:
        raise D1Refused("incomplete-runtime-trace",
                        "D-gate needs run_start, rowset_init, mtp, tg_tokens and run_end (%s)" % path)
    if "tg" not in phases or not demand_parts or spec_events == 0:
        raise D1Refused("incomplete-runtime-trace",
                        "D-gate needs a tg phase, rowset_fetch and spec events (%s)" % path)
    if run_end.get("failed") is not False:
        raise D1Refused("failed-runtime-trace", "runtime did not finish successfully (%s)" % path)
    if _integer(run_start.get("row_spec_mode"), "row_spec_mode") != 2:
        raise D1Refused("not-mode2", "D-gate requires a mode-2 trace (%s)" % path)
    demands = hits + fetched
    identity = {
        "model": run_start.get("model"),
        "arch": run_start.get("arch"),
        "n_layer": _integer(run_start.get("n_layer"), "n_layer"),
        "ctx_size": _integer(run_start.get("ctx_size"), "ctx_size"),
        "retain_k": _integer(rowset_init.get("retain_k"), "retain_k"),
        "window_tokens": _integer(rowset_init.get("window_tokens"), "window_tokens"),
        "n_expert": _integer(rowset_init.get("n_expert"), "n_expert"),
        "top_k": _integer(rowset_init.get("top_k"), "top_k"),
        "mtp_drafted": _integer(mtp.get("mtp_drafted"), "mtp_drafted"),
        "mtp_accepted": _integer(mtp.get("mtp_accepted"), "mtp_accepted"),
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


def _session_evidence(path: str | None, expected: dict[str, str]) -> dict | None:
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8", errors="strict") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise D1Refused("session-manifest", str(exc))
    if not isinstance(manifest, dict) or manifest.get("schema") != "tg59-d-gate-session/1":
        raise D1Refused("session-manifest", "unsupported D-gate session manifest")
    legs = manifest.get("legs")
    if not isinstance(legs, dict) or set(legs) != set(expected):
        raise D1Refused("session-manifest", "manifest must describe exactly the D1 and frontier legs")
    spread = manifest.get("spread")
    if spread is not None and (isinstance(spread, bool) or not isinstance(spread, (int, float)) or
                               not math.isfinite(spread) or spread < 0):
        raise D1Refused("session-spread", "manifest spread must be a finite non-negative number or null")
    identities = []
    command_identity = None
    for label, trace_path in expected.items():
        entry = legs[label]
        if not isinstance(entry, dict):
            raise D1Refused("session-manifest", "%s manifest entry is not an object" % label)
        listed_trace = entry.get("trace")
        verdict_path = entry.get("verdict")
        if not isinstance(listed_trace, str) or not isinstance(verdict_path, str):
            raise D1Refused("session-manifest", "%s needs trace and verdict paths" % label)
        if os.path.normcase(os.path.realpath(listed_trace)) != os.path.normcase(os.path.realpath(trace_path)):
            raise D1Refused("session-trace-mismatch", "%s manifest trace does not match the CLI trace" % label)
        try:
            with open(verdict_path, encoding="utf-8", errors="strict") as handle:
                verdict = json.load(handle)
        except (OSError, ValueError) as exc:
            raise D1Refused("session-verdict", "%s: %s" % (label, exc))
        fields = verdict.get("fields") if isinstance(verdict, dict) else None
        if (not isinstance(verdict, dict) or verdict.get("status") != "PASS" or verdict.get("exit_code") != 0 or
                not isinstance(fields, dict) or fields.get("mutex_acquired") is not True or
                fields.get("gpu_pre_run") != "clean" or fields.get("gpu_post_run") != "clean"):
            raise D1Refused("session-verdict", "%s does not record a clean PASS under the GPU mutex" % label)
        launch = fields.get("provenance")
        post = fields.get("provenance_post")
        delta = fields.get("provenance_delta")
        if (not isinstance(launch, dict) or not isinstance(post, dict) or not isinstance(delta, dict) or
                delta.get("binary_changed_during_run") is not False or delta.get("source_changed_during_run") is not False):
            raise D1Refused("session-verdict", "%s has incomplete or changed provenance" % label)
        for provenance in (launch, post):
            if (not isinstance(provenance.get("source_digest"), str) or not provenance["source_digest"] or
                    not isinstance(provenance.get("binary_exe_sha"), str) or not provenance["binary_exe_sha"] or
                    not isinstance(provenance.get("streaming_lib_sha"), str) or not provenance["streaming_lib_sha"]):
                raise D1Refused("session-verdict", "%s has incomplete binary provenance" % label)
        identity = (launch["source_digest"], launch["binary_exe_sha"], launch["streaming_lib_sha"])
        post_identity = (post["source_digest"], post["binary_exe_sha"], post["streaming_lib_sha"])
        if identity != post_identity:
            raise D1Refused("session-verdict", "%s provenance changed during the run" % label)
        command = verdict.get("command", fields.get("command"))
        if not isinstance(command, str) or not command:
            raise D1Refused("session-verdict", "%s has no launch command" % label)
        normalized = re.sub(r"--telemetry\s+\S+", "--telemetry <path>", command)
        normalized = re.sub(r"--residency-tg-spec-arm\s+\S+", "--residency-tg-spec-arm <arm>", normalized)
        normalized = re.sub(r"--residency-tg-spec-null\s+\S+", "--residency-tg-spec-null <null>", normalized)
        normalized = re.sub(r"\s>\s+\S+\s+2>&1", " > <log> 2>&1", normalized)
        if command_identity is None:
            command_identity = normalized
        elif normalized != command_identity:
            raise D1Refused("session-command-mismatch", "%s used a different model, prompt, or resolved command" % label)
        identities.append(identity)
    if len(set(identities)) != 1:
        raise D1Refused("session-binary-mismatch", "session legs did not use one source and binary stack")
    assert identities and command_identity is not None
    return {
        "verified": True,
        "source_digest": identities[0][0],
        "binary_exe_sha": identities[0][1],
        "streaming_lib_sha": identities[0][2],
        "command_sha256": hashlib.sha256(command_identity.encode()).hexdigest(),
        "spread": float(spread) if spread is not None else None,
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
          frontier_nulls: dict[str, str], session_manifest: str | None = None) -> dict:
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
    session = _session_evidence(session_manifest, {
        "d1-arm": d1_arm,
        "d1-null": d1_null,
        **{"%s-arm" % label: path for label, path in frontier_arms.items()},
        **{"%s-null" % label: path for label, path in frontier_nulls.items()},
    })
    spread_floor = session["spread"] if session is not None else None
    passed = (session is not None and spread_floor is not None and
              d1_gain > best["gain"] + spread_floor)
    if session is None:
        reason = "D1 gate is FAIL: session provenance and measured spread are unverified"
    elif spread_floor is None:
        reason = "D1 gate is FAIL: the session spread is unmeasured"
    elif passed:
        reason = "D1 runtime gain beats the best measured past-only arm by more than the session spread"
    else:
        reason = "D1 does not add runtime demand-hit value over the best past-only frontier"
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
        "session_provenance": session,
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
        "reason": reason,
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
        shape = {"event": "d1_shape", "run_id": 7, "n_expert": 16, "top_k": 4, "n_layer": 2}
        good = _fixture(tmp, "good.jsonl", [
            shape,
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0, "accepted": False, "demand_ids": [9, 8, 7, 6]},
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0, "accepted": True, "demand_ids": [3, 1, 2, 4]},
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0, "accepted": True, "demand_ids": [3, 1, 2, 4]},
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0, "accepted": True, "demand_ids": [3, 1, 2, 4]},
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0, "accepted": True, "demand_ids": [1, 2, 4, 5]},
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
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0, "accepted": False, "demand_ids": [9, 8, 7, 6]},
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0, "accepted": True, "demand_ids": [3, 1, 2, 4]},
        ])
        result = replay(rejected, 2)
        checks += 1
        assert not result["pairing_valid"]

        mismatch = _fixture(tmp, "prediction-mismatch.jsonl", [
            {"event": "d1_shape", "run_id": 7, "n_expert": 16, "top_k": 4, "n_layer": 2, "predicted_ids": True},
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0,
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
            "event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0,
            "draft_token": 7, "layer": 0, "accepted": True, "demand_ids": [3, 3],
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

        null = _fixture(tmp, "null.jsonl", [{"event": "d1_shape", "run_id": 7, "n_expert": 16, "top_k": 4, "n_layer": 2,
              "null": True, "predicted_ids": True},
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0,
             "accepted": True, "demand_ids": [1, 2, 3, 4], "predicted_ids": [5, 6]},
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0,
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
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0,
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
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0,
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

        out_of_order = _fixture(tmp, "out-of-order.jsonl", [shape,
            {"event": "d1_pair", "run_id": 7, "round": 2, "draft_index": 0,
             "draft_token": 7, "layer": 0, "accepted": False, "demand_ids": [1, 2, 3, 4]},
            {"event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0,
             "draft_token": 7, "layer": 0, "accepted": True, "demand_ids": [1, 2, 3, 4]},
        ])
        try:
            replay(out_of_order, 4)
        except D1Refused as exc:
            checks += 1
            assert exc.code == "d1-order"
        else:
            raise AssertionError("out-of-order D1 pairs were accepted")

        layer_out_of_range = _fixture(tmp, "layer-out-of-range.jsonl", [shape, {
            "event": "d1_pair", "run_id": 7, "round": 1, "draft_index": 0,
            "draft_token": 7, "layer": 2, "accepted": True, "demand_ids": [1, 2, 3, 4],
        }])
        try:
            replay(layer_out_of_range, 4)
        except D1Refused as exc:
            checks += 1
            assert exc.code == "d1-shape-mismatch"
        else:
            raise AssertionError("out-of-range D1 layer was accepted")

        # D-gate lifecycle fixtures: the four MTP legs share exact model, shape,
        # MTP acceptance, generated ids and demand stream. Every arm submits zero
        # rows, so proposal precision may be positive while the runtime gate ties.
        def runtime_fixture(name: str, arm: str, null: bool, share_num: int, proposals: int) -> str:
            hits = share_num
            fetched = 100 - share_num
            events = [
                {"event": "run_start", "model": "fixture", "arch": "qwen35moe", "n_layer": 1,
                 "ctx_size": 128, "row_spec_mode": 2, "row_spec_arm": arm, "row_spec_null": null},
                {"event": "rowset_init", "retain_k": 4, "window_tokens": 3,
                 "n_expert": 16, "top_k": 8},
                {"event": "rowset_fetch", "ubatch": 1, "layer": 0, "phase": 1,
                 "union_ids": "1,2", "n_rows_resident": hits, "n_rows_fetched": fetched},
                {"event": "spec", "submitted": 0, "consumed": 0, "cancelled": 0,
                 "matched_by_demand": 0, "waited_by_demand": 0, "prop": proposals,
                 "arm_count": proposals, "stop_reason": "no_free"},
                {"event": "phase_summary", "phase": "tg", "wall_us": 10,
                 "bytes_transferred": 100},
                {"event": "mtp", "mtp_drafted": 8, "mtp_accepted": 8},
                {"event": "tg_tokens", "ids": [1, 2, 3]},
                {"event": "d1_shape", "source": "d1", "n_expert": 16, "top_k": 8, "n_layer": 1,
                 "null": null, "predicted_ids": bool(null)},
                {"event": "d1_pair", "round": 1, "draft_index": 0, "draft_token": 7, "layer": 0,
                 "accepted": True, "demand_ids": [1, 2, 3, 4, 5, 6, 7, 8], "predicted_ids": [1, 2]},
                {"event": "run_end", "exit_status": 0, "failed": False, "peak_vram_bytes": 1},
            ]
            events = [{"run_id": 7, **event} for event in events]
            return _fixture(tmp, name, events)

        gate_d1 = runtime_fixture("gate-d1.jsonl", "d1", False, 50, 8)
        gate_d1_null = runtime_fixture("gate-d1-null.jsonl", "d1", True, 50, 8)
        gate_coact = runtime_fixture("gate-coact.jsonl", "coact", False, 50, 4)
        gate_coact_null = runtime_fixture("gate-coact-null.jsonl", "coact", True, 50, 4)
        gate = dgate(gate_d1, gate_d1_null, {"coact": gate_coact}, {"coact": gate_coact_null})
        checks += 1
        assert gate["verdict"] == "FAIL" and gate["runtime"]["d1_submitted"] == 0
        duplicate_tg = runtime_fixture("duplicate-tg.jsonl", "d1", False, 50, 8)
        with open(duplicate_tg, encoding="utf-8") as handle:
            duplicate_events = [json.loads(line) for line in handle if line.strip()]
        duplicate_events.insert(-1, {"event": "tg_tokens", "run_id": 7, "ids": [4]})
        with open(duplicate_tg, "w", encoding="utf-8") as handle:
            for event in duplicate_events:
                handle.write(json.dumps(event) + "\n")
        try:
            _runtime_evidence(duplicate_tg)
        except D1Refused as exc:
            checks += 1
            assert exc.code == "duplicate-tg-tokens"
        else:
            raise AssertionError("duplicate tg_tokens was accepted")
        no_spec = runtime_fixture("no-spec.jsonl", "d1", False, 50, 8)
        no_spec_events = []
        with open(no_spec, encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("event") != "spec":
                    no_spec_events.append(event)
        with open(no_spec, "w", encoding="utf-8") as handle:
            for event in no_spec_events:
                handle.write(json.dumps(event) + "\n")
        try:
            _runtime_evidence(no_spec)
        except D1Refused as exc:
            checks += 1
            assert exc.code == "incomplete-runtime-trace"
        else:
            raise AssertionError("runtime trace without spec events was accepted")
        good_share = runtime_fixture("gate-future-share.jsonl", "coact", False, 60, 4)
        good_gate = dgate(gate_d1, gate_d1_null, {"coact": good_share}, {"coact": gate_coact_null})
        checks += 1
        assert good_gate["verdict"] == "FAIL"  # frontier gain +0.10 beats D1 gain 0
        future_d1 = runtime_fixture("gate-d1-future-share.jsonl", "d1", False, 60, 8)
        passed_gate = dgate(future_d1, gate_d1_null, {"coact": gate_coact}, {"coact": gate_coact_null})
        checks += 1
        assert passed_gate["verdict"] == "FAIL"
        assert "unverified" in passed_gate["reason"]
        verdict_entries = {}
        trace_entries = {
            "d1-arm": future_d1,
            "d1-null": gate_d1_null,
            "coact-arm": gate_coact,
            "coact-null": gate_coact_null,
        }
        identity = {
            "source_digest": "fixture-source",
            "binary_exe_sha": "fixture-exe",
            "streaming_lib_sha": "fixture-lib",
        }
        for label, trace_path in trace_entries.items():
            verdict_path = os.path.join(tmp, label + ".verdict.json")
            verdict = {
                "status": "PASS",
                "exit_code": 0,
                "fields": {
                    "mutex_acquired": True,
                    "gpu_pre_run": "clean",
                    "gpu_post_run": "clean",
                    "command": "--model model.gguf --prompt-file prompt.txt --telemetry %s" % trace_path,
                    "provenance": identity,
                    "provenance_post": identity,
                    "provenance_delta": {"binary_changed_during_run": False, "source_changed_during_run": False},
                },
            }
            with open(verdict_path, "w", encoding="utf-8") as handle:
                json.dump(verdict, handle)
            verdict_entries[label] = {"trace": trace_path, "verdict": verdict_path}
        manifest_path = os.path.join(tmp, "session.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({"schema": "tg59-d-gate-session/1", "spread": 0.0, "legs": verdict_entries}, handle)
        verified_gate = dgate(future_d1, gate_d1_null, {"coact": gate_coact}, {"coact": gate_coact_null}, manifest_path)
        checks += 1
        assert verified_gate["verdict"] == "PASS"
        assert verified_gate["session_provenance"]["verified"]
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
