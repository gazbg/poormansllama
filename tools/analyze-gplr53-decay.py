#!/usr/bin/env python3
"""T-GPLR-53 A3 pre-registered decay discriminator (host-only, no GPU).

Reads the recon-gated `embed_chunk` events that A2 landed and applies the
decision rule the T-GPLR-53 block PRE-REGISTERED before A3 ran:

  per chunk c:  excess(c) = span_us(pass 1, c) - median{ span_us(pass p, c) : p >= 2 }
  statistic (i): first-decile share = sum(excess of the first ceil(10%) chunks)
                                    / sum(all positive excess)
  statistic (ii): rho = Spearman(chunk index, excess)

  FRONT_LOADED  rho <= -0.7 AND first-decile share >= 0.50 -> H1 or H3
                (H1-vs-H3 is A4's modality question, never this tool's)
  FLAT          |rho| < 0.3 (or a uniform ~5-6.4 ms/chunk profile) -> the
                page/driver stories are refuted: ESCALATE, do not tune
  CHUNK0_SPIKE  chunk 0 carries >= 50% of the excess and every later chunk sits
                at the warm band (<= 2x the pass-2 median) -> H3
  MIXED         anything else: report the numbers, claim nothing

Exit contract: 0 = a verdict was produced (any of the four), 1 = the trace is
malformed, 2 = there is nothing to evaluate (no embed_chunk event, or fewer
than two eval[LP]uable passes). Exit 2 is NEVER a pass.

Usage:
  python tools/analyze-gplr53-decay.py T1 [T2 ...]   # arms are evaluated independently
  python tools/analyze-gplr53-decay.py --selftest
"""

import json
import statistics
import sys
from pathlib import Path

DECILE = 0.10
RHO_FRONT = -0.7
SHARE_FRONT = 0.50
RHO_FLAT = 0.30
CHUNK0_SHARE = 0.50


def rank(values):
    """Average ranks (ties share the mean rank), 1-based."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rho with average-rank tie handling. None when undefined."""
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def load_chunks(path, diag=None):
    """{pass_index: {chunk: span_us}} from the embed_chunk stream.

    T-GPLR-53 (2026-09-22): the pass is derived from the CHUNK-SEQUENCE RESET, not
    from an event field. `embed_chunk` carries no pass index at all in the recorded
    traces, and `phase` -- which this loader used to read as the pass -- means
    PP-vs-TG elsewhere in the format (values 0/1 on the per-op events, 'pp'/'tg' on
    `phase_summary`). Reading it as a pass put every chunk in pass 0, so the rule
    refused its own input on both A3 arms. The emitter writes each pass's chunks in
    order 0..n-1, so a pass boundary is exactly where the chunk index stops
    increasing; verified on the A3 traces (294 = 98x3 at 50k, 1173 = 391x3 at 200k,
    monotone t_us, three equal blocks at each shape).

    Ambiguity is REFUSED, never guessed: every block must be exactly 0..n-1, all
    blocks must be the same length, and when the trace also carries per-pass
    boundary markers (`host_ledger`, one per pass) their count must agree with the
    block count. `pp_pass` is advisory only -- it fires once per pass TRANSITION and
    is measured at n-1 (2 markers for 3 passes), so it is recorded, never gated on.
    """
    seq = []
    ledger = 0
    ppmark = 0
    run_saw = False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"event":"host_ledger"' in line or '"event": "host_ledger"' in line:
                ledger += 1
            if '"event":"pp_pass"' in line or '"event": "pp_pass"' in line:
                ppmark += 1
            if '"event":"embed_chunk"' not in line and '"event": "embed_chunk"' not in line:
                if '"event":"run_start"' in line or '"event": "run_start"' in line:
                    run_saw = True
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError("malformed JSON on an embed_chunk line")
            if ev.get("event") != "embed_chunk":
                continue
            c = ev.get("chunk")
            s = ev.get("span_us")
            if c is None or s is None:
                raise ValueError("embed_chunk without chunk/span_us")
            seq.append((int(c), float(s)))
    if not run_saw and not seq:
        raise ValueError("no run_start and no embed_chunk: not a streaming trace")

    blocks = []
    cur = []
    for c, s in seq:
        if cur and c <= cur[-1][0]:
            blocks.append(cur)
            cur = []
        cur.append((c, s))
    if cur:
        blocks.append(cur)

    if diag is not None:
        diag["bucketing"] = "chunk-sequence reset"
        diag["pass_boundary_events"] = {"host_ledger": ledger, "pp_pass": ppmark}
        diag["block_lengths"] = [len(b) for b in blocks]

    for i, b in enumerate(blocks):
        ids = [c for c, _ in b]
        if ids != list(range(len(ids))):
            raise ValueError("pass %d chunk ids are not exactly 0..%d (gapped, duplicated "
                             "or out of order): the pass boundary cannot be trusted"
                             % (i + 1, len(ids) - 1))
    if len({len(b) for b in blocks}) > 1:
        raise ValueError("uneven pass blocks %s: a truncated or interleaved trace cannot "
                         "be bucketed by chunk reset" % [len(b) for b in blocks])
    if ledger >= 2 and ledger != len(blocks):
        raise ValueError("chunk-reset bucketing found %d pass(es) but the trace carries %d "
                         "host_ledger boundaries: refusing to guess which is the pass count"
                         % (len(blocks), ledger))

    return {i: {c: s for c, s in b} for i, b in enumerate(blocks)}


def evaluate(path):
    """Returns (code, verdict, facts). code 0 = verdict, 2 = nothing to evaluate."""
    diag = {}
    by_pass = load_chunks(path, diag)
    pp = [p for p in sorted(by_pass) if len(by_pass[p]) >= 2]
    if len(pp) < 2:
        out = {"reason": "fewer than two passes with per-chunk timing"}
        out.update(diag)
        return 2, "NOT_EVALUABLE", out
    first = pp[0]
    later = pp[1:]
    chunks = sorted(by_pass[first])
    if len(chunks) < 3:
        return 2, "NOT_EVALUABLE", {"reason": "fewer than three chunks in pass 1"}

    warm = {}
    for c in chunks:
        vals = [by_pass[p][c] for p in later if c in by_pass[p]]
        if vals:
            warm[c] = statistics.median(vals)
    if len(warm) < len(chunks):
        return 2, "NOT_EVALUABLE", {"reason": "later passes do not cover every pass-1 chunk"}

    excess = {c: by_pass[first][c] - warm[c] for c in chunks}
    total_pos = sum(max(0.0, e) for e in excess.values())
    # The per-pass numbers the rule decides on, kept in the verdict so the decision
    # can be re-derived without re-reading the trace.
    facts = {
        "n_chunks": len(chunks),
        "bucketing": diag.get("bucketing", "unknown"),
        "pass_count": len(pp),
        "chunks_per_pass": len(chunks),
        "pass_totals_ms": [round(sum(by_pass[p].values()) / 1000.0, 2) for p in sorted(by_pass)],
        "pass_boundary_events": diag.get("pass_boundary_events", {}),
        "pass1_total_ms": round(sum(by_pass[first].values()) / 1000.0, 2),
        "warm_median_ms": round(statistics.median(warm.values()) / 1000.0, 3),
        "excess_total_ms": round(sum(excess.values()) / 1000.0, 2),
        "excess_positive_ms": round(total_pos / 1000.0, 2),
        "pass1_chunk0_ms": round(by_pass[first][chunks[0]] / 1000.0, 3),
        "warm_chunk0_ms": round(warm[chunks[0]] / 1000.0, 3),
        "excess_chunk0_ms": round(excess[chunks[0]] / 1000.0, 2),
        "max_excess_excl_chunk0_ms": round(
            max([max(0.0, excess[c]) for c in chunks[1:]] or [0.0]) / 1000.0, 3),
    }
    if total_pos <= 0.0:
        facts["reason"] = "no positive pass-1 excess: nothing to discriminate"
        return 0, "NO_EXCESS", facts

    k = max(1, int(len(chunks) * DECILE + 0.999))
    head = sorted(chunks, key=lambda c: excess[c], reverse=True)[:k]
    first_decile = [c for c in chunks[:k]]
    facts["first_decile_share"] = round(sum(max(0.0, excess[c]) for c in first_decile) / total_pos, 4)
    facts["top_excess_chunks"] = head
    facts["chunk0_share"] = round(max(0.0, excess[chunks[0]]) / total_pos, 4)

    rho = spearman(list(chunks), [excess[c] for c in chunks])
    facts["rho_spearman"] = None if rho is None else round(rho, 4)

    # "the rest sits at the warm band" is a statement about the PASS-1 spans of the
    # later chunks, not about the warm medians themselves (a mistake this selftest
    # caught: with the warm values compared against themselves the guard was always
    # true, so any chunk-0-heavy profile read as CHUNK0_SPIKE and FRONT_LOADED was
    # unreachable). A later chunk is warm when its own pass-1 span is within 2x the
    # later-pass median.
    warm_med = statistics.median(warm.values())
    later_warm = all(by_pass[first][c] <= warm_med * 2.0 for c in chunks[1:])
    if facts["chunk0_share"] >= CHUNK0_SHARE and later_warm and len(chunks) > 1:
        return 0, "CHUNK0_SPIKE", facts
    if rho is not None and rho <= RHO_FRONT and facts["first_decile_share"] >= SHARE_FRONT:
        return 0, "FRONT_LOADED", facts
    if rho is None or abs(rho) < RHO_FLAT:
        return 0, "FLAT", facts
    return 0, "MIXED", facts


def synth(chunks, passes=2, ledger=None, truncate_last=0, gap_at=None):
    """Synthetic trace in the RECORDED shape: `chunks` is the pass-1 per-chunk span
    list (us), later passes are warm (640 us). No pass field is emitted -- that is
    the whole point of these fixtures: bucketing must work off the chunk reset.
    `ledger` overrides the number of host_ledger boundary markers (default: one per
    pass); `truncate_last` drops that many chunks from the final pass; `gap_at`
    renumbers one pass's chunks so they are not 0..n-1.
    """
    out = []
    for p in range(passes):
        spans = chunks if p == 0 else [640.0] * len(chunks)
        body = list(enumerate(spans))
        if p == passes - 1 and truncate_last:
            body = body[:-truncate_last]
        if gap_at is not None and p == gap_at:
            body = [(c + 1 if c >= 1 else c, s) for c, s in body]   # 0,2,3,.. -> not 0..n-1
        for c, s in body:
            out.append({"event": "embed_chunk", "chunk": c, "span_us": s})
        out.append({"event": "host_ledger", "embed_us": 1000.0, "t_us": p})
    if ledger is not None:
        extra = ledger - passes
        while extra > 0:
            out.append({"event": "host_ledger", "embed_us": 1000.0, "t_us": 99})
            extra -= 1
        while extra < 0:
            for i in range(len(out)):
                if out[i].get("event") == "host_ledger":
                    del out[i]
                    extra += 1
                    break
    return "\n".join(json.dumps(e) for e in out)


def selftest():
    import tempfile

    cases = []
    # front-loaded: big excess on the first chunk, warm afterwards
    cases.append(("front-loaded", synth([6000, 1500, 800, 700, 690, 660, 650, 645, 640, 640]),
                  "FRONT_LOADED", 0))
    # uniform: every chunk pays the same excess -> the page/driver stories are refuted
    cases.append(("uniform", synth([1200] * 10), "FLAT", 0))
    # chunk-0 spike with the rest at the warm band
    cases.append(("chunk0-spike", synth([5000] + [660] * 9), "CHUNK0_SPIKE", 0))
    # no excess at all
    cases.append(("no-excess", synth([640] * 10), "NO_EXCESS", 0))
    # single pass -> nothing to evaluate (exit 2, NOT a pass)
    cases.append(("one-pass", synth([6000, 700], passes=1), "NOT_EVALUABLE", 2))

    ok = 0
    with tempfile.TemporaryDirectory() as td:
        for name, body, want, want_code in cases:
            p = Path(td) / (name + ".jsonl")
            p.write_text(body + "\n", encoding="utf-8")
            try:
                code, verdict, _ = evaluate(str(p))
            except ValueError as exc:
                print(f"[selftest] {name:14s} raised {exc}")
                continue
            good = (code == want_code and verdict == want)
            ok += good
            print(f"[selftest] {name:14s} -> {verdict} (exit {code}) "
                  f"{'ok' if good else 'MISMATCH, want ' + want}")
        # negative control: a trace with no embed_chunk at all must REFUSE
        p = Path(td) / "no-event.jsonl"
        p.write_text(json.dumps({"event": "run_start"}) + "\n", encoding="utf-8")
        code, verdict, _ = evaluate(str(p))
        good = (code == 2 and verdict == "NOT_EVALUABLE")
        ok += good
        print(f"[selftest] {'no-event':14s} -> {verdict} (exit {code}) "
              f"{'ok' if good else 'MISMATCH, want NOT_EVALUABLE exit 2'}")

        # T-GPLR-53 (2026-09-22): the bucketing itself must be covered, because the
        # defect that blocked A3 was in the loader, not the rule. Each negative
        # control must REFUSE (exit 2) rather than bucket a trace it cannot trust.
        refuse = []
        refuse.append(("truncated-pass", synth([6000, 700, 700, 700], truncate_last=2),
                       "uneven pass blocks"))
        refuse.append(("gapped-chunks", synth([6000, 700, 700, 700], gap_at=1),
                       "not exactly 0.."))
        refuse.append(("ledger-mismatch", synth([6000, 700, 700, 700], ledger=3),
                       "refusing to guess"))
        for name, body, want_sub in refuse:
            p = Path(td) / (name + ".jsonl")
            p.write_text(body + "\n", encoding="utf-8")
            try:
                code, verdict, facts = evaluate(str(p))
                exc = ""
            except ValueError as err:
                code, verdict, exc = 2, "REFUSED", str(err)
            good = (code == 2 and (want_sub in exc or want_sub in json.dumps(facts)))
            ok += good
            print(f"[selftest] {name:14s} -> {verdict} (exit {code}) "
                  f"{'ok' if good else 'MISMATCH, want a refusal naming: ' + want_sub}")
        # positive control for the same shape: 3 passes, boundaries agreeing
        p = Path(td) / "three-pass.jsonl"
        p.write_text(synth([5000] + [660] * 9, passes=3) + "\n", encoding="utf-8")
        code, verdict, facts = evaluate(str(p))
        good = (code == 0 and verdict == "CHUNK0_SPIKE" and facts.get("pass_count") == 3)
        ok += good
        print(f"[selftest] {'three-pass':14s} -> {verdict} (exit {code}, passes "
              f"{facts.get('pass_count')}) {'ok' if good else 'MISMATCH, want CHUNK0_SPIKE over 3 passes'}")
    total = len(cases) + 1 + len(refuse) + 1
    print(f"SELFTEST: {'PASS' if ok == total else 'FAIL'} {ok}/{total}")
    return 0 if ok == total else 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not paths:
        print(__doc__.strip())
        return 2
    worst = 0
    for p in paths:
        try:
            code, verdict, facts = evaluate(p)
        except (OSError, ValueError) as exc:
            print(f"{p}: ERROR {exc}")
            worst = max(worst, 1)
            continue
        print(f"{p}: {verdict} (exit {code}) {json.dumps(facts, sort_keys=True)}")
        worst = max(worst, code if code == 2 else 0)
    return worst


if __name__ == "__main__":
    sys.exit(main())
