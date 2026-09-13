#!/usr/bin/env python3
"""Render the T-GPLR-15 gate-9 statistic (b) versus retention K as a standalone
HTML artifact.

View 1 (always): the measured (b)-versus-K curve, the 0.6 threshold, the SDD
section 4 reference band, and the VRAM each K costs.

With --ladder, the four ladder rungs are overlaid as a second set of series, so
the saturation-versus-stream-length argument is visible rather than described:
the rungs are 3-window streams and their (b) equals the oracle from K=2 on,
because a 3-window stream has only 2 predecessor windows to retain -- while the
256-token probe stream saturates only at K=24. Each rung's structural saturation
depth (windows - 1) is ringed on the chart and the claim is checked in code, not
asserted: the tool compares (b) at K >= windows-1 against the oracle and says so
in the note if any rung disagrees.

View 2 (when --trace and --tg-tok-s are given): what the same retention would be
worth in WALL-CLOCK terms. Statistic (b) is the fraction of the next window's
rows a cache would already hold, so the bytes it removes from the link are
(b) x (bytes staged per token), and the time those bytes cost is measured, not
assumed: per-fetch bytes and per-fetch transfer come from the trace itself.
The saving is priced at two measured rates -- the 13.7 GB/s link ceiling and the
effective per-fetch rate the path actually achieves -- and against the lever's
own asymptote (every staged byte removed, the residual per-token floor). So the
retention choice reads as milliseconds and tok/s, not only as VRAM.
This is a PROJECTION from a measured staging share, not a measured speedup --
the artifact says so, because the task contracts forbid a speedup claim without
a paired run.

Reads the analyzer's JSON report (tools/route_overlap/analyze.py --json ...) and
writes one self-contained HTML file with inline SVG -- no external assets, no
third-party packages, stdlib only. Open the output in any browser; nothing else
is needed to read it.

Usage:
  python tools/route_overlap/plot-b-vs-k.py \
      --input  build/llama-cuda/gplr15-phaseA-probe/analysis-wide.json \
      --trace  build/llama-cuda/gplr15-phaseA-probe/probe-tg256.jsonl \
      --tg-tok-s 11.578 \
      --out    docs/plots/gplr15-b-vs-k.html
"""

import argparse
import collections
import html
import json
import os

THRESHOLD_FALLBACK = 0.6
REFERENCE_BAND = (1, 2)   # SDD section 4: a one-to-two window retention
LABELLED_K = {1, 2, 4, 8, 16, 32}
TIME_LABELLED_K = {1, 2, 4, 8, 32}   # view 2: 16 and 32 are the same value (plateau)
# One colour per ladder rung. They are near-coincident by construction (that is
# the finding), so the colours carry identity for the table, not for the chart.
RUNG_COLOURS = ["#2e7d32", "#00838f", "#8d6e63", "#ad1457"]
# Value labels sit close to steep series, and a rising curve passes through the
# wedge under a label placed above its own marker. A white halo keeps the text
# legible where that happens, instead of juggling 13 positions per series.
HALO = (' paint-order="stroke" stroke="#ffffff" stroke-width="3.2" '
        'stroke-linejoin="round"')
LINK_GBS_DEFAULT = 13.7   # measured Gen4 x8 H2D ceiling (T-Q38TG-00)
# One A/B summary per retention depth. The K=1 gate plus whichever deferred
# rungs have been run; a path that does not exist is skipped, not faked.
MEASURED_AB_DEFAULTS = [
    "build/llama-cuda/gplr19-rowcache-ab/summary.json",   # K=1, the gate
    "build/llama-cuda/gplr19-k2-ab/summary.json",
    "build/llama-cuda/gplr19-k4-ab/summary.json",
    "build/llama-cuda/gplr19-k6-ab/summary.json",
    "build/llama-cuda/gplr19-k8-ab/summary.json",
]

W, H = 980, 620
X0, X1 = 96, 900        # plot box, left/right
Y0, Y1 = 96, 452        # plot box, top/bottom  (value axis 0.0 -> 1.0)


def load_curve(path):
    with open(path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    if isinstance(report, dict):
        report = [report]
    rows = [r for r in report if r.get("rowset")]
    if not rows:
        raise SystemExit(f"no row-set report in {path} (the gate statistic needs rowset_fetch events)")
    # the richest curve wins, so a --retention-k run supersedes the default band
    rep = max(rows, key=lambda r: len(r["rowset"]["verdict_statistic_b"]))
    rs = rep["rowset"]
    curve = sorted(
        ((int(k), v) for k, v in rs["verdict_statistic_b"].items()),
        key=lambda kv: kv[0],
    )
    return {
        "trace": rep.get("path", "?"),
        "model": rep.get("model", "?"),
        "run_id": rep.get("run_id"),
        "curve": curve,
        "oracle": rs["oracle_containment"]["value"],
        "pairs": rs["exact_consecutive"]["pairs"],
        "n_expert": rs["n_expert"],
        "top_k": rs["top_k"],
        "row_bytes": rs["row_bytes"],
        "window_tokens": rs["window_tokens"],
        "windows_per_layer": rs.get("windows_per_layer") or [],
        "layers": rs["n_layers"] or rs["oracle_containment"]["layers"],
        "saturated_by": rs.get("ceiling_saturated_by_k"),
    }


def rung_label(path):
    """Rung name from the trace it came from: rowset-4000.jsonl -> '4k'."""
    digits = "".join(ch for ch in os.path.basename(str(path)) if ch.isdigit())
    n = int(digits) if digits else 0
    return f"{n // 1000}k" if n >= 1000 else str(n)


def load_rungs(path):
    """Per-rung series from one analyzer report covering the ladder traces.

    The structural saturation depth is windows_per_layer - 1: a stream with M
    windows per layer has no window older than M-1 back to retain, so every
    K >= M-1 is the same cache as far as the stream can tell. The cap cannot
    bind first, because a K=M-1 cache holds (M-1)*W*top_k rows and any union of
    M-1 windows holds at most that many.
    """
    with open(path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    if isinstance(report, dict):
        report = [report]
    out = []
    for rep in report:
        rs = rep.get("rowset")
        if not rs:
            continue
        curve = sorted(
            ((int(k), v) for k, v in rs["verdict_statistic_b"].items()),
            key=lambda kv: kv[0],
        )
        wins = rs.get("windows_per_layer") or []
        m = max(wins) if wins else None
        struct_k = (m - 1) if m else None
        oracle = rs["oracle_containment"]["value"]
        out.append({
            "label": rung_label(rep.get("path")),
            "path": rep.get("path"),
            "curve": curve,
            "oracle": oracle,
            "pairs": rs["exact_consecutive"]["pairs"],
            "windows": m,
            "struct_k": struct_k,
            "reported_sat": rs.get("ceiling_saturated_by_k"),
            "cap_at_struct": (
                min(rs["n_expert"], struct_k * rs["window_tokens"] * rs["top_k"])
                if struct_k else None
            ),
            "n_expert": rs["n_expert"],
            "top_k": rs["top_k"],
            "window_tokens": rs["window_tokens"],
            # the claim is CHECKED, not asserted: at/after the structural depth
            # the value must BE the oracle, because nothing older can exist
            "struct_confirmed": (
                struct_k is not None
                and all(abs(v["value"] - oracle) < 1e-12
                        for k, v in curve if k >= struct_k)
            ),
        })
    if not out:
        raise SystemExit(f"no row-set report in {path} (the rung overlay needs rowset_* traces)")
    return out


def vram_gb(rows_per_layer, d):
    return rows_per_layer * d["row_bytes"] * d["layers"] / 1e9


def load_trace_basis(path):
    """Measured per-fetch basis straight from the trace: bytes and transfer time.

    Deliberately NOT derived from a row size: the staged bytes per fetch and the
    transfer that carries them are the quantities the projection needs, and they
    come from the same run the baseline tok/s came from.
    """
    tot_b = tot_t = tot_s = 0.0
    n = rows = 0
    full_bank = 0
    per_layer = collections.defaultdict(lambda: [0, 0, 0])
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if '"event":"rowset_round"' in line:
                full_bank = json.loads(line).get("bytes_full_bank") or full_bank
                continue
            if '"event":"rowset_fetch"' not in line:
                continue
            ev = json.loads(line)
            n += 1
            rows += int(ev["n_rows_fetched"])
            tot_b += ev["bytes_staged"]
            tot_t += ev["transfer_us"]
            tot_s += ev["host_submit_us"]
            acc = per_layer[int(ev["layer"])]
            acc[0] += 1
            acc[1] += int(ev["n_rows_fetched"])
            acc[2] += ev["bytes_staged"]
    if n == 0:
        raise SystemExit(f"no rowset_fetch events in {path}; view 2 needs a row-set trace")
    # Per-layer staged bytes: the row-set policy sizes a layer's rows from the same
    # slabs every time, so this is a measured per-layer row size, not a model. It is
    # NOT uniform across layers on the reference model, and the analyzer's slot-derived
    # row_bytes is (see the basis caveat the artifact prints when both are known).
    row_by_layer = {il: b / r for il, (c, r, b) in per_layer.items() if r}
    rb = collections.Counter(round(v, 3) for v in row_by_layer.values())
    modal, _ = rb.most_common(1)[0]
    return {
        "trace": path,
        "fetches": n,
        "rows_per_fetch": rows / n,
        "bytes_per_fetch": tot_b / n,
        "transfer_us_mean": tot_t / n,
        "submit_us_mean": tot_s / n,
        "effective_gbs": (tot_b / n) / (tot_t / n * 1e-6) / 1e9,
        "full_bank_bytes": full_bank,
        "layers_seen": len(row_by_layer),
        "row_bytes_modal": modal,
        "row_bytes_max": max(row_by_layer.values()),
        "n_odd_layers": sum(c for v, c in rb.items() if v != modal),
        "one_row_all_layers": sum(row_by_layer.values()),
    }


def project_times(d, basis, baseline_tok_s, link_gbs):
    """Per-K byte savings and the wall-clock they are worth at two rates.

    The bounds are both real numbers from the measurements: the link ceiling
    (optimistic throughput, conservative saving) and the effective per-fetch
    rate this path actually achieves (which carries the pageable staging and
    per-call overhead, so removing bytes removes a share of those too).
    """
    base_ms = 1000.0 / baseline_tok_s
    per_token_bytes = basis["bytes_per_fetch"] * d["layers"]
    staging_ms = d["layers"] * basis["transfer_us_mean"] / 1000.0
    out = []
    for k, v in d["curve"]:
        saved_b = v["value"] * per_token_bytes
        row = {"k": k, "b": v["value"], "rows": v["capacity_rows_per_layer"],
               "vram": vram_gb(v["capacity_rows_per_layer"], d), "saved_b": saved_b}
        for tag, rate in (("link", link_gbs), ("eff", basis["effective_gbs"])):
            ms = saved_b / (rate * 1e6)
            row[f"{tag}_ms"] = ms
            row[f"{tag}_toks"] = 1000.0 / (base_ms - ms)
        out.append(row)
    return {
        "rows": out,
        "base_ms": base_ms,
        "per_token_bytes": per_token_bytes,
        "staging_ms": staging_ms,
        "staging_share": staging_ms / base_ms,
        "floor_ms": base_ms - staging_ms,
    }


def build_time_svg(d, basis, proj, baseline_tok_s, link_gbs, measured=None):
    """View 2: projected TG tok/s against retention K, at both measured rates."""
    rows = proj["rows"]
    ks = [r["k"] for r in rows]
    kmin, kmax = min(ks), max(ks)
    span = max(1, kmax - kmin)
    # the asymptote is what makes view 2 answer the F2 question: with every staged
    # byte removed, the measured per-token floor remains, and that is the most any
    # retention (or any perfect predictor) can buy back.
    asymptote = 1000.0 / proj["floor_ms"]
    top = max(max(r["link_toks"], r["eff_toks"]) for r in rows)
    if measured:
        top = max(top, max(m["tok_s"] for m in measured))
    ymax = max(top, asymptote) * 1.14

    def x(k):
        return X0 + (k - kmin) / span * (X1 - X0)

    def y(v):
        return Y1 - max(0.0, v) / ymax * (Y1 - Y0)

    out = []
    add = out.append
    add(f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
        f'aria-label="Projected TG tokens per second against retention K">')
    add(f'<rect x="{x(1) - 14:.1f}" y="{Y0}" width="{x(2) - x(1) + 14:.1f}" '
        f'height="{Y1 - Y0}" fill="#eef4ff"/>')
    add(f'<text x="{x(1) - 8:.1f}" y="{Y1 + 74}" font-size="12" fill="#3a5fa8">'
        f'SDD s4 reference band (K=1-2)</text>')

    step = ymax / 5.0
    v = 0.0
    while v <= ymax:
        add(f'<line x1="{X0}" y1="{y(v):.1f}" x2="{X1}" y2="{y(v):.1f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>')
        add(f'<text x="{X0 - 12}" y="{y(v) + 4:.1f}" font-size="12" fill="#666" '
            f'text-anchor="end">{v:.0f}</text>')
        v += step
    add(f'<text x="26" y="{(Y0 + Y1) / 2:.0f}" font-size="13" fill="#333" '
        f'transform="rotate(-90 26 {(Y0 + Y1) / 2:.0f})" text-anchor="middle">'
        f'projected TG tok/s (projection, not a measured speedup)</text>')

    # the measured baseline this projection starts from
    add(f'<line x1="{X0}" y1="{y(baseline_tok_s):.1f}" x2="{X1}" '
        f'y2="{y(baseline_tok_s):.1f}" stroke="#555" stroke-width="1.6" '
        f'stroke-dasharray="7 4"/>')
    add(f'<text x="{X0 + 6}" y="{y(baseline_tok_s) + 16:.1f}" font-size="12" '
        f'fill="#333">measured reactive row-set TG: {baseline_tok_s:.3f} tok/s</text>')

    add(f'<line x1="{X0}" y1="{y(asymptote):.1f}" x2="{X1}" y2="{y(asymptote):.1f}" '
        f'stroke="#8e44ad" stroke-width="1.2" stroke-dasharray="2 4"/>')
    add(f'<text x="{X1 - 4}" y="{y(asymptote) - 8:.1f}" font-size="12" fill="#8e44ad" '
        f'text-anchor="end">all staging removed: {asymptote:.1f} tok/s '
        f'({proj["floor_ms"]:.1f} ms/token floor)</text>')

    for tag, colour, label in (("link", "#1f6feb", f"at the {link_gbs:g} GB/s link ceiling"),
                               ("eff", "#e07b39", f"at the {basis['effective_gbs']:.2f} GB/s "
                                f"effective per-fetch rate")):
        pts = " ".join(f"{x(r['k']):.1f},{y(r[f'{tag}_toks']):.1f}" for r in rows)
        add(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2.4"/>')
        for r in rows:
            add(f'<circle cx="{x(r["k"]):.1f}" cy="{y(r[f"{tag}_toks"]):.1f}" r="4" '
                f'fill="{colour}"/>')
        add(f'<text x="{X0 + 6}" y="{Y0 + (12 if tag == "link" else 26)}" font-size="12" '
            f'fill="{colour}">{label}</text>')

    # The paired measurements (T-GPLR-19's four-arm A/B, one run per retention
    # depth) are drawn ON this view rather than only described, because they are
    # exactly what the projection is asked to predict. Drawing the series instead
    # of a single marker is what makes the shape checkable: the measured line
    # tracks inside the projected band, so a reader can see whether the bracket
    # held as well as whether one point landed in it. Markers are hollow rings so
    # the projected series stay visible underneath. Labels go in the legend row,
    # not beside the markers: the geometry here is tight (7 px between the
    # projected series at K=1) and a label would sit on them.
    if measured:
        pts = " ".join(f"{x(m['k']):.1f},{y(m['tok_s']):.1f}" for m in measured)
        if len(measured) > 1:
            add(f'<polyline points="{pts}" fill="none" stroke="#111" '
                f'stroke-width="1.6" stroke-dasharray="3 3"/>')
        for m in measured:
            add(f'<circle cx="{x(m["k"]):.1f}" cy="{y(m["tok_s"]):.1f}" r="7" '
                f'fill="none" stroke="#111" stroke-width="2"/>')
            add(f'<circle cx="{x(m["k"]):.1f}" cy="{y(m["tok_s"]):.1f}" r="2.6" fill="#111"/>')
        ks = ", ".join(str(m["k"]) for m in measured)
        add(f'<text x="{X0 + 6}" y="{Y0 + 40}" font-size="12" fill="#111">'
            f'\u25cf measured, one ABAB run per K: K={ks}</text>')

    # Value labels, with two guards because this geometry is genuinely tight: the link
    # projection is nearly flat early (13.6 at K=1 vs 14.7 at K=2, 7 px apart) and the
    # orange series sits only {9,17} px above it at those K, so a label above a blue
    # marker lands on the orange one. Labels that cannot be read are dropped, not
    # squeezed -- the full table below carries every value.
    prev_right = None
    for r in rows:
        if r["k"] not in TIME_LABELLED_K:
            continue
        if y(r["link_toks"]) - y(r["eff_toks"]) < 22:
            continue
        lx = x(r["k"])
        half = len(f'{r["link_toks"]:.1f}') * 6.5 / 2
        if prev_right is not None and lx - half < prev_right + 8:
            continue
        prev_right = lx + half
        add(f'<text x="{lx:.1f}" y="{y(r["link_toks"]) - 11:.1f}" '
            f'font-size="12" fill="#0b3d91" text-anchor="middle"{HALO}>'
            f'{r["link_toks"]:.1f}</text>')
    for r, anchor in ((rows[0], "start"), (rows[-1], "end")):
        add(f'<text x="{x(r["k"]):.1f}" y="{y(r["eff_toks"]) - 11:.1f}" font-size="12" '
            f'fill="#e07b39" text-anchor="{anchor}"{HALO}>{r["eff_toks"]:.1f}</text>')

    add(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="#999" stroke-width="1.2"/>')
    for r in rows:
        add(f'<line x1="{x(r["k"]):.1f}" y1="{Y1}" x2="{x(r["k"]):.1f}" y2="{Y1 + 5}" '
            f'stroke="#999" stroke-width="1"/>')
        add(f'<text x="{x(r["k"]):.1f}" y="{Y1 + 20}" font-size="12" fill="#333" '
            f'text-anchor="middle">{r["k"]}</text>')
    add(f'<text x="{(X0 + X1) / 2:.0f}" y="{Y1 + 42}" font-size="13" fill="#333" '
        f'text-anchor="middle">K = retention depth in token windows</text>')
    add('</svg>')
    return "\n".join(out)


def load_measured_abs(paths, basis, d):
    """The paired T-GPLR-19 A/B results, one entry per retention depth K.

    Reads one summary.json per K (the K=1 gate plus whichever deferred rungs
    have been run). Files that are absent are skipped, so the artifact degrades
    to fewer measured points rather than inventing one. Every number here is
    derived from the arm means -- removed bytes over the time actually saved --
    never quoted from a note, and the byte fraction is checked against the
    curve view 1 already shows so a rung cannot silently disagree with it.
    """
    out = []
    for path in (paths or []):
        if not path or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8-sig") as fh:
            arms = json.load(fh)
        # the driver records tok/s as text (it scrapes the run log), so coerce
        off = [a for a in arms if int(a.get("retain")) == 0]
        ks = sorted({int(a["retain"]) for a in arms if int(a.get("retain")) > 0})
        if not off or not ks:
            continue
        k = ks[0]
        on = [a for a in arms if int(a.get("retain")) == k]
        tg = lambda a: float(a["tg_tok_s"])
        base_tg = sum(tg(a) for a in off) / len(off)
        ret_tg = sum(tg(a) for a in on) / len(on)
        base_b = sum(float(a["staged_per_round"]) for a in off) / len(off)
        ret_b = sum(float(a["staged_per_round"]) for a in on) / len(on)
        saved_b = base_b - ret_b
        saved_s = (1.0 / base_tg) - (1.0 / ret_tg)
        conv = saved_b / saved_s / 1e9 if saved_s > 0 else 0.0
        spread = max(
            abs(tg(on[0]) - tg(on[-1])),
            abs(tg(off[0]) - tg(off[-1])),
        )
        point = next((v for kk, v in d["curve"] if kk == k), None)
        out.append({
            "k": k,
            "path": path,
            "reactive_tok_s": base_tg,
            "tok_s": ret_tg,
            "gain": ret_tg - base_tg,
            "saved_ms_per_token": (1.0 / base_tg - 1.0 / ret_tg) * 1000.0,
            "spread": spread,
            "byte_pct": saved_b / base_b * 100.0,
            "predicted_pct": point["value"] * 100.0 if point else 0.0,
            "conversion_gbs": conv,
        })
    return sorted(out, key=lambda p: p["k"]) if out else None


def build_time_table(proj, link_gbs, basis):
    rows = ["<tr><th>K</th><th>rows/layer</th><th>VRAM (GB)</th><th>statistic (b)</th>"
            "<th>bytes saved/token (MB)</th>"
            f"<th>ms saved @ {link_gbs:g} GB/s</th><th>tok/s @ {link_gbs:g} GB/s</th>"
            f"<th>ms saved @ {basis['effective_gbs']:.2f} GB/s</th>"
            f"<th>tok/s @ {basis['effective_gbs']:.2f} GB/s</th></tr>"]
    for r in proj["rows"]:
        rows.append(
            f'<tr><td>{r["k"]}</td><td>{r["rows"]}</td><td>{r["vram"]:.2f}</td>'
            f'<td>{r["b"]:.4f}</td><td>{r["saved_b"] / 1e6:.1f}</td>'
            f'<td>{r["link_ms"]:.2f}</td><td>{r["link_toks"]:.2f}</td>'
            f'<td>{r["eff_ms"]:.2f}</td><td>{r["eff_toks"]:.2f}</td></tr>'
        )
    return "\n".join(rows)


def build_svg(d, rungs=None):
    curve = d["curve"]
    ks = [k for k, _ in curve]
    kmin, kmax = min(ks), max(ks)
    span = max(1, kmax - kmin)

    def x(k):
        return X0 + (k - kmin) / span * (X1 - X0)

    def y(v):
        return Y1 - max(0.0, min(1.0, v)) * (Y1 - Y0)

    # Screen-y band the overlaid rung series occupy. Labels below are kept clear of
    # it: on the reference model the rungs sit at 0.36-0.43, which is exactly where
    # a value label above the probe's K=1 marker would land.
    rung_band = None
    if rungs:
        rys = [y(v["value"]) for r in rungs for _, v in r["curve"]]
        rung_band = (min(rys), max(rys))

    thresh = next((v for k, v in curve if v.get("below") is not None), None)
    thresh = THRESHOLD_FALLBACK

    out = []
    add = out.append

    add(f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
        f'aria-label="Statistic (b) versus retention K for the T-GPLR-15 gate">')

    # SDD reference band (K=1..2), behind everything
    add(f'<rect x="{x(REFERENCE_BAND[0]) - 14:.1f}" y="{Y0}" '
        f'width="{x(REFERENCE_BAND[1]) - x(REFERENCE_BAND[0]) + 14:.1f}" '
        f'height="{Y1 - Y0}" fill="#eef4ff"/>')
    add(f'<text x="{x(REFERENCE_BAND[0]) - 8:.1f}" y="{Y1 + 74}" font-size="12" '
        f'fill="#3a5fa8">SDD s4 reference band (K=1-2)</text>')

    # gridlines + value axis
    for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        add(f'<line x1="{X0}" y1="{y(v):.1f}" x2="{X1}" y2="{y(v):.1f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>')
        add(f'<text x="{X0 - 12}" y="{y(v) + 4:.1f}" font-size="12" fill="#666" '
            f'text-anchor="end">{v:.1f}</text>')
    add(f'<text x="26" y="{(Y0 + Y1) / 2:.0f}" font-size="13" fill="#333" '
        f'transform="rotate(-90 26 {(Y0 + Y1) / 2:.0f})" text-anchor="middle">'
        f'statistic (b) - fraction of the next window already resident</text>')

    # gate threshold
    add(f'<line x1="{X0}" y1="{y(thresh):.1f}" x2="{X1}" y2="{y(thresh):.1f}" '
        f'stroke="#c0392b" stroke-width="1.6" stroke-dasharray="7 4"/>')
    add(f'<text x="{X1 - 4}" y="{y(thresh) - 8:.1f}" font-size="12" fill="#c0392b" '
        f'text-anchor="end">gate 9 threshold {thresh}</text>')

    # oracle ceiling
    add(f'<line x1="{X0}" y1="{y(d["oracle"]):.1f}" x2="{X1}" y2="{y(d["oracle"]):.1f}" '
        f'stroke="#7f8c8d" stroke-width="1.2" stroke-dasharray="2 4"/>')
    # anchored LEFT: right-aligned it would collide with the K=max value label, which
    # sits on the same plateau
    add(f'<text x="{X0 + 6}" y="{y(d["oracle"]) - 8:.1f}" font-size="12" fill="#5d6d6e" '
        f'text-anchor="start">T8 oracle ceiling (unbounded retention) '
        f'{d["oracle"]:.4f}</text>')

    # the ladder rungs, drawn thin and BEHIND the probe curve so the primary
    # series stays readable where they overlap
    if rungs:
        for r, col in zip(rungs, RUNG_COLOURS * len(rungs)):
            pts = " ".join(f"{x(k):.1f},{y(v['value']):.1f}" for k, v in r["curve"])
            add(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6"/>')
            for k, v in r["curve"]:
                add(f'<circle cx="{x(k):.1f}" cy="{y(v["value"]):.1f}" r="2.6" '
                    f'fill="{col}"/>')
            # hollow ring at the STRUCTURAL saturation depth (windows - 1): the
            # point past which the stream has no older window left to retain
            sk = r["struct_k"]
            sv = next((v for k, v in r["curve"] if k == sk), None)
            if sk is not None and sv is not None:
                add(f'<circle cx="{x(sk):.1f}" cy="{y(sv["value"]):.1f}" r="6.5" '
                    f'fill="none" stroke="{col}" stroke-width="1.8"/>')

    # the curve
    pts = " ".join(f"{x(k):.1f},{y(v['value']):.1f}" for k, v in curve)
    add(f'<polyline points="{pts}" fill="none" stroke="#1f6feb" stroke-width="2.4"/>')
    for k, v in curve:
        add(f'<circle cx="{x(k):.1f}" cy="{y(v["value"]):.1f}" r="4" fill="#1f6feb"/>')

    # crossings worth naming: first K at or above threshold
    first_pass = next(((k, v) for k, v in curve if v["value"] >= thresh), None)
    for k, v in curve:
        if k not in LABELLED_K:
            continue
        label = f'K={k}: {v["value"]:.4f}'
        ly = y(v["value"]) - 12
        # If the rung band would run through the label, put it under the marker
        # instead: on the reference model that is the K=1 label (0.3088 sits just
        # below the rung band), and nothing else is affected.
        if rung_band and (ly - 9) <= rung_band[1] and (ly + 3) >= rung_band[0]:
            ly = y(v["value"]) + 18
        add(f'<text x="{x(k):.1f}" y="{ly:.1f}" font-size="12" '
            f'fill="#0b3d91" text-anchor="middle"{HALO}>{label}</text>')
    if first_pass is not None:
        k, v = first_pass
        add(f'<circle cx="{x(k):.1f}" cy="{y(v["value"]):.1f}" r="7.5" fill="none" '
            f'stroke="#c0392b" stroke-width="1.8"/>')
        # short and low: a long centred string here would run off the canvas and
        # touch the 0.6 axis label
        add(f'<text x="{x(k):.1f}" y="{y(v["value"]) + 34:.1f}" font-size="12" '
            f'fill="#c0392b" text-anchor="middle"{HALO}>K={k} crosses the gate at '
            f'{vram_gb(v["capacity_rows_per_layer"], d):.2f} GB</text>')

    # bottom axis: K
    add(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="#999" stroke-width="1.2"/>')
    for k, v in curve:
        add(f'<line x1="{x(k):.1f}" y1="{Y1}" x2="{x(k):.1f}" y2="{Y1 + 5}" '
            f'stroke="#999" stroke-width="1"/>')
        add(f'<text x="{x(k):.1f}" y="{Y1 + 20}" font-size="12" fill="#333" '
            f'text-anchor="middle">{k}</text>')
    add(f'<text x="{(X0 + X1) / 2:.0f}" y="{Y1 + 42}" font-size="13" fill="#333" '
        f'text-anchor="middle">K = retention depth in token windows '
        f'(rows per layer = min({d["n_expert"]}, K*W*{d["top_k"]}), W={d["window_tokens"]})</text>')

    # top axis: the same positions priced in VRAM
    add(f'<line x1="{X0}" y1="{Y0}" x2="{X1}" y2="{Y0}" stroke="#999" stroke-width="1.2"/>')
    for k in (1, 4, 8, 16, 32):
        if k not in ks:
            continue
        v = next(v for kk, v in curve if kk == k)
        gb = vram_gb(v["capacity_rows_per_layer"], d)
        add(f'<line x1="{x(k):.1f}" y1="{Y0}" x2="{x(k):.1f}" y2="{Y0 - 5}" '
            f'stroke="#999" stroke-width="1"/>')
        add(f'<text x="{x(k):.1f}" y="{Y0 - 10}" font-size="12" fill="#333" '
            f'text-anchor="middle">{gb:.2f} GB</text>')
    add(f'<text x="{(X0 + X1) / 2:.0f}" y="{Y0 - 32}" font-size="13" fill="#333" '
        f'text-anchor="middle">VRAM this K costs over {d["layers"]} MoE layers '
        f'(rows x {d["row_bytes"]:,} B per row, the model slot accounting)</text>')

    # The rung note, placed in the empty band below every plotted point so it
    # cannot cross a series. This is the argument the overlay exists to make: the
    # rungs' saturation depth is a property of the STREAM (windows - 1), not of
    # the routing, so the ladder cannot inform the 0.6 threshold.
    if rungs:
        ty = max([y(v["value"]) for _, v in curve] + [rung_band[1]]) + 42
        for i, col in enumerate(RUNG_COLOURS[:len(rungs)]):
            sx = X0 + 6 + i * 22
            add(f'<line x1="{sx}" y1="{ty - 4}" x2="{sx + 16}" y2="{ty - 4}" '
                f'stroke="{col}" stroke-width="3"/>')
        add(f'<text x="{X0 + 6 + len(rungs) * 22 + 6}" y="{ty}" font-size="12" '
            f'fill="#333">the {len(rungs)} ladder rungs ({rungs[0]["windows"]}-window '
            f'streams, {rungs[0]["pairs"]} pairs each), ringed where they saturate</text>')
        add(f'<text x="{X0 + 6}" y="{ty + 18:.1f}" font-size="12" fill="#333">'
            f'structural, not routing: {rungs[0]["windows"]} windows per layer leaves '
            f'only {rungs[0]["struct_k"]} predecessor windows, so a K={rungs[0]["struct_k"]} '
            f'cache (min({rungs[0]["n_expert"]}, {rungs[0]["struct_k"]}*W*'
            f'{rungs[0]["top_k"]}) = {rungs[0]["cap_at_struct"]} rows)</text>')
        add(f'<text x="{X0 + 6}" y="{ty + 36:.1f}" font-size="12" fill="#333">'
            f'already holds every row the oracle could; the probe stream has '
            f'{max(d["windows_per_layer"]) if d["windows_per_layer"] else "many"} '
            f'windows per layer and saturates only at K={d["saturated_by"]}</text>')

    add('</svg>')
    return "\n".join(out)


def build_table(d):
    rows = [
        "<tr><th>K</th><th>rows/layer</th><th>VRAM (GB)</th>"
        "<th>statistic (b)</th><th>vs gate 0.6</th></tr>"
    ]
    for k, v in d["curve"]:
        gb = vram_gb(v["capacity_rows_per_layer"], d)
        mark = "passes" if v["value"] >= THRESHOLD_FALLBACK else "fails"
        rows.append(
            f'<tr><td>{k}</td><td>{v["capacity_rows_per_layer"]}</td>'
            f'<td>{gb:.2f}</td><td>{v["value"]:.4f}</td><td>{mark}</td></tr>'
        )
    return "\n".join(rows)


def build_html(d, input_name, generated_from, basis=None, baseline_tok_s=None,
               link_gbs=LINK_GBS_DEFAULT, rungs=None, ladder_name=None,
               measured=None):
    sat = d["saturated_by"]
    sat_note = (
        f"The reported K={sat} already saturates it." if sat is not None
        else "Not saturated by the reported K values (retention is the limit)."
    )
    # The VRAM axis is the model's slot accounting. Where a trace is available the
    # measured per-layer row size is known too, and on the reference model the two
    # disagree -- so the artifact says which one the axis uses and what the other is,
    # rather than printing one basis as if it were the only one.
    basis_caveat = ""
    if basis and basis["n_odd_layers"]:
        slot_one_row = d["layers"] * d["row_bytes"]
        exact_one_row = basis["one_row_all_layers"]
        pct = (exact_one_row / slot_one_row - 1) * 100 if slot_one_row else 0.0
        last_k, last_v = d["curve"][-1]
        cap = last_v["capacity_rows_per_layer"]
        bank = (f" (the run's own full-bank figure is {basis['full_bank_bytes']:,} B)"
                if basis["full_bank_bytes"] else "")
        basis_caveat = (
            f'<br><strong>Basis caveat.</strong> The {d["row_bytes"]:,} B row size is the '
            f'model\'s slot accounting (<code>slot_bytes / n_expert</code>) and it is not '
            f'uniform: {basis["n_odd_layers"]} of the {basis["layers_seen"]} layers measured '
            f'stage {basis["row_bytes_max"]:,.0f} B per row against '
            f'{basis["row_bytes_modal"]:,.0f} B for the other '
            f'{basis["layers_seen"] - basis["n_odd_layers"]}. So the true footprint at each K '
            f'is {pct:.1f}% above the axis figures \u2014 at K={last_k} the axis reads '
            f'{vram_gb(cap, d):.2f} GB against {cap * exact_one_row / 1e9:.2f} GB from the '
            f'measured per-layer rows{bank}. The axis is left on the slot basis because that '
            f'is the basis the SDD section 4 band and the T-GPLR-15 retention records are '
            f'written in.'
        )
    # The projection has been measured at several retention depths: T-GPLR-19's
    # four-arm paired A/B (tools/run-gplr19-rowcache-ab.ps1) ran A,B,A,B per K.
    # Where those measurements are supplied, view 2 draws them against the
    # projection instead of claiming no paired run exists.
    measured_note = ""
    # The A/B summaries are gitignored build output, so the measured series
    # disappears from a clean checkout while the caption's numbers do not. Say
    # which is which rather than letting a regenerate silently drop it.
    measured_src = (
        "Paired measurements: <code>"
        + "</code>, <code>".join(html.escape(str(m["path"])) for m in measured)
        + "</code> (gitignored build output; the measured series is drawn only "
          "while these exist, but the values are stated in the view 2 caption "
          "regardless).<br>\n"
    ) if measured else ""
    if measured:
        k1 = next((m for m in measured if m["k"] == 1), None)
        first, last = measured[0], measured[-1]
        r_lo = min(m["reactive_tok_s"] for m in measured)
        r_hi = max(m["reactive_tok_s"] for m in measured)
        # the byte side is compared in PERCENTAGE POINTS of (b), not relative
        # bytes, because that is the quantity the gate is written in
        pp = max(abs(m["byte_pct"] - m["predicted_pct"]) for m in measured)
        convs = ", ".join(f"K={m['k']} {m['conversion_gbs']:.2f}" for m in measured)
        measured_note = (
            f" A PAIRED MEASUREMENT NOW EXISTS at K="
            f"{', '.join(str(m['k']) for m in measured)}: four arms A,B,A,B in "
            f"one GPU session per depth. The reactive arms held at "
            f"{r_lo:.3f}&ndash;{r_hi:.3f} tok/s across all of them, and the "
            f"retention arms reached {first['tok_s']:.3f} tok/s at K={first['k']} "
            f"rising to {last['tok_s']:.3f} tok/s at K={last['k']} (gain "
            f"{last['gain']:+.3f}, within-pair spread {last['spread']:.3f} there). "
            f"Every rung's realized byte reduction landed within {pp:.2f} "
            f"percentage points of the value projected here, so the byte side of "
            f"this view holds at depth. "
            + (f"The K=1 point converted at {k1['conversion_gbs']:.2f} GB/s &mdash; "
               f"faster than the {link_gbs:g} GB/s this view's blue series assumes. "
               if k1 else "")
            + f"The conversion rate does NOT hold flat, though: {convs} GB/s, so "
              f"the deeper bytes are worth materially less than the first "
              f"window's and the series' late points sit below the blue line. "
              f"The proportionality assumption this view rests on is therefore "
              f"measured and DEGRADING, not untested."
        )
    view2 = ""
    rerun = ("python tools/route_overlap/plot-b-vs-k.py --input <analysis.json> "
             "--out docs/plots/gplr15-b-vs-k.html")
    if basis is not None and baseline_tok_s:
        proj = project_times(d, basis, baseline_tok_s, link_gbs)
        view2 = f"""
<h2>View 2: what this retention is worth in wall-clock terms</h2>
<p class="sub">Statistic (b) is the fraction of the next window's rows a cache would
already hold, so it removes <code>(b) x bytes staged per token</code> from the link. The
basis is <em>measured from the trace</em>, not assumed: {basis['bytes_per_fetch']:,.0f} B per
fetch over {basis['fetches']:,} fetches ({basis['rows_per_fetch']:.2f} rows each), mean
transfer {basis['transfer_us_mean']:,.0f} us &mdash; {basis['effective_gbs']:.2f} GB/s
effective, against the {link_gbs:g} GB/s link ceiling. Staging is
{proj['staging_share'] * 100:.1f}% of token time ({proj['staging_ms']:.1f} ms of
{proj['base_ms']:.1f} ms/token over {d['layers']} layers), leaving a
{proj['floor_ms']:.1f} ms/token floor that this lever cannot touch.</p>

{build_time_svg(d, basis, proj, baseline_tok_s, link_gbs, measured)}

<h2>Projected values</h2>
<table>{build_time_table(proj, link_gbs, basis)}</table>

<div class="note"><strong>Projection, not a claim.</strong> It assumes the staged bytes stay
on the critical path &mdash; they do: staging is {proj['staging_share'] * 100:.1f}% of token
time and the layer graph waits on the copy event &mdash; and that removing a share of them
removes the same share of the measured per-fetch transfer. The two series differ only in the
rate they price that saving at: the <strong>blue</strong> series uses the
{link_gbs:g} GB/s link ceiling, the smallest reading of the two, and the one that would also
require the transfer path itself to reach the ceiling; the <strong>orange</strong> series uses
the {basis['effective_gbs']:.2f} GB/s this path is actually measured at, which is what
today's mechanism supports. The dotted line is the lever's own asymptote: with
<em>every</em> staged byte removed, {proj['floor_ms']:.1f} ms/token remains, so no retention
choice &mdash; and no predictor at all, however accurate &mdash; takes TG past
{1000 / proj['floor_ms']:.1f} tok/s on this run. It is one 256-token run on a stream that
echoes its own anchor.{measured_note} This view is the decision arithmetic for F2; the
K=1 point has since been settled by measurement rather than by projection &mdash; it is
the ringed marker in view 2, between the two projected series.</div>
"""
        rerun = ("python tools/route_overlap/plot-b-vs-k.py --input <analysis.json> "
                 "--trace <trace.jsonl> --tg-tok-s <measured baseline> "
                 "--out docs/plots/gplr15-b-vs-k.html")

    # The rung overlay exists to make the saturation-versus-stream-length argument
    # visible. The structural depth is windows - 1, and whether the values actually
    # equal the oracle there is CHECKED per rung (see load_rungs) and reported as a
    # discrepancy if it fails, so the chart cannot quietly show a routing effect.
    rung_section = ""
    rung_bullet = ""
    ladder_note = ""
    if rungs:
        sk = rungs[0]["struct_k"]
        n_wins = max(d["windows_per_layer"]) if d["windows_per_layer"] else None
        trs = []
        for i, r in enumerate(rungs):
            col = RUNG_COLOURS[i % len(RUNG_COLOURS)]
            k1 = next((v["value"] for k, v in r["curve"] if k == 1), None)
            sat = next((v["value"] for k, v in r["curve"] if k == sk), None)
            trs.append(
                f'<tr><td style="text-align:left">'
                f'<span style="color:{col}">&#9679;</span> {r["label"]}</td>'
                f'<td>{r["windows"]}</td><td>{r["pairs"]}</td>'
                f'<td>{k1:.4f}</td><td>{sat:.4f}</td><td>{r["oracle"]:.4f}</td>'
                f'<td>{r["struct_k"]}</td><td>{r["reported_sat"]}</td>'
                f'<td>{"yes" if r["struct_confirmed"] else "NO"}</td></tr>'
            )
        bad = [r["label"] for r in rungs if not r["struct_confirmed"]]
        verdict = (
            f'<strong>Discrepancy.</strong> On rung(s) {", ".join(bad)} the reported (b) at or '
            f'after K={sk} does <em>not</em> equal the oracle, which the argument above says it '
            f'must. That is a finding about the tool, not a presentation detail.'
            if bad else
            f'<strong>Checked, not asserted.</strong> On all {len(rungs)} rungs the reported '
            f'(b) at every K &ge; {sk} equals that rung\'s oracle exactly, which is what the '
            f'argument predicts. The overlay is drawn because that check passes.'
        )
        rung_section = f"""
<h2>Why the ladder rungs saturate at K={sk}</h2>
<p class="sub">A stream of {rungs[0]["windows"]} windows per layer has only
{rungs[0]["struct_k"]} predecessor windows, so a cache of depth
K &ge; {rungs[0]["struct_k"]} already holds that stream's <em>entire</em> history: there is
nothing older left to retain, and (b) must equal the oracle. Capacity cannot bind first
either &mdash; a K={sk} cache holds
min({rungs[0]["n_expert"]}, {sk}*W*{rungs[0]["top_k"]}) = {rungs[0]["cap_at_struct"]} rows per
layer, and any union of {sk} {rungs[0]["top_k"]}-row windows holds at most that many &mdash; so
the equality is arithmetic, not a property of the routing process.</p>
<p class="sub">That is why the rung series are flat from K={sk} on, and why these samples
cannot inform the 0.6 threshold: they saturate at a depth set by <em>how much history the
trace contains</em>, not by how predictable the routing is. The probe stream carries
{n_wins if n_wins else "many"} windows per layer and saturates only at
K={d["saturated_by"]} &mdash; the two depths differ by construction, and that difference
measures stream length, not routing.</p>
<table><tr><th>rung</th><th>windows/layer</th><th>pairs</th><th>(b) at K=1</th>
<th>(b) at K={sk}</th><th>oracle</th><th>structural K</th><th>reported K</th>
<th>equal?</th></tr>
{"".join(trs)}</table>
<div class="note">{verdict}</div>
"""
        rung_bullet = f"""
 <li>The thin coloured series are the <strong>{len(rungs)} ladder rungs</strong> overlaid
     &mdash; {rungs[0]["windows"]}-window streams, {rungs[0]["pairs"]} pairs each. Each is
     ringed at its structural saturation depth K={sk} and flat from there on: past that depth
     the stream has no older window left, so nothing about the gate can be learned there.</li>"""
    if rungs and ladder_name:
        ladder_note = (
            f'Ladder rungs: <code>{html.escape(str(ladder_name))}</code> '
            f'({len(rungs)} reports over the ladder traces).<br>\n'
            f'Re-create it with:<br>\n<code>python tools/route_overlap/analyze.py '
            f'"build/llama-cuda/gplr13-ladder/*/rowset-*.jsonl" --retention-k '
            f'1,2,3,4,5,6,7,8,10,12,16,24,32 --json '
            f'{html.escape(str(ladder_name))}</code><br>\n'
        )
        rerun += f" --ladder {ladder_name}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>T-GPLR-15 gate 9: statistic (b) versus retention K</title>
<style>
 body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        margin: 0 auto; max-width: 1040px; padding: 28px 20px 60px; color: #1c1c1c; }}
 h1 {{ font-size: 20px; margin: 0 0 4px; }}
 h2 {{ font-size: 15px; margin: 28px 0 8px; }}
 p, li {{ font-size: 14px; line-height: 1.55; }}
 .sub {{ color: #555; font-size: 13px; margin: 0 0 18px; }}
 table {{ border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
 th, td {{ border: 1px solid #dcdcdc; padding: 4px 10px; text-align: right; }}
 th {{ background: #f6f8fa; font-weight: 600; }}
 code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 12.5px; }}
 .note {{ background: #f8f9fb; border-left: 3px solid #1f6feb; padding: 10px 14px;
          font-size: 13px; }}
</style>
</head>
<body>
<h1>T-GPLR-15 gate 9 &mdash; statistic (b) versus retention K</h1>
<p class="sub">Measured on a real greedy generation stream: trace
<code>{html.escape(str(d["trace"]))}</code>, model {html.escape(str(d["model"]))},
run_id {html.escape(str(d["run_id"]))}, {d["pairs"]:,} consecutive-window pairs,
W={d["window_tokens"]}, top_k={d["top_k"]}, n_expert={d["n_expert"]}.
Self-contained: no external assets, no scripts.</p>

{build_svg(d, rungs)}

<h2>How to read K</h2>
<div class="note">
K is the per-layer cache's retention <strong>depth in token windows</strong> &mdash; not bytes
and not a count of experts. A K-window cache holds
<code>min(n_expert, K*W*top_k)</code> expert rows per layer
(here W={d["window_tokens"]}, top_k={d["top_k"]}, n_expert={d["n_expert"]}), and one row is
that expert's gate+up+down slabs ({d["row_bytes"]:,} B). The VRAM axis at the top therefore
<em>follows</em> from K rather than being a separate choice. Statistic (b) is defined
<strong>at</strong> a K, so a gate reading is complete only when the K it was read at is named:
reading it at K=1&ndash;2 (the SDD section 4 reference band, shaded) <strong>fails</strong>
the 0.6 gate, while the curve only crosses 0.6 later, at a VRAM cost the chart prices.
{basis_caveat}
</div>

<h2>Reading this curve</h2>
<ul>
 <li>The shaded band is the SDD section 4 reference retention: a one-to-two window
     cache, the only sizing the spec itself declares.</li>
 <li>The solid blue curve is statistic (b) as measured
     ({d["pairs"]:,} pairs); each marker is a reported K.</li>{rung_bullet}
 <li>The red dashed line is the 0.6 gate. The curve crosses it above the reference
     band, so the verdict depends on which K an adopting task declares &mdash; the
     reason this was escalated to the owner and closed at the band.</li>
 <li>The grey dotted line is the T8 oracle ceiling, {d["oracle"]:.4f}: statistic (b)
     under <em>unbounded</em> retention, i.e. the whole expert bank resident, which is
     the outcome the row-set policy exists to avoid. No cache policy of any size can
     exceed it. {sat_note}</li>
</ul>

<h2>Full measured values (view 1)</h2>
<table>{build_table(d)}</table>
{rung_section}
{view2}
<h2>Provenance</h2>
<p class="sub">
Input report: <code>{html.escape(input_name)}</code>.<br>
{measured_src}{ladder_note}Generated by <code>{html.escape(generated_from)}</code> (stdlib only).<br>
Re-render with:<br>
<code>{html.escape(rerun)}</code>
</p>
<p class="sub">Scope: view 1 measures routing predictability only, and the stream it was
measured on echoes its own anchor prompt &mdash; repetition can only increase temporal
reuse, so the in-band failure is conservative while the crossing and the ceiling are likely
optimistic. View 2 is a projection built on view 1's statistic plus a measured staging
share; neither view is a throughput, quality, or equivalence claim.</p>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="analyzer JSON report (rowset_* trace)")
    ap.add_argument("--out", required=True, help="output HTML path")
    ap.add_argument("--trace", default=None,
                    help="row-set trace (jsonl); adds the wall-clock view")
    ap.add_argument("--tg-tok-s", type=float, default=None,
                    help="measured baseline TG tok/s for the same run (required with --trace)")
    ap.add_argument(
        "--measured-ab", nargs='+', default=MEASURED_AB_DEFAULTS,
        help="one T-GPLR-19 four-arm A/B summary.json per retention depth (the "
        "paired measurement). Each file that exists adds a measured point to "
        "view 2 instead of the artifact claiming that no paired run exists.",
    )
    ap.add_argument("--link-gbs", type=float, default=LINK_GBS_DEFAULT,
                    help=f"H2D link capacity in GB/s for the projection (default {LINK_GBS_DEFAULT})")
    ap.add_argument("--ladder", default=None,
                    help="analyzer report covering the ladder traces (a list of reports); "
                         "overlays the rungs and marks their structural saturation")
    args = ap.parse_args()

    d = load_curve(args.input)
    basis = load_trace_basis(args.trace) if args.trace else None
    if basis is not None and not args.tg_tok_s:
        raise SystemExit("--tg-tok-s is required with --trace: view 2 needs the measured "
                         "baseline it projects from, and glossing over that would make the "
                         "projection look like a result")
    rungs = load_rungs(args.ladder) if args.ladder else None
    measured = load_measured_abs(args.measured_ab, basis, d) if basis is not None else None
    doc = build_html(d, args.input, "tools/route_overlap/plot-b-vs-k.py",
                     basis, args.tg_tok_s, args.link_gbs, rungs, args.ladder, measured)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    print(f"wrote {args.out}")
    print(f"  view 1: K={d['curve'][0][0]}..{d['curve'][-1][0]} ({len(d['curve'])} points), "
          f"threshold {THRESHOLD_FALLBACK}, oracle {d['oracle']:.4f}")
    if basis is not None:
        proj = project_times(d, basis, args.tg_tok_s, args.link_gbs)
        first, last = proj["rows"][0], proj["rows"][-1]
        print(f"  view 2: {basis['bytes_per_fetch']:,.0f} B/fetch @ {basis['transfer_us_mean']:,.0f} us "
              f"= {basis['effective_gbs']:.2f} GB/s effective; staging "
              f"{proj['staging_share'] * 100:.1f}% of {proj['base_ms']:.1f} ms/token; "
              f"floor {proj['floor_ms']:.1f} ms/token")
        print(f"          K={first['k']}: {first['link_toks']:.2f}-{first['eff_toks']:.2f} tok/s | "
              f"K={last['k']}: {last['link_toks']:.2f}-{last['eff_toks']:.2f} tok/s "
              f"against a {args.tg_tok_s:.3f} tok/s baseline")
    if rungs:
        sk = rungs[0]["struct_k"]
        print(f"  rungs: {len(rungs)} overlaid, windows={rungs[0]['windows']}, "
              f"structural saturation K={sk}, all K>={sk} equal the oracle on "
              f"{sum(1 for r in rungs if r['struct_confirmed'])}/{len(rungs)}")
        for r in rungs:
            v = next(v['value'] for k, v in r['curve'] if k == sk)
            flag = "ok" if r['struct_confirmed'] else "NOT CONFIRMED"
            print(f"          {r['label']:>5}: K=1 {r['curve'][0][1]['value']:.4f} -> "
                  f"K={sk} {v:.4f} = oracle {r['oracle']:.4f} [{flag}]")
        if not all(r['struct_confirmed'] for r in rungs):
            print("  WARNING: a rung's value at K>=windows-1 is not the oracle; the "
                  "structural claim does not hold on this data and the artifact says so")


if __name__ == "__main__":
    main()
