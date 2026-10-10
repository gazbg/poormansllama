#!/usr/bin/env python3
"""Fit the streaming path's wall time against its own staged bytes.

Every T-GPLR-19 A/B run records, per arm, the bytes it staged per round and the
throughput it achieved, at identical peak VRAM and identical routes (same
`route_digest`).  Arm A stages everything (h = 0); arm B stages only the misses.
So each run is a two-point measurement of

    ms_per_token = fixed + slope * staged_bytes_per_token

with no model parameter anywhere in it.  `slope` is the marginal feed rate the
path actually achieves end to end, and `fixed` is everything the path costs that
is NOT transfer -- device expert compute, attention, the rowset machinery
itself.  `fixed` is the number that decides whether the path can ever win:
compare it against the incumbent's whole token.

Usage: stream-slope.py <summary.json> [...]
"""

import json
import sys

STOCK_MS = 1000.0 / 40.96  # stock anchor, --fit-target 16000, 1 session


def arms(path):
    """-> list of (label, h, bytes_per_token, ms_per_token, pp_tok_s, peak_vram).

    Handles both artifact shapes: the T-GPLR-19 A/B `summary.json` array, and
    the T-GPLR-58 `gplr58-kprobe-summary/1` object.  The kprobe legs are NOT a
    matched contrast -- each K is its own configuration with its own resident
    footprint -- so its intercept is not comparable to a stock anchor.  Its
    slope still is, because bytes still vary.
    """
    with open(path, "r", encoding="utf-8-sig") as fh:
        doc = json.load(fh)

    if isinstance(doc, dict) and str(doc.get("schema", "")).startswith("gplr58-kprobe"):
        n_tg = doc["shape"]["tg_tokens"]
        out = []
        for name, leg in doc.get("legs", {}).items():
            tg = leg.get("leg_tg_toks")
            if not tg:
                continue  # halted or telemetry-gapped leg
            out.append((
                name.replace("gplr58-kprobe-", ""),
                leg.get("leg_share", 0.0),
                tg["tg_bytes_gib"] * (1024 ** 3) / n_tg,
                1000.0 * tg["tg_wall_s"] / n_tg,
                0.0,  # this harness records no PP rate per leg
                int(tg["peak_vram_gib"] * (1024 ** 3)),
            ))
        return out

    out = []
    for r in doc:
        res = r.get("rows_resident", 0)
        fet = r.get("rows_fetched", 0)
        tot = res + fet
        tg = float(r["tg_tok_s"])
        out.append((
            r.get("arm", "?"),
            (res / tot) if tot else 0.0,
            float(r.get("staged_per_round", 0)),
            1000.0 / tg,
            float(r.get("pp_tok_s", 0)),
            r.get("peak_vram_bytes", 0),
        ))
    return out


def fit(points):
    """Least squares of ms against bytes. -> (fixed_ms, gb_per_s)."""
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx == 0:
        raise SystemExit("all arms staged the same bytes: nothing to fit")
    slope = sxy / sxx                      # ms per byte
    fixed = my - slope * mx
    return fixed, 1.0 / (slope * 1e6)      # ms/B -> B/s /1e9 == 1/(slope*1e6) GB/s


def main(argv):
    pts = []
    for path in argv:
        try:
            rows = arms(path)
        except (OSError, ValueError, KeyError) as e:
            print("!! %s: %s" % (path, e), file=sys.stderr)
            continue
        print("\n== %s" % path)
        print("   %-14s %8s %14s %10s %10s %10s"
              % ("arm", "h", "staged B/tok", "ms/tok", "tg tok/s", "pp tok/s"))
        vram = set()
        for label, h, b, ms, pp, pv in rows:
            print("   %-14s %8.4f %14.0f %10.2f %10.2f %10.2f"
                  % (label, h, b, ms, 1000.0 / ms, pp))
            pts.append((b, ms))
            vram.add(pv)
        if len(vram) > 1:
            print("   !! arms differ in peak VRAM %s -- the intercept is NOT a "
                  "matched measurement; use the slope only" % sorted(vram),
                  file=sys.stderr)
        pp_all = [r[4] for r in rows]
        if min(pp_all) > 0:
            print("   prefill spread across arms: %.3f - %.3f tok/s (%.2f%%)"
                  % (min(pp_all), max(pp_all),
                     100.0 * (max(pp_all) - min(pp_all)) / min(pp_all)))

    if len(pts) < 2:
        return 1

    fixed, gbs = fit(pts)
    print("\n== pooled fit over %d arms" % len(pts))
    print("   marginal feed          %.2f GB/s" % gbs)
    print("   fixed cost per token   %.2f ms   (everything that is NOT transfer)"
          % fixed)
    print("   stock anchor           %.2f ms = 40.96 tok/s" % STOCK_MS)
    verdict = ("CANNOT reach stock even at h = 1"
               if fixed > STOCK_MS else
               "could in principle beat stock at high enough h")
    print("   => streaming path %s (%.2f tok/s at h = 1)"
          % (verdict, 1000.0 / fixed))

    print("\n   residuals:")
    slope_ms_per_b = 1.0 / (gbs * 1e6)
    for b, ms in sorted(pts, key=lambda p: -p[0]):
        pred = fixed + slope_ms_per_b * b
        print("   staged %10.0f B  measured %6.2f ms  fit %6.2f ms  %+5.1f%%"
              % (b, ms, pred, 100.0 * (pred - ms) / ms))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
