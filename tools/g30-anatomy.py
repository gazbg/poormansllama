#!/usr/bin/env python3
"""G30 anatomy of a decode step with and without MTP (host-only; reads the run logs and the hybrid engine's jsonl).

    python tools/g30-anatomy.py <session> [<session> ...]       per run: TG, acceptance, tokens per step, the engine's timeline of one decode call, the draft stage
    python tools/g30-anatomy.py --pairs <session> ...            per session: mean of every arm and the ratio MTP / plain of TG and of each term
    python tools/g30-anatomy.py --dir <dir> <prefix>             a session directory that has no verdict (e.g. g41-qadm50000 / QKFMD)

Terms (microseconds per decode CALL; a call is one token without MTP and one verify round with MTP; the engine's `timeline` event):
  span   the call's GPU span (from the first layer to the last logits)        attn   the GPU front (attention + recurrent layers up to the host hand-off points)
  stall  GPU time waiting for host-served experts                              gap    time between two calls (sampling, launch, fetch)
  hserve host service of the missed experts (mean per layer with misses)       miss_layers  layers with at least one miss per call
  draft  the draft stage of an MTP round (from the runner's `MTP host timing` line)
Model of a call: plain token = span + gap; MTP round = span + gap too (the engine's gap runs from the end of one call to the launch of the next, so it CONTAINS the draft stage: TG tok/s x rounds reproduces span + gap),
yielding 1 + acceptance tokens (one draft per round). R_pred = (1 + a) * T_plain / T_round; it reproduced the measured R of g93-c512 (1.05 against 1.056).
"""
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "llama-cuda"))


def read(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace")


def last(pat, text, cast=float):
    m = None
    for m in re.finditer(pat, text):
        pass
    return cast(m.group(1)) if m else None


def run_info(logpath, hybpath):
    t = read(logpath)
    d = {"tg": last(r"TG: \d+ tokens in [\d.]+ ms \(([\d.]+) tok/s\)", t) or last(r"phase tg : \d+ tokens, ([\d.]+) tok/s", t),
         "pp_s": last(r"PP: \d+ tokens in ([\d.]+) ms", t), "pp_tokens": last(r"PP: (\d+) tokens in", t, int),
         "acc": last(r"MTP: drafted \d+, accepted \d+, acceptance ([\d.]+)", t),
         "rounds": last(r"MTP batch verify: rounds (\d+)", t, int),
         "draft_us": last(r"draft stage ([\d.]+)", t), "verify_host_us": last(r"verify llama_decode \(host side, up to the launch\) ([\d.]+)", t),
         "fetch_us": last(r"verify fetch \+ sampling ([\d.]+)", t),
         "slots": last(r"cache resized to (\d+) slots", t, int), "catchup_ms": last(r"draft-head catch-up over \d+ prompt positions .*? in ([\d.]+) ms", t),
         "free_min": last(r"free-watch: minimum free device memory ([\d.]+) MiB", t)}
    if hybpath and os.path.exists(hybpath):
        tl = hits = lookups = None
        for line in read(hybpath).splitlines():
            if '"timeline"' in line:
                try:
                    tl = json.loads(line)
                except Exception:
                    pass
            elif '"summary"' in line:
                try:
                    s = json.loads(line)
                    hits, lookups = s.get("hits"), s.get("lookups")
                except Exception:
                    pass
        if tl:
            for k in ("tokens", "span_us", "attn_us", "ovl_us", "stall_us", "gap_us", "hserve_us", "rt_us", "miss_layers"):
                d[k] = tl.get(k)
        if hits is not None and lookups:
            d["hit"] = hits / lookups
    return d


def fmt(v, spec="%.1f"):
    return "-" if v is None else spec % v


def session_runs(session):
    sdir = os.path.join(ROOT, session)
    runs = []
    for lp in sorted(glob.glob(os.path.join(sdir, "%s-*.log" % session))):
        base = os.path.basename(lp)[len(session) + 1:-4]
        m = re.match(r"^(.*?)(?:_r(\d+))?$", base)
        arm, rep = m.group(1), int(m.group(2) or 1)
        hyb = os.path.join(sdir, "hybrid_%s.jsonl" % base)
        runs.append((arm, rep, run_info(lp, hyb)))
    return runs


def print_runs(session, runs):
    print("== %s" % session)
    print("%-6s %3s %8s %6s %7s %7s %7s %7s %7s %7s %6s %6s %6s %6s %8s" % ("arm", "rep", "TG", "acc", "span", "attn", "stall", "gap", "hserve", "draft", "miss_l", "hit", "slots", "PP s", "catchup"))
    for arm, rep, d in runs:
        print("%-6s %3d %8s %6s %7s %7s %7s %7s %7s %7s %6s %6s %6s %6s %8s" % (
            arm, rep, fmt(d["tg"], "%.2f"), fmt(d["acc"], "%.3f"), fmt(d.get("span_us"), "%.0f"), fmt(d.get("attn_us"), "%.0f"), fmt(d.get("stall_us"), "%.0f"), fmt(d.get("gap_us"), "%.0f"),
            fmt(d.get("hserve_us"), "%.0f"), fmt(d["draft_us"], "%.0f"), fmt(d.get("miss_layers"), "%.1f"), fmt(d.get("hit"), "%.3f"), fmt(d["slots"], "%d"),
            fmt(None if d["pp_s"] is None else d["pp_s"] / 1000.0, "%.1f"), fmt(d["catchup_ms"], "%.0f")))


def mean_of(runs, arm, key):
    v = [d[key] for a, _, d in runs if a == arm and d.get(key) is not None]
    return statistics.mean(v) if v else None


def print_pairs(session, runs):
    arms = sorted({a for a, _, _ in runs})
    plain = "B0" if "B0" in arms else next((a for a in arms if not a.endswith("M")), None)
    mtps = [a for a in arms if a != plain and a not in ("B0S", "B0old", "B0new")]
    for mtp in mtps:
        print_pair(session, runs, plain, mtp)


def print_pair(session, runs, plain, mtp):
    print("== %s pairs (%s vs %s)" % (session, mtp, plain))
    if not plain or not mtp:
        return
    per_rep = {}
    for a, rep, d in runs:
        per_rep.setdefault(rep, {})[a] = d
    ratios = [per_rep[r][mtp]["tg"] / per_rep[r][plain]["tg"] for r in sorted(per_rep) if plain in per_rep[r] and mtp in per_rep[r] and per_rep[r][mtp]["tg"] and per_rep[r][plain]["tg"]]
    for key, label in (("tg", "TG tok/s"), ("acc", "acceptance"), ("span_us", "span us"), ("attn_us", "attn us"), ("stall_us", "stall us"), ("gap_us", "gap us"), ("hserve_us", "hserve us"),
                       ("miss_layers", "miss layers"), ("hit", "hit rate"), ("slots", "slots"), ("pp_s", "PP ms"), ("draft_us", "draft us"), ("catchup_ms", "catch-up ms")):
        a, b = mean_of(runs, plain, key), mean_of(runs, mtp, key)
        print("  %-12s %s %12s   %s %12s   ratio %s" % (label, plain, fmt(a, "%.3f" if key in ("acc", "hit") else "%.1f"), mtp, fmt(b, "%.3f" if key in ("acc", "hit") else "%.1f"),
                                                      "-" if not a or b is None else "%.3f" % (b / a)))
    pa, pb = mean_of(runs, plain, "pp_s"), mean_of(runs, mtp, "pp_s")
    if pa and pb:
        print("  PP ratio B0 / %s = %.3f (criterion G30-2: >= 0.95)" % (mtp, pa / pb))
    if ratios:
        print("  R = TG(%s)/TG(%s) per counterbalanced pair: %s -> mean %.3f, min %.3f, pairs > 1.00: %d of %d" % (
            mtp, plain, " ".join("%.3f" % r for r in ratios), statistics.mean(ratios), min(ratios), sum(1 for r in ratios if r > 1.0), len(ratios)))
    # the model of the round
    tp = mean_of(runs, plain, "span_us"), mean_of(runs, plain, "gap_us")
    tm = mean_of(runs, mtp, "span_us"), mean_of(runs, mtp, "gap_us"), mean_of(runs, mtp, "draft_us"), mean_of(runs, mtp, "acc")
    if None not in tp and None not in tm:
        t_plain = tp[0] + tp[1]
        t_round = tm[0] + tm[1]
        print("  model: plain call %.0f us; MTP round %.0f us (span %.0f + gap %.0f, the draft stage %.0f us is inside the gap) = %.2fx a plain call for %.3f tokens -> R_pred = %.3f" % (
            t_plain, t_round, tm[0], tm[1], tm[2], t_round / t_plain, 1 + tm[3], (1 + tm[3]) * t_plain / t_round))


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    pairs = False
    if args[0] == "--pairs":
        pairs, args = True, args[1:]
    if args[0] == "--dir":
        sdir, prefix = args[1], args[2]
        lp = os.path.join(ROOT, sdir, "%s-%s.log" % (sdir, prefix))
        d = run_info(lp, os.path.join(ROOT, sdir, "hybrid_%s.jsonl" % prefix))
        print_runs(sdir, [(prefix, 1, d)])
        return 0
    for s in args:
        runs = session_runs(s)
        if not runs:
            print("no runs found for", s)
            continue
        if pairs:
            print_pairs(s, runs)
        else:
            print_runs(s, runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
