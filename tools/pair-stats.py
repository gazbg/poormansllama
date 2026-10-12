#!/usr/bin/env python3
"""Paired statistics of two arms of one run-tg-arms.py session (counterbalanced reps).

    python tools/pair-stats.py <session verdict.json> <control arm> <test arm> [--bar PCT] [--min-pos N]

Per rep (pair): TG tok/s of both arms, the paired gain in percent, the generated-text hash of each arm, the whole-GPU VRAM peak of the poller, the engine's own
statistics read from the arm log (hit rate, host requests, serve avg / max, slow-request warnings, slots) and the guards. The summary prints the mean of each arm,
the mean paired gain, the number of positive pairs, and whether the criteria "mean gain >= --bar % and >= --min-pos positive pairs" hold (default 2.0 and all but one).
Nothing here moves a criterion; it only computes what the contract names.
"""
import argparse
import hashlib
import json
import re
import statistics
import sys


def read_log(path):
    try:
        raw = open(path, "rb").read()
    except OSError:
        return ""
    return raw.replace(b"\x00", b"").decode("utf-8", "replace")


def text_hash(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    except OSError:
        return "-"


def engine_stats(log):
    out = {}
    m = re.search(r"tokens (\d+), hit rate ([\d.]+) \((\d+)/(\d+)\).*?host requests (\d+), serve avg ([\d.]+) max ([\d.]+) us", log)
    if m:
        out.update(tokens=int(m.group(1)), hit=float(m.group(2)), requests=int(m.group(5)), serve_avg=float(m.group(6)), serve_max=float(m.group(7)))
    m = re.search(r"timeline per token over (\d+) tokens \(us\): span ([\d.]+) = attn ([\d.]+) \+ ovl ([\d.]+) \+ stall ([\d.]+), gap ([\d.]+)", log)
    if m:
        out.update(span=float(m.group(2)), attn=float(m.group(3)), ovl=float(m.group(4)), stall=float(m.group(5)), gap=float(m.group(6)))
    m = re.search(r"cpu graph avg by miss count \[1\.\.\d+\] ([\d.,]+)", log)
    if m:
        out["cg"] = [float(x) for x in m.group(1).split(",") if x]
    out["slow"] = len(re.findall(r"slow host request", log))
    m = re.search(r"cache resized to (\d+) slots", log)
    if m:
        out["slots"] = int(m.group(1))
    m = re.search(r"lean host leg mode (\d): (\d+) requests through the lean team(?:, each compared bit for bit with the graph: (\d+) DIFFER)?", log)
    if m:
        out["lean_mode"] = int(m.group(1))
        out["lean_requests"] = int(m.group(2))
        if m.group(3) is not None:
            out["lean_differ"] = int(m.group(3))
    return out


def fit(cg):
    """fixed + slope*m least-squares over the miss counts that have a value"""
    pts = [(i + 1, v) for i, v in enumerate(cg) if v > 0]
    if len(pts) < 3:
        return None
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - slope * sx) / n, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdict")
    ap.add_argument("control")
    ap.add_argument("test")
    ap.add_argument("--bar", type=float, default=2.0, help="required mean paired gain in percent")
    ap.add_argument("--min-pos", type=int, default=None, help="required number of positive pairs (default: all but one)")
    a = ap.parse_args()

    d = json.load(open(a.verdict))
    arms = d["arms"]

    def rep_arm(name):
        out = {}
        for k, v in arms.items():
            if v.get("base") == name:
                out[v["rep"]] = v
        return out

    c, t = rep_arm(a.control), rep_arm(a.test)
    reps = sorted(set(c) & set(t))
    if not reps:
        print("no common reps")
        return 2
    print("session %s | power %s | attest %s" % (d.get("session"), d.get("power", {}).get("ac_line"), ",".join(d.get("attest", []))))
    print("%-4s %9s %9s %8s  %-12s %-12s  %6s %6s  %9s %9s  %5s %5s" % ("rep", a.control, a.test, "gain%", "text(c)", "text(t)", "hit(c)", "hit(t)", "serve(c)", "serve(t)", "slots", "peak"))
    gains, tc, tt, serve_c, serve_t, cgs_c, cgs_t = [], [], [], [], [], [], []
    ok = True
    same_text = 0
    for r in reps:
        ac, at = c[r], t[r]
        if ac["rc"] != 0 or at["rc"] != 0:
            ok = False
        sc, st = ac["summary"]["tg_tokens_per_s"], at["summary"]["tg_tokens_per_s"]
        g = 100.0 * (st / sc - 1.0)
        gains.append(g)
        tc.append(sc)
        tt.append(st)
        ec, et = engine_stats(read_log(ac["log"])), engine_stats(read_log(at["log"]))
        hc, ht = text_hash(ac["tgtext"]), text_hash(at["tgtext"])
        same_text += hc == ht
        if "serve_avg" in ec:
            serve_c.append(ec["serve_avg"])
        if "serve_avg" in et:
            serve_t.append(et["serve_avg"])
        if "cg" in ec:
            cgs_c.append(ec["cg"])
        if "cg" in et:
            cgs_t.append(et["cg"])
        print("%-4d %9.2f %9.2f %+8.2f  %-12s %-12s  %6s %6s  %9s %9s  %5s %5s%s%s" % (
            r, sc, st, g, hc, ht, ec.get("hit", "-"), et.get("hit", "-"), ec.get("serve_avg", "-"), et.get("serve_avg", "-"),
            et.get("slots", "-"), at.get("vram_peak_mib", "-"),
            "  SLOW-REQ(c=%d,t=%d)" % (ec.get("slow", 0), et.get("slow", 0)) if ec.get("slow") or et.get("slow") else "",
            "  GUARDS" if any(ac["guards"].values()) or any(at["guards"].values()) else ""))
        if any(ac["guards"].values()) or any(at["guards"].values()) or ac.get("check_telemetry_rc") or at.get("check_telemetry_rc"):
            ok = False

    n_pos = sum(1 for g in gains if g > 0)
    need = a.min_pos if a.min_pos is not None else max(1, len(gains) - 1)
    mean_gain = statistics.mean(gains)
    print()
    print("%s mean %.3f (min %.2f max %.2f) | %s mean %.3f (min %.2f max %.2f)" % (a.control, statistics.mean(tc), min(tc), max(tc), a.test, statistics.mean(tt), min(tt), max(tt)))
    print("mean paired gain %+.2f%% | ratio of means %+.2f%% | positive pairs %d of %d (need %d) | identical text in %d of %d pairs" % (
        mean_gain, 100.0 * (statistics.mean(tt) / statistics.mean(tc) - 1.0), n_pos, len(gains), need, same_text, len(gains)))
    if serve_c and serve_t:
        print("engine serve avg per request: %s %.1f us, %s %.1f us (saved %.1f us)" % (a.control, statistics.mean(serve_c), a.test, statistics.mean(serve_t), statistics.mean(serve_c) - statistics.mean(serve_t)))
    for name, cgs in ((a.control, cgs_c), (a.test, cgs_t)):
        if cgs:
            avg = [statistics.mean(col) for col in zip(*cgs)]
            f = fit(avg)
            print("host leg time by miss count [1..8] %s: %s%s" % (name, ",".join("%.1f" % x for x in avg), "  fit %.1f us fixed + %.1f us per expert" % f if f else ""))
    verdict = ok and mean_gain >= a.bar and n_pos >= need
    print("criterion: mean gain >= %.1f%% and >= %d positive pairs and all runs clean: %s" % (a.bar, need, "MET" if verdict else "NOT MET"))
    return 0 if verdict else 3


if __name__ == "__main__":
    sys.exit(main())
