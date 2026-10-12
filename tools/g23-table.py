#!/usr/bin/env python3
"""G23 (the G22 prototype, stages A, B, C): tables and criterion checks of the sessions g50-<N> (C), g56-stock-<N> (A) and g57-pace-<N> (B). Host-only.

    python tools/g23-table.py hybrid [N ...]    g50-<N>: every arm's prefill wall per rep, C2/C3/C4 (default N = 32 64 128 256 512 1024 2048)
    python tools/g23-table.py stock  [N ...]    g56-stock-<N>: runner stock path against the stock binary, A1/A2 (default N = 3 10 14 29 63 125 256 511 1023 2047)
    python tools/g23-table.py pace   [N ...]    g57-pace-<N>: paced bank copies, B1-B4 (default N = 63 256 1023)
    python tools/g23-table.py final  [N ...]    g68-<N>: the final-build comparison (STK STKNO STKU LM2W LM2P HYC HY4P HY4Q HY8Q), C2-C5 (default N = 29 63 125 256 511 1023 2047)
    python tools/g23-table.py ident  [N ...]    g69-ident-<N>: B4 on the final build, three reps (default N = 63 256 1023)
    python tools/g23-table.py deep   [N ...]    g70-pacedeep-<N>: paced copies on longer prompts (default N = 4096 24000 50000)
    python tools/g23-table.py keep   [N ...]    g84-<N> (CPU-only paths against stock, with and without --gpu-keepalive) and g85-<N> (cold and warm pass), both passes shown (default N = 511 1023 / 63 256 1023)

The prefill wall of a stock arm is the `prompt eval time` of llama-completion; of a runner arm the LAST `PP: N tokens in X ms` line (the warm pass of --pp-repeat 2). Token identity comes from the
`tg_tokens` event of the runner trace. Reps are paired by their rep id (the arms of a rep run in one counterbalanced round).
"""
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "llama-cuda")


def read(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace")


def prefill(text):
    m = None
    for m in re.finditer(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", text):
        pass
    if m:
        return float(m.group(1)), int(m.group(2))
    for m in re.finditer(r"PP: (\d+) tokens in ([\d.]+) ms", text):
        pass
    return (float(m.group(2)), int(m.group(1))) if m else None


def passes(text):
    """every `PP: N tokens in X ms` line (one per in-process pass)"""
    return [float(m.group(2)) for m in re.finditer(r"PP: (\d+) tokens in ([\d.]+) ms", text)]


def tg_ids(trace):
    if not os.path.exists(trace):
        return None
    for line in open(trace, encoding="utf-8", errors="replace"):
        if '"event":"tg_tokens"' in line.replace(" ", "") or '"event": "tg_tokens"' in line:
            try:
                return json.loads(line)["ids"]
            except Exception:
                return None
    return None


def attn_compute(trace):
    """median layer_compute (ms) of the attention layers (3, 7, ...) and of the recurrent layers in the LAST layer-major pass"""
    if not os.path.exists(trace):
        return None
    lc = []
    for line in open(trace, encoding="utf-8", errors="replace"):
        if '"layer_compute"' in line and '"phase": 0' in line or '"layer_compute"' in line and '"phase":0' in line:
            try:
                lc.append(json.loads(line))
            except Exception:
                pass
    if not lc:
        return None
    lc = lc[len(lc) // 2:]
    att = [d["compute_us"] / 1e3 for d in lc if d["layer"] % 4 == 3]
    rec = [d["compute_us"] / 1e3 for d in lc if d["layer"] % 4 != 3 and d["layer"] > 0]
    return (statistics.median(att) if att else None, statistics.median(rec) if rec else None)


def collect(session, arms):
    d = os.path.join(ROOT, session)
    out = {}
    for a in arms:
        out[a] = {}
        for p in sorted(glob.glob(os.path.join(d, "%s-%s_r*.log" % (session, a)))):
            r = int(re.search(r"_r(\d+)\.log$", p).group(1))
            v = prefill(read(p))
            rec = {"ms": v[0] if v else None, "tokens": v[1] if v else None, "log": p, "passes": passes(read(p))}
            tr = p[:-4] + ".jsonl"
            rec["ids"] = tg_ids(tr)
            rec["trace"] = tr
            out[a][r] = rec
    return out


def predicted(session, arm):
    p = os.path.join(ROOT, "specs", session + ".json")
    if not os.path.exists(p):
        return None
    spec = json.load(open(p))
    for a in spec["arms"]:
        if a["name"] == arm:
            m = re.search(r"predicted (\d+) ms", a.get("why", ""))
            return float(m.group(1)) if m else None
    return None


def fmt(v):
    return "%7.0f" % v if v is not None else "      -"


def hybrid(ns):
    arms = ("STK", "STKNO", "STKU", "LM2W", "HYC", "HY2", "HY2P", "HY4", "HY4P")
    refs = ("STK", "STKNO", "STKU", "LM2W")
    hys = ("HYC", "HY2", "HY2P", "HY4", "HY4P")
    for n in ns:
        session = "g50-%d" % n
        data = collect(session, arms)
        reps = sorted({r for a in arms for r in data[a]})
        if not reps:
            print("%s: no data yet" % session)
            continue
        tok = next((v["tokens"] for a in arms for v in data[a].values() if v["tokens"]), 0)
        print("== %s (%d prompt tokens), prefill wall ms per rep (last pass), reps %s" % (session, tok, reps))
        for a in arms:
            vals = [data[a][r]["ms"] if r in data[a] else None for r in reps]
            ok = [v for v in vals if v is not None]
            ids = {tuple(data[a][r]["ids"]) for r in data[a] if data[a][r]["ids"]}
            pred = predicted(session, a)
            print("  %-6s %s  median %s%s" % (a, " ".join(fmt(v) for v in vals), fmt(statistics.median(ok)) if ok else "      -",
                                             ("   (model %.0f)" % pred) if pred else ""))
        # C4: every rep of an interleave arm against the best reference of the same rep
        best_ref = {}
        for r in reps:
            v = [data[a][r]["ms"] for a in refs if r in data[a] and data[a][r]["ms"]]
            best_ref[r] = min(v) if v else None
        for a in hys:
            ratios = []
            for r in reps:
                if r in data[a] and data[a][r]["ms"] and best_ref[r]:
                    ratios.append(data[a][r]["ms"] / best_ref[r])
            if ratios:
                claim = len(ratios) == len(reps) and all(x < 0.97 for x in ratios)
                print("  C4 %-5s vs best reference per rep: %s -> %s" % (a, " ".join("%.3f" % x for x in ratios), "BEATS (all reps < 0.97)" if claim else "not demonstrated"))
        # C3: token identity with the all-GPU arm
        ref_ids = {tuple(data["LM2W"][r]["ids"]) for r in data["LM2W"] if data["LM2W"][r]["ids"]}
        for a in ("HYC", "HY2", "HY2P", "HY4", "HY4P"):
            ids = {tuple(data[a][r]["ids"]) for r in data[a] if data[a][r]["ids"]}
            if ids and ref_ids:
                print("  C3 %-5s tokens %s" % (a, "identical to LM2W" if ids == ref_ids else "DIFFER from LM2W: %s vs %s" % (sorted(ids), sorted(ref_ids))))
        print()


def stock(ns):
    pairs = (("STK", "SP1"), ("STKNO", "SP0"), ("STKU", "SPU"))
    for n in ns:
        session = "g56-stock-%d" % n
        data = collect(session, ("STK", "STKNO", "STKU", "SP1", "SP0", "SPU"))
        reps = sorted({r for a in data for r in data[a]})
        if not reps:
            print("%s: no data yet" % session)
            continue
        tok = next((v["tokens"] for a in data for v in data[a].values() if v["tokens"]), 0)
        print("== %s (%d prompt tokens), prefill wall ms per rep" % (session, tok))
        for ref, sp in pairs:
            rv = [data[ref][r]["ms"] if r in data[ref] else None for r in reps]
            sv = [data[sp][r]["ms"] if r in data[sp] else None for r in reps]
            dev = [(s / b - 1.0) * 100.0 for b, s in zip(rv, sv) if b and s]
            a2 = bool(dev) and all(-10.0 <= x <= 20.0 for x in dev)
            print("  %-5s %s | %-4s %s | deviation %s -> A2 %s" % (ref, " ".join(fmt(v) for v in rv), sp, " ".join(fmt(v) for v in sv),
                                                              " ".join("%+.1f%%" % x for x in dev), "holds" if a2 else "OUTSIDE +20%/-10%"))
        print()


def pace(ns):
    arms = ("LM40", "LM41", "BPA", "BPB")
    for n in ns:
        session = "g57-pace-%d" % n
        data = collect(session, arms)
        reps = sorted({r for a in data for r in data[a]})
        if not reps:
            print("%s: no data yet" % session)
            continue
        tok = next((v["tokens"] for a in data for v in data[a].values() if v["tokens"]), 0)
        print("== %s (%d prompt tokens), warm prefill wall ms per rep" % (session, tok))
        base = {r: data["LM41"][r]["ms"] for r in data["LM41"] if data["LM41"][r]["ms"]}
        ctl = {r: data["LM40"][r]["ms"] for r in data["LM40"] if data["LM40"][r]["ms"]}
        ids0 = {tuple(data["LM40"][r]["ids"]) for r in data["LM40"] if data["LM40"][r]["ids"]}
        for a in arms:
            vals = [data[a][r]["ms"] if r in data[a] else None for r in reps]
            ac = [attn_compute(data[a][r]["trace"]) for r in reps if r in data[a]]
            ac = [x for x in ac if x]
            ids = {tuple(data[a][r]["ids"]) for r in data[a] if data[a][r]["ids"]}
            extra = ""
            if ac:
                extra = " | attention layer compute %.1f ms, recurrent %.2f ms" % (statistics.median(x[0] for x in ac if x[0] is not None), statistics.median(x[1] for x in ac if x[1] is not None))
            print("  %-5s %s | tokens %s%s" % (a, " ".join(fmt(v) for v in vals), "identical to LM40" if ids == ids0 and ids else "DIFFER/none", extra))
        for a in ("BPA", "BPB"):
            dev = [(data[a][r]["ms"] / base[r] - 1.0) * 100.0 for r in reps if r in data[a] and r in base and data[a][r]["ms"]]
            if dev:
                print("  B3 %s vs LM41 per rep: %s -> %s" % (a, " ".join("%+.1f%%" % x for x in dev), "within +6%" if all(x <= 6.0 for x in dev) else "OVER +6%"))
        dev = [(base[r] / ctl[r] - 1.0) * 100.0 for r in reps if r in base and r in ctl]
        if dev:
            print("  B4 LM41 vs LM40 per rep: %s -> %s" % (" ".join("%+.1f%%" % x for x in dev), "within +-2%" if all(abs(x) <= 2.0 for x in dev) else "OUTSIDE +-2%"))
        print()


def final(ns):
    arms = ("STK", "STKNO", "STKU", "LM2W", "LM2P", "HYC", "HY4P", "HY4Q", "HY8Q")
    refs = ("STK", "STKNO", "STKU", "LM2W")
    news = ("LM2P", "HYC", "HY4P", "HY4Q", "HY8Q")
    for n in ns:
        session = "g68-%d" % n
        data = collect(session, arms)
        reps = sorted({r for a in arms for r in data[a]})
        if not reps:
            print("%s: no data yet" % session)
            continue
        tok = next((v["tokens"] for a in arms for v in data[a].values() if v["tokens"]), 0)
        print("== %s (%d prompt tokens), prefill wall ms per rep (last pass)" % (session, tok))
        for a in arms:
            if not data[a]:
                continue
            vals = [data[a][r]["ms"] if r in data[a] else None for r in reps]
            ok = [v for v in vals if v is not None]
            pred = predicted(session, a)
            print("  %-6s %s  median %s%s" % (a, " ".join(fmt(v) for v in vals), fmt(statistics.median(ok)) if ok else "      -", ("   (model %.0f)" % pred) if pred else ""))
        best_ref = {}
        for r in reps:
            v = [data[a][r]["ms"] for a in refs if r in data[a] and data[a][r]["ms"]]
            best_ref[r] = min(v) if v else None
        for a in news:
            ratios = [data[a][r]["ms"] / best_ref[r] for r in reps if r in data[a] and data[a][r]["ms"] and best_ref[r]]
            if ratios:
                claim = len(ratios) == len(reps) and all(x < 0.97 for x in ratios)
                pred = predicted(session, a)
                med = statistics.median(data[a][r]["ms"] for r in data[a] if data[a][r]["ms"])
                err = (" model error %+.1f%%" % ((pred / med - 1.0) * 100.0)) if pred else ""
                print("  C4 %-5s vs best reference per rep: %s -> %s%s" % (a, " ".join("%.3f" % x for x in ratios), "BEATS (all reps < 0.97)" if claim else "not demonstrated", err))
        ref_ids = {tuple(data["LM2W"][r]["ids"]) for r in data["LM2W"] if data["LM2W"][r]["ids"]}
        for a in news:
            ids = {tuple(data[a][r]["ids"]) for r in data[a] if data[a][r]["ids"]}
            if ids and ref_ids:
                print("  C3 %-5s tokens %s" % (a, "identical to LM2W" if ids == ref_ids else "DIFFER from LM2W: %s vs %s" % (sorted(ids), sorted(ref_ids))))
        print()


def ident(ns):
    for n in ns:
        session = "g69-ident-%d" % n
        data = collect(session, ("LM40", "LM44"))
        reps = sorted({r for a in data for r in data[a]})
        if not reps:
            print("%s: no data yet" % session)
            continue
        print("== %s, warm prefill wall ms per rep" % session)
        for a in ("LM40", "LM44"):
            print("  %-5s %s" % (a, " ".join(fmt(data[a][r]["ms"]) if r in data[a] else "      -" for r in reps)))
        dev = [(data["LM44"][r]["ms"] / data["LM40"][r]["ms"] - 1.0) * 100.0 for r in reps if r in data["LM44"] and r in data["LM40"] and data["LM44"][r]["ms"] and data["LM40"][r]["ms"]]
        ids0 = {tuple(data["LM40"][r]["ids"]) for r in data["LM40"] if data["LM40"][r]["ids"]}
        ids1 = {tuple(data["LM44"][r]["ids"]) for r in data["LM44"] if data["LM44"][r]["ids"]}
        print("  B4 LM44 vs LM40 per rep: %s -> %s; tokens %s" % (" ".join("%+.1f%%" % x for x in dev), "within +-2%" if dev and all(abs(x) <= 2.0 for x in dev) else "OUTSIDE +-2%", "identical" if ids0 == ids1 and ids0 else "DIFFER"))
        print("     median-based: %+.1f%%" % ((statistics.median(data["LM44"][r]["ms"] for r in data["LM44"]) / statistics.median(data["LM40"][r]["ms"] for r in data["LM40"]) - 1.0) * 100.0))
        print()


def deep(ns):
    for n in ns:
        session = "g70-pacedeep-%d" % n
        data = collect(session, ("LM44", "BPA", "BPB"))
        reps = sorted({r for a in data for r in data[a]})
        if not reps:
            print("%s: no data yet" % session)
            continue
        print("== %s, prefill wall ms per rep (last pass)" % session)
        for a in ("LM44", "BPA", "BPB"):
            vals = [data[a][r]["ms"] if r in data[a] else None for r in reps]
            ok = [v for v in vals if v is not None]
            print("  %-5s %s  median %s" % (a, " ".join(fmt(v) for v in vals), fmt(statistics.median(ok)) if ok else "      -"))
        for a in ("BPA", "BPB"):
            dev = [(data[a][r]["ms"] / data["LM44"][r]["ms"] - 1.0) * 100.0 for r in reps if r in data[a] and r in data["LM44"] and data[a][r]["ms"] and data["LM44"][r]["ms"]]
            if dev:
                print("  %s vs LM44 per rep: %s" % (a, " ".join("%+.1f%%" % x for x in dev)))
        print()


def keep(ns):
    for n in ns:
        for session, arms in (("g84-%d" % n, ("STKNO", "SP0X", "SP0K", "HYC", "HYCK")), ("g85-%d" % n, ("LM2P", "LM2PK", "HY4P", "HY4PK"))):
            data = collect(session, arms)
            reps = sorted({r for a in data for r in data[a]})
            if not reps:
                continue
            print("== %s: prefill wall ms per pass (pass 1 = cold process, last pass = warm), per rep" % session)
            for a in arms:
                if not data[a]:
                    continue
                cells = []
                for r in reps:
                    p = data[a][r]["passes"] if r in data[a] else []
                    cells.append("/".join("%.0f" % x for x in p) if p else (("%.0f" % data[a][r]["ms"]) if r in data[a] and data[a][r]["ms"] else "-"))
                print("  %-6s %s" % (a, "   ".join("rep%d %s" % (r, c) for r, c in zip(reps, cells))))
            print()


def main(argv):
    if not argv or argv[0] not in ("hybrid", "stock", "pace", "final", "ident", "deep", "keep"):
        print(__doc__)
        return 2
    kind, ns = argv[0], [int(x) for x in argv[1:]]
    if kind == "hybrid":
        hybrid(ns or [32, 64, 128, 256, 512, 1024, 2048])
    elif kind == "stock":
        stock(ns or [3, 10, 14, 29, 63, 125, 256, 511, 1023, 2047])
    elif kind == "final":
        final(ns or [29, 63, 125, 256, 511, 1023, 2047])
    elif kind == "ident":
        ident(ns or [63, 256, 1023])
    elif kind == "deep":
        deep(ns or [4096, 24000, 50000])
    elif kind == "keep":
        keep(ns or [63, 256, 511, 1023])
    else:
        pace(ns or [63, 256, 1023])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
