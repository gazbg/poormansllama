#!/usr/bin/env python3
"""G24 tables: the final TG verification under the strict (whole-run) reading of the keep-free rule.

    python tools/g24-table.py speed g87-short-B g87-n24k ...      per-run table + per-session summary + criteria G24-1 / G24-2
    python tools/g24-table.py qa g87-qab24000 g87-qab50000 ...    answer-stability sessions + G24-1
    python tools/g24-table.py doc                                 the owner's llama-server baseline reproduction (g87-doc)
    python tools/g24-table.py md [sessions]                       markdown rows (one per session and hybrid arm) for the plan
    python tools/g24-table.py all                                 every session of the G24 chain that has a verdict

Host-only. Reads build/llama-cuda/<session>/<session>-verdict.json and the per-run logs next to it. The criteria are the ones written in the plan's G24 CONTRACT:
  G24-1  every hybrid run: in-process watcher minimum free >= 16,000 MiB AND poller peak (vram_peak_raw_mib, whole process life) <= 7,370 MiB
  G24-2  every hybrid run: TG > 30.68 tok/s (the document) AND faster than its own pair's stock run (same rep)
  G24-3  QA sessions: positive grade PASS and negative control fails as designed
Nothing here moves a bar: the numbers 16,000 / 7,370 / 30.68 are constants of the contract.
"""
import json
import os
import re
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "llama-cuda"))
FREE_BAR_MIB = 16000.0
POLLER_BAR_MIB = 7370.0
DOC_TG = 30.68
SPEED = ["g87-short-B", "g87-short-A", "g87-n24k", "g87-n50k", "g87-n100k", "g87-n200k", "g87-n250k"]
QA = ["g87-qab24000", "g87-qab50000", "g87-qab100000"]


def read(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if b"\x00" in raw[:4000]:
        return raw.replace(b"\x00", b"").decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


def last(pat, text):
    m = None
    for m in re.finditer(pat, text):
        pass
    return m.groups() if m else None


def parse_log(path):
    if not path or not os.path.exists(path):
        return {}
    t = read(path)
    d = {}
    g = last(r"PP: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\)", t)
    if g:
        d["pp_n"], d["pp_ms"], d["pp_rate"] = int(g[0]), float(g[1]), float(g[2])
    g = last(r"TG: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\)", t)
    if g:
        d["tg_n"], d["tg_ms"], d["tg_rate"] = int(g[0]), float(g[1]), float(g[2])
    g = last(r"auto slots:.*-> (\d+) slots", t)
    if g:
        d["slots"] = int(g[0])
    g = last(r"free-watch: minimum free device memory ([\d.]+) MiB", t)
    if g:
        d["free_min"] = float(g[0])
    g = last(r"free-watch \(decode phase[^)]*\): minimum free device memory ([\d.]+) MiB", t)
    if g:
        d["free_tg"] = float(g[0])
    g = last(r"moe_hybrid: tokens (\d+), hit rate ([\d.]+) .*?promotions/token ([\d.]+)", t)
    if g:
        d["hit"] = float(g[1])
    g = last(r"timeline per token over (\d+) tokens \(us\): span ([\d.]+) = attn ([\d.]+) \+ ovl ([\d.]+) \+ stall ([\d.]+), gap ([\d.]+)", t)
    if g:
        d["span"], d["attn"] = float(g[1]), float(g[2])
    g = last(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens .*?([\d.]+) tokens per second", t)
    if g:
        d["spp_ms"], d["spp_n"], d["spp_rate"] = float(g[0]), int(g[1]), float(g[2])
    g = last(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) runs .*?([\d.]+) tokens per second", t)
    if g:
        d["stg_ms"], d["stg_n"], d["stg_rate"] = float(g[0]), int(g[1]), float(g[2])
    return d


def load(session):
    p = os.path.join(ROOT, session, session + "-verdict.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def fmt(x, nd=2):
    return "-" if x is None else ("%.*f" % (nd, x))


def runs_of(v):
    out = []
    for key, rec in v["arms"].items():
        base = rec.get("base") or key.split("#")[0]
        rep = rec.get("rep", 1)
        eng = rec.get("engine")
        log = parse_log(rec.get("log"))
        if eng == "stock":
            tg = rec.get("stock_tok_s") or log.get("stg_rate")
            pp_rate, pp_s = log.get("spp_rate"), (log.get("spp_ms") or 0) / 1000.0 if log.get("spp_ms") else None
        else:
            tg = rec.get("summary", {}).get("tg_tokens_per_s")
            pp_rate, pp_s = log.get("pp_rate"), (log.get("pp_ms") or 0) / 1000.0 if log.get("pp_ms") else None
        out.append({"key": key, "arm": base, "rep": rep, "stock": eng == "stock", "tg": tg, "rc": rec.get("rc"), "pp_rate": pp_rate, "pp_s": pp_s,
                    "slots": log.get("slots"), "hit": log.get("hit"), "free_min": log.get("free_min"), "free_tg": log.get("free_tg"),
                    "poller": rec.get("vram_peak_raw_mib"), "attn": log.get("attn"), "span": log.get("span"), "wall": rec.get("wall_s"),
                    "tg_n": log.get("tg_n") or log.get("stg_n"), "qa": rec.get("qa"), "ratio6": (rec.get("summary") or {}).get("rep6", {}).get("ratio")})
    return out


def speed_session(session):
    v = load(session)
    if v is None:
        print("%s: no verdict yet" % session)
        return None
    runs = runs_of(v)
    prov = v.get("provenance") or {}
    print("\n== %s  (power %s, exe %s, prompt %s)" % (session, (v.get("power") or {}).get("ac_line"), prov.get("exe_sha256_16"), prov.get("prompt_sha256_16")))
    print("%-7s %3s %8s %7s %8s %6s %6s %8s %8s %8s %8s %6s" % ("arm", "rep", "TG tok/s", "slots", "PP tok/s", "PP s", "hit", "freeMin", "freeTG", "poller", "attn us", "6gram"))
    for r in sorted(runs, key=lambda x: (x["rep"], x["arm"])):
        print("%-7s %3s %8s %7s %8s %6s %6s %8s %8s %8s %8s %6s  rc=%s" % (
            r["arm"], r["rep"], fmt(r["tg"]), r["slots"] if r["slots"] is not None else "-", fmt(r["pp_rate"], 0), fmt(r["pp_s"], 1), fmt(r["hit"], 4),
            fmt(r["free_min"], 1), fmt(r["free_tg"], 1), r["poller"] if r["poller"] is not None else "-", fmt(r["attn"], 0), fmt(r["ratio6"], 3), r["rc"]))
    stk = {r["rep"]: r for r in runs if r["stock"]}
    stk_mean = mean([r["tg"] for r in stk.values()])
    verdicts = {}
    print("-- per arm: mean (min-max), x STK mean, x %.2f (document); pairs faster than their stock run; G24-1 / G24-2" % DOC_TG)
    for arm in sorted({r["arm"] for r in runs}):
        rs = [r for r in runs if r["arm"] == arm]
        tgs = [r["tg"] for r in rs if r["tg"] is not None]
        if not tgs:
            print("%-5s no TG" % arm)
            continue
        m = statistics.mean(tgs)
        line = "%-5s n=%d  %.2f (%.2f-%.2f)" % (arm, len(tgs), m, min(tgs), max(tgs))
        if arm != "STK":
            ratio = (m / stk_mean) if stk_mean else None
            pairs = [(r["tg"] > stk[r["rep"]]["tg"]) for r in rs if r["rep"] in stk and r["tg"] and stk[r["rep"]]["tg"]]
            g2 = all(t > DOC_TG for t in tgs) and len(pairs) == len(tgs) and all(pairs)
            frees = [r["free_min"] for r in rs]
            pol = [r["poller"] for r in rs]
            g1 = all(f is not None and f >= FREE_BAR_MIB for f in frees) and all(p is not None and p <= POLLER_BAR_MIB for p in pol)
            line += "  x STK %s  x doc %s  pairs %d/%d  slots %s  hit %s  minFree %s  poller %s  G24-1 %s  G24-2 %s" % (
                fmt(ratio, 2) + "x" if ratio else "-", fmt(m / DOC_TG, 2) + "x", sum(pairs), len(rs), sorted({r["slots"] for r in rs if r["slots"]}),
                fmt(mean([r["hit"] for r in rs]), 4), fmt(min([f for f in frees if f is not None] or [None]) if any(f is not None for f in frees) else None, 1),
                max([p for p in pol if p is not None] or [0]), "PASS" if g1 else "FAIL", "PASS" if g2 else "FAIL")
            verdicts[arm] = (g1, g2)
        else:
            line += "  PP %s tok/s" % fmt(mean([r["pp_rate"] for r in rs]), 0)
        print(line)
    return verdicts


def qa_session(session):
    v = load(session)
    if v is None:
        print("%s: no verdict yet" % session)
        return
    print("\n== %s" % session)
    for r in runs_of(v):
        qa = r["qa"] if isinstance(r["qa"], dict) else {}
        g1 = r["free_min"] is not None and r["free_min"] >= FREE_BAR_MIB and r["poller"] is not None and r["poller"] <= POLLER_BAR_MIB
        print("%-7s TG %s tok/s  PP %s tok/s (%s s)  slots %s  freeMin %s freeTG %s poller %s  positive_pass=%s negative_fails_as_designed=%s  rc=%s  G24-1 %s  G24-3 %s" % (
            r["arm"], fmt(r["tg"]), fmt(r["pp_rate"], 0), fmt(r["pp_s"], 1), r["slots"], fmt(r["free_min"], 1), fmt(r["free_tg"], 1), r["poller"],
            qa.get("positive_pass"), qa.get("negative_fails_as_designed"), r["rc"], "PASS" if g1 else "FAIL",
            "PASS" if (qa.get("positive_pass") and qa.get("negative_fails_as_designed") and r["rc"] == 0) else "FAIL"))


def md_summary(names):
    """One markdown row per (session, hybrid arm): the table the plan's G24 RESULTS block carries."""
    rows = []
    print("| session | prompt | arm | n | TG tok/s mean (min-max) | x STK | x 30.68 | x server | slots | hit | PP tok/s (s) | STK PP tok/s (s) | min free MiB (watcher) | poller peak MiB | G24-1 | G24-2 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    srv = None
    sp = os.path.join(ROOT, "g87-doc", "summary.json")
    if os.path.exists(sp):
        reqs = {r["label"]: r["timings"].get("predicted_per_second") for r in json.load(open(sp, encoding="utf-8"))["requests"]}
        srv = reqs.get("sampled-1")
    for session in names:
        v = load(session)
        if v is None:
            continue
        runs = runs_of(v)
        stk = {r["rep"]: r for r in runs if r["stock"]}
        stk_mean = mean([r["tg"] for r in stk.values()])
        stk_pp = mean([r["pp_rate"] for r in stk.values()])
        stk_pps = mean([r["pp_s"] for r in stk.values()])
        prompt = os.path.basename(((v.get("spec") or {}) if isinstance(v.get("spec"), dict) else {}).get("prompt", "") or session)
        for arm in sorted({r["arm"] for r in runs if not r["stock"]}):
            rs = [r for r in runs if r["arm"] == arm]
            tgs = [r["tg"] for r in rs if r["tg"]]
            if not tgs:
                continue
            m = statistics.mean(tgs)
            pairs = [(r["tg"] > stk[r["rep"]]["tg"]) for r in rs if r["rep"] in stk and r["tg"] and stk[r["rep"]]["tg"]]
            g2 = all(t > DOC_TG for t in tgs) and len(pairs) == len(tgs) and all(pairs)
            frees = [r["free_min"] for r in rs]
            pol = [r["poller"] for r in rs]
            g1 = all(f is not None and f >= FREE_BAR_MIB for f in frees) and all(p is not None and p <= POLLER_BAR_MIB for p in pol)
            print("| %s | %s | %s | %d | %.2f (%.2f-%.2f) | %.2fx | %.2fx | %s | %s | %s | %.0f (%.1f) | %.0f (%.1f) | %s | %s | %s | %s |" % (
                session, prompt, arm, len(tgs), m, min(tgs), max(tgs), (m / stk_mean) if stk_mean else 0, m / DOC_TG, ("%.2fx" % (m / srv)) if srv else "-",
                "/".join(str(x) for x in sorted({r["slots"] for r in rs if r["slots"]})), fmt(mean([r["hit"] for r in rs]), 4), mean([r["pp_rate"] for r in rs]) or 0, mean([r["pp_s"] for r in rs]) or 0,
                stk_pp or 0, stk_pps or 0, fmt(min([f for f in frees if f is not None] or [None]), 0) if any(f is not None for f in frees) else "-", max([p for p in pol if p is not None] or [0]),
                "PASS" if g1 else "FAIL", "PASS" if g2 else "FAIL"))


def doc_session():
    p = os.path.join(ROOT, "g87-doc", "summary.json")
    if not os.path.exists(p):
        print("g87-doc: no summary yet")
        return
    s = json.load(open(p, encoding="utf-8"))
    print("\n== g87-doc  (the owner's llama-server, build of %s, ready after %s s)" % (s.get("server_sha256_16"), s.get("ready_s")))
    for r in s["requests"]:
        t = r["timings"]
        print("%-10s prompt %s tok  TG %s tok in %s ms = %s tok/s  (wall %s s)" % (r["label"], t.get("prompt_n"), t.get("predicted_n"), fmt(t.get("predicted_ms"), 0), fmt(t.get("predicted_per_second")), r["wall_s"]))
    smi = os.path.join(ROOT, "g87-doc", "smi.csv")
    if os.path.exists(smi):
        used = [int(l.split(",")[1]) for l in open(smi).read().splitlines()[1:] if l.count(",") == 2]
        if used:
            print("nvidia-smi used MiB: max %d (poller unit), samples %d; free-in-runtime-view equivalent = 24462 - max - 1086 = %d MiB" % (max(used), len(used), 24462 - max(used) - 1086))
    log = os.path.join(ROOT, "g87-doc", "server.log")
    if os.path.exists(log):
        t = read(log)
        for pat in (r"common_params_fit_impl: (id=0, n_layer=\d+.*)", r"offloaded (\d+/\d+) layers", r"llama_kv_cache: (size = [^\n]*)"):
            g = last(pat, t)
            if g:
                print("log:", g[0][:200])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    names = sys.argv[2:]
    if mode == "speed":
        for s in names:
            speed_session(s)
    elif mode == "qa":
        for s in names:
            qa_session(s)
    elif mode == "md":
        md_summary(names or SPEED)
    elif mode == "doc":
        doc_session()
    elif mode == "all":
        doc_session()
        for s in QA[:2]:
            qa_session(s)
        for s in SPEED:
            speed_session(s)
        qa_session(QA[2])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
