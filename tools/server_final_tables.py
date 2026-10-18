#!/usr/bin/env python3
"""Print the G31 acceptance tables (markdown) from the recorded sessions of the final matrix (tags g331..g344 plus the STOCK references g267 / g233 / g311).

  python tools/server_final_tables.py [--root build/llama-cuda]

Nothing is computed from memory: every number is read from steps.jsonl / report.json / soak.json / server.log of its session.
"""
import argparse
import json
import os
import re
import statistics

ROOT = "build/llama-cuda"


def steps(tag):
    p = os.path.join(ROOT, tag, "steps.jsonl")
    return {s["name"]: s for s in (json.loads(l) for l in open(p, "r", encoding="utf-8") if l.strip())} if os.path.exists(p) else {}


def report(tag):
    p = os.path.join(ROOT, tag, "report.json")
    return json.load(open(p, "r", encoding="utf-8")) if os.path.exists(p) else None


def rows(rep):
    return {(r["scenario"], r["name"]): r for r in rep["rows"] if not r.get("check")} if rep else {}


def log_text(tag):
    p = os.path.join(ROOT, tag, "server.log")
    return open(p, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace") if os.path.exists(p) else ""


def watcher_min(tag):
    w = [int(x) for x in re.findall(r"free-watch: (?:new minimum of free device memory|the engine is gone; lowest free device memory) (\d+) MiB", log_text(tag))]
    return min(w) if w else None


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    ROOT = a.root

    print("### SRV-1 default-off identity (A = stock s0, B = the final build without the new flags; A B B A)")
    print("| step | A1 PP / TG | B1 PP / TG | B2 PP / TG | A2 PP / TG | texts equal | counts equal |")
    print("|---|---|---|---|---|---|---|")
    S = [steps(t) for t in ("g501-off-A1", "g502-off-B1", "g503-off-B2", "g504-off-A2")]
    for name in ("p4k-cold", "p4k-warm", "p4k-ext"):
        if not all(name in s for s in S):
            continue
        cells = []
        for s in S:
            t = s[name]["timings"]
            cells.append("%.1f / %.1f" % (t.get("prompt_per_second") or 0, t.get("predicted_per_second") or 0))
        texts = len(set(s[name]["text"] for s in S)) == 1
        counts = len(set((s[name]["timings"].get("prompt_n"), s[name]["timings"].get("cache_n"), s[name]["timings"].get("predicted_n")) for s in S)) == 1
        print("| %s | %s | %s | %s |" % (name, " | ".join(cells), "yes" if texts else "NO", "yes" if counts else "NO"))
    print()

    print("### SRV-2 / SRV-7 cells (greedy, 128 tokens): STOCK sessions g505 / g508, OPT sessions g506 / g507")
    print("| cell | STOCK TG | OPT TG | TG ratio | STOCK PP | OPT PP | PP ratio |")
    print("|---|---|---|---|---|---|---|")
    st = [steps("g505-cells-S1"), steps("g508-cells-S2")]
    op = [steps("g506-cells-O1"), steps("g507-cells-O2")]
    for name in ("p4k-cold", "p4k-warm", "p24k-cold", "p24k-warm", "p100k-cold", "p100k-warm"):
        if not all(name in s for s in st + op):
            continue
        stg = [s[name]["timings"]["predicted_per_second"] for s in st]
        otg = [s[name]["timings"]["predicted_per_second"] for s in op]
        spp = [s[name]["timings"]["prompt_per_second"] for s in st]
        opp = [s[name]["timings"]["prompt_per_second"] for s in op]
        ratios = [o / statistics.mean(stg) for o in otg]
        pp = ""
        if "cold" in name:
            pr = [o / statistics.mean(spp) for o in opp]
            pp = "%.1f | %.1f | %.2fx-%.2fx" % (statistics.mean(spp), statistics.mean(opp), min(pr), max(pr))
        else:
            pp = "- | - | -"
        print("| %s | %.2f / %.2f | %.2f / %.2f | %.2fx / %.2fx | %s |" % (name, stg[0], stg[1], otg[0], otg[1], ratios[0], ratios[1], pp))
    print()

    print("### Multi-turn (fixtures A, B) OPT g509 against STOCK g267: prompt tokens processed per request and TG")
    rs, ro = rows(report("g267-stock512")), rows(report("g509-opt-mt"))
    worse = []
    n_same = 0
    n_less = 0
    ratios = []
    for k, o in ro.items():
        s = rs.get(k)
        if not s or o.get("prompt_n") is None or s.get("prompt_n") is None:
            continue
        if k[0] in ("prefix", "agentic", "branch", "alternate", "sampling"):
            if o["prompt_n"] == s["prompt_n"]:
                n_same += 1
            elif o["prompt_n"] < s["prompt_n"]:
                n_less += 1
            elif o["prompt_n"] > s["prompt_n"] + 8:
                worse.append((k, s["prompt_n"], o["prompt_n"]))
        if o.get("tps") and s.get("tps") and (o.get("predicted_n") or 0) >= 16 and (s.get("predicted_n") or 0) >= 16:
            ratios.append(o["tps"] / s["tps"])
    print("requests with the same prompt_n: %d; OPT processed fewer: %d; OPT processed more than STOCK + 8: %s" % (n_same, n_less, worse if worse else "none"))
    if ratios:
        print("TG ratio over %d requests: min %.2fx, median %.2fx, max %.2fx" % (len(ratios), min(ratios), statistics.median(ratios), max(ratios)))
    for tag in ("g509-opt-mt",):
        rep = report(tag)
        if rep:
            print("expectations failed in %s: %s; server alive: %s; watcher min %s MiB; nvidia-smi min free %s MiB" % (tag, rep.get("fails"), rep.get("alive"), watcher_min(tag), rep.get("free_min_smi_mib")))
    print()

    print("### Growth to 138k tokens (fixture G): OPT g511 against STOCK g233")
    rs, ro = rows(report("g233-stock-growth")), rows(report("g511-opt-growth"))
    print("| request | ctx tokens | STOCK PP / TG | OPT PP / TG | prompt_n S = O |")
    print("|---|---|---|---|---|")
    for k, o in ro.items():
        s = rs.get(k)
        if not s:
            continue
        ctx = (o.get("prompt_n") or 0) + (o.get("cache_n") or 0)
        print("| %s | %d | %.0f / %.1f | %.0f / %.1f | %s |" % (k[1], ctx, s.get("prompt_per_second") or 0, s.get("tps") or 0, o.get("prompt_per_second") or 0, o.get("tps") or 0, "yes" if o.get("prompt_n") == s.get("prompt_n") else "NO (%s vs %s)" % (s.get("prompt_n"), o.get("prompt_n"))))
    rep = report("g511-opt-growth")
    if rep:
        print("expectations failed: %s; watcher min %s MiB" % (rep.get("fails"), watcher_min("g511-opt-growth")))
    print()

    print("### Soaks (SRV-6)")
    for tag in ("g513-soak-np1", "g514-soak-np2"):
        p = os.path.join(ROOT, tag, "soak.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p, "r", encoding="utf-8"))
        s = d["summary"]
        t = log_text(tag)
        kinds = {}
        for r in d["results"]:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print("- %s: %d requests %s, census %s, failed %d, server alive %s, wall %.0f s, working set %s MiB (start / after probe / end), idle free VRAM (nvidia-smi) %s MiB, watcher min %s MiB, log hits %s, prefill windows %d, cache shrinks %d"
              % (tag, s["requests"], kinds, s["census"], s["failed"], s["alive"], s["wall_s"], s["rss_mib"], s["smi_free_mib"], s["watcher_min_mib"], s["log_hits"] or "none", t.count("prefill window #"), t.count("short by")))
    print()

    print("### SRV-5 with the expert cache frozen (LLAMA_MOE_HYBRID_B=0): the pairs of g515 / g516 (console lines), the cross-server check against STOCK g267 is run by tools/server_mt_xcheck.py")
    for out in ("g515", "g516", "g509"):
        p = os.path.join(ROOT, out + ".out")
        if not os.path.exists(p):
            continue
        print("- %s:" % out)
        for line in open(p, "r", encoding="utf-8", errors="replace"):
            if re.match(r"^(equiv|branch|alternate) +(same-|rollback-vs|restore-vs)", line):
                print("    " + line.strip()[:200])
    print()

    print("### Shrink stress (LLAMA_HYBRID_SLACK_MIB=0)")
    for tag in ("g512-stress1", "g512-stress2"):
        rep = report(tag)
        if rep:
            t = log_text(tag)
            print("- %s: shrink events %d, windows %d, failed expectations %s, server alive %s" % (tag, t.count("short by"), t.count("prefill window #"), rep.get("fails"), rep.get("alive")))


if __name__ == "__main__":
    main()
