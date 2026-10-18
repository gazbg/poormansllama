#!/usr/bin/env python3
"""Print the G32 acceptance tables (markdown) from the recorded sessions (tags g601 .. g738, STOCK / WINDOW references g267 / g509 / g511 / g513 / g514 / g614).

  python tools/server_lm_final_tables.py [--root build/llama-cuda]

Nothing is computed from memory: every number is read from steps.jsonl / report.json / soak.json / server.log / firsttok.json of its session.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys

ROOT = "build/llama-cuda"
HERE = os.path.dirname(os.path.abspath(__file__))


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


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    return (r.stdout + r.stderr).rstrip()


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    ROOT = a.root

    print("### LM-1 default-off identity (A = stock s0, B = the final build without the new flags; A B B A, g672-g675)")
    print("| step | A1 PP / TG | B1 PP / TG | B2 PP / TG | A2 PP / TG | texts equal | counts equal |")
    print("|---|---|---|---|---|---|---|")
    S = [steps(t) for t in ("g672-off-A1", "g673-off-B1", "g674-off-B2", "g675-off-A2")]
    for name in ("p4k-cold", "p4k-warm", "p4k-ext"):
        if not all(name in s for s in S):
            continue
        cells = ["%.1f / %.1f" % (s[name]["timings"].get("prompt_per_second") or 0, s[name]["timings"].get("predicted_per_second") or 0) for s in S]
        texts = len(set(s[name]["text"] for s in S)) == 1
        counts = len(set((s[name]["timings"].get("prompt_n"), s[name]["timings"].get("cache_n"), s[name]["timings"].get("predicted_n")) for s in S)) == 1
        print("| %s | %s | %s | %s |" % (name, " | ".join(cells), "yes" if texts else "NO", "yes" if counts else "NO"))
    print()

    print("### LM-2 cold-prompt prefill, final build (WINDOW = s20 --hybrid-moe, LM = s24 --hybrid-moe --lm-prefill; g733 g736 g734 g735; A B B A) and STOCK (g614)")
    print(run([os.path.join(HERE, "server_lm_table.py"), "--arm", "WIN-A=%s/g733-win-a" % ROOT, "--arm", "WIN-B=%s/g736-win-b" % ROOT, "--arm", "LM-A=%s/g734-lm-a" % ROOT, "--arm", "LM-B=%s/g735-lm-b" % ROOT,
                "--arm", "STOCK=%s/g614-stock" % ROOT, "--base", "WIN-A"]).split("\n\nLM-A")[0])
    print()

    print("### --lm-prefill-min sweep (g630: cold prompts, 3 repetitions, prefill seconds; the minimum decides which batches are taken)")
    print("| min | 1,024 tokens | 2,047 tokens | 4,093 tokens |")
    print("|---|---|---|---|")
    for m in (2048, 1024, 512, 256):
        st = steps("g630-min%d" % m)
        cells = []
        for k in ("p1k", "p2k", "p4k"):
            v = [st[k + "-" + r]["timings"]["prompt_ms"] / 1000.0 for r in "abc" if k + "-" + r in st]
            cells.append("%.2f (%.2f-%.2f)" % (statistics.mean(v), min(v), max(v)) if v else "-")
        print("| %d | %s |" % (m, " | ".join(cells)))
    print()

    print("### Checkpoint chunk (g640, --lm-prefill-ckpt-chunk 24, build s23) against the stock rule (g620, s22)")
    for t in ("g620-lm-smoke2", "g640-lm-ckpt24"):
        st = steps(t)
        print("- %s: 4k cold %.0f t/s, 24k cold %.0f t/s, 4k again %.0f t/s" % (t, st["p4k-a"]["timings"]["prompt_per_second"], st["p24k-a"]["timings"]["prompt_per_second"], st["p4k-b"]["timings"]["prompt_per_second"]))
    print()

    print("### Multi-turn per request: STOCK g267 / WINDOW g509 / LM g737 (prompt tokens processed, PP t/s, TG t/s)")
    rs, rw, rl = rows(report("g267-stock512")), rows(report("g509-opt-mt")), rows(report("g737-lm-mt"))
    print("| request | prompt_n S / W / L | PP S | PP W | PP L | TG S | TG W | TG L |")
    print("|---|---|---|---|---|---|---|---|")
    n_same = n_less = 0
    worse = []
    ratios = []
    for k, l in rl.items():
        s, w = rs.get(k), rw.get(k)
        if not s or not w or l.get("prompt_n") is None:
            continue
        if k[0] not in ("prefix", "agentic", "branch", "alternate", "sampling", "equiv", "xcheck"):
            continue
        f = lambda r, x: ("%.0f" % r[x]) if r.get(x) is not None else "-"
        print("| %s/%s | %s / %s / %s | %s | %s | %s | %s | %s | %s |" % (k[0], k[1], s.get("prompt_n"), w.get("prompt_n"), l.get("prompt_n"), f(s, "prompt_per_second"), f(w, "prompt_per_second"), f(l, "prompt_per_second"), f(s, "tps"), f(w, "tps"), f(l, "tps")))
        if k[0] in ("prefix", "agentic", "branch", "alternate", "sampling"):
            if l["prompt_n"] == s["prompt_n"]:
                n_same += 1
            elif l["prompt_n"] < s["prompt_n"]:
                n_less += 1
            elif l["prompt_n"] > s["prompt_n"] + 8:
                worse.append((k, s["prompt_n"], l["prompt_n"]))
        if l.get("tps") and s.get("tps") and (l.get("predicted_n") or 0) >= 16 and (s.get("predicted_n") or 0) >= 16:
            ratios.append(l["tps"] / s["tps"])
    print("\nrequests with the same prompt_n as STOCK: %d; LM processed fewer: %d; LM processed more than STOCK + 8: %s" % (n_same, n_less, worse if worse else "none"))
    if ratios:
        print("TG ratio LM / STOCK over %d requests: min %.2fx, median %.2fx, max %.2fx" % (len(ratios), min(ratios), statistics.median(ratios), max(ratios)))
    rep = report("g737-lm-mt")
    if rep:
        print("expectations failed in g737: %s; server alive %s; watcher min %s MiB; nvidia-smi min free %s MiB" % (rep.get("fails"), rep.get("alive"), watcher_min("g737-lm-mt"), rep.get("free_min_smi_mib")))
    print()

    print("### Fresh continuations (xcheck): LM g737 against STOCK g267 and WINDOW g509")
    for ref in ("g267-stock512", "g509-opt-mt"):
        print("- ref %s:" % ref)
        print("\n".join("    " + l for l in run([os.path.join(HERE, "server_mt_xcheck.py"), "--ref", "%s/%s/report.json" % (ROOT, ref), "--cand", "%s/g737-lm-mt/report.json" % ROOT]).split("\n")))
    print()

    print("### Equivalence with the expert cache frozen (--hybrid-moe-slots 16 + LLAMA_MOE_HYBRID_B=0): WINDOW g660, LM g661, LM + 24-token checkpoint chunk g662")
    for t, f in (("g660", "g660-frozen-win"), ("g661", "g661-frozen-lm"), ("g662", "g662-frozen-lm-ck24")):
        p = os.path.join(ROOT, t + ".out")
        print("- %s (%s):" % (t, f))
        if os.path.exists(p):
            for line in open(p, "r", encoding="utf-8", errors="replace"):
                if re.match(r"^(equiv|branch|alternate) +(same-|rollback-vs|restore-vs)", line):
                    print("    " + line.strip()[:190])
    print("- fresh continuations, WINDOW g660 against LM g661:")
    print("\n".join("    " + l for l in run([os.path.join(HERE, "server_mt_xcheck.py"), "--ref", "%s/g660-frozen-win/report.json" % ROOT, "--cand", "%s/g661-frozen-lm/report.json" % ROOT]).split("\n")))
    print()

    print("### First generated token from the prefill alone (--ctx-checkpoints 0; 3 repetitions): determinism and distance between the prefill paths")
    for t in ("g690s-ft-win", "g691s-ft-lm", "g692s-ft-lm2", "g693s-ft-stock", "g694-ft-seg4096", "g695-ft-slots1", "g696-ft-chunk1024", "g697-ft-bf16"):
        p = os.path.join(ROOT, t, "firsttok.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        worst = 0.0
        for k, runs in d.items():
            base = dict((x, lp) for x, lp in runs[0]["top"])
            for r in runs[1:]:
                oth = dict((x, lp) for x, lp in r["top"])
                sh = [x for x in base if x in oth and base[x] is not None and oth[x] is not None]
                worst = max([worst] + [abs(base[x] - oth[x]) for x in sh])
        print("- %s: %d prompts, repetition-to-repetition max |dlogprob| over the top-10 = %.4f" % (t, len(d), worst))
    for a_, b_, nm in (("g690s-ft-win", "g691s-ft-lm", "WINDOW vs LM"), ("g690s-ft-win", "g693s-ft-stock", "WINDOW vs STOCK"), ("g691s-ft-lm", "g693s-ft-stock", "LM vs STOCK"),
                       ("g691s-ft-lm", "g696-ft-chunk1024", "LM vs LM chunk 1024"), ("g691s-ft-lm", "g697-ft-bf16", "LM vs LM bf16 slab"), ("g691s-ft-lm", "g694-ft-seg4096", "LM vs LM segment 4096"),
                       ("g691s-ft-lm", "g695-ft-slots1", "LM vs LM 1 slot")):
        print("- %s:" % nm)
        print("\n".join("    " + l for l in run([os.path.join(HERE, "server_firsttok_cmp.py"), "%s/%s/firsttok.json" % (ROOT, a_), "%s/%s/firsttok.json" % (ROOT, b_)]).split("\n")))
    print()

    print("### Stock decode with the layer-major prefill (-ngl 99 -ncmoe 37 -fit off; g670 STOCK s0, g671 LM s23) and the TG attribution runs (smoke scenario, 128 generated tokens)")
    print(run([os.path.join(HERE, "server_lm_table.py"), "--arm", "STOCK37=%s/g670-s37-stock" % ROOT, "--arm", "LM37=%s/g671-s37-lm" % ROOT]).split("\n\nLM37")[0])
    for t in ("g695-s0-n37", "g696-s0-n37-mlock", "g697-s23-n37", "g698-s23-n37-lm"):
        st = steps(t)
        if "p4k-cold" in st:
            print("- %s: 4k cold PP %.0f t/s, TG cold %.2f / warm %.2f t/s" % (t, st["p4k-cold"]["timings"]["prompt_per_second"], st["p4k-cold"]["timings"]["predicted_per_second"], st["p4k-warm"]["timings"]["predicted_per_second"]))
    print()

    print("### Growth to 138k tokens (fixture G): WINDOW g511 against LM g700")
    rw, rl = rows(report("g511-opt-growth")), rows(report("g700-lm-growth"))
    print("| request | ctx tokens | prompt_n W / L | PP W | PP L | TG W | TG L |")
    print("|---|---|---|---|---|---|---|")
    for k, l in rl.items():
        w = rw.get(k)
        if not w:
            continue
        ctx = (l.get("prompt_n") or 0) + (l.get("cache_n") or 0)
        print("| %s | %d | %s / %s | %.0f | %.0f | %.1f | %.1f |" % (k[1], ctx, w.get("prompt_n"), l.get("prompt_n"), w.get("prompt_per_second") or 0, l.get("prompt_per_second") or 0, w.get("tps") or 0, l.get("tps") or 0))
    rep = report("g700-lm-growth")
    print("expectations failed: %s; watcher min %s MiB; nvidia-smi min free %s MiB" % (rep.get("fails"), watcher_min("g700-lm-growth"), rep.get("free_min_smi_mib")))
    print()

    print("### Soaks")
    for tag in ("g513-soak-np1", "g710-lm-soak-np1", "g514-soak-np2", "g711-lm-soak-np2"):
        p = os.path.join(ROOT, tag, "soak.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p, "r", encoding="utf-8"))["summary"]
        t = log_text(tag)
        print("- %s: %d requests, census %s, failed %d, server alive %s, wall %.0f s, working set %s MiB, idle free VRAM (nvidia-smi) %s MiB, watcher min %s MiB, log hits %s, layer-major passes %d, prefill windows %d"
              % (tag, s["requests"], s["census"], s["failed"], s["alive"], s["wall_s"], s["rss_mib"], s["smi_free_mib"], s["watcher_min_mib"], s["log_hits"] or "none", t.count("lm-prefill: pass"), t.count("prefill window #")))
    for tag in ("g701-lm-stress1", "g701-lm-stress2", "g702-lm-slots"):
        rep = report(tag)
        if rep:
            t = log_text(tag)
            print("- %s: failed expectations %s, server alive %s, layer-major passes %d, cache shrinks %d" % (tag, rep.get("fails"), rep.get("alive"), t.count("lm-prefill: pass"), t.count("short by")))
    print()

    print("### Fault injection, variants, memcheck")
    for tag in ("g676-fail-acquire", "g677-fail-run1", "g730-fail2-once", "g731-fail1-once", "g732-min-high"):
        st = steps(tag)
        print("- %s: %s" % (tag, "; ".join("%s %s%s" % (k, v["status"], (" %.0f t/s" % v["timings"].get("prompt_per_second", 0)) if v["status"] == 200 and v["timings"].get("prompt_n", 0) > 100 else "") for k, v in st.items())))
    for tag in ("g680-seg4096", "g681-chunk1024", "g682-slots1", "g683-bf16"):
        rep = report(tag)
        if rep:
            print("- %s: failed expectations %s" % (tag, rep.get("fails")))
    sp = os.path.join(ROOT, "g738-lm-memcheck", "sanitizer.txt")
    if os.path.exists(sp):
        txt = open(sp, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
        m = re.findall(r"ERROR SUMMARY: (\d+) errors?", txt)
        print("- g738 compute-sanitizer memcheck: %s" % ("ERROR SUMMARY: " + ", ".join(m) if m else "no summary line (%d bytes)" % len(txt)))
        print("  layer-major passes in the log: %d" % log_text("g738-lm-memcheck").count("lm-prefill: pass"))


if __name__ == "__main__":
    main()
