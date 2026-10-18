#!/usr/bin/env python3
"""G32 review rounds (final binary s42 = s40 + the G37 diagnostic + G38 window close): the tables of the sessions k26xx / k27xx / k28xx (final, s42), k21xx (s40), k11xx / k9xx / k10xx (s36 / s33 history), g97xq / g85x / g77x plus the references they are read against. Host-only; reads build/llama-cuda/<tag>/.
  python tools/server_lm_review_tables.py [section ...]       (no argument: all sections)
Sections: np2flows, suites, soaks, soakseries, soakworkloads, canonical, canoncost, invariance, equiv, firsttok, frozen, ckpt, memcheck, lm1, speed, runner, fshape, fckpt, fcont, hitfresh"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

B = "build/llama-cuda/"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import server_canon_tables as canon_tables   # noqa: E402  (G41: the canonical-mode tables)


def tail_lines(tag, pat, n=0):
    p = B + tag + "/wrapper.txt"
    if not os.path.exists(p):
        return []
    out = [l.rstrip() for l in open(p, "rb").read().decode("utf-8", "replace").replace("\0", "").split("\n") if re.search(pat, l)]
    return out[-n:] if n else out


def log_text(tag):
    p = B + tag + "/server.log"
    return open(p, "rb").read().decode("utf-8", "replace").replace("\0", "") if os.path.exists(p) else ""


def passes(tag):
    return len(re.findall(r"W lm-prefill: pass \d+:", log_text(tag)))


def declines(tag):
    t = log_text(tag)
    c = {}
    for m in re.finditer(r"takes the ordinary path: ([^\n\[]*?)\s*(\[[^\n]*\])?\n", t):
        why = m.group(1)
        key = "several sequences" if "several sequences" in why else ("layout: " + re.sub(r"\d+", "N", (m.group(2) or "")[:70]) if m.group(2) else why[:60])
        c[key] = c.get(key, 0) + 1
    g = len(re.findall(r"could not make \d+ MiB of device memory", t))
    if g:
        c["memory guard (lm_acquire)"] = g
    return c


def report(tag):
    p = B + tag + "/report.json"
    return json.load(open(p)) if os.path.exists(p) else None


def sec_np2flows():
    print("### Two-slot flows (conversation X = natdeep-24000, Y = mtpprose-24000; extensions of 3,000 tokens; the server picks the slots)")
    for tag, label in (("k2851-np2-lm", "LM FINAL s42, default cache"), ("k2852-np2cr0-lm", "LM FINAL s42, --cache-ram 0"), ("k2860-np2kvu-lmstock", "LM-STOCK FINAL s42 (stock decode, -kvu)"), ("k2712-np2-lm", "LM s42 (second run)"), ("k2713-np2cr0-lm", "LM s42, --cache-ram 0 (second run)"),
                       ("k2151-np2-lm", "LM s40, default cache"), ("k2152-np2cr0-lm", "LM s40, --cache-ram 0"), ("k2160-np2kvu-lmstock", "LM-STOCK s40 (stock decode, -kvu)"),
                       ("g971q-np2-lm", "LM s33, default cache"), ("k932-np2cr0-lm", "LM, --cache-ram 0"), ("k940-np2kvu-lmstock", "LM-STOCK (stock decode, -kvu)"),
                       ("g806-np2-win", "WINDOW, default cache (s20)"), ("g807-np2-stock", "STOCK, default (s0)"), ("g809-np2cr0-win", "WINDOW, --cache-ram 0"), ("g810-np2cr0-stock", "STOCK, --cache-ram 0")):
        L = tail_lines(tag, r"^(s[01]-|both-)")
        if not L:
            continue
        print("- %s (%s): %d passes; declines %s" % (tag, label, passes(tag), declines(tag) or "none"))
        for l in L:
            print("    " + l[:150])


def sec_suites():
    print("### Multi-turn suites on the final binary")
    for tag in ("k3925-dyn-mt-np2-s44", "k3926-dyn-mt-np1-s44", "k3924-dyn-mt-np2-swap-s44", "k3927-dyn-mt-np1-swap-s44", "k3922-lm-growth-s44", "k3921-lm-stress-s44", "k3923-lm-slots-s44",
                "k2853-lm-mt-np2", "k2855-lm-mt-np1", "k2856-lm-growth", "k2857-lm-stress", "k2858-lm-slots", "k2708-dyn-mt-np1", "k2710-dyn-mt-np2", "k2702-frozen-mt-np1", "k2703-frozen-mt-np2", "k2821-dyn-mt-np1-swap", "k2822-dyn-mt-np2-swap", "k2704-frozen-mt-np1-swap", "k2707-lmstock-mt-np1",
                "k2901-frozen512-mt-np1", "k2904-frozen512-mt-np2", "k2902-frozen512-mt-np1-swap", "k2903-frozen512-mt-np2-swap", "k3001-dyn512-mt-np1", "k3002-dyn512-mt-np2", "k2905-dyn512-mt-np1-swap", "k3003-dyn512-mt-np2-swap",
                "k2401-swap-stock", "k2402-swap-lmstock2048", "k2403-swap-lmstock512", "k2153-lm-mt-np2", "k2155-lm-mt-np1", "k2156-lm-growth", "k2157-lm-stress", "k2158-lm-slots", "k933-lm-mt-np2", "k935-lm-mt-np1", "k936-lm-growth", "k937-lm-stress", "k938-lm-slots", "g761-lm-mt-np2", "g763-stock-mt-np2", "g762-win-mt-np2"):
        r = report(tag)
        if r:
            ws = re.findall(r"free-watch: new minimum of free device memory (\d+) MiB", log_text(tag))
            print("- %-18s fails %s, alive %s, nvidia-smi free min %s MiB, watcher min %s MiB, LM passes %d, batches >= min that took the ordinary path: %s" % (tag, r["fails"], r["alive"], r["free_min_smi_mib"], min(map(int, ws)) if ws else "-", passes(tag), declines(tag) or "none"))
    # prompt_n equality of the np2 suites
    def rows(tag):
        r = report(tag)
        return {(x["scenario"], x["name"]): x for x in r["rows"] if x.get("prompt_n") is not None} if r else {}
    S, W, L = rows("g763-stock-mt-np2"), rows("g762-win-mt-np2"), rows("k2853-lm-mt-np2")
    if S and L:
        keys = [k for k in S if k in L]
        eq = sum(1 for k in keys if S[k]["prompt_n"] == L[k]["prompt_n"])
        print("  np2 suite, FINAL s42 LM against STOCK: %d requests, same prompt_n %d, fewer %d, more %d" % (
            len(keys), eq, sum(1 for k in keys if L[k]["prompt_n"] < S[k]["prompt_n"]), sum(1 for k in keys if L[k]["prompt_n"] > S[k]["prompt_n"])))


def sec_soaks():
    print("### Soaks (160 requests, mixed lengths, cancelled streams)")
    for tag, label in (("k2854-lm-soak-np2", "np 2, FINAL s42"), ("k2859-lm-soak-np1", "np 1, FINAL s42"), ("k2154-lm-soak-np2", "np 2, s40"), ("k2159-lm-soak-np1", "np 1, s40"), ("g711-lm-soak-np2", "np 2, first round (s23)"), ("g831-lm-soak-np2", "np 2, s28"), ("g972q-lm-soak-np2", "np 2, FINAL code s33 (gate)"), ("g710-lm-soak-np1", "np 1, first round"), ("k939-lm-soak-np1", "np 1, FINAL s33"),
                       ("g514-soak-np2", "np 2 WINDOW (s20)")):
        p = B + tag + "/soak.json"
        if not os.path.exists(p):
            continue
        s = json.load(open(p))["summary"]
        print("- %-20s %-26s census %s failed %s alive %s wall %s s, idle free VRAM %s MiB, free min %s, watcher min %s, rss %s, LM passes %d; declines %s" % (
            tag, label, s.get("census"), s.get("failed"), s.get("alive"), s.get("wall_s"), s.get("smi_free_mib"), s.get("free_min_smi_mib"), s.get("watcher_min_mib"), s.get("rss_mib"), passes(tag), declines(tag) or "none"))


def sec_soakseries():
    import math
    print("### np 2 soak series (160 requests, 3 clients, tools/server_soak.py), every run kept, per build; cascade = 10 or more memory-guard declines")
    rows = []
    for d in sorted(glob.glob(B + "k*soak-np2*/")):
        tag = os.path.basename(d.rstrip("/\\"))
        p = d + "soak.json"
        if not os.path.exists(p):
            continue
        m = re.search(r"-s(\d+)$", tag)
        build = "s" + m.group(1) if m else {"k1254-lm-soak-np2": "s36", "k1854-lm-soak-np2": "s39", "k2154-lm-soak-np2": "s40", "k2854-lm-soak-np2": "s42"}.get(tag, "?")
        s = json.load(open(p))["summary"]
        t = log_text(tag)
        guard = len(re.findall(r"could not make \d+ MiB of device memory|no room for", t))
        # one decline logs both the hook line and the acquire line in s41/s42: count declines by the acquire line
        guard = len(re.findall(r"could not make \d+ MiB of device memory", t)) or guard
        rows.append((build, tag, passes(tag), guard, len(re.findall(r"several sequences", t)), s.get("census"), s.get("wall_s"), s.get("watcher_min_mib")))
    print("%-5s %-26s %6s %6s %8s  %-14s %7s %8s" % ("build", "tag", "passes", "guard", "severalS", "census", "wall s", "watcher"))
    for r in sorted(rows):
        print("%-5s %-26s %6d %6d %8d  %-14s %7s %8s%s" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], "  <- cascade" if r[3] >= 10 else ""))
    def split(builds):
        sel = [r for r in rows if r[0] in builds]
        return sum(1 for r in sel if r[3] >= 10), len(sel)
    for name, bs in (("s36/s39/s40", ("s36", "s39", "s40")), ("s41 (s40 + diagnostic only)", ("s41",)), ("s42 (window close)", ("s42",))):
        c, n = split(bs)
        print("  %-28s %d of %d runs with a cascade" % (name, c, n))
    ca, na = split(("s36", "s39", "s40"))
    cb, nb = split(("s42",))
    if na and nb:
        def pm(a, b, c, d):  # P(first row has a successes) for a hypergeometric table, two-sided Fisher exact
            n1, n2, k = a + b, c + d, a + c
            tot = math.comb(n1 + n2, k)
            def pr(x):
                return math.comb(n1, x) * math.comb(n2, k - x) / tot
            p0 = pr(a)
            return sum(pr(x) for x in range(max(0, k - n2), min(n1, k) + 1) if pr(x) <= p0 * (1 + 1e-9))
        print("  Fisher exact (two-sided), cascade rate before (s36/s39/s40) %d/%d against s42 %d/%d: p = %.3f" % (ca, na, cb, nb, pm(ca, na - ca, cb, nb - cb)))
        pre = [r for r in rows if r[0] in ("s36", "s39", "s40", "s41")]
        post = [r for r in rows if r[0] == "s42"]
        a = sum(1 for r in pre if r[3] >= 1)
        c = sum(1 for r in post if r[3] >= 1)
        print("  runs with at least one memory-guard decline: before (s36-s41) %d/%d against s42 %d/%d: Fisher p = %.2g" % (a, len(pre), c, len(post), pm(a, len(pre) - a, c, len(post) - c)))


def sec_soakworkloads():
    print("### np 2 soak on six workloads other than the seed-11 mix, s40 (before G38) / s42 (G38) / s43 (G38 + G39) / s44 (G38 + G39 + G40); same seed / workers / mix per workload for every build; repeats of W3 / W4 are the k36NN and k40NN tags")
    W = {"01": "seed 21, 3 workers, 160 req", "02": "seed 22, 4 workers, 160 req", "03": "seed 23, 2 workers, 160 req", "04": "seed 24, 3 workers, max-ctx 70000", "05": "seed 25, 3 workers, fixtures A+B only", "06": "seed 26, 3 workers, 240 req"}
    tot = {}
    print("%-3s %-36s | %-18s %-5s %6s %6s %8s %-13s %6s %8s" % ("w", "workload", "tag", "build", "passes", "guard", "severalS", "census", "wall s", "watcher"))
    for n, label in W.items():
        tags = []
        for pre in ("k33%s", "k35%s", "k36%s", "k39%s", "k40%s"):
            for b in ("s40", "s42", "s43", "s44"):
                tags.append(("%s-soak-np2-%s" % (pre % n, b), b))
        for tag, b in tags:
            p = B + tag + "/soak.json"
            if not os.path.exists(p):
                continue
            s = json.load(open(p))["summary"]
            t = log_text(tag)
            guard = len(re.findall(r"could not make \d+ MiB of device memory", t))
            row = tot.setdefault(b, [0, 0, 0, 0, 0])
            row[0] += 1
            row[1] += 1 if guard >= 1 else 0
            row[2] += 1 if guard >= 10 else 0
            row[3] += passes(tag)
            row[4] += guard
            print("%-3s %-36s | %-18s %-5s %6d %6d %8d %-13s %6s %8s" % (n, label, tag[:18], b, passes(tag), guard, len(re.findall(r"several sequences", t)), s.get("census"), s.get("wall_s"), s.get("watcher_min_mib")))
    for b in sorted(tot):
        r = tot[b]
        print("  %s: %d runs, with at least one memory-guard decline %d, cascades (10 or more) %d, passes %d, guard declines %d" % (b, r[0], r[1], r[2], r[3], r[4]))
    if "s44" in tot and "s40" in tot and "s42" in tot:
        import math
        def fisher(a, b, c, d):
            n1, n2, k = a + b, c + d, a + c
            def pr(x):
                return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
            p0 = pr(a)
            return min(1.0, sum(pr(x) for x in range(max(0, k - n2), min(n1, k) + 1) if pr(x) <= p0 * (1 + 1e-9)))
        pre_n = tot["s40"][0] + tot["s42"][0]
        for idx, name in ((1, "at least one memory-guard decline"), (2, "a cascade (10 or more)")):
            a = tot["s40"][idx] + tot["s42"][idx]
            c = tot["s44"][idx]
            print("  %s: s40+s42 %d/%d against s44 %d/%d, Fisher exact p = %.4f" % (name, a, pre_n, c, tot["s44"][0], fisher(a, pre_n - a, c, tot["s44"][0] - c)))


def sec_invariance():
    print("### np 1 invariance: the 64-token LM-STOCK continuations of 16 prompts: s28 (g772) against s33 (k941), and s33 (k941) against the final s40 with the opt-out (k2181: LLAMA_LM_FA_EXACT=0 FAFIX_CAUSAL_GQA=0)")
    for a, b in ((B + "g772-eq-lmstock/gen.json", B + "k941-eq-lmstock/gen.json"), (B + "k941-eq-lmstock/gen.json", B + "k2181-eq-lmstock-optout/gen.json"), (B + "k941-eq-lmstock/gen.json", B + "k4181-eq-lmstock-optout/gen.json")):
      if os.path.exists(a) and os.path.exists(b):
        A, Bj = json.load(open(a)), json.load(open(b))
        same = sum(1 for k in A if k in Bj and [s["token"] for s in A[k]["steps"]] == [s["token"] for s in Bj[k]["steps"]])
        lp = max((abs((x["logprob"] or 0) - (y["logprob"] or 0)) for k in A if k in Bj for x, y in zip(A[k]["steps"], Bj[k]["steps"])), default=0)
        print("  %s -> %s: %d of %d prompts token-identical; max |logprob difference| over all generated tokens %.6f" % (os.path.basename(os.path.dirname(a)), os.path.basename(os.path.dirname(b)), same, len(A), lp))


def sec_equiv():
    print("### 64-token greedy continuations, 16 prompts: pair classes (reference -> candidate)")
    runs = [("S512", "g770-eq-stock512"), ("S2048", "g771-eq-stock2048"), ("LMS", "g772-eq-lmstock"), ("WIN", "g773-eq-window"), ("LM", "g774-eq-lm")]
    args = [sys.executable, os.path.join(HERE, "server_gen_matrix.py")] + ["%s=%s%s/gen.json" % (l, B, t) for l, t in runs if os.path.exists(B + t + "/gen.json")]
    for pr in ("S512,S2048", "S512,LMS", "S2048,LMS", "S512,WIN", "S512,LM", "WIN,LM"):
        args += ["--pair", pr]
    print(subprocess.run(args, capture_output=True, text=True).stdout)


def sec_firsttok():
    print("### First generated token from the prefill alone (--ctx-checkpoints 0), 16 prompts, stock decode")
    runs = [("S512", "g854-ft16-stock512"), ("S2048", "g855-ft16-stock2048"), ("L512", "g856-ft16-lm512"), ("L2048", "g857-ft16-lm2048")]
    args = [sys.executable, os.path.join(HERE, "server_firsttok_matrix.py")] + ["%s=%s%s/firsttok.json" % (l, B, t) for l, t in runs if os.path.exists(B + t + "/firsttok.json")]
    for pr in ("S512,S2048", "S512,L512", "S2048,L2048", "S512,L2048", "S2048,L512"):
        args += ["--pair", pr]
    print(subprocess.run(args, capture_output=True, text=True).stdout)


def sec_frozen():
    print("### Shipping decode path with the expert cache FROZEN (--hybrid-moe-slots 16, LLAMA_MOE_HYBRID_B=0): WINDOW and LM, two sessions each, 16 prompts x 64 tokens")
    runs = [("WINa", "k960-eqfz-window-a"), ("LMa", "k961-eqfz-lm-a"), ("WINb", "k962-eqfz-window-b"), ("LMb", "k963-eqfz-lm-b")]
    args = [sys.executable, os.path.join(HERE, "server_gen_matrix.py")] + ["%s=%s%s/gen.json" % (l, B, t) for l, t in runs if os.path.exists(B + t + "/gen.json")]
    for pr in ("WINa,WINb", "LMa,LMb", "WINa,LMa", "WINb,LMb", "WINa,LMb"):
        args += ["--pair", pr]
    print(subprocess.run(args, capture_output=True, text=True).stdout)


def sec_ckpt():
    print("### First token WITH the default context checkpoints (the realistic server flow), stock decode")
    runs = [("S512", "k964-ftck-stock512"), ("S2048", "k965-ftck-stock2048"), ("LM", "k966-ftck-lm2048")]
    args = [sys.executable, os.path.join(HERE, "server_firsttok_matrix.py")] + ["%s=%s%s/firsttok.json" % (l, B, t) for l, t in runs if os.path.exists(B + t + "/firsttok.json")]
    for pr in ("S512,S2048", "S512,LM", "S2048,LM"):
        args += ["--pair", pr]
    print(subprocess.run(args, capture_output=True, text=True).stdout)


def sec_memcheck():
    print("### compute-sanitizer memcheck (graceful stop, final binary)")
    for tag in ("k4631-canon-memcheck-np2", "k3932-memcheck-seg-s44", "k3931-memcheck-np2-s44", "k2714-memcheck-seg", "k2715-memcheck-np2", "k2014-memcheck-seg", "k2015-memcheck-np2", "k1714-memcheck-seg", "k1715-memcheck-np2", "k1201-memcheck-seg", "k1202-memcheck-np2", "k1101-memcheck-seg", "k1102-memcheck-np2", "k1103-memcheck-window", "k974-memcheck-seg", "k975-memcheck-np2", "k970-memcheck-seg", "k971-memcheck-np2", "g790-memcheck-seg"):
        p = B + tag + "/sanitizer.txt"
        if not os.path.exists(p):
            continue
        t = open(p, "rb").read().decode("utf-8", "replace").replace("\0", "")
        summ = re.findall(r"ERROR SUMMARY[^\n]*", t)
        errs = {}
        for m in re.finditer(r"========= (Program hit [^\n]*|Invalid [^\n]*|Race [^\n]*|Uninitialized [^\n]*)", t):
            k = re.sub(r"0x[0-9a-f]+", "", m.group(1))[:140]
            errs[k] = errs.get(k, 0) + 1
        calls = {}
        for c in re.findall(r"on CUDA API call to (\w+)", t):
            calls[c] = calls.get(c, 0) + 1
        print("- %s: %s; reports: %s; by API call: %s" % (tag, [s.strip() for s in summ] or "no ERROR SUMMARY line", errs or "none", calls or "-"))
        ls = re.findall(r"lm-prefill: pass (\d+): (\d+) tokens at position (\d+) in (\d+) chunks / (\d+) segments: ([\d.]+) s", log_text(tag))
        for x in ls:
            print("    pass %s: %s tokens at position %s, %s chunks, %s segments, %s s" % x)
        for l in tail_lines(tag, r"stopped gracefully|graceful stop"):
            print("    " + l[:120])


def sec_lm1():
    print("### LM-1 default-off identity on the final binary (A B B A)")
    for name, tags in (("FINAL flags off: STOCK s0 / s44 / s44 / s0", ("k4161-off-A1", "k4162-off-B1", "k4163-off-B2", "k4164-off-A2")), ("FINAL --hybrid-moe without --lm-prefill: s20 / s44 / s44 / s20", ("k4165-hyb-A1", "k4166-hyb-B1", "k4167-hyb-B2", "k4168-hyb-A2")),
                       ("s42 flags off: STOCK s0 / s42 / s42 / s0", ("k2861-off-A1", "k2862-off-B1", "k2863-off-B2", "k2864-off-A2")), ("FINAL --hybrid-moe without --lm-prefill: s20 / s42 / s42 / s20", ("k2865-hyb-A1", "k2866-hyb-B1", "k2867-hyb-B2", "k2868-hyb-A2")),
                       ("s40 flags off: STOCK s0 / s40 / s40 / s0", ("k2161-off-A1", "k2162-off-B1", "k2163-off-B2", "k2164-off-A2")), ("s40 --hybrid-moe without --lm-prefill: s20 / s40 / s40 / s20", ("k2165-hyb-A1", "k2166-hyb-B1", "k2167-hyb-B2", "k2168-hyb-A2")),
                       ("flags off: STOCK s0 / s33 / s33 / s0", ("k942-off-A1", "k943-off-B1", "k944-off-B2", "k945-off-A2")), ("--hybrid-moe without --lm-prefill: s20 / s33 / s33 / s20", ("k946-hyb-A1", "k947-hyb-B1", "k948-hyb-B2", "k949-hyb-A2"))):
        if all(os.path.exists(B + t + "/steps.jsonl") for t in tags):
            print("- " + name)
            print(subprocess.run([sys.executable, os.path.join(HERE, "server_smoke_cmp.py")] + list(tags), capture_output=True, text=True, cwd=os.path.dirname(HERE)).stdout)


def sec_speed():
    print("### LM-2 speed on the FINAL binary s42: LM k2871/k2872 against WINDOW k2870/k2873 (A B B A), and the same on s40 (k2171/k2172 against k2170/k2173); older: s33 k950/k951, s24 g734/g735, WINDOW g733/g736")
    def steps(tag):
        out = {}
        p = B + tag + "/steps.jsonl"
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                out[r["name"]] = r
        return out
    S = {t: steps(t) for t in ("g734-lm-a", "g735-lm-b", "k950-lm-a", "k951-lm-b", "g733-win-a", "g736-win-b", "k2170-win-a", "k2171-lm-a", "k2172-lm-b", "k2173-win-b", "k2870-win-a", "k2871-lm-a", "k2872-lm-b", "k2873-win-b")}
    for n in [n for n in S.get("k2171-lm-a", {}) if n.startswith("p") and n != "warmup"]:
        def pp2(t):
            tm = S[t].get(n, {}).get("timings") or {}
            return tm.get("prompt_per_second", 0), tm.get("prompt_n")
        a, b, wa, wb = (pp2(t) for t in ("k2171-lm-a", "k2172-lm-b", "k2170-win-a", "k2173-win-b"))
        mw = (wa[0] + wb[0]) / 2 or 1
        a2, b2, wa2, wb2 = (pp2(t) for t in ("k2871-lm-a", "k2872-lm-b", "k2870-win-a", "k2873-win-b"))
        mw2 = (wa2[0] + wb2[0]) / 2 or 1
        print("FINAL %-9s %7s | s42 LM %5.0f %5.0f | WINDOW %5.0f %5.0f | LM / WINDOW %.2fx || s40 LM %5.0f %5.0f | WINDOW %5.0f %5.0f | %.2fx" % (n, a2[1], a2[0], b2[0], wa2[0], wb2[0], (a2[0] + b2[0]) / 2 / mw2, a[0], b[0], wa[0], wb[0], (a[0] + b[0]) / 2 / mw))
    print("%-9s %7s | s24 LM a/b | s33 LM a/b | WINDOW a/b | s33 LM / WINDOW" % ("step", "prompt"))
    for n in [n for n in S["k950-lm-a"] if n.startswith("p") and n != "warmup"]:
        def pp(t):
            tm = S[t].get(n, {}).get("timings") or {}
            return tm.get("prompt_per_second", 0), tm.get("prompt_n")
        a24, b24, a33, b33, wa, wb = (pp(t) for t in ("g734-lm-a", "g735-lm-b", "k950-lm-a", "k951-lm-b", "g733-win-a", "g736-win-b"))
        mw = (wa[0] + wb[0]) / 2 or 1
        print("%-9s %7s | %5.0f %5.0f | %5.0f %5.0f | %5.0f %5.0f | %.2fx" % (n, a33[1], a24[0], b24[0], a33[0], b33[0], wa[0], wb[0], (a33[0] + b33[0]) / 2 / mw))


def sec_runner():
    print("### Runner identity cells (LM-8): the final llama.dll (build g80 = s42) k2869-*, then k2169-* (g79 = s40), k1169-* (g74), against the last accepted runner (g58); older: g72 k969-*")
    for s in ("lc4k", "sc24", "sgc50k"):
      for pre in ("k4169", "k2869", "k2169", "k1169", "k969"):
        p = B + "%s-%s.wrapper.log" % (pre, s)
        if not os.path.exists(p):
            continue
        t = open(p, "rb").read().decode("utf-8", "replace").replace("\0", "")
        m = re.search(r"arm +n +mean.*?\n(.*?)\nverdict", t, re.S)
        h = {}
        for f in sorted(glob.glob(B + "%s-%s/*.txt" % (pre, s))):
            h[os.path.basename(f)] = hashlib.sha256(open(f, "rb").read()).hexdigest()[:10]
        dl = re.findall(r"^B0\w+#\d +\d +[\d.]+ +- +\S+ +\S+ +\d +\S+ +([\d.-]+)", t, re.M)
        print("- %s-%s: texts %s (%d files), dlogitmed %s, provenance %s" % (pre, s, sorted(set(h.values())), len(h), sorted(set(dl)), "0 refused" if "0 refused" in t else "CHECK"))
        if m:
            print("    " + m.group(1).replace("\n", "\n    "))


def ftm(runs, pairs, tool="server_firsttok_matrix.py", key="firsttok.json", verbose=False):
    args = [sys.executable, os.path.join(HERE, tool)] + ["%s=%s%s/%s" % (l, B, t, key) for l, t in runs if os.path.exists(B + t + "/" + key)]
    for pr in pairs:
        args += ["--pair", pr]
    if verbose:
        args.append("-v")
    print(subprocess.run(args, capture_output=True, text=True).stdout)


def sec_fshape():
    print("### EQUAL CHUNK SHAPE, hybrid decode (the shipping decode), expert cache frozen, first token of the whole prompt in one pass (--ctx-checkpoints 0), 16 prompts")
    print("W512 / W2048 = the window path with a static ubatch (s20), L512 / L2048 = the pass with chunk 512 / 2048 on s35 (exact flash attention by default, k1104/k1105); L512F / L2048F = the same on the FINAL s40 (k1204/k1205); S34L* / S34X* = the s34 arms with the old / the exact variant (k1041/42/51/52)")
    ftm([("W512", "k1040-w512"), ("L512", "k1104-ftfz-lm512"), ("W2048", "k1050-w2048"), ("L2048", "k1105-ftfz-lm2048"), ("L512F", "k1204-ftfz-lm512"), ("L2048F", "k1205-ftfz-lm2048"), ("S34L512", "k1041-lm512"), ("S34X512", "k1042-lm512x"), ("S34L2048", "k1051-lm2048"), ("S34X2048", "k1052-lm2048x")],
        ("W512,L512", "W2048,L2048", "W512,W2048", "L512,L2048", "W512,L512F", "W2048,L2048F", "W512,S34L512", "W512,S34X512", "W2048,S34L2048", "W2048,S34X2048", "W512,L2048", "W2048,L512"))


def sec_fckpt():
    print("### The realistic flow (default context checkpoints), first token of 20 prompts, stock decode: S512 (stock, k990), S2048, LM chunk 2048, LM chunk 512 (= the shape of S512), control with the pass never taken")
    ftm([("S512", "k990-ftck20-stock512"), ("S2048", "k1106-ftck20-stock2048"), ("LM2048", "k1107-ftck20-lm2048"), ("LM512", "k1108-ftck20-lm512"), ("NOLM", "k1109-ftck20-nolm"), ("LMU2048", "k1219-ftck20-lm-ub2048"),
         ("S33S512", "k990-ftck20-stock512"), ("S33LM2048", "k992-ftck20-lm2048")],
        ("S512,S2048", "S512,LM2048", "S2048,LM2048", "S512,LM512", "S512,NOLM", "S2048,LMU2048", "S33LM2048,LM2048"))


def sec_fcont():
    print("### 64-token greedy continuations, 16 prompts: stock decode (S512 = g770, S2048, LM, LM chunk 512) and hybrid decode, cache frozen, no checkpoint cut, equal shapes (window static ubatch against the pass)")
    ftm([("S512", "g770-eq-stock512"), ("S2048", "k1110-eq-stock2048"), ("LMS", "k1111-eq-lmstock"), ("LMS512", "k1112-eq-lmstock512"), ("W2048", "k1115-eqfz0-w2048s"), ("L2048", "k1116-eqfz0-lm2048"), ("L2048F", "k1216-eqfz0-lm2048"), ("L512F", "k1218-eqfz0-lm512"),
         ("W512", "k1117-eqfz0-w512"), ("L512", "k1118-eqfz0-lm512"), ("LMSU2048", "k1220-eq-lmstock-ub2048")],
        ("S512,S2048", "S512,LMS", "S2048,LMS", "S512,LMS512", "S2048,LMSU2048", "W2048,L2048", "W512,L512", "W2048,L2048F", "W512,L512F", "W512,W2048"), tool="server_gen_matrix.py", key="gen.json")


def sec_hitfresh():
    print("### Replicated pairs, hit-versus-fresh (HF) and same-request-twice (FF), 17 prompts x (3 fresh + 2 hit runs), 48 tokens (round 3; tools/server_hitfresh.py)")
    runs = [("WIN", "k1401-hf-win"), ("LM", "k1402-hf-lm"), ("LM512", "k1403-hf-lm512"), ("LM33", "k1404-hf-lm33"), ("LMSTOCK", "k1405-hf-lmstock"), ("LMFZ", "k1406-hf-lm-frozen16"),
            ("WINFZ", "k1407-hf-win-frozen16"), ("STOCK", "k1408-hf-stock"),
            ("LM42", "k2709-hf-lm-dyn"), ("FROZEN512", "k2906-hf-frozen512"), ("DYN512", "k2907-hf-dyn512"), ("LMSTOCK42", "k2706-hf-lmstock"), ("FROZEN42", "k2705-hf-frozen-flag-lm"),
            ("LM40", "k2009-hf-lm-dyn"), ("LMSTOCK40", "k2006-hf-lmstock"), ("FROZEN36", "k1606-hf-frozen-flag-lm"), ("FROZEN40", "k2005-hf-frozen-flag-lm")]
    args = [sys.executable, os.path.join(HERE, "server_hitfresh.py"), "analyze"] + ["%s=%s%s/hitfresh.json" % (l, B, t) for l, t in runs if os.path.exists(B + t + "/hitfresh.json")]
    print(subprocess.run(args, capture_output=True, text=True).stdout)
    print("### The suite's five hit-versus-fresh checks with the conversation fixtures swapped (k1411 WINDOW, k1412 LM)")
    for tag in ("k1411-swap-win", "k1412-swap-lm", "k2121-dyn-mt-np1-swap", "k2122-dyn-mt-np2-swap", "k2004-frozen-mt-np1-swap", "k2821-dyn-mt-np1-swap", "k2822-dyn-mt-np2-swap", "k2704-frozen-mt-np1-swap", "k2401-swap-stock", "k2402-swap-lmstock2048", "k2403-swap-lmstock512"):
        r = report(tag)
        if r:
            print("- %s: fails %s, LM passes %d" % (tag, r["fails"], passes(tag)))


SECTIONS = {"np2flows": sec_np2flows, "suites": sec_suites, "soaks": sec_soaks, "soakseries": sec_soakseries, "soakworkloads": sec_soakworkloads, "canonical": canon_tables.sec_canonical, "canoncost": canon_tables.sec_canoncost, "invariance": sec_invariance, "equiv": sec_equiv, "firsttok": sec_firsttok, "frozen": sec_frozen,
            "ckpt": sec_ckpt, "memcheck": sec_memcheck, "lm1": sec_lm1, "speed": sec_speed, "runner": sec_runner, "fshape": sec_fshape, "fckpt": sec_fckpt, "fcont": sec_fcont, "hitfresh": sec_hitfresh}

if __name__ == "__main__":
    want = sys.argv[1:] or list(SECTIONS)
    for w in want:
        SECTIONS[w]()
        print()
