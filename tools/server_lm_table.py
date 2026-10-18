#!/usr/bin/env python3
"""G32 LM-2: the cold-prompt prefill table of the arms of one comparison (steps.jsonl of tools/server-probe.py sessions of tools/server-scenarios/lm-scale*.json).

  python tools/server_lm_table.py --arm STOCK=build/llama-cuda/g603 --arm WINDOW=build/llama-cuda/g602 --arm LM=build/llama-cuda/g601 [--base STOCK] [--out table.md]

Per step (p1k .. p100k, repetitions a / b) it prints prompt_n, the prefill rate of every arm (t/s, from the response's timings), the wall time of the prefill (prompt_ms) and the ratio of each arm to --base (default: the first arm).
When an arm's server.log holds the layer-major pass lines (LLAMA_LM_LOG=1) they are summarised under the table. stdlib only; nothing is computed from memory.
"""
import argparse
import json
import os
import re
import statistics


def load_steps(d):
    p = os.path.join(d, "steps.jsonl")
    out = {}
    if os.path.exists(p):
        for l in open(p, "r", encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[r["name"]] = r
    return out


def passes(d):
    p = os.path.join(d, "server.log")
    if not os.path.exists(p):
        return []
    txt = open(p, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
    res = []
    for m in re.finditer(r"lm-prefill: pass (\d+): (\d+) tokens at position (\d+) in (\d+) chunks / (\d+) segments: ([\d.]+) s \(([\d.]+) t/s\); embed ([\d.]+) ms, slot wait ([\d.]+) ms, head ([\d.]+) ms, (\d+) launches: build ([\d.]+) ms, alloc ([\d.]+) ms, inputs ([\d.]+) ms, compute ([\d.]+) ms", txt):
        g = m.groups()
        res.append(dict(n=int(g[0]), tokens=int(g[1]), pos=int(g[2]), chunks=int(g[3]), segs=int(g[4]), s=float(g[5]), tps=float(g[6]), embed=float(g[7]), wait=float(g[8]), head=float(g[9]),
                        launches=int(g[10]), build=float(g[11]), alloc=float(g[12]), inputs=float(g[13]), compute=float(g[14])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=DIR (a session directory)")
    ap.add_argument("--base")
    ap.add_argument("--out")
    a = ap.parse_args()
    arms = []
    for s in a.arm:
        n, d = s.split("=", 1)
        arms.append((n, d, load_steps(d)))
    base = a.base or arms[0][0]
    names = []
    for _, _, st in arms:
        for k in st:
            if k not in names and k != "warmup":
                names.append(k)

    def key(n):
        m = re.match(r"p(\d+)k-([a-z])", n)
        return (int(m.group(1)), m.group(2)) if m else (10**9, n)

    names.sort(key=key)
    lines = []
    head = "| step | prompt_n | " + " | ".join("%s t/s (s)" % n for n, _, _ in arms) + " | " + " | ".join("%s / %s" % (n, base) for n, _, _ in arms if n != base) + " |"
    lines.append(head)
    lines.append("|" + "---|" * (2 + len(arms) + len(arms) - 1))
    stb = dict((n, st) for n, _, st in arms)[base]
    for nm in names:
        row = [nm]
        pn = None
        cells = []
        for n, _, st in arms:
            r = st.get(nm)
            if r is None or r.get("status") != 200:
                cells.append("-")
                continue
            tm = r["timings"]
            pn = pn or tm.get("prompt_n")
            cells.append("%.0f (%.2f)" % (tm.get("prompt_per_second", 0), tm.get("prompt_ms", 0) / 1000.0))
        ratios = []
        for n, _, st in arms:
            if n == base:
                continue
            r, rb = st.get(nm), stb.get(nm)
            if r is None or rb is None or r.get("status") != 200 or rb.get("status") != 200:
                ratios.append("-")
            else:
                ratios.append("%.2fx" % (r["timings"]["prompt_per_second"] / rb["timings"]["prompt_per_second"]))
        lines.append("| " + " | ".join([nm, str(pn)] + cells + ratios) + " |")
    out = "\n".join(lines)
    for n, d, st in arms:
        ps = passes(d)
        if ps:
            out += "\n\n%s: %d layer-major passes (server.log). tokens / seconds / t/s | sum of build, alloc, inputs, compute (ms) over the launches | slot wait (ms)\n" % (n, len(ps))
            for p in ps:
                out += "  pass %2d: %6d tok at %6d: %6.2f s %6.0f t/s | %d launches: build %.0f, alloc %.0f, inputs %.0f, compute %.0f | wait %.0f, embed %.0f, head %.0f\n" % (
                    p["n"], p["tokens"], p["pos"], p["s"], p["tps"], p["launches"], p["build"], p["alloc"], p["inputs"], p["compute"], p["wait"], p["embed"], p["head"])
    print(out)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(out + "\n")


if __name__ == "__main__":
    main()
