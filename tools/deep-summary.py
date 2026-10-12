#!/usr/bin/env python3
"""One row per run of a session: the numbers the deep-fill contract (G19) reports.

    python tools/deep-summary.py <session> [<session> ...]      (reads build/llama-cuda/<session>/*.log)

Per run: prompt tokens and PP rate, TG tokens and rate, the auto-sized slots, the lazily committed KV rows, the free-memory watcher's minimum (the owner's keep-free
definition needs >= 16,000 MiB), the engine's hit rate and decode timeline (span = attn + ovl + stall, gap), and for stock runs the `prompt eval` / `eval` lines of
llama-completion. Host-only; the logs are UTF-16 with BOM when the runner wrote them through PowerShell, otherwise UTF-8.
"""
import glob
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "llama-cuda")


def read(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if b"\x00" in raw[:4000]:
        return raw.replace(b"\x00", b"").decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


PATTERNS = [
    ("pp_tok", r"PP: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\)"),
    ("tg", r"TG: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\)"),
    ("slots", r"auto slots:.*-> (\d+) slots"),
    ("kv", r"kv-lazy: committed ([\d.]+) MiB of KV cache \((\d+) of"),
    ("free", r"free-watch: minimum free device memory ([\d.]+) MiB"),
    ("freetg", r"free-watch \(decode phase[^)]*\): minimum free device memory ([\d.]+) MiB"),
    ("hit", r"moe_hybrid: tokens (\d+), hit rate ([\d.]+) .*?promotions/token ([\d.]+)"),
    ("timeline", r"timeline per token over (\d+) tokens \(us\): span ([\d.]+) = attn ([\d.]+) \+ ovl ([\d.]+) \+ stall ([\d.]+), gap ([\d.]+)"),
    ("hosttiming", r"tg host timing over (\d+) steps: llama_decode \(host side, up to the graph launch\) ([\d.]+) us/step, logits fetch \(the GPU wait\) ([\d.]+) us/step"),
    ("stock_pp", r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens .*?([\d.]+) tokens per second"),
    ("stock_tg", r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) runs .*?([\d.]+) tokens per second"),
]


def summarize(path):
    text = read(path)
    out = {}
    for key, pat in PATTERNS:
        m = None
        for m in re.finditer(pat, text):
            pass
        if m:
            out[key] = m.groups()
    return out


def fmt(d):
    parts = []
    if "pp_tok" in d:
        n, ms, rate = d["pp_tok"]
        parts.append("PP %s tok %.1f s %.0f tok/s" % (n, float(ms) / 1000.0, float(rate)))
    if "stock_pp" in d:
        ms, n, rate = d["stock_pp"]
        parts.append("PP(stock) %s tok %.1f s %.0f tok/s" % (n, float(ms) / 1000.0, float(rate)))
    if "tg" in d:
        n, ms, rate = d["tg"]
        parts.append("TG %s tok %.2f tok/s" % (n, float(rate)))
    if "stock_tg" in d:
        ms, n, rate = d["stock_tg"]
        parts.append("TG(stock) %s tok %.2f tok/s" % (n, float(rate)))
    if "slots" in d:
        parts.append("slots %s" % d["slots"][0])
    if "kv" in d:
        parts.append("kv %s rows %s MiB" % (d["kv"][1], d["kv"][0]))
    if "free" in d:
        parts.append("minfree %s MiB" % d["free"][0] + ((" (decode phase %s)" % d["freetg"][0]) if "freetg" in d else ""))
    if "hit" in d:
        parts.append("hit %.4f prom/tok %s" % (float(d["hit"][1]), d["hit"][2]))
    if "timeline" in d:
        n, span, attn, ovl, stall, gap = d["timeline"]
        parts.append("span %.0f=attn %.0f+ovl %.0f+stall %.0f gap %.0f" % (float(span), float(attn), float(ovl), float(stall), float(gap)))
    if "hosttiming" in d:
        n, launch, wait = d["hosttiming"]
        parts.append("host launch %.0f us GPU wait %.0f us" % (float(launch), float(wait)))
    return " | ".join(parts)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    for session in argv:
        d = os.path.join(ROOT, session)
        logs = sorted(glob.glob(os.path.join(d, session + "-*.log")))
        print("== %s (%d logs)" % (session, len(logs)))
        for p in logs:
            name = os.path.basename(p)[len(session) + 1:-4]
            print("  %-12s %s" % (name, fmt(summarize(p))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
