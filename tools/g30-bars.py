#!/usr/bin/env python3
"""G30-4 strict bars per cell: for every arm of the named sessions the poller peak (vram_peak_raw_mib, whole process life) and the in-process watcher minimum free memory.

    python tools/g30-bars.py g99-sc24 g99-lc24k ...

The bars (contract G24-1, constants): watcher minimum free >= 16,000 MiB AND poller peak <= 7,370 MiB. A cell is judged by G30-4 only where its B0 arm holds both bars; there MTP must hold them too.
Host-only; reads build/llama-cuda/<session>/<session>-verdict.json and the per-run logs.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "llama-cuda"))
FREE, PEAK = 16000.0, 7370.0


def text(p):
    raw = open(p, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace")


def main():
    for s in sys.argv[1:]:
        vp = os.path.join(ROOT, s, s + "-verdict.json")
        if not os.path.exists(vp):
            print("no verdict", s)
            continue
        v = json.load(open(vp))
        per = {}
        for k, a in v["arms"].items():
            name = k.split("#")[0]
            per.setdefault(name, {"peak": [], "free": []})
            per[name]["peak"].append(a.get("vram_peak_raw_mib"))
            lp = glob.glob(os.path.join(ROOT, s, "%s-%s_r%s.log" % (s, name, k.split("#")[1]))) if "#" in k else []
            if lp:
                m = re.findall(r"free-watch: minimum free device memory ([\d.]+) MiB", text(lp[0]))
                if m:
                    per[name]["free"].append(float(m[-1]))
        print("== %s" % s)
        base_ok = None
        for name, d in per.items():
            pk = max(x for x in d["peak"] if x is not None) if any(x is not None for x in d["peak"]) else None
            fr = min(d["free"]) if d["free"] else None
            ok = (pk is not None and pk <= PEAK) and (fr is not None and fr >= FREE)
            if name == "B0":
                base_ok = ok
            print("  %-5s poller peak max %s MiB, watcher min free %s MiB  -> bars %s" % (name, "-" if pk is None else "%.0f" % pk, "-" if fr is None else "%.0f" % fr, "HELD" if ok else "NOT held"))
        if base_ok is False:
            print("  (B0 itself does not hold the bars in this cell: G30-4 does not apply)")


if __name__ == "__main__":
    main()
