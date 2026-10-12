#!/usr/bin/env python3
"""G22: compare the last-position top-8 logits of a layer-major prefill arm with a control arm of the same session (the runner prints them with --tg-logit-margin 8).

    python tools/g22-logit-compare.py <session> <control arm> <test arm> [<test arm> ...] [--rep N]

Reads build/llama-cuda/<session>/<session>-<arm>_r<rep>.log. Per test arm and rep: the top-1 id of both, whether the top-8 id sets agree (and in which order), the largest absolute logit
difference over the ids both list, and the top-1 minus top-2 gap of each. Host-only. Criterion G22-1 (plan): the top-1 id equals the control's and the gap differs by at most 0.5.
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
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace")


def top8(path):
    out = []
    for m in re.finditer(r"PP: last-position top(\d+) id (-?\d+) logit (-?[\d.]+)", read(path)):
        out.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    out.sort()
    return [(i, v) for _, i, v in out]


def main(argv):
    reps = None
    if "--rep" in argv:
        k = argv.index("--rep")
        reps = [int(argv[k + 1])]
        del argv[k:k + 2]
    if len(argv) < 3:
        print(__doc__)
        return 2
    session, ctl, tests = argv[0], argv[1], argv[2:]
    d = os.path.join(ROOT, session)
    rep_ids = reps or sorted({int(re.search(r"_r(\d+)\.log$", p).group(1)) for p in glob.glob(os.path.join(d, "%s-%s_r*.log" % (session, ctl)))})
    bad = 0
    for r in rep_ids:
        c = top8(os.path.join(d, "%s-%s_r%d.log" % (session, ctl, r)))
        if not c:
            print("rep %d: control %s has no top-8 lines" % (r, ctl))
            bad += 1
            continue
        cmap = dict(c)
        cgap = c[0][1] - c[1][1]
        for t in tests:
            p = os.path.join(d, "%s-%s_r%d.log" % (session, t, r))
            if not os.path.exists(p):
                continue
            x = top8(p)
            if not x:
                print("rep %d %-5s: no top-8 lines" % (r, t))
                bad += 1
                continue
            xmap = dict(x)
            common = [i for i in cmap if i in xmap]
            dmax = max(abs(cmap[i] - xmap[i]) for i in common) if common else float("nan")
            xgap = x[0][1] - x[1][1]
            same_top1 = c[0][0] == x[0][0]
            same_order = [i for i, _ in c] == [i for i, _ in x]
            ok = same_top1 and abs(cgap - xgap) <= 0.5
            if not ok:
                bad += 1
            print("rep %d %-5s: top1 %d vs %d %s | ids shared %d/8 same order %s | max|dlogit| %.3f | gap %.3f vs %.3f (d %.3f) | %s" % (
                r, t, x[0][0], c[0][0], "same" if same_top1 else "DIFFERENT", len(common), same_order, dmax, xgap, cgap, xgap - cgap, "G22-1 OK" if ok else "G22-1 FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
