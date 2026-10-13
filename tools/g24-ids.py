#!/usr/bin/env python3
"""Compare the GENERATED TOKEN IDS of two arms of one session, rep by rep (host-only).

    python tools/g24-ids.py <session> <armA> <armB> [<armC> ...]      first arm = reference; every other arm is compared with it

The ids come from the `tg_tokens` event of each run's telemetry trace (build/llama-cuda/<session>/<session>-<ARM>[_rN].jsonl). For every rep both arms ran it prints the
number of ids, the first divergence (none = identical) and, where they diverge, the two ids at that step. Exit 0 only if every compared pair is identical.
Used for the default-off identity checks (G27 / G28: new flags absent, same tokens as the previous build) and for the segmented-prefill identity (a segmented prefill must
generate the same tokens as the unsegmented one).
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "llama-cuda"))


def ids_of(path):
    for line in open(path, encoding="utf-8", errors="replace"):
        if '"tg_tokens"' in line:
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("event") == "tg_tokens":
                return o.get("ids")
    return None


def runs(session, arm):
    out = {}
    for p in glob.glob(os.path.join(ROOT, session, "%s-%s*.jsonl" % (session, arm))):
        m = re.match(r"^%s-%s(?:_r(\d+))?\.jsonl$" % (re.escape(session), re.escape(arm)), os.path.basename(p))
        if m:
            out[int(m.group(1) or 1)] = p
    return out


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    session, ref, others = sys.argv[1], sys.argv[2], sys.argv[3:]
    rref = runs(session, ref)
    ok = True
    for arm in others:
        r = runs(session, arm)
        for rep in sorted(set(rref) & set(r)):
            a, b = ids_of(rref[rep]), ids_of(r[rep])
            if a is None or b is None:
                print("%s rep %d vs %s: no ids (%s / %s)" % (ref, rep, arm, a is not None, b is not None))
                ok = False
                continue
            n = min(len(a), len(b))
            first = next((i for i in range(n) if a[i] != b[i]), None)
            same = first is None and len(a) == len(b)
            ok = ok and same
            print("%s rep %d vs %s: %d / %d ids, %s" % (ref, rep, arm, len(a), len(b),
                  "IDENTICAL" if same else ("first divergence at step %s (%s vs %s)" % (first, a[first] if first is not None else "-", b[first] if first is not None else "-"))))
    print("IDS:", "IDENTICAL" if ok else "DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
