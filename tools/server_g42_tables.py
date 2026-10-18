#!/usr/bin/env python3
"""G42 tables (host-only, stdlib): the check sessions of --lm-prefill-fuse, the replicated pairs, the checkpoint positions of the legacy and the fused flow, the checkpoint hashes of the canonical mode.

  python tools/server_g42_tables.py checks      # the five hit-versus-fresh checks, session by session (canonical + fuse), and the default mode with and without fuse
  python tools/server_g42_tables.py pairs       # server_hitfresh.py analyze over the new runs
  python tools/server_g42_tables.py positions   # created / restored checkpoint positions of k5391 (BASE) and k5392 (FUSE), default mode
  python tools/server_g42_tables.py hashes      # checkpoint hashes of k5393 (legacy canonical) and k5394 (fused canonical)
"""
import os
import re
import subprocess
import sys

B = "build/llama-cuda/"
HERE = os.path.dirname(os.path.abspath(__file__))

CANON_FUSE = [
    ("stock decode", "swapped", "k5361-cf-lmstock-mt-swap"), ("stock decode", "original", "k5362-cf-lmstock-mt"),
    ("frozen cache", "swapped", "k5363-cf-frozen-mt-swap"), ("frozen cache", "original", "k5364-cf-frozen-mt"),
    ("dynamic cache", "swapped", "k5365-cf-dyn-mt-swap"), ("dynamic cache", "original", "k5366-cf-dyn-mt"),
]
DEFAULT = [
    ("default s47", "original", "k5381-def47-dyn-mt"), ("default s47", "swapped", "k5382-def47-dyn-mt-swap"),
    ("default s51 + fuse", "original", "k5383-def51fuse-dyn-mt"), ("default s51 + fuse", "swapped", "k5384-def51fuse-dyn-mt-swap"),
]
CHECK_RE = re.compile(r"^(branch|alternate|equiv) +(rollback-vs-fresh|restore-vs-first|same-turn\d+) +(ok|FAIL): (.*)$", re.M)
SHORT = {"rollback-vs-fresh": "rollback", "restore-vs-first": "restore"}


def read_out(tag):
    num = re.match(r"k(\d+)", tag).group(1)
    p = B + "k%s.out" % num
    return open(p, "rb").read().decode("utf-8", "replace").replace(chr(0), "") if os.path.exists(p) else None


def checks(sessions, title):
    print("### " + title)
    n_ok = n_all = n_ident = 0
    for dec, sess, tag in sessions:
        txt = read_out(tag)
        if txt is None:
            print("- %-18s %-9s %s: not run" % (dec, sess, tag))
            continue
        cells = []
        for m in CHECK_RE.finditer(txt):
            name = SHORT.get(m.group(2), m.group(2))
            msg = m.group(4)
            n_all += 1
            n_ok += 1 if m.group(3) == "ok" else 0
            if "identical over" in msg:
                n_ident += 1
                cells.append("%s identical" % name)
            else:
                d = re.search(r"first difference at token (\d+), margin ([\d.]+)", msg)
                cells.append("%s %s%s" % (name, ("d%s m%s" % d.groups()) if d else msg[:30], "" if m.group(3) == "ok" else " FAIL"))
        ft = re.findall(r"1st token \(from the prefill\): max\|dlogprob\| ([\d.]+)", txt)
        status = "all met" if "all expectations met" in txt else "EXPECTATIONS FAILED"
        print("- %-18s %-9s %-34s %s | first token max|dlogprob| %s | %s" % (dec, sess, tag, "; ".join(cells), max(map(float, ft)) if ft else "-", status))
    print("  %d of %d checks pass, %d of them bit-identical" % (n_ok, n_all, n_ident))


def pairs():
    runs = [("CF-LMSTOCK", "k5371-hf-cf-lmstock"), ("CF-FROZEN", "k5372-hf-cf-frozen"), ("CF-DYN", "k5373-hf-cf-dyn"), ("DEF47-DYN", "k5385-hf-def47-dyn"), ("DEF51FUSE-DYN", "k5386-hf-def51fuse-dyn"),
            ("OLD-CANON-LMSTOCK", "k4205-hf-canon-lmstock"), ("OLD-CANON-FROZEN", "k4206-hf-canon-frozen"), ("OLD-CANON-DYN", "k4215-hf-canon-dyn")]
    args = [sys.executable, os.path.join(HERE, "server_hitfresh.py"), "analyze"] + ["%s=%s%s/hitfresh.json" % (lab, B, t) for lab, t in runs if os.path.exists(B + t + "/hitfresh.json")]
    print(subprocess.run(args, capture_output=True, text=True).stdout)


def trace_ckpt(tag):
    """per request: the checkpoints created (n_tokens) and the one restored (n_tokens)"""
    reqs = []
    cur = None
    for raw in open(B + tag + "/server.log", "rb").read().decode("utf-8", "replace").replace(chr(0), "").split("\n"):
        if "launch_slot_" in raw and "processing task" in raw:
            cur = {"created": [], "restored": None, "erased": []}
            reqs.append(cur)
        elif cur is not None:
            m = re.search(r"created context checkpoint \d+ of \d+ \(pos_min = (\d+), pos_max = (\d+), n_tokens = (\d+)", raw)
            if m:
                cur["created"].append(int(m.group(3)))
            m = re.search(r"restored context checkpoint \(pos_min = (\d+), pos_max = (\d+), n_tokens = (\d+)", raw)
            if m:
                cur["restored"] = int(m.group(3))
            m = re.search(r"erasing context checkpoint too close to an earlier one \(pos_min = (\d+), pos_max = (\d+), n_tokens = (\d+)", raw)
            if m:
                cur["erased"].append(int(m.group(3)))
    return reqs


def positions():
    a, b = trace_ckpt("k5391-trace-def-base"), trace_ckpt("k5392-trace-def-fuse")
    same = 0
    for i, (x, y) in enumerate(zip(a, b)):
        ok = x["created"] == y["created"] and x["restored"] == y["restored"] and x["erased"] == y["erased"]
        same += ok
        print("request %2d: restored %s | created BASE %s FUSE %s | erased BASE %s FUSE %s | %s" % (i, x["restored"], x["created"], y["created"], x["erased"], y["erased"], "same" if ok else "DIFFERENT"))
    print("%d of %d requests have the same restored, created and erased checkpoints (BASE s47 against FUSE s51, default mode)" % (same, min(len(a), len(b))))


def hashes():
    def rows(tag):
        out = []
        for raw in open(B + tag + "/server.log", "rb").read().decode("utf-8", "replace").replace(chr(0), "").split("\n"):
            m = re.search(r"lm-fuse: checkpoint n_tokens = (\d+), pos_min = (\d+), pos_max = (\d+), (\d+) bytes, fnv ([0-9a-f]+), taken (.*)$", raw)
            if m:
                out.append((int(m.group(1)), int(m.group(4)), m.group(5), m.group(6).strip()))
        return out
    a, b = rows("k5393-hash-canon-base"), rows("k5394-hash-canon-fuse")
    da, db = {}, {}
    for n, sz, h, via in a:
        da.setdefault(n, []).append((h, via))
    for n, sz, h, via in b:
        db.setdefault(n, []).append((h, via))
    for n in sorted(set(da) | set(db)):
        print("n_tokens %6d: legacy %s | fused %s" % (n, da.get(n), db.get(n)))
    common = sorted(set(da) & set(db))
    eq = [n for n in common if da[n][0][0] == db[n][0][0]]
    print("%d positions in both; %d with the same hash (the first checkpoint of each position)" % (len(common), len(eq)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "checks"
    if cmd == "checks":
        checks(CANON_FUSE, "canonical + fuse (np 1; stock decode, frozen cache, dynamic cache; swapped and original fixture order): the five hit-versus-fresh checks")
        checks(DEFAULT, "default mode, dynamic cache: s47 against s51 + fuse")
    elif cmd == "pairs":
        pairs()
    elif cmd == "positions":
        positions()
    elif cmd == "hashes":
        hashes()
