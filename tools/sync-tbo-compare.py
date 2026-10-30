#!/usr/bin/env python3
"""Compare test-backend-ops logs of the upstream-sync gates (docs/plans/upstream-sync-plan.md, G3).

usage: sync-tbo-compare.py <synced.log> <upstream.log> [<control.log>]

Case strings are normalised (upstream adds src_overlap / m_v / pad fields over time). Prints, per log, the failing cases
by (op, quant type) and the cases that fail on the synced build but not on the others. On this machine (RTX 5090 Laptop,
CUDA 13.2, MSVC) MUL_MAT and MUL_MAT_ID with iq1_s / iq2_s / iq3_s weights fail erratically on EVERY build, pure upstream
included, and the failing subset differs between builds; the gate is therefore the family, not the case: the synced
build may fail nothing outside it (exit status 1 if it does). Measured 2026-10-30: three runs of the SAME control binary
fail 28 / 26 / 26 of the iq1_s..iq3_s MUL_MAT cases with different sets, so a per-case comparison is meaningless here.
"""
import collections
import re
import sys

ansi = re.compile(r"\x1b\[[0-9;]*m")
# grid / codebook quantisations whose CUDA matmul fails erratically here, and the one LIGHTNING_INDEXER case that fails on
# the control and on pure upstream too; a failing case is "outside" when it is none of those and fails on no other build
FAMILY_RE = re.compile(r"type(?:_a)?=iq\d|^LIGHTNING_INDEXER")


def norm(k):
    return re.sub(r",(src_overlap|m_v|pad)=\d+", "", k)


def load(path):
    """Failing cases of one log. A case header line is '  OP(args): ' followed by OK / FAIL on the same line or, when the
    CUDA graph warm-up logs in between, by a bare OK / FAIL line later; NMSE failures print '[OP] ERR = x > y   OP(args): FAIL'."""
    raw = open(path, "rb").read().replace(b"\x00", b"")
    t = ansi.sub("", raw.decode("utf-8", "replace"))
    fails = {}
    last = None
    for l in t.splitlines():
        m = re.search(r"\]\s+ERR = ([0-9.eE+-]+) > [0-9.eE+-]+\s+(.*?): FAIL", l)
        if m:
            fails[norm(m.group(2).strip())] = m.group(1)
            last = None
            continue
        m = re.match(r"^  (\S.*?\)): (.*)$", l)
        if m:
            last = norm(m.group(1).strip())
            if m.group(2).strip().endswith("FAIL"):
                fails.setdefault(last, "")
            continue
        if l.strip() == "FAIL" and last is not None:
            fails.setdefault(last, "")
    summary = [l.strip() for l in t.splitlines() if "tests passed" in l or "backends passed" in l][-2:]
    return fails, summary


def bucket(f):
    c = collections.Counter()
    for k in f:
        m = re.search(r"type_a=(\w+)", k)
        c[(k.split("(")[0], m.group(1) if m else "-")] += 1
    return dict(c)


logs = [("synced", sys.argv[1]), ("upstream", sys.argv[2])] + ([("control", sys.argv[3])] if len(sys.argv) > 3 else [])
data = {n: load(p) for n, p in logs}
for n, (f, s) in data.items():
    print(f"{n}: {len(f)} failing cases  {s}")
    print("   by op/type:", bucket(f))
synced = data["synced"][0]
others = set().union(*[set(data[n][0]) for n in data if n != "synced"])
outside = [k for k in synced if not FAMILY_RE.search(k) and k not in others]
print("synced failures outside the IQ-quantisation family and not failing elsewhere:", len(outside))
for k in outside[:20]:
    print("   ", k)
for n in ("upstream", "control"):
    if n in data:
        print(f"synced failing, passing on {n}: {len(set(synced) - set(data[n][0]))}")
sys.exit(1 if outside else 0)
