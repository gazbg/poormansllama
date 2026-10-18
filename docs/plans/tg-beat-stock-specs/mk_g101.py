"""G30-1 default-off identity: build g52 (all G30 edits) against build g47 (none) with the SAME B0 flags, no --mtp, in one counterbalanced session per cell.

    python mk_g101.py          writes build/llama-cuda/specs/g101-*.json and copies them next to this file
Cells: 24 tokens on the stock path (`g99-sc24`), 4,093 tokens layer-major (`g98-lc4k`), 49,999 tokens segmented (`g99-sgc50k`). Criterion G30-1: the generated token ids are identical (tools/g24-ids.py)
and PP / TG agree within 1.5%.
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "build", "llama-cuda", "specs")
for tag, src, reps in (("sc24", "g99-sc24", 3), ("lc4k", "g98-lc4k", 3), ("sgc50k", "g99-sgc50k", 2)):
    s = json.load(open(os.path.join(HERE, src + ".json")))
    b0 = next(a for a in s["arms"] if a["name"] == "B0")
    new = copy.deepcopy(b0)
    new["name"] = "B0new"
    new["exe_dir"] = os.environ.get("G101_EXE", "build/llama-cuda/bin/g52")
    new["why"] = "G30-1: build g52 (all G30 edits), no --mtp"
    old = copy.deepcopy(b0)
    old["name"] = "B0old"
    old["exe_dir"] = "build/llama-cuda/bin/g47"
    old["why"] = "G30-1: build g47 (before G30), the same flags"
    s["session"] = os.environ.get("G101_PREFIX", "g101") + "-" + tag
    s["exe_dir"] = os.environ.get("G101_EXE", "build/llama-cuda/bin/g52")
    s["arms"] = [new, old]
    s["control"] = "B0old"
    s["reps"] = reps
    data = json.dumps(s, indent=1).encode("utf-8")
    open(os.path.join(OUT, s["session"] + ".json"), "wb").write(data)
    open(os.path.join(HERE, s["session"] + ".json"), "wb").write(data)
    print(s["session"], [a["name"] for a in s["arms"]], "reps", reps)
