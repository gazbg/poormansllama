"""MTP draft depth study (owner request 2026-10-14): --mtp-draft 1 / 2 / 4 / 6 / 8, each without and with --mtp-draft-p-min 0.1, against B0, on the layer-major 4,093-token code and prose cells (recipe MF). Sessions g120-*, build g58.

    python mk_g120.py [cell ...]       writes build/llama-cuda/specs/g120-*.json and copies them next to this file
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {"g120-lc4k": "g108-lc4k.json", "g120-lp4k": "g108-lp4k.json", "g120-sc500": "g108-sc500.json", "g120-lc24k": "g107-lc24k.json"}
EXE = os.environ.get("G120_EXE", "build/llama-cuda/bin/g58")
OUT = os.path.join(HERE, "..", "..", "..", "build", "llama-cuda", "specs")
DEPTHS = [int(x) for x in os.environ.get("G120_DEPTHS", "1,2,4,6,8").split(",")]
REPS = int(os.environ.get("G120_REPS", "2"))
for name, src in SRC.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    s = json.load(open(os.path.join(HERE, src)))
    by = {a["name"]: a for a in s["arms"]}
    base = by["B0"]
    mtp = by.get("MF") or by.get("MFLt")
    arms = [base]
    for n in DEPTHS:
        for pmin in (0.0, 0.1):
            a = copy.deepcopy(mtp)
            a["name"] = "D%d%s" % (n, "p" if pmin > 0 else "")
            a["flags"][a["flags"].index("--mtp-draft") + 1] = str(n)
            if pmin > 0:
                a["flags"] += ["--mtp-draft-p-min", str(pmin)]
            a["why"] = "MTP draft depth %d%s" % (n, ", draft p-min 0.1" if pmin > 0 else "")
            arms.append(a)
    s["session"] = name
    s["exe_dir"] = EXE
    s["arms"] = arms
    s["control"] = "B0"
    s["reps"] = REPS
    data = json.dumps(s, indent=1).encode("utf-8")
    open(os.path.join(OUT, name + ".json"), "wb").write(data)
    open(os.path.join(HERE, name + ".json"), "wb").write(data)
    print(name, [a["name"] for a in arms], "reps", REPS)
