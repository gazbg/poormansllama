"""G30-5 answer stability with MTP on: needle QA sessions g100-qa24000 and g100-qa50000 on build g52.

    python mk_g100.py          writes build/llama-cuda/specs/g100-qa*.json and copies them next to this file

The common part, the prompt and the grading come from the g47 QA sessions (programmatic grading + the must-fail negative control, never token identity); the arms are the final MTP recipe MF of the matrix
cell of the same fill (g99-lc24k for 24,000 tokens, g99-sgc50k for 50,000) and its pre-registered tail variant MFt. The MTP loop's first-token convention (no `--tg-first-pp-token`, `--pp-no-logits`)
replaces the QA sessions' `--tg-first-pp-token`.
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "build", "llama-cuda", "specs")
EXE = os.environ.get("G100_EXE", "build/llama-cuda/bin/g52")
PREFIX = os.environ.get("G100_PREFIX", "g100")
CELLS = os.environ.get("G100_CELLS", "g99-lc24k,g99-sgc50k").split(",")
ARMS = os.environ.get("G100_ARMS", "MF,MFt").split(",")
for fill, cell in ((24000, CELLS[0]), (50000, CELLS[1])):
    qa = json.load(open(os.path.join(HERE, "g47-qa%d.json" % fill)))
    cellspec = json.load(open(os.path.join(HERE, cell + ".json")))
    by = {a["name"]: a for a in cellspec["arms"]}
    arms = []
    for name in ARMS:
        a = copy.deepcopy(by[name])
        a["why"] = "G30-5: " + a["why"]
        arms.append(a)
    s = copy.deepcopy(qa)
    s["session"] = "%s-qa%d" % (PREFIX, fill)
    s["exe_dir"] = EXE
    s["arms"] = arms
    s["control"] = ARMS[0]
    # the cell spec's common differs from the QA one only in the cell's own prompt / tg settings: keep the QA common, but the band flags (layer slots, slab dtype, segments) are the cell's own
    qc = list(s["common"])
    for flag in ("--layer-slots", "--act-slab-dtype"):
        if flag in qc:
            i = qc.index(flag)
            del qc[i:i + 2]
    s["common"] = qc
    data = json.dumps(s, indent=1).encode("utf-8")
    open(os.path.join(OUT, s["session"] + ".json"), "wb").write(data)
    open(os.path.join(HERE, s["session"] + ".json"), "wb").write(data)
    print(s["session"], [a["name"] for a in arms], "prompt", s["prompt"].split("/")[-1])
