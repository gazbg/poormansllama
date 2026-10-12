"""G23 regression guard for the decode: the g44 runner without any new flag against g40 on a 4,096-token prompt with 128 generated tokens, three counterbalanced reps (tok/s of the TG phase).

    python mk_g73.py [EXE_DIR] [CONTROL_DIR]     writes build/llama-cuda/specs/g73-tgreg.json (+ a copy next to this file)
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402
import mk_qa   # noqa: E402

ROOT = mk_g21.root
EXE = sys.argv[1] if len(sys.argv) > 1 else "build/llama-cuda/bin/g44"
CONTROL = sys.argv[2] if len(sys.argv) > 2 else "build/llama-cuda/bin/g40"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")

session = "g73-tgreg"
spec, arms = mk_qa.make(session, "anchor-4096.txt", EXE, ["QKFP"])


def arm(name, why, exe_dir=None):
    a = copy.deepcopy(arms[0])
    a["name"] = name
    fl = list(a["flags"])
    fl[fl.index("--ubatch") + 1] = "2048"
    fl += ["--layer-slots", "2"]
    a["flags"] = fl
    a["env"]["LLAMA_FATTN_GQA8"] = "1"
    if exe_dir:
        a["exe_dir"] = exe_dir
    a["why"] = why
    return a


spec.update({"qa": False, "tg_tokens": 128, "reps": 3, "interleave": True, "counterbalance": True, "control": "TG40",
             "arms": [arm("TG40", "g40 control: final configuration, 4,096-token prompt, 128 generated tokens", CONTROL), arm("TG44", "g44 without any new flag")]})
json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
print(session, [a["name"] for a in spec["arms"]])
