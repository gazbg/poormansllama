"""G23 criterion B4 (default-off identity) on the FINAL build g44: the first run of g57-pace-63 had LM40 (the g40 control) 6% slower than LM41 in its first rep (1438 against 1352 ms; the other
five pairs of g57 were within +-1% of each other). This reruns the identity pair with THREE reps at 63, 256 and 1,023 tokens, g44 without any new flag against g40, both attested.

    python mk_g69.py [EXE_DIR] [CONTROL_DIR]     writes build/llama-cuda/specs/g69-ident-<N>.json (+ copies next to this file)
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

for n, tokens in ((64, 63), (256, 256), (1024, 1023)):
    session = "g69-ident-%d" % tokens
    spec, arms = mk_qa.make(session, "sweep-%d.txt" % n, EXE, ["QKFP"])
    c = list(spec["common"])
    for flag in ("--layer-slots", "--act-slab-dtype", "--tg-tokens", "--tg-logit-margin"):
        i = c.index(flag)
        if flag in ("--tg-tokens", "--tg-logit-margin"):
            c[i + 1] = "8"
        else:
            del c[i:i + 2]
    spec["common"] = c

    def arm(name, why, exe_dir=None):
        a = copy.deepcopy(arms[0])
        a["name"] = name
        fl = list(a["flags"])
        fl[fl.index("--ubatch") + 1] = "2048"
        fl += ["--layer-slots", "2", "--act-slab-dtype", "f32", "--pp-repeat", "2"]
        a["flags"] = fl
        a["env"]["LLAMA_FATTN_GQA8"] = "1"
        if exe_dir:
            a["exe_dir"] = exe_dir
        a["why"] = why
        return a

    spec.update({"qa": False, "tg_tokens": 8, "reps": 3, "interleave": True, "counterbalance": True, "control": "LM40",
                 "arms": [arm("LM40", "g40 control: all-GPU layer-major prefill, warm pass", CONTROL), arm("LM44", "g44 without any new flag")]})
    json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
    print(session, [a["name"] for a in spec["arms"]])
