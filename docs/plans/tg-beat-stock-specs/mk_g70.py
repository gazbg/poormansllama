"""G23 stage B on longer prompts (does pacing keep paying, or cost, once the compute outgrows the copy?), final build g44.

    python mk_g70.py [EXE_DIR]     writes build/llama-cuda/specs/g70-pacedeep-<N>.json (+ copies next to this file); default EXE_DIR build/llama-cuda/bin/g44

Arms: LM44 (no new flag), BPA (paced 16 MiB x 2), BPB (paced 4 MiB x 3); prompts anchor-4096 / anchor-24000 / anchor-50000 of the QA ladder; 2 counterbalanced reps; `--pp-repeat 2` at 4,096 tokens, 1 beyond.
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
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")

for prompt_name, tokens, repeat in (("anchor-4096.txt", 4096, 2), ("anchor-24000.txt", 24000, 1), ("anchor-50000.txt", 50000, 1)):
    session = "g70-pacedeep-%d" % tokens
    spec, arms = mk_qa.make(session, prompt_name, EXE, ["QKFP"])
    c = list(spec["common"])
    for flag in ("--layer-slots", "--act-slab-dtype", "--tg-tokens", "--tg-logit-margin"):
        i = c.index(flag)
        if flag in ("--tg-tokens", "--tg-logit-margin"):
            c[i + 1] = "8"
        else:
            del c[i:i + 2]
    spec["common"] = c

    def arm(name, why, extra):
        a = copy.deepcopy(arms[0])
        a["name"] = name
        fl = list(a["flags"])
        fl[fl.index("--ubatch") + 1] = "2048"
        fl += ["--layer-slots", "2", "--act-slab-dtype", "f32"] + (["--pp-repeat", str(repeat)] if repeat > 1 else []) + extra
        a["flags"] = fl
        a["env"]["LLAMA_FATTN_GQA8"] = "1"
        a["why"] = why
        return a

    spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "control": "LM44",
                 "arms": [arm("LM44", "g44 without a new flag", []),
                          arm("BPA", "paced bank copies 16 MiB x 2", ["--pp-copy-piece-mib", "16", "--pp-copy-window", "2"]),
                          arm("BPB", "paced bank copies 4 MiB x 3", ["--pp-copy-piece-mib", "4", "--pp-copy-window", "3"])]})
    json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
    print(session, [a["name"] for a in spec["arms"]])
