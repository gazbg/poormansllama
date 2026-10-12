"""G22 follow-up: does a deeper slot window speed up the all-GPU layer-major prefill of a short prompt?

The trace of the 2,047-token prefill (g46-2048) shows the copy engine idle for tens of ms whenever a layer's compute is longer than a bank's copy (the 10 attention layers take 39-53 ms at that
length against about 9 ms for the 30 recurrent ones) because two slots forbid the next-but-one bank from being submitted before the layer that holds its slot has ended (`tools/g22-trace-sim.py`).
More slots let the copy engine run ahead.

    python mk_g52.py        writes build/llama-cuda/specs/g52-<N>.json (arms LM2, LM3, LM4, LM6; 2 counterbalanced reps) and copies them next to this file

Prompts: sweep-<N>.txt of g46 (N = 511, 1023, 2047 tokens) and anchor-4096.txt. Nothing else differs between the arms but --layer-slots.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402
import mk_g46  # noqa: E402
import mk_qa   # noqa: E402

ROOT = mk_g21.root
EXE = "build/llama-cuda/bin/g40"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")

for n, prompt_name in ((511, "sweep-512.txt"), (1023, "sweep-1024.txt"), (2047, "sweep-2048.txt"), (4096, "anchor-4096.txt")):
    session = "g52-%d" % n
    spec, arms = mk_qa.make(session, prompt_name, EXE, ["QKFP"])
    c = list(spec["common"])
    for flag in ("--layer-slots", "--act-slab-dtype", "--tg-tokens", "--tg-logit-margin"):
        i = c.index(flag)
        if flag in ("--tg-tokens", "--tg-logit-margin"):
            c[i + 1] = "8"
        else:
            del c[i:i + 2]
    spec["common"] = c
    out_arms = []
    for slots in (2, 3, 4, 6):
        a = copy.deepcopy(arms[0])
        a["name"] = "LM%d" % slots
        fl = list(a["flags"])
        fl[fl.index("--ubatch") + 1] = "2048"
        fl += ["--layer-slots", str(slots), "--act-slab-dtype", "f32"]
        a["flags"] = fl
        a["env"]["LLAMA_FATTN_GQA8"] = "1"
        a["why"] = "all-GPU layer-major prefill, ubatch 2048, %d layer slots with the async feed, f32 slab" % slots
        out_arms.append(a)
    spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": out_arms, "control": "LM2"})
    json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
    print(session, [a["name"] for a in out_arms])
