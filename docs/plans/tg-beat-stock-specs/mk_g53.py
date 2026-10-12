"""G22 follow-up: the layer-major prefill of a short prompt, WARM (second in-process pass) and with a deeper slot window.

The sweep of g46 times the first and only pass of a fresh process. The trace of that pass (g46-2048) shows (1) ~190 ms of one-time kernel loading in the first layer (layer 0 computes in 186-200 ms
against 9 ms for the layers that follow; stock llama-completion warms up before its timer, so the stock column of g46 does not pay it), and (2) with two slots the copy engine idles whenever a
layer's compute exceeds a bank's copy (replay: `tools/g22-trace-sim.py`: 3 slots -3%, 4 -5%, 6-8 -9..-11%). This session measures both: arms run `--pp-repeat 2` (the table tools read the LAST pass, i.e.
the warm one) with 2, 4, 6 and 8 layer slots.

    python mk_g53.py        writes build/llama-cuda/specs/g53-<N>.json (arms LM2W LM4W LM6W LM8W; 2 counterbalanced reps) and copies them next to this file

Prompts: sweep-<N>.txt of g46 (63, 255, 511, 1023, 2047 tokens) and anchor-4096.txt. Nothing else differs between the arms.
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
EXE = "build/llama-cuda/bin/g40"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")

for n, prompt_name in ((63, "sweep-64.txt"), (255, "sweep-256.txt"), (511, "sweep-512.txt"), (1023, "sweep-1024.txt"), (2047, "sweep-2048.txt"), (4096, "anchor-4096.txt")):
    session = "g53-%d" % n
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
    for slots in (2, 4, 6, 8):
        a = copy.deepcopy(arms[0])
        a["name"] = "LM%dW" % slots
        fl = list(a["flags"])
        fl[fl.index("--ubatch") + 1] = "2048"
        fl += ["--layer-slots", str(slots), "--act-slab-dtype", "f32", "--pp-repeat", "2"]
        a["flags"] = fl
        a["env"]["LLAMA_FATTN_GQA8"] = "1"
        a["why"] = "all-GPU layer-major prefill, ubatch 2048, %d layer slots with the async feed, f32 slab, TWO in-process passes (the last one is the warm one)" % slots
        out_arms.append(a)
    spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": out_arms, "control": "LM2W"})
    json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
    print(session, [a["name"] for a in out_arms])
