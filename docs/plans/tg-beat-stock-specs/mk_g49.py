"""G22 follow-up: what does the stock prefill really do for a batch of N tokens? Stock llama-completion with the owner's flags (STK, op offload on, the default) against the same
flags with `--no-op-offload` (STKNO: host tensor operations stay on the CPU) and with `-ub 2048` (STKU), on the sweep prompts of g46. With the offload on, the scheduler copies the experts a batch needs to the
GPU over PCIe for batches of 32 tokens or more; with it off the expert FFN runs on the CPU from the host banks, which is the CPU leg of the hybrid prefill (tools/g22-model.py).

    python mk_g49.py        writes build/llama-cuda/specs/g49-<N>.json (arms STK, STKNO and STKU, 2 counterbalanced reps) and copies them next to this file
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
PROMPTS = mk_g46.PROMPTS
MODEL = mk_g21.MODEL

for n in mk_g46.NS:
    session = "g49-%d" % n
    spec, arms = mk_qa.make(session, "sweep-%d.txt" % n, EXE, ["QKFP"])
    stk = {"name": "STK", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-mmap"], "env": {}, "why": "stock llama-completion, the owner's document configuration (op offload on)"}
    stkno = copy.deepcopy(stk)
    stkno["name"] = "STKNO"
    stkno["flags"] = stk["flags"] + ["--no-op-offload"]
    stkno["why"] = "the same with --no-op-offload: the expert FFN of the CPU layers runs on the CPU from the host banks"
    stku = copy.deepcopy(stk)
    stku["name"] = "STKU"
    stku["flags"] = stk["flags"] + ["-ub", "2048"]
    stku["why"] = "the owner's flags with --ubatch-size 2048 (the default is 512): one copy of the used experts per 2048 tokens instead of per 512"
    prompt = PROMPTS.replace("\\", "/") + "/sweep-%d.txt" % n
    spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": [stk, stkno, stku], "control": "STK"})
    spec["stock_common"] = ["-m", MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", "8",
                            "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt"]
    json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
    print(session, [a["name"] for a in spec["arms"]])
