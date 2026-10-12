"""G23 criteria A3 and C3 (strict gate): the needle QA smoke tier (512 tokens, programmatic grading, the negative control must fail) through the new prefill paths.

    python mk_g59.py [EXE_DIR]     writes build/llama-cuda/specs/g59-qa512.json (+ a copy next to this file); default EXE_DIR build/llama-cuda/bin/g43

Arms (the prompt is anchor-512.txt of the QA ladder, 128 generated tokens, ctx 262,144):
    Q0    the final configuration's layer-major prefill (ubatch 2048, 2 slots, f32 slab): the reference
    QSP   the stock graph path for the prompt (--pp-stock-below 4096, op offload on, ubatch 512)
    QSP0  the same with the experts on the CPU (--pp-stock-offload 0)
    QHYC  every layer on the CPU leg
    QHY   the model's 2-slot plan for 512 tokens with paced copies (16 MiB x 2)
"""
import copy
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402
import mk_qa   # noqa: E402

ROOT = mk_g21.root
EXE = sys.argv[1] if len(sys.argv) > 1 else "build/llama-cuda/bin/g43"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")

_spec = importlib.util.spec_from_file_location("g22model", os.path.join(ROOT, "tools", "g22-model.py"))
g22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g22)
t, seq = g22.optimize(512, 41.0, 2, 0, 1.163, 1.0)
plan = ",".join(str(i) for i, k in enumerate(seq) if k == 'C')

spec, arms = mk_qa.make("g59-qa512", "anchor-512.txt", EXE, ["QKFP"])
base = arms[0]


def arm(name, why, ubatch, extra):
    a = copy.deepcopy(base)
    a["name"] = name
    fl = list(a["flags"])
    fl[fl.index("--ubatch") + 1] = str(ubatch)
    fl += list(extra)
    a["flags"] = fl
    a["env"]["LLAMA_FATTN_GQA8"] = "1"
    a["why"] = why
    return a


spec["arms"] = [
    arm("Q0", "layer-major prefill, ubatch 2048 (reference)", 2048, []),
    arm("QSP", "stock graph path for the prompt, op offload on, ubatch 512", 512, ["--pp-stock-below", "4096"]),
    arm("QSP0", "stock graph path for the prompt, experts on the CPU, ubatch 512", 512, ["--pp-stock-below", "4096", "--pp-stock-offload", "0"]),
    arm("QHYC", "every layer on the CPU leg", 2048, ["--pp-cpu-layers", "all"]),
    arm("QHY", "model plan for 512 tokens, 2 slots (CPU layers %s), paced copies" % plan, 2048, ["--pp-cpu-layers", plan, "--pp-copy-piece-mib", "16", "--pp-copy-window", "2"]),
]
spec["control"] = "Q0"
json.dump(spec, open(os.path.join(OUT, "g59-qa512.json"), "w"), indent=1)
json.dump(spec, open(os.path.join(HERE, "g59-qa512.json"), "w"), indent=1)
print("g59-qa512", [a["name"] for a in spec["arms"]], "plan", plan)
