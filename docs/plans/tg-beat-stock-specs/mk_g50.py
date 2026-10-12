"""G22 stage C: the interleaved CPU/GPU layer-major prefill (--pp-cpu-layers) against the all-GPU layer-major prefill and the stock prefill, per prompt length.

    python mk_g50.py [EXE_DIR]   writes build/llama-cuda/specs/g50-<N>.json (+ copies next to this file); default EXE_DIR build/llama-cuda/bin/g41

Arms per session (prompt sweep-<N>.txt of g46, 8 generated tokens, 2 counterbalanced reps; only the prefill is of interest):
    STK    stock llama-completion, the owner's flags
    LM2    the layer-major prefill of g46 (ubatch 2048, 2 layer slots with the async feed, f32 slab): every bank streamed
    HYC    every layer on the CPU leg (--pp-cpu-layers all): the CPU path through the layer-major controller, no bank streamed
    HY2    the model's best static plan for 2 slots (tools/g22-model.py, c = 41 ms, DRAM contention 1.163): the CPU layers named in --pp-cpu-layers
    HY4    the same for 4 slots (--layer-slots 4)
The plans are printed next to the arm names so that a session record shows exactly what was run.
"""
import copy
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402
import mk_g46  # noqa: E402
import mk_qa   # noqa: E402

ROOT = mk_g21.root
EXE = sys.argv[1] if len(sys.argv) > 1 else "build/llama-cuda/bin/g41"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")
PROMPTS = mk_g46.PROMPTS
MODEL = mk_g21.MODEL
NS = (32, 64, 128, 256, 512, 1024, 2048)

_spec = importlib.util.spec_from_file_location("g22model", os.path.join(ROOT, "tools", "g22-model.py"))
g22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g22)


def plan_for(n_tokens, slots):
    """CPU layers (comma list) of the model's best static plan for a prompt of n_tokens tokens and `slots` layer slots (no bank resident at the start)."""
    t, seq = g22.optimize(n_tokens, 41.0, slots, 0, 1.163, 1.0)
    return ",".join(str(i) for i, k in enumerate(seq) if k == 'C'), t, seq.count('G')


for n in NS:
    session = "g50-%d" % n
    spec, arms = mk_qa.make(session, "sweep-%d.txt" % n, EXE, ["QKFP"])
    c = list(spec["common"])
    for flag in ("--layer-slots", "--act-slab-dtype", "--tg-tokens", "--tg-logit-margin"):
        i = c.index(flag)
        if flag in ("--tg-tokens", "--tg-logit-margin"):
            c[i + 1] = "8"
        else:
            del c[i:i + 2]
    spec["common"] = c

    def lm_arm(name, slots, extra, why):
        a = copy.deepcopy(arms[0])
        a["name"] = name
        fl = list(a["flags"])
        fl[fl.index("--ubatch") + 1] = "2048"
        fl += ["--layer-slots", str(slots), "--act-slab-dtype", "f32"] + extra
        a["flags"] = fl
        a["env"]["LLAMA_FATTN_GQA8"] = "1"
        a["why"] = why
        return a

    tokens = {32: 29, 64: 63, 128: 125, 256: 256, 512: 511, 1024: 1023, 2048: 2047}[n]   # the real token counts of the sweep prompts
    l2, t2, g2 = plan_for(tokens, 2)
    l4, t4, g4 = plan_for(tokens, 4)
    lm2 = lm_arm("LM2", 2, [], "all-GPU layer-major prefill: ubatch 2048, two layer slots with the async feed, f32 slab")
    hyc = lm_arm("HYC", 2, ["--pp-cpu-layers", "all"], "every layer on the CPU leg, no bank streamed")
    hy2 = lm_arm("HY2", 2, ["--pp-cpu-layers", l2], "model plan, 2 slots: %d GPU layers, predicted %.0f ms; CPU layers %s" % (g2, t2, l2))
    hy4 = lm_arm("HY4", 4, ["--pp-cpu-layers", l4], "model plan, 4 slots: %d GPU layers, predicted %.0f ms; CPU layers %s" % (g4, t4, l4))
    stk = {"name": "STK", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-mmap"], "env": {}, "why": "stock llama-completion, the owner's document configuration"}
    prompt = PROMPTS.replace("\\", "/") + "/sweep-%d.txt" % n
    spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": [stk, lm2, hyc, hy2, hy4], "control": "STK"})
    spec["stock_common"] = ["-m", MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", "8",
                            "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt"]
    json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
    print(session, "tokens", tokens, "| HY2 %d GPU layers pred %.0f ms | HY4 %d GPU layers pred %.0f ms" % (g2, t2, g4, t4))
