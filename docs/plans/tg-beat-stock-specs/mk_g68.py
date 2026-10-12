"""G23 final comparison on build g44 (stage B, fix 3 and the v4 plans in): the prefill wall of the paced hybrid prefill and of the paced all-GPU prefill against the stock binary, per prompt length.
Contract: docs/plans/tg-beat-stock-8gb-262k.md, "G23 CONTRACT" C2-C5 and B3 (the evidence of g50 on build g43 is the intermediate state).

    python mk_g68.py [EXE_DIR]     writes build/llama-cuda/specs/g68-<N>.json (+ copies next to this file); default EXE_DIR build/llama-cuda/bin/g44

Arms (prompt sweep-<N>.txt, 8 generated tokens, 2 counterbalanced reps; the runner arms run `--pp-repeat 2`, the table tools read the last pass; hybrid engine ON as in the final configuration):
    STK / STKNO / STKU   stock llama-completion, the owner's flags / + --no-op-offload / + -ub 2048
    LM2W   the all-GPU layer-major prefill of the final configuration (ubatch 2048, 2 slots, async feed, f32 slab), unpaced
    LM2P   the same with paced bank copies (16 MiB x 2)
    HYC    every layer on the CPU leg (lengths up to 125 tokens only)
    HY4P   the v3 model's 4-slot plan, paced copies              (the plan of g50 / g66)
    HY4Q   the v4 (paced) model's 4-slot plan, paced copies
    HY8Q   the v4 model's 8-slot plan, paced copies              (lengths from 256 tokens)
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
EXE = sys.argv[1] if len(sys.argv) > 1 else "build/llama-cuda/bin/g44"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")
PROMPTS = mk_g46.PROMPTS
MODEL = mk_g21.MODEL
PACE = ["--pp-copy-piece-mib", "16", "--pp-copy-window", "2"]
NS = ((32, 29), (64, 63), (128, 125), (256, 256), (512, 511), (1024, 1023), (2048, 2047))   # (sweep file number, real token count)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g22 = load("g22model", "g22-model.py")
g23 = load("g23model", "g23-model.py")


def plan(mod, tokens, slots, c):
    t, seq = mod.optimize(tokens, c, slots, 0, 1.163, 1.0)
    return ",".join(str(i) for i, k in enumerate(seq) if k == 'C'), t, seq.count('G')


for n, tokens in NS:
    session = "g68-%d" % tokens
    spec, arms = mk_qa.make(session, "sweep-%d.txt" % n, EXE, ["QKFP"])
    c = list(spec["common"])
    for flag in ("--layer-slots", "--act-slab-dtype", "--tg-tokens", "--tg-logit-margin"):
        i = c.index(flag)
        if flag in ("--tg-tokens", "--tg-logit-margin"):
            c[i + 1] = "8"
        else:
            del c[i:i + 2]
    spec["common"] = c

    def runner(name, why, slots, extra):
        a = copy.deepcopy(arms[0])
        a["name"] = name
        fl = list(a["flags"])
        fl[fl.index("--ubatch") + 1] = "2048"
        fl += ["--layer-slots", str(slots), "--act-slab-dtype", "f32", "--pp-repeat", "2"] + extra
        a["flags"] = fl
        a["env"]["LLAMA_FATTN_GQA8"] = "1"
        a["why"] = why
        return a

    stk = {"name": "STK", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-mmap"], "env": {}, "why": "stock llama-completion, the owner's document configuration (op offload on, ubatch 512)"}
    stkno = copy.deepcopy(stk)
    stkno["name"] = "STKNO"
    stkno["flags"] = stk["flags"] + ["--no-op-offload"]
    stkno["why"] = "the same with --no-op-offload"
    stku = copy.deepcopy(stk)
    stku["name"] = "STKU"
    stku["flags"] = stk["flags"] + ["-ub", "2048"]
    stku["why"] = "the owner's flags with ubatch 2048"
    arms_out = [stk, stkno, stku,
                runner("LM2W", "all-GPU layer-major prefill, unpaced, two slots, warm pass", 2, []),
                runner("LM2P", "the same with paced bank copies (16 MiB x 2)", 2, PACE)]
    if tokens <= 125:
        arms_out.append(runner("HYC", "every layer on the CPU leg", 2, ["--pp-cpu-layers", "all"]))
    l3, t3, g3 = plan(g22, tokens, 4, 41.0)
    l4, t4, g4 = plan(g23, tokens, 4, 33.0)
    arms_out.append(runner("HY4P", "v3 model 4-slot plan: %d GPU layers; CPU layers %s" % (g3, l3), 4, ["--pp-cpu-layers", l3] + PACE))
    if l4 != l3:
        arms_out.append(runner("HY4Q", "v4 model 4-slot plan: %d GPU layers, predicted %.0f ms; CPU layers %s" % (g4, t4, l4), 4, ["--pp-cpu-layers", l4] + PACE))
    if tokens >= 256:
        l8, t8, g8 = plan(g23, tokens, 8, 33.0)
        if l8 not in (l3, l4):
            arms_out.append(runner("HY8Q", "v4 model 8-slot plan: %d GPU layers, predicted %.0f ms; CPU layers %s" % (g8, t8, l8), 8, ["--pp-cpu-layers", l8] + PACE))
    prompt = PROMPTS.replace("\\", "/") + "/sweep-%d.txt" % n
    spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": arms_out, "control": "STK"})
    spec["stock_common"] = ["-m", MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", "8",
                            "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt"]
    json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
    print(session, [a["name"] for a in arms_out])
