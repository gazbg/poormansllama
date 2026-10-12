"""G23 root-cause session (criterion C5): why is the CPU path of the runner 30-60% SLOWER than the stock binary's CPU path from ~500 tokens (g56 SP0 against STKNO, g50 HYC against STKNO), and 15-20% faster
below ~125 tokens? Hypothesis H1: the hybrid decode engine of the runner (LLAMA_MOE_HYBRID=14: 8 spinning host-leg threads + the lean team, created at bring-up) keeps cores busy during the
prefill, so the 16 threads of the CPU matmul compete with them (24 cores). Test: the same arms with the engine OFF (LLAMA_MOE_HYBRID=0, no generation: `--tg-tokens 0`).

    python mk_g64.py [EXE_DIR]     writes build/llama-cuda/specs/g64-<N>.json (+ copies next to this file); default EXE_DIR build/llama-cuda/bin/g43

Arms (prompt sweep-<N>.txt, `--pp-repeat 2`, 2 counterbalanced reps; the prefill of the last pass is the datum):
    STKNO  stock llama-completion, owner's flags + --no-op-offload (the CPU path of the stock engine)
    SP0    runner, stock graph path, experts on the CPU, engine ON
    SP0X   the same with the engine OFF
    HYC    runner, every layer on the CPU leg, engine ON
    HYCX   the same with the engine OFF
    HY4P   runner, the model's 4-slot plan with paced copies, engine ON
    HY4PX  the same with the engine OFF
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
EXE = sys.argv[1] if len(sys.argv) > 1 else "build/llama-cuda/bin/g43"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")
PROMPTS = mk_g46.PROMPTS
MODEL = mk_g21.MODEL
PACE = ["--pp-copy-piece-mib", "16", "--pp-copy-window", "2"]

_spec = importlib.util.spec_from_file_location("g22model", os.path.join(ROOT, "tools", "g22-model.py"))
g22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g22)

for n, tokens in ((256, 256), (512, 511), (1024, 1023)):
    session = "g64-%d" % n
    spec, arms = mk_qa.make(session, "sweep-%d.txt" % n, EXE, ["QKFP"])
    c = list(spec["common"])
    for flag in ("--layer-slots", "--act-slab-dtype", "--tg-tokens", "--tg-logit-margin"):
        i = c.index(flag)
        if flag in ("--tg-tokens", "--tg-logit-margin"):
            c[i + 1] = "8"
        else:
            del c[i:i + 2]
    spec["common"] = c
    t4, seq4 = g22.optimize(tokens, 41.0, 4, 0, 1.163, 1.0)
    l4 = ",".join(str(i) for i, k in enumerate(seq4) if k == 'C')

    def runner(name, why, ubatch, slots, extra, engine_on):
        a = copy.deepcopy(arms[0])
        a["name"] = name
        fl = list(a["flags"])
        fl[fl.index("--ubatch") + 1] = str(ubatch)
        fl += ["--layer-slots", str(slots), "--act-slab-dtype", "f32", "--pp-repeat", "2"] + extra
        if not engine_on:
            fl += ["--tg-tokens", "0"]
            a["env"]["LLAMA_MOE_HYBRID"] = "0"
        a["flags"] = fl
        a["env"]["LLAMA_FATTN_GQA8"] = "1"
        a["why"] = why
        return a

    stkno = {"name": "STKNO", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-mmap", "--no-op-offload"], "env": {}, "why": "stock llama-completion, the owner's flags with --no-op-offload"}
    arms_out = [
        stkno,
        runner("SP0", "stock graph path, experts on the CPU, hybrid engine ON", 512, 2, ["--pp-stock-below", "4096", "--pp-stock-offload", "0"], True),
        runner("SP0X", "stock graph path, experts on the CPU, hybrid engine OFF (LLAMA_MOE_HYBRID=0)", 512, 2, ["--pp-stock-below", "4096", "--pp-stock-offload", "0"], False),
        runner("HYC", "every layer on the CPU leg, engine ON", 2048, 2, ["--pp-cpu-layers", "all"], True),
        runner("HYCX", "every layer on the CPU leg, engine OFF", 2048, 2, ["--pp-cpu-layers", "all"], False),
        runner("HY4P", "model 4-slot plan (CPU layers %s), paced copies, engine ON" % l4, 2048, 4, ["--pp-cpu-layers", l4] + PACE, True),
        runner("HY4PX", "the same plan, engine OFF", 2048, 4, ["--pp-cpu-layers", l4] + PACE, False),
    ]
    prompt = PROMPTS.replace("\\", "/") + "/sweep-%d.txt" % n
    spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": arms_out, "control": "STKNO"})
    spec["stock_common"] = ["-m", MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", "8",
                            "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt"]
    json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
    print(session, [a["name"] for a in arms_out], "plan", l4)
