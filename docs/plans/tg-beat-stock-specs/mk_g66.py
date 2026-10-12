"""G23 root-cause session, third step (criterion C5): H5 = the runner's CPU matmuls run on OpenMP regions, the stock tools attach an explicit persistent threadpool (power throttling off, polling workers).
g64 falsified H1 (hybrid engine threads), g65 falsified H2 (mmap) and H3 (thread count). Build g44 adds `--pp-threadpool 1` (explicit pool for the prefill, freed before the generation) and fix 3
(paced preload through the worker, unbounded request queue, smallest-layer-first worker).

    python mk_g66.py [EXE_DIR]     writes build/llama-cuda/specs/g66-<N>.json (+ copies next to this file); default EXE_DIR build/llama-cuda/bin/g44

Arms (prompt sweep-<N>.txt, `--pp-repeat 2`, 2 counterbalanced reps; hybrid engine ON everywhere, as in the final configuration):
    STKNO  stock llama-completion, owner's flags + --no-op-offload
    SP0    runner stock graph path, experts on the CPU           SP0P  the same with --pp-threadpool 1
    HYC    runner, every layer on the CPU leg                    HYCP  the same with --pp-threadpool 1
    HY4P   the v3 model's 4-slot plan, paced copies              HY4PP the same with --pp-threadpool 1
    HY4Q   the v4 (paced) model's 4-slot plan, paced copies, --pp-threadpool 1     (tools/g23-model.py, c = 33 ms, no preload blocking)
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


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g22 = load("g22model", "g22-model.py")
g23 = load("g23model", "g23-model.py")

for n, tokens in ((256, 256), (512, 511), (1024, 1023)):
    session = "g66-%d" % n
    spec, arms = mk_qa.make(session, "sweep-%d.txt" % n, EXE, ["QKFP"])
    c = list(spec["common"])
    for flag in ("--layer-slots", "--act-slab-dtype", "--tg-tokens", "--tg-logit-margin"):
        i = c.index(flag)
        if flag in ("--tg-tokens", "--tg-logit-margin"):
            c[i + 1] = "8"
        else:
            del c[i:i + 2]
    spec["common"] = c
    t3, seq3 = g22.optimize(tokens, 41.0, 4, 0, 1.163, 1.0)
    l3 = ",".join(str(i) for i, k in enumerate(seq3) if k == 'C')
    t4, seq4 = g23.optimize(tokens, 33.0, 4, 0, 1.163, 1.0)
    l4 = ",".join(str(i) for i, k in enumerate(seq4) if k == 'C')

    def runner(name, why, ubatch, slots, extra):
        a = copy.deepcopy(arms[0])
        a["name"] = name
        fl = list(a["flags"])
        fl[fl.index("--ubatch") + 1] = str(ubatch)
        fl += ["--layer-slots", str(slots), "--act-slab-dtype", "f32", "--pp-repeat", "2"] + extra
        a["flags"] = fl
        a["env"]["LLAMA_FATTN_GQA8"] = "1"
        a["why"] = why
        return a

    stkno = {"name": "STKNO", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-mmap", "--no-op-offload"], "env": {}, "why": "stock llama-completion, the owner's flags with --no-op-offload"}
    sp0 = ["--pp-stock-below", "4096", "--pp-stock-offload", "0"]
    pool = ["--pp-threadpool", "1"]
    arms_out = [
        stkno,
        runner("SP0", "stock graph path, experts on the CPU", 512, 2, sp0),
        runner("SP0P", "the same with the explicit prefill threadpool", 512, 2, sp0 + pool),
        runner("HYC", "every layer on the CPU leg", 2048, 2, ["--pp-cpu-layers", "all"]),
        runner("HYCP", "the same with the explicit prefill threadpool", 2048, 2, ["--pp-cpu-layers", "all"] + pool),
        runner("HY4P", "v3 model 4-slot plan (CPU layers %s), paced copies; predicted %.0f ms" % (l3, t3), 2048, 4, ["--pp-cpu-layers", l3] + PACE),
        runner("HY4PP", "the same with the explicit prefill threadpool", 2048, 4, ["--pp-cpu-layers", l3] + PACE + pool),
        runner("HY4Q", "v4 (paced) model 4-slot plan (CPU layers %s), paced copies, pool; predicted %.0f ms" % (l4, t4), 2048, 4, ["--pp-cpu-layers", l4] + PACE + pool),
    ]
    prompt = PROMPTS.replace("\\", "/") + "/sweep-%d.txt" % n
    spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": arms_out, "control": "STKNO"})
    spec["stock_common"] = ["-m", MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", "8",
                            "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt"]
    json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
    print(session, "v3 plan", l3, "| v4 plan", l4, "(%d GPU layers, predicted %.0f ms)" % (seq4.count('G'), t4))
