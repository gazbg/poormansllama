"""G23 root-cause session, fourth step (criterion C5): the runner's CPU-only path is 30-60% slower than the stock binary's at 1,023 tokens although the per-logical-CPU load is the same (g72: 16 of 32 logical CPUs
at ~45% in both). g64 (engine threads), g65 (mmap, thread count) and g66 (explicit threadpool) were falsified. What else do the runner arms carry that the stock arm does not? Their ENVIRONMENT: the
free-memory watcher (`LLAMA_TG_FREE_WATCH_MS=20`: a thread that calls cudaMemGetInfo every 20 ms for the whole run) and `LLAMA_CUDA_INPUT_BATCH=1` (the scheduler's small input uploads are batched behind an
event instead of synchronised one by one), `LLAMA_FATTN_GQA8`. This switches them off one at a time, engine off, 1,023 tokens, prefill of the last pass.

    python mk_g78.py [EXE_DIR]     writes build/llama-cuda/specs/g78-1024.json (+ a copy next to this file); default EXE_DIR build/llama-cuda/bin/g44

Arms: STKNO (stock, owner's flags + --no-op-offload), SP0X (runner stock path, experts on the CPU, engine off), SPNW (no free-memory watcher), SPNB (no input batching), SPBARE (neither, no GQA8 flag).
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
EXE = sys.argv[1] if len(sys.argv) > 1 else "build/llama-cuda/bin/g44"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")
PROMPTS = mk_g46.PROMPTS
MODEL = mk_g21.MODEL

session = "g78-1024"
spec, arms = mk_qa.make(session, "sweep-1024.txt", EXE, ["QKFP"])
c = list(spec["common"])
for flag in ("--layer-slots", "--act-slab-dtype", "--tg-tokens", "--tg-logit-margin"):
    i = c.index(flag)
    if flag in ("--tg-tokens", "--tg-logit-margin"):
        c[i + 1] = "8"
    else:
        del c[i:i + 2]
spec["common"] = c


def runner(name, why, drop):
    a = copy.deepcopy(arms[0])
    a["name"] = name
    fl = list(a["flags"])
    fl[fl.index("--ubatch") + 1] = "512"
    fl += ["--layer-slots", "2", "--act-slab-dtype", "f32", "--pp-repeat", "2", "--pp-stock-below", "4096", "--pp-stock-offload", "0", "--tg-tokens", "0"]
    a["flags"] = fl
    a["env"]["LLAMA_MOE_HYBRID"] = "0"
    a["env"]["LLAMA_FATTN_GQA8"] = "1"
    for k in drop:
        a["env"].pop(k, None)
    a["why"] = why
    return a


stkno = {"name": "STKNO", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-mmap", "--no-op-offload"], "env": {}, "why": "stock llama-completion, owner's flags + --no-op-offload"}
arms_out = [
    stkno,
    runner("SP0X", "runner, stock graph path, experts on the CPU, engine off, the QKFP environment", []),
    runner("SPNW", "the same without the free-memory watcher (LLAMA_TG_FREE_WATCH_MS)", ["LLAMA_TG_FREE_WATCH_MS"]),
    runner("SPNB", "the same without input batching (LLAMA_CUDA_INPUT_BATCH)", ["LLAMA_CUDA_INPUT_BATCH"]),
    runner("SPBARE", "the same without either and without LLAMA_FATTN_GQA8", ["LLAMA_TG_FREE_WATCH_MS", "LLAMA_CUDA_INPUT_BATCH", "LLAMA_FATTN_GQA8"]),
]
prompt = PROMPTS.replace("\\", "/") + "/sweep-1024.txt"
spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": arms_out, "control": "STKNO"})
spec["stock_common"] = ["-m", MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", "8",
                        "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt"]
json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
print(session, [a["name"] for a in arms_out])
