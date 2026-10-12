"""G23 root-cause session, second step (criterion C5): g64 falsified H1 (the hybrid engine's threads are not why the runner's CPU path is 30-60% slower than the stock binary's from ~500 tokens: engine ON and
OFF give the same 6.0-6.4 s at 1,023 tokens against 4.0 s). H2: the runner maps the model (file-backed pages, page-locked by cudaHostRegister), the owner's stock flags read it into ordinary memory
(`--no-mmap`). Test: stock WITH the mapping (no --no-mmap) and the runner WITHOUT it (--no-mmap), engine off, 1,023 tokens, prefill only.

    python mk_g65.py [EXE_DIR]     writes build/llama-cuda/specs/g65-1024.json (+ a copy next to this file); default EXE_DIR build/llama-cuda/bin/g43

Arms: STKNO (stock, owner's flags + --no-op-offload), STKNOM (the same, memory-mapped), SP0X (runner, stock graph path, experts on the CPU, engine off), SP0N (the same with --no-mmap),
      SP0T (the same as SP0X with 24 threads: is the pool simply too small next to the stock run?)
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
EXE = sys.argv[1] if len(sys.argv) > 1 else "build/llama-cuda/bin/g43"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")
PROMPTS = mk_g46.PROMPTS
MODEL = mk_g21.MODEL

session = "g65-1024"
spec, arms = mk_qa.make(session, "sweep-1024.txt", EXE, ["QKFP"])
c = list(spec["common"])
for flag in ("--layer-slots", "--act-slab-dtype", "--tg-tokens", "--tg-logit-margin"):
    i = c.index(flag)
    if flag in ("--tg-tokens", "--tg-logit-margin"):
        c[i + 1] = "8"
    else:
        del c[i:i + 2]
spec["common"] = c


def runner(name, why, extra):
    a = copy.deepcopy(arms[0])
    a["name"] = name
    fl = list(a["flags"])
    fl[fl.index("--ubatch") + 1] = "512"
    fl += ["--layer-slots", "2", "--act-slab-dtype", "f32", "--pp-repeat", "2", "--pp-stock-below", "4096", "--pp-stock-offload", "0", "--tg-tokens", "0"] + extra
    a["flags"] = fl
    a["env"]["LLAMA_MOE_HYBRID"] = "0"
    a["env"]["LLAMA_FATTN_GQA8"] = "1"
    a["why"] = why
    return a


stkno = {"name": "STKNO", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-mmap", "--no-op-offload"], "env": {}, "why": "stock llama-completion, owner's flags + --no-op-offload"}
stknom = {"name": "STKNOM", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-op-offload"], "env": {}, "why": "the same with the model memory-mapped (no --no-mmap)"}
arms_out = [
    stkno,
    stknom,
    runner("SP0X", "runner, stock graph path, experts on the CPU, engine off", []),
    runner("SP0N", "the same with --no-mmap", ["--no-mmap"]),
    runner("SP0T", "the same as SP0X with 24 CPU threads", ["--cpu-threads", "24"]),
]
prompt = PROMPTS.replace("\\", "/") + "/sweep-1024.txt"
spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": arms_out, "control": "STKNO"})
spec["stock_common"] = ["-m", MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", "8",
                        "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt"]
json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
print(session, [a["name"] for a in arms_out])
