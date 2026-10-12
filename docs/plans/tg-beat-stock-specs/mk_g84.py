"""G23 root-cause validation for ledger row 63 and the cold start (build g45: `--gpu-keepalive 1` = a one-block spin kernel on a private stream, so that the PCIe link stays at Gen4).

Found before this session (standalone, tools/pcie-state-bench; nvidia-smi in g81; nsys in g80): after ~10 s of GPU utilisation under ~10% the PCIe link of the laptop GPU falls Gen4 -> Gen2 -> Gen1 (a 32 MiB
host-to-device copy 2.33 -> 9.15 -> 18.3 ms) and needs ~0.25 s of traffic to climb back; the runner's CPU-only prefill (80 round trips of 32 MiB per pass) runs into it in its second pass.

    python mk_g84.py [EXE_DIR]     writes build/llama-cuda/specs/g84-<N>.json (CPU-only paths against stock) and g85-<N>.json (cold start of the layer-major and the hybrid prefill), + copies next to this file

g84-<N> (N = 511, 1023; engine off, no generation, `--pp-repeat 2`, 2 counterbalanced reps): STKNO (stock, owner's flags + --no-op-offload), SP0X (runner stock path, experts on the CPU), SP0K (the same
with --gpu-keepalive 1), HYC (every layer on the CPU leg), HYCK (the same with --gpu-keepalive 1).
g85-<N> (N = 63, 256, 1023; engine on as in the final configuration, `--pp-repeat 2`; the table tool prints BOTH passes): LM2P (paced all-GPU), LM2PK (the same + keepalive), HY4P (paced hybrid plan), HY4PK (+ keepalive).
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
EXE = sys.argv[1] if len(sys.argv) > 1 else "build/llama-cuda/bin/g45"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")
PROMPTS = mk_g46.PROMPTS
MODEL = mk_g21.MODEL
PACE = ["--pp-copy-piece-mib", "16", "--pp-copy-window", "2"]
KA = ["--gpu-keepalive", "1"]

_spec = importlib.util.spec_from_file_location("g22model", os.path.join(ROOT, "tools", "g22-model.py"))
g22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g22)


def base(session, n):
    spec, arms = mk_qa.make(session, "sweep-%d.txt" % n, EXE, ["QKFP"])
    c = list(spec["common"])
    for flag in ("--layer-slots", "--act-slab-dtype", "--tg-tokens", "--tg-logit-margin"):
        i = c.index(flag)
        if flag in ("--tg-tokens", "--tg-logit-margin"):
            c[i + 1] = "8"
        else:
            del c[i:i + 2]
    spec["common"] = c
    return spec, arms[0]


def runner(b, name, why, ubatch, slots, extra, engine_on=True):
    a = copy.deepcopy(b)
    a["name"] = name
    fl = list(a["flags"])
    fl[fl.index("--ubatch") + 1] = str(ubatch)
    fl += ["--layer-slots", str(slots), "--act-slab-dtype", "f32", "--pp-repeat", "2"] + list(extra)
    if not engine_on:
        fl += ["--tg-tokens", "0"]
        a["env"]["LLAMA_MOE_HYBRID"] = "0"
    a["flags"] = fl
    a["env"]["LLAMA_FATTN_GQA8"] = "1"
    a["why"] = why
    return a


def write(session, spec):
    json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
    print(session, [a["name"] for a in spec["arms"]])


for n_file, tokens in ((512, 511), (1024, 1023)):
    session = "g84-%d" % tokens
    spec, b = base(session, n_file)
    sp0 = ["--pp-stock-below", "4096", "--pp-stock-offload", "0"]
    stkno = {"name": "STKNO", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-mmap", "--no-op-offload"], "env": {}, "why": "stock llama-completion, owner's flags + --no-op-offload"}
    arms = [stkno,
            runner(b, "SP0X", "runner stock path, experts on the CPU, engine off", 512, 2, sp0, False),
            runner(b, "SP0K", "the same with --gpu-keepalive 1", 512, 2, sp0 + KA, False),
            runner(b, "HYC", "every layer on the CPU leg, engine off", 2048, 2, ["--pp-cpu-layers", "all"], False),
            runner(b, "HYCK", "the same with --gpu-keepalive 1", 2048, 2, ["--pp-cpu-layers", "all"] + KA, False)]
    prompt = PROMPTS.replace("\\", "/") + "/sweep-%d.txt" % n_file
    spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": arms, "control": "STKNO"})
    spec["stock_common"] = ["-m", MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", "8",
                            "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt"]
    write(session, spec)

for n_file, tokens in ((64, 63), (256, 256), (1024, 1023)):
    session = "g85-%d" % tokens
    spec, b = base(session, n_file)
    t4, seq4 = g22.optimize(tokens, 41.0, 4, 0, 1.163, 1.0)
    l4 = ",".join(str(i) for i, k in enumerate(seq4) if k == 'C')
    arms = [runner(b, "LM2P", "paced all-GPU layer-major prefill", 2048, 2, PACE),
            runner(b, "LM2PK", "the same with --gpu-keepalive 1", 2048, 2, PACE + KA),
            runner(b, "HY4P", "paced hybrid plan, 4 slots (CPU layers %s)" % l4, 2048, 4, ["--pp-cpu-layers", l4] + PACE),
            runner(b, "HY4PK", "the same with --gpu-keepalive 1", 2048, 4, ["--pp-cpu-layers", l4] + PACE + KA)]
    spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": arms, "control": "LM2P"})
    write(session, spec)
