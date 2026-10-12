"""G23 (the G22 prototype, stages A, B, C): session specs for build g41. Contract: docs/plans/tg-beat-stock-8gb-262k.md, "G23 CONTRACT".

    python mk_g54.py [EXE_DIR] [CONTROL_DIR]     writes build/llama-cuda/specs/<session>.json and copies next to this file
                                                 default EXE_DIR build/llama-cuda/bin/g41, CONTROL_DIR build/llama-cuda/bin/g40

Sessions (prompts sweep-<N>.txt of g46; 8 generated tokens; `--pp-repeat 2`, so the last pass is the warm one; ctx 262,144, q8 KV, lazy KV, GQA8, keep-free auto slots):
    g55-smoke-<N>   function and exactness on any power state: LM40 (the g40 control), LM41 (g41 without a new flag), BPA / BPB (paced 16 MiB x 2 / 4 MiB x 3),
                    SP1 / SP0 (stock graph path, op offload on / off), HYC (every layer on the CPU leg), HY2 (the model's 2-slot plan), HY2P (the plan + paced copies)
    g56-stock-<N>   stage A against the stock binary, same session: STK / STKNO / STKU (llama-completion) against SP1 (ubatch 512, offload on), SP0 (offload off), SPU (ubatch 2048)
    g57-pace-<N>    stage B on the all-GPU prefill: LM40, LM41, BPA, BPB (B1-B4)
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
CONTROL = sys.argv[2] if len(sys.argv) > 2 else "build/llama-cuda/bin/g40"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")
PROMPTS = mk_g46.PROMPTS
MODEL = mk_g21.MODEL

_spec = importlib.util.spec_from_file_location("g22model", os.path.join(ROOT, "tools", "g22-model.py"))
g22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g22)

# sweep prompt file number -> real token count (g46: sweep-3 / -8 / -16 / -32 hold 3 / 10 / 14 / 29 tokens)
TOKENS = {3: 3, 8: 10, 16: 14, 32: 29, 64: 63, 128: 125, 256: 256, 512: 511, 1024: 1023, 2048: 2047}


def plan_for(n_tokens, slots):
    t, seq = g22.optimize(n_tokens, 41.0, slots, 0, 1.163, 1.0)
    return ",".join(str(i) for i, k in enumerate(seq) if k == 'C'), t, seq.count('G')


def base_spec(session, prompt_name):
    spec, arms = mk_qa.make(session, prompt_name, EXE, ["QKFP"])
    c = list(spec["common"])
    for flag in ("--layer-slots", "--act-slab-dtype", "--tg-tokens", "--tg-logit-margin"):
        i = c.index(flag)
        if flag in ("--tg-tokens", "--tg-logit-margin"):
            c[i + 1] = "8"
        else:
            del c[i:i + 2]
    spec["common"] = c
    return spec, arms[0]


def runner_arm(base, name, why, slots=2, ubatch=2048, extra=(), exe_dir=None, repeat=2, f32=True):
    a = copy.deepcopy(base)
    a["name"] = name
    fl = list(a["flags"])
    fl[fl.index("--ubatch") + 1] = str(ubatch)
    fl += ["--layer-slots", str(slots)]
    if f32:
        fl += ["--act-slab-dtype", "f32"]
    if repeat > 1:
        fl += ["--pp-repeat", str(repeat)]
    fl += list(extra)
    a["flags"] = fl
    a["env"]["LLAMA_FATTN_GQA8"] = "1"
    if exe_dir:
        a["exe_dir"] = exe_dir
    a["why"] = why
    return a


def stock_arms(prompt, n):
    stk = {"name": "STK", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-mmap"], "env": {}, "why": "stock llama-completion, the owner's document configuration (op offload on, ubatch 512)"}
    stkno = copy.deepcopy(stk)
    stkno["name"] = "STKNO"
    stkno["flags"] = stk["flags"] + ["--no-op-offload"]
    stkno["why"] = "the same with --no-op-offload: the expert FFN of the CPU layers runs on the CPU"
    stku = copy.deepcopy(stk)
    stku["name"] = "STKU"
    stku["flags"] = stk["flags"] + ["-ub", "2048"]
    stku["why"] = "the owner's flags with ubatch 2048"
    return [stk, stkno, stku]


def stock_common(prompt):
    return ["-m", MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", "8",
            "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt"]


def write(session, spec):
    json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
    print(session, [a["name"] for a in spec["arms"]])


# ----------------------------------------------------------------------------- g55: smoke
for n_file, tokens in ((64, 63), (256, 256)):
    session = "g55-smoke-%d" % tokens
    spec, base = base_spec(session, "sweep-%d.txt" % n_file)
    l2, t2, g2 = plan_for(tokens, 2)
    arms = [
        runner_arm(base, "LM40", "g40 control: all-GPU layer-major prefill, two slots, warm pass", exe_dir=CONTROL),
        runner_arm(base, "LM41", "g41 without any new flag (identity with g40)"),
        runner_arm(base, "BPA", "paced bank copies 16 MiB x 2", extra=["--pp-copy-piece-mib", "16", "--pp-copy-window", "2"]),
        runner_arm(base, "BPB", "paced bank copies 4 MiB x 3", extra=["--pp-copy-piece-mib", "4", "--pp-copy-window", "3"]),
        runner_arm(base, "SP1", "stock graph path, op offload on, ubatch 512", ubatch=512, extra=["--pp-stock-below", "4096"]),
        runner_arm(base, "SP0", "stock graph path, op offload off, ubatch 512", ubatch=512, extra=["--pp-stock-below", "4096", "--pp-stock-offload", "0"]),
        runner_arm(base, "HYC", "every layer on the CPU leg (no bank streamed)", extra=["--pp-cpu-layers", "all"]),
        runner_arm(base, "HY2", "model plan, 2 slots: %d GPU layers, predicted %.0f ms; CPU layers %s" % (g2, t2, l2), extra=["--pp-cpu-layers", l2]),
        runner_arm(base, "HY2P", "the same plan with paced copies 16 MiB x 2", extra=["--pp-cpu-layers", l2, "--pp-copy-piece-mib", "16", "--pp-copy-window", "2"]),
    ]
    prompt = PROMPTS.replace("\\", "/") + "/sweep-%d.txt" % n_file
    spec.update({"qa": False, "tg_tokens": 8, "reps": 1, "interleave": True, "counterbalance": False, "arms": arms, "control": "LM40"})
    write(session, spec)

# ----------------------------------------------------------------------------- g56: stage A against the stock binary
for n_file, tokens in sorted(TOKENS.items()):
    session = "g56-stock-%d" % tokens
    spec, base = base_spec(session, "sweep-%d.txt" % n_file)
    prompt = PROMPTS.replace("\\", "/") + "/sweep-%d.txt" % n_file
    arms = stock_arms(prompt, tokens) + [
        runner_arm(base, "SP1", "stock graph path in the runner, op offload on, ubatch 512", ubatch=512, extra=["--pp-stock-below", "4096"]),
        runner_arm(base, "SP0", "stock graph path in the runner, op offload off, ubatch 512", ubatch=512, extra=["--pp-stock-below", "4096", "--pp-stock-offload", "0"]),
        runner_arm(base, "SPU", "stock graph path in the runner, op offload on, ubatch 2048", ubatch=2048, extra=["--pp-stock-below", "4096"]),
    ]
    spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": arms, "control": "STK"})
    spec["stock_common"] = stock_common(prompt)
    write(session, spec)

# ----------------------------------------------------------------------------- g57: stage B on the all-GPU prefill
for n_file, tokens in ((64, 63), (256, 256), (1024, 1023)):
    session = "g57-pace-%d" % tokens
    spec, base = base_spec(session, "sweep-%d.txt" % n_file)
    arms = [
        runner_arm(base, "LM40", "g40 control: all-GPU layer-major prefill, warm pass", exe_dir=CONTROL),
        runner_arm(base, "LM41", "g41 without any new flag"),
        runner_arm(base, "BPA", "paced bank copies 16 MiB x 2", extra=["--pp-copy-piece-mib", "16", "--pp-copy-window", "2"]),
        runner_arm(base, "BPB", "paced bank copies 4 MiB x 3", extra=["--pp-copy-piece-mib", "4", "--pp-copy-window", "3"]),
    ]
    spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": arms, "control": "LM40"})
    write(session, spec)
