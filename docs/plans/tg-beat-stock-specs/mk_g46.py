"""G22: where does the layer-major prefill beat the stock prefill? Prefill time against the prompt length N, stock llama-completion (the owner's flags) against the layer-major prefill on a
dedicated GPU (2 layer slots with the async feed, f32 slab, ubatch 2048), both reading the same prompt file, 8 generated tokens (only the prefill is of interest).

    python mk_g46.py prompts     writes build/llama-cuda/gplr07-qa-prompts/sweep-<N>.txt (N = 3 .. 2048 tokens; the real token count is printed by both engines)
    python mk_g46.py specs       writes build/llama-cuda/specs/g46-<N>.json (arms STK and LM2, 2 counterbalanced reps) and copies them next to this file
"""
import copy
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402
import mk_qa   # noqa: E402

ROOT = mk_g21.root
EXE = "build/llama-cuda/bin/g40"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")
PROMPTS = os.path.join(ROOT, "build", "llama-cuda", "gplr07-qa-prompts")
NS = (3, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
TINY = {3: "Please continue.", 8: "Okay, please continue with the next step now."}
MODEL = mk_g21.MODEL


def make_prompts():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    for n in NS:
        path = os.path.join(PROMPTS, "sweep-%d.txt" % n)
        if n in TINY:
            open(path, "w", encoding="utf-8", newline="").write(TINY[n])
            print(path, "(handwritten)")
            continue
        gen = os.path.join(ROOT, "tools", "semantic-qa", "gen-needle-prompt.py")
        tok = os.path.join(ROOT, EXE.replace("/", os.sep), "llama-tokenize.exe")
        r = subprocess.run([sys.executable, gen, "--target-tokens", str(n), "--calibrate", tok, "--model", MODEL, "--out", path], capture_output=True, text=True, env=env)
        print(path, r.stdout.strip().splitlines()[0] if r.stdout else r.stderr[-200:])
        # the prompt must not end in a newline: llama-completion -f strips one, the runner does not
        data = open(path, "rb").read()
        if data.endswith(b"\n"):
            open(path, "wb").write(data.rstrip(b"\n"))


def make_specs():
    for n in NS:
        session = "g46-%d" % n
        spec, arms = mk_qa.make(session, "sweep-%d.txt" % n, EXE, ["QKFP"])
        c = list(spec["common"])
        for flag in ("--layer-slots", "--act-slab-dtype", "--tg-tokens", "--tg-logit-margin"):
            i = c.index(flag)
            if flag in ("--tg-tokens", "--tg-logit-margin"):
                c[i + 1] = "8"
            else:
                del c[i:i + 2]
        spec["common"] = c
        lm = copy.deepcopy(arms[0])
        lm["name"] = "LM2"
        fl = list(lm["flags"])
        fl[fl.index("--ubatch") + 1] = "2048"
        fl += ["--layer-slots", "2", "--act-slab-dtype", "f32"]
        lm["flags"] = fl
        lm["env"]["LLAMA_FATTN_GQA8"] = "1"
        lm["why"] = "layer-major prefill on a dedicated GPU: ubatch 2048, two layer slots with the async feed, f32 slab"
        prompt = PROMPTS.replace("\\", "/") + "/sweep-%d.txt" % n
        stk = {"name": "STK", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-mmap"], "env": {}, "why": "stock llama-completion, the owner's document configuration"}
        spec.update({"qa": False, "tg_tokens": 8, "reps": 2, "interleave": True, "counterbalance": True, "arms": [stk, lm], "control": "STK"})
        spec["stock_common"] = ["-m", MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", "8",
                                "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt"]
        json.dump(spec, open(os.path.join(OUT, session + ".json"), "w"), indent=1)
        json.dump(spec, open(os.path.join(HERE, session + ".json"), "w"), indent=1)
        print(session, [a["name"] for a in spec["arms"]])


if __name__ == "__main__":
    {"prompts": make_prompts, "specs": make_specs}[sys.argv[1]]()
