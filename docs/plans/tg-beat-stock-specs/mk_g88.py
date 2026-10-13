"""G25 sessions: the PREFILL of the strict recipe. Question: how fast can a prompt be prefilled while the owner's keep-free rule (>= 16,000 MiB free) holds over the WHOLE run?

    python mk_g88.py            writes build/llama-cuda/specs/g88-*.json and copies them next to this file

The strict recipe of G24 ('HB' = G19 QKFBG) is the plain batch-major prefill at ubatch 512: no activation slab, so the footprint does not grow with the prompt except for the
lazily committed KV, but every ubatch streams all 40 expert banks (16.9 GiB over PCIe Gen4 x8, ~1.3-1.6 s) and the prefill runs at 257 tok/s (0.59x stock at 200k). The
batch-major prefill is link-bound: time ~ (prompt / ubatch) x max(bank copy time, compute time of one ubatch), so a larger ubatch divides the number of passes. The price is the
scheduler compute buffer, 937 / 1,218 / 1,780 / 3,032 MiB at ubatch 512 / 1,024 / 2,048 / 4,096 (AGENTS.md). Budget model of the whole-run minimum free memory:
    21,196.5 (after the model load) - [2-slot initial cache 136.5 + two layer slots 996 + C_eff(ubatch) + lazily committed KV 10.625 KiB per prompt token], C_eff = sched_reserve buffer + ~259 MiB
Every arm is one needle-QA run (128 generated tokens, programmatic grading, negative control) of one ubatch; nothing else differs from QKFBG (GQA8, 2-slot initial cache).
Sessions: g88-bm50k (BM512 BM1024 BM2048 BM4096), g88-bm100k (BM1024 BM2048), g88-bm200k (BM1024 BM1536 BM2048), g88-bm250k (BM768 BM1024).
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_deep  # noqa: E402
import mk_g21   # noqa: E402
import mk_qa    # noqa: E402

ROOT = mk_g21.root
EXE = "build/llama-cuda/bin/g45"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")


def save(name, spec):
    json.dump(spec, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, name + ".json"), "w"), indent=1)
    print(name, [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", spec["tg_tokens"])


def session(name, needle, ubatches):
    spec, arms = mk_qa.make(name, needle, EXE, ["QKFM"])
    arms = mk_deep.deepen(spec, arms)
    base = arms[0]
    fl = list(base["flags"])
    for flag, n in (("--mtp", 1), ("--mtp-draft", 2), ("--mtp-draft-experts-gpu", 1)):
        i = fl.index(flag)
        del fl[i:i + n]
    base["flags"] = fl
    base["env"]["LLAMA_FATTN_GQA8"] = "1"
    out = []
    for u in ubatches:
        a = copy.deepcopy(base)
        a["name"] = "BM%d" % u
        f = list(a["flags"])
        f[f.index("--ubatch") + 1] = str(u)
        a["flags"] = f
        a["why"] = "strict recipe (QKFBG) with ubatch %d: plain batch-major prefill, no activation slab, GQA8 decode attention, 2-slot initial cache" % u
        out.append(a)
    spec["arms"] = out
    spec["control"] = out[0]["name"]
    save(name, spec)


if __name__ == "__main__":
    session("g88-bm50k", "needle-50000.txt", (512, 1024, 2048, 4096))
    session("g88-bm100k", "needle-100000.txt", (1024, 2048))
    session("g88-bm200k", "needle-200000.txt", (1024, 1536, 2048))
    session("g88-bm250k", "needle-250000.txt", (768, 1024))
