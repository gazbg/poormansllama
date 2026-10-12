"""The FINAL configuration on the dedicated GPU (G21 decisions: two layer slots with the async feed, ubatch 2048, f32 slab, the default 14-slot initial cache, the GQA8 decode attention),
build g40. The keep-free rule is read from the decode-phase line of the runner's free-watch.

    python mk_g47.py     writes build/llama-cuda/specs/g47-*.json and copies them next to this file

QA (arm QKFE, 128 generated tokens, needle prompts):   g47-qa4096 g47-qa24000 g47-qa50000 g47-qa100000 g47-qa200000 g47-qa250000
Headline against stock (arms STK, DKFE, 512 tokens, counterbalanced pairs, the code-continuation prompt after the haystack):   g47-fin50000 g47-fin100000 g47-fin200000
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
EXE = "build/llama-cuda/bin/g40"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")


def save(name, spec):
    json.dump(spec, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, name + ".json"), "w"), indent=1)
    print(name, [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", spec["tg_tokens"])


def final_arm(base, name):
    """QKFP (layer-major, ubatch 512, f32 slab, two layer slots, 14-slot initial cache) with ubatch 2048 and the GQA8 decode attention."""
    a = copy.deepcopy(base)
    a["name"] = name
    fl = list(a["flags"])
    fl[fl.index("--ubatch") + 1] = "2048"
    a["flags"] = fl
    a["env"]["LLAMA_FATTN_GQA8"] = "1"
    a["why"] = "final configuration: layer-major prefill ubatch 2048, two layer slots with the async feed, f32 slab, keep-free hybrid decode, GQA8 decode attention"
    return a


for n, needle in ((4096, "anchor-4096.txt"), (24000, "anchor-24000.txt"), (50000, "needle-50000.txt"), (100000, "needle-100000.txt"), (200000, "needle-200000.txt"), (250000, "needle-250000.txt")):
    session = "g47-qa%d" % n
    spec, arms = mk_qa.make(session, needle, EXE, ["QKFP"])
    spec["arms"] = [final_arm(arms[0], "QKFE")]
    spec["control"] = "QKFE"
    save(session, spec)

for n, reps in ((50000, 3), (100000, 3), (200000, 2)):
    session = "g47-fin%d" % n
    prompt = "deepcode-%d.txt" % n
    spec = mk_deep.make_speed(session, prompt, reps, EXE, ["STK"])            # the stock arm, its stock_common, 512 tokens, the counterbalanced pairs
    base_spec, base_arms = mk_qa.make(session, prompt, EXE, ["QKFP"])         # the QKFP shape; its shared flags carry the f32 slab and the two layer slots
    common = list(base_spec["common"])
    for flag in ("--tg-tokens", "--tg-logit-margin"):
        common[common.index(flag) + 1] = "512"
    spec["common"] = common
    dk = final_arm(base_arms[0], "DKFE")
    dk["why"] += " (512 tokens)"
    spec["arms"] = spec["arms"] + [dk]
    save(session, spec)
