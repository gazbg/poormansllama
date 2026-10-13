"""G27 sessions (build g47 = g45 + the opt-in sampler of stage-c/g27_sampler_apply.py.txt): the hybrid engine against the stock engine under the document's SAMPLED decoding.

    python mk_g90.py            writes build/llama-cuda/specs/g90-*.json and copies them next to this file

g90-sampled-B / g90-sampled-A : prompts nat-B / nat-A, 918 generated tokens, arms STKS<s> (stock llama-completion, --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.05 --seed s, --ignore-eos so that
    every arm generates exactly 918 tokens), HBS<s> and HFS<s> (the strict and the fast hybrid recipes with --tg-sampler doc --tg-seed s), seeds 1..4 (B) / 1..3 (A), one run per arm and
    seed (the seed is the replicate: the generated text, hence the expert-cache hit rate, changes with it). Arms alternate stock / strict / fast per seed.
g90-ident : default-off identity of the sampler and segmentation patches: the strict recipe HB and the fast recipe HF (greedy, no new flag) on build g45 and on build g47, nat-B, 3 counterbalanced reps.
    Tokens must be identical and the rates within 1.5%.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g87  # noqa: E402  (its __main__ block does not run on import)
import mk_g21  # noqa: E402

ROOT = mk_g21.root
NEW = "build/llama-cuda/bin/g47"
OLD = "build/llama-cuda/bin/g45"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")
SAMPLER = ["--temp", "1.0", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.05"]


def save(name, spec):
    json.dump(spec, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, name + ".json"), "w"), indent=1)
    print(name, [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", spec["tg_tokens"])


def base_speed(session, prompt, reps):
    mk_g87.EXE = NEW
    mk_g87.speed(session, prompt, reps, ["HF", "HB"])
    spec = json.load(open(os.path.join(OUT, session + ".json")))
    spec["exe_dir"] = NEW
    return spec


def sampled(session, prompt, seeds):
    spec = base_speed(session, prompt, 1)
    stk, hf, hb = spec["arms"]
    arms = []
    for s in seeds:
        a = copy.deepcopy(stk)
        a["name"] = "STKS%d" % s
        a["flags"] = list(stk["flags"]) + SAMPLER + ["--seed", str(s)]
        a["why"] = "stock llama-completion, the owner's flags, SAMPLED like the document (top-k 20, top-p 0.95, min-p 0.05, temp 1.0), seed %d" % s
        b = copy.deepcopy(hb)
        b["name"] = "HBS%d" % s
        b["flags"] = list(hb["flags"]) + ["--tg-sampler", "doc", "--tg-seed", str(s)]
        b["why"] = "strict recipe (batch-major prefill) with the doc sampler, seed %d" % s
        f = copy.deepcopy(hf)
        f["name"] = "HFS%d" % s
        f["flags"] = list(hf["flags"]) + ["--tg-sampler", "doc", "--tg-seed", str(s)]
        f["why"] = "fast recipe (layer-major prefill, ubatch 2048) with the doc sampler, seed %d" % s
        arms += [a, b, f]
    spec["arms"] = arms
    spec["control"] = arms[0]["name"]
    spec["reps"] = 1
    spec["counterbalance"] = False
    save(session, spec)


def ident(session, prompt):
    spec = base_speed(session, prompt, 3)
    hf, hb = spec["arms"][1], spec["arms"][2]
    arms = []
    for base, tag, what in ((hb, "HB", "strict recipe (batch-major prefill)"), (hf, "HF", "fast recipe (layer-major prefill, which the G28 patch restructures)")):
        a45 = copy.deepcopy(base)
        a45["name"] = tag + "45"
        a45["exe_dir"] = OLD
        a45["why"] = what + " on build g45 (control)"
        a46 = copy.deepcopy(base)
        a46["name"] = tag + "47"
        a46["why"] = what + " on build g47 (the sampler and segmentation patches, flags absent)"
        arms += [a45, a46]
    spec["arms"] = arms
    spec["control"] = "HB45"
    spec["counterbalance"] = True
    save(session, spec)


if __name__ == "__main__":
    sampled("g90-sampled-B", "nat-B.txt", (1, 2, 3, 4))
    sampled("g90-sampled-A", "nat-A.txt", (1, 2, 3))
    ident("g90-ident", "nat-B.txt")
