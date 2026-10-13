"""G28 sessions (build g47: g45 + the sampler patch + the segmented layer-major prefill, both opt-in): the fast prefill under the strict keep-free rule.

    python mk_g91.py            writes build/llama-cuda/specs/g91-*.json and copies them next to this file

g91-ident     : needle-50000, ubatch 2048, f32 slab, two layer slots with the async feed (the G21 shape): control LM (no new flag) against SEG8 (--pp-segment-tokens 8192) and SEG16 (16384).
                QA (128 tokens, negative control). The generated ids must be identical (the harness reports the first divergence against the control).
g91-pp-<N>    : the strict-rule recipes per fill (budget model of the G28 contract: free = 21,196.5 - [136.5 + 996 + C_eff(ubatch) + 16 KiB x S + 10.625 KiB x N]); every arm = one needle-QA run:
                50k:  SF2048S16 (ubatch 2048, S 16384), SF2048S8 (S 8192), SF512S16 (ubatch 512)
                100k: SF2048S16, SF2048S32 (S 32768)
                200k: SF1024S16 (ubatch 1024), SF512S16
                250k: SF512S8, SF512S4 (S 4096), SF512S16
g91-tg-50k / g91-tg-200k : natural prompts (natdeepv2-50000 / natdeep-200000), 918 tokens, arms STK, HB (the G24 strict recipe), SF (50k: ubatch 2048 S 16384; 200k: ubatch 1024 S 16384): the decode after the
                fast strict prefill against the decode after the batch-major prefill and against stock.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402
import mk_g87  # noqa: E402
import mk_qa   # noqa: E402

ROOT = mk_g21.root
NEW = "build/llama-cuda/bin/g47"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")
mk_g87.EXE = NEW


def save(name, spec):
    json.dump(spec, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, name + ".json"), "w"), indent=1)
    print(name, [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", spec["tg_tokens"])


def lm_arm(base, name, ubatch, seg, strict_cache=True):
    """QKFP (layer-major, f32 slab, two layer slots with the async feed, keep-free hybrid decode) with the ubatch and the segment size; GQA8; the 2-slot initial cache of the strict recipes."""
    a = copy.deepcopy(base)
    a["name"] = name
    fl = list(a["flags"])
    fl[fl.index("--ubatch") + 1] = str(ubatch)
    if seg:
        fl += ["--pp-segment-tokens", str(seg)]
    a["flags"] = fl
    a["env"]["LLAMA_FATTN_GQA8"] = "1"
    if strict_cache:
        a["env"]["LLAMA_MOE_HYBRID"] = "2"
    a["why"] = "layer-major prefill, ubatch %d, f32 slab, two layer slots%s, keep-free hybrid decode, GQA8" % (ubatch, (", segments of %d tokens" % seg) if seg else ", NO segmentation (control)")
    return a


def qa_session(session, needle, arm_specs):
    spec, arms = mk_qa.make(session, needle, NEW, ["QKFP"])
    spec["arms"] = [lm_arm(arms[0], n, u, s, sc) for (n, u, s, sc) in arm_specs]
    spec["control"] = spec["arms"][0]["name"]
    save(session, spec)


def tg_session(session, prompt, reps, ubatch, seg):
    mk_g87.speed(session, prompt, reps, ["HB"])
    spec = json.load(open(os.path.join(OUT, session + ".json")))
    spec["exe_dir"] = NEW
    base_spec, base_arms = mk_qa.make(session, prompt, NEW, ["QKFP"])
    sf = lm_arm(base_arms[0], "SF", ubatch, seg)
    spec["arms"] = list(spec["arms"]) + [sf]
    spec["control"] = "STK"
    save(session, spec)


if __name__ == "__main__":
    qa_session("g91-ident", "needle-50000.txt", [("LM", 2048, 0, False), ("SEG8", 2048, 8192, False), ("SEG16", 2048, 16384, False)])
    qa_session("g91-pp-50k", "needle-50000.txt", [("SF2048S16", 2048, 16384, True), ("SF2048S8", 2048, 8192, True), ("SF512S16", 512, 16384, True)])
    qa_session("g91-pp-100k", "needle-100000.txt", [("SF2048S16", 2048, 16384, True), ("SF2048S32", 2048, 32768, True)])
    qa_session("g91-pp-200k", "needle-200000.txt", [("SF1024S16", 1024, 16384, True), ("SF512S16", 512, 16384, True)])
    qa_session("g91-pp-250k", "needle-250000.txt", [("SF512S8", 512, 8192, True), ("SF512S4", 512, 4096, True), ("SF512S16", 512, 16384, True)])
    tg_session("g91-tg-50k", "natdeepv2-50000.txt", 3, 2048, 16384)
    tg_session("g91-tg-200k", "natdeep-200000.txt", 2, 1024, 16384)
