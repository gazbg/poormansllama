"""Specs for the g43 sessions: the build of record g38 (g36 runner + the GQA8 decode attention with its final shape checks), run in this order.

    python mk_g43.py          writes build/llama-cuda/specs/g43-*.json and copies them next to this file

QA with the flag on (QKFPG = the final plain configuration, layer-major prefill, QKFDG = its deep variant, QKFBG = plain batch-major prefill):
    g43-qa4096, g43-qa24000 (QKFPG); g43-qad50000, g43-qad100000, g43-qad200000 (QKFDG); g43-qab200000 (QKFBG)
Speed against stock, certified headline of the final configuration at depth (STK vs DKFDG): g43-fin50000, g43-fin100000, g43-fin200000
Diagnostic, f16 KV (tensor-core GQA kernel) against the q8 vector kernel and GQA8 at 100k: g43-kv16-100k (DKFD, DKFDG, DKFD16)
MTP on the fixed runner at a short fill, with the old padding as a third arm: g43-kfm-code, g43-kfm-prose (KFP, KFM, KFMPAD)
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_deep   # noqa: E402
import mk_final  # noqa: E402
import mk_g21    # noqa: E402
import mk_qa     # noqa: E402

ROOT = mk_g21.root
EXE = "build/llama-cuda/bin/g38"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")


def save(name, spec):
    json.dump(spec, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, name + ".json"), "w"), indent=1)
    print(name, [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", spec["tg_tokens"])


def qa_plain(session, needle):
    spec, arms = mk_qa.make(session, needle, EXE, ["QKFP"])
    arms[0]["name"] = "QKFPG"
    arms[0]["env"]["LLAMA_FATTN_GQA8"] = "1"
    arms[0]["why"] += " [GQA8 decode attention]"
    spec["control"] = "QKFPG"
    spec["arms"] = arms
    save(session, spec)


def qa_deep(session, needle):
    spec, arms = mk_qa.make(session, needle, EXE, ["QKFP"])
    arms = mk_deep.deepen(spec, arms)
    arms[0]["name"] = "QKFDG"
    arms[0]["env"]["LLAMA_FATTN_GQA8"] = "1"
    arms[0]["why"] += " [GQA8 decode attention]"
    spec["control"] = "QKFDG"
    spec["arms"] = arms
    save(session, spec)


def qa_batch_major(session, needle):
    spec, arms = mk_qa.make(session, needle, EXE, ["QKFM"])
    arms = mk_deep.deepen(spec, arms)
    a = arms[0]
    fl = list(a["flags"])
    for flag, n in (("--mtp", 1), ("--mtp-draft", 2), ("--mtp-draft-experts-gpu", 1)):
        i = fl.index(flag)
        del fl[i:i + n]
    a["flags"] = fl
    a["name"] = "QKFBG"
    a["env"]["LLAMA_FATTN_GQA8"] = "1"
    a["why"] = "plain batch-major prefill (no act slab, no MTP), GQA8 decode attention, 2-slot initial cache: the candidate that holds the keep-free rule at 200k"
    spec["control"] = "QKFBG"
    spec["arms"] = [a]
    save(session, spec)


qa_plain("g43-qa4096", "anchor-4096.txt")
qa_plain("g43-qa24000", "anchor-24000.txt")
for n in (50000, 100000, 200000):
    qa_deep("g43-qad%d" % n, "needle-%d.txt" % n)
qa_batch_major("g43-qab200000", "needle-200000.txt")
qa_batch_major("g43-qab250000", "needle-250000.txt")   # the near-full window: 250,000 prompt tokens + 128 generated in ctx 262,144

for n, reps in ((50000, 3), (100000, 3), (200000, 2)):
    save("g43-fin%d" % n, mk_deep.make_speed("g43-fin%d" % n, "deepcode-%d.txt" % n, reps, EXE, ["STK", "DKFDG"]))
save("g43-kv16-100k", mk_deep.make_speed("g43-kv16-100k", "deepcode-100000.txt", 2, EXE, ["DKFD", "DKFDG", "DKFD16"]))

for tag, prompt in (("code", "div-code.txt"), ("prose", "div-prose.txt")):
    session = "g43-kfm-" + tag
    spec = mk_final.make(session, prompt, 512, 4, ["KFP", "KFM"], EXE)
    kfm = next(a for a in spec["arms"] if a["name"] == "KFM")
    pad = copy.deepcopy(kfm)
    pad["name"] = "KFMPAD"
    pad["env"]["LLAMA_MTP_PAD_PROMPT"] = "1"
    pad["why"] = "KFM with the OLD prompt padding (LLAMA_MTP_PAD_PROMPT=1): up to ubatch-1 end-of-text tokens appended after the prompt"
    spec["arms"].append(pad)
    spec["control"] = "KFP"
    save(session, spec)
