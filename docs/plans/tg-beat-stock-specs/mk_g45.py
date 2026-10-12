"""G21: the prefill on a DEDICATED GPU (owner direction, chat 2026-10-12: "revisit the use of --layer-slots 2, there should be enough vram to use a 2 slot for PP",
"assume the gpu will be dedicated to the process and no others will run"). The keep-free rule (at least 16,000 MiB free) is measured over the DECODE phase (the runner prints a
second `free-watch (decode phase ...)` line since build g40); the prefill may use the free memory of the card.

    python mk_g45.py            writes build/llama-cuda/specs/g45-*.json and copies them next to this file

g45-pp4k, g45-pp50k : one session per prompt, every arm is one needle-QA run (128 tokens, GQA8 decode attention on) of a prefill configuration:
    P512F  : the original shape of the final configuration: ubatch 512, TWO layer slots with the async feed, f32 act slab, 14-slot initial cache
    P512B  : the same with the bf16 slab
    P1024F, P1024B, P2048F : larger ubatch (the compute buffer grows 1,196 -> about 1,780 MiB at 2048; the prefill rate may grow with it)
    D1S    : the deep variant of G19 (ONE layer slot, no async feed, bf16 slab, 2-slot initial cache), the configuration the owner questioned
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402
import mk_qa   # noqa: E402

ROOT = mk_g21.root
EXE = "build/llama-cuda/bin/g40"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")


def save(name, spec):
    json.dump(spec, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, name + ".json"), "w"), indent=1)
    print(name, [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", spec["tg_tokens"])


def strip_common(spec):
    c = list(spec["common"])
    for flag in ("--layer-slots", "--act-slab-dtype"):
        i = c.index(flag)
        del c[i:i + 2]
    spec["common"] = c


def variant(base, name, slab, ubatch, slots, async_feed, init_slots=None, no_logits=False):
    a = copy.deepcopy(base)
    a["name"] = name
    fl = list(a["flags"])
    fl[fl.index("--ubatch") + 1] = ubatch
    fl += ["--act-slab-dtype", slab, "--layer-slots", slots]
    if not async_feed:
        fl += ["--pp-async-feed", "0"]
    if no_logits:
        fl += ["--pp-no-logits"]
    a["flags"] = fl
    a["env"]["LLAMA_FATTN_GQA8"] = "1"
    if init_slots:
        a["env"]["LLAMA_MOE_HYBRID"] = init_slots
    a["why"] = "prefill configuration %s: ubatch %s, %s layer slot(s)%s, %s slab%s" % (name, ubatch, slots, "" if async_feed else " without async feed", slab, "" if not init_slots else ", initial cache of %s slots" % init_slots)
    return a


def pp_session(session, needle):
    spec, arms = mk_qa.make(session, needle, EXE, ["QKFP"])
    strip_common(spec)
    base = arms[0]
    spec["arms"] = [
        variant(base, "P512F", "f32", "512", "2", True),
        variant(base, "P512B", "bf16", "512", "2", True),
        variant(base, "P1024F", "f32", "1024", "2", True),
        variant(base, "P1024B", "bf16", "1024", "2", True),
        variant(base, "P2048F", "f32", "2048", "2", True),
        variant(base, "D1S", "bf16", "512", "1", False, init_slots="2", no_logits=True),
    ]
    spec["control"] = "P512F"
    save(session, spec)


pp_session("g45-pp4k", "anchor-4096.txt")
pp_session("g45-pp50k", "needle-50000.txt")


def pp_session_b(session, needle, arms_wanted):
    """Second sweep on the dedicated GPU: how far does a bigger ubatch go (compute buffer 1.2 -> 1.8 -> 3 -> 6 GiB)?"""
    spec, arms = mk_qa.make(session, needle, EXE, ["QKFP"])
    strip_common(spec)
    base = arms[0]
    table = {"P512F": ("f32", "512"), "P2048F": ("f32", "2048"), "P4096F": ("f32", "4096"), "P8192F": ("f32", "8192")}
    spec["arms"] = [variant(base, n, table[n][0], table[n][1], "2", True) for n in arms_wanted]
    spec["control"] = arms_wanted[0]
    save(session, spec)


pp_session_b("g45-ppb4k", "anchor-4096.txt", ["P2048F", "P4096F", "P8192F"])
pp_session_b("g45-ppb50k", "needle-50000.txt", ["P2048F", "P4096F", "P8192F"])
pp_session_b("g45-ppb200k", "needle-200000.txt", ["P512F", "P2048F", "P4096F"])
