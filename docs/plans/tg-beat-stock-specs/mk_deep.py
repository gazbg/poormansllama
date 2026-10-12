"""G19 deep-fill sessions (the final configuration at 50,000 / 100,000 / 200,000 prompt tokens, ctx 262,144):

    python mk_deep.py prompts                                  build deepcode-<N>.txt from needle-<N>.txt (speed prompts; needles stay in the haystack)
    python mk_deep.py qa    <session> <needle file> <exe_dir> <ARMS comma>      answer-stability session (arms QKFP, QKFM; see mk_qa.py)
    python mk_deep.py qad   <session> <needle file> <exe_dir> <ARMS comma>      the same with the deep variants QKFD, QKFMD (bf16 slab, 1 layer slot, 2-slot initial cache)
    python mk_deep.py speed <session> <deepcode file> <reps> <exe_dir> [ARMS comma, default STK,DKFD; DKFM adds MTP]

Speed arms: STK (stock llama-completion, the owner's flags), DKFD (= the QA arm QKFD, the deep variant of QKFP: bf16 slab, one layer slot without async feed, 2-slot initial cache; layer-major prefill, ubatch 512, keep-free auto slots, lazy KV, lean host leg + K=2
prefetch, async scan, fast sampler, input batching) with 512 generated tokens, counterbalanced pairs. The speed prompt is the needle file WITHOUT its instruction line and
WITHOUT its answer contract, followed by the code-continuation prompt of the short-prompt sessions (div-code.txt), so that the text generated after the long fill is code (the
hit rate depends on the generated text, never compare a long-fill run on a prompt whose continuation is the repeating filler). The prompt has no trailing newline because
llama-completion -f strips one (a prompt that differs in its last byte is not the same prompt).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402
import mk_qa  # noqa: E402

root = mk_g21.root
PROMPTS = root + r"\build\llama-cuda\gplr07-qa-prompts"
LENGTHS = (50000, 100000, 200000)


def build_prompts():
    code = open(os.path.join(PROMPTS, "div-code.txt"), "rb").read().decode("utf-8").rstrip("\n")
    for n in LENGTHS:
        src = os.path.join(PROMPTS, "needle-%d.txt" % n)
        if not os.path.exists(src):
            print("missing", src)
            continue
        text = open(src, "rb").read().decode("utf-8")
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        if not lines[0].startswith("You are answering questions about the document below."):
            raise SystemExit("unexpected first line in " + src)
        if not lines[-1].startswith("From the document above, list the four archive access codes"):
            raise SystemExit("unexpected last line in " + src)
        body = "\n".join(lines[1:-1])
        out = os.path.join(PROMPTS, "deepcode-%d.txt" % n)
        data = (body + "\n\n" + code).encode("utf-8")
        open(out, "wb").write(data)
        print(out, len(data), "bytes,", len(lines) - 2, "document lines")


def deepen(spec, arms, layer_slots="1", slab="bf16", init_slots="2"):
    """Repair of the G19 keep-free miss at 100k (g39-qa100000: QKFP minimum free 15,387 MiB, at the START of the prefill): the prefill-time footprint grows with the prompt
    (act slab 16,384 B/token in f32, the lazily committed KV 11.7 KB/token), so the deep arms use the bf16 slab (the default of the tool, half the bytes), ONE layer slot with
    --pp-async-feed 0 (a layer's weights are loaded once and used for the whole prompt, so the load is not on the critical path at these lengths; g40-qad100000's first try was
    refused loudly: "--pp-async-feed requires --layer-slots >= 2") and a 2-slot initial hybrid cache (the auto sizing grows it after the prefill anyway). The batch-major MTP arm
    keeps two layer slots because its prefill reloads every layer for every ubatch. Arms are renamed QKFP -> QKFD, QKFM -> QKFMD."""
    common = list(spec["common"])
    common[common.index("--act-slab-dtype") + 1] = slab
    i = common.index("--layer-slots")
    del common[i:i + 2]
    spec["common"] = common
    renamed = []
    for a in arms:
        a = dict(a)
        a["name"] = {"QKFP": "QKFD", "QKFM": "QKFMD"}.get(a["name"], a["name"])
        a["env"] = dict(a["env"])
        a["env"]["LLAMA_MOE_HYBRID"] = init_slots
        if a["name"] == "QKFD":
            # one slot forbids the async feed (it stages the next layer into the second slot); the batch-major MTP arm keeps two slots, its prefill reloads every layer per ubatch
            a["flags"] = list(a["flags"]) + ["--layer-slots", layer_slots, "--pp-async-feed", "0", "--pp-no-logits"]
            a["why"] = a["why"] + " [deep variant: bf16 slab, 1 layer slot without async feed, logits only for the last prompt token, 2-slot initial cache]"
        else:
            a["flags"] = list(a["flags"]) + ["--layer-slots", "2"]
            a["why"] = a["why"] + " [deep variant: 2-slot initial cache]"
        renamed.append(a)
    spec["control"] = renamed[0]["name"]
    return renamed


def make_speed(session, prompt_name, reps, exe_dir, arms_wanted):
    spec, arms = mk_qa.make(session, prompt_name, exe_dir, ["QKFP", "QKFM"])
    arms = deepen(spec, arms)
    common = list(spec["common"])
    for flag in ("--tg-tokens", "--tg-logit-margin"):
        common[common.index(flag) + 1] = "512"
    spec.update({"common": common, "qa": False, "tg_tokens": 512, "reps": reps, "interleave": True, "counterbalance": True})
    prompt = root.replace("\\", "/") + "/build/llama-cuda/gplr07-qa-prompts/" + prompt_name
    dkfd = dict(arms[0])
    dkfd["name"] = "DKFD"
    dkfd["why"] = "final plain configuration, deep variant (= QKFD), 512 generated tokens, deep fill"
    dkfm = dict(arms[1])
    dkfm["name"] = "DKFM"
    dkfm["why"] = "final configuration with MTP, deep variant (= QKFMD), 512 generated tokens, deep fill"
    # DKFBG: the plain batch-major prefill (no act slab, no MTP) with the GQA8 decode attention: the arm that holds the keep-free rule at 200k.
    dkfb = dict(dkfm)
    fl = list(dkfb["flags"])
    for flag, n in (("--mtp", 1), ("--mtp-draft", 2), ("--mtp-draft-experts-gpu", 1)):
        i = fl.index(flag)
        del fl[i:i + n]
    dkfb["flags"] = fl
    dkfb["name"] = "DKFBG"
    dkfb["env"] = dict(dkfb["env"])
    dkfb["env"]["LLAMA_FATTN_GQA8"] = "1"
    dkfb["why"] = "plain batch-major prefill (no act slab), GQA8 decode attention, 2-slot initial cache: holds the keep-free rule at 200k, at a slower prefill"
    # DKFD16: the same plain arm with an f16 KV cache. At decode with an f16 KV cache and a long KV the CUDA backend picks the tensor-core kernel with the GQA packing (fattn.cu), not the
    # vector kernel that quantized KV is tied to; this arm measures what that kernel is worth at depth and what the doubled KV costs the cache (a diagnostic, not a candidate).
    dkfd16 = dict(dkfd)
    dkfd16["name"] = "DKFD16"
    dkfd16["why"] = "DKFD with an f16 KV cache (diagnostic: the tensor-core GQA-packed decode attention against the vector kernel)"
    i = common.index("--kv-cache")
    del common[i:i + 2]
    spec["common"] = common
    for a in (dkfd, dkfm):
        a["flags"] = list(a["flags"]) + ["--kv-cache", "q8"]
    dkfd16["flags"] = list(dkfd16["flags"]) + ["--kv-cache", "f16"]
    out_arms = []
    for name in arms_wanted:
        if name == "STK":
            out_arms.append({"name": "STK", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-mmap"], "env": {},
                             "why": "stock llama-completion, the owner's document configuration"})
        elif name == "DKFD":
            out_arms.append(dkfd)
        elif name == "DKFM":
            out_arms.append(dkfm)
        elif name == "DKFD16":
            out_arms.append(dkfd16)
        elif name == "DKFBG":
            out_arms.append(dkfb)
        elif name in ("DKFDG", "DKFMG"):
            g = dict(dkfd if name == "DKFDG" else dkfm)
            g["name"] = name
            g["env"] = dict(g["env"])
            g["env"]["LLAMA_FATTN_GQA8"] = "1"
            g["why"] += " [GQA8 decode attention, LLAMA_FATTN_GQA8=1]"
            out_arms.append(g)
        else:
            raise SystemExit("unknown arm " + name)
    spec["arms"] = out_arms
    spec["control"] = arms_wanted[0]
    spec["stock_common"] = ["-m", mk_g21.MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", "512",
                            "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt"]
    return spec


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prompts":
        build_prompts()
    elif mode == "qa":
        session, needle, exe, arms = sys.argv[2:6]
        names = arms.split(",")
        # QKFPG / QKFMG = the arm with the GQA8 decode attention (LLAMA_FATTN_GQA8=1)
        spec, arm_list = mk_qa.make(session, needle, exe, [n.rstrip("G") if n.endswith("PG") or n.endswith("MG") else n for n in names])
        for a, nm in zip(arm_list, names):
            if nm != a["name"]:
                a["name"] = nm
                a["env"]["LLAMA_FATTN_GQA8"] = "1"
                a["why"] += " [GQA8 decode attention]"
        spec["control"] = arm_list[0]["name"]
        spec["arms"] = arm_list
        out = root + r"\build\llama-cuda\specs\%s.json" % session
        json.dump(spec, open(out, "w"), indent=1)
        print(out, [a["name"] for a in arm_list])
    elif mode == "qad":
        session, needle, exe, arms = sys.argv[2:6]
        names = arms.split(",")
        # QKFDU256 = QKFD with --ubatch 256 (a diagnostic sweep of the prefill footprint); QKFDG = QKFD with the GQA8 decode attention (LLAMA_FATTN_GQA8=1); QKFDGU128 both
        base_of = lambda x: "QKFMD" if x.startswith("QKFMD") else "QKFD"
        wanted = [{"QKFD": "QKFP", "QKFMD": "QKFM"}[base_of(x)] for x in names]
        spec, arm_list = mk_qa.make(session, needle, exe, wanted)
        arm_list = deepen(spec, arm_list)
        for a, nm in zip(arm_list, names):
            suffix = nm[len(base_of(nm)):]
            a["name"] = nm
            if suffix.startswith("G"):
                a["env"]["LLAMA_FATTN_GQA8"] = "1"
                a["why"] += " [GQA8 decode attention]"
                suffix = suffix[1:]
            if suffix.startswith("U"):
                u = suffix[1:]
                fl = list(a["flags"])
                fl[fl.index("--ubatch") + 1] = u
                a["flags"] = fl
                a["why"] += " [ubatch %s]" % u
        spec["control"] = arm_list[0]["name"]
        spec["arms"] = arm_list
        out = root + r"\build\llama-cuda\specs\%s.json" % session
        json.dump(spec, open(out, "w"), indent=1)
        print(out, [a["name"] for a in arm_list])
    elif mode == "speed":
        session, prompt_name, reps, exe = sys.argv[2:6]
        arms = sys.argv[6].split(",") if len(sys.argv) > 6 else ["STK", "DKFD"]
        spec = make_speed(session, prompt_name, int(reps), exe, arms)
        out = root + r"\build\llama-cuda\specs\%s.json" % session
        json.dump(spec, open(out, "w"), indent=1)
        print(out, [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", spec["tg_tokens"])
    else:
        raise SystemExit("unknown mode " + mode)
