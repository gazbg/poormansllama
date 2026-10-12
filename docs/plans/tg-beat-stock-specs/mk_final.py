"""Final integrated comparison under the owner's own definition of "8 GB" (at least 16,000 MiB stay free in the CUDA runtime's view, llama-server --fit-target 16000):

    python mk_final.py <session> <prompt file name> <tg tokens> <reps> <ARMS comma> <exe_dir>

arms: STK (stock llama-completion with the owner's flags), KFP (hybrid: keep-free auto sizing, lazy KV, lean host leg, K=2 prefetch, async scan, fast sampler, input batching),
KFM (KFP + MTP self-speculation with the draft layer's experts on the GPU), KFS (KFP + the per-layer slot split), KFB (the same as KFP but with the graph host leg and no
prefetch, the old team placement: the G9 configuration plus the accepted G10 / G11 options, i.e. the state before G14).
Every hybrid arm watches the runtime's free memory every 20 ms (LLAMA_TG_FREE_WATCH_MS), so the owner's definition is checked per run, not assumed.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402

root = mk_g21.root


def make(session, prompt_name, tg, reps, arms_wanted, exe_dir):
    base = mk_g21.make(session, prompt_name, 24, tg, reps, ["PLAIN"], False, exe_dir)
    prompt = root.replace("\\", "/") + "/build/llama-cuda/gplr07-qa-prompts/" + prompt_name
    keepfree = ["--tg-vram-cap-mib", "0", "--tg-vram-keep-free-mib", "16000"]
    plain = next(a for a in base["arms"] if a["name"] == "PLAIN")
    arms = []
    for name in arms_wanted:
        if name == "STK":
            arms.append({"name": "STK", "engine": "stock", "flags": ["-ncmoe", "37", "-t", "16", "--no-mmap"], "env": {}, "why": "stock llama-completion, the owner's document configuration"})
            continue
        a = {"name": name, "flags": list(plain["flags"]) + keepfree, "env": dict(plain["env"])}
        a["env"]["LLAMA_TG_FREE_WATCH_MS"] = "20"
        if name == "KFB":
            a["env"]["LLAMA_MOE_HYBRID_LEAN"] = "0"
            a["env"]["LLAMA_MOE_HYBRID_MAIN_CPU"] = "-1"
            a["why"] = "keep-free hybrid with the graph host leg (state before G14), async scan, fast sampler, input batching"
        elif name == "KFP":
            a["env"]["LLAMA_MOE_HYBRID_LEAN"] = "1"
            a["env"]["LLAMA_MOE_HYBRID_LEAN_PF"] = "2"
            a["why"] = "keep-free hybrid with the lean host leg and K=2 prefetch (G14 + G15), async scan, fast sampler, input batching"
        elif name == "KFS":
            a["env"]["LLAMA_MOE_HYBRID_LEAN"] = "1"
            a["env"]["LLAMA_MOE_HYBRID_LEAN_PF"] = "2"
            a["env"]["LLAMA_MOE_HYBRID_SPLIT"] = root.replace("\\", "/") + "/build/llama-cuda/aer-split-misses-pooled.txt"
            a["env"]["LLAMA_MOE_HYBRID_SPLIT_ALPHA"] = "0.5"
            a["why"] = "KFP + the per-layer slot split from the calibration misses (G17c arm P1)"
        elif name == "KFM":
            a["env"]["LLAMA_MOE_HYBRID_LEAN"] = "1"
            a["env"]["LLAMA_MOE_HYBRID_LEAN_PF"] = "2"
            a["flags"] += ["--mtp", "--mtp-draft", "1", "--mtp-draft-experts-gpu"]
            a["why"] = "KFP + MTP self-speculation (1 draft, draft-layer experts on the GPU)"
        else:
            raise SystemExit("unknown arm " + name)
        arms.append(a)
    spec = dict(base)
    spec["arms"] = arms
    spec["control"] = arms_wanted[0]
    spec["stock_common"] = ["-m", mk_g21.MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", str(tg),
                            "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt"]
    return spec


if __name__ == "__main__":
    session, prompt_name, tg, reps, arms, exe = sys.argv[1:7]
    spec = make(session, prompt_name, int(tg), int(reps), arms.split(","), exe)
    out = root + r"\build\llama-cuda\specs\%s.json" % session
    json.dump(spec, open(out, "w"), indent=1)
    print(out, [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", tg)
