"""G30 / M2j: sessions g98-* (build g52 = g51 + g30g): what the fused 2-row MoE router (LLAMA_CUDA_TOPK_MOE_SMALL=1) is worth, clean.
Arms: B0 (the current optimal solution), M (B0 + MTP), MT (M + the fused router for 2-4 rows), MVT (MT + --mtp-draft-vocab 32768).
  python docs/plans/tg-beat-stock-specs/mk_g98.py        writes g98-lc4k.json, g98-lp4k.json from the g97 specs (same prompts, reps, flags)
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.environ.get("G98_EXE", "build/llama-cuda/bin/g52")
for tag in ("lc4k", "lp4k"):
    s = json.load(open(os.path.join(HERE, "g97-%s.json" % tag)))
    s["session"] = "g98-" + tag
    s["exe_dir"] = EXE
    by = {a["name"]: a for a in s["arms"]}
    arms = [by["B0"], by["M"]]
    mt = copy.deepcopy(by["M"])
    mt["name"] = "MT"
    mt["env"]["LLAMA_CUDA_TOPK_MOE_SMALL"] = "1"
    mt["why"] = "M + the fused MoE router for the 2-row verify"
    arms.append(mt)
    mvt = copy.deepcopy(by["MV"])
    mvt["name"] = "MVT"
    mvt["env"]["LLAMA_CUDA_TOPK_MOE_SMALL"] = "1"
    mvt["why"] = "MT + the draft head draws only from token ids < 32768"
    arms.append(mvt)
    s["arms"] = arms
    out = os.path.join(HERE, "g98-%s.json" % tag)
    json.dump(s, open(out, "w"), indent=1)
    print("wrote", out)
