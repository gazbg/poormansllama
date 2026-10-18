"""G30 / M3 round 2: the final matrix on build g52 (= g47 + stage-c/g30 .. g30g). Sessions g99-*.

    python mk_g99.py [cell ...]        writes build/llama-cuda/specs/g99-*.json and copies them next to this file

Arms per cell (counterbalanced, same session):
  B0   the band's best non-MTP configuration (as in mk_g94.py)
  MF   the final MTP recipe: B0 + `--mtp --mtp-draft 1 --mtp-draft-experts-gpu --mtp-draft-vocab 32768` + LLAMA_CUDA_TOPK_MOE_SMALL=1 (the fused router for the 2-row verify)
  MFt  (24k / 50k cells only) MF + `--mtp-mirror-tail 2048`: the pre-registered second arm of criterion G30-2 (the catch-up of the last 2,048 positions only)
  MT   (24k / 50k cells only) MF without the pruned draft vocabulary, to show what the pruning is worth at depth
The two 4,096-token cells are not repeated: `g98-lc4k` / `g98-lp4k` ran the same four arms on the same build.
"""
import copy
import json
import os
import sys

os.environ.setdefault("G94_PREFIX", "g99")
os.environ.setdefault("G94_EXE", "build/llama-cuda/bin/g52")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g94  # noqa: E402

SKIP = set() if os.environ.get("G99_ALL") else {"g94-lc4k", "g94-lp4k"}        # G99_ALL=1: also the two 4,096-token cells (the final-build rerun g108)
TOPK = {"LLAMA_CUDA_TOPK_MOE_SMALL": "1"}


def final_arms(name, deep):
    path = os.path.join(mk_g94.OUT, name + ".json")
    spec = json.load(open(path))
    by = {a["name"]: a for a in spec["arms"]}
    b0, m = by["B0"], by["M"]
    mf = copy.deepcopy(m)
    mf["name"] = "MF"
    mf["flags"] = m["flags"] + ["--mtp-draft-vocab", "32768"]
    mf["env"] = dict(m["env"], **TOPK)
    mf["why"] = "final MTP recipe: B0 + MTP + pruned draft vocabulary (ids < 32768) + fused 2-row router"
    arms = [b0, mf]
    if deep:
        mft = copy.deepcopy(mf)
        mft["name"] = "MFt"
        mft["flags"] = mf["flags"] + ["--mtp-mirror-tail", "2048"]
        mft["why"] = "MF with the draft head mirroring only the last 2,048 prompt positions (pre-registered G30-2 variant)"
        mt = copy.deepcopy(m)
        mt["name"] = "MT"
        mt["env"] = dict(m["env"], **TOPK)
        mt["why"] = "MF without the pruned draft vocabulary (full-vocabulary draft pick)"
        arms += [mft, mt]
    spec["arms"] = arms
    spec["control"] = "B0"
    mk_g94.save(name, spec)


if __name__ == "__main__":
    only = set(sys.argv[1:])
    for name, prompt, band, tokens, reps, tail in mk_g94.CELLS:
        if name in SKIP or (only and name not in only and mk_g94.PREFIX + name[3:] not in only):
            continue
        mk_g94.session(name, prompt, band, tokens, reps, False)
        final_arms(mk_g94.PREFIX + name[3:], tokens >= 20000)
