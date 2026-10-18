"""G30-4 repair, round 2 (build g54 = g53 + stage-c/g30i: the head's experts staged in page-locked memory at start-up).

    python mk_g104.py          writes build/llama-cuda/specs/g104-*.json and copies them next to this file
ic1k arms: B0, MF, MFL (MF + --mtp-draft-late), B0_3 (B0 with 3 layer slots instead of 4), MF3 (MF with 3 layer slots and --mtp-catchup 128: no migration, one slot given up instead).
24k / 50k arms: B0, MFL, MFLt (MFL + --mtp-mirror-tail 2048).
"""
import copy
import json
import os
import sys

os.environ["G94_PREFIX"] = "g104"
os.environ["G94_EXE"] = "build/llama-cuda/bin/g54"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g94  # noqa: E402
import mk_g99  # noqa: E402

CELLS = (("g94-ic1k", "mtpcode-1024.txt", "inter", 1023, 3), ("g94-lc24k", "mtpcode-24000.txt", "lm", 23995, 3), ("g94-sgc50k", "mtpcode-50000.txt", "seg", 49999, 2))


def slots(arm, n):
    a = copy.deepcopy(arm)
    a["flags"][a["flags"].index("--layer-slots") + 1] = str(n)
    return a


if __name__ == "__main__":
    only = set(sys.argv[1:])
    for name, prompt, band, tokens, reps in CELLS:
        real = mk_g94.PREFIX + name[3:]
        if only and real not in only:
            continue
        mk_g94.session(name, prompt, band, tokens, reps, False)
        mk_g99.final_arms(real, False)
        spec = json.load(open(os.path.join(mk_g94.OUT, real + ".json")))
        by = {a["name"]: a for a in spec["arms"]}
        mf = by["MF"]
        mfl = copy.deepcopy(mf)
        mfl["name"] = "MFL"
        mfl["flags"] = mf["flags"] + ["--mtp-draft-late"]
        mfl["env"] = dict(mf["env"], LLAMA_TG_VRAM_TRACE="1")
        mfl["why"] = "MF + the late draft bring-up"
        if tokens < 20000:
            b03 = slots(by["B0"], 3)
            b03["name"] = "B0_3"
            b03["why"] = "B0 with 3 layer slots (what giving one slot up costs without MTP)"
            mf3 = slots(mf, 3)
            mf3["name"] = "MF3"
            mf3["flags"] = mf3["flags"] + ["--mtp-catchup", "128"]
            mf3["why"] = "MF with 3 layer slots and a 128-row draft context: no migration, one layer slot given up"
            arms = [by["B0"], mf, mfl, b03, mf3]
        else:
            mflt = copy.deepcopy(mfl)
            mflt["name"] = "MFLt"
            mflt["flags"] = mfl["flags"] + ["--mtp-mirror-tail", "2048"]
            mflt["why"] = "MFL + the 2,048-position catch-up"
            arms = [by["B0"], mfl, mflt]
        spec["arms"] = arms
        mk_g94.save(real, spec)
