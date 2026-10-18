"""G30-4 repair, round 1 (build g53 = g52 + stage-c/g30h): the three cells whose whole-run bars failed with MTP, with the late draft bring-up.

    python mk_g103.py [cell ...]       writes build/llama-cuda/specs/g103-*.json and copies them next to this file
Arms: B0, MF (the g99 recipe), MFL (MF + --mtp-draft-late), MFLt (MFL + --mtp-mirror-tail 2048, 24k / 50k only). Cells: ic1k (interleave 1,023), lc24k (layer-major 23,995), sgc50k (segmented 49,999).
"""
import copy
import json
import os
import sys

os.environ["G94_PREFIX"] = "g103"
os.environ.setdefault("G94_EXE", "build/llama-cuda/bin/g53")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g94  # noqa: E402
import mk_g99  # noqa: E402

CELLS = (("g94-ic1k", "mtpcode-1024.txt", "inter", 1023, 3), ("g94-lc24k", "mtpcode-24000.txt", "lm", 23995, 3), ("g94-sgc50k", "mtpcode-50000.txt", "seg", 49999, 2))
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
        mfl["why"] = "MF + the late draft bring-up (the head's experts and context come up after the prefill)"
        arms = [by["B0"], mf, mfl]
        if tokens >= 20000:
            mflt = copy.deepcopy(mfl)
            mflt["name"] = "MFLt"
            mflt["flags"] = mfl["flags"] + ["--mtp-mirror-tail", "2048"]
            mflt["why"] = "MFL + the 2,048-position catch-up"
            arms.append(mflt)
        spec["arms"] = arms
        mk_g94.save(real, spec)
