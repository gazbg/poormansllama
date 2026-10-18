"""G30-4 repair, round 4 (build g55): the remaining cells of the matrix with the band recipes that hold the bars. Sessions g106-*.

    python mk_g106.py          writes build/llama-cuda/specs/g106-*.json and copies them next to this file
Band recipes (the final MTP recipe per band):
  stock path, layer-major 4,093 tokens, interleave 2,047 tokens : MF  (B0 + --mtp --mtp-draft 1 --mtp-draft-experts-gpu --mtp-draft-vocab 32768 + LLAMA_CUDA_TOPK_MOE_SMALL=1)
  interleave 1,023 tokens                                       : MFu (MF + --ubatch 1024 --mtp-catchup 128: the prompt is one chunk, the scheduler reserves for 1,024 rows, the draft context for 128)
  layer-major >= 23,995 tokens, segmented                       : MFL / MFLt (MF + --mtp-draft-late [+ --mtp-mirror-tail 2048])
Cells: ic1k, lc24k, sgc50k (code), ip1k, lp24k, sgp50k (prose), sgc100k and sgc200k (code, B0 itself fails the bars there). G94_PREFIX / G94_EXE choose the session prefix and the build (g107 on g56 for the final evidence).
"""
import copy
import json
import os
import sys

os.environ.setdefault("G94_PREFIX", "g106")
os.environ.setdefault("G94_EXE", "build/llama-cuda/bin/g55")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g94  # noqa: E402
import mk_g99  # noqa: E402
import mk_g105  # noqa: E402

CELLS = (("g94-ic1k", "mtpcode-1024.txt", "inter", 1023, 3), ("g94-lc24k", "mtpcode-24000.txt", "lm", 23995, 3), ("g94-sgc50k", "mtpcode-50000.txt", "seg", 49999, 2),
         ("g94-ip1k", "mtpprose-1024.txt", "inter", 1023, 3), ("g94-lp24k", "mtpprose-24000.txt", "lm", 23995, 3), ("g94-sgp50k", "mtpprose-50000.txt", "seg", 49999, 2),
         ("g94-sgc100k", "mtpcode-100000.txt", "seg", 99990, 2), ("g94-sgc200k", "mtpcode-200000.txt", "seg", 199995, 2))

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
        if tokens < 20000:
            mfu = mk_g105.ub(mf, 1024)
            mfu["name"] = "MFu"
            mfu["flags"] = mfu["flags"] + ["--mtp-catchup", "128"]
            mfu["why"] = "MF with --ubatch 1024 and a 128-row draft context"
            arms = [by["B0"], mfu]
        else:
            mfl = copy.deepcopy(mf)
            mfl["name"] = "MFL"
            mfl["flags"] = mf["flags"] + ["--mtp-draft-late"]
            mfl["env"] = dict(mf["env"], LLAMA_TG_VRAM_TRACE="1")
            mfl["why"] = "MF + the late draft bring-up"
            mflt = copy.deepcopy(mfl)
            mflt["name"] = "MFLt"
            mflt["flags"] = mfl["flags"] + ["--mtp-mirror-tail", "2048"]
            mflt["why"] = "MFL + the 2,048-position catch-up"
            arms = [by["B0"], mfl, mflt]
        spec["arms"] = arms
        mk_g94.save(real, spec)
