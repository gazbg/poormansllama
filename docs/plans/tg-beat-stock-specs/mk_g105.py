"""G30-4 repair, round 3 (build g55 = g54 + stage-c/g30j: the late bring-up's warm-up at start-up).

    python mk_g105.py          writes build/llama-cuda/specs/g105-*.json and copies them next to this file
ic1k arms: B0, B0u (B0 with --ubatch 1024: the 1,023-token prompt is ONE chunk either way), MF, MFu (MF with --ubatch 1024 and --mtp-catchup 128: the scheduler reserves for 1,024 rows instead of 2,048
and the draft context for 128: -562 and -221 MiB during the prefill, no migration).
24k / 50k arms: B0, MFL, MFLt (the late bring-up with and without the 2,048-position catch-up).
"""
import copy
import json
import os
import sys

os.environ.setdefault("G94_PREFIX", "g105")
os.environ.setdefault("G94_EXE", "build/llama-cuda/bin/g55")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g94  # noqa: E402
import mk_g99  # noqa: E402

CELLS = (("g94-ic1k", "mtpcode-1024.txt", "inter", 1023, 3), ("g94-lc24k", "mtpcode-24000.txt", "lm", 23995, 3), ("g94-sgc50k", "mtpcode-50000.txt", "seg", 49999, 2))


def ub(arm, n):
    a = copy.deepcopy(arm)
    a["flags"][a["flags"].index("--ubatch") + 1] = str(n)
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
        if tokens < 20000:
            b0u = ub(by["B0"], 1024)
            b0u["name"] = "B0u"
            b0u["why"] = "B0 with --ubatch 1024 (the prompt is one chunk either way): what the smaller scheduler reservation costs without MTP"
            mfu = ub(mf, 1024)
            mfu["name"] = "MFu"
            mfu["flags"] = mfu["flags"] + ["--mtp-catchup", "128"]
            mfu["why"] = "MF with --ubatch 1024 and a 128-row draft context: the prefill's VRAM headroom stays"
            arms = [by["B0"], b0u, mf, mfu]
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
