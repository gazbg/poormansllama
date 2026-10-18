"""G30 / M2e-M2i: sessions g97-* (build g51): what the draft-stage trims are worth, clean (no host load during the sessions).
Arms: B0 (the current optimal solution of the cell), B0S (B0 with the cache fixed at the slot count the MTP arm ends up with: the price of the draft head's residency alone, an upper bound
for what an acceptance gate could reach), M (B0 + MTP), MV (M + --mtp-draft-vocab 32768).
  python docs/plans/tg-beat-stock-specs/mk_g97.py            writes g97-lc4k.json (code 4k), g97-lp4k (prose 4k)
Environment: G97_EXE (default build/llama-cuda/bin/g51), G97_PREFIX (default g97), G97_SLOTS (the cache slots of B0S, default 57), G97_LEN (default 4096).
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.environ.get("G97_EXE", "build/llama-cuda/bin/g51")
PRE = os.environ.get("G97_PREFIX", "g97")
SLOTS = os.environ.get("G97_SLOTS", "57")
LEN = int(os.environ.get("G97_LEN", "4096"))
SRC = {4096: "g94-lc4k.json", 24000: "g94-lc24k.json"}[LEN]
WT = "C:/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1"

base = json.load(open(os.path.join(HERE, SRC)))
b0 = next(a for a in base["arms"] if a["name"] == "B0")
m = next(a for a in base["arms"] if a["name"] == "M")
tag_len = "4k" if LEN == 4096 else "24k"
for kind, letter in (("mtpcode", "c"), ("mtpprose", "p")):
    s = copy.deepcopy(base)
    s["session"] = "%s-l%s%s" % (PRE, letter, tag_len)
    s["exe_dir"] = EXE
    s["tg_tokens"] = 512
    s["reps"] = 3
    pf = "build/llama-cuda/gplr07-qa-prompts/%s-%d.txt" % (kind, LEN if LEN == 4096 else 24000)
    s["prompt"] = pf
    c = s["common"]
    c[c.index("--prompt-file") + 1] = WT + "/" + pf
    arms = []
    a = copy.deepcopy(b0)
    a["why"] = "B0: the current optimal solution (control)"
    arms.append(a)
    a = copy.deepcopy(b0)
    a["name"] = "B0S"
    a["flags"][a["flags"].index("--tg-hybrid-slots") + 1] = SLOTS
    a["why"] = "B0 with the cache fixed at %s slots per layer (what MTP leaves after the draft head's residency)" % SLOTS
    arms.append(a)
    a = copy.deepcopy(m)
    a["why"] = "B0 + MTP (1 draft, draft experts on the GPU)"
    arms.append(a)
    a = copy.deepcopy(m)
    a["name"] = "MV"
    a["flags"] += ["--mtp-draft-vocab", "32768"]
    a["why"] = "M + the draft head draws only from token ids < 32768"
    arms.append(a)
    s["arms"] = arms
    s["control"] = "B0"
    out = os.path.join(HERE, "%s.json" % s["session"])
    json.dump(s, open(out, "w"), indent=1)
    print("wrote", out)
