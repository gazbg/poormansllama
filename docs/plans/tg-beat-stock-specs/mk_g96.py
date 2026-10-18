"""G30 / M2e: sessions g96-* (build g50): the draft stage split and the pruned draft vocabulary, measured on the layer-major 4k cell.
Arms: M (full vocabulary, the control), V32 (the draft's pick scans token ids < 32768), V16 (< 16384), V8 (< 8192) via LLAMA_MTP_DRAFT_VOCAB_N.
The logits are still computed in full: this measures only what the acceptance rate does when the draft can only pick from a small vocabulary,
i.e. what a pruned draft lm head could keep. The verify is unchanged, so the text is the same in every arm.
  python docs/plans/tg-beat-stock-specs/mk_g96.py        writes g96-dvc4k.json (code) and g96-dvp4k.json (prose)
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = json.load(open(os.path.join(HERE, "g95-smoke-lc4k.json")))
EXE = "build/llama-cuda/bin/g50"
WT = "C:/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1"

for tag, kind in (("dvc4k", "mtpcode"), ("dvp4k", "mtpprose")):
    s = copy.deepcopy(BASE)
    s["session"] = "g96-" + tag
    s["exe_dir"] = EXE
    s["tg_tokens"] = 256
    s["reps"] = 2
    pf = "build/llama-cuda/gplr07-qa-prompts/%s-4096.txt" % kind
    s["prompt"] = pf
    c = s["common"]
    c[c.index("--prompt-file") + 1] = WT + "/" + pf
    c[c.index("--tg-tokens") + 1] = "256"
    c[c.index("--model") + 1] = c[c.index("--model") + 1]
    base_arm = s["arms"][0]
    arms = []
    for name, n in (("M", 0), ("V32", 32768), ("V16", 16384), ("V8", 8192)):
        a = copy.deepcopy(base_arm)
        a["name"] = name
        if n:
            a["env"]["LLAMA_MTP_DRAFT_VOCAB_N"] = str(n)
            a["why"] = "same as M, the draft's pick scans only token ids < %d" % n
        else:
            a["why"] = "B0 + MTP, full-vocabulary draft pick (control)"
        arms.append(a)
    s["arms"] = arms
    s["control"] = "M"
    s["items"] = "N1,N2,N3,N4,K1"
    out = os.path.join(HERE, "g96-%s.json" % tag)
    json.dump(s, open(out, "w"), indent=1)
    print("wrote", out)
