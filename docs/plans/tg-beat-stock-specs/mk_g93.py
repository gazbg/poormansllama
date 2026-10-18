"""G30 / M1 sessions: the CLEAN decode effect of MTP self-speculation on the current decode engine (build g47, no source edit).

    python mk_g93.py            writes build/llama-cuda/specs/g93-*.json and copies them next to this file

Two arms per session, run counterbalanced in the SAME session, differing in nothing but MTP:
  HB   the strict recipe of G24 (plain batch-major prefill, ubatch 512, act_slab 0 B, 2-slot initial cache, keep-free hybrid decode, lean host leg, K = 2 prefetch, lazy KV, GQA8)
  HBM  HB + `--mtp --mtp-draft 1 --mtp-draft-experts-gpu` (the draft head's experts on the GPU, one draft per round), SAME prefill and SAME environment (GQA8 on)
The old depth comparison (ledger row 48) ran the MTP arm without LLAMA_FATTN_GQA8 and with a different prefill; here nothing but MTP differs.
Cells: code and prose at 512 tokens (the existing div-code / div-prose prompts), 4,096, 24,000 and 50,000 tokens (mk_g93_prompts.py); 512 generated tokens;
reps 4 / 3 / 3 / 2 (cost: the batch-major prefill is ~260 tok/s). Session names: g93-c512 g93-p512 g93-c4k g93-p4k g93-c24k g93-p24k g93-c50k g93-p50k.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402
import mk_g87  # noqa: E402

ROOT = mk_g21.root
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")
EXE = os.environ.get("G93_EXE", "build/llama-cuda/bin/g47")
NTOK = 512
MTP_FLAGS = ["--mtp", "--mtp-draft", "1", "--mtp-draft-experts-gpu"]


def save(name, spec):
    # binary writes: text mode would turn LF into CRLF on Windows
    data = json.dumps(spec, indent=1).encode("utf-8")
    open(os.path.join(OUT, name + ".json"), "wb").write(data)
    open(os.path.join(HERE, name + ".json"), "wb").write(data)
    print(name, [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", spec["tg_tokens"], "prompt", spec["prompt"])


def session(name, prompt_name, reps):
    hb, spec = mk_g87.strict_arm(name, prompt_name, "HB")
    spec = copy.deepcopy(spec)
    hbm = copy.deepcopy(hb)
    hbm["name"] = "HBM"
    hbm["flags"] = list(hb["flags"]) + MTP_FLAGS
    hbm["why"] = "HB + MTP self-speculation (1 draft, the draft layer's experts on the GPU), same batch-major prefill and environment as HB"
    common = list(spec["common"])
    for flag in ("--tg-tokens", "--tg-logit-margin"):
        common[common.index(flag) + 1] = str(NTOK)
    spec.update({"session": name, "exe_dir": EXE, "tg_tokens": NTOK, "common": common, "control": "HB", "qa": False, "reps": reps, "interleave": True,
                 "counterbalance": True, "arms": [hb, hbm]})
    spec["prompt"] = "build/llama-cuda/gplr07-qa-prompts/" + prompt_name
    save(name, spec)


if __name__ == "__main__":
    session("g93-c512", "div-code.txt", 4)
    session("g93-p512", "div-prose.txt", 4)
    session("g93-c4k", "mtpcode-4096.txt", 3)
    session("g93-p4k", "mtpprose-4096.txt", 3)
    session("g93-c24k", "mtpcode-24000.txt", 3)
    session("g93-p24k", "mtpprose-24000.txt", 3)
    session("g93-c50k", "mtpcode-50000.txt", 2)
    session("g93-p50k", "mtpprose-50000.txt", 2)
