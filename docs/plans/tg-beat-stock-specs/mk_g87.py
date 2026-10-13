"""G24 sessions: the final TG verification under the STRICT (whole-run) reading of the owner's keep-free rule, on natural text, build g45.

    python mk_g87.py            writes build/llama-cuda/specs/g87-*.json and copies them next to this file

Arms (all with --tg-tokens 918 = the owner's document generation length; prompts are byte-identical for both engines):
  STK  stock llama-completion with the owner's flags (-ncmoe 37, -fit off, q8 KV, -c 262000, greedy, --ignore-eos): the same-session stock
  HF   the final fast configuration (G21 'DKFE'): layer-major prefill, ubatch 2048, f32 slab, two layer slots with the async feed, hybrid decode, GQA8 attention.
       Holds the whole-run keep-free rule only for short prompts (G21: 16,163 / 16,169 MiB at 4,096 / 24,000 tokens; 15,555 at 50,000).
  HB   the strict recipe (G19 'QKFBG'): plain batch-major prefill (act_slab = 0 B), 2-slot initial cache, GQA8 attention; holds the whole-run rule at every length
       (G19: 16,161 / 16,173 MiB at 200,000 / 250,000 tokens). Its flags are copied from the validated QKFBG arm, including `--act-slab-dtype bf16` (act_slab = 0 B there).
Sessions:
  g87-short-B   doc-shape prompt nat-B (839 tokens, chat template, thinking mode; the stock repetition probe g87-natprobe picked it: rep6 0.0), STK HF HB, 4 reps
  g87-short-A   the second natural prompt nat-A (830 tokens), STK HF HB, 2 reps (robustness)
  g87-n24k      natdeep-24000, STK HF HB, 3 reps;  g87-n50k natdeep-50000 STK HB 3 reps;  g87-n100k STK HB 3 reps;  g87-n200k STK HB 2 reps;  g87-n250k STK HB 1 rep
  g87-qab24000 / qab50000 / qab100000   answer stability of the strict recipe (QKFBG, needle QA, 128 tokens) at the lengths the earlier QA runs did not cover
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_deep  # noqa: E402
import mk_g21   # noqa: E402
import mk_qa    # noqa: E402

ROOT = mk_g21.root
EXE = "build/llama-cuda/bin/g45"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")
NTOK = 918


def save(name, spec):
    json.dump(spec, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(HERE, name + ".json"), "w"), indent=1)
    print(name, [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", spec["tg_tokens"])


def strip_mtp(flags):
    fl = list(flags)
    for flag, n in (("--mtp", 1), ("--mtp-draft", 2), ("--mtp-draft-experts-gpu", 1)):
        i = fl.index(flag)
        del fl[i:i + n]
    return fl


def final_arm(base, name):
    """QKFP (layer-major, ubatch 512, f32 slab, two layer slots, 14-slot initial cache) with ubatch 2048 and the GQA8 decode attention (= G21's DKFE)."""
    a = copy.deepcopy(base)
    a["name"] = name
    fl = list(a["flags"])
    fl[fl.index("--ubatch") + 1] = "2048"
    a["flags"] = fl
    a["env"]["LLAMA_FATTN_GQA8"] = "1"
    a["why"] = "final fast configuration (G21 DKFE): layer-major prefill ubatch 2048, two layer slots with the async feed, f32 slab, keep-free hybrid decode, GQA8 decode attention"
    return a


def strict_arm(session, prompt_name, name):
    """The validated QKFBG arm (g48-qab200000 / g48-qab250000): batch-major prefill, no act slab, GQA8, 2-slot initial cache."""
    spec, arms = mk_qa.make(session, prompt_name, EXE, ["QKFM"])
    arms = mk_deep.deepen(spec, arms)
    a = arms[0]
    a["flags"] = strip_mtp(a["flags"]) + ["--act-slab-dtype", "bf16"]
    a["name"] = name
    a["env"]["LLAMA_FATTN_GQA8"] = "1"
    a["why"] = "strict recipe (G19 QKFBG): plain batch-major prefill (act_slab = 0 B), GQA8 decode attention, 2-slot initial cache; holds the whole-run keep-free rule at 200k / 250k"
    return a, spec


def speed(session, prompt_name, reps, arms_wanted):
    spec = mk_deep.make_speed(session, prompt_name, reps, EXE, ["STK"])
    base_spec, base_arms = mk_qa.make(session, prompt_name, EXE, ["QKFP"])
    common = list(base_spec["common"])
    for flag in ("--tg-tokens", "--tg-logit-margin"):
        common[common.index(flag) + 1] = str(NTOK)
    spec["common"] = common
    spec["tg_tokens"] = NTOK
    sc = list(spec["stock_common"])
    sc[sc.index("-n") + 1] = str(NTOK)
    spec["stock_common"] = sc
    arms = list(spec["arms"])                       # STK
    if "HF" in arms_wanted:
        arms.append(final_arm(base_arms[0], "HF"))
    if "HB" in arms_wanted:
        arms.append(strict_arm(session, prompt_name, "HB")[0])
    spec["arms"] = arms
    spec["control"] = "STK"
    save(session, spec)


def qa_strict(session, needle):
    a, spec = strict_arm(session, needle, "QKFBG")
    spec["control"] = "QKFBG"
    spec["arms"] = [a]
    save(session, spec)


if __name__ == "__main__":
    speed("g87-short-B", "nat-B.txt", 4, ["HF", "HB"])
    speed("g87-short-A", "nat-A.txt", 2, ["HF", "HB"])
    speed("g87-n24k", "natdeep-24000.txt", 3, ["HF", "HB"])
    speed("g87-n50k", "natdeep-50000.txt", 3, ["HB"])
    speed("g87-n100k", "natdeep-100000.txt", 3, ["HB"])
    speed("g87-n200k", "natdeep-200000.txt", 2, ["HB"])
    speed("g87-n250k", "natdeep-250000.txt", 1, ["HB"])
    # v2: the natural long prompts without backslashes (llama-completion processes escape sequences by default, the runner does not; the v1 24k prompt was 24,000 tokens for the stock arm and 24,004 for the runner)
    speed("g87-n24kv2", "natdeepv2-24000.txt", 3, ["HF", "HB"])
    speed("g87-n50kv2", "natdeepv2-50000.txt", 3, ["HB"])
    qa_strict("g87-qab24000", "anchor-24000.txt")
    qa_strict("g87-qab50000", "needle-50000.txt")
    qa_strict("g87-qab100000", "needle-100000.txt")
