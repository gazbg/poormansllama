"""G30 / M3 sessions: MTP on top of the CURRENT OPTIMAL solution of each prefill band (build g48 = g47 + stage-c/g30_mtp_catchup_apply + g30_gqa8_rows_apply).

    python mk_g94.py            writes build/llama-cuda/specs/g94-*.json and copies them next to this file

Two arms per session, counterbalanced in the same session, differing in nothing but MTP:
  B0   the band's best non-MTP configuration: the prefill of the band + the keep-free hybrid decode (lean host leg, K = 2 prefetch, lazy KV, GQA8 on), `--pp-no-logits` and NO `--tg-first-pp-token`
       (the MTP loop's first-token convention, D99: the same convention in both arms, so a cell compares like with like)
  M    B0 + `--mtp --mtp-draft 1 --mtp-draft-experts-gpu`
Bands (the prefill that wins there, G21 / G23 / G28):
  stock  prompt <= ~600 tokens : the stock graph path (`--pp-stock-below 4096`, op offload on, ubatch 512); batch-major schedule so that the MTP branch takes the stock arm
  inter  ~500-2,100 tokens     : the paced CPU/GPU interleave (`--pp-cpu-layers <plan>`, paced copies 16 MiB x 2, f32 slab) with the plan of the G23 model (HY4Q / HY8Q of g68)
  lm     ~4,000-24,000 tokens  : the unsegmented fast layer-major prefill (ubatch 2048, f32 slab, 2 slots)
  seg    >= 50,000 tokens      : the segmented layer-major prefill (ubatch 2048, segments of 16,384; 200k: ubatch 1024), the strict bars
Cells: code and prose per length; 512 generated tokens; the prefill is ONE pass (`--pp-repeat` is refused with --mtp), cold in both arms.
Extra arm for the catch-up question: `Mt` = M with `--mtp-mirror-tail 2048` (the draft head mirrors only the last 2,048 positions), made only for the cells named in TAIL_CELLS.
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
EXE = os.environ.get("G94_EXE", "build/llama-cuda/bin/g48")
PREFIX = os.environ.get("G94_PREFIX", "g94")        # session-name prefix: g94 = round 1 (build g48), g95 = round 2 (build g49: + stage-c/g30c, g30d)
NTOK = 512
MTP_FLAGS = ["--mtp", "--mtp-draft", "1", "--mtp-draft-experts-gpu"]
PACE = ["--pp-copy-piece-mib", "16", "--pp-copy-window", "2"]
KEEPFREE = ["--kv-graph-pad", "256", "--tg-hybrid-slots", "auto", "--tg-release-compute", "--tg-fast-sample", "--kv-lazy", "--tg-vram-cap-mib", "0", "--tg-vram-keep-free-mib", "16000"]
PLANS = {511: ("0,3,7,9,14,18,20,25,28,33,36", 4), 1023: ("0,6,12,18,25,33", 4), 2047: ("0,12,26", 8)}


def save(name, spec):
    data = json.dumps(spec, indent=1).encode("utf-8")      # binary writes: text mode would turn LF into CRLF on Windows
    open(os.path.join(OUT, name + ".json"), "wb").write(data)
    open(os.path.join(HERE, name + ".json"), "wb").write(data)
    print(name, [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", spec["tg_tokens"], "prompt", spec["prompt"].split("/")[-1])


def band_flags(band, tokens):
    if band == "stock":
        return ["--ubatch", "512", "--pp-schedule", "batch-major", "--pp-no-logits", "--pp-stock-below", "4096"] + KEEPFREE + ["--layer-slots", "2", "--act-slab-dtype", "f32"]
    if band == "inter":
        plan, slots = PLANS[tokens]
        return ["--ubatch", "2048", "--pp-schedule", "layer-major", "--pp-no-logits"] + KEEPFREE + ["--layer-slots", str(slots), "--act-slab-dtype", "f32", "--pp-cpu-layers", plan] + PACE
    if band == "lm":
        return ["--ubatch", "2048", "--pp-schedule", "layer-major", "--pp-no-logits"] + KEEPFREE + ["--layer-slots", "2", "--act-slab-dtype", "f32"]
    if band == "seg":
        ub = "1024" if tokens >= 150000 else "2048"
        return ["--ubatch", ub, "--pp-schedule", "layer-major", "--pp-no-logits"] + KEEPFREE + ["--layer-slots", "2", "--act-slab-dtype", "f32", "--pp-segment-tokens", "16384"]
    raise SystemExit("band " + band)


def session(name, prompt_name, band, tokens, reps, tail=False, hybrid_env="14", ntok=NTOK, only_m=False):
    name = PREFIX + name[3:] if name.startswith("g94") else name
    hb, spec = mk_g87.strict_arm(name, prompt_name, "B0")        # env + common of the validated strict arm, then replaced below
    spec = copy.deepcopy(spec)
    env = dict(hb["env"])
    env["LLAMA_MOE_HYBRID"] = hybrid_env
    env["LLAMA_FATTN_GQA8"] = "1"
    b0 = {"name": "B0", "flags": band_flags(band, tokens), "env": dict(env), "why": "band '%s' best non-MTP configuration (keep-free hybrid decode, GQA8)" % band}
    m = copy.deepcopy(b0)
    m["name"] = "M"
    m["flags"] = b0["flags"] + MTP_FLAGS
    m["why"] = "B0 + MTP self-speculation (1 draft, draft-layer experts on the GPU), the same prefill"
    arms = [m] if only_m else [b0, m]
    if tail and not only_m:
        mt = copy.deepcopy(m)
        mt["name"] = "Mt"
        mt["flags"] = m["flags"] + ["--mtp-mirror-tail", "2048"]
        mt["why"] = "M with the draft head mirroring only the last 2,048 prompt positions"
        arms.append(mt)
    common = list(spec["common"])
    for flag in ("--tg-tokens", "--tg-logit-margin"):
        common[common.index(flag) + 1] = str(ntok)
    for flag in ("--layer-slots", "--act-slab-dtype"):
        if flag in common:
            i = common.index(flag)
            del common[i:i + 2]
    spec.update({"session": name, "exe_dir": EXE, "tg_tokens": ntok, "common": common, "control": arms[0]["name"], "qa": False, "reps": reps, "interleave": True,
                 "counterbalance": True, "arms": arms})
    spec["prompt"] = "build/llama-cuda/gplr07-qa-prompts/" + prompt_name
    save(name, spec)


# (session, prompt, band, tokens, reps, tail?)
CELLS = [
    ("g94-sc24", "div-code.txt", "stock", 24, 4, False),
    ("g94-sp24", "div-prose.txt", "stock", 24, 4, False),
    ("g94-sc500", "mtpcode-512.txt", "stock", 511, 3, False),
    ("g94-sp500", "mtpprose-512.txt", "stock", 511, 3, False),
    ("g94-ic1k", "mtpcode-1024.txt", "inter", 1023, 3, False),
    ("g94-ip1k", "mtpprose-1024.txt", "inter", 1023, 3, False),
    ("g94-ic2k", "mtpcode-2048.txt", "inter", 2047, 3, False),
    ("g94-ip2k", "mtpprose-2048.txt", "inter", 2047, 3, False),
    ("g94-lc4k", "mtpcode-4096.txt", "lm", 4093, 3, False),
    ("g94-lp4k", "mtpprose-4096.txt", "lm", 4093, 3, False),
    ("g94-lc24k", "mtpcode-24000.txt", "lm", 23995, 3, True),
    ("g94-lp24k", "mtpprose-24000.txt", "lm", 23995, 3, False),
    ("g94-sgc50k", "mtpcode-50000.txt", "seg", 49999, 2, True),
    ("g94-sgp50k", "mtpprose-50000.txt", "seg", 49999, 2, False),
]

if __name__ == "__main__":
    args = sys.argv[1:]
    smoke = bool(args) and args[0] == "--smoke"        # one M run of 64 tokens per named cell: function check, session name g94-smoke-<cell>
    only = set(args[1:] if smoke else args)
    for name, prompt, band, tokens, reps, tail in CELLS:
        if only and name not in only:
            continue
        if smoke:
            session(PREFIX + "-smoke-" + name[4:], prompt, band, tokens, 1, False, ntok=64, only_m=True)
        else:
            session(name, prompt, band, tokens, reps, tail)
