"""Answer-stability gate (needle QA, programmatic grading, negative control failing as designed) for the FINAL configuration:

    python mk_qa.py <session> <prompt file name> <exe_dir> <ARMS comma>

arms: QKFP (the final plain configuration under the keep-free rule: lean host leg + K=2 prefetch, async scan, fast sampler, input batching, lazy KV, layer-major prefill),
      QKFM (the same with MTP self-speculation: batch-major prefill, draft layer's experts on the GPU).
The grading is tools/semantic-qa/check-answer.py through tools/run-tg-arms.py ("qa": true); 128 generated tokens; ctx 262,144.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402

root = mk_g21.root
MODEL = mk_g21.MODEL
FREQ = mk_g21.FREQ


def make(session, prompt_name, exe_dir, arms_wanted):
    prompt = root.replace("\\", "/") + "/build/llama-cuda/gplr07-qa-prompts/" + prompt_name
    base_common = ["--model", MODEL, "--prompt-file", prompt, "--tg-tokens", "128", "--tg-logit-margin", "128", "--residency", "moe-expert-bank", "--kv-cache", "q8",
                   "--allow-full-residency", "--no-handoff", "--layer-slots", "2", "--tg-hybrid", "--ctx-size", "262144", "--act-slab-dtype", "f32"]
    env = {"LLAMA_MOE_HYBRID": "14", "LLAMA_MOE_HYBRID_B": "16", "LLAMA_MOE_HYBRID_FREQ": FREQ, "LLAMA_MOE_HYBRID_LOG": "{sdir}/hybrid_{name}.jsonl",
           "LLAMA_CUDA_INPUT_BATCH": "1", "LLAMA_MOE_HYBRID_SCAN": "2", "LLAMA_MOE_HYBRID_LEAN": "1", "LLAMA_MOE_HYBRID_LEAN_PF": "2", "LLAMA_TG_FREE_WATCH_MS": "20"}
    keepfree = ["--kv-graph-pad", "256", "--tg-hybrid-slots", "auto", "--tg-release-compute", "--tg-fast-sample", "--kv-lazy", "--tg-vram-cap-mib", "0", "--tg-vram-keep-free-mib", "16000"]
    arms = []
    for name in arms_wanted:
        if name == "QKFP":
            arms.append({"name": name, "flags": ["--ubatch", "512", "--pp-schedule", "layer-major", "--tg-first-pp-token"] + keepfree, "env": dict(env),
                         "why": "final plain configuration, layer-major prefill"})
        elif name == "QKFM":
            arms.append({"name": name, "flags": ["--ubatch", "512", "--pp-schedule", "batch-major", "--pp-no-logits"] + keepfree + ["--mtp", "--mtp-draft", "1", "--mtp-draft-experts-gpu"],
                         "env": dict(env), "why": "final configuration with MTP self-speculation, batch-major prefill"})
        else:
            raise SystemExit("unknown arm " + name)
    return {"session": session, "exe_dir": exe_dir, "tg_tokens": 128, "control": arms_wanted[0], "qa": True, "items": "N1,N2,N3,N4,K1",
            "prompt": "build/llama-cuda/gplr07-qa-prompts/" + prompt_name, "common": base_common, "reps": 1, "interleave": False, "poll_ms": 50}, arms


if __name__ == "__main__":
    session, prompt_name, exe, arms = sys.argv[1:5]
    spec, arm_list = make(session, prompt_name, exe, arms.split(","))
    spec["arms"] = arm_list
    out = root + r"\build\llama-cuda\specs\%s.json" % session
    json.dump(spec, open(out, "w"), indent=1)
    print(out, [a["name"] for a in arm_list])
