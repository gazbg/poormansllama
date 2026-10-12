import json, copy, sys
root = r"C:\Development\AI\qwen38GZI\.claude\worktrees\attribution-handoff-doc-8ed1d1"
MODEL = "C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf"
FREQ = root.replace("\\", "/") + "/build/llama-cuda/aer-freq-train.txt"

def make(session, prompt_name, ubatch, tg, reps, arms_wanted, battery_ok, exe_dir, ctx=262144, lazy=True):
    prompt = root.replace("\\", "/") + "/build/llama-cuda/gplr07-qa-prompts/" + prompt_name
    common = ["--model", MODEL, "--prompt-file", prompt, "--tg-tokens", str(tg), "--tg-logit-margin", str(tg),
              "--residency", "moe-expert-bank", "--kv-cache", "q8", "--allow-full-residency", "--no-handoff",
              "--layer-slots", "2", "--tg-hybrid", "--ctx-size", str(ctx), "--act-slab-dtype", "f32"]
    arm_flags = ["--ubatch", str(ubatch), "--pp-schedule", "batch-major", "--pp-no-logits", "--kv-graph-pad", "256",
                 "--tg-hybrid-slots", "auto", "--tg-release-compute", "--tg-fast-sample"] + (["--kv-lazy"] if lazy else [])
    env = {"LLAMA_MOE_HYBRID": "14", "LLAMA_MOE_HYBRID_B": "16", "LLAMA_MOE_HYBRID_FREQ": FREQ,
           "LLAMA_MOE_HYBRID_LOG": "{sdir}/hybrid_{name}.jsonl", "LLAMA_CUDA_INPUT_BATCH": "1", "LLAMA_MOE_HYBRID_SCAN": "2"}
    defs = {
        "PLAIN": ([], "hybrid decode, no MTP (same flags, batch-major PP, auto slots)"),
        "MTP1":  (["--mtp", "--mtp-draft", "1"], "hybrid decode + MTP, 1 draft per round"),
        "MTP2":  (["--mtp", "--mtp-draft", "2"], "hybrid decode + MTP, 2 drafts per round"),
        "MTP3":  (["--mtp", "--mtp-draft", "3"], "hybrid decode + MTP, 3 drafts per round"),
        "MTP1G": (["--mtp", "--mtp-draft", "1", "--mtp-draft-experts-gpu"], "hybrid decode + MTP, 1 draft per round, the draft layer's routed experts on the GPU"),
        "MTP1U": (["--mtp", "--mtp-draft", "1"], "hybrid decode + MTP, 1 draft per round, mirror fill NOT fused into the next seed decode (LLAMA_MTP_FUSE=0)"),
        # G14: the host leg that serves the misses (LLAMA_MOE_HYBRID_LEAN: 0 graph, 1 lean team, 2 both compared bit for bit)
        "LEAN0":  ([], "hybrid decode, no MTP, host leg = the ggml graph (LLAMA_MOE_HYBRID_LEAN=0)"),
        "LEAN1":  ([], "hybrid decode, no MTP, host leg = the lean team (LLAMA_MOE_HYBRID_LEAN=1)"),
        "LEAN1P": ([], "LEAN1 with the team's stage timers (LLAMA_MOE_HYBRID_LEAN_PROF=1), diagnostic"),
        "LEAN2":  ([], "SELF-CHECK: both legs on every request, compared bit for bit (LLAMA_MOE_HYBRID_LEAN=2), not a timing arm"),
        "MTP1GL": (["--mtp", "--mtp-draft", "1", "--mtp-draft-experts-gpu"], "MTP1G with the lean host leg (LLAMA_MOE_HYBRID_LEAN=1)"),
        "MTP1GC": (["--mtp", "--mtp-draft", "1", "--mtp-draft-experts-gpu"], "SELF-CHECK of the lean leg under MTP: both legs on every request of the 2-row verify steps (LLAMA_MOE_HYBRID_LEAN=2), not a timing arm"),
        # G15: the lean leg (LEAN=1) with K predicted experts prefetched into the workers' caches after every reply (LLAMA_MOE_HYBRID_LEAN_PF=K)
        "PF0": ([], "lean host leg, no prefetch (LLAMA_MOE_HYBRID_LEAN=1, PF=0)"),
        "PF1": ([], "lean host leg + prefetch of K=1 predicted expert per reply"),
        "PF2": ([], "lean host leg + prefetch of K=2 predicted experts per reply (the G15 headline)"),
        "PF3": ([], "lean host leg + prefetch of K=3 predicted experts per reply"),
        "PF2P": ([], "PF2 with the team's stage timers (diagnostic)"),
        "LEAN2PF": ([], "SELF-CHECK: both legs on every request with the prefetch stage on (LEAN=2, PF=2), not a timing arm"),
        "MTP1GLP": (["--mtp", "--mtp-draft", "1", "--mtp-draft-experts-gpu"], "MTP1GL + prefetch K=2 (skipped whenever a second request is waiting, i.e. in every verify step)"),
        # team CPU layouts (the stall diagnostic showed every request over 3 ms waiting for the worker pinned to CPU 0, the core that takes the machine's interrupts)
        "PF2B": ([], "PF2 with the team on CPUs 2,4,..,14 and 16 (core 0, the interrupt core, left alone; the 8th thread on CCD1)"),
        "PF2C": ([], "PF2 with 7 threads on CPUs 2,4,..,14 (cores 0 left alone)"),
        "PF2D": ([], "PF2 with the team on CPUs 4,6,..,16,18 (cores 0 and 1 left alone)"),
        "PF2M": ([], "PF2 (team on 0,2,..,14) with the decoding thread pinned to CPU 18"),
        "PF2BM": ([], "PF2B (team on 2,4,..,14,16) with the decoding thread pinned to CPU 18"),
        "PF2BN": ([], "PF2B with the decoding thread pinned to CPU 22 (core 11)"),
        # the build g34 makes the layout that leaves core 0 alone and the pinned decoding thread the DEFAULT of the lean team; this arm restores the old one
        "PF2A": ([], "PF2 with the pre-G16 placement: team on CPUs 0,2,..,14, decoding thread unpinned"),
        # G17c: cache policy variants (all on top of LEAN=1, PF=2)
        "PF2N": (["--tg-probe-nosample"], "PF2 with the sampling probe: no logits copy, no host top-2, no per-step print (MEASUREMENT, the text is meaningless)"),
        "P0": ([], "policy baseline: uniform split, B=16"),
        "P1": ([], "policy: per-layer split from the pooled calibration misses (alpha 0.5), B=16"),
        "P2": ([], "policy: split (alpha 0.5), B=24"),
        "P3": ([], "policy: split (alpha 0.5), B=32, recency filter"),
        "P4": ([], "policy: uniform split, B=32, recency filter"),
    }
    lean_env = {"LEAN0": "0", "LEAN1": "1", "LEAN1P": "1", "LEAN2": "2", "MTP1GL": "1", "MTP1GC": "2",
                "PF0": "1", "PF1": "1", "PF2": "1", "PF3": "1", "PF2P": "1", "LEAN2PF": "2", "MTP1GLP": "1", "PF2B": "1", "PF2C": "1", "PF2D": "1", "PF2M": "1", "PF2BM": "1", "PF2BN": "1", "PF2A": "1", "PF2N": "1",
                "P0": "1", "P1": "1", "P2": "1", "P3": "1", "P4": "1"}
    pf_env = {"PF1": "1", "PF2": "2", "PF3": "3", "PF2P": "2", "LEAN2PF": "2", "MTP1GLP": "2", "PF2B": "2", "PF2C": "2", "PF2D": "2", "PF2M": "2", "PF2BM": "2", "PF2BN": "2", "PF2A": "2", "PF2N": "2", "P0": "2", "P1": "2", "P2": "2", "P3": "2", "P4": "2"}
    cpu_layouts = {"PF2B": "2,4,6,8,10,12,14,16", "PF2C": "2,4,6,8,10,12,14", "PF2D": "4,6,8,10,12,14,16,18", "PF2BM": "2,4,6,8,10,12,14,16", "PF2BN": "2,4,6,8,10,12,14,16"}
    main_cpus = {"PF2M": "18", "PF2BM": "18", "PF2BN": "22", "PF2A": "-1"}
    cpu_layouts["PF2A"] = "0,2,4,6,8,10,12,14"
    split_file = root.replace("\\", "/") + "/build/llama-cuda/aer-split-misses-pooled.txt"
    pol_env = {"P1": {"LLAMA_MOE_HYBRID_SPLIT": split_file, "LLAMA_MOE_HYBRID_SPLIT_ALPHA": "0.5"},
               "P2": {"LLAMA_MOE_HYBRID_SPLIT": split_file, "LLAMA_MOE_HYBRID_SPLIT_ALPHA": "0.5", "LLAMA_MOE_HYBRID_B": "24"},
               "P3": {"LLAMA_MOE_HYBRID_SPLIT": split_file, "LLAMA_MOE_HYBRID_SPLIT_ALPHA": "0.5", "LLAMA_MOE_HYBRID_B": "32", "LLAMA_MOE_HYBRID_RECENCY": "1"},
               "P4": {"LLAMA_MOE_HYBRID_B": "32", "LLAMA_MOE_HYBRID_RECENCY": "1"}}
    arms = []
    for name in arms_wanted:
        extra, why = defs[name]
        aenv = dict(env)
        if name == "MTP1U":
            aenv["LLAMA_MTP_FUSE"] = "0"
        if name in lean_env:
            aenv["LLAMA_MOE_HYBRID_LEAN"] = lean_env[name]
        if name in ("LEAN1P", "PF2P"):
            aenv["LLAMA_MOE_HYBRID_LEAN_PROF"] = "1"
        if name in pf_env:
            aenv["LLAMA_MOE_HYBRID_LEAN_PF"] = pf_env[name]
        if name in cpu_layouts:
            aenv["LLAMA_MOE_HYBRID_LEAN_CPUS"] = cpu_layouts[name]
        if name in main_cpus:
            aenv["LLAMA_MOE_HYBRID_MAIN_CPU"] = main_cpus[name]
        if name in pol_env:
            aenv.update(pol_env[name])
        if name in ("LEAN2", "MTP1GC", "LEAN2PF"):
            # both legs run on every request: the lean team must NOT share cores with the graph's spinning thread pool (CPUs 0,2,..,14), or the two pools time-slice each other
            # (first attempt: 2.7 s per token, 100 ms per request). CCD1, away from the scan helper on CPU 20; the last CPU is the engine thread's.
            aenv["LLAMA_MOE_HYBRID_LEAN_CPUS"] = "16,18,22,24,26,28,30"
        arms.append({"name": name, "flags": arm_flags + extra, "env": aenv, "why": why})
    spec = {"session": session, "exe_dir": exe_dir, "tg_tokens": tg, "control": arms_wanted[0], "qa": False,
            "items": "N1,N2,N3,N4,K1", "prompt": "build/llama-cuda/gplr07-qa-prompts/" + prompt_name,
            "common": common, "reps": reps, "interleave": True, "counterbalance": True, "arms": arms, "poll_ms": 50}
    if battery_ok:
        spec["allow_battery"] = True
    return spec

if __name__ == "__main__":
    session, prompt_name, ubatch, tg, reps, arms, battery, exe = sys.argv[1:9]
    ctx = int(sys.argv[9]) if len(sys.argv) > 9 else 262144
    lazy = (sys.argv[10] != "0") if len(sys.argv) > 10 else True
    spec = make(session, prompt_name, int(ubatch), int(tg), int(reps), arms.split(","), battery == "1", exe, ctx, lazy)
    out = root + r"\build\llama-cuda\specs\%s.json" % session
    json.dump(spec, open(out, "w"), indent=1)
    print(out, [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", tg, "ubatch", ubatch)
