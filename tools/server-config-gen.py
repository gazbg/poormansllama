#!/usr/bin/env python3
"""Write a server-probe config: python tools/server-config-gen.py <out.json> [--exe EXE] [--no-hybrid] [--drop ENVKEY ...] [--env K=V ...] [--arg ARG ...]"""
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--exe", default="build/llama-cuda/bin/Release/llama-server.exe")
ap.add_argument("--no-hybrid", action="store_true")
ap.add_argument("--drop", action="append", default=[])
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--arg", action="append", default=[])
ap.add_argument("--ctx", default="262000")
ap.add_argument("--no-override", action="store_true")
ap.add_argument("--np", default="1", help="--parallel value")
ap.add_argument("--wrap", action="append", default=[], help="a command prefix (e.g. the compute-sanitizer and its options), one --wrap per word")
ap.add_argument("--ub", default="512", help="-ub value; 0 omits the flag (the server default, or the --hybrid-moe preset's)")
ap.add_argument("--batch", default="8192", help="-b value; 0 omits the flag (the server default, or the --lm-prefill preset's)")
a = ap.parse_args()
root = os.getcwd().replace("\\", "/")
model = "C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf"
args = ["--model", model, "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-c", a.ctx] + ([] if a.batch == "0" else ["-b", a.batch]) + ["-np", a.np] + ([] if a.ub == "0" else ["-ub", a.ub])
if not a.no_override:
    args += ["-ot", r"blk\..*\.ffn_(gate|up|down|gate_up)_exps\.weight=CPU", "-fit", "off"]
env = {}
if not a.no_hybrid:
    env = {"LLAMA_MOE_HYBRID": "14", "LLAMA_MOE_HYBRID_B": "16", "LLAMA_MOE_HYBRID_FREQ": root + "/build/llama-cuda/aer-freq-train.txt", "LLAMA_CUDA_INPUT_BATCH": "1",
           "LLAMA_MOE_HYBRID_SCAN": "2", "LLAMA_MOE_HYBRID_LEAN": "1", "LLAMA_MOE_HYBRID_LEAN_PF": "2", "LLAMA_FATTN_GQA8": "1"}
for k in a.drop:
    env.pop(k, None)
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v
args += a.arg
exe = a.exe if os.path.isabs(a.exe) else root + "/" + a.exe
cfg = {"exe": exe, "args": args, "env": env}
if a.wrap:
    cfg["wrap"] = a.wrap
json.dump(cfg, open(a.out, "w"), indent=1)
print(a.out, "env keys:", sorted(env))
