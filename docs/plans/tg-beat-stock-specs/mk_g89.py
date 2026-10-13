"""G26 probe: why is the owner's llama-server (30.65 tok/s today) 19% slower than llama-completion with --no-mmap (36.9-37.2 tok/s) on the same flags otherwise?

    python mk_g89.py            writes build/llama-cuda/specs/g89-mmap.json and copies it next to this file

Hypothesis H (derived from the two observations of G24: g87-natprobe/g87-short-B vs g87-doc): the difference is the weight memory. The server maps the model file (load_mode mmap, the
expert tensors the fit moved to the CPU are read straight from the mapped pages) while the STK arm uses --no-mmap (heap copy). Prediction: the same llama-completion binary WITH the
mapping (`-lm mmap`, no --no-mmap) decodes within +-1.5 tok/s of 30.65. If it does not, the cause is somewhere else in the server (jinja chat template, parallel slots, logging at -lv 4,
batch size 8192, sampler chain) and the hybrid engine's choice of a mapped host copy (reg_bytes 18.3 GB, `mmap-registered`) is not implicated. Arms: STK (flags of the other G24 sessions),
STKM (the same without --no-mmap, with `-lm mmap` appended to override the common `-lm none`) and STKL (`-lm mmap+mlock`: mapped and locked, which separates soft page faults / working-set trimming from the mapping itself), 4 counterbalanced reps, prompt nat-B, 918 generated tokens, greedy.
This is an explanation probe of the BASELINE, not a claim about the hybrid engine; the hybrid's host leg is not changed by it.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_deep  # noqa: E402
import mk_g21   # noqa: E402

ROOT = mk_g21.root
EXE = "build/llama-cuda/bin/g45"
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")

spec = mk_deep.make_speed("g89-mmap", "nat-B.txt", 4, EXE, ["STK"])
sc = list(spec["stock_common"])
sc[sc.index("-n") + 1] = "918"
spec["stock_common"] = sc
spec["tg_tokens"] = 918
stk = spec["arms"][0]
stkm = copy.deepcopy(stk)
stkm["name"] = "STKM"
stkm["flags"] = ["-ncmoe", "37", "-t", "16", "-lm", "mmap"]
stkm["why"] = "stock llama-completion WITH the model mapped (-lm mmap, the server's default load mode), otherwise the owner's flags"
stkl = copy.deepcopy(stk)
stkl["name"] = "STKL"
stkl["flags"] = ["-ncmoe", "37", "-t", "16", "-lm", "mmap+mlock"]
stkl["why"] = "stock llama-completion with the model mapped AND locked in RAM (-lm mmap+mlock): separates page-fault / working-set trimming from the mapping itself"
spec["arms"] = [stk, stkm, stkl]
spec["control"] = "STK"
json.dump(spec, open(os.path.join(OUT, "g89-mmap.json"), "w"), indent=1)
json.dump(spec, open(os.path.join(HERE, "g89-mmap.json"), "w"), indent=1)
print("g89-mmap", [a["name"] for a in spec["arms"]], "reps", spec["reps"], "tg", spec["tg_tokens"])

# G26b (after g89-mmap REJECTED the mapping: STKM 35.64 and STKL 36.79 against STK 37.17 tok/s, the server's 30.65 is not the model mapping): the same stock flags with the OWNER'S OWN
# llama-completion.exe (build 10892, C:\Development\AI\llamacppCUDA, the binary that sits next to the llama-server of the document). If it decodes ~30.7 with --no-mmap too, the 21% is the
# BUILD (CPU backend / kernels of the owner's binary against the vendored tree's), not the server; if it decodes ~37, the difference is inside llama-server. Not attestable (an external
# binary): the wrapper is called with -AllowUnprovenanced (recorded).
spec_b = copy.deepcopy(spec)
spec_b["session"] = "g89b-ownerbin"
spec_b["exe_dir"] = "C:/Development/AI/llamacppCUDA"
own_nommap = copy.deepcopy(stk)
own_nommap["name"] = "OWN"
own_nommap["why"] = "the OWNER'S llama-completion.exe (build 10892) with the STK flags (--no-mmap)"
own_mmap = copy.deepcopy(stk)
own_mmap["name"] = "OWNM"
own_mmap["flags"] = ["-ncmoe", "37", "-t", "16", "-lm", "mmap"]
own_mmap["why"] = "the OWNER'S llama-completion.exe (build 10892), model mapped (the server's default load mode)"
spec_b["arms"] = [own_nommap, own_mmap]
spec_b["control"] = "OWN"
json.dump(spec_b, open(os.path.join(OUT, "g89b-ownerbin.json"), "w"), indent=1)
json.dump(spec_b, open(os.path.join(HERE, "g89b-ownerbin.json"), "w"), indent=1)
print("g89b-ownerbin", [a["name"] for a in spec_b["arms"]], "reps", spec_b["reps"], "exe_dir", spec_b["exe_dir"])
