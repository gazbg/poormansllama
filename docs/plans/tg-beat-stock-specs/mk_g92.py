"""G29 confirmation session: the one arm that failed on build g45 (g88-bm200k BM1024, rc=1: duplicate staging of layer 26 in the async-feed queue) rerun on build g47 (the guard is unconditional there).
    python mk_g92.py       writes build/llama-cuda/specs/g92-bm1024-200k.json and copies it next to this file
Same spec as g88-bm200k with the build changed and only the BM1024 arm kept, so any difference is the build."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "build", "llama-cuda", "specs")

spec = json.load(open(os.path.join(OUT, "g88-bm200k.json")))
spec["session"] = "g92-bm1024-200k"
spec["exe_dir"] = "build/llama-cuda/bin/g47"
spec["arms"] = [a for a in spec["arms"] if a["name"] == "BM1024"]
assert len(spec["arms"]) == 1
spec["arms"][0]["why"] = "G29 confirmation: g88-bm200k BM1024 (rc=1 on g45) on build g47"
for name in ("g92-bm1024-200k",):
    # binary writes: text mode would turn LF into CRLF on Windows
    data = json.dumps(spec, indent=1).encode("utf-8")
    open(os.path.join(OUT, name + ".json"), "wb").write(data)
    open(os.path.join(HERE, name + ".json"), "wb").write(data)
print("g92-bm1024-200k", [a["name"] for a in spec["arms"]])
