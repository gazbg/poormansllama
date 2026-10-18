"""G30 / M3 round 2, supplements on build g52 (sessions g102-*): the stock-band 500-token code cell with 8 reps (its g99 session had one pair at 0.978), and the deep code cells at 100k and 200k tokens.

    python mk_g102.py          writes build/llama-cuda/specs/g102-*.json and copies them next to this file
Arms: B0, MF (final recipe), MFt (MF + --mtp-mirror-tail 2048). The deep cells run the segmented prefill (ubatch 2048 / 1024 at 200k, segments of 16,384) like g99-sgc50k.
"""
import json
import os
import sys

os.environ["G94_PREFIX"] = "g102"
os.environ.setdefault("G94_EXE", "build/llama-cuda/bin/g52")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g94  # noqa: E402
import mk_g99  # noqa: E402

for name, prompt, band, tokens, reps in (("g94-sc500", "mtpcode-512.txt", "stock", 511, 8),
                                         ("g94-sgc100k", "mtpcode-100000.txt", "seg", 99990, 2),
                                         ("g94-sgc200k", "mtpcode-200000.txt", "seg", 199995, 2)):
    mk_g94.session(name, prompt, band, tokens, reps, False)
    real = mk_g94.PREFIX + name[3:]
    mk_g99.final_arms(real, tokens >= 20000)
    path = os.path.join(mk_g94.OUT, real + ".json")
    spec = json.load(open(path))
    spec["arms"] = [a for a in spec["arms"] if a["name"] != "MT"]        # the pruned-vocabulary question is settled at 24k / 50k (g99)
    mk_g94.save(real, spec)
