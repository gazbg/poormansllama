"""G24 prompt probe (STOCK engine only): which prompt makes the greedy 918-token continuation prose instead of a loop?

    powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag g87-natprobe -Command "python docs\\plans\\tg-beat-stock-specs\\g87_natprobe.py <exe_dir> <prompt> [<prompt> ...]"

Runs `llama-completion` (the STK arm's flags: the owner's document configuration, -ncmoe 37, greedy, --ignore-eos, 918 generated tokens = the document's
generation length) once per prompt, keeps the generated text, and scores repetition with two engine-independent measures:
  rep6   = 1 - distinct/total over word 6-grams of the generated text (the plan's G18/G19 "6-gram repetition ratio": 0 = never repeats, 1 = pure loop)
  zratio = zlib-compressed bytes / raw bytes of the generated text (prose ~0.4-0.5, a loop far below)
The probe decides the G24 doc-shape prompt BEFORE any hybrid run: the lowest-rep6 candidate that is not worse than 0.5 on either measure (rule fixed in the
G24 contract). Results: build/llama-cuda/g87-natprobe/<prompt>.{out,err}.txt and summary.json.
"""
import json
import os
import re
import subprocess
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402

ROOT = mk_g21.root
MODEL = mk_g21.MODEL
PROMPTS = os.path.join(ROOT, "build", "llama-cuda", "gplr07-qa-prompts")
OUT = os.path.join(ROOT, "build", "llama-cuda", "g87-natprobe")
N_GEN = 918


def rep6(text):
    w = text.split()
    grams = [tuple(w[i:i + 6]) for i in range(len(w) - 5)]
    return (1.0 - len(set(grams)) / len(grams)) if grams else float("nan"), len(w)


def main():
    exe_dir, names = sys.argv[1], sys.argv[2:]
    exe = os.path.join(ROOT, exe_dir, "llama-completion.exe") if not os.path.isabs(exe_dir) else os.path.join(exe_dir, "llama-completion.exe")
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for name in names:
        prompt = os.path.join(PROMPTS, name + ".txt")
        cmd = [exe, "-m", MODEL, "-f", prompt, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off", "-c", "262000", "-n", str(N_GEN),
               "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt", "-ncmoe", "37", "-t", "16", "--no-mmap"]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True)
        wall = time.time() - t0
        out = r.stdout.decode("utf-8", errors="replace")
        err = r.stderr.decode("utf-8", errors="replace")
        open(os.path.join(OUT, name + ".out.txt"), "w", encoding="utf-8").write(out)
        open(os.path.join(OUT, name + ".err.txt"), "w", encoding="utf-8").write(err)
        m = re.search(r"eval time =\s*([\d.]+) ms /\s*(\d+) runs.*?([\d.]+) tokens per second", err.replace("prompt eval time", "PROMPT"))
        p = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", err)
        ratio, nwords = rep6(out)
        z = len(zlib.compress(out.encode("utf-8"), 9)) / max(1, len(out.encode("utf-8")))
        summary[name] = {"rc": r.returncode, "wall_s": round(wall, 1), "prompt_tokens": int(p.group(2)) if p else None,
                         "tg_tokens": int(m.group(2)) if m else None, "tg_tok_s": float(m.group(3)) if m else None,
                         "rep6": round(ratio, 4), "zratio": round(z, 4), "words": nwords, "first_80": out[:80].replace("\n", " ")}
        print(name, json.dumps(summary[name]), flush=True)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
