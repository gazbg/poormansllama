"""G24 natural-text prompts (doc shape): ~839 prompt tokens whose greedy continuation is prose, not a loop.

    python mk_g87_prompts.py            writes build/llama-cuda/gplr07-qa-prompts/nat-{A,B,C}.txt (no trailing newline) and prints their exact token counts

Why. The speed prompts of G18-G21 (div-code.txt, div-prose.txt, deepcode-<N>.txt) end in a code or essay continuation that loops under greedy
`--ignore-eos` decoding (6-gram repetition ratio 0.91-1.0, plan G18/G19). A looping text routes through the same experts again and again, which
flatters an expert cache. The owner's document baseline is a natural chat (thinking mode, sampled), 839 prompt tokens, 918 generated tokens. These
prompts reproduce that SHAPE with text the repository already contains (so nothing is invented) and are scored for repetition on STOCK before any
hybrid run (g87-natprobe), so the choice of the prompt does not depend on the hybrid engine.

A: chat template, thinking mode (assistant turn opened with <think>), a design-notes excerpt to critique (qwen38-layer-streaming-sdd.md)
B: chat template, thinking mode (the prompt ends right after <think>, no newline: llama-completion -f strips one trailing newline), a performance review excerpt to summarise for a non-expert and turn into an action list
C: no chat template: the same kind of excerpt as raw text that continues into a plain-language explanation (layer-residency-sdd.md)
The excerpt length is calibrated so that the whole prompt has TARGET tokens under the model's own tokenizer (llama-tokenize, special tokens parsed, BOS rules
of the model), the same call the runner and llama-completion make (add_special=true, parse_special=true).
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402

ROOT = mk_g21.root
MODEL = mk_g21.MODEL
TOK = os.path.join(ROOT, "build", "llama-cuda", "bin", "g45", "llama-tokenize.exe")
OUT = os.path.join(ROOT, "build", "llama-cuda", "gplr07-qa-prompts")
TARGET = 839


def count_tokens(text):
    tmp = os.path.join(OUT, "_tok_probe.txt")
    open(tmp, "wb").write(text.encode("utf-8"))
    r = subprocess.run([TOK, "-m", MODEL, "-f", tmp, "--show-count", "--ids"], capture_output=True, text=True)
    m = re.search(r"Total number of tokens:\s*(\d+)", r.stdout + r.stderr)
    if not m:
        raise SystemExit("tokenize failed: " + (r.stdout + r.stderr)[-400:])
    return int(m.group(1))


def paragraphs(path):
    text = open(os.path.join(ROOT, path), "rb").read().decode("utf-8")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 60 and not p.strip().startswith(("|", "```", "#"))]
    return paras


def calibrate(build, paras, target):
    """Largest number of paragraphs (cut at a sentence boundary for the last one) whose full prompt stays <= target; then pad the last one with words to hit it exactly."""
    lo, hi = 1, len(paras)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_tokens(build("\n\n".join(paras[:mid])))
        if n <= target:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    body = "\n\n".join(paras[:best])
    # add words of the next paragraph until the target is reached exactly
    words = paras[best].split(" ") if best < len(paras) else []
    cur = body
    n = count_tokens(build(cur))
    k = 0
    while n < target and k < len(words):
        step = max(1, min(8, (target - n)))
        k += step
        cur = body + "\n\n" + " ".join(words[:k])
        n = count_tokens(build(cur))
        if n > target:
            k -= step
            # fall back to word-by-word
            for j in range(1, step + 1):
                cur = body + "\n\n" + " ".join(words[:k + j])
                n = count_tokens(build(cur))
                if n > target:
                    cur = body + "\n\n" + " ".join(words[:k + j - 1])
                    n = count_tokens(build(cur))
                    break
            break
    return build(cur), n


def chat(user_head, user_tail):
    return lambda body: "<|im_start|>user\n" + user_head + "\n\n" + body + "\n\n" + user_tail + "<|im_end|>\n<|im_start|>assistant\n<think>"


def raw(tail):
    return lambda body: body + "\n\n" + tail


def main():
    os.makedirs(OUT, exist_ok=True)
    specs = {
        "nat-A": (chat("Here are design notes for a GPU inference engine.",
                       "Write a thorough critical review of these notes. Identify the weakest assumptions, the likely failure modes, and propose concrete experiments to check each one. Be detailed."),
                  "docs/qwen38-layer-streaming-sdd.md"),
        "nat-B": (chat("Here is a performance review of a language-model inference stack.",
                       "Summarise the findings in plain language for a reader who is not an expert, explain the reasoning behind each, and then give a prioritised action list."),
                  "docs/inference-performance-review-2026-09-11.md"),
        "nat-C": (raw("In plain language, the idea above works for the following reasons, and it can fail in these situations:"),
                  "docs/layer-residency-sdd.md"),
    }
    for name, (build, src) in specs.items():
        text, n = calibrate(build, paragraphs(src), TARGET)
        path = os.path.join(OUT, name + ".txt")
        open(path, "wb").write(text.encode("utf-8"))
        print(name, "tokens", n, "bytes", len(text.encode("utf-8")), path)


if __name__ == "__main__":
    main()
