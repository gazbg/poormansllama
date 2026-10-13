"""G24 natural long-context prompts: real repository prose at a chosen exact token count, then an instruction whose answer is natural reasoning text.

    python mk_g87_natdeep.py [N ...]       default N = 24000 50000 100000 200000 250000; writes build/llama-cuda/gplr07-qa-prompts/natdeep-<N>.txt

The speed prompts of G19-G21 (deepcode-<N>.txt) are a synthetic filler haystack followed by a code prompt, and the continuation loops under greedy
`--ignore-eos` decoding. A looping continuation flatters an expert cache, so G24 also measures on text that is long, real and natural: the repository's own
documents (docs/*.md, docs/plans/*.md, tools/*.py ... in a fixed order, headed by `=== FILE: <path> ===`), cut at a word boundary so that the WHOLE prompt has
exactly N tokens under the model's tokenizer (llama-tokenize, special tokens parsed, the call the runner and llama-completion make), inside the chat template
of the model with the assistant turn opened in thinking mode:

    <|im_start|>user\\n{preface}\\n\\n{documents}\\n\\n{instruction}<|im_end|>\\n<|im_start|>assistant\\n<think>

No trailing newline (llama-completion -f strips one). The literal sequences `<|` and `<think>` / `</think>` inside the documents are broken up (`< |`, `< think>`)
so that no special token can appear inside the document part. Both engines receive the same bytes.
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
PREFACE = "Below are the working documents of a software research project."
INSTRUCTION = ("Write a thorough review of the project documents above: summarise the goals and the main results, explain the reasoning behind them in plain language, "
               "point out the biggest risks and open questions, and propose the next experiments. Be detailed.")


def count_tokens(text):
    tmp = os.path.join(OUT, "_tok_probe_deep.txt")
    open(tmp, "wb").write(text.encode("utf-8"))
    r = subprocess.run([TOK, "-m", MODEL, "-f", tmp, "--show-count", "--ids"], capture_output=True, text=True)
    m = re.search(r"Total number of tokens:\s*(\d+)", r.stdout + r.stderr)
    if not m:
        raise SystemExit("tokenize failed: " + (r.stdout + r.stderr)[-400:])
    return int(m.group(1))


def corpus_files():
    files = []
    for sub in ("docs", "docs/plans", "docs/findings", "docs/currentstate"):
        d = os.path.join(ROOT, sub.replace("/", os.sep))
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p) and name.endswith((".md", ".txt")) and "nested-worktree" not in name:
                files.append(p)
    return files


def corpus_text():
    parts = []
    for p in corpus_files():
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        t = open(p, "rb").read().decode("utf-8", errors="replace").replace("\r\n", "\n")
        t = t.replace("<|", "< |").replace("<think>", "< think>").replace("</think>", "< /think>")
        # llama-completion (and llama-tokenize) process escape sequences of the prompt by default (-e), the runner does not: a backslash makes the two engines read different
        # text (measured 2026-10-13, g87-n24k: the runner tokenized 24,004 tokens where llama-completion and llama-tokenize counted 24,000). No backslash, no difference.
        t = t.replace("\\", "/")
        parts.append("=== FILE: %s ===\n%s" % (rel, t))
    return "\n\n".join(parts)


def build(doc):
    return "<|im_start|>user\n" + PREFACE + "\n\n" + doc + "\n\n" + INSTRUCTION + "<|im_end|>\n<|im_start|>assistant\n<think>"


def make(n_target, words):
    cache = {}

    def cnt(k):
        if k not in cache:
            cache[k] = count_tokens(build(" ".join(words[:k])))
        return cache[k]

    k = min(len(words), int(n_target * 0.85))
    n = cnt(k)
    for _ in range(3):
        k2 = min(len(words), max(1, int(k * n_target / n)))
        if k2 == k:
            break
        k, n = k2, cnt(k2)
        if abs(n - n_target) <= 200:
            break
    if n < n_target and k >= len(words):
        raise SystemExit("corpus too small for %d tokens (max %d)" % (n_target, n))
    # bracket and bisect on the word count to land exactly on n_target
    lo, hi = k, k
    while cnt(lo) > n_target:
        lo = max(1, int(lo * 0.99))
    while cnt(hi) < n_target and hi < len(words):
        hi = min(len(words), int(hi * 1.01) + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cnt(mid) <= n_target:
            lo = mid
        else:
            hi = mid
    return build(" ".join(words[:lo])), cnt(lo)


def main():
    args = sys.argv[1:]
    prefix = "natdeep"
    if args and args[0] == "--prefix":
        prefix, args = args[1], args[2:]
    lengths = [int(a) for a in args] or [24000, 50000, 100000, 200000, 250000]
    os.makedirs(OUT, exist_ok=True)
    words = corpus_text().split(" ")
    print("corpus words", len(words), "files", len(corpus_files()))
    for n in lengths:
        path = os.path.join(OUT, "%s-%d.txt" % (prefix, n))
        text, got = make(n, words)
        open(path, "wb").write(text.encode("utf-8"))
        print("%s-%d" % (prefix, n), "tokens", got, "bytes", len(text.encode("utf-8")), path, flush=True)


if __name__ == "__main__":
    main()
