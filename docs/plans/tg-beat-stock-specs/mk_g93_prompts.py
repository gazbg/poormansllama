"""G30 prompts: a long repository-prose haystack at an exact token count, followed by a request whose answer is CODE or PROSE (the two text classes that decide an MTP draft head's acceptance).

    python mk_g93_prompts.py [N ...]       default N = 4096 24000 50000; writes build/llama-cuda/gplr07-qa-prompts/mtpcode-<N>.txt and mtpprose-<N>.txt

Why: the speed prompts of G24 (natdeep) end with a thinking-mode review whose continuation is natural reasoning text; the MTP draft head pays on predictable text (code, structured
text) and not on prose (G12: acceptance 0.965 against 0.793), so G30 needs a code prompt and a prose prompt at the same depth. The haystack is the one of mk_g87_natdeep.py (the
repository's own documents, backslash-free: llama-completion processes escape sequences, the runner does not), inside the chat template with thinking CLOSED so the answer is
the code or the essay itself:

    <|im_start|>user{NL}{preface}{NL}{NL}{documents}{NL}{NL}{instruction}<|im_end|>{NL}<|im_start|>assistant{NL}<think>{NL}{NL}</think>{NL}{NL}

The whole prompt has exactly N tokens under the model's tokenizer (llama-tokenize, special tokens parsed).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g87_natdeep as nd  # noqa: E402

PREFACE = nd.PREFACE
INSTR = {
    "mtpcode": ("Ignore the documents above. Write a complete, well commented Python implementation of a red-black tree with insert, delete and search, "
                "followed by a full set of unit tests. Output only code."),
    "mtpprose": ("Ignore the documents above. Write a long, detailed essay about the history of the Roman Republic: its institutions, its great figures, its wars "
                 "and the causes of its fall. Be thorough."),
}
OPEN = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def make_class(prefix, n_target, words):
    instr = INSTR[prefix]

    def build(doc):
        return "<|im_start|>user\n" + PREFACE + "\n\n" + doc + "\n\n" + instr + OPEN

    cache = {}

    def cnt(k):
        if k not in cache:
            cache[k] = nd.count_tokens(build(" ".join(words[:k])))
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
    lengths = [int(a) for a in sys.argv[1:]] or [4096, 24000, 50000]
    os.makedirs(nd.OUT, exist_ok=True)
    words = nd.corpus_text().split(" ")
    print("corpus words", len(words))
    for n in lengths:
        for prefix in ("mtpcode", "mtpprose"):
            text, got = make_class(prefix, n, words)
            assert "\\" not in text, "backslash in the prompt"
            path = os.path.join(nd.OUT, "%s-%d.txt" % (prefix, n))
            open(path, "wb").write(text.encode("utf-8"))
            print("%s-%d" % (prefix, n), "tokens", got, "bytes", len(text.encode("utf-8")), path, flush=True)


if __name__ == "__main__":
    main()
