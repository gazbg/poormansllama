#!/usr/bin/env python3
"""First-token distributions of the layer-major pass for one attention configuration (GTX 1070 box, plain stock decode so that repeated runs are bit-identical).

  python3 -I tools/pascal/ftk_arm.py --label ref [--env K=V ...] [--add "--flag v"] [--prompts 4096,8192,16384] [--offsets 0,150000] [--bin DIR]
  python3 -I tools/pascal/ftk_arm.py --compare ref a b        # KL(top-N) of every arm against the first one, per prompt

Starts the server (Qwen3.6-35B-A3B Q4_K_M, experts of 37 layers on the CPU, q8_0 KV, --lm-prefill, no context checkpoints so that the last batch goes through the
pass under test), tokenizes prefixes of build/ctx100k/base.txt cut at the given character offsets, sends each as a fresh request (n_predict 1, greedy, n_probs 20) and
writes build/pascal/ftk/LABEL.json. GPU 0 only; refuses to start while another compute process is on the GPUs."""
import argparse, json, math, os, shlex, subprocess, sys, time, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = os.path.expanduser("~/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
OUT = os.path.join(REPO, "build/pascal/ftk")
TEXT = os.path.join(REPO, "build/ctx100k/base.txt")
if not os.path.exists(TEXT):
    TEXT = os.path.expanduser("~/Development/poormansllama/build/ctx100k/base.txt")


def api(port, path, body=None, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gpu_busy():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return out


def softmax_top(lp):
    m = max(v for _, v in lp)
    z = sum(math.exp(v - m) for _, v in lp)
    return {t: math.exp(v - m) / z for t, v in lp}


def ppl_compare(labels):
    arms = {l: json.load(open(os.path.join(OUT, l + ".json"))) for l in labels}
    ref = arms[labels[0]]
    def lp(entry):
        for t in entry["top"]:
            if (t[2] is not None and t[2] == entry["target"]) or t[0] == entry["piece"]:
                return t[1]
        return None
    keys = [k for k in ref if all(k in arms[l] for l in labels) and all(lp(arms[l][k]) is not None for l in labels)]
    print(f"{len(keys)} of {len(ref)} windows have the target token in the top 200 of every arm")
    for grp in sorted({k.split("-")[0] for k in keys}):
        ks = [k for k in keys if k.startswith(grp + "-")]
        print(f"-- windows of {grp[1:]} tokens: {len(ks)}")
        base = [-lp(ref[k]) for k in ks]
        print(f"   {labels[0]:>8}: mean NLL {sum(base)/len(ks):.4f} (perplexity of these positions {math.exp(sum(base)/len(ks)):.3f})")
        for l in labels[1:]:
            d = [(-lp(arms[l][k])) - (-lp(ref[k])) for k in ks]
            m = sum(d)/len(d)
            sd = (sum((x - m)**2 for x in d)/max(1, len(d) - 1))**0.5
            print(f"   {l:>8}: mean NLL {sum(-lp(arms[l][k]) for k in ks)/len(ks):.4f}  dNLL vs {labels[0]} {m:+.4f} +- {sd/len(d)**0.5:.4f} (sd {sd:.3f}, max |d| {max(abs(x) for x in d):.3f})")
    # KL over the shared top tokens is reported by the first-token comparison; here also the top-1 agreement
    for l in labels[1:]:
        same = sum(1 for k in keys if arms[l][k]["top"][0][0] == ref[k]["top"][0][0])
        print(f"   top-1 token equal to {labels[0]}: {l} {same}/{len(keys)}")


def compare(labels):
    arms = {l: json.load(open(os.path.join(OUT, l + ".json"))) for l in labels}
    ref = arms[labels[0]]
    print(f"{'prompt':>14} " + " ".join(f"{l:>22}" for l in labels[1:]))
    tot = {l: [] for l in labels[1:]}
    for k in ref:
        row = []
        for l in labels[1:]:
            a, b = ref[k], arms[l][k]
            pa = softmax_top([(t[0], t[1]) for t in a["top"][:20]]); pb = softmax_top([(t[0], t[1]) for t in b["top"][:20]])
            toks = set(pa) | set(pb)
            eps = 1e-9
            # KL(ref || arm) over the union of the top tokens, each side renormalised over the union
            sa = sum(pa.get(t, eps) for t in toks); sb = sum(pb.get(t, eps) for t in toks)
            kl = sum((pa.get(t, eps) / sa) * math.log((pa.get(t, eps) / sa) / (pb.get(t, eps) / sb)) for t in toks)
            top1 = a["top"][0][0] == b["top"][0][0]
            d1 = abs(a["top"][0][1] - dict((t[0], t[1]) for t in b["top"]).get(a["top"][0][0], -99))
            tot[l].append(kl)
            row.append(f"KL {kl:.5f} d1 {d1:.3f} {'same' if top1 else 'DIFF'}")
        print(f"{k:>14} " + " ".join(f"{r:>22}" for r in row))
    print("mean KL: " + "  ".join(f"{l} {sum(v)/len(v):.5f}" for l, v in tot.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--bin", default=os.path.join(REPO, "build/llama-cuda/bin"))
    ap.add_argument("--prompts", default="4096,8192,16384")
    ap.add_argument("--offsets", default="0,150000", help="character offsets into the text, one set of prompts per offset")
    ap.add_argument("--port", type=int, default=8193)
    ap.add_argument("--ctx", type=int, default=20480)
    ap.add_argument("--text", default=None, help="text file to cut the prompts from (default: the harness text)")
    ap.add_argument("--piece", type=int, default=200000, help="characters tokenized per offset")
    ap.add_argument("--ppl", default="", help="paired next-token log-likelihood: LENxCOUNT[,LENxCOUNT...] windows of the text; the target is the token after each window")
    ap.add_argument("--ppl-step", type=int, default=1200, help="start offset step, in tokens, between windows of one length")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.compare:
        if a.compare[0] == "ppl":
            ppl_compare(a.compare[1:])
        else:
            compare(a.compare)
        return
    if gpu_busy():
        sys.exit("GPU busy: " + gpu_busy())
    cmd = [os.path.join(a.bin, "llama-server"), "--model", MODEL, "--host", "127.0.0.1", "--port", str(a.port)] + shlex.split(
        "-dev CUDA0 -sm none -mg 0 -fa on -np 1 -c %d -ctk q8_0 -ctv q8_0 -ngl 99 -ncmoe 37 -fit off --lm-prefill --lm-prefill-ubatch 2048 "
        "--lm-prefill-segment 4096 -b 4096 --fattn-head-tile --ctx-checkpoints 0 --no-warmup" % a.ctx)
    for x in a.add:
        cmd += shlex.split(x)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", LD_LIBRARY_PATH=os.path.abspath(a.bin))
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    log = open(os.path.join(OUT, a.label + ".log"), "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    res = {}
    try:
        for _ in range(900):
            if srv.poll() is not None:
                sys.exit("server exited, see " + log.name)
            try:
                if api(a.port, "/health", timeout=3).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        text = open(a.text or TEXT, "r", encoding="utf-8", errors="replace").read()
        if a.ppl:
            toks = api(a.port, "/tokenize", {"content": text})["tokens"]
            for spec in a.ppl.split(","):
                ln, cnt = [int(x) for x in spec.split("x")]
                for w in range(cnt):
                    st = w*a.ppl_step
                    if st + ln + 1 > len(toks):
                        break
                    t0 = time.time()
                    js = api(a.port, "/completion", {"prompt": toks[st:st + ln], "n_predict": 1, "temperature": 0, "cache_prompt": False, "n_probs": 200, "seed": 1})
                    cp = js["completion_probabilities"][0]
                    tops = cp.get("top_logprobs") or cp.get("top_probs") or []
                    tgt = toks[st + ln]
                    piece = api(a.port, "/detokenize", {"tokens": [tgt]}).get("content")
                    res[f"w{ln}-{st}"] = {"target": tgt, "piece": piece, "top": [(t["token"], t["logprob"] if "logprob" in t else math.log(max(t["prob"], 1e-30)), t.get("id")) for t in tops], "s": round(time.time() - t0, 2)}
                    if w % 8 == 0:
                        print(f"{a.label} len {ln} window {w}/{cnt}: {time.time() - t0:.1f} s", flush=True)
        for off in [int(x) for x in ([] if a.ppl else a.offsets.split(","))]:
            piece = text[off:off + a.piece]
            toks = api(a.port, "/tokenize", {"content": piece})["tokens"]
            for n in [int(x) for x in a.prompts.split(",")]:
                t0 = time.time()
                js = api(a.port, "/completion", {"prompt": toks[:n], "n_predict": 1, "temperature": 0, "cache_prompt": False, "n_probs": 20, "seed": 1})
                cp = js["completion_probabilities"][0]
                tops = cp.get("top_logprobs") or cp.get("top_probs") or []
                top = [(t["token"], t["logprob"] if "logprob" in t else math.log(max(t["prob"], 1e-30))) for t in tops]
                res[f"o{off}-n{n}"] = {"top": top, "s": round(time.time() - t0, 2), "tokens_evaluated": js.get("tokens_evaluated")}
                print(f"{a.label} off {off} n {n}: {time.time() - t0:.1f} s, top1 {top[0][0]!r} {top[0][1]:.4f}", flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(60)
        except Exception:
            srv.kill()
    json.dump(res, open(os.path.join(OUT, a.label + ".json"), "w"))


main()
