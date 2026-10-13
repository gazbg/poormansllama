#!/usr/bin/env python3
"""G24 audit: re-derive every run's TG rate and free-memory figures from the RAW artifacts by routes independent of tools/g24-table.py, and flag any disagreement.

    python tools/g24-audit.py g87-short-B g87-n24k ...        exit 0 only if every check passes

Per run (hybrid arms) the generation rate is computed four ways:
  verdict   the harness's summary.tg_tokens_per_s
  phase     the telemetry's phase_summary(tg): tokens / wall_us
  log       the runner's "TG: N tokens in X ms" line
  steps     (n_steps - 1) / (t_last - t_first) over the tg_logits events' own timestamps (no runner arithmetic at all)
and for the stock arm: the "eval time ... runs ... tokens per second" line of llama-completion and the harness's stock_tok_s.
Checks (tolerances are the audit's, not the contract's bars): the four hybrid rates agree within 1.5%; N generated tokens equals the contract's 918; the stock and hybrid prompt
token counts agree (same bytes, same tokenizer call); the watcher's minimum free memory agrees with the poller peak after the constant offset (24,462 - 1,086 - poller) within
150 MiB; exit codes and guard counters are clean; the arm's flags carry the keep-free flags and the watcher env; every session ran on mains power; the wrapper provenance
line says 0 refused. Host-only. It deliberately re-implements the parsing instead of importing tools/g24-table.py.
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "llama-cuda"))
TOTAL_MIB = 24462.0
OFFSET_MIB = 1086.0
N_GEN = 918


def text_of(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace")


def find(pat, text, flags=0):
    ms = list(re.finditer(pat, text, flags))
    return ms[-1].groups() if ms else None


def step_rate(trace):
    ts = []
    for line in open(trace, encoding="utf-8", errors="replace"):
        if '"tg_logits"' in line:
            try:
                ts.append(json.loads(line)["t_us"])
            except Exception:
                pass
    if len(ts) < 3:
        return None, len(ts)
    return (len(ts) - 1) / ((ts[-1] - ts[0]) / 1e6), len(ts)


def phase_rate(trace):
    for line in open(trace, encoding="utf-8", errors="replace"):
        if "phase_summary" in line:
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("event") == "phase_summary" and o.get("phase") == "tg":
                return o["tokens"] / (o["wall_us"] / 1e6), o["tokens"]
    return None, None


def audit(session):
    bad = []
    vp = os.path.join(ROOT, session, session + "-verdict.json")
    if not os.path.exists(vp):
        print("%s: no verdict" % session)
        return ["no verdict"]
    v = json.load(open(vp, encoding="utf-8"))
    wl = os.path.join(ROOT, session + ".wrapper.log")
    wtxt = text_of(wl) if os.path.exists(wl) else ""
    g = find(r"PROVENANCE GATE:[^\n]*?(\d+) refused", wtxt)
    print("\n== %s: wrapper provenance line: %s" % (session, ("%s refused" % g[0]) if g else "NOT FOUND"))
    if not g or g[0] != "0":
        bad.append("provenance gate not clean")
    if (v.get("power") or {}).get("ac_line") != "mains":
        bad.append("not on mains power")
    pp_by_engine = {}
    for key, rec in sorted(v["arms"].items()):
        log = text_of(rec["log"]) if rec.get("log") and os.path.exists(rec["log"]) else ""
        eng = rec.get("engine")
        line = "%-8s" % key
        if eng == "stock":
            e = find(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) runs[^\n]*?([\d.]+) tokens per second", log)
            p = find(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", log)
            rate_log = float(e[2]) if e else None
            rate_chk = (float(e[1]) / (float(e[0]) / 1e3)) if e else None
            n_gen = int(e[1]) + 1 if e else None            # llama-completion counts 'runs' = generated tokens - 1 (the first comes from the prompt pass)
            line += " stock: log %.2f recomputed %.2f harness %.2f  runs %s  prompt %s tok" % (rate_log or 0, rate_chk or 0, rec.get("stock_tok_s") or 0, e[1] if e else "-", p[1] if p else "-")
            if rate_log is None or abs(rate_log - (rec.get("stock_tok_s") or 0)) > 0.02:
                bad.append("%s: stock rate log vs harness" % key)
            if n_gen != N_GEN:
                bad.append("%s: stock generated %s tokens, contract says %d" % (key, n_gen, N_GEN))
            if p:
                pp_by_engine.setdefault("stock", set()).add(int(p[1]))
        else:
            vr = (rec.get("summary") or {}).get("tg_tokens_per_s")
            pr, pn = phase_rate(rec["trace"]) if rec.get("trace") and os.path.exists(rec["trace"]) else (None, None)
            sr, sn = step_rate(rec["trace"]) if rec.get("trace") and os.path.exists(rec["trace"]) else (None, 0)
            lg = find(r"TG: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\)", log)
            lr = float(lg[2]) if lg else None
            rates = [x for x in (vr, pr, lr, sr) if x]
            spread = (max(rates) / min(rates) - 1) * 100 if len(rates) >= 2 else None
            line += " hybrid: verdict %s phase %s log %s steps %s  spread %s%%  tokens %s/%s" % (
                "%.2f" % vr if vr else "-", "%.2f" % pr if pr else "-", "%.2f" % lr if lr else "-", "%.2f" % sr if sr else "-", "%.2f" % spread if spread is not None else "-", pn, lg[0] if lg else "-")
            if spread is None or spread > 1.5:
                bad.append("%s: hybrid rate methods disagree (%s%%)" % (key, spread))
            if (pn or 0) != N_GEN and rec.get("summary") and "qa" not in rec:
                bad.append("%s: %s tokens generated, contract says %d" % (key, pn, N_GEN))
            pp = find(r"PP: (\d+) tokens in", log)
            if pp:
                pp_by_engine.setdefault("hybrid", set()).add(int(pp[0]))
            fm = find(r"free-watch: minimum free device memory ([\d.]+) MiB", log)
            if fm is None:
                bad.append("%s: no free-watch line" % key)
            else:
                derived = TOTAL_MIB - OFFSET_MIB - float(rec.get("vram_peak_raw_mib") or 0)
                line += "  watcher %.1f MiB, poller %s -> derived free %.0f (diff %+.0f)" % (float(fm[0]), rec.get("vram_peak_raw_mib"), derived, derived - float(fm[0]))
                if abs(derived - float(fm[0])) > 150:
                    bad.append("%s: watcher and poller disagree by %.0f MiB" % (key, derived - float(fm[0])))
            flags = rec.get("flags") or []
            env = rec.get("env") or {}
            if "--tg-vram-keep-free-mib" not in flags or "16000" not in flags or env.get("LLAMA_TG_FREE_WATCH_MS") != "20":
                bad.append("%s: keep-free flags or watcher env missing" % key)
        if rec.get("rc") != 0:
            bad.append("%s: rc=%s" % (key, rec.get("rc")))
        gd = rec.get("guards") or {}
        if any(gd.values()):
            bad.append("%s: guards %s" % (key, gd))
        print(line)
    if len(pp_by_engine) == 2 and pp_by_engine["stock"] != pp_by_engine["hybrid"]:
        # the hybrid PP line may carry the prompt length only; report, do not hide
        print("   prompt token counts: stock %s hybrid %s" % (sorted(pp_by_engine["stock"]), sorted(pp_by_engine["hybrid"])))
        bad.append("prompt token counts differ between engines")
    print("   RESULT:", "OK" if not bad else "; ".join(bad))
    return bad


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    allbad = []
    for s in sys.argv[1:]:
        allbad += audit(s)
    print("\nAUDIT:", "PASS" if not allbad else "FAIL (%d)" % len(allbad))
    return 1 if allbad else 0


if __name__ == "__main__":
    sys.exit(main())
