#!/usr/bin/env python3
"""T-GPLR-41: compare pass-1 vs pass-2 memory-lifecycle streams from mctx /
mctx_kvview events. Failure isolation only - no performance claims.

Usage: python tools/analyze-gplr41.py <trace.jsonl> [more ...]

Verdict rules (pre-registered in the plan):
  CONFIRM  - any pass-2 vs pass-1 divergence in live_n_kv / rs_z / set fields
             (at the same chunk) or any k_ne2/v_ne2 that differs between
             passes at the same (chunk, layer).
  REFUTE   - all emitted facts identical across passes (per this analyzer's
             reach) - the stale-reference family as instrumented is not the
             mechanism.
Pointer values (k_storage/v_storage/k_view_src/v_view_src) are reported for
the last chunk of each pass but intentionally excluded from the verdict:
ggml buffer base addresses may legitimately differ between passes only if a
reallocation happened, which the analyzer surfaces as an explicit note.
"""
import json
import sys


def load(path):
    mctx, kvv = [], []
    try:
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            ev = e.get("event")
            if ev == "mctx":
                mctx.append(e)
            elif ev == "mctx_kvview":
                kvv.append(e)
    except FileNotFoundError:
        print("  (no trace)", path)
    return mctx, kvv


def main():
    for path in sys.argv[1:]:
        print("==", path)
        mctx, kvv = load(path)
        if not mctx:
            print("  no mctx events (flag off or old binary)")
            continue
        passes = sorted({e["pass"] for e in mctx})
        print(f"  passes seen: {passes}  mctx={len(mctx)} kvview={len(kvv)}")
        # per-pass per-chunk lifecycle facts
        facts = {}
        for e in mctx:
            facts[(e["pass"], e["chunk"])] = (
                e["live_n_kv"], e["rs_z"], e["set_chunk_id"], e["set_rs_z"], e["set_n_out"])
        p1 = passes[0]
        divergences = []
        for (p, c), v in sorted(facts.items()):
            if p == p1:
                continue
            base = facts.get((p1, c))
            if base is None:
                divergences.append((c, "no pass-1 counterpart", v))
            elif base != v:
                divergences.append((c, base, v))
        if divergences:
            print("  LIFECYCLE DIVERGENCE pass1 vs pass2 (chunk: pass1 -> pass2):")
            for c, a, b in divergences:
                print(f"    chunk {c}: {a} -> {b}")
            print("  VERDICT: CONFIRM (stale-reference family has a fingerprint)")
        else:
            print("  lifecycle identical across passes")
            print("  VERDICT (lifecycle only): REFUTE so far; check kvview below")
        # baked KV-view shapes per (chunk, layer): compare across passes
        shapes = {}
        for e in kvv:
            shapes[(e["pass"], e["chunk"], e["layer"])] = (e["k_ne2"], e["v_ne2"])
        shape_div = []
        for (p, c, il), v in sorted(shapes.items()):
            if p == p1:
                continue
            base = shapes.get((p1, c, il))
            if base is None:
                shape_div.append((c, il, "no pass-1 counterpart", v))
            elif base != v:
                shape_div.append((c, il, base, v))
        if shape_div:
            print("  KVVIEW SHAPE DIVERGENCE (chunk, layer: pass1 -> pass2):")
            for c, il, a, b in shape_div:
                print(f"    chunk {c} layer {il}: {a} -> {b}")
        elif kvv:
            print("  baked KV-view shapes identical across passes")
        # storage pointer stability across passes (informational)
        ptrs = {}
        for e in kvv:
            ptrs[(e["pass"], e["chunk"], e["layer"])] = (
                e["k_storage"], e["v_storage"], e["k_view_src"], e["v_view_src"])
        ptr_div = []
        for key, v in ptrs.items():
            p, c, il = key
            if p == p1:
                continue
            base = ptrs.get((p1, c, il))
            if base is not None and base != v:
                ptr_div.append((c, il, base, v))
        if ptr_div:
            n = len(ptr_div)
            print(f"  KV STORAGE/VIEW POINTERS DIFFER BETWEEN PASSES at {n} (chunk,layer) sites:")
            for c, il, a, b in ptr_div[:6]:
                print(f"    chunk {c} layer {il}: {a} -> {b}")
            if n > 6:
                print(f"    ... {n-6} more")
        else:
            print("  KV storage/view pointers identical across passes")
        # last chunk seen per pass (context for where a crash interrupted)
        last = {}
        for e in mctx:
            last[e["pass"]] = max(last.get(e["pass"], -1), e["chunk"])
        print("  last chunk emitted per pass:", dict(sorted(last.items())))


if __name__ == "__main__":
    main()
