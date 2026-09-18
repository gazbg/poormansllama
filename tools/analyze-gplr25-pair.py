# Relocated from build/analyze_gplr25_pair.py (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
#!/usr/bin/env python3
"""T-GPLR-25 paired A/B analyzer: af=0 vs af=1 at fixed --layer-slots 2.

Reads two layer-streaming telemetry JSONL files (the two arms of one paired
run) and reports, per arm:
  - PP window (first phase-0 copy_submit -> last phase-0 consumption end)
  - busy coverage inside the window: copy transfer, ready-wait, compute+setup
  - idle inside the window (window minus coverage) = the async-feed opportunity
  - feed serial chain: slot-0 lead-in + sum of same-slot copy-to-copy gaps
  - effective staging bandwidth = PP bytes / PP wall

The af delta is the only flag difference; both arms must show n_slots=2.
Exit code 0 always; verdict is printed, not enforced.
"""
import json, sys
from collections import defaultdict

def load(path):
    arm = {"path": path, "t0": None, "n_slots": None, "schedule": None,
           "copy_source": None, "run_id": None,
           "submits": [], "completes": [], "waits": [], "computes": []}
    for line in open(path):
        if '"run_start"' in line:
            e = json.loads(line)
            arm["t0"] = e["t0_us"]; arm["n_slots"] = e["n_slots"]
            arm["schedule"] = e["schedule"]; arm["copy_source"] = e["copy_source"]
            arm["run_id"] = e["run_id"]
        elif '"copy_submit"' in line:
            e = json.loads(line)
            if e.get("phase") == 0:
                arm["submits"].append((e["t_us"], e["slot"], e["bytes"]))
        elif '"copy_complete"' in line:
            e = json.loads(line)
            if e.get("phase") == 0:
                arm["completes"].append((e["t_us"], e["slot"], e["transfer_us"]))
        elif '"layer_ready_wait"' in line:
            e = json.loads(line)
            if e.get("phase") == 0:
                arm["waits"].append((e["t_us"], e["wait_us"]))
        elif '"layer_compute"' in line:
            e = json.loads(line)
            if e.get("phase") == 0:
                arm["computes"].append((e["t_us"], e["compute_us"], e["setup_us"]))
    arm["submits"].sort(); arm["completes"].sort(); arm["waits"].sort()
    return arm

def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out

def analyze(arm):
    subs, comps, waits, calcs = arm["submits"], arm["completes"], arm["waits"], arm["computes"]
    if not subs or not comps:
        return None
    w0 = min(s[0] for s in subs)
    w1 = max(t + w for t, w in waits) if waits else max(c[0] for c in comps)
    wall = w1 - w0
    byts = sum(s[2] for s in subs)
    iv = [(s[0], c[0]) for s, c in zip(subs, comps) if c[0] > s[0]]
    iv += [(t, t + w) for t, w in waits if w > 0]
    iv += [(t - cu - su, t) for t, cu, su in calcs if cu + su > 0]
    busy = sum(b - a for a, b in merge(iv))
    idle = wall - busy
    # feed serial chain: slot lead-in + same-slot copy-to-copy gaps
    per_slot = defaultdict(list)
    for t, s, _ in subs:
        per_slot[s].append(t)
    comp_end = {s: c[0] for (_, s, _), c in zip(subs, comps)}
    chain = 0
    for s, ts in per_slot.items():
        chain += ts[0] - w0  # lead-in from window start to first submit of slot
        for i in range(1, len(ts)):
            prev_end = comp_end.get(s)
            # previous submit of the same slot completed at comps[i-1] of that slot
            pass
    # simpler chain: for each slot, submit[i+1] - complete[i]
    per_slot_sub = defaultdict(list)
    for t, s, _ in subs:
        per_slot_sub[s].append(t)
    per_slot_com = defaultdict(list)
    for t, s, _ in comps:
        per_slot_com[s].append(t)
    chain = sum(ts[0] - w0 for ts in per_slot_sub.values())
    for s in per_slot_sub:
        for a, b in zip(per_slot_com[s], per_slot_sub[s][1:]):
            if b > a:
                chain += b - a
    return {"wall": wall, "bytes": byts, "busy": busy, "idle": idle,
            "chain": chain, "n_sub": len(subs),
            "gbps": byts / wall / 1e3 if wall else 0.0}

def fmt(arm, r, label):
    ms = lambda x: x / 1e3
    print(f"--- {label}: {arm['path'].split('/')[-1]}")
    print(f"    n_slots={arm['n_slots']} sched={arm['schedule']} src={arm['copy_source']} copies={r['n_sub']}")
    print(f"    PP wall {ms(r['wall']):,.0f} ms | busy {ms(r['busy']):,.0f} ms | idle {ms(r['idle']):,.0f} ms ({100*idle_frac(r):.1f}%)")
    print(f"    feed serial chain {ms(r['chain']):,.0f} ms | effective staging {r['gbps']:.2f} GB/s")

def idle_frac(r):
    return r["idle"] / r["wall"] if r["wall"] else 0.0

def main():
    a = load(sys.argv[1]); b = load(sys.argv[2])
    ra, rb = analyze(a), analyze(b)
    fmt(a, ra, "arm A"); fmt(b, rb, "arm B")
    if ra and rb and rb["wall"]:
        d = ra["wall"] - rb["wall"]
        print(f"=== paired verdict: wall A {ra['wall']/1e3:,.0f} ms vs B {rb['wall']/1e3:,.0f} ms "
              f"-> delta {d/1e3:+,.0f} ms ({100*d/ra['wall']:+.1f}%); "
              f"idle A {ra['idle']/1e3:,.0f} ms vs B {rb['idle']/1e3:,.0f} ms")

if __name__ == "__main__":
    main()
