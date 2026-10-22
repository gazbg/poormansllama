#!/usr/bin/env python3
"""Print the placement flags of a plan: host layers + streamed subset.
  python tools/dense/win-plan.py --host even --k 8                # even layers on the host, 8 of them streamed (spread evenly)
  python tools/dense/win-plan.py --host 0-37 --k 10 --pick att    # 10 streamed, preferring full-attention layers
  --pick spread (default): evenly spread over the host list; att: the attention layers of the host list first (3,7,...), then spread
  --pick lookahead|spaced --lookahead L: the owner's formula (same as the server's --lm-decode-stream lookahead:L; --k is ignored): the first SLOTS host layers fill the ring; when a ring
      layer ends, the next host layer after L CPU-computed host layers (ring layers and non-host layers do not count) goes into the freed slot
      (spaced: the count starts at the previous ring layer, so consecutive ring layers have L CPU layers between them)
  --host res:N : every N-th layer (N-1, 2N-1, ...) resident on the device, all the others host layers (res:8 = 8 resident, 56 host; res:4 = the 16 attention layers resident)
  --host natt:N : the non-attention layers among 0..N-1 (the full-attention layers 3,7,... stay on the device)
"""
import argparse, sys
sys.path.insert(0, __file__.rsplit("tools", 1)[0] + "tools")
from dense_plan import spread, fmt_list, is_att

def lookahead_plan(host, slots, look, spaced=False):
    """the formula: see the docstring (spaced: counted from the previous ring layer instead of from each ring layer that ends)"""
    ring = list(host[:slots])
    r = len(ring) - 1 if spaced else 0
    while r < len(ring):
        cpu, nxt = 0, None
        base = ring[-1] if spaced else ring[r]
        for il in host:
            if il <= base or il in ring:
                continue
            if cpu == look:
                nxt = il
                break
            cpu += 1
        if nxt is None or nxt <= ring[-1]:
            break
        ring.append(nxt)
        r += 1
    return ring[:len(ring) // slots * slots]

def parse(spec):
    if spec.startswith("res:"):
        n = int(spec[4:])
        return [l for l in range(64) if l % n != n - 1]       # every n-th layer (n-1, 2n-1, ...) stays on the device, all the others are host layers
    if spec.startswith("natt:"):
        return [l for l in range(int(spec[5:])) if not is_att(l)]
    if spec == "even":
        return list(range(0, 64, 2))
    if spec == "odd":
        return list(range(1, 64, 2))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True)
ap.add_argument("--k", type=int, default=0)
ap.add_argument("--slots", type=int, default=2)
ap.add_argument("--pick", choices=["spread", "att", "lookahead", "spaced"], default="spread")
ap.add_argument("--lookahead", type=int, default=3)
a = ap.parse_args()
host = parse(a.host)
if a.pick in ("lookahead", "spaced"):
    stream = lookahead_plan(host, a.slots, a.lookahead, a.pick == "spaced")
    print(f"--lm-prefill-host-layers {fmt_list(host)} --lm-decode-stream {fmt_list(stream)} --lm-decode-slots {a.slots}")
    print(f"# ring {len(stream)} of {len(host)} host layers: {','.join(map(str, stream))}", file=sys.stderr)
    sys.exit(0)
if a.k == 0:
    print(f"--lm-prefill-host-layers {fmt_list(host)}"); sys.exit(0)
if a.pick == "att":
    att = [l for l in host if is_att(l)]
    rest = [l for l in host if not is_att(l)]
    stream = sorted(att[:a.k]) if len(att) >= a.k else sorted(att + spread(rest, a.k - len(att)))
else:
    stream = spread(host, a.k)
assert len(stream) == a.k, (len(stream), a.k)
print(f"--lm-prefill-host-layers {fmt_list(host)} --lm-decode-stream {fmt_list(stream)} --lm-decode-slots {a.slots}")
