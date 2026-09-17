#!/usr/bin/env python3
"""T-GPLR-39: report the host_ledger sub-buckets (route/tel/loop) and closure.

Usage: python tools/analyze-gplr39.py <trace.jsonl> [more.jsonl ...]
"""
import json
import sys

for path in sys.argv[1:]:
    evs = [json.loads(l) for l in open(path) if l.strip()]
    h = [e for e in evs if e.get('event') == 'host_ledger']
    print('==', path)
    if not h:
        print('  no host_ledger events (needs --pp-recon 1)')
        continue
    for i, e in enumerate(h, 1):
        o = e['other_us']
        r = e.get('route_us', 0.0)
        t = e.get('tel_us', 0.0)
        l = e.get('loop_us', 0.0)
        err = o - (r + t + l)
        print(f"  pass {i}: sweep {e['sweep_us']/1e3:9.1f} ms | other {o/1e3:8.1f} = "
              f"route {r/1e3:8.1f} + tel {t/1e3:7.1f} + loop {l/1e3:8.1f} ms "
              f"(closure err {err:+.1f} us, launches {e.get('launches')})")
    print('  T-GPLR-39 prediction: route carries the ubatch growth (~500-800 ms at u2048); '
          'falsifier = loop carries it or route is flat vs u512')
