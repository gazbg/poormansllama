#!/usr/bin/env python3
"""Self-reported cache hit rate per trace, straight from rowset_fetch.

h = sum(n_rows_resident) / sum(n_rows_resident + n_rows_fetched), TG events
only.  Used to pick which traces can anchor a hit-rate comparison: a trace that
reports 0.0000 either ran with the cache off or never populated the field, and
must not be mixed with ones that did.
"""

import json
import sys

def survey(path):
    """-> (n_events, n_layers, h_all, h_second_half).

    h_second_half excludes the cold-start half of the stream.  It is the number
    to compare against a warm-then-pin policy scored on the same tail, since
    that policy's warm-up misses are likewise not counted in its figure.
    """
    rows = []
    layers = set()
    for line in open(path, "r", encoding="utf-8"):
        if '"rowset_fetch"' not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") != "rowset_fetch" or ev.get("n_tokens") != 1:
            continue
        rows.append((ev.get("n_rows_resident", 0), ev.get("n_rows_fetched", 0)))
        layers.add(ev.get("layer"))

    def rate(seq):
        r = sum(a for a, _ in seq)
        t = r + sum(b for _, b in seq)
        return (r / t) if t else None

    half = len(rows) // 2
    return len(rows), len(layers), rate(rows), rate(rows[half:])


def main(argv):
    print("%-56s %7s %4s %8s %10s" % ("trace", "events", "L", "h_all", "h_2nd_half"))
    for p in argv:
        try:
            n, nl, h, h2 = survey(p)
        except OSError as e:
            print("!! %s: %s" % (p, e), file=sys.stderr)
            continue
        if not n:
            continue
        fmt = lambda v: ("%8.4f" % v) if v is not None else "     n/a"
        print("%-56s %7d %4d %s %s" % (p[-56:], n, nl, fmt(h), fmt(h2)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
