#!/usr/bin/env python3
"""Host-to-device copy share of an nsys capture (sqlite export of one prefill). Host-only.

  nsys export --type sqlite --output cap.sqlite cap.nsys-rep
  python tools/nsys-copy-share.py cap.sqlite

Prints the capture wall (first to last GPU activity), the union of kernel time, the host-to-device copy time (sum of durations, and the
union of the intervals), the part of the copy time that no kernel overlaps (the stall the GPU actually sees), and the GPU idle time.
Times in the sqlite are NANOSECONDS (AGENTS.md).  copyKind 1 = host to device (CUPTI memcpy kinds).

FL-0 of docs/plans/flint-support-plan.md reads the first share: h2d_union / wall.  The exposed share is reported beside it, never instead of it.
"""
import argparse
import sqlite3


def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def length(merged):
    return sum(e - s for s, e in merged)


def subtract_length(a, b):
    """total length of a that is not covered by b (both merged and sorted)"""
    total = 0
    j = 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                total += b[k][0] - cur
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            total += e - cur
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    a = ap.parse_args()
    c = sqlite3.connect(a.db)
    kern = [(s, e) for s, e in c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL")]
    copies = [(s, e, k, b) for s, e, k, b in c.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY")]
    h2d = [(s, e) for s, e, k, b in copies if k == 1]
    h2d_bytes = sum(b for s, e, k, b in copies if k == 1)
    allact = kern + [(s, e) for s, e, k, b in copies]
    if not kern or not allact:
        raise SystemExit("no kernel activity in the capture")
    t0 = min(s for s, e in allact)
    t1 = max(e for s, e in allact)
    wall = t1 - t0
    km = merge(kern)
    hm = merge(h2d)
    am = merge(allact)
    kernel_busy = length(km)
    h2d_sum = sum(e - s for s, e in h2d)
    h2d_union = length(hm)
    exposed = subtract_length(hm, km)
    idle = wall - length(am)
    ns = 1e9
    print("kernels: %d, h2d copies: %d (%.2f GiB), all copies: %d" % (len(kern), len(h2d), h2d_bytes / 2**30, len(copies)))
    print("capture wall            : %8.3f s" % (wall / ns))
    print("kernel busy (union)     : %8.3f s  %5.1f%% of wall" % (kernel_busy / ns, 100.0 * kernel_busy / wall))
    print("h2d copy, sum of copies : %8.3f s  %5.1f%% of wall" % (h2d_sum / ns, 100.0 * h2d_sum / wall))
    print("h2d copy, union         : %8.3f s  %5.1f%% of wall   <- FL-0 metric" % (h2d_union / ns, 100.0 * h2d_union / wall))
    print("h2d copy not under a kernel (exposed): %.3f s  %5.1f%% of wall" % (exposed / ns, 100.0 * exposed / wall))
    print("h2d rate while copying  : %8.2f GB/s" % (h2d_bytes / max(h2d_union, 1) ))
    print("GPU idle (no kernel, no copy): %.3f s  %5.1f%% of wall" % (idle / ns, 100.0 * idle / wall))


if __name__ == "__main__":
    main()
