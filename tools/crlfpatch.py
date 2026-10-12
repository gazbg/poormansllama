"""Exact-match patch helper for CRLF source files: edits are given in LF form, every `old` must occur exactly once, the file keeps CRLF.
edit kinds:  (old, new)                       exact replacement
             ("@between", start, end, new)    replace from the unique `start` (inclusive) up to the unique `end` (exclusive)
Environment: PATCH_DRY=1 checks every edit and writes NOTHING (prints DRY-OK per file)."""
import os
import sys


def patch(path, edits):
    raw = open(path, "rb").read().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for i, e in enumerate(edits):
        if e[0] == "@between":
            _, start, end, new = e
            ns, ne_ = text.count(start), text.count(end)
            if ns != 1 or ne_ != 1:
                print("EDIT %d (between): start found %d, end found %d\n--- start ---\n%s\n--- end ---\n%s" % (i, ns, ne_, start[:200], end[:200]))
                sys.exit(1)
            a = text.index(start)
            b = text.index(end)
            if b <= a:
                print("EDIT %d (between): end precedes start" % i)
                sys.exit(1)
            text = text[:a] + new + text[b:]
            continue
        old, new = e
        n = text.count(old)
        if n != 1:
            print("EDIT %d: expected exactly 1 match, found %d\n--- old (first 300 chars) ---\n%s" % (i, n, old[:300]))
            sys.exit(1)
        text = text.replace(old, new)
    if os.environ.get("PATCH_DRY"):
        print("DRY-OK %s (%d edits, crlf=%s): every edit matched exactly once, NOTHING written" % (path, len(edits), crlf))
        return
    if crlf:
        text = text.replace("\n", "\r\n")
    open(path, "wb").write(text.encode("utf-8"))
    print("patched %s (%d edits, crlf=%s)" % (path, len(edits), crlf))
