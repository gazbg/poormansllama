"""Resolve one conflict hunk of a merged file by choice.

usage: resolve.py <file> <hunk 1-based or 'all'> <ours|theirs|both|base>
HEAD side = the fork ('ours'); the other side = upstream ('theirs'). Works on diff3-less and diff3 markers.
Operates on bytes so line endings are preserved.
"""
import re
import sys

path, which, choice = sys.argv[1:4]
d = open(path, "rb").read()
pat = re.compile(
    rb"<<<<<<< [^\n]*\n(.*?)(?:\|\|\|\|\|\|\| [^\n]*\n(.*?))?=======\n(.*?)>>>>>>> [^\n]*\n", re.S)
n = 0


def sub(m):
    global n
    n += 1
    if which != "all" and n != int(which):
        return m.group(0)
    ours, base, theirs = m.group(1), m.group(2) or b"", m.group(3)
    return {"ours": ours, "theirs": theirs, "both": ours + theirs, "base": base}[choice]


out = pat.sub(sub, d)
open(path, "wb").write(out)
print(f"{path}: {n} hunk(s) seen, resolved {'all' if which == 'all' else which} with {choice}; "
      f"{len(re.findall(rb'^<<<<<<< ', out, re.M))} left")
