#!/usr/bin/env python3
"""Find the upstream commit whose tree best matches a vendored tree.

usage: find-upstream-base.py <outer_repo> <vendored_commit> <prefix> <upstream_git_dir> <branch>

example (host-only, no GPU, no mutex):
  python -I tools/find-upstream-base.py . HEAD third_party/llama.cpp <scratch>/upstream.git master

<upstream_git_dir> is a (blobless is fine) bare clone of ggml-org/llama.cpp.
Each vendored file's blob id is looked up in the upstream first-parent history;
the file is "live" from the commit that introduced that blob until the next
commit that changes the path. The upstream position covered by the most files
is the fork point. Fork-added or fork-modified files match nowhere and are
written to upstream-base-unmatched.txt in the current directory.

Why it exists: third_party/README.md used to name `36f0ae09` as the upstream
base, but that id is a fork commit, not on ggml-org (a fetch by id is refused
with "not our ref"). Measured 2026-10-30 on the vendored base commit 6dae153:
3,403 of 3,488 files match upstream 04b569142da2 (b10488) and no other
position comes within one file.
"""
import subprocess
import sys
from collections import defaultdict

outer, vcommit, prefix, up, branch = sys.argv[1:6]


def run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, capture_output=True, check=True).stdout.decode("utf-8", "replace")


# vendored tree: path -> blob oid
vend = {}
for line in run("git", "ls-tree", "-r", f"{vcommit}", "--", prefix, cwd=outer).splitlines():
    meta, path = line.split("\t", 1)
    oid = meta.split()[2]
    vend[path[len(prefix):].lstrip("/")] = oid
print(f"vendored files: {len(vend)}", flush=True)

# upstream history, newest first
log = run("git", f"--git-dir={up}", "log", "--raw", "--no-abbrev", "--no-renames", "-m", "--first-parent",
          "--format=C %H %cI", branch)
commits = []  # newest first: (sha, date)
changes = defaultdict(list)  # path -> [(idx, newoid)] newest first
for line in log.splitlines():
    if line.startswith("C "):
        _, sha, date = line.split()
        commits.append((sha, date))
    elif line.startswith(":"):
        meta, path = line.split("\t", 1)
        parts = meta.split()
        newoid = parts[3]
        status = parts[4]
        changes[path].append((len(commits) - 1, None if status == "D" else newoid))
n = len(commits)
print(f"upstream commits walked: {n}", flush=True)


def age(idx):  # oldest = 0
    return n - 1 - idx


# coverage by age position
diff = defaultdict(int)
unmatched = []
for path, oid in vend.items():
    lst = changes.get(path)
    if not lst:
        unmatched.append(path)
        continue
    hit = None
    for k, (idx, new) in enumerate(lst):
        if new == oid:
            hit = k
            break
    if hit is None:
        unmatched.append(path)
        continue
    lo = age(lst[hit][0])
    hi = age(lst[hit - 1][0]) if hit > 0 else n  # next newer change
    diff[lo] += 1
    diff[hi] -= 1

cov = 0
best = []
for a in sorted(diff):
    cov += diff[a]
    best.append((cov, a))
mx = max(c for c, _ in best)
print(f"matched files at best position: {mx} / {len(vend)}  (unmatched anywhere: {len(unmatched)})")
# report all positions at max coverage (interval ends at next breakpoint)
keys = sorted(diff)
cov = 0
for i, a in enumerate(keys):
    cov += diff[a]
    if cov >= mx - 3:
        end = keys[i + 1] if i + 1 < len(keys) else n
        sha, date = commits[n - 1 - a]
        sha2, date2 = commits[n - 1 - (end - 1)] if end - 1 < n else commits[0]
        print(f"cov={cov} age {a}..{end-1}  first={sha[:12]} {date}  last={sha2[:12]} {date2}")
open("upstream-base-unmatched.txt", "w").write("\n".join(sorted(unmatched)))
