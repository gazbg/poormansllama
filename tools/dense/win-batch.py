#!/usr/bin/env python3
"""Run several win-arm.py arms one after the other (run it under tools/with-gpu-lock.ps1 so the mutex is held for the whole batch).
Arm file: one arm per line, '#' comments:   label | win-arm.py options (shell-quoted)
  python tools/dense/win-batch.py build/dense/win/batch1.txt
A line may contain the token {plan:HOST,K[,PICK[,SLOTS[,LOOKAHEAD]]]} which is replaced by the flags of tools/dense/win-plan.py.
"""
import re, shlex, subprocess, sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def expand(line):
    def rep(m):
        parts = m.group(1).split(",")
        cmd = [sys.executable, os.path.join(ROOT, "tools", "dense", "win-plan.py"), "--host", parts[0], "--k", parts[1]]
        if len(parts) > 2: cmd += ["--pick", parts[2]]
        if len(parts) > 3: cmd += ["--slots", parts[3]]
        if len(parts) > 4: cmd += ["--lookahead", parts[4]]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    return re.sub(r"\{plan:([^}]*)\}", rep, line)
rc = 0
for raw in open(sys.argv[1], encoding="utf-8"):
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        continue
    label, opts = [s.strip() for s in raw.split("|", 1)]
    opts = expand(opts)
    cmd = [sys.executable, os.path.join(ROOT, "tools", "dense", "win-arm.py"), "--label", label] + shlex.split(opts)
    print("=== ARM", label, flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    rc = rc or r.returncode
sys.exit(rc)
