#!/usr/bin/env python3
"""T-GPLR-16 MIN_BATCH=1 revert - host-only, no GPU, no CUDA, no model.

The runner used to force `GGML_OP_OFFLOAD_MIN_BATCH=1` under `--handoff-partial`,
with `LLAMA_LS_NO_MINBATCH_PROBE` as its bypass. That route deadlocked the device
stream on the third TG token, so T-GPLR-16 was closed with option (a): revert to
the pre-fix partial TG. This checker proves the revert is COMPLETE and keeps it
complete, because "the flag is gone" is otherwise only ever asserted by grep.

What it checks, over the source tree (build/ is untracked scratch and docs/ is
historical narrative, so neither is scanned for mentions):
  1. NO source file SETS GGML_OP_OFFLOAD_MIN_BATCH - the forced override.
  2. NO source file mentions LLAMA_LS_NO_MINBATCH_PROBE - the bypass.
  3. NO source file prints the `gplr16:` announce line the override emitted.
  4. Every file that READS the variable is on the upstream allowlist (the ggml
     backend `... ? atoi(getenv(...)) : 32` defaults). A new READ on this
     project's own path fails the check - a read is how the value gets used.
  5. The MODE survived: `tools/layer-streaming/main.cpp` still parses
     `--handoff-partial` and keeps its compatibility refusals. The revert removed
     the override, not the mode.
  6. Refusal over silence: exit 2 when the tree or a required surface is absent,
     so a moved/renamed file can never read as "nothing to check, pass".

    python tools/check-gplr16-revert.py
    python tools/check-gplr16-revert.py --selftest

Exit: 0 = the revert holds, 1 = a surface came back, 2 = cannot evaluate.
"""
import argparse
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VAR = "GGML_OP_OFFLOAD_MIN_BATCH"
PROBE = "LLAMA_LS_NO_MINBATCH_PROBE"

SCOPE_ROOTS = ["tools", "third_party/llama.cpp"]
SOURCE_EXT = {".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".m", ".mm",
              ".py", ".ps1", ".sh", ".cmd", ".bat", ".txt", ".cmake", ".yml", ".yaml"}
SKIP_DIRS = {".git", "build", "vendor", "node_modules", "__pycache__", ".venv"}

# A SET is what the revert removed: anything that writes the variable. Patterns are
# case-sensitive on purpose - the reverted comment in main.cpp mentions the name.
SET_PATTERNS = [
    (r'_putenv_s\s*\(\s*"%s"' % VAR, "C _putenv_s"),
    (r'\bsetenv\s*\(\s*"%s"' % VAR, "C setenv"),
    (r'\bputenv\s*\([^)]*%s' % VAR, "C putenv"),
    (r'SetEnvironmentVariable\w*\s*\([^)]*%s' % VAR, "Win32 SetEnvironmentVariable"),
    (r'os\.environ\s*\[\s*[\'"]%s' % VAR, "python os.environ[...] ="),
    (r'putenv\s*\(\s*[\'"]%s' % VAR, "python os.putenv"),
    (r'\$env:%s\s*=' % VAR, "powershell $env:... ="),
    (r'\bexport\s+%s\s*=' % VAR, "shell export"),
    (r'(?:^|[\s;&|])set\s+%s\s*=' % VAR, "cmd set"),
    (r'^\s*%s\s*=\s*\S' % VAR, "bare NAME=value assignment"),
]
ANNOUNCE_RE = re.compile(r"gplr16:\s*" + VAR)
PROBE_RE = re.compile(re.escape(PROBE))
READ_RE = re.compile(r'getenv\w*\s*\(\s*"%s"' % VAR)
READER_ALLOWED_PREFIX = ("third_party/llama.cpp/ggml/src/",)

MODE_FILE = "third_party/llama.cpp/tools/layer-streaming/main.cpp"
# The mode's own surfaces: if one of these vanishes, the revert took the mode with it.
MODE_SURFACES = [
    ("--handoff-partial help", "--handoff-partial N"),
    ("declaration", "handoff_partial_n"),
    ("argument parse", 'arg == "--handoff-partial"'),
    ("requires moe-expert-bank refusal", "--handoff-partial requires --residency moe-expert-bank"),
    ("incompatible with --residency off", "--handoff-partial (partial handoff TG is a streaming-module mode)"),
    ("incompatible with --residency-tg", "row-set TG stages rows reactively instead of migrating whole packages"),
]


def read_text(path):
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    return raw.decode("utf-8", "replace")


# This checker necessarily NAMES both strings, so it must not scan itself: its own
# text is not a surface. Reported in the output so the exclusion is visible.
SELF = "tools/check-gplr16-revert.py"


def iter_source_files(root):
    """Relative paths of scannable source files under the scope roots."""
    out = []
    for rel in SCOPE_ROOTS:
        base = os.path.join(root, rel)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in sorted(filenames):
                if os.path.splitext(name)[1].lower() not in SOURCE_EXT:
                    continue
                relpath = os.path.relpath(os.path.join(dirpath, name), root).replace("\\", "/")
                if relpath == SELF:
                    continue
                out.append(relpath)
    return sorted(out)


def scan(files):
    """files: list of (relpath, text). Returns the finding lists."""
    sets, probes, announces, reads = [], [], [], []
    for rel, text in files:
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat, kind in SET_PATTERNS:
                if re.search(pat, line):
                    sets.append((rel, lineno, kind, line.strip()[:120]))
                    break
            if ANNOUNCE_RE.search(line):
                announces.append((rel, lineno, line.strip()[:120]))
            if PROBE_RE.search(line):
                probes.append((rel, lineno, line.strip()[:120]))
            if READ_RE.search(line):
                allowed = rel.startswith(READER_ALLOWED_PREFIX)
                reads.append((rel, lineno, allowed, line.strip()[:100]))
    return sets, probes, announces, reads


def evaluate(files, scanned):
    """Returns (ok, lines). Pure function of the scanned content, so the selftest
    can drive it with fixtures instead of the real tree: the mode file is looked up
    INSIDE the given set, so a fixture cannot silently be judged against the real
    tree's mode text."""
    out = []
    ok = True
    sets, probes, announces, reads = scan(files)
    mode_text = next((t for p, t in files if p == MODE_FILE), None)

    out.append("scanned %d source file(s) under %s (build/ and docs/ excluded, %s excluded as self)"
               % (scanned, ", ".join(SCOPE_ROOTS), SELF))
    if scanned == 0 or mode_text is None:
        return None, ["cannot evaluate: %d file(s) scanned, mode file %s"
                      % (scanned, "missing" if mode_text is None else "present")]

    if sets:
        ok = False
        out.append("FAIL: %d place(s) SET %s - the T-GPLR-16 revert is undone:" % (len(sets), VAR))
        for rel, lineno, kind, line in sets:
            out.append("   %s:%d [%s] %s" % (rel, lineno, kind, line))
    else:
        out.append("OK: no source file sets %s (%d set-pattern(s) checked)"
                   % (VAR, len(SET_PATTERNS)))

    if probes:
        ok = False
        out.append("FAIL: %d reference(s) to the %s bypass:" % (len(probes), PROBE))
        for rel, lineno, line in probes:
            out.append("   %s:%d %s" % (rel, lineno, line))
    else:
        out.append("OK: the %s bypass flag does not exist anywhere in source" % PROBE)

    if announces:
        ok = False
        out.append("FAIL: the override's announce line is back:")
        for rel, lineno, line in announces:
            out.append("   %s:%d %s" % (rel, lineno, line))
    else:
        out.append("OK: no `gplr16:` announce line (the override never ran)")

    stray = [r for r in reads if not r[2]]
    out.append("reads of %s: %d, all on the upstream allowlist %s: %s"
               % (VAR, len(reads), READER_ALLOWED_PREFIX,
                  "yes" if not stray else "NO"))
    for rel, lineno, _allowed, line in reads:
        out.append("   %s%s:%d %s" % ("" if rel.startswith(READER_ALLOWED_PREFIX) else "STRAY ",
                                      rel, lineno, line))
    if stray:
        ok = False
        out.append("FAIL: %d read(s) outside the upstream backends - a read is how the"
                   " value is used again" % len(stray))
    if not reads:
        out.append("note: no reader anywhere; upstream's default (32) is gone too -"
                   " check that the vendored ggml backends are intact")

    missing = [name for name, needle in MODE_SURFACES if needle not in mode_text]
    if missing:
        ok = False
        out.append("FAIL: the MODE lost surfaces (the revert removed the override, not"
                   " --handoff-partial): %s" % ", ".join(missing))
    else:
        out.append("OK: --handoff-partial and its %d compatibility refusal(s) intact"
                   % (len(MODE_SURFACES) - 2))
    return ok, out


def run_real():
    rels = iter_source_files(ROOT)
    files = []
    for rel in rels:
        text = read_text(os.path.join(ROOT, rel))
        if text is not None:
            files.append((rel, text))
    ok, lines = evaluate(files, len(files))
    for line in lines:
        print(line)
    if ok is None:
        print("RESULT: CANNOT-EVALUATE (no source found to check; refusing to pass)")
        return 2
    print("RESULT: %s - %s" % ("PASS" if ok else "FAIL",
                               "the T-GPLR-16 MIN_BATCH=1 revert holds"
                               if ok else "a reverted surface is back"))
    return 0 if ok else 1


def selftest():
    """Fixtures, including the negative controls: each way the revert can be undone
    must be DETECTED, and the checker must refuse rather than pass on no input."""
    good_mode = "\n".join(n for _n, n in MODE_SURFACES)
    upstream = ('third_party/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu',
                'const int min_batch_size = getenv("%s") ? atoi(getenv("%s")) : 32;' % (VAR, VAR))
    ours = (MODE_FILE, good_mode)
    cases = []

    def case(name, files, want_ok):
        ok, _lines = evaluate(files, len(files))
        cases.append((name, ok is want_ok, ok, want_ok))

    case("clean tree (upstream reader only)", [upstream, ours], True)
    case("override came back",
         [upstream, ours, (MODE_FILE, good_mode + '\n_putenv_s("%s", "1");' % VAR)], False)
    case("bypass flag came back",
         [upstream, ours, ("tools/other.cpp", 'getenv("%s") != nullptr' % PROBE)], False)
    case("announce line came back",
         [upstream, ours, ("tools/x.ps1", 'fprintf(stderr, "gplr16: %s=1")' % VAR)], False)
    case("launcher exports it",
         [upstream, ours, ("tools/run-foo.ps1", '$env:%s = "1"' % VAR)], False)
    case("python sets it",
         [upstream, ours, ("tools/x.py", 'os.environ["%s"] = "1"' % VAR)], False)
    case("READ outside the upstream backends",
         [upstream, ours, (MODE_FILE, good_mode + '\ngetenv("%s");' % VAR)], False)
    case("mode surface removed",
         [upstream, (MODE_FILE, good_mode.replace("--handoff-partial N", ""))], False)
    case("mode file missing entirely", [upstream], None)
    # Refusal, not a pass: no input at all.
    ok_none, _ = evaluate([], 0)
    cases.append(("empty tree refuses", ok_none is None, ok_none, None))

    bad = 0
    for name, good, got, want in cases:
        if not good:
            bad += 1
        print("  %s  %s (got %s, want %s)" % ("PASS" if good else "FAIL", name, got, want))
    print("SELFTEST: %s %d/%d" % ("PASS" if bad == 0 else "FAIL", len(cases) - bad, len(cases)))
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description="T-GPLR-16 MIN_BATCH=1 revert checker (host-only)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return run_real()


if __name__ == "__main__":
    sys.exit(main())
