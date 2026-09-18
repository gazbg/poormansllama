#!/usr/bin/env python3
"""Doc-size guard: fail loudly when a tracked file approaches lean-ctx's per-file search cap.

WHY THIS EXISTS
  `ctx_search` SILENTLY SKIPS files above a compiled-in size cap and still prints
  "0 matches", which reads as a negative result. Measured 2026-09-18 in this
  workspace:

    * `docs/currentstate.md` at 557,144 B was the "1 large files skipped" note in
      every search result -- never scanned -- and two searches for strings known
      to be present returned "0 matches in 24 files (1 large files skipped)".
    * `docs/plans/generic-prefill-layer-residency-plan.md` at 174,707 B WAS
      scanned (it returned a hit and reported "scanned 9").

  So a `ctx_search` zero is NOT evidence while a file over the cap could hold the
  answer, and the canonical state file is exactly where a stale claim hides.

WHY THE THRESHOLDS ARE A BAND, NOT A NUMBER
  The cap's VALUE is not observable from this machine: there is no `LCTX_*`
  environment override for it, no documented configuration key, and a compiled
  integer cannot be read out of the binary (grepping for `512000` returns 0 and
  proves nothing). Web research reports 512,000 B, which is CONSISTENT with the two
  measurements above but is not independently confirmed here. The thresholds are
  therefore calibrated to the measured BRACKET (174,707 B searched, 557,144 B
  skipped), not to a number this tool can verify:

    FAIL >= 480,000 B   at or past the reported cap: treat the file as already
                        invisible to `ctx_search` and split it now
    WARN >= 250,000 B   comfortably inside the bracket's safe side but within a
                        doubling of the fail band: plan the split before it grows

  `--warn`/`--fail` override both, which is also how the alarm itself is tested
  (see USAGE). If a future measurement narrows the bracket, move the thresholds
  and say so here -- do not treat 512,000 as verified.

USAGE
  python tools/check-doc-sizes.py                       # all tracked text files
  python tools/check-doc-sizes.py --top 10              # + the largest N, with band %
  python tools/check-doc-sizes.py --ext .md             # scope by extension
  python tools/check-doc-sizes.py --selftest            # classification negative controls
  python tools/check-doc-sizes.py --fail 120000         # negative control on real files
  python tools/check-doc-sizes.py --warn 100000 --fail 200000 --strict

  Wired into `tools/run-tests.ps1` for the host-only suites (no GPU, no mutex), so
  the routine host check covers it. Standalone it is one cheap command.

EXIT CODES  0 = PASS (no file in the FAIL band), 1 = FAIL (at least one file is at
or past the fail threshold, or --strict and a warning), 2 = cannot evaluate (bad
arguments, not a git work tree).
"""
import argparse
import os
import subprocess
import sys

WARN_DEFAULT = 250_000
FAIL_DEFAULT = 480_000

# Extensions whose CONTENT search never reads as text (lean-ctx skips binaries).
# Everything else tracked is checked, because a 500 KB source file is just as
# invisible to `ctx_search` as a 500 KB document.
BINARY_EXT = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', '.pdf',
    '.gguf', '.ggml', '.bin', '.safetensors', '.pt', '.onnx',
    '.zip', '.gz', '.xz', '.7z', '.tar', '.bz2',
    '.exe', '.dll', '.so', '.dylib', '.obj', '.o', '.a', '.lib', '.pdb',
    '.class', '.jar', '.pyc', '.woff', '.woff2', '.ttf', '.otf',
    '.mp4', '.mov', '.mkv', '.wav', '.mp3', '.flac',
}


def fmt(n):
    if n >= 1024 * 1024:
        return '%.2f MiB' % (n / (1024 * 1024))
    if n >= 1024:
        return '%.1f KiB' % (n / 1024)
    return '%d B' % n


def classify(size, warn, fail):
    if size >= fail:
        return 'FAIL'
    if size >= warn:
        return 'WARN'
    return 'PASS'


def tracked_files(repo):
    """Tracked paths (gitlink entries like third_party/llama.cpp are not files and
    are reported as missing, so they drop out naturally)."""
    out = subprocess.run(['git', '-C', repo, 'ls-files', '-z'],
                         capture_output=True)
    if out.returncode != 0:
        raise SystemExit('cannot evaluate: `git ls-files` failed in %s (exit %d)\n%s'
                         % (repo, out.returncode, out.stderr.decode('utf-8', 'replace').strip()))
    return [p.decode('utf-8', 'replace') for p in out.stdout.split(b'\0') if p]


def selftest(warn, fail):
    """Negative controls: the classification must move at the exact boundaries, and
    the alarm must fire on the sizes this repo actually measured."""
    cases = [
        (0,            'PASS', 'empty file'),
        (174_707,      'PASS', 'the plan, which measurably WAS scanned'),
        (249_999,      'PASS', 'one byte under the warn threshold'),
        (250_000,      'WARN', 'exactly the warn threshold'),
        (479_999,      'WARN', 'one byte under the fail threshold'),
        (480_000,      'FAIL', 'exactly the fail threshold'),
        (557_144,      'FAIL', 'the file that measurably WAS skipped'),
    ]
    bad = 0
    for size, want, why in cases:
        got = classify(size, warn, fail)
        ok = got == want
        bad += 0 if ok else 1
        print('  %-5s %-9s %-9s %s  (%s)' % ('ok' if ok else 'FAIL', fmt(size),
                                             got if ok else got + ' != ' + want, why,
                                             'expected ' + want))
    print('selftest: %d/%d classification cases PASS' % (len(cases) - bad, len(cases)))
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                    help='repository root (default: this script\'s parent directory)')
    ap.add_argument('--warn', type=int, default=WARN_DEFAULT,
                    help='WARN threshold in bytes (default %d)' % WARN_DEFAULT)
    ap.add_argument('--fail', type=int, default=FAIL_DEFAULT,
                    help='FAIL threshold in bytes (default %d)' % FAIL_DEFAULT)
    ap.add_argument('--ext', action='append', default=None,
                    help='only check this extension (repeatable), e.g. --ext .md')
    ap.add_argument('--top', type=int, default=0,
                    help='also print the largest N checked files')
    ap.add_argument('--strict', action='store_true',
                    help='treat a WARN as a failure (exit 1)')
    ap.add_argument('--selftest', action='store_true',
                    help='run the classification negative controls and exit')
    args = ap.parse_args()

    if args.warn > args.fail:
        print('cannot evaluate: --warn (%d) is above --fail (%d)' % (args.warn, args.fail))
        return 2

    if args.selftest:
        return selftest(args.warn, args.fail)

    if not os.path.isdir(os.path.join(args.repo, '.git')) and \
       subprocess.run(['git', '-C', args.repo, 'rev-parse', '--git-dir'],
                      capture_output=True).returncode != 0:
        print('cannot evaluate: %s is not a git work tree' % args.repo)
        return 2

    files = tracked_files(args.repo)
    checked = []          # (size, relpath)
    skipped_binary = 0
    for rel in files:
        ext = os.path.splitext(rel)[1].lower()
        if args.ext and ext not in [e.lower() for e in args.ext]:
            continue
        if not args.ext and ext in BINARY_EXT:
            skipped_binary += 1
            continue
        p = os.path.join(args.repo, rel)
        try:
            size = os.path.getsize(p)
        except OSError:
            continue      # gitlink / removed / unreadable: not a searchable file
        checked.append((size, rel))

    checked.sort(reverse=True)
    warn_hits = [(s, p) for s, p in checked if s >= args.warn]
    fail_hits = [(s, p) for s, p in checked if s >= args.fail]

    print('doc-size guard: %d tracked files, %d checked as text (%d binary skipped)'
          % (len(files), len(checked), skipped_binary))
    print('band: WARN >= %d B (%s), FAIL >= %d B (%s)   [cap not observable; see the docstring]'
          % (args.warn, fmt(args.warn), args.fail, fmt(args.fail)))
    if args.top:
        print('largest checked files:')
        for s, p in checked[:args.top]:
            print('  %-11s %5.1f%% of FAIL band  %s' % (fmt(s), 100.0 * s / args.fail, p))

    for s, p in sorted(warn_hits, reverse=True):
        band = classify(s, args.warn, args.fail)
        print()
        print('%s: %s is %d B (%s), %.1f%% of the FAIL band'
              % (band, p, s, fmt(s), 100.0 * s / args.fail))
        if band == 'FAIL':
            print('  AT OR PAST THE REPORTED CAP: assume ctx_search ALREADY skips this file,')
            print('  so a "0 matches" result is not evidence about it. Split it now -- the')
            print('  verbatim shard method used for docs/currentstate.md, its manifest and its')
            print('  index are described in docs/AGENT_ENVIRONMENT.md, "Where current state lives".')
        else:
            print('  Approaching the cap. Findable in ctx_search today, but adopt the same')
            print('  split before it crosses: docs/AGENT_ENVIRONMENT.md, "Where current state lives".')

    if fail_hits:
        print()
        print('RESULT: FAIL -- %d file(s) at or past %d B (%s); search may already be skipping them.'
              % (len(fail_hits), args.fail, fmt(args.fail)))
        return 1
    if warn_hits and args.strict:
        print()
        print('RESULT: FAIL (--strict) -- %d warning(s).' % len(warn_hits))
        return 1
    print()
    print('RESULT: PASS -- no file at or past %d B (%s)%s.'
          % (args.fail, fmt(args.fail), '' if not warn_hits else
             ' (%d warning(s), inside the safe band)' % len(warn_hits)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
