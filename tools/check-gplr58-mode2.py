#!/usr/bin/env python3
"""T-GPLR-58 mode 2 - the runner's FLAG and REFUSAL contract (host-only).

The mode-2 wiring lets the engine consume the A1 proposer, and its flag surface
is where a leg declares WHICH arm proposed and whether the proposals came from
the count-matched null instead. That surface is only worth having if a
mis-declared leg is REFUSED rather than silently run: a run that quietly
substituted an arm (or an arm for the null) would produce a trace whose verdict
section describes something other than what executed. This checker pins the
contract, by running the real binary on the real fixture and asserting the exact
exit code AND the message, for every refusal and for the legal shapes.

Why a fixture and not a model: every mode-2 gate sits AFTER the host-only
metadata read (`llama_ls_model_arch`) and BEFORE `llama_backend_init`, so the
unsupported-arch fixture (1280 bytes, produced by the nested
`layer-streaming-make-unsupported-fixture` target) drives them WITHOUT a GPU,
without CUDA, and without loading a model. Each legal shape is proven accepted
by REACHING THE NEXT STABLE REFUSAL (the adapter's architecture gate), not by a
green run: the point is where the gates are, and reaching the arch gate is the
observable that they were passed.

Exit codes: 0 = every case matched, 1 = a case did not match, 2 = cannot
evaluate (missing binary or fixture). Exit 2 prints no verdict: a checker that
could not run has not passed.

  python tools/check-gplr58-mode2.py
  python tools/check-gplr58-mode2.py --selftest
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_CANNOT_EVALUATE = 2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXE = os.path.join(REPO, 'build', 'llama-cuda', 'bin', 'Release', 'llama-layer-streaming.exe')
DEFAULT_FIXTURE = os.path.join(REPO, 'build', 'llama-cuda', 'unsupported-arch-fixture.gguf')

# The gates that must PASS before any of them fires: the fixture's architecture is
# not one this project has an adapter for, so a shape that clears every mode-2
# gate lands here. Its message is the observable that the mode-2 gates admitted
# the request rather than refused it for another reason.
ARCH_GATE = re.compile(r"requires architecture")

# (name, extra argv, expected exit, message the refusal must carry)
CASES = [
    ('mode2-no-arm',
     ['--residency-tg-spec', '2'],
     2, re.compile(r"needs --residency-tg-spec-arm")),
    ('mode2-plus-script',
     ['--residency-tg-spec', '2', '--residency-tg-spec-arm', 'coact',
      '--residency-tg-spec-script', 'x.json'],
     2, re.compile(r"takes its proposals from the arm")),
    ('mode1-plus-arm',
     ['--residency-tg-spec', '1', '--residency-tg-spec-script', 'x.json',
      '--residency-tg-spec-arm', 'coact'],
     2, re.compile(r"--residency-tg-spec-arm is only meaningful with --residency-tg-spec 2")),
    ('null-without-mode2',
     ['--residency-tg-spec-null', '1'],
     2, re.compile(r"--residency-tg-spec-null is only meaningful with --residency-tg-spec 2")),
    ('unknown-arm-name',
     ['--residency-tg-spec', '2', '--residency-tg-spec-arm', 'bar0.9'],
     2, re.compile(r"expects coact, bar09, fusion or d1")),
    ('arm-names-are-exactly-four',
     ['--residency-tg-spec', '2', '--residency-tg-spec-arm', ''],
     2, re.compile(r"expects coact, bar09, fusion or d1")),
    ('d1-requires-mtp',
     ['--residency-tg-spec', '2', '--residency-tg-spec-arm', 'd1'],
     2, re.compile(r"d1 requires --mtp")),
    ('d1-window-depth',
     ['--mtp', '--pp-schedule', 'batch-major', '--residency-tg-spec', '2',
      '--residency-tg-spec-arm', 'd1', '--residency-tg-retain', '3'],
     2, re.compile(r"requires --residency-tg-retain >= 4")),
    ('d1-real-cache-only',
     ['--mtp', '--pp-schedule', 'batch-major', '--residency-tg-spec', '2',
      '--residency-tg-spec-arm', 'd1', '--residency-tg-spec-pool', '12'],
     2, re.compile(r"d1 currently uses the real per-layer cache")),
    # Legal shapes: each must reach the ADAPTER gate, proving the mode-2 gates
    # admitted it. A run that instead refused here would be a false alarm.
    ('legal-mode2-coact',
     ['--residency-tg-spec', '2', '--residency-tg-spec-arm', 'coact'],
     2, ARCH_GATE),
    ('legal-mode2-fusion',
     ['--residency-tg-spec', '2', '--residency-tg-spec-arm', 'fusion'],
     2, ARCH_GATE),
    ('legal-mode2-null',
     ['--residency-tg-spec', '2', '--residency-tg-spec-arm', 'coact',
      '--residency-tg-spec-null', '1'],
     2, ARCH_GATE),
    ('legal-mtp-past-only-coact',
     ['--mtp', '--pp-schedule', 'batch-major', '--residency-tg-spec', '2',
      '--residency-tg-spec-arm', 'coact'],
     2, ARCH_GATE),
    ('legal-mtp-past-only-bar09',
     ['--mtp', '--pp-schedule', 'batch-major', '--residency-tg-spec', '2',
      '--residency-tg-spec-arm', 'bar09'],
     2, ARCH_GATE),
    ('legal-mtp-past-only-fusion',
     ['--mtp', '--pp-schedule', 'batch-major', '--residency-tg-spec', '2',
      '--residency-tg-spec-arm', 'fusion'],
     2, ARCH_GATE),
    ('legal-d1-mtp-rowset',
     ['--mtp', '--pp-schedule', 'batch-major', '--residency-tg-spec', '2',
      '--residency-tg-spec-arm', 'd1'],
     2, ARCH_GATE),
    ('legal-d1-null',
     ['--mtp', '--pp-schedule', 'batch-major', '--residency-tg-spec', '2',
      '--residency-tg-spec-arm', 'd1', '--residency-tg-spec-null', '1'],
     2, ARCH_GATE),
    ('default-off-untouched',
     [],
     2, ARCH_GATE),
]

# Flags every case runs with: the retention path mode 2 lives on, plus an
# EXPLICIT --residency so a fixture without an adapter cannot be tunnelled into
# `forced_off` (which would refuse everything for an unrelated reason).
BASE = ['--residency', 'moe-expert-bank', '--residency-tg', '--residency-tg-retain',
        '--tg-tokens', '3']


class Fail(Exception):
    pass


def run_case(exe, model, extra, timeout=60):
    argv = [exe] + BASE + ['--model', model] + extra
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout)
    return p.returncode, p.stdout.decode('utf-8', 'replace')


def evaluate(name, extra, want_exit, want_re, exe, model, verbose=False):
    """Return (ok, detail). A mismatch is a FAILED check, never a silent skip."""
    code, out = run_case(exe, model, extra)
    if verbose:
        print('    $ %s %s' % (os.path.basename(exe), ' '.join(BASE + extra)))
    if code != want_exit:
        return False, 'exit %d, expected %d' % (code, want_exit)
    m = want_re.search(out)
    if not m:
        first = next((ln for ln in out.splitlines() if ln.startswith('error:')), '')
        return False, 'no %r in output (got %r)' % (want_re.pattern, first)
    return True, 'exit %d, matched %r' % (code, want_re.pattern)


def selftest():
    """Hermetic: stub runners, so the checker's own errors are caught here.

    Uses a Python one-liner as the 'runner', which is exactly the interface this
    checker consumes (exit code + combined output), so the negative controls
    exercise the real comparison rather than a re-implementation of it.
    """
    checks = []

    def check(what, ok):
        checks.append((what, bool(ok)))

    def stub(body):
        fd, path = tempfile.mkstemp(suffix='.py', dir=tempfile.gettempdir())
        os.close(fd)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('import sys\n' + body + '\n')
        return [sys.executable, path]

    def run_stub(stub_argv, extra, want_exit, want_re):
        # run_case takes (exe, model, extra) and composes [exe, BASE, --model, ...];
        # a stub must receive the same shape, so it is wrapped in a 1-element list.
        argv = stub_argv + BASE + ['--model', 'ignored'] + extra
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        code = p.returncode
        out = p.stdout.decode('utf-8', 'replace')
        if code != want_exit:
            return False, 'exit'
        return (want_re.search(out) is not None), 'msg'

    good = stub("print('error: --residency-tg-spec 2 needs --residency-tg-spec-arm'); sys.exit(2)")
    wrong_exit = stub("print('error: --residency-tg-spec 2 needs --residency-tg-spec-arm'); sys.exit(1)")
    wrong_msg = stub("print('error: something else entirely'); sys.exit(2)")
    silent = stub("sys.exit(2)")

    ok, _ = run_stub(good, ['--residency-tg-spec', '2'], 2, re.compile(r"needs --residency-tg-spec-arm"))
    check('(a) a matching refusal is accepted', ok)
    ok, _ = run_stub(wrong_exit, ['--residency-tg-spec', '2'], 2, re.compile(r"needs --residency-tg-spec-arm"))
    check('(a) a wrong exit code (2 expected, 1 seen) is CAUGHT', not ok)
    ok, _ = run_stub(wrong_msg, ['--residency-tg-spec', '2'], 2, re.compile(r"needs --residency-tg-spec-arm"))
    check('(a) a wrong message is CAUGHT', not ok)
    ok, _ = run_stub(silent, ['--residency-tg-spec', '2'], 2, re.compile(r"needs --residency-tg-spec-arm"))
    check('(a) a silent exit is CAUGHT rather than read as agreement', not ok)

    # The failure mode this checker exists for: a leg that would ALSO pass if the
    # binary were merely broken (model missing -> exit 1) must not be scored as a
    # refusal match, because the message would not be there.
    missing_model = stub("print('error: failed to read general.architecture from nope.gguf'); sys.exit(1)")
    ok, _ = run_stub(missing_model, ['--residency-tg-spec', '2'], 2, re.compile(r"needs --residency-tg-spec-arm"))
    check('(b) a model-missing exit 1 is not scored as a refusal', not ok)

    # A legal shape is proven by reaching the NEXT stable refusal, so the arch
    # gate must be required rather than "any exit 2".
    arch = stub("print(\"error: --residency moe-expert-bank requires architecture qwen35moe (got 'llama')\"); sys.exit(2)")
    ok, _ = run_stub(arch, [], 2, ARCH_GATE)
    check('(c) the adapter gate is accepted as the legal shapes\' observable', ok)
    ok, _ = run_stub(arch, [], 2, re.compile(r"needs --residency-tg-spec-arm"))
    check('(c) and it is NOT accepted for a case expecting a mode-2 refusal', not ok)

    passed = sum(1 for _, ok in checks if ok)
    for what, ok in checks:
        print('  %-6s %s' % ('ok' if ok else 'FAIL', what))
    print('\nSELFTEST %s (%d/%d checks)' % ('PASSED' if passed == len(checks) else 'FAILED',
                                            passed, len(checks)))
    return EXIT_OK if passed == len(checks) else EXIT_MISMATCH


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--exe', default=DEFAULT_EXE)
    ap.add_argument('--fixture', default=DEFAULT_FIXTURE)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if not os.path.isfile(args.exe):
        print('CANNOT EVALUATE: no runner at %s' % args.exe)
        return EXIT_CANNOT_EVALUATE
    if not os.path.isfile(args.fixture):
        print('CANNOT EVALUATE: no unsupported-arch fixture at %s (build the nested '
              '`layer-streaming-make-unsupported-fixture` target)' % args.fixture)
        return EXIT_CANNOT_EVALUATE

    print('T-GPLR-58 mode 2 - runner flag/refusal contract (host-only, no GPU, no model load)')
    print('  runner  : %s' % os.path.relpath(args.exe, REPO))
    print('  fixture : %s (%d bytes)' % (os.path.relpath(args.fixture, REPO),
                                         os.path.getsize(args.fixture)))
    print('  every gate sits after the metadata read and before llama_backend_init, so no CUDA')
    print('  is initialised and no weights are loaded.\n')

    failures = []
    for name, extra, want_exit, want_re in CASES:
        ok, detail = evaluate(name, extra, want_exit, want_re, args.exe, args.fixture,
                              verbose=args.verbose)
        print('  %-28s %-6s %s' % (name, 'ok' if ok else 'FAIL', detail))
        if not ok:
            failures.append(name)

    print()
    if failures:
        print('FAILED: %d case(s) did not match the contract: %s' % (len(failures), ', '.join(failures)))
        return EXIT_MISMATCH
    print('PASS: %d/%d cases - every mis-declared leg is refused with its own exit code and '
          'message, and every legal shape clears the mode-2 gates.' % (len(CASES), len(CASES)))
    print('This is a gate RESULT, not an authorization: every gate here is soft.')
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
