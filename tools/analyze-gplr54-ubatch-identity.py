#!/usr/bin/env python3
"""T-GPLR-54 arm A1: the single-chunk prefill ID-CHECK (host-only, no GPU).

WHY THIS EXISTS
  T-GPLR-38's ubatch sweep recorded the caution "no arm at or above u4096 can be
  called output-equivalent" after its 4k arms showed different greedy ids. The
  T-GPLR-54 drafting recon (2026-09-21) sharpened that observation before
  spending anything on it: the divergence is NOT a region or a repeat loop - it
  is exactly ONE bounded insertion of template ids, after which the single-chunk
  stream rejoins the multi-chunk stream byte-exactly. This tool makes that
  identity check reproducible and falsifiable instead of a drafting-session
  hand calculation.

WHAT IT DOES, PER PAIR
  Given a REFERENCE stream (a multi-chunk arm, the id-equivalence oracle) and a
  CANDIDATE stream (the single-chunk arm under test), both from one `tg_tokens`
  event, let D = first differing index, K = the candidate's insertion length:

  * C1 (prefix)      candidate[:D] == reference[:D] and D >= D_MIN (the
                     recorded answer-span end; any divergence inside the answer
                     region fails, whatever the tail does).
  * C2 (insertion)   the candidate is the reference plus exactly ONE contiguous
                     insertion at D: with g = len(cand) - len(ref), there is a
                     K (1 <= K <= K_MAX, K >= g) with
                     cand[D+K:] == ref[D:len(ref)-(K-g)], i.e. every reference
                     id from D onward is preserved (the last K-g fall out of a
                     fixed-length window when g = 0, as on the recorded arms;
                     the stream simply runs K longer when g = K). A
                     substitution or deletion is NOT an insertion: a torn tail
                     fails because the reference's ids from D are not all
                     preserved. K = 0 means id-identical streams.
  * C3 (template)    every inserted id belongs to the tokenizer's
                     template/turn-marker set: {248045, 248046} plus 198 (the
                     newline), measured on the Qwen3.6-35B-A3B arms. A content-
                     token insertion fails C3 even if C2 holds.
  * C4 (determinism) a repeat arm at the same shape reproduces (D, K, inserted)
                     EXACTLY. Enforced only when a repeat pair is requested
                     (--repeat); C4 needs a second run and is never inferred.

  Verdicts: IDENTICAL (K = 0) / BOUNDED-INSERTION (C1-C3 pass) / FAIL.

CHECKED EXPECTATIONS ON THE RECORDED ARMS (falsifiable; pre-registered in the
plan block T-GPLR-54 under "Established facts" F1-F3):
  * CHK-u1024 / CHK-u2048 : id-identical to u512 over all 96 ids (K = 0)
  * CHK-u4096             : D = 41, K = 7, inserted = the measured sequence,
                            tail realigned, C3 template-only
  A pair that predates `tg_tokens` SKIPS with exit 2 (cannot evaluate, NOT a
  pass): the 100k arms carry no `tg_tokens` at all.

PRE-REGISTERED FIXTURES (--selftest): positive (identical pair), positive
(the recorded 7-id template insertion with exact realignment), negative
(content-token insertion), negative (non-realigning tail), negative (divergence
inside the answer region, D < D_MIN), refusal (a stream with no tg_tokens).

USAGE
  python tools/analyze-gplr54-ubatch-identity.py --selftest
  python tools/analyze-gplr54-ubatch-identity.py TAG REF            # one pair
  python tools/analyze-gplr54-ubatch-identity.py                    # the
      recorded default set: u512 vs u1024, u2048, u4096 (+ C4 skip note)
  python tools/analyze-gplr54-ubatch-identity.py --repeat TAG_A TAG_B
      # C4 determinism gate for A3's x2 arms: two same-shape arms must be
      # id-identical to each other (the divergence structure itself reproduces)
  Trailing --json prints the rows as JSON for later joins.

EXIT CODES
  0 = PASS  every evaluated pair satisfied its expectations
  1 = FAIL  at least one checked expectation failed (named in the output)
  2 = cannot evaluate: no arguments with no default set, a missing trace, or a
      trace with no `tg_tokens` (a run that produced no decode is NOT a pass)
"""
import io
import json
import os
import sys

# --- pre-registered constants (plan block T-GPLR-54, "Established facts") ----

REF = 'gplr38-u512'                    # the multi-chunk id-equivalence oracle
DEFAULT_CANDIDATES = ['gplr38-u1024', 'gplr38-u2048', 'gplr38-u4096']
D_MIN = 41                             # C1: answer span ends at index 40
K_MAX = 8                              # C2: one short detour, not a region
TEMPLATE_IDS = frozenset((248045, 248046, 198))  # C3, measured on qwen35moe

# F2, the measured insertion on the recorded arms (selftest fixture + CHK)
RECORDED_D = 41
RECORDED_K = 7
RECORDED_INSERTED = [248045, 248045, 248045, 248045, 248046, 198, 248045]


def load_tg_ids(path):
    """Return (ids, reason). reason is set when the trace cannot be evaluated."""
    if not os.path.isfile(path):
        return None, 'missing trace'
    ids = None
    with io.open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '"tg_tokens"' not in line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get('event') == 'tg_tokens' and isinstance(e.get('ids'), list):
                ids = [int(t) for t in e['ids']]
    if ids is None:
        return None, 'no tg_tokens in trace (run produced no decode - exit 2, NOT a pass)'
    return ids, None


def id_check(ref, cand, d_min=D_MIN, k_max=K_MAX, template_ids=TEMPLATE_IDS):
    """The pre-registered ID-CHECK. Returns a verdict dict.  C2 works from the tail, not from the divergence: K is the smallest k with
  cand[D+k:] == ref[D:len(ref)-(k-g)], g = len(cand)-len(ref). That makes "no
  realignment" a structural answer (no k exists) rather than a heuristic, and
  covers both the recorded fixed-window arms (g = 0) and natural-length
  insertions (g = k).
    """
    out = {'D': None, 'K': None, 'inserted': [], 'realigns': False,
           'template_only': False, 'prefix_ok': False, 'd_min_ok': False}
    n = min(len(ref), len(cand))
    D = next((i for i in range(n) if ref[i] != cand[i]), None)
    out['D'] = D
    if D is None:
        # identical up to the shorter length; require equal lengths for K = 0
        out['K'] = 0 if len(ref) == len(cand) else None
        out['realigns'] = out['K'] == 0
        out['prefix_ok'] = out['K'] == 0
        out['d_min_ok'] = True
        out['template_only'] = True
        out['verdict'] = 'IDENTICAL' if out['K'] == 0 else 'FAIL'
        return out
    out['prefix_ok'] = cand[:D] == ref[:D]
    out['d_min_ok'] = D >= d_min
    g = len(cand) - len(ref)
    K = None
    for k in range(1, k_max + 1):
        if k < g:
            continue  # candidate consumed more reference ids than an insertion can explain
        end = len(ref) - (k - g)
        if end < D:
            continue  # the window drop would eat into the matched prefix
        if cand[D + k:] == ref[D:end]:
            K = k
            break
    out['K'] = K
    if K is None:
        out['verdict'] = 'FAIL'
        return out
    out['realigns'] = True
    out['inserted'] = cand[D:D + K]
    out['template_only'] = K == 0 or all(t in template_ids for t in out['inserted'])
    out['verdict'] = 'BOUNDED-INSERTION' if (out['prefix_ok'] and out['d_min_ok']
                                             and out['template_only']) else 'FAIL'
    return out


def fmt_verdict(v):
    if v['verdict'] == 'IDENTICAL':
        return 'IDENTICAL (K=0, %d ids)' % len(v.get('_ref', []))
    if v['verdict'] == 'BOUNDED-INSERTION':
        return ('BOUNDED-INSERTION D=%d K=%d inserted=%s tail-realigned=%s '
                'template-only=%s' % (v['D'], v['K'], v['inserted'],
                                      v['realigns'], v['template_only']))
    d = v['D'] if v['D'] is not None else 'none'
    k = v['K'] if v['K'] is not None else 'none'
    return ('FAIL D=%s K=%s prefix_ok=%s d_min_ok=%s realigns=%s template_only=%s '
            'inserted=%s' % (d, k, v['prefix_ok'], v['d_min_ok'],
                             v['realigns'], v['template_only'], v['inserted']))


# --- selftest fixtures --------------------------------------------------------

def _fixture_ref():
    # 96 synthetic ids: answer span 0..40, then a template run, then content.
    ids = ([100 + i for i in range(35)]        # 0..34 answer span
           + [248045] * 6                      # 35..40 template region
           + [846, 198, 2523, 513]             # 41..44 content continuation
           + [700 + i for i in range(51)])     # 45..95 content tail
    assert len(ids) == 96
    return ids


def _fixture_candidates(ref):
    # identical / the recorded insertion shape / the three failure modes
    same = list(ref)
    ins = list(ref[:41]) + [248045, 248045, 248045, 248045, 248046, 198, 248045] + list(ref[41:])
    content = list(ref[:41]) + [12345, 6789] + list(ref[41:])       # K=2 content ids
    torn = list(ref[:41]) + [248045, 248046] + list(ref[43:])       # no k realigns
    early = list(ref) ; early[12] = 999                              # D=12 < D_MIN
    return [('identical', same, 'IDENTICAL'),
            ('recorded-insertion', ins, 'BOUNDED-INSERTION'),
            ('content-insertion', content, 'FAIL'),
            ('torn-tail', torn, 'FAIL'),
            ('early-divergence', early, 'FAIL')]


def run_selftest():
    ref = _fixture_ref()
    cases = _fixture_candidates(ref)
    bad = 0
    for name, cand, want in cases:
        got = id_check(ref, cand)
        ok = got['verdict'] == want
        bad += 0 if ok else 1
        print('[selftest] %-18s want=%-17s got=%-17s %s' %
              (name, want, got['verdict'], 'ok' if ok else 'MISMATCH'))
    # recorded-shape fixture must reproduce the measured (D, K, inserted)
    _, ins, _ = cases[1]
    got = id_check(ref, ins)
    shape_ok = (got['D'] == RECORDED_D and got['K'] == RECORDED_K
                and got['inserted'] == RECORDED_INSERTED and got['realigns'])
    bad += 0 if shape_ok else 1
    print('[selftest] recorded-shape     want=D=%d K=%d seq ok  got=D=%s K=%s %s' %
          (RECORDED_D, RECORDED_K, got['D'], got['K'],
           'ok' if shape_ok else 'MISMATCH'))
    # refusal fixture: a stream with no tg_tokens is exit 2, never a pass
    ref_code, ref_reason = load_tg_ids('definitely-missing-trace.jsonl')
    ref_ok = ref_code is None and 'missing trace' in ref_reason
    bad += 0 if ref_ok else 1
    print('[selftest] missing-trace        want=refusal          got=%s %s' %
          (ref_reason, 'ok' if ref_ok else 'MISMATCH'))
    print('[selftest] %d/%d as expected' % (len(cases) + 2 - bad, len(cases) + 2))
    return 1 if bad else 0


# --- real pairs ---------------------------------------------------------------

def check_pair(ref_tag, cand_tag, root, expect=None):
    ref, r_reason = load_tg_ids(os.path.join(root, ref_tag + '.jsonl'))
    cand, c_reason = load_tg_ids(os.path.join(root, cand_tag + '.jsonl'))
    row = {'ref': ref_tag, 'cand': cand_tag}
    if ref is None or cand is None:
        reason = r_reason or c_reason
        print('%-22s vs %-22s SKIP: %s' % (ref_tag, cand_tag, reason))
        row['skip'] = reason
        return 'SKIP', row
    v = id_check(ref, cand)
    v['_ref'] = ref
    print('%-22s vs %-22s %s' % (ref_tag, cand_tag, fmt_verdict(v)))
    row.update({'verdict': v['verdict'], 'D': v['D'], 'K': v['K'],
                'inserted': v['inserted'], 'realigns': v['realigns'],
                'template_only': v['template_only'],
                'prefix_ok': v['prefix_ok'], 'd_min_ok': v['d_min_ok'],
                'n_ids': len(ref)})
    if expect is None:
        return 'EVAL', row
    # CHK-*: the pre-registered expectation for this recorded pair
    ok = (v['verdict'] == expect.get('verdict')
          and (expect.get('K') is None or v['K'] == expect['K'])
          and (expect.get('D') is None or v['D'] == expect['D'])
          and (not expect.get('inserted') or v['inserted'] == expect['inserted']))
    print('%46s CHK-%s: %s' % ('', cand_tag.split('-')[-1],
                               'PASS' if ok else 'FAIL (expected %s)' % expect))
    return ('EVAL' if ok else 'CHKFAIL'), row


def main(argv):
    args = [a for a in argv if a != '--json']
    as_json = '--json' in argv
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'build', 'llama-cuda')
    if '--selftest' in args:
        return run_selftest()
    rows = []
    if len(args) == 3 and args[0] == '--repeat':
        # C4: two same-shape arms reproduce each other exactly. The check is
        # id-identity between the two repeats - any (D, K) they share with the
        # reference shape is then already enforced by the standard pair check.
        verdict, row = check_pair(args[1], args[2], root,
                                  expect={'verdict': 'IDENTICAL', 'K': 0})
        rows.append(row)
        print('RESULT: %s (C4 determinism: repeats %s)' %
              ('FAIL' if verdict == 'CHKFAIL' else
               ('CANNOT-EVALUATE' if verdict == 'SKIP' else 'PASS'),
               'disagree' if verdict == 'CHKFAIL' else 'agree'))
        return 0 if verdict == 'EVAL' else (2 if verdict == 'SKIP' else 1)
    if len(args) == 2:
        verdict, row = check_pair(args[0], args[1], root)
        rows.append(row)
        return 0 if verdict in ('EVAL',) else (2 if verdict == 'SKIP' else 1)
    if not args:
        # the recorded default set: F1 (K=0 pairs) then F2/F3 (the u4096 shape)
        checks = [(REF, 'gplr38-u1024',
                   {'verdict': 'IDENTICAL', 'K': 0}),
                  (REF, 'gplr38-u2048',
                   {'verdict': 'IDENTICAL', 'K': 0}),
                  (REF, 'gplr38-u4096',
                   {'verdict': 'BOUNDED-INSERTION', 'D': RECORDED_D,
                    'K': RECORDED_K, 'inserted': RECORDED_INSERTED})]
        bad = skip = 0
        for ref_tag, cand_tag, expect in checks:
            verdict, row = check_pair(ref_tag, cand_tag, root, expect=expect)
            rows.append(row)
            bad += 1 if verdict == 'CHKFAIL' else 0
            skip += 1 if verdict == 'SKIP' else 0
        if skip == len(checks):
            print('nothing to evaluate: no default-set trace carries tg_tokens')
            print('RESULT: CANNOT-EVALUATE (exit 2, NOT a pass)')
            return 2
        print('RESULT: %s (%d checked, %d skipped, C4 determinism needs a repeat '
              'pair: pass the same tags twice with --repeat)' %
              ('FAIL' if bad else 'PASS', len(checks) - skip, skip))
        if as_json:
            print(json.dumps(rows, indent=1))
        return 1 if bad else 0
    print('usage: analyze-gplr54-ubatch-identity.py [--selftest] | [TAG REF] [--json]')
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
