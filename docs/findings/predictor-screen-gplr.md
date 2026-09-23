# Predictor screening on recorded routing streams (2026-09-23, host-only)

Offline screening of TG expert-router predictors against the history
baseline, ported from the colibri/deepseek-v4-baseline predictor-funnel
(T-pf-02 replay pattern) to this repository's `rowset_fetch` schema.
Tool: `tools/predictor-screen-gplr.py`. No GPU, no engine change, no
performance claim. This is a feasibility/instrumentation result, not a
wire-in verdict.

## Motivation

T-GPLR-15 closed (2026-09-13) with the explicit note that "the predictive
hypothesis is not falsified, the budgeted variant is": the measured reuse
ceiling on the echo stream was 0.9385 while retention-only reaches 0.3088
at K=1. The colibri funnel (docs/findings/predictor-funnel-findings.md,
DeepSeek-V4 routing) supplied the candidate families and the methodology;
this pass ports the screening harness and produces our own numbers on our
own model (Qwen3.6-35B-A3B, top_k=8 of 256).

## Method

- Demand stream: one row per (TG token, layer) = the exact routed set
  (`rowset_fetch.union_ids`). Online-faithfulness per colibri plan D1: a
  predictor for (t, L) sees only tokens < t (all layers) and token t at
  layers < L; `predict()` precedes `observe()` for every row. A selftest
  plants known history/markov structure and asserts it is detected while
  faithfulness holds (`--selftest`, PASS).
- Candidates (faithful ports): history (the K=1 retention baseline),
  markov-1 (per-layer id-bigram transitions), coact (normalized per-layer
  co-activation), coupling-copula (cross-layer, lift-weighted).
- Metrics: recall@B (colibri D3: |pred ∩ true| / |true| per observation),
  ppb@B (useful/proposed), extra-hits-vs-history computed per observation
  (colibri used unique-hits-over-live-union; our live baseline IS history,
  so the per-observation delta is the directly meaningful form here).

## Corpus

74 traces collapse to 7 distinct demand streams (content hash of the
demand rows). Screened one representative per stream; same-stream files
reproduce identical tables (verified A1 vs B1):

| stream | tokens x layers | shape | note |
|---|---|---|---|
| gplr19 adaptive-prev F1 | 256 x 40 | 10240 rows | the ECHO vehicle (T-GPLR-15/19/26) |
| gplr19 honest-k6 A1/A2/B1/B2 | 256 x 40 | 10240 rows | non-echoing 256-token stream |
| gplr24-f3-50k PRIOR | 96 x 40 | 3840 rows | 50k-context |
| qa-depth-k4/k8 | 96 x 40 | 3840 rows | QA prompt, retention smoke |
| gplr57 a1-512 rung | 3 x 40 | 120 rows | too small to screen; reported for completeness |
| gplr57 a1-200k rung | 3 x 40 | 120 rows | too small to screen; reported for completeness |

## Results (recall% / ppb% / extra-hits-vs-history)

### Echo stream (the T-GPLR-15/19/26 vehicle) — arm F1

| algorithm | B=8 | B=16 | B=32 |
|---|---|---|---|
| history | 30.8 / 30.9 / 0 | 30.8 / 30.9 / 0 | 30.8 / 30.9 / 0 |
| markov-1 | 39.5 / 39.9 / 14472 | 58.0 / 29.4 / 25755 | 75.0 / 19.3 / 37616 |
| coact | **42.6 / 43.0 / 16573** | 60.5 / 30.7 / 27567 | 76.0 / 19.5 / 38374 |
| coupling-copula | 31.9 / 32.9 / 17831 | 55.9 / 29.0 / 29573 | **77.2 / 20.3 / 40815** |

Anchor: history@B=8 = 0.3076 vs recorded (b) K=1 = 0.3088 — OK (port
validates against the repo's own recording to ~1e-3).

### Honest non-echoing stream — arm A1 (A2/B1/B2 identical)

| algorithm | B=8 | B=16 | B=32 |
|---|---|---|---|
| history | 24.6 / 24.7 / 0 | 24.6 / 24.7 / 0 | 24.6 / 24.7 / 0 |
| markov-1 | 27.4 / 27.7 / 11403 | 41.0 / 20.8 / 19386 | 56.8 / 14.6 / 29696 |
| coact | **28.1 / 28.4 / 11489** | 41.4 / 21.0 / 19338 | 57.5 / 14.8 / 29788 |
| coupling-copula | 20.6 / 21.2 / 12456 | 37.7 / 19.5 / 22126 | **59.0 / 15.5 / 34010** |

Anchor: history@B=8 = 0.2458 vs recorded honest (b) K=1 = 0.2468 — OK.

### 50k-context stream (gplr24-f3) and QA-depth stream

| algorithm | 50k B=8 | 50k B=32 | QA B=8 | QA B=32 |
|---|---|---|---|---|
| history | 38.5 / 38.9 / 0 | 38.5 / 38.9 / 0 | 33.3 / 33.7 / 0 | 33.3 / 33.7 / 0 |
| markov-1 | 39.0 / 40.0 / 3717 | 70.1 / 18.9 / 10400 | 35.5 / 36.4 / 3993 | 67.0 / 18.1 / 11148 |
| coact | 40.8 / 41.8 / 3979 | 70.3 / 18.9 / 10423 | 36.8 / 37.7 / 4294 | 67.0 / 18.1 / 11166 |
| coupling-copula | 30.8 / 32.1 / 4808 | 72.5 / 19.9 / 11471 | 30.2 / 31.4 / 5341 | 71.3 / 19.7 / 12554 |

No recorded (b) K=1 exists for these shapes; their history values are
fresh measurements, not validations (the tool prints N/A for the anchor
there; the first run mis-flagged 50k — anchor scope fixed to the 10240-row
vehicle only).

### gplr57 rungs (3 TG tokens each — too small to screen)

Both rungs: history ~24, markov-1/coact ~6-7 (no history to learn from in
three tokens), coupling-copula ~24-25 with 7-18 extra hits. Recorded for
completeness only; no conclusion.

## Reading

1. **The port is anchored.** History@B=8 reproduces the recorded (b) K=1
   on both 256-token vehicles (0.3076 vs 0.3088 echoed; 0.2458 vs 0.2468
   honest). The tables share denominators and faithfulness discipline with
   colibri's F4, so the family ranking is comparable.
2. **Coact/markov-1 beat history at every budget on every substantive
   stream**, exactly as colibri's H1 predicted: +3-12pp recall at B=8 at
   EQUAL-or-better precision, and the margin grows with budget. On the
   echo stream coact@B=8 delivers 42.6% recall at 43.0% precision —
   history's precision with +11.8pp recall.
3. **Coupling-copula is the unique-coverage and high-budget leader**
   (highest extra-hits at B=8 on three of four streams; highest recall at
   B=32 everywhere) — colibri's H2 regime finding (long context favors
   cross-layer structure) reproduces qualitatively on our model.
4. **Implication for the retention cache (measured shapes):** at B=8 the
   same-layer families reach coact 28.1 on the honest stream vs history
   24.6 — a same-precision +3.5pp — and at B=16 they reach 41+ vs
   history's budget-invariant 24.6. Read against the K-ladder: a
   predictor feeding a K=2-sized window could plausibly deliver what
   K=4-6 retention delivers (0.536-0.649), i.e. the charter's VRAM
   objective has a candidate lever. THIS IS PROJECTION FROM RECALL TO
   (b), NOT A MEASUREMENT — (b) and recall are different quantities
   (resident-window share vs per-observation set overlap).
5. **Why this is not a wire-in verdict:** colibri's own end-to-end A/B
   (T-pf-06) shrank its fusion advantage to noise on larger samples, and
   its slot-thrash incident (UPDATE101) showed predicted fills must yield
   to exact demand. A shadow win is a hypothesis; an engine wire-in needs
   the exact-demand-priority invariant and a pre-registered acceptance
   quantity of its own.

## Limits

- n=1 prompt per stream (two independent prompts at 256 tokens, one each
  at 50k/QA); no fresh GPU capture was taken for this screening.
- The (b)-projection in reading 4 is arithmetic extrapolation, labeled as
  such; it is the hypothesis a Planner task would test, not a result.
- Echo-stream numbers inherit the echo bias the T-GPLR-15 close recorded
  (repetition inflates temporal reuse; the honest stream is the
  conservative one, and the ranking there is narrower).
- The three small streams (gplr57) carry 3 TG tokens — screening is
  meaningless there; included only because the task named them.

## Extension: confidence-bar frontier and fusion (2026-09-23, same day)

Added to `tools/predictor-screen-gplr.py` (colibri barsweep.py/fusion.py
operators ported verbatim; selftest extended: `--mode bar`, `--mode fusion`).

### Confidence-bar sweep (--mode bar, B=8)

Rank-geometric bar: publish rank r while bar^r >= 0.5. Frontier per stream:

| variant | honest rec/ppb | echo rec/ppb |
|---|---|---|
| history (unbarred) | 24.6 / 24.7 | 30.8 / 30.9 |
| markov-1@bar0.9 | **25.1 / 29.0** | **36.1 / 41.7** |
| coact@bar0.9 | 25.8 / 29.8 | 38.9 / 44.9 |
| coact (unbarred, best B=8) | 28.1 / 28.4 | 42.6 / 43.0 |
| anything@bar0.7 | 10.1-10.3 / 41-42 | 13.5-14.2 / 54-57 |

**Reading:** a conservative bar (0.9) strictly DOMINATES history on both
axes on both streams (markov-1@0.9: +0.5pp recall AND +4.3pp ppb honest;
+5.3/+10.8 echo) — the safest possible family. But the frontier vs the
unbarred leaders is unfavorable (colibri's H3-disconfirm reproduces):
loosening toward bar 0.7 buys precision (41-57pp) at a ~2.7x recall
collapse, and unbarred coact beats every barred variant on recall at
near-equal ppb. The bar's role is risk control, not frontier-pushing.

### Fusion (--mode fusion, tune/held split 80/20 by token, grid at B=8)

| row | honest held rec/ppb | echo held rec/ppb |
|---|---|---|
| history baseline (held) | 22.7 / 22.7 | 27.9 / 27.9 |
| weighted-fusion (per-stream tuned) | 28.8 / 28.8 | 48.8 / 48.8 |
| weighted-fusion (colibri portable h1 m2 c1 k2) | 28.7 / 28.7 | 46.4 / 46.4 |
| vote-fusion (param-free) | **29.6 / 29.6** | 47.3 / 47.3 |
| vote-fusion + null source [pool control] | 28.0 / 28.0 | 41.0 / 41.0 |

**Reading:**
1. Fusion dominates every single candidate on BOTH axes on both streams
   (best: 29.6 vs coact 28.1 honest; 48.8 vs coact 42.6 echo) — the merge
   is real, not averaging noise.
2. The portable weight vector lands within 0.1pp (honest) / 2.4pp (echo)
   of per-stream tuning — colibri's portability result reproduces; no
   per-cell tuning is required for a future contract.
3. **The null control exposes the pool-growth share:** a history-clone
   added to the vote pool alone lifts history 22.7 -> 28.0 (honest).
   Structure above that: +1.6pp (honest) / +6.3pp (echo). A wire-in case
   must be argued against the null row, not the history row.
4. Best-vs-history margin at held-out, honest stream: +6.9pp recall at
   +6.9pp precision (vote-fusion). Echo stream: +20.9/+20.9 (tuned).

### Combined implication for a wire-in decision

The precision-recall frontier now has three operating points measured on
our model: (a) unbarred coact/markov-1 at B=8 (+3.5pp honest, equal ppb);
(b) bar-0.9 same-layer families (+0.5 recall, +4.3 ppb — the risk-free
point); (c) fusion (+6.9/+6.9 honest, +20.9 echo, needs 4 sources and the
null-share caveat). None of this changes the T-pf-06 lesson: an offline
win is a hypothesis; the engine A/B remains the verdict instrument.

## Reproduce

```
python tools/predictor-screen-gplr.py --selftest
python tools/predictor-screen-gplr.py --b 8,16,32 \
  build/llama-cuda/gplr19-adaptive-prev/arm-F1.jsonl \
  build/llama-cuda/gplr19-honest-k6-ab/arm-A1.jsonl \
  build/llama-cuda/gplr24-f3-50k/qa-50000-retention-PRIOR.jsonl \
  build/llama-cuda/qa-depth-k4.jsonl
python tools/predictor-screen-gplr.py --mode bar --b 8 \
  build/llama-cuda/gplr19-honest-k6-ab/arm-A1.jsonl
python tools/predictor-screen-gplr.py --mode fusion --b 8 \
  build/llama-cuda/gplr19-honest-k6-ab/arm-A1.jsonl
```

Deterministic: identical tables on re-run (verified for base mode; bar and
fusion are deterministic by construction - no sampling anywhere).
