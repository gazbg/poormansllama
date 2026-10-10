# optimizaionV1: PP sweep results (plan section 6 P4, authorised as D7, 2026-10-10)

**Status: MEASURED, not accepted.** One mutex hold, 27 arms, session tag `optv1-pp50k`. Every arm is **UNCERTIFIED by provenance** (nested tree dirty; `-AllowUnprovenanced` waiver logged). Gate PASS/FAIL below is advisory; acceptance is the owner's call.
**Commit:** _(unfilled. No commit authorised.)_

## 1. Question answered

For a 50,000-token prompt, layer-major PP, which configuration gives the highest steady prefill throughput under an 8 GB ceiling (8,192 MiB) and under a 16 GB ceiling (16,384 MiB)?

## 2. What was run

- **Shape:** 50,000-token anchor prompt, layer-major (`--pp-schedule layer-major`), `--pp-repeat 3`. Statistic: steady = mean of passes 2 and 3. Slab dtype bf16, `--pp-async-feed 1`, `--pp-recon 1`, `--tg-tokens 3`, `--pp-no-logits`. Same flags as ab16 except the varied knobs.
- **Varied:** ctx (262,144 or 51,456), KV type (q8 or f16), slots (2 or 3), ubatch (1024, 2048, 4096), copy source (pinned or mmap-registered), FA causal-GQA on or off.
- **Binary for FA arms:** `build/llama-cuda/bin/optv1-fa` = research working tree (T-GPLR-66 alloc_spill fix, T-GPLR-71 present but default-off) plus the `FAFIX_CAUSAL_GQA` opt-in gate in `ggml/src/ggml-cuda/fattn.cu` (default off). Identity: `llama-layer-streaming.exe` 5EE16398D333507B, `llama.dll` 2E96AB55278A9C9C, `ggml-cuda.dll` 7150C4938E8ADAE9, ggml commit `9b90025c4-dirty`, FA patch digest 2b0e714c…, full hashes in `build/llama-cuda/bin/optv1-fa/IDENTITY.sha256`.
- **Anchor arm `ctl1024`:** the ab16 f1024 configuration on its own binary (`gplr66-fixed`, exe 70E35F47ACF27E17), to test whether this session reproduces ab16.
- **Order:** mirrored (forward block arms 1–7, reverse replicates 8–13), then S2 ctx arms 14–17, S3 slot arms 18–21, S4 copy arms 22–25, S5 KV arms 26–27.

## 3. Results (all 27 arms)

Peak = WDDM per-process dedicated maximum from `logs/<tag>.mem.csv` (1 s poller; a lower bound). Gate: the sweep driver's only failing rule for pinned arms is `copy_source != mmap-registered` (`build/stock-levers/run-lm-ubatch-sweep.ps1` line 246). Telemetry column: `check-telemetry` exit (0 = OK). Every arm: GPU clean before and after.

| # | Arm | ctx | KV | slots | ubatch | copy | FA | steady tok/s | peak MiB | gate | tel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ctl1024 (ab16 binary) | 262,144 | q8 | 2 | 1024 | mmap | off | 2,960.9 | 7,782.4 | PASS, UNCERTIFIED | 0 |
| 2 | u1024off | 262,144 | q8 | 2 | 1024 | pinned | off | 2,923.3 | 7,749.7 | copy_source only | 0 |
| 3 | u1024on | 262,144 | q8 | 2 | 1024 | pinned | on | 3,310.1 | 7,749.7 | copy_source only | 0 |
| 4 | u2048off | 262,144 | q8 | 2 | 2048 | pinned | off | 3,136.6 | 8,378.7 | copy_source only | 0 |
| 5 | u2048on | 262,144 | q8 | 2 | 2048 | pinned | on | 3,617.2 | 8,378.7 | copy_source only | 0 |
| 6 | u4096off | 262,144 | q8 | 2 | 4096 | pinned | off | 3,179.7 | 9,764.8 | copy_source only | 0 |
| 7 | u4096on | 262,144 | q8 | 2 | 4096 | pinned | on | 3,690.5 | 9,764.8 | copy_source only | 0 |
| 8 | r4096on | 262,144 | q8 | 2 | 4096 | pinned | on | 3,698.7 | 9,764.8 | copy_source only | 0 |
| 9 | r4096off | 262,144 | q8 | 2 | 4096 | pinned | off | 3,164.0 | 9,764.8 | copy_source only | 0 |
| 10 | r2048on | 262,144 | q8 | 2 | 2048 | pinned | on | 3,627.1 | 8,378.7 | copy_source only | 0 |
| 11 | r2048off | 262,144 | q8 | 2 | 2048 | pinned | off | 3,151.8 | 8,378.7 | copy_source only | 0 |
| 12 | r1024on | 262,144 | q8 | 2 | 1024 | pinned | on | 3,309.5 | 7,749.7 | copy_source only | 0 |
| 13 | r1024off | 262,144 | q8 | 2 | 1024 | pinned | off | 2,917.1 | 7,749.7 | copy_source only | 0 |
| 14 | c51u1024on | 51,456 | q8 | 2 | 1024 | pinned | on | 3,300.1 | 4,832.8 | copy_source only | 0 |
| 15 | c51u4096on | 51,456 | q8 | 2 | 4096 | pinned | on | 3,696.3 | 5,723.5 | copy_source only | 0 |
| 16 | c51u4096onb | 51,456 | q8 | 2 | 4096 | pinned | on | 3,690.1 | 5,723.5 | copy_source only | 0 |
| 17 | c51u1024onb | 51,456 | q8 | 2 | 1024 | pinned | on | 3,302.6 | 4,832.8 | copy_source only | 0 |
| 18 | s3u4096off | 262,144 | q8 | 3 | 4096 | pinned | off | 3,209.4 | 10,265.8 | copy_source only | 0 |
| 19 | s3u4096on | 262,144 | q8 | 3 | 4096 | pinned | on | 3,750.3 | 10,265.8 | copy_source only | 0 |
| 20 | s3u4096onb | 262,144 | q8 | 3 | 4096 | pinned | on | 3,749.6 | 10,265.8 | copy_source only | 0 |
| 21 | s3u4096offb | 262,144 | q8 | 3 | 4096 | pinned | off | 3,183.8 | 10,265.8 | copy_source only | 0 |
| 22 | m1024on | 262,144 | q8 | 2 | 1024 | mmap | on | 3,308.9 | 7,782.4 | PASS | 0 |
| 23 | m4096on | 262,144 | q8 | 2 | 4096 | mmap | on | 3,736.8 | 9,797.6 | PASS | 0 |
| 24 | m4096onb | 262,144 | q8 | 2 | 4096 | mmap | on | 3,729.1 | 9,797.6 | PASS | 0 |
| 25 | m1024onb | 262,144 | q8 | 2 | 1024 | mmap | on | 3,312.7 | 7,782.4 | PASS | 0 |
| 26 | f16s3u4096on | 262,144 | f16 | 3 | 4096 | pinned | on | 3,826.4 | 12,459.8 | copy_source only | 0 |
| 27 | f16s3u4096onb | 262,144 | f16 | 3 | 4096 | pinned | on | 3,812.9 | 12,459.8 | copy_source only | 0 |

**Gate note.** 22 of 27 arms fail the driver's gate with the single reason `copy_source pinned`. That rule was written for the ab16 mmap curve and does not accept the declared pinned source. The other gates (telemetry, GPU before and after, identity) are clean on every arm. The steady numbers stand. The session verdict is FAIL, and the wrapper's provenance check refused for the same dirty-tree reason as before.

## 4. Replicates (precision)

Largest relative difference between the two arms of each pair (steady tok/s): 0.02% to 0.80%. Pairs: 1024 on 0.02%, 1024 off 0.21%, 2048 on 0.27%, 2048 off 0.48%, 4096 on 0.22%, 4096 off 0.50%, ctx51 1024 0.08%, ctx51 4096 0.17%, 3-slot on 0.02%, 3-slot off 0.80%, mmap 1024 0.12%, mmap 4096 0.21%, f16 3-slot 0.35%. Differences between arms larger than about 1% are real within this session.

## 5. Effects (same session, from the table)

| Effect | Comparison | Result | VRAM |
|---|---|---|---|
| FA causal-GQA | ub1024, 2 slots | 2,920.2 → 3,309.8, **+13.3%** | 0 (identical peak) |
| FA causal-GQA | ub2048, 2 slots | 3,144.2 → 3,622.2, **+15.2%** | 0 |
| FA causal-GQA | ub4096, 2 slots | 3,171.9 → 3,694.6, **+16.5%** | 0 |
| FA causal-GQA | ub4096, 3 slots (q8) | 3,196.6 → 3,750.0, **+17.3%** | 0 |
| Ubatch 1024 → 2048 | FA on | +9.4% | +629 MiB |
| Ubatch 2048 → 4096 | FA on | +2.0% | +1,386 MiB |
| Ubatch 1024 → 2048 | FA off | +7.7% | +629 MiB |
| Ubatch 2048 → 4096 | FA off | +0.9% | +1,386 MiB |
| Context 262,144 → 51,456 | ub1024, FA on | 3,309.8 → 3,301.4 (−0.25%) | **−2,917 MiB** |
| Context 262,144 → 51,456 | ub4096, FA on | 3,694.6 → 3,693.2 (−0.04%) | **−4,041 MiB** |
| Slots 2 → 3 | ub4096, FA on | 3,694.6 → 3,750.0 (**+1.5%**) | +501 MiB |
| KV q8 → f16 | ub4096, 3 slots, FA on | 3,750.0 → 3,819.7 (**+1.9%**) | **+2,194 MiB** |
| Copy pinned → mmap | ub1024, FA on | 3,309.8 → 3,310.8 (+0.03%) | +33 MiB |
| Copy pinned → mmap | ub4096, FA on | 3,694.6 → 3,733.0 (**+1.0%**) | +33 MiB |

Two things from this table matter most. The FA gain is real at every ubatch and costs no VRAM. Job-sized context costs no PP and saves 2.9 to 4.0 GB, more than the KV difference alone, because the compute buffers also shrink with context.

## 6. Answer per ceiling

**8 GB ceiling (8,192 MiB), 50,000-token prompt**

| Option | Steady PP | Peak | Margin | Status |
|---|---|---|---|---|
| Ctx kept 262,144: ubatch 1024, FA on | 3,309.8 | 7,749.7 | 442 MiB | **measured, fits as is** |
| Ctx kept 262,144: ubatch 2048, FA on | 3,622.2 | 8,378.7 | **−187 MiB (over)** | measured, over; needs the activation slab moved to host (expected −391 MiB, so about 7,988 MiB) |
| Job-sized ctx 51,456: ubatch 4096, FA on | **3,693.2** | **5,723.5** | 2,469 MiB | **measured, best 8 GB option with headroom** (valid only if no later session needs 262k KV) |
| Job-sized ctx 51,456: ubatch 1024, FA on | 3,301.4 | 4,832.8 | 3,359 MiB | measured |

Best 8 GB option that keeps 262k context: ubatch 1024 with FA. It fits without new code (only the opt-in FA hunk). Best 8 GB option with headroom: ubatch 4096 with FA at job-sized context.

**16 GB ceiling (16,384 MiB), 50,000-token prompt, ctx 262,144**

| Option | Steady PP | Peak | Margin |
|---|---|---|---|
| ubatch 4096, FA on, 2 slots, q8 | 3,694.6 | 9,764.8 | 6,619 MiB |
| ubatch 4096, FA on, 3 slots, q8 | 3,750.0 | 10,265.8 | 6,118 MiB |
| ubatch 4096, FA on, 3 slots, f16 | **3,819.7** | 12,459.8 | 3,924 MiB |

Best measured 16 GB option: ubatch 4096, FA on, 3 slots, f16 KV at 3,819.7 tok/s. Ubatch 4096 is the knee (2048 to 4096 is +2.0%).

## 7. Corrections to the plan's expected numbers

| Plan item (rev 2, section 2.1) | Expected | Measured | Verdict |
|---|---|---|---|
| 8 GB, ctx 262k, ubatch 1024 + FA | ≈ 3,640 at 7,778 MiB | 3,309.8 at 7,749.7 MiB | Expected used the ubatch-4096 FA gain (+18%). Measured gain at 1024 is +13.3% |
| 8 GB, ctx 262k, ubatch 2048 + slab on host + FA | ≈ 3,900 at ≈ 8,020 MiB | FA on at 2048: 3,622.2 at 8,378.7 MiB. Slab not built | PP lower than expected; peak with slab on host ≈ 7,988 MiB (expected) |
| 8 GB, job-sized ctx, ubatch 4096 + FA | ≈ 3,930 at ≈ 7,612 MiB | 3,693.2 at 5,723.5 MiB | Memory better than expected (context saves more than KV alone); PP lower |
| 16 GB, ubatch 4096, 3 slots + FA | ≈ 4,800 at ≈ 10,296 MiB | 3,750.0 at 10,265.8 MiB (q8); 3,819.7 at 12,459.8 MiB (f16) | **Not reproduced.** Slots and FA combined give ~3,750, not ~4,800 |
| doc1 campaign 4,079 tok/s at ub4096, 3 slots (10.05 GiB) | reference | Same session: 3 slots, FA off = 3,196.6; FA on = 3,750.0 | **Still unexplained.** Slot count is +0.8 to +1.5% here, not +22%. Campaign binary or driver state is the likely cause; not reproduced |

Cross-session drift: the anchor arm runs the ab16 binary and configuration and gets 2,960.9 against ab16's 3,087.5 (−4.1%); its peak matches (7,782.4 against 7,778.3). Cross-session comparisons therefore carry roughly ±4%. The within-session deltas in section 5 do not.

## 8. Decisions this sweep informs

- **D7 (sweep):** done.
- **D8 (job-sized ctx for 8 GB prefill jobs):** the data favour it. It gives the best measured 8 GB number with 2.5 GB headroom and costs no PP. Trade-off: the 262k KV is not available in that session.
- **D9 (copy source):** equal at ubatch 1024; mmap about +1% at ubatch 4096 for +33 MiB. Recommendation: keep pinned as the default. It is safer for cold file cache at 200k, which the earlier copy-tail finding describes, and the gain is within the cross-session drift.

## 9. Open items and limits

1. **FA stays opt-in.** The FA causal path has not been answer-tested. The stock QA suite (`run-tests.ps1 -Suite qa-smoke`) hard-wires `--pp-schedule batch-major`, so it cannot exercise this path. An answer gate for layer-major PP needs its own run (layer-major PP plus handoff-partial TG, as in ab17), graded by `check-answer.py` with a negative control. Needs a GPU hold and authorisation.
2. **Tooling:** `run-lm-ubatch-sweep.ps1` line 246 accepts only `mmap-registered`. Make it accept the declared copy source. Not changed during this sweep (changing a driver mid-sweep would alter later arms). Safe to change now.
3. **Activation slab on host (B1)** is still unbuilt. It is needed for 8 GB at ubatch 2048 with ctx 262k, and for prompts of 144k and above at 8 GB. Its PCIe cost is unmeasured.
4. **doc1 4,079 tok/s** is unreproduced. It needs the campaign binary to settle.
5. **Stock comparison** is not in this sweep. The single-pass stock figure (837 tok/s, ab18) is still the only matched number, and it is an upper bound on the ratio.
6. **Peak VRAM** is a 1-second poller maximum, so it is a lower bound. Real-card reserve on an 8 GB card is unmeasured; the 8,192 MiB line is the campaign convention.
7. **Pre-existing doc-size FAIL:** `docs/plans/generic-prefill-layer-residency-plan.md` is 485,036 B, above the 480,000 B FAIL band. Found by `tools/check-doc-sizes.py` today. Not caused by this work; not touched here. It needs the shard split described in `docs/AGENT_ENVIRONMENT.md` ("Where current state lives").

## 10. Artifacts

- Manifest: `build/stock-levers/optv1-pp50k.manifest.jsonl`
- Plan: `build/stock-levers/optv1-pp50k.plan`
- Wrapper log: `build/stock-levers/optv1-pp50k.wrapper.log`
- Verdicts: `build/llama-cuda/verdicts/optv1-pp50k-*-a1-u*.verdict.json`
- Telemetry, logs, poller and CPU CSVs: `build/llama-cuda/logs/optv1-pp50k-*.{jsonl,log,mem.csv,cpu.csv}`
- Binary set and hashes: `build/llama-cuda/bin/optv1-fa/` (`IDENTITY.sha256`); build log `build/llama-cuda-optv1-build.log`
- Source change: `third_party/llama.cpp/ggml/src/ggml-cuda/fattn.cu` (uncommitted; FA opt-in hunk)

## 11. Owner summary (for decision)

1. **Goal.** Find the fastest long-prompt prefill configuration (50k tokens) under 8 GB and 16 GB VRAM ceilings, using layer-major streaming, with prefill throughput as the priority.
2. **How it works, plainly.** Layer-major streaming loads one layer's expert weights into a VRAM slot, runs that layer over the whole prompt, then moves on. Four knobs were measured: the FA causal kernel (a kernel choice on the no-mask path), the chunk size (ubatch), the context allocated (KV cache and compute buffers scale with it), and the staging mode.
3. **Glossary.** PP: prompt-processing throughput, tokens per second. Steady: mean of passes 2 and 3 after a cold first pass. Ubatch: prompt tokens per chunk. Slot: a VRAM place for one layer's expert weights (about 498 MiB). KV: stored attention history, sized by context. FA: flash attention. Ceiling: the peak VRAM limit used as the budget. Peak: Windows per-process VRAM maximum, from the poller.
4. **Pros and cons.**
   - FA causal kernel: **measured** +13% to +17% PP, identical VRAM. Limit: answer-untested; opt-in only.
   - Job-sized context: **measured** −2.9 to −4.0 GB VRAM, no PP cost. Limit: the job cannot later use 262k KV.
   - Ubatch: **measured** +9% (1024 to 2048) and +2% (2048 to 4096), at +0.6 and +1.4 GB. Limit: the 8 GB ceiling blocks 2048 at 262k without the slab move.
   - Slots: +1.5% for +0.5 GB. Marginal.
   - KV f16: +1.9% for +2.2 GB. Only for 16 GB.
   - Copy source: within 1%. Keep pinned.
5. **Recommendation.**
   - 8 GB, job-sized context, ubatch 4096, FA on: **adopt for prefill-first jobs** (3,693 tok/s, 5.7 GB peak). Needs the answer gate for FA first.
   - 8 GB, context 262k kept: ubatch 1024, FA on (3,310 tok/s, 7.7 GB peak). Fits as measured.
   - 16 GB: ubatch 4096, 3 slots, f16 KV, FA on (3,820 tok/s, 12.5 GB peak).
   - Next: answer-gate the FA path (open item 1); fix the copy-source gate (open item 2); build the activation-slab move (open item 3) if 8 GB at 262k with ubatch 2048 is wanted.
   - Nothing here is approved. Every gate in this repo is advisory evidence.
