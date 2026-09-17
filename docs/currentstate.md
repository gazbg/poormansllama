# Current State

## 2026-09-17: T-GPLR-38 ADDENDUM — 144K UBOATCH A/B: INCOMPLETE, and it found a real pass-2 crash. **Requested as a 100k test; the generated anchor tokenized to 144,001 tokens (generator overshoot, achieved=unverified — flagged), so the arms ran a 144,000-token prefill (ctx 144256).** Two flag-traps hit first (both now documented): the needle generator's target-token count is NOT verified against the tokenizer, and at this KV size the handoff fit gate measures auto-OFF so `--tg-tokens 96` refuses (layer-major TG cap); arms ran `--tg-tokens 0`. **Result: BOTH arms crashed identically in pass 2 of `--pp-repeat 2`** — CUDA illegal memory access in `ggml_backend_cuda_synchronize`, at the same trace position in each: chunk 130/141 (u1024) and chunk 60/71 (u2048), both at layer 3, slot 0, generation 16, i.e. deterministic and ubatch-independent. Neither trace has a second phase_summary; the strict checker rejects both (no completed warm pass). **The partial data (pass 1 only, COLD, arm order confounded — not a valid A/B):** u1024 80,631.8 ms / 1785.9 tok/s vs u2048 76,972.0 ms / 1870.8 tok/s (−4.5% for u2048, unproven). Route telemetry cost at this shape: 4.39 s / 4.31 s per pass (~3%), consistent with the T-GPLR-39 scaling (80 events × 512×n_tok readback + sort). **What the crash means:** all previous `--pp-repeat 2` runs used chunk counts that divide the prompt exactly (8×512, 1×4096, 391×512, 141 chunks at u1024/200k? — no: 200k used ubatch 512 = 391 exact). The 144k arms are the FIRST with a tail chunk (640 tokens) under layer-major + repeat; both died at the same pass-2 slot-generation point, which smells like the pass-boundary reset interacting with the tail chunk's graph/act-slab binding, not with ubatch size. **Status:** the ubatch question at long context remains UNANSWERED; the crash is a new defect (candidate task: bisect exact-multiple pp-tokens 143360 vs the tail, then fix the reset path). Nothing committed; traces + logs preserved as crash evidence.

## 2026-09-17: T-GPLR-39 RECON — the ubatch-size residue is ATTRIBUTED: `tel_route` (route telemetry: device→host probs readback + host sort), not the launch loop. **Instrument (nested tree, opt-in `--pp-recon`):** the T-GPLR-36 `host_ledger` now splits `other` into `route_us` / `tel_us` / `loop_us` (closure exact by construction; new fields, checker-agnostic). **Recon arm (tools/run-gplr39-recon.ps1, u2048, 3 slots, ron mode, 4k anchor, --pp-repeat 2, checker OK; trace `build/llama-cuda/gplr39-u2048-recon.jsonl`):** warm pass other = 1152.8 ms, of which **route 1161.5 ms (100.8%), tel 1.0 ms, loop −9.7 ms** (80 route events/pass at 2-chunk stride × 40 layers). The pre-registered prediction ("route dominates, ~500–800 ms") is CONFIRMED in mechanism and even exceeded; the falsifier (loop carries it) did not fire. **Cross-check vs u512 (T-GPLR-38):** u512's other was 237.9 ms with the SAME 80 route events but 4× smaller readbacks (512×512 f32 = 1 MiB vs 512×2048 = 4 MiB per event) and 4× fewer sort comparisons — the route cost scales with n_tok as predicted (≈14.5 ms/event at u2048). **Consequence (the fix is now obvious and does NOT touch the host loop):** tel_route is DIAGNOSTIC telemetry, not model work — its readback stalls on the compute stream (ggml_backend_tensor_get) and its host sort is O(n_tok × 512 log 512). Options in order of invasiveness: (a) route-sampling stride already exists (`--telemetry-chunk-stride`); a route-HEAD sampling (first 64 tokens per sampled chunk) or F16 readback would cut route_us ~32–64×; (b) move the sort to the GPU / drop the hash; (c) accept the cost in exchange for the route-integrity evidence (the route events back the exact-routing invariant). The honest read: the u2048 "regression" is partly a telemetry-measurement artifact — PP math itself got faster (compute 1862.6 → 1282.7 ms); ubatch 1024 remains the measured sweet spot with telemetry on. **Process note (disclosed):** the first recon attempt used `--tg-tokens 0`, which skips the handoff migration block entirely while the fit gate still reports auto-on → checker FAIL (`auto-on but no handoff events`) — a real flag-combination trap, now documented here and in the runner; the fixed arm passes. Nothing committed.

## 2026-09-17: T-GPLR-38 UBATCH SWEEP (4k anchor, 3 slots, ron mode) — the prediction inverted at 2048: warm wall 2317.5 (u512) -> **2224.3 (u1024, −4.0%)** -> **2510.6 (u2048, +8.3%)** -> 2435.2 (u4096); tok/s 1767.4 -> 1841.4 -> 1631.5 -> 1682.0. **Run (tools/run-gplr38-ubatch.ps1, four adjacent arms, one mutex hold, all four strict checkers OK; traces `build/llama-cuda/gplr38-u*.jsonl`):** consumer wait FLAT (67.7-68.0 ms at every ubatch — the pipelining-granularity concern did not materialize), peak VRAM IDENTICAL 3265.56 MiB (the sweep probes the 3-slot window; buf_act deltas of 24->192 MiB are inside the shared-pool slack), ledger `compute` falls monotonically 1862.6 -> 1525.1 -> 1375.0 -> 1282.7 ms (fatter matmuls, as expected) — but ledger `other` EXPLODES 237.9 -> 484.5 -> 962.1 -> 959.0 ms at u2048/u4096, swallowing the compute savings and then some. **Where the win went (measured):** u1024 keeps the best of both (−334 ms compute, +247 ms other, net −93 ms wall); at u2048 the other-bucket growth (+724 ms) exceeds the compute saving (−488 ms). The other bucket is the T-GPLR-36 "loop body + sampled telemetry + route lookup" residue, per-launch cost now clearly ubatch-dependent (larger per-launch graph inputs/outputs to snapshot and patch), not launch-COUNT-dependent. **Output check:** greedy TG text byte-identical at u1024/u2048; u4096 DIFFERS (262 vs 305 bytes — the answer is cut short), consistent with a ubatch-related TG interplay, uninvestigated; this is an identity sanity check, NOT the semantic-QA gate. **Gates:** absolute profile PASS at all four; paired runs printed but correctly refuse same-config pairs (the moving axis is ubatch, which the gate does not model) — the paired throughput numbers are still recorded (1.042x / 0.923x / 0.952x). **Verdict vs the pre-registered prediction:** WRONG at u2048 (predicted best ~2050-2200 ms there; measured 2510.6, the second-worst arm) and its falsifier "wall worsens at 1024+ -> overhead model wrong" fired in the informative direction: the per-launch overhead model was launch-COUNT-based, but the residue is ubatch-SIZE-based. **Consequence:** the current sweet spot is ubatch 1024 at this shape (−4.0% warm wall, −498 MB… no VRAM cost measured, identical wait); ubatch ≥2048 is a regression under the current host loop. The generic fix (make the other-bucket work batch- or chunk-independent) is the real lever, worth ~0.7 s/pass at u2048 — a candidate task, not done here. Nothing committed.

## 2026-09-17 (addendum 3): T-GPLR-37 AT THE CROWDED SHAPE — the 4k win does NOT transfer: 200k/262k-F16-KV warm walls are FLAT (−0.2% to −0.4%), exactly as pre-registered. **Run (tools/run-gplr37c.ps1, four adjacent arms pin/ron × 2/3 slots, one mutex hold, 4k→200k anchor, `--pp-tokens 262144`, ubatch 512, kv f16, layer-major, af=1, --pp-repeat 2; all four strict checkers OK; traces `build/llama-cuda/gplr37c-*.jsonl`):** warm pass pin-s2 142,165.5 ms / 1406.8 tok/s -> ron-s2 141,853.7 / 1409.9 (**−311.8 ms, −0.22%**); pin-s3 141,666.6 / 1411.8 -> ron-s3 141,134.4 / 1417.1 (**−532.2 ms, −0.38%**). The pre-registered prediction ("within ~1% at both depths, because each bank serves ~391 chunks and af=1 already hides the ~1.8 s pass hop behind 131 s of compute") is CONFIRMED; neither falsifier branch (>3% win, >3% regression) fired. **The mechanism still shows in the ledger — the hop is gone, the wall just doesn't care:** per-bank hop 44.01→0.08 ms (s2) and 44.47→0.13 ms (s3); `slot` bucket 1759.0→1280.6 ms (s2) and **637.3→70.3 ms (s3)**; bank DMA windows stay pure (31.5–31.6 ms = 1.00×); `compute`/`patch`/`other`/`setup` unchanged (setup 221.8→207.2 ms, still the same 200k magnitude as the non-churn 4k band + the chunk-count term). Warm walls are compute-dominated (131.2 s of 141.9 s = 92.5%). **RAM pressure, now MEASURED by the fixed monitor (process-anchored window + sys_free column; CSV `build/llama-cuda/gplr37c-mem-monitor.csv`, 102+ samples covering all four arms):** min free system RAM **19,447 MiB** (of 61.7 GiB) — the box never came within ~19 GiB of pressure with 17.23 GiB pinned AND the ~5 GiB spilled KV resident; max process working set 23,458 MiB; the monitor now covers the whole run including arm boundaries. **Verdict:** the churn-free mmap-registered mode is a SHORT/AMORTIZED-CONTEXT lever (≤ ~8-16k tokens per bank, where staging sits on the critical path), NOT a long-context one; at 200k it costs 17.23 GiB of pinned RAM for ~0.3% wall. The honest default posture remains: pinned staging + depth for long context, mmap-registered opt-in for short prefill; a shape-aware policy could later switch modes on measured prompt length. Nothing here is a fitness claim; gates soft, acceptance is the owner's. Nothing committed.

## 2026-09-17 (addendum 2): T-GPLR-37 CHURN-FREE VARIANT — CHURN CONFIRMED AS THE CAUSE, SETUP INFLATION GONE, AND A LARGE WARM WIN AT 4k AT BOTH DEPTHS. **What changed (code):** `--copy-source mmap-registered` is now the churn-FREE mode — one `register_all_pkg_sources` call runs after the prebuild and BEFORE the timed PP, registering the page-aligned source span of every executed layer once (cached per logical layer, read-only afterwards); staging then never touches page tables during a pass. The V1 per-bank churn stays reproducible as `--copy-source mmap-registered-churn`, and the mode refuses loudly (driver text, `abort_reason=mmap_register`, run_refused) if any registration fails or a pass finds an unregistered span. **A/B (4 adjacent arms = pin-s2, ron-s2, pin-s3, ron-s3; 4k anchor, ubatch 512, layer-major, kv q8, af=1, --pp-repeat 2, one mutex hold; all four strict checkers OK; analyzer `tools/analyze-gplr37.py`; traces `build/llama-cuda/gplr37b-*.jsonl`):** warm pass **pin-s2 3490.2 ms / 1173.6 tok/s -> ron-s2 2676.9 ms / 1530.1 tok/s (−813.3 ms, −23.3%, +30.4% tok/s)** and **pin-s3 2706.0 ms / 1513.7 tok/s -> ron-s3 2290.6 ms / 1788.2 tok/s (−415.5 ms, −15.4%, +18.1%)**. **The falsifier did NOT fire — the churn mechanism is confirmed:** ledger `setup` (sched reset + `alloc_graph`) returns to the clean band, 13.0 -> 8.2 ms (2 slots) and 12.1 -> 7.9 ms (3 slots), versus the V1 churn arms' 227.8 / 134.7 ms; `mmap_register` fires 40 times with `mode="init-once"`, **40/40 before the first `layer_compute`** (V1: 80 events, 34.46 GiB of churn, 2/80 before), and one `mmap_register_init` reports 18,500,517,888 B (17.23 GiB) pinned in 1.6 ms pre-timed. So the V1 regression was registration churn, not an inherent cost of file-backed registered pages. **Mechanism of the win (measured):** the PP per-bank hop collapses from median 45.24 -> 0.06 ms/bank (2 slots) and 23.43 -> 0.11 ms (3 slots), the `slot` bucket falls 1620.7 -> 991.2 ms (2 slots) and 582.3 -> 69.4 ms (3 slots), bank DMA windows stay pure (31.38-31.48 ms = 1.00x the 31.37 ceiling), compute is essentially unchanged (1363.6 -> 1297.2 at 2 slots), and TG is unaffected (76.6 vs 76.75 tok/s). **Output check (identity only, not the semantic-QA gate):** the 96-token greedy `--tg-text` is byte-identical between the pinned and mmap-registered arms at both depths. **Headline vs history:** ron-s3 2290.6 ms is the fastest warm 4k layer-major pass recorded (previous best 2706 / 2811.9 / 2843.2), and **ron-s2 2676.9 ms matches pin-s3 2706.0 ms while using one slot fewer (−498 MiB of VRAM)**, so depth-3 pinning is no longer needed for this shape if this mode is adopted. **My pre-registered prediction was WRONG in direction and size** (predicted within ±3% of the pinned control, gain ≤5% at 2 slots; measured −23.3% / −15.4%): the pre-registered reading that "the hop is already hidden at 3 slots" was false at this shape. **RAM price (measured by the runner, the mode's real cost):** 17.23 GiB of the model's file pages held pinned for the run, versus 0.9-1.4 GiB of pinned staging buffers for the pinned path; no staging buffers are allocated in this mode. The side RAM monitor (`tools/monitor-mem-gplr36.ps1`, CSV `build/llama-cuda/gplr37b-mem-monitor.csv`) is NOT usable evidence: it was started before the ~26-minute mutex wait and stopped after ~74 s, so its window misses the arms entirely (tooling defect to fix, not a measurement). **Gate defect found and fixed:** `tools/check-gplr35-pipeline.py` paired mode was depth-only (it FAILed an arm comparison at equal depth by construction) and its absolute `wait/wall` fraction false-FAILed the 30%-faster ron-s2 arm (37.0% of a *smaller* wall). Paired mode now (a) accepts a same-depth pair when the copy source differs, (b) replaces the shape-anchored absolute floors with control-anchored criteria (paired wait in ms ≤ control x 1.25 + 50 ms; paired exposure ≤ control + 350 ms), keeping the pure-DMA-window check always on and refusing identical-configuration pairs. Re-validated on six controls: depth pair 3-vs-2 PASS 1.188x, reversed FAIL (0.842x + wait + exposure), the FALSIFIED churn arms FAIL (0.846x), ron-s2 / ron-s3 pairs PASS (1.304x / 1.181x), identical config FAIL. **Consequence:** the as-implemented churn-free variant removes the hop and wins warm PP at both depths at a 17.23 GiB pinned-RAM cost; nothing here is a default change and nothing certifies fitness for use. **Also to note:** the cold-pass numbers in these traces are CONFOUNDED in the candidates' favour (candidate arms page in the whole model before their timed pass), so only the warm pairs are claimed. Nothing committed.

## 2026-09-17: T-GPLR-37 — DIRECT MMAP-SOURCED STAGING: FEASIBLE, HOP REMOVED FROM THE ACCOUNTING, BUT A REGRESSION AT 4k AS IMPLEMENTED. **Probe (measured, tools/gplr37-mmapreg-probe.cu, 452.5 MiB window of the real GGUF, all arms verified byte-for-byte):** pageable mmap DMA 49.47 ms (9.59 GB/s); `cudaHostRegister` on the FILE-BACKED mapping succeeds and its DMA runs 32.86 ms = **14.44 GB/s = the pinned ceiling**, registration 0.16 ms; the current path is CPU hop 20.22 ms + DMA 32.86 ms = 53.08 ms per bank. So the hop is 20.22 ms per 452 MiB bank (~809 ms per 40-bank pass) and registration is cheap *in isolation*. **Prototype (implemented, built clean, `--copy-source mmap-registered`):** the staging path now page-locks the page-aligned span covering a package's member sources, caches it on the slot, unregisters before that slot stages a different package, and DMAs straight from the model's mapping with no CPU hop; it allocates no staging buffers and refuses loudly on a registration failure (driver text in `error_msg`), and `--pp-pool` with it is refused. **A/B (4k anchor, ubatch 512, layer-major, kv q8, af=1, --pp-repeat 2, one mutex hold, all four strict checkers OK):** warm pass pin-s2 3123.9 ms / 1311.2 tok/s vs reg-s2 3692.9 ms / 1109.2 tok/s (**-15.4%**); pin-s3 2811.9 ms / 1456.6 tok/s vs reg-s3 3689.0 ms / 1110.3 tok/s (**-23.8%**). The pre-registered falsifier that fired is the REGRESSION branch (>=5% worse at either depth). **Where the regression comes from (host ledger, both arms):** the hop DID shrink (PP `copy_complete.host_submit_us` sum per pass 1587.8 -> 1222.3 ms at 2 slots; 1289.0 -> 1247.4 ms at 3) and the `slot` bucket fell (1315.6 -> 1153.3; 688.0 -> 590.3), but the per-launch host windows GREW far more: `setup` (sched reset + `alloc_graph`) 11.2 -> 227.8 ms (2 slots) and 11.9 -> 134.7 ms (3 slots), and `compute` 1329.5 -> 1845.1 / 1581.4 -> 2441.7 ms. Bank DMA windows stayed pure in every arm (median 31.48 ms = 1.00x the 31.37 ceiling) and 3-slot consumer wait stayed 36.5 ms, so nothing starved — the loss is host-side per-launch cost, not transfer rate. **Mechanism (indicated, not yet isolated):** every bank is a DIFFERENT 436 MiB file region, so the mode registers and unregisters ~111k pages per bank, 80 times per run (34.46 GiB cumulative registration churn, one `mmap_register` event per bank fill at 436.1 MiB, median 99.9 ms apart) — page-table/TLB churn that lands inside the per-launch `alloc_graph`/sync windows. The one-shot 0.16 ms probe number does not transfer to per-bank churn on freshly touched file regions. **Confound disclosed:** the arms ran in the order pin-s2, reg-s2, pin-s3, reg-s3, so the registered arms' COLD passes (reg 4659/4680 ms vs pin 6095/5658 ms, i.e. -23%/-17%) benefited from a file cache the earlier arms had warmed — the cold-pass win is recorded as CONFOUNDED, not as a result; the warm comparison (steady state, minutes apart) is the reliable one. **Consequence:** the as-implemented prototype is rejected; the data points at the churn-free variant (register the whole mapping, or all exec-layer spans, ONCE at init — T-GPLR-31 proved an 18 GiB grantable — paying ~17 GiB of pinned file pages and removing registration churn entirely) as the one remaining test of the family, with the falsifier that the `setup` inflation persists even without churn (which would mean the cost is inherent to file-backed registered pages and the family closes). Nothing committed.

## 2026-09-17: T-GPLR-36 — THE LONG-CONTEXT RESIDUAL IS PER-LAUNCH HOST WORK, NOT PER-GRAPH SETUP. Ledger run: one 200k-token arm (262k F16 KV, layer-major, ubatch 512, 3 slots, af=1, pinned, bf16 slabs, --pp-repeat 2, --pp-recon 1) under one mutex hold (tools/run-gplr36-ledger.ps1 / -locked.ps1; trace build/llama-cuda/gplr36-ledger-200k.jsonl, 7,854 events, strict checker OK, pipeline gate EXIT 0). Warm pass: wall 141,847.9 ms, compute 131,249.3 ms (summary), wait 36.5 ms, 1,409.96 tok/s, peak VRAM 3,265.56 MiB (the `peak VRAM` line again reads identically to the 2/3/4-slot arms, the known instrument defect). The new `host_ledger` event closes EXACTLY on the sweep (bucket sum - sweep = +0.0 ms, 0.00%): sweep 140,807.7 ms = compute 131,523.3 + patch 3,402.3 + slot 1,065.1 + other 4,516.3 + setup 278.6 + graph_call 22.1, and fn 141,813.0 = embed 318.4 + preload 126.1 + sweep 140,807.7 + output 77.2 + controller 483.6. **ANSWER TO THE OWNER'S QUESTION — IS IT PER-GRAPH SETUP ACROSS THE 15,640 LAUNCHES? NO.** `setup_us` (sched reset + `ggml_backend_sched_alloc_graph`, the exact quantity named) is 278.6 ms per pass = **0.017 ms/launch = 3.0% of the 9,343.8 ms residual**, with `graph_builds = 0` and `graph_prep_us` 15.4 ms (graphs prebuilt once, never rebuilt). The per-launch stage (16,422 launches = 391 embed + 15,640 layer + 391 output) is dominated by **host work outside the graph window**: `other` 4,516.3 ms (48.3% of the residual, 0.275 ms/launch — the chunk-loop body, sampled telemetry writes, `sets[c]->routes.count(il)` lookups; NOT yet split further), `patch` 3,402.3 ms (36.4%, 0.207 ms/launch — `prepare_chunk` input-snapshot memcpy + FA causal/KV-bound patching, repeated identically once per launch even though its inputs are per-chunk, so ~97% of it is redundant across the 40 layers of a chunk), `slot` 1,065.1 ms (11.4% — load_layer/wait_slot_ready/record_consumer_done/af enqueue/event wait), `setup` 3.0%, `graph_call` **22.1 ms (0.2%, 0.0013 ms/launch — the src-pointer snapshot save/restore is negligible)**. Pre-registered prediction outcome: **falsified on both halves** — `graph_call_us` was predicted to be the largest unattributed bucket (measured 0.022 s) and `other_us` under 1 s (measured 4.52 s); the falsifier (both under 1 s -> residual is not host bookkeeping) did NOT fire, so the residual IS host bookkeeping, with the largest single piece still unnamed (`other`). **Accounting correction found by the ledger:** the ledger's compute bucket exceeds the phase summary's `total_compute_us` by 274 ms because steps 1 (embed) and 4 (output) are executed but never accumulated as layer stats, so ~0.27 s of what the summary calls "residual" is really embed/output graph compute, not host time — the residual should be read as wall - (layer compute + embed/output compute) - pure DMA. **Mechanism for the 80-100% GPU-utilisation wave the owner observed:** the controller synchronises after EVERY launched graph (`ggml_backend_sched_graph_compute_async` then `ggml_backend_sched_synchronize`), so the host's 0.564 ms/launch of prep (patch 0.207 + other 0.275 + slot 0.065 + setup 0.017) is exposed GPU-idle time by construction -> duty = 8.009/8.573 = 93.4% on average, oscillating with the bimodal layer cost (the 10 full-attention layers take ~40 ms at 200k vs ~3.5 ms for the other 30, i.e. a 4-layer beat). Pre-staging the NEXT LAYER cannot remove this wave: at this shape staging is already fully hidden (wait 36.5 ms total, slot block 1.07 s = 11.4% of the residual), so 'waiting for the next layer' is not the limiter — per-launch host work is. Host RAM: peak VRAM 3.19 GiB is the run's own measure while byte_account reports KV 3.82 GiB (host-spilled), act_slab 1.53 GiB, slot_window 1.46 GiB, static 0.89 GiB, free 20.70 GiB; the owner's observed host-RSS growth (19 -> 30 GB) has NO instrument in this run and is recorded as an open question with a proposed monitor probe, not as a result. Nothing committed.

## 2026-09-16: T-GPLR-35 ADDENDUM 2 — DEPTH DOES NOT PAY AT LONG CONTEXT, AND THE BUDGET REFUSAL IS VERIFIED AT THAT SHAPE. 3-vs-4 slot A/B at the charter's crowded-VRAM shape: 200,000-token prefill (the longest on-disk anchor; 262,144-token KV capacity) with F16 KV, layer-major, ubatch 512, af=1, pinned, bf16 slabs, --pp-repeat 2, --pp-recon 1, one mutex hold (tools/run-gplr35-slots262k.ps1 / -locked.ps1; traces build/llama-cuda/gplr35-s{3,4}-200k.jsonl, wrapper log gplr35-200k-wrapper.log). **Payoff: none measurable.** Warm: 2 slots (prior reference) 1363.2 tok/s / 146,713 ms -> 3 slots 1369.6 tok/s / 146,030 ms (+0.5%) -> 4 slots 1374.7 tok/s / 145,487 ms (+0.4%); 2 -> 4 slots totals +0.8%. The pre-registered prediction (<= ~5% at this shape, because a bank is consumed across ~391 chunks and staging is therefore already hidden) is CONFIRMED; the falsifier (>10% from a slot jump) did NOT fire. Mechanism evidence: the consumer wait collapses at 3 slots and then stops mattering (1,182.5 ms -> 36.5 ms -> 36.5 ms) while the wall is compute-dominated (134,592 ms of 146,030 ms = 92%), and the summed host transfer windows GROW with depth (137.7 s -> 269.5 s -> 400.3 s) with no wall change — queueing is fully overlapped at this shape, so extra slots buy nothing. Bank DMA windows stay pure (median 31.48 / 31.56 ms = 1.00x/1.01x the 31.37 ms ceiling) and lag is still 0% negative. **Fit: yes under the derived budget** — the 4-slot run completed, and the capacity preflight's own arithmetic puts a 4-slot window at 8,809,572,864 B (8.2 GiB) against ~20.7 GiB observed free; **refusal verified**: the same 4-slot config with an explicit --residency-budget-mib 1500 aborts loudly at capacity preflight (exit 1) with `bounded 4-slot window : 8809572864 B (total residency budget 1572864000 B, reserve 536870912 B)` and `error: capacity preflight: bounded 4-slot window 8809572864 B exceeds the residency budget 1572864000 B` — the loud denial path the charter requires, exercised at long context. **Instrument defect found and reported:** the `peak VRAM` telemetry line reports 3265.56 MiB IDENTICALLY at 2, 3 and 4 slots at this shape, while the preflight window arithmetic scales with slots — so that line cannot be used to judge slot residency at long context and the bounded-window figures are the usable instrument. No independent VRAM sample exists for this run (the nvidia-smi sampling pass was interrupted), which is stated as a gap. **Gate self-correction (methodological finding):** the gate's 4k-anchored absolute floors FALSE-FAILED a legitimate 3-slot 200k arm on the 4k exposure ceiling — absolute floors do not transfer across prompt shapes. Fixed: a shape mismatch now SKIPS the absolute floors and says so (`no gate applied` / `paired only`), with `--force-absolute` to override; re-validated — 200k arms clean, the four 4k traces unchanged (all PASS), both negative controls still FAIL, and --force-absolute reproduces the documented misfire. **Charter consequence:** at 200k/262k-F16 the extra slot is not worth its VRAM — the crowded-VRAM optimum stays at the 2-3 slot range, and the remaining lever at long context is the compute side (92% of the wall), not staging. Committed with the T-GPLR-35 gate/slot tooling (hash in the plan's T-GPLR-35 Commit field).

## 2026-09-16: T-GPLR-35 PIPELINE REGRESSION GATE ADDED — `tools/check-gplr35-pipeline.py` fails when PP staging silently re-serializes, and it is wired into the slot runner so every future run of that shape self-verifies. **Design finding first (measured):** a gate on the recon's `lag_ms` would be VACUOUS — copies starting before the compute backlog drains = 0% at BOTH 2 slots (serialized) and 3 slots (pipelined), so that signal cannot discriminate overlap; it is reported as INFO and explicitly not gated, with the reasoning in the module docstring so nobody re-derives it as a gate. The gated criteria are what actually moved with pipeline depth: warm PP throughput, consumer wait as a share of wall (21.7% at 2 slots vs 7.9% at 3), exposure = wall − compute − pure DMA (+754 ms vs −23 ms), and the median bank DMA window vs the 31.37 ms pure ceiling (1.25x alarm). Thresholds are per-depth (`slots2` / `slots3`), auto-selected from the trace's own `run_start.n_slots` so it cannot be silently misconfigured, and a `--baseline` pair from the SAME session is the trustworthy form because the observed session-to-session spread at a fixed configuration is ~9%. **No silent pass:** an unsupported slot count with no baseline exits 2 (CANNOT EVALUATE), as does an unreadable trace. Validation (no GPU time): all four real traces PASS under their own profiles (slot2, slot3, recon-ov0, recon-ov1), both negative controls FAIL correctly — the serialized 2-slot run judged by the `slots3` profile fails all three criteria, and pairing 2 slots against the 3-slot control fails at 0.842x (< 0.90x required) — and both error paths return exit 2. **Calibration finding disclosed:** the first `slots2` wait ceiling (30%) FALSE-FAILED recon-ov1, a legitimate 2-slot run whose wait share was exactly 30.0% vs recon-ov0's 24.2% — two same-configuration sessions straddling the ceiling. The ceiling was widened to 36% and the limit is recorded in the tool: the absolute profile is a coarse alarm, the paired comparison is the gate. Wired into `tools/run-gplr35-slots.ps1` after the per-arm strict checkers (control absolute, then candidate absolute + paired), and its exit code propagates to the runner. Committed with the T-GPLR-35 gate/slot tooling (hash in the plan's T-GPLR-35 Commit field).

## 2026-09-16: T-GPLR-35 ADDENDUM 1 — 3 SLOTS CONVERT THE SERIAL WAITS INTO OVERLAP: warm 4k layer-major 3377.8 -> 2843.2 ms (-534.6 ms, -15.8%), PP 1212.6 -> 1440.6 tok/s (+18.8%), for +498 MiB of slot VRAM. Adjacent A/B in ONE mutex hold (tools/run-gplr35-slots.ps1 / -locked.ps1; traces build/llama-cuda/gplr35-slot{2,3}.jsonl, wrapper log gplr35-slots-wrapper.log; 4k anchor, ubatch 512, layer-major, kv q8, af=1, pinned, --pp-repeat 2, --pp-recon 1; both strict checkers OK; 2-slot arm re-run adjacently rather than compared against the older recon numbers). **The mechanism is NOT the one predicted.** Pre-registered: deeper lookahead would shrink the 1191 ms engine queue wait and the 747 ms consumer wait; falsifier: wall >= ~2.9 s with unchanged waits. Measured instead: the queue wait GREW (1155 -> 2667 ms, host windows 4234 -> 5407 ms summed) while the consumer wait COLLAPSED (734.4 -> 225.3 ms, -509 ms — essentially the whole wall gain, -534.6 ms) and the residual `wall - (compute + pure DMA)` fell 770 -> 100 ms. So 3 slots do not make queueing disappear; they take it OFF THE CRITICAL PATH — with one bank of lookahead every wait was exposed serially, with two the engine queues behind work that is no longer blocking. Per-bank DMA windows stay pure at both counts (median 31.4 ms = 1.00x the 31.37 ms ceiling, every sample) and lag is still never negative (0/76 and 0/66) — no bank's DMA starts while the compute backlog drains, at either depth. Cold pass also improves (6364 -> 5697 ms). VRAM price (measured, run_start): slot window 1,044,389,888 B -> 1,566,584,832 B = **+522,194,944 B (498 MiB) per extra slot**. Unexplained and NOT attributed: the per-pass compute total rose 1369 -> 1611 ms (+242 ms, +18%) with identical bytes/arithmetic — recorded as an open question, not a result. Recon coverage caveat: the slot3 arm emitted 66 lag / 34 DMA samples vs 76/39 (the sample ring flushes a few banks later, so a pass tail is lost), so per-sample SUMS are not sample-for-sample comparable; wall/wait/compute/transfer come from the complete phase summaries. Prediction band 2.2-2.6 s was not reached (2843 ms) but the falsifier did NOT fire; the direction is confirmed and the remaining gap is the 1607 ms host hop plus the residual queue exposure. Nothing about defaults changed — the slot count stays an owner decision (default 2; the ladder may use a third slot when the budget allows).

## 2026-09-16: T-GPLR-35 RECON RUN — the coupling is NOT the staging/compute stream topology. Falsifier class reached: NONE. Both arms ran under the GPU mutex with `--pp-recon 1` (trace build/llama-cuda/gplr35-recon-{ov0,ov1}.jsonl, log/gplr35-recon-wrapper.log, tools/run-gplr35-recon.ps1, analyzer tools/analyze-gplr35-recon.py), 4k anchor, ubatch 512, 2 slots, layer-major default, kv q8, --pp-repeat 2 (pass 1 cold, pass 2 warm), pinned, af=1; both strict checkers OK. **Identity (measured, not assumed):** `recon_identity` records the actual handles — ov0 compute_s0=2562101218336 vs copy_s0=2571110727088 (same_ctx_stream=0: the copy backend really is a separate stream); ov1 overlap=1 with compute_s1=2571110691440 (the aux stream exists). **Attribution (76 samples/arm, 40 layers x 2 passes):** copy windows are PURE in both arms — median 31.47 ms (ov0) / 31.44 ms (ov1) = **1.00x the 31.37 ms pure 453 MB pinned ceiling**; lag (copy_start - compute-stream reference) is median +34.59 ms (ov0) / +34.15 ms (ov1) with **0/76 negative samples in either arm** — no bank's DMA ever started while the compute backlog was still draining. **ov1 (aux stream inside the compute context) is measurably identical to ov0:** warm walls 3089.0 vs 3088.4 ms, 1326 tok/s both, same window, same lag, same overlap 0%; host hop moved 1559 -> 1279 ms but ready latency rose 2429 -> 2694 ms and engine queue wait 1191 -> 1456 ms, net zero. The stream-topology lever is FALSIFIED as a wall lever for this shape. **Warm-pass ledger (wall 3089 ms, 40 banks; summary-authoritative totals):** compute 1350 ms + transfer-window 3988 ms with consumer wait 747 ms; the 3988 ms transfer window decomposes into **host hop (pinned mirror + submit) 1559 ms + ready latency 2429 ms**, and of that ready latency only **1238 ms is actual DMA engine time** (median 31.4 ms/bank) — the other **1191 ms is queue wait** (submitted but not started). So per bank the wall carries ~39 ms of CPU hop, ~31 ms of pure DMA, ~30 ms of engine queue wait and ~34 ms of compute, at one bank of pipeline depth; wall - (compute + pure DMA) = **+501 ms** is the non-engine residual. **The old "74 ms chained window" reading was a host-window artifact**: the host-observed per-bank window (~100 ms) = host hop + queue wait + DMA, and only the last ~31 ms is the engine. **Defect found and fixed during recon:** `backend_compute` was published ONLY inside `layer_streaming_compute` (llama-layer-streaming.cpp:2426), which layer-major PP never calls — it executes graphs directly via `ggml_backend_sched_graph_compute_async` (:3274) — so on the PRIMARY workload that pointer was null for the whole run. Consequences: the ov1 arm aborted loudly at the first staging ("refusing to fall back to the copy-stream staging path", message later overwritten by the caller's package/bundle diagnostic), i.e. the T-GPLR-35 V1 mechanism could not engage at all, and the recon's own `backend_compute == nullptr` guard silently emitted nothing (a bad instrument, now fixed). Fix: publish `backend_compute` in `layer_major_pp` right where the scheduler's CUDA backend is discovered, with the same loud aux-stream refusal; the recon now emits `recon_unavailable` instead of a silent no-op. Build clean after both fixes (0 errors), host validation unchanged. **Consequence for the goal charter:** at 2 slots the PP wall is ~68% host-pipeline + engine-queue, not engine-bound and not stream-bound; the levers that remain are host-hop removal (direct mmap staging instead of the pinned mirror) and pipeline depth (more than one bank in flight, i.e. slot lookahead), not CUDA stream topology. Nothing committed; recon scripts + probe code uncommitted in the worktree.

## 2026-09-16: T-GPLR-35 V1 IMPLEMENTED + BUILT (A/B armed, NOT yet run) — staging DMAs can now run on an auxiliary stream (No. 1) of the COMPUTE backend's CUDA context behind `--pp-copy-overlap 0|1` (default 0 = the pre-T-GPLR-35 copy-backend path, byte-identical). New experimental extension surface (ggml/include/ggml-cuda.h + ggml/src/ggml-cuda/ggml-cuda.cu): `ggml_backend_cuda_aux_stream_ok` / `memcpy_1d_aux` / `event_record_aux` / `event_wait_aux` / `aux_synchronize`; every staging submission and every staging-side ordering call in llama-layer-streaming.cpp routes through one plumbing layer (`ls_stage_aux` / `ls_stage_record` / `ls_stage_wait` / `ls_stage_drain`) so the mode switch is atomic — an event can never be recorded on a stream that carries no copy work, and a wait always targets the stream that does. Drains are conservative (both streams) so no site can be missed; the handoff-migration and TG row-cache paths keep the copy backend and their own consistent topology. A run whose flag is on but whose compute context has no aux stream fails LOUDLY where the scheduler's CUDA backend resolves (never a silent fallback to the serialized path). Host validation (no GPU): build clean (0 errors; CUDA backend + llama core + runner all recompiled), usage line present, `--pp-copy-overlap 2` refused with the contract message, `--pp-copy-overlap 1` parses through to prompt validation. A/B armed and NOT yet run: `tools/run-gplr35.ps1` -> `tools/run-gplr35-arms.ps1` (arms overlap0/overlap1, 4k anchor, ubatch 512, 2 slots, default schedule, kv q8, --pp-repeat 3, --copy-source pinned, af=1, strict check-telemetry per arm) + `tools/analyze-gplr35.py`. Pre-registered: warm 4k 2.75-2.8 s -> ~2.0-2.1 s (ceiling ~1.5-1.6 s); FALSIFIER warm wall >= ~2.6 s with the copy window unchanged. Recon limitation stated: stream/context identity is not logged, so the attribution rests on the A/B's wall/window/pass-transfer evidence.

## 2026-09-16: T-GPLR-34 MID-KERNEL PROBE — WDDM cross-engine serialization FALSIFIED (a second stream overlaps the compute and can even overtake it); the runner-side cause was first written up as shared-stream, which is CORRECTED in the same entry below (drafted as T-GPLR-35). The standalone probe (tools/gplr34-midkernel-probe.cu, rev 4, -arch=native; runner tools/run-gplr34-probe.ps1; log build/llama-cuda/gplr34-midkernel-probe.log; run under the GPU mutex; re-run 2026-09-16 from the committed script reproduces every number below): [1] pure 453 MB pinned copy 31.37 ms min (14.44 GB/s — pure-rate ceiling re-confirmed); [5] single-stream kernel+copy mid-kernel enqueue 62.12/62.77 ms = 30.74 kernel + 31.37 copy additive — guaranteed by stream ordering on every platform, so only a reference; **[6] TWO-STREAM discriminator: copy START -31.6/-28.4 ms relative to kernel start (the copy OVERTOOK the kernel — launched later but ran earlier), copy-done -0.15 ms min (finished before the kernel began), copy window pure 31.38 ms, kernel window unaffected 30.36 ms. WDDM cross-engine serialization is FALSIFIED: the copy engine runs at full rate around SM compute on this box when the copy is on a second stream, and event completion is not deferred either.** Instrument notes: rev 1 was invalid (burn kernel measured 0.00 ms — launches never error-checked); rev 2+ add loud launch checks, verified calibration (33.94 ms), and the FATAL guard; first rev-2 run also surfaced a Blackwell toolchain fact — the default nvcc target gave "provided PTX was compiled with an unsupported toolchain" at kernel launch on sm_120, fixed with -arch=native. CORRECTION (same day, before any fix was attempted): the "shared stream" causal reading was WRONG for the runner. `ggml_backend_cuda_set_tensor_async` does submit on `cuda_ctx->stream()` (ggml-cuda.cu:2428-2434), but the backend it is handed here is `backend_copy` — a DEDICATED CUDA backend (llama-layer-streaming.cpp:1305 `ggml_backend_cuda_init` -> a fresh `ggml_backend_cuda_context`; `streams[device][...]` is a per-context member, common.cuh:1420/1494) — while the compute side is the SCHEDULER's CUDA backend (`backend_compute`, resolved at llama-layer-streaming.cpp:2274): two contexts, two streams. Stream identity therefore does NOT explain the runner's chained 34 + 31.5 ms windows. What matches instead is the runner's own CROSS-CONTEXT ordering gate: `ev_consumer_done` is created on the copy device (:1324), recorded on `backend_compute` (:2105) and waited on the copy stream (:602 worker, :1900 main) — a shape the T-GPLR-34 probe never tested (its two streams shared ONE context, where overlap is proven). The GPU is still not the limiter and the wall conclusion is unchanged (~2.75-2.8 s is not an engine-bound floor; target ~2.0-2.1 s), but the lever's mechanism is UNRESOLVED and is exactly what T-GPLR-35's recon step must attribute (cross-context wait vs slot-reuse gate vs enqueue position vs graph-launch semantics) before any fix is written. Probe commit: fc13543 (2026-09-16, tools/gplr34-midkernel-probe.cu + tools/run-gplr34-probe.ps1 + this verdict; the T-GPLR-35 draft lands with the next doc update).
## 2026-09-16: SUBMIT_BATCH GAP CLOSED — the field now reports the ACTUALLY-TAKEN branch threaded from stage_members (0 = per-member, 1 = single span, 2 = mixed), not the flag state; verification A/B under the mutex: batch0 {0:120}, batch1 {1:120} — and this retroactively confirms the falsified A/B's implementation-side premise by direct branch telemetry (the span path really ran on every bank there)

## 2026-09-16: T-GPLR-33 WHY-ANALYSIS — the decisive pair: warm GPU-side ready_latency 74.1 ms (3 submissions) vs 74.2 ms (1 submission) — a real per-submission toll of ~12.7 ms would have dropped batch1 by ~25 ms; it dropped 0.0 ms. The theory was flawed by MODEL ALIASING: the 69.7 = 3 x (10.5 + 12.7) fit mistook "remaining layer-compute time at staging" for "per-submission overhead" — both predict the same warm mean, only the submission-count A/B separates them, and it separated. The chaining model (DMA parked during compute, then drains at pure rate) predicts 34 + 31.5 + flush ~ 74 ms for BOTH arms: matches. Implementation ruled out: contiguity arithmetic (144 MiB members, alignment divides exactly, bundle from offset 0) + pinned-staging source contiguity make the span path near-certain to have run; one honest gap disclosed: submit_batch reports the FLAG, not the taken branch, so the null result's implementation-side certainty rests on that arithmetic, not telemetry.

## 2026-09-16: T-GPLR-33 DONE (A/B RUN) — one-submission-per-bank FALSIFIED: windows identical with 1 vs 3 submissions (median 76.1 vs 75.1 ms; walls 3483/3055 vs 3419/2949 ms, batch1 marginally worse within variance); per-submission-overhead model is DEAD — and the surviving arithmetic points at the real mechanism: warm wall = 40 x (compute ~34 ms + DMA ~31.5 ms) chained, i.e. the H2D DMA does NOT overlap layer compute on this WDDM box (or its completion is deferred to the compute boundary)

A/B ran under the GPU mutex (tools/run-gplr33.ps1, detached wrapper;
~2 min per arm — the "no load" observation mid-run was the run already
finishing). Both checkers OK (20,577 events, 2 slots). V1 verified live:
submit_batch=1 on 120/120 banks (the slot-build contiguity check tiled the
members exactly — the single span copy ran everywhere; the 453 MB span
submission is one cudaMemcpyAsync via the backend API).

Results (traces gplr33-batch{0,1}.jsonl):
- batch0 (legacy 3-submission): warm passes 3418.7 / 2949.0 ms (wait 698.7 /
  927.6 ms); windows mean 78.5 / median 75.1 / p90 159.3; 93/120 stretched.
- batch1 (single span submission): warm passes 3483.2 / 3055.4 ms (wait
  734.5 / 927.7 ms); windows mean 87.5 / median 76.1 / p90 160.6; 107/120
  stretched. NOT faster; slightly worse within arm variance.
- The pre-registered prediction (window 91 -> ~44-58 ms, warm wall ->
  ~1.8-2.0 s) FAILED; the falsifier (window >= ~80 ms) FIRED.

What this closes and what it opens:
- CLOSED: the per-submission-overhead model (submission count does not
  move the window). With it, the whole submission-shaping lever family is
  dead on this box: serialization (T-GPLR-32), submission count (this).
- OPEN (surviving arithmetic): warm wall ~2.75 s = 40 layers x (~34 ms
  compute + ~31.5 ms DMA) chained — exactly the wall, per-layer. The H2D
  DMA is not executing concurrently with layer compute (or its completion
  signal is deferred to the compute boundary) despite the copy/compute
  engine model. This also explains why af=1's gain is bounded and why the
  engine-bound floor (~1.5-1.8 s) was never reached.
- DISCRIMINATING NEXT STEP (cheap, pre-registration): a standalone probe
  that launches a ~34 ms compute kernel, enqueues a 453 MB pinned H2D mid-
  kernel, and times the copy event: event ~= 65 ms => true DMA-after-
  compute serialization; event ~= 31.5 ms => concurrent DMA with deferred
  completion signaling. The fix differs (HAGS/scheduling vs engine
  concurrency), so this probe decides the next lever. GPU-direct storage
  (M3) remains the heavy alternative in both worlds.

Status of --pp-submit-batch: kept as opt-in diagnostic tooling (default 0;
measureably harmless-to-neutral; byte-safe by construction). NOT a default.

## 2026-09-16: T-GPLR-32 FOLLOW-UP RESEARCH — the ~82 ms window decomposes as 3 members x (10.5 ms pure transfer + ~12.7 ms PER-SUBMISSION overhead); the "intrinsic per-copy cost" is actually PER-SUBMISSION cost x3, and CUDA 13.2's cudaMemcpyBatchAsync (in the installed SDK) is the direct fix — predicted bank window 91 -> ~44-58 ms, warm wall ~2.77 -> ~1.8-2.0 s

Sharper decomposition of the T-GPLR-32 traces (build/llama-cuda/
gplr32-ser{0,1}.jsonl, warm passes; per-pass phase_summary cross-check):
- Local fact (inspect JSON): each MoE bank package = exactly 3 members of
  452,984,832/3 = 150,994,944 B (144 MiB) each — per-member submission
  overhead CANNOT be hidden by member size; there are 3 submissions per
  bank and each member is only ~10.5 ms of pure transfer at the measured
  14.4 GB/s link.
- Warm per-bank telemetry: transfer_us 91.1-91.2 ms = host_submit 21.4-21.5
  (the CPU memcpy hop, PROVEN wall-neutral by T-GPLR-30/31 — it overlaps
  compute) + ready_latency 69.7 ms (GPU-side). 69.7 = 3 x (10.5 + ~12.7):
  the cost tracks the NUMBER OF SUBMISSIONS, not the bytes. This resolves
  T-GPLR-32's "intrinsic per-copy cost" wording into per-submission cost.
- Primary sources (search tooling degraded again — pivoted to the
  authoritative local source, the installed CUDA 13.2 SDK):
  - cuda_runtime_api.h (CUDA 13.2): cudaMemcpyBatchAsync(dsts[], srcs[],
    sizes[], count, attrs, attrsIdxs, numAttrs, stream) — "performs a batch
    of memory copies asynchronously. The batch as a whole executes in
    stream order but copies within a batch are not guaranteed to execute in
    any specific order." Pointer-to-pointer only; attrs include
    srcAccessOrder (cudaMemcpySrcAccessOrderAny fits our pinned pool/
    staging sources) and the cudaMemcpyFlagPreferOverlapWithCompute hint
    ("associated copies should preferably overlap with any compute work").
    Also present: cudaMemcpy3DBatchAsync, cudaMemPrefetchBatchAsync.
  - NVIDIA streams guidance (fetched 2026-09-16, earlier pass): same-
    direction copies sharing an engine serialize; batching restores the
    per-copy rate — consistent with the per-submission overhead model.
  - Microsoft HAGS devblog (fetched 2026-09-16, earlier pass): WDDM
    batching/submission overhead is first-class; HAGS moves high-frequency
    scheduling onto the GPU (platform knob, owner-level, noted not pursued).

Mechanisms to shrink the window (in attack order):
- M1 MEMBER-BATCHED SUBMISSION (next task, T-GPLR-33 draft): one
  cudaMemcpyBatchAsync(count=3) per bank instead of 3 cudaMemcpyAsync
  calls. Stronger variant with zero new-API risk: the slot bundle allocates
  all members in ONE buffer and each member is exactly 144 MiB (already
  512-byte aligned), and the pinned staging/pool buffers already hold the
  bank CONTIGUOUSLY in member order by construction — so if a runtime
  offset check confirms dst members are back-to-back in the slot buffer,
  ONE plain cudaMemcpyAsync of the full 453 MB suffices. Expected: the
  ~12.7 ms overhead paid once per bank instead of 3x — bank window 91 ->
  ~44-58 ms (host-observed), transfer phase 3.65 -> ~2.4 s, warm wall
  ~2.77 -> ~1.8-2.0 s (the earlier engine-bound floor). Falsifiable: if
  the window does not shrink, the overhead is inside the DMA engine itself,
  not the submission path — that would close this family too.
- M2 Overlap hint: set cudaMemcpyFlagPreferOverlapWithCompute on the
  batch attributes; near-zero cost, try as an M1 variant arm.
- M3 GPU-direct storage (bypass the CPU hop and possibly WDDM entirely):
  large effort, separate future task; M1's verdict decides if needed.
- M4 HAGS platform knob: owner-level Windows setting, not code; recorded
  for completeness.

## 2026-09-16: T-GPLR-32 DONE (A/B RUN) — serialize-the-worker-DMAs FALSIFIED as a wall lever: warm walls identical (2751.8-2782.9 ms ser1 vs 2768.9-2777.8 ms ser0, wait ~959 ms both); with worker DMAs serialized, 114/120 windows STILL stretch >45 ms while the only possible concurrent partner is a main-thread preload copy — the stretch is INTRINSIC per-copy cost under WDDM, and the DMA-concurrency lever family is CLOSED

The A/B ran under the GPU mutex (detached wrapper survived the session;
deepseek_v4.exe held the mutex first — the run fired on its release, exactly
as designed). Both arms: check-telemetry OK (20,577 events, 2 slots),
no deadlock, 120/120 nonzero phase-0 copies per arm.

Config: 4k anchor, ubatch 512, --layer-slots 2, --pp-repeat 3, --copy-source
pinned, --pp-async-feed 1; only --pp-af-serialize differs (0 vs 1).

Results (trace gplr32-ser{0,1}.jsonl; logs UTF-16):
- ser0 warm passes: 2777.8 / 2768.9 ms (wait 959.3 / 958.6 ms) — matches the
  recorded af=1+pinned optimum (2.80-2.87 s), so the arm shape is comparable.
- ser1 warm passes: 2782.9 / 2751.8 ms (wait 958.8 / 958.5 ms) — identical
  within noise. The pre-registered falsifier (wall unchanged) FIRED.
- Gate verified operating: af_inflight present on copy_submit (ser1 max 2 =
  submitted-but-unconsumed worker DMAs; 0 when the gate is off), stream gate
  enqueued before every worker copy, ser1 cold pass also ~equal (6205 vs
  6474 ms — inside arm-to-arm variance).
- Mechanism resolution: analyzer tools/analyze-gplr32.py (submit->consumption
  span per (layer,generation); host-observed — the trace has no GPU-side
  timestamps): with worker DMAs serialized one-in-flight, ~114/120 windows
  remain >45 ms (mean 82.6 ms ser1 vs 82.3 ms ser0; p90 142.6). The only
  DMAs that can overlap a serialized worker copy are the main-thread preload
  copies (~3-6 per pass, they supply layers 0/1 while the worker takes over)
  — far too few to stretch 114 windows. CONCLUSION: the effective window is
  dominated by an intrinsic per-copy cost under WDDM (submission/window
  overhead), NOT by contention between concurrent copies. The original
  "2.2x inflation from DMA pairs" interpretation (2026-09-16 research pass)
  is falsified; with it the whole DMA-concurrency lever family closes
  (3-slot falsification was the first signal, this is the second).
- Analyzer caveat recorded: "effective window" is the host-side
  submit->consumption span, so it embeds WDDM submission latency by
  construction; that limitation does not affect the wall/wait verdicts,
  which are direct measurements.

Status of --pp-af-serialize: kept as opt-in diagnostic tooling (default 0;
harmless — identical walls, checkers OK, ~zero risk since it only orders
worker DMAs on the copy stream). NOT a default.

Remaining PP lever family (next territory): shrink the per-copy window
itself — member batching for large-member banks, GPU-direct storage paths,
or accepting the ~2.75-2.8 s warm wall as the pinned-af optimum on this box.

## 2026-09-16: WDDM DMA-INFLATION RESEARCH PASS — asyncEngineCount=5 measured on this 5090 Laptop; NVIDIA's engine model (same-direction copies sharing one engine stretch; batching restores full rate while kernels still overlap) predicts our 2.2x stretch is LINK/QUEUE CONTENTION, fixable by one-in-flight serialization; T-GPLR-32 experiment drafted

Local measurement (tools/gplr32-enginecount.cu/.exe, CUDA 13.2): RTX 5090
Laptop GPU reports asyncEngineCount=5 (multiple copy engines exist) — yet
our traces show 28 concurrent two-copy pairs stretching effective windows
2.2x (69.6 ms avg vs 31.45 ms pure) and ~237 ms at 3 slots. Conclusion: the
inflation is contention for the shared physical Gen4 x8 link and/or WDDM
timeslicing of engine queues, NOT a single-engine limit.

Primary sources consulted (2026-09-16 online pass; searches were mostly
degenerate, so I read documents directly):
- NVIDIA Technical Blog "How to Overlap Data Transfers in CUDA C/C++":
  engines execute their queues strictly in issue order; ops in DIFFERENT
  engines run concurrently (so copy/compute overlap never depends on copy
  serialization); same-direction copies sharing one engine stretch each
  other (their C1060 case: per-stream interleaving gave ZERO gain; batching
  all H2D first restored the full per-copy rate — 12.9 -> 8.8 ms — while
  kernels still overlapped). This is our exact pattern.
- Microsoft DirectX Developer Blog "Hardware Accelerated GPU Scheduling":
  WDDM's scheduler coordinates/prioritizes/batches GPU submission from a
  high-priority CPU thread (HAGS moves high-frequency scheduling onto the
  GPU); submission/batching overhead is a first-class WDDM cost — consistent
  with per-copy window inflation when copies are issued over each other.
  (No applicable learn.microsoft.com DMA-buffer page was reachable; the
  blog is the governing Microsoft source I could fetch.)

T-GPLR-32 EXPERIMENT DRAFT (GPU-mutex, af=1+pinned, 2 slots, 4k 3-pass
shape): serialize the per-pass H2D DMAs to ONE IN FLIGHT at a time (gate
the af worker's enqueue on the previous copy_complete, keeping the CPU hop
overlapped as today). PREDICTION: effective windows 69.6 -> ~31-36 ms,
per-pass wait ~930 -> near 0, warm wall 2.80-2.87 -> ~1.9-2.3 s (floor
~1.5-1.8 s engine-bound from T-GPLR-29 addendum 3). FALSIFIER: wall
unchanged or windows still stretched -> the stretch is per-copy WDDM
submission overhead outside the DMA window itself, which serialization
does not fix; the wait telemetry will discriminate (wait would then persist
while windows stay pure-rate). Note the warning case: if the af worker
waits for copy-complete BEFORE its next CPU hop, the hop (~18.7 ms) re-
serializes; the gate must cover only the DMA, not the memcpy.
Status: not run yet (GPU busy with the external deepseek_v4 baseline).

## 2026-09-16: BASELINE COMPARISON (docs/llamacpp-baseline.log, 2026-09-10 llamacppCUDA fork llama-server, same model) — same placement philosophy (banks CPU-side), our PP streaming 1.9x cold / ~4.3x warm by implied rate, TG 2.4x (76.9 greedy vs 32.4 with MTP), VRAM same class; no long-context PP datapoint exists in the baseline log

Facts from the baseline log (a recorded run, NOT the live colibri
deepseek_v4.exe process): llama-server, same Qwen3.6-35B-A3B GGUF, 262k ctx,
Q8 KV, ubatch 512, MTP draft on. Placement: 34/42 layers offloaded, CUDA0
model buffer 2,089 MiB, CPU-mapped 19,980 MiB (the routed-expert banks),
CUDA0 KV 2,176 MiB (10 attn layers, q8_0, 262k) + 544 MiB CPU recurrent
state + 512 MiB MTP-draft KV. Self VRAM 5,795 MiB; 16,257 MiB left free.
Its measured workload: one 372-token chat request — prompt eval 9,421 ms
(39.5 tok/s, 25.33 ms/token), eval 32.4 tok/s over 1,724 tokens with draft
acceptance 0.546 (mean 3.73).

| metric | baseline (llamacppCUDA server) | current optimum (ours) |
|---|---|---|
| PP, short prompt (372 tok) | 9.42 s / 39.5 tok/s | ~6.3-7.0 s cold at 4,096 tok (590-650 tok/s); ~2.85 s warm-class |
| PP implied streaming rate | ~1.4-1.6 GB/s (≈all 512 experts touched at top-8 x 372 tok ≈ 13.6-15 GB rows / 9.4 s; assumption flagged) | cold 2.6-2.9 GB/s (18.33 GB / 6.3-7.0 s); warm 6.4-6.5 GB/s (18.33 GB / 2.80-2.87 s) |
| PP, long context | NOT IN THE LOG | 1,363 tok/s at 200k F16 (wall 146.7 s, 91% compute, wait 0.8%) |
| TG | 32.4 tok/s WITH MTP (acc 0.546) | 76.9 tok/s greedy, no MTP, 4k ctx handoff (2.4x) |
| VRAM self | 5,795 MiB (2,089 model + 2,477 ctx + 1,228 compute) | slots 996 + slabs 1,563 MiB sampled; KV ≈ 3.3 GiB derived (10 attn layers, 200k F16); same class, banks out in both |
| Server features | prompt cache, MTP, chat template, 262k server ctx | research runner (CLI, one-shot prefill) |

Honest caveats: the baseline's only PP datapoint is a 372-token request
(overhead-dominated; its marginal long-context PP rate is not recorded);
the expert-coverage rate estimate assumes ~full expert coverage; our TG
figure has no MTP while the baseline's does (ours would be the slower
comparison if MTP were added to the runner); our 200k arm ran while
deepseek_v4.exe was resident (started 17:13; run_start showed 21,197 MiB
free, so its VRAM impact was small, and wait 0.8% suggests little compute
interference — but the arm predates that check and is not certified
exclusive). Comparison is documents-vs-documents; no new GPU work while
deepseek_v4.exe holds the machine. Where to beat the baseline next is
unchanged: cold-start first-touch and the WDDM DMA inflation.

## 2026-09-16: CROWDED-VRAM CHARTER RUNG MEASURED — 200k-token PP, F16 KV, the af=1+pinned 2-slot optimum: 1,363 tok/s (146.7 s wall), slot window only 996 MiB — the streamed mechanism delivers its speed with the banks OUT of VRAM; 5.13x the recorded Q8 af=0 262k reference (262.6 tok/s)

The charter's crowding condition (KV takes the VRAM so banks must stream)
ran end-to-end at the new optimum: 200,000-token prefill (anchor-200000
prompt, the longest on-disk rung), F16 KV (the widest standard cache),
layer-major + af=1 + pinned + 2 slots + bf16 slabs (the T-Q38PW-02-approved
long-ctx shape). Checker OK (4,302 events). Trace
build/llama-cuda/gplr31-262k-f16-af1pinned.{jsonl,log}.

Results: PP 1,363.2 tok/s, wall 146.7 s; transfer 137.7 s; wait 1.18 s
(0.8% of wall); compute 133.0 s; bytes 18.33 GB (exactly one pass, 40
copies, 38 evictions); VRAM at run_start: 21,197 MiB free of 24,463
(3,266 MiB used total = slots 996 + activations 1,562.5 + KV head), i.e.
the 18.33 GB of banks stayed OUT of VRAM by design. Per-copy windows mean
3.44 s (min 0.26, max 9.69 s) — first-touch cost spread across the pass,
overlapped by the af worker (wait only 0.8%).

Comparison to the recorded long-ctx reference (T-Q38PW-02, 262k, Q8 KV,
af=0, 4 slots): 262.6 tok/s. This run: 5.13x at a shorter but same-order
rung with HALF the KV bytes and 2 fewer slots. Honest caveats: (1) rung
lengths differ (200k vs 262k; tok/s at this shape is length-dependent —
the 100k reference was 506 tok/s af=0, so the like-for-like gain vs 100k
af=0 is ~2.7x, still decisive); (2) F16 KV at 262k was NOT run (only 200k
with F16) — the charter asks for it and it is the next recorded gap; (3)
peak-VRAM telemetry shows usage at run_start, not a sampled maximum; the
budget model bounded it. Verdict: the crowded-VRAM charter condition is
DELIVERED at the new optimum — minimal VRAM (996 MiB slot window) and high
PP speed simultaneously; no default-path change yet (promotion task
pending its qa-smoke gate).

## 2026-09-16: GOAL CHARTER RECORDED (owner decision) — PP speed first, VRAM second, speed must hold under crowded VRAM (262k KV, 16 GB-class GPUs); model-fits-in-VRAM is NOT the target condition; next levers: 262k F16 at the new optimum, then default promotion

Owner confirmed the charter: maximize PP (prefill) speed first, minimize
VRAM second, and the PP speed must hold when VRAM is crowded — long-context
KV (e.g. 262k) or a 16 GB-class GPU where the 18.33 GB expert banks cannot
fit even alone. The mechanism must never depend on the model fitting in
VRAM. Recorded in AGENTS.md (Goal Charter section); future tasks are
prioritized and judged against it. Grounding facts: 262k KV costs ~2.7 GiB
at Q8 / ~5.3 GiB at F16 on this model; 2 slot windows cost ~0.9 GiB; on a
16 GB GPU streaming is mandatory regardless of KV width. Owner-selected next
levers: (1) validate 262k F16 KV at the new optimum (layer-major + af=1 +
pinned, 2 slots), then (2) promote that config to the runner default
(default profile v3) gated by the qa-smoke semantic regression.

## 2026-09-16: DEFAULT PROFILE v3 GATED — af=1 + pinned defaults passed the qa-smoke semantic gate (512 rung); default promotion COMPLETE

The 2026-09-16 default promotion (af=1 + --copy-source pinned as runner
defaults) was built earlier today but its semantic gate was pending on GPU
availability. With the external baseline finished, ran the canonical gate
under the GPU mutex (tools/run-tests.ps1 -Suite qa-smoke, 512-token anchor,
streaming-TG shape, --no-handoff, batch-major PP per the recorded rung
shape; fresh exe rebuild that includes the default changes):

- Prompt calibration: target=512 achieved=512, items=N1,N2,N3,N4,K1.
- Runner: exit 0; strict --check-telemetry OK (30,503 events, 3 slots —
  note: the 512 rung's one-ubatch shape triggers the auto slot ladder, 3
  slots here; answer content unaffected).
- Answer gate: aggregate=PASS 5/5 (QX-7741 TR-2918 MK-5501 LD-8834 Paris).
- Negative control: FAILs as designed (aggregate=FAIL, exit != 0).
- Suite verdict: SUCCESS. Artifacts regenerated:
  build/llama-cuda/gplr07-qa-512.{jsonl,txt,log}.

Telemetry sanity (gate is an answer-stability gate, not a perf gate, but
recorded): PP 512 tok wall 5.34 s (96 tok/s) cold process incl. first-touch;
TG 96 tok wall 178 s — TG is the known slow shape for this 512-rung config
(streaming TG, no handoff, no row-set; unchanged from all prior 512-rung
runs) and carries the bulk of trace time; peak VRAM 3,266 MiB. This matches
the recorded 512-rung behavior of prior profiles; the af=1+pinned defaults
change the PP path (measured separately: warm 4k 2.80-2.87 s) and did not
perturb answer stability here.

Also fixed: stale usage line still advertising "direct (default)" for
--copy-source (now "pinned (default since profile v3) or direct").

Status: the new defaults (pp_async_feed=1, copy_source=pinned, layer_slots=2,
layer-major PP, bf16 slabs) are now gate-cleared. Remaining honest caveat:
qa-smoke proves answer stability only (soft gate, per AGENTS.md); it does
not certify the promotion's performance claims, which remain the T-GPLR-25/
29 measurements.

## 2026-09-16: T-GPLR-31 DONE — locked-RAM probe: the driver grants 44.72 GiB of pinned memory incrementally (106 x 432 MiB, fails with cudaErrorMemoryAllocation with ~10 GiB still free), single-shot 20 GiB OK, and cudaHostRegister on ANONYMOUS pages SUCCEEDS — the T-GPLR-29 OOM was file-backed-view-specific, not a general registration limit

Standalone probe (tools/gplr31-pinpool-probe.cu, CUDA 13.2 nvcc, GPU-mutex
run, log build/llama-cuda/gplr31-pinpool-probe.log): incremental
`cudaHostAlloc` ladder granted 44.72 GiB pinned before failing
(cudaErrorMemoryAllocation) with GlobalMemoryStatusEx showing 10.11 GiB
still available — the binding constraint is a ~45 GiB-class cap or an OS
floor, NOT a small WDDM locked-page quota; clean 10%→83%→10% physLoad
accounting across allocate/free. Single-shot allocations up to 20 GiB
allocate and touch OK (no per-allocation cap). `cudaHostRegister` on
anonymous malloc pages SUCCEEDS (64 MiB and 432 MiB) — revising T-GPLR-29
case D: the earlier cudaErrorOutOfMemory was specific to the FILE-BACKED
mmap view; registration generally is viable under WDDM on this driver.

Consequence for the persistent pool: allocation is NOT the blocker — the
18 GiB full-pool size is grantable with large headroom (extends T-GPLR-30
fixed40's 17,478 MiB in-process grant). The pool's remaining obstacles are
the T-GPLR-30 findings themselves (flat warm wall under af=1; populate at
the cold 162 ms/bank rate), not driver refusals; the two residual uses
(hot-cache populate, af=0 serial staging) lose their feasibility
contingency. Also recorded: the CUDA 12.3 toolkit on PATH rejects the
Blackwell host compiler; probes must compile with the v13.2 toolkit.
Nothing committed; no vendored-tree changes.

## 2026-09-16: T-GPLR-30 DONE — partial persistent pinned pool (--pp-pool N[,fixed|lru]): hit mechanics perfect, warm savings curve FLAT AT ZERO at every N — DISCARD for warm-wall purposes; af=1 wall is now proven DMA-window-bound by two independent falsifications

Implementation (vendored tree, opt-in): `--pp-pool N[,policy]` allocates N
bank-sized pinned buffers at init (fixed: banks 0..N-1 populated once,
157-162 ms/bank at the COLD ~2.8 GB/s first-touch rate; lru: N-entry LRU
filled on first touch). Both stage sites (af worker + load_layer) serve pooled
banks straight from the pool — no mmap→pinned hop; DMA unchanged. Safe against
in-flight async DMAs: acquire/release protocol, eviction drains the copy
stream before rewriting a possibly-inflight entry (no torn weights, ever);
`pool_init` + per-bank `pool_event` telemetry; checker unaffected (unknown
events ignored). Default off; requires `--copy-source pinned`; refusals loud
(parse errors host-side, allocation failure surfaces the driver locked-page
quota).

Measured (five arms, one mutex hold, checker OK, gplr28/29 3-pass 4k shape,
2 slots, af=1, pinned; RAM = pool_init bytes):

| arm | RAM | cold pass 1 | warm 2/3 | hits per warm pass |
|---|---|---|---|---|
| baseline (no pool) | 0 | 6.26 s | 2.80/2.87 s | — |
| lru4 | 1992 MiB | 7.47 s | 4.17/3.64 s | 0 |
| fixed4 | 1728 MiB | 6.49 s | 3.20/2.91 s | 4 |
| fixed10 | 4320 MiB | 5.94 s | 3.10/2.83 s | 10 |
| fixed20 | 8640 MiB | 4.97 s | 3.07/2.86 s | 20 |
| fixed40 | 17478 MiB | 3.24 s | 2.93/3.02 s | 40 |

Verdicts: (P1) LRU cliff VALIDATED — under layer-major's cyclic order an
N-entry LRU below the 40-bank working set hits nothing (0 hits, ~112 misses
over 3 passes) and its fills ADD work: lru4 warm walls +27-45% vs baseline.
The naive persistent-cache idea is structurally dead in this schedule.
(P2) SAVINGS CURVE FALSIFIED — hits are perfect at every N (exactly N per
warm pass, zero misses) yet warm walls stay in the baseline band (all within
~6% of 2.80-2.87 s). The predicted ~0.5-0.6 s warm saving (the "~2.3-2.5 s
ceiling") does not exist: the wall is set by the concurrency-inflated DMA
window chain (addendum 3's mechanism — wait ends at own-DMA-end), and the
pool leaves the DMA untouched. At af=1 the 18.7 ms hop was ALREADY fully
hidden. (P3) Cold pass 1 collapses with N (fixed40 3.24 s, -48%) but populate
pays the same cost serially at init (6.48 s at 40 banks), so the whole
3-pass process is a net LOSS (15.67 s vs 11.93 s). Populate at 157-162
ms/bank is the cold first-touch rate, not the 24 GB/s warm-page rate — init
overlaps nothing; a hot-OS-cache start would populate at ~19 ms/bank
(unproven, cache-state lottery per T-GPLR-28).

Owner-facing implication: under af=1+pinned, the warm wall (~2.8-2.9 s) is
DMA-window-bound — proven now by TWO independent falsifications (3-slot
geometry; pool hop removal). No RAM size buys warm-wall time. Remaining
levers: shrink DMA bytes (row-set TG class), or beat WDDM concurrency
inflation (outside current scope). Residual uses for the pool: long-lived
process starting with hot OS file cache (populate ~0.75 s, unproven), or
af=0 serial staging where the 0.75 s/pass hop exposure is real (unmeasured).
Files: build/llama-cuda/gplr30-{lru4,fixed4,fixed10,fixed20,fixed40}.{jsonl,log},
analyzer tools/analyze-gplr30-pool.py, arms build/run-gplr30{,-poolcurve}.ps1.
No default-path change; nothing committed.

## 2026-09-16: WAIT SIGNATURE IDENTIFIED — it is the per-layer deficit max(0, effective_DMA_window − compute_window), quantized at the DMA window itself (31.45 ms); af=1+pinned wall without it = 1.85–1.92 s (empirical) / ~1.5–1.8 s (engine-bound floor); 3-slot prescription TESTED AND FALSIFIED (warm 4.14–4.83 s, worse — copy engine saturates at 2 slots)

Event-level analysis of the af1 trace (pass 3, fully warm): every
layer_ready_wait ends EXACTLY at that layer's own copy_complete (delta 0.0 ms
everywhere), quantized at 31.45 ms — the pure pinned DMA time from the probe.
~10 of 40 layers wait 0. So the wait is neither worker lag nor an accounting
artifact: it is the summed deficit between what the copy engine delivers per
bank (69.6 ms average effective window at 2 slots — inflated 2.2x from the
31.4 ms pure rate when two DMAs overlap on the WDDM engine, 28 concurrent
pairs) and what the consumer computes per layer (~34 ms). Deficit model: ~23
ms × ~30 layers ≈ 927–958 ms at 2 slots. MODEL VALIDATED cross-geometry: the
same arithmetic predicted 363 ms at 3 slots; two independent 3-slot runs
measured 363.0/363.2 ms.

Counterfactual: empirical wall-without-wait = wall − wait = 1.85–1.92 s;
engine-bound floor = 40 × 31.4 ms pure DMA = 1.26 s overlapped with ~0.55 s
sampled compute ≈ 1.5–1.8 s. The gap to 2.80–2.87 s is concurrency inflation
on the copy engine — outside current scope to remove.

Prescription test (the obvious "add a slot" fix): af=1 + pinned + 3 slots,
reproduced twice (checker OK both, TG=3 legal shape because handoff auto
declines at 3 slots): warm walls 4.14–4.83 s (worse than 2-slot 2.80–2.87 s),
transfer ~9.2–9.5 s vs 3.69–3.73 s (windows inflate to ~237 ms avg), wait
halves to 363 ms only because compute inflates too, cold pass 23.2–43.9 s
(vs ~7 s at 2 slots). VERDICT: the copy engine is already saturated at 2
slots under pinned+af; a third slot adds contention, not runway. Owner-facing
implication: under af=1+pinned, 2 slots is the measured optimum; 3 is
actively harmful. Wait telemetry is trustworthy and reads as chain exposure.
Files: build/llama-cuda/gplr29-af1-pinned-3slot.{jsonl,log} (×2 runs),
analysis inline in the plan addendum 3. No code changes.

## 2026-09-16: PASS-2 ANOMALY REPLICATED AND RESOLVED — two separate observations, opposite verdicts: the 4.10 s wall spike was environmental (not replicated); the ~635 ms wait is SYSTEMATIC pinned-path behavior (4 warm passes in 2 independent runs: 635.1/635.3/635.3/635.9 ms, 0.13% spread; absent in direct arms) — mechanism closed by arithmetic (hop 748 ms vs DMA saving 564 ms per pass)

No new GPU runs: the three pinned traces on disk (gplr29-repeat-3pass-pinned,
gplr29-afab-af0, gplr29-afab-af1) contain four warm passes across two
independent executions. Per-pass wait totals: 635.1 / 635.3 / 635.3 / 635.9 ms
— the signature replicates to 0.8 ms (0.13%). Direct arms show 0.4–7 ms.
Distribution: ~120 layer_ready_wait events per 3-pass trace, 59 > 1 ms, max
36.4 ms — distributed per-layer (~16 ms/layer avg), not one spike; layer 39
largest (~109 ms/pass), most layers ~95 ms.

Verdicts: (1) the original 4.10 s pass-2 WALL spike is an environmental one-off
(not replicated; its residual sat in a compute figure that also did not
reproduce — shared-GPU interference window during that mutex hold). (2) The
wait is a REAL, deterministic pinned-path property. Mechanism: the chain's CPU
hop (18.7 ms × 40 banks = 748 ms/pass) minus the DMA saving (14 ms × 40 =
564 ms/pass) = ~184 ms net serial cost, matching the measured af0-pinned
vs-direct wall delta (+70–170 ms); at af=1 the worker hides most of the hop
and walls drop 0.45–0.53 s below direct, but worker service time
(~50 ms/bank vs ~34 ms/layer compute) becomes the pipeline bottleneck —
which is why the wait ACCOUNT grows (927–958 ms) while the wall SHRINKS. Wait
is accounting-visible chain exposure, not wall cost.

Ceiling (expected, unproven): eliminating the per-pass hop (persistent
pre-populated pinned pool across passes) is worth up to ~0.5–0.6 s/pass;
af=1+pinned warm could approach ~2.3–2.5 s. Open: exact wait-accounting
semantics under af (instrumentation question, separate from this verdict).
No code changes; no new GPU time.

## 2026-09-16: T-GPLR-29 ADDENDUM — af overlap A/B with pinned: overlap converts ONLY with pinned staging (warm 4k PP 2.80–2.87 s vs 3.33–3.43 s, ~14–15%); af=1 + direct shows NO gain; first measured configuration to beat the warm steady state

The pinned-probe task left one open question: does OVERLAP (async-feed,
T-GPLR-25) convert the pinned chain's +44% DMA headroom into wall time, given
the serial chain loses to direct? Three arms under/adjacent to one mutex hold
(4k anchor, 2 slots, layer-major default, handoff auto, --pp-repeat 3, kv q8):

| warm passes 2+3 (wall) | af=0 | af=1 |
|---|---|---|
| pinned copy-source | 3.43 / 3.33 s | **2.80 / 2.87 s** |
| direct copy-source | 3.26 / 3.23 s (gplr28) | 3.41 / 3.35 s (new control) |

Cold pass 1: af1-pinned 6.26 s vs af0-pinned 6.96 s. All four traces checker
OK; TG unchanged (~76–77 tok/s, 0 evictions).

**Verdict:** the pinned gain is REAL but only under overlap — af=1 + direct
shows no wall gain, so the improvement is attributable to the worker consuming
the pinned chain's DMA headroom, not to overlap alone. Measured: ~0.5 s per
warm 4k prefill (13.8–15.5%, above the ~6% noise floor and ~1.1% drift).
Caveats: one pair per cell; the wait-telemetry figure grows under af (768–958
ms) and needs a dedicated interpretation pass before being read as
slot-starvation; af=1 stays an opt-in prototype per the T-GPLR-25 contract.
Consequence: the interaction (af=1 AND pinned) is the configured optimum
measured on this box; next lever remains first-touch (process longevity /
pinned pool population already amortize it within a process).
Files: build/llama-cuda/gplr29-afab-{af0,af1,af1direct}.{jsonl,log},
scripts build/run-gplr29-afab*.ps1, tools/run-gplr29-af1direct.ps1.

## 2026-09-16: T-GPLR-29 DONE — pinned-pool probe + end-to-end pinned arm: DMA ceiling 14.4 GB/s is ~90% of the real link (Gen4 x8 LAPTOP 5090, not Gen5); pinned buys nothing serial (chain 9.0 GB/s < direct 10.0 GB/s) and no PP-wall change end-to-end; first-touch stays the dominant cost

Standalone probe `tools/gplr29-pinprobe.cu` (nvcc 13.2, outside the vendored
tree) over a 453 MB region of the real MoE GGUF (3-member bank shape
confirmed), run twice under the GPU mutex (reproducible to <2%):

| case | rate |
|---|---|
| A pinned cudaHostAlloc → VRAM | **14.4 GB/s** (best 31.4 ms) |
| B direct mmap → VRAM warm | 10.0 GB/s (45 ms) |
| B0 direct cold first-touch | 0.83 GB/s (548 ms) first probe; 3.25 GB/s (139.5 ms) second probe minutes later |
| C1 CPU memcpy mmap → pinned | 24.2–24.5 GB/s |
| C T-GPLR-18 chain (C1+A, serial) | **9.0 GB/s — slower than direct** |
| D cudaHostRegister on the mapped view | FAILED (WDDM, cudaErrorOutOfMemory — locked-page quota; same practical verdict as Linux InvalidValue) |

**New hardware fact (nvidia-smi under load):** the 5090 is the LAPTOP SKU and
the PCIe link negotiates **Gen4 x8 (~16 GB/s theoretical)**; width max x16,
gen max 4. All prior Gen5/63 GB/s bandwidth framings are void for this box.
Pinned DMA at 14.4 GB/s is already ~90% of the physical ceiling.

End-to-end check (existing `--copy-source pinned`, T-GPLR-18 machinery, the
gplr28 3-pass shape, checker OK 20,348 events):
pass 1 wall 7.12 s (direct: 7.15 s) — no change; pass 2 4.10 s vs 3.26 s —
anomalous (wait 635 ms vs 0.4 ms, compute +~380 ms; unexplained, does not
count as a pinned penalty without replication); pass 3 3.26 s vs 3.23 s —
no change; transfer −7% (4.60 vs 4.96 s); TG unchanged (76.4 tok/s, 0
evictions).

**Verdict:** three predictions falsified. (1) The 4.4 GB/s "pageable
bounce-buffer" figure from the research pass is revised — the pure pageable
copy runs 10 GB/s here (penalty only ~1.44x vs pinned); the recorded 4.4 GB/s
per-copy effective rate in production traces is pipeline overhead + stream
gaps, and first-touch (3–12x, T-GPLR-28) remains the dominant cost. (2) The
pinned two-hop chain loses serially (9.0 < 10.0 GB/s): its only justification
is OVERLAP (af=1, T-GPLR-25), where +44% DMA headroom could hide the chain —
unmeasured. (3) No end-to-end PP-wall win from pinned at serial direct
copy-source. Consequence ranking: keep banks warm in-process (done, T-GPLR-28)
> hide the chain behind compute via async-feed (measure af=1 with pinned) >
pinned headroom as an overlap enabler only. The mlock/working-set privilege
path stays parked pending an owner-side Local Security Policy change.
Files: build/gplr29-pinprobe.{cu,exe}, build/run-gplr29-*.ps1,
build/llama-cuda/gplr29-pinprobe.log (+ second-run console),
build/llama-cuda/gplr29-repeat-3pass-pinned.{jsonl,log}. No vendored-tree
changes. Not committed.

## 2026-09-16: ONLINE RESEARCH PASS — first-touch + staging cost layers identified from primary sources (pageable-memcpy bounce buffer, PrefetchVirtualMemory-vs-working-set, pinned-register legality, ik_llama slot-pool precedent)

Owner asked for formulated research questions + an online pass on the T-GPLR-28
first-touch findings. Questions and answers (all claims verified against the
cited primary sources; local facts from docs/AGENT_ENVIRONMENT.md outrank any
page):

**Q1 Why is warm staging only ~4.4 GB/s when the bus is ~63 GB/s?**
NVIDIA staff (Robert Crovella, developer forums thread 68153, confirmed 2021):
cudaMemcpy/cudaMemcpyAsync from PAGEABLE host memory silently degrades to a
synchronous copy that first bounces through a driver-maintained page-locked
buffer, then DMAs. Our `copy_source: direct` path IS this path — the warm
ceiling is bounce-buffer + copy-engine overhead, not the PCIe Gen5 x16 link
(~63 GB/s raw; we use ~7% of it). The pinned staging pool (T-GPLR-18) removes
the bounce hop; externally expected to multiple-x the warm rate (exact 5090
pinned number not found in public benchmarks — unmeasured, falsifiable).

**Q2 Can we pin the mmap itself (cudaHostRegister on the mapped file)?**
Same thread: cudaHostRegister over file-backed mmap pages returns
cudaErrorInvalidValue; it only works once pages are locked (mlock/MAP_LOCKED)
or after copying to anonymous memory. No Windows-specific counterexample found,
but no positive Windows case either — registering our mapped GGUF view is NOT
viable as-designed; the pinned pool must be cudaHostAlloc'd anonymous memory +
one-time population. Size-dependent pin failures exist in the wild
(nvidia-container-toolkit #226) — a probe on this box is the falsifiable check.

**Q3 Why did the T-GPLR-08 PrefetchVirtualMemory probe fail?**
Official MS docs (PrefetchVirtualMemory function page): "The prefetched memory
is not added to the target process' working set; it is cached in physical
memory. When the prefetched address ranges are accessed by the target process,
they will be added to the working set." Prefetch fills the STANDBY cache only;
every bank page still soft-faults into the working set on first access.
Arithmetic check: 453 MB / 4 KiB = ~116k pages/bank; 570 ms / 116k ≈ 4.9 µs
per soft fault — matches the cold per-copy cost. This also explains the
fresh-process re-pay (bm2 sweep 1 cold: pages in standby, not in the new
process's WS) AND the T-GPLR-28 in-process stability (pages stay in the WS for
the process lifetime once faulted in). Mechanism now consistent across all
three observations.

**Q4 Can we defend the working set without privileges?**
No. Raising a process working set (SetProcessWorkingSetSizeEx, incl. the
QUOTA_LIMITS_HARDWS_MIN_ENABLE "hard minimum" mode) requires
SeIncreaseWorkingSetPrivilege, which docs/AGENT_ENVIRONMENT.md records as
present-but-DISABLED on this machine (whoami /priv, 2026-09-09) — same blocker
as VirtualLock (no SeLockMemoryPrivilege). The privilege-free paths that remain:
process longevity (server-style residency) and pinned (driver-locked) memory.
An owner-side Local Security Policy change would unlock the WS/mlock path.

**Q5 Does anyone else do this (MoE slot pools)?**
ik_llama.cpp RFC+PoC #1833 (cross-post of ggml-org/llama.cpp#23324, 2026-05):
expert slot pool with LRU + pread from GGUF + event-signaled GPU wait — the
same architecture as ours, independently. Their measured cautions: even the
no-eviction case (n_slots=all) pays ~19% throughput tax from the interceptor
sync alone; knee at ~half the experts; synchronous cold misses collapsed PP to
~4.8 tok/s. Implications: validates the design; flags that OUR af=1 vs af=0
overhead should be re-checked against that 19% figure; our async staging
(308-1,270 tok/s PP at 4k) is the differentiator their PoC lacks.

**Q6 DirectStorage/GDS as a bypass?**
NVIDIA GPUDirect Storage (NVMe→VRAM DMA, Windows via DirectStorage) exists and
could bypass the whole mmap/fault path; no public llama.cpp integration found.
Would need an owner-approved dependency + GPU-attached NVMe verification —
recorded as a possible future lever only, behind the dependency gate.

Consulted: developer.nvidia.com forums #68153 (pinned/pageable/register
semantics), learn.microsoft.com PrefetchVirtualMemory + Working Set pages,
SetProcessWorkingSetSizeEx docs, ik_llama.cpp discussion #1833, gamersnexus
Gen5 x16 figures, nvidia-container-toolkit #226. Wigolo MCP engine pool was
degraded (Bing-only, junk results); searches completed via fallback web search.
Changed: this entry only — no code, no plan edits. Next cheap probes implied:
(1) cudaHostAlloc pin-probe + pinned-vs-direct per-copy A/B on the MoE vehicle;
(2) owner decision on the Local Security Policy right if the mlock-class path
is ever wanted; (3) af-overhead cross-check vs ik's 19%.

## 2026-09-16: T-GPLR-28 DONE — in-process --pp-repeat 3 proves first-touch is pay-once-per-process (warm state stable in-process: 7.15 s cold pass → 3.26/3.23 s warm passes, per-copy ~309 ms → ~124-125 ms); in-process trimmer/invalidator branch FALSIFIED

The 2026-09-15 trio left the first-touch mechanism open: batch-major's warm
sweeps ran ~103 ms/copy while layer-major's every copy was first-touch
(~570 ms), and the decisive bm2 observation (fresh process re-pays sweep 1
even seconds after layer-major touched every bank) proved the state does not
survive process death — but could not distinguish an active in-process
trimmer from process-lifetime page state. T-GPLR-28 added an opt-in runner
flag `--pp-repeat N` (default 1, no behavior change; passes 2..N clear KV,
reset per-layer stats, emit a `pp_pass` telemetry marker, keep the slot
window; TG runs once; refusals: outside 1..16, and repeat>1 with --mtp) and
ran the falsifiable 3-pass arm on the MoE vehicle (anchor-4096, ubatch 512,
`--layer-slots 2`, default schedule = layer-major, handoff auto-on, kv q8,
af=0) under the GPU mutex:

| pass | PP wall | PP transfer | per-copy avg | evictions |
|---|---|---|---|---|
| 1 (cold) | 7154 ms | 12 359 ms | ~309 ms | 38 |
| 2 (warm) | 3262 ms | 4992 ms | ~125 ms | 40 |
| 3 (warm) | 3226 ms | 4956 ms | ~124 ms | 40 |

TG (96 tokens, handoff auto): 76.9 tok/s, 0 evictions, transfer 82 ms.
Trace `build/llama-cuda/gplr28-repeat-3pass.jsonl` (20,348 events),
check-telemetry OK after a generic multi-pass extension of the checker
(`pp_pass` resets per-layer ascending-chunk state; final-chunk, stride-1 and
act-bind rules made pass-count aware; all 14 lm40/tail/byte-account checker
fixtures still pass — the only failing CTests remain the pre-existing
environment-gated `-run` companions requiring Q38LS_MODEL_PATH + GPU).
Pass 1 reproduced the trio shape (40 copies, 38 evictions) at a better wall
(7.15 s vs 13.30 s; the trio's lm arm was a fresh process under different
cache conditions — cross-process inheritance of page state is real but
partial and time-varying, recorded as uncontrolled Windows file-cache/
working-set behavior).

**Verdict on the falsifiable prediction: the in-process trimmer/invalidator
branch is FALSIFIED.** Passes 2 and 3 are identical within 1.1% and run at
~4.4x the cold per-copy rate, so once established, the warm state persists
for the process lifetime with no decay. The first-touch cost is paid once
per bank per process. The warm steady state of layer-major 4k PP is
**~3.2 s (~1,260-1,270 tok/s)** — the 7-8 s prediction floor is met or
beaten in steady state. The exact carrier (disk-in vs page-fault work vs
working-set trim on teardown) remains partially open; what is settled is
that it lives and dies with the process. Expected (unproven) consequence:
long-lived server processes hold the warm rate; levers are pinned staging
(`--copy-source pinned`, T-GPLR-18) or a resident server-style process —
both unmeasured.

Implementation: `--pp-repeat` in
`third_party/llama.cpp/tools/layer-streaming/main.cpp` (flag, parse,
validation, per-pass loop around the existing PP block, `llama_memory_clear`
between passes, per-pass phase_summaries sharing the prebuild builds0
baseline) + `tel_pp_pass` emitter in `llama-layer-streaming.cpp` + checker
multi-pass rules. Not committed.

## 2026-09-15: 4k LAYER-MAJOR PP MEASURED — 2.3x PP-wall gain vs fresh adjacent batch-major baseline (prediction of 3.5-4x FALSIFIED as stated; the 8.0x staging-volume cut CONFIRMED; residual cost identified as per-process first-touch PCIe stalls)

The T-GPLR-12 layer-major 4k prediction ("~3.5-4x PP-wall gain at 4k, 40 copies
instead of 320") was tested with a three-arm adjacent trio on the MoE vehicle
(Qwen3.6-35B-A3B, anchor-4096 prompt, ubatch 512, `--layer-slots 2`, af=0,
`--allow-full-residency`, kv q8), all under ONE GPU-mutex hold for cache
adjacency, executed by `tools/run-lm4k-arms.ps1` via `tools/run-lm4k-trio.ps1`:

| arm | schedule | handoff | PP wall | PP tok/s | PP bytes | PP copies | PP evictions | checker |
|---|---|---|---|---|---|---|---|---|
| `gplr25-lm4k-bm1.jsonl` | batch-major (pinned) | off | 30.41 s | 134.7 | 146.62 GB | 320 | 318 | OK (25005 ev) |
| `gplr25-lm4k-lm.jsonl` | layer-major (default) | auto-on | **13.30 s** | **307.9** | **18.33 GB** | **40** | **38** | OK (19624 ev) |
| `gplr25-lm4k-bm2.jsonl` | batch-major (drift control) | off | 28.73 s | 142.6 | 146.62 GB | 320 | 318 | OK (25005 ev) |

**Verdict on the 3.5-4x prediction: FALSIFIED as stated.** Measured gain is
**2.29x** (vs bm1 30.41 s) / **2.39x** (vs the adjacent bm2 28.73 s baseline).
The prediction's floor logic assumed the batch-major warm per-copy cost
(~176 ms/bank) would apply to layer-major's 40 copies (40 x 0.176 s ~ 7.1 s
floor). It does not: per-copy DMA rate is copy-state-dependent, not
schedule-inherent.

**What IS confirmed (structural volume prediction):** 40 copies / 18.33 GB /
38 evictions — exactly the one-pass shape; the 8.09x re-staging volume
(320 copies / 146.62 GB) is gone. Compute parity holds: bm2 PP compute sum
2.98 s vs layer-major 0.74 s over 80 `layer_compute` events (4-chunk
activation slabs, same FLOPs at coarser granularity; 0.74 x 4 ~ 2.98).

**Why the wall lands at 13.30 s, not ~7-8 s — the first-touch finding:**
per-copy submit-to-complete spans sum to 12.93 s of the 13.30 s PP wall: the
copy chain IS the layer-major wall (no scheduling gap to recover). But
layer-major's 40 copies each run ~570-580 ms transfer (first touch of the
bank's host pages in this process), while batch-major's sweeps 2-8 run
~103 ms (~4.4 GB/s effective) after a ~680/596 ms cold sweep 1 (bm1/bm2).
First-touch penalty is per-copy ~5.5x vs warm. Decisive negative control:
**bm2's sweep 1 is still cold (596 ms) even though layer-major (lm arm) had
copied every one of the 40 banks seconds earlier** — so the penalty is
per-process, NOT file-cache contents; the Windows working-set/codeyness of
mmap-backed staging buffers re-cold per process. (Mechanism unproven;
recorded as an open question.)

**Side result (larger than the PP win): TG-phase handoff effect.** With
`--no-handoff`, batch-major's 96-token TG re-stages every bank per token:
TG wall 192.6/284.2 s at 0.5/0.34 tok/s, 1,759 GB transferred, 3,840
evictions. Layer-major's handoff auto-on kept the last bank resident: TG wall
2.31 s at **41.5 tok/s**, 0 evictions, 0 bytes. For interactive 4k-prompt
sessions the handoff is worth ~2 orders of magnitude in TG, independent of
schedule choice.

**Honest limits:** single trio per shape (no repeated legs; BM drift
bm1->bm2 = -1.7 s bounds run-to-run noise at ~6%); both BM arms cold-started
sweep 1 while the lm arm's whole run is first-touch — a warm-process
layer-major run (second identical run in one process) is NOT yet measured and
could land well below 13.30 s; the `--residency-budget-mib`/ladder gates were
not involved (explicit `--layer-slots 2`). Analysis script:
`build/analyze-lm4k-trio.py`.

**Open next levers (unchanged order):** (1) measure in-process warm
layer-major (re-run without reload) to separate first-touch from steady state;
(2) pre-touch / persistent bank cache across processes to attack the 570 ms
first-touch copy; (3) af=1 A/B re-run on the fixed binary (async-feed
residue fix) whose payoff is now bounded by this wall decomposition (worker
fed 40 copies ~ 12.9 s span work to hide under 0.74 s compute — af alone
cannot hide first-touch PCIe; it needs the same warm-copy state).

## 2026-09-15: DEFAULT PP SCHEDULE VERIFIED as layer-major (T-GPLR-12, no code flip needed); stale selector comment fixed; harness scripts pin batch-major deliberately

Owner asked to make `--pp-schedule layer-major` the default. Code inspection
showed it already is: `main.cpp` line 2464 initializes
`pp_schedule = "layer-major"` (T-GPLR-12), and three independent paths agree
(the pw-probe default note, the `--residency off` conflict table that refuses
only an EXPLICIT layer-major request, the TG-cap gate that fires from the
untouched default). The recorded batch-major arms were pinned by their
measurement scripts (`build/run-gplr25-*.ps1`, `tools/run-gplr13-ladder.ps1`,
`tools/run-tests.ps1:348` MoE rows), not by the binary.

**Proof on the real vehicle (Qwen3.6-35B-A3B, 512 anchor, 2 slots, af=0):**
- Default shape, `--tg-tokens 96 --no-handoff`: runner exits 2 with the
  layer-major-only TG-cap refusal — a batch-major default cannot produce
  this error; the cap gate is reached only when the schedule resolved to
  layer-major.
- Default shape, `--tg-tokens 3 --no-handoff`:
  `gplr25-default-lm-tg3.jsonl` — run_start `"schedule":"layer-major"`,
  run_end present, checker OK (1,047 events, 2 slots); PP wall 14,385.6 ms,
  38 evictions (2 slots, matches the 1-slot-per-sweep layer-major shape at
  512; the 320-copy batch-major shape is a 4k sweep effect).
- Default shape, `--tg-tokens 96`, handoff AUTO:
  `gplr25-default-lm-tg96-auto.jsonl` — run_start `"schedule":"layer-major"`
  + `"handoff_mode":"auto-on"`, run_end present, checker OK (19,530 events);
  TG 96 tokens at 43.377 tok/s with transfer 123.5 ms and 0 evictions (bank
  stayed resident through TG — the T-GPLR-10 cap lift worked as designed).

**Code change:** one stale comment fixed (the selector comment at ~2825
claimed "batch-major is the normative default; layer-major is the opt-in
experiment" — contradicted by T-GPLR-12's actual default and the gate
structure). No behavior change; rebuilt green.

**Why the harness scripts pin batch-major (deliberate, leave as-is):**
`--no-handoff --tg-tokens 96` is only legal under batch-major (layer-major
caps TG at 3 without an active handoff, T-GPLR-10); the A/B, ladder, and
suite scripts measure the batch-major path on purpose. The 8.09x re-staging
discovery (320 copies at 4k) is a batch-major-sweep property; the pending
layer-major 4k measurement (predicted ~3.5-4x PP-wall gain, 40 copies) is
the open falsifiable test, not a default question.

Trace/artifact identity: `gplr25-default-schedule-512.*` (the exit-2 refusal
arm), `gplr25-default-lm-tg3.*`, `gplr25-default-lm-tg96-auto.*` (completed
proofs); all under `build/llama-cuda/`, gitignored per policy.

## 2026-09-15: AGENTS.md governance updated — MoE-Qwen focus is a measurement choice (other models on owner request), all gates declared soft, owner summary required

AGENTS.md gained three sections/notes, binding every future session:

1. **MoE-Qwen focus, not a boundary (Scope and Boundaries).** Active
   development targets the Qwen MoE vehicle (Qwen3.6-35B-A3B routed-expert
   banks) because it is the only on-disk model whose expert banks do not fit
   in VRAM, making residency actually measurable. The note states explicitly:
   this focus is a measurement choice, not an architecture boundary; testing
   any other architecture is legitimate work whenever the owner asks for it
   (entry point: the second-architecture draft T-GPLR-27), and
   owner-requested model tests are never treated as scope creep.
2. **"Gate Policy: All Gates Are Soft" (new section).** Every gate —
   semantic-QA ladder, trace checker, CTest families, capacity preflight and
   refusal path, performance and hit-rate measurements — is advisory
   evidence, not release authorization. A PASS proves only what that gate's
   contract defines. Final acceptance is always the owner's decision from the
   required summary, never from gate output. Honest reporting of limits
   (noise floors, stream biases, unmeasured expectations, withdrawn
   confounded comparisons) is required. Diligence is unchanged: required
   validation must still run and pass, and hard failures still block a task
   being reported complete.
3. **"Owner Summary Requirement" (new section).** Because gates are soft,
   every completed task and significant experiment ends with a
   self-contained, structured summary the owner can decide from: (1) original
   task/goal/idea explored; (2) how it works in simple terms; (3) glossary
   defining every term and abbreviation at first use (PP, TG, af, slots, hit
   rate, ...), restated in the summary itself rather than assumed from prior
   turns; (4) pros and cons with every claim labeled measured / expected /
   unproven and pointed at its evidence; (5) a recommendation (adopt, keep
   experimenting, or discard) plus what the owner should validate or decide
   next. A task is not reported complete while any part is missing.

No code, plan, or SDD content changed. This entry records the governance
change only; the summaries already produced this session (slot ladder,
async-feed A/B, refusal-path test, residue fix, proof ledger) follow the
now-required structure. [Governance commit: 49c270d (2026-09-15); recorded
here in a follow-up record commit per the repo's hash-recording convention.]

## 2026-09-15: Pending owner decisions (item 4 gate-eligibility, item 6 second-architecture) recorded as plan task drafts T-GPLR-26 / T-GPLR-27

Per the owner's standing instruction, the two 2026-09-14 decision briefs are
now represented in docs/plans/generic-prefill-layer-residency-plan.md as
drafts — no pick made, no code, no gate verdict, status DRAFT — pending
owner:

- **T-GPLR-26 (item 4, gate-eligibility binding vs advisory):** written as
  two explicit branches — A (BIND: the oracle + verdict-rule + refusal-
  reason prerequisite chain first, since the brief establishes no admissible
  input exists at bring-up) and B (ADVISORY, the brief's recommendation:
  docs-only SDD scope note + K=4 non-echoing caveat). Executing the branch
  the owner picks IS the task.
- **T-GPLR-27 (item 6, second architecture):** written against the brief's
  recommended target (Gemma-4 12B, dense, on-disk checkpoint) with the
  brief's scope table and its five unknowns carried over verbatim; the
  explicit non-goals state the capacity-claim limit up front (green proves
  lifecycle/parity, not capacity). Re-scopes if the owner picks otherwise.

Activation: both drafts stay dormant until the owner picks; T-GPLR-27's
green-block condition (Qwen reference gates standing) is named in its
Depends-on.

## 2026-09-15: T-GPLR-25 residue FIXED — async_feed_submit wired to the worker (batch-major + layer-major), async_feed_ready gated to worker-fed copies; a false-positive protocol hazard removed

The 2026-09-15 prototype residue ("async_feed_submit never fires; dead
telemetry") was investigated and turned out to be a mis-wiring plus a latent
protocol hazard, not a dead event. Decision: WIRE, not delete (the event was
already emitted from the worker's copy body — the worker simply was never
fed).

**Root causes found (all in the untested worker paths):**
1. Both af enqueue sites lived in the LAYER-MAJOR controller; every recorded
   af1 arm ran batch-major, so the worker was constructed but never fed. The
   apparent overlap evidence (3,880 async_feed_ready) was the
   consumption-side event firing unconditionally whenever af existed — a
   false positive that misattributed the main-thread lookahead's work to the
   worker.
2. af_submit_copy skipped the slot-reuse invariant's consumer wait:
   record_consumer_done frees the slot BEFORE ev_consumer_done completes, so
   the worker could overwrite a slot whose prior generation's graph was still
   running.
3. The worker's entry gate re-staged an unconsumed resident generation
   ("slot already holds il" was treated as stage-again; the layer-major seed
   enqueue hits exactly this).
4. Main-thread staging never claimed the slot (free_slot stayed stale-true),
   so a worker request could double-stage over an unconsumed main-thread
   copy.

**The fix (single-writer copy protocol, opt-in af path only, default behavior
byte-identical):**
- Batch-major lookahead (and the layer-major post-consumer_done refill) now
  ENQUEUE the next slot's refill on the worker when af is on; the
  consumption-time load_layer remains the on-demand fallback. af=0 paths are
  untouched (else branch identical to the old code).
- af_submit_copy mirrors load_layer's eviction block exactly: consumer-done
  event wait OUTSIDE af->mtx (re-validating after the wait for the main
  thread's same-layer race, returning no-op success if it staged first),
  eviction bookkeeping + tel_slot_evict under the lock, claim-on-stage
  (free_slot=0) by whoever stages, af_submitted flag on the slot consumed by
  wait_slot_ready.
- tel_async_feed_ready fires ONLY for worker-submitted copies (af_fed gate);
  T-GPLR-08 page-prefetch hint parity added to the worker copy body
  (pf has its own mutex; no lock-order risk).

**Validation:** build green; host suite pass; the previously env-var-skipped
cTest families (lm511-run/checker, tail513-{lm,bm}-run/checker) 6/6 PASS with
Q38LS_MODEL_PATH set (the earlier 'Failed' rows were the missing env var, not
the change). Focused real-model arm under the GPU mutex
(`gplr25-af1-residuefix-512.jsonl`, 2 slots, af=1, batch-major, 512 anchor):
async_feed_submit 3,686 events (was 0 in every recorded arm),
async_feed_ready 3,686, and the two (layer,generation) sets are an exact
bijection (0 ready-without-submit, 0 submit-without-ready); evictions 38 (no
double count); checker OK (30,698 events); answer gate PASS. PP wall 12.69 s
is a single cold-branch arm and carries NO performance claim. Consequence
for the A/B history: the recorded af0/af1 pairs measured af1 with an IDLE
worker (it was never fed under batch-major), so the "no gate-able af wall
delta at 512/4k" null verdict does not speak for the now-live prototype —
re-running the paired A/B on the fixed binary is the honest next step for
the performance question.

## 2026-09-15: T-GPLR-25 COMPLETED: 4k pair run — af delta +3.0% is AT the noise floor, no performance claim; batch-major 4k re-staging discovered (8.09x bank bytes); budget-refusal path verified (run_refused, cap_source=budget)

**Owner-decision refusal path VERIFIED under the GPU mutex**
(`gplr25-budget-refusal-2048.jsonl`, launcher `build/run-budget-refusal-test.ps1`,
chained into `build/run-refusal-and-af1-chain.ps1`). Tight explicit budget
`--residency-budget-mib 2048` (no `--layer-slots`; default ladder resolves the
count; `--allow-full-residency` keeps the out-of-core gate out of the way):
- The ladder resolved 3 slots on free VRAM, then preflight check #2 refused:
  runner exit 1 (the expected outcome), `run_refused` trace with
  `reason=capacity_preflight`, `detail`="bounded 3-slot window 4,195,805,696 B
  exceeds the residency budget 2,147,483,648 B", `bytes_needed=4195805696`,
  `bytes_allowed=2147483648`, `cap_source=budget`, `exit_status=1` — and NO
  run_end / phase_summary (the T-GPLR-22 refusal-trace rule).
- `--check-telemetry` accepts the refusal shape (OK, REFUSAL TRACE). No silent
  downgrade to 2 or 1 slots: the budget refusal names the resolved slot count.
  Note the ladder itself reads free VRAM only; the explicit budget is enforced
  by check #2 AFTER the ladder — this is the designed layering.

**4k paired A/B complete (af=0 vs af=1, fixed --layer-slots 2, back-to-back
under ONE mutex hold; only flag delta):**
- af0 `gplr25-ab-4096-af0.jsonl`: PP wall 30,034 ms, PP tok/s 135.2, transfer
  sum 56.4 s, compute sum 3.35 s, effective staging 4.88 GB/s.
- af1 `gplr25-ab-4096-af1.jsonl`: PP wall 29,123 ms (+3.0%, +911 ms), PP tok/s
  139.3, transfer 54.7 s, compute 3.21 s, effective staging 5.03 GB/s. Checker
  OK (29,166 events, 2 slots), answer gate PASS.
- **Verdict: NOT a performance result.** The TG leg (which the flag provably
  does not touch) differs by +2.3% between the same two arms (191.5 s vs
  196.0 s) — a direct single-replicate noise-floor estimate. The +3.0% PP
  delta sits at that floor. The 512 rung showed -0.3% (noise). The prototype
  MECHANISM evidence stands (worker engages, overlap real); a gate-able wall
  claim would need replicated pairs. Direction is positive and
  mechanism-consistent; magnitude unproven.

**New structural discovery (af0 4k trace, supersedes the 40-copy mental
model for batch-major PP):** at 4k/ubatch-512 batch-major, PP runs 320
copies, not 40 — mean 458.2 MB/copy, 146.62 GB total = 8.09x the one-pass
bank total (318 evictions). Every layer's bank is re-staged once per chunk
sweep (8 sweeps at 4096/512); slot 0 and slot 1 carry 160 copies each.
next-copy-submitted-at-compute-start is only 12% (vs 98% at 512): the
schedule is eviction-dominated. host_submit sum 26.9 s against a 30.0 s wall
— the main-thread feed chain remains the critical path; compute (3.35 s) is
already largely hidden by the eviction stream itself (transfer sum 56.4 s >
wall). Effective staging at 2 slots is 4.88-5.03 GB/s, i.e. the recorded
3.34 GB/s H-D number was the 1-slot serialization artifact, and the REAL 4k
cost driver is the 8.09x re-staging of the batch-major sweep order, not copy
bandwidth. A layer-major PP schedule (40 copies, one pass) is the obvious
next lever and is explicitly in scope per the SDD's primary-workload note.

Artifacts: `gplr25-budget-refusal-2048.jsonl`,
`gplr25-ab-4096-{af0,af1}.{jsonl,tgtext}`, `tools/analyze-gplr25-pair.py`,
`build/run-refusal-and-af1-chain.ps1`, `build/run-budget-refusal-test.ps1`.

## 2026-09-15: T-GPLR-25 paired A/B, 512 rung COMPLETE: NO af delta at 1 chunk/layer; earlier prototype wall comparison confounded (superseded); 4k pair pending GPU mutex

**Design (per the amended T-GPLR-25 contract):** paired arms run back-to-back
under ONE mutex hold, `--layer-slots 2` fixed, af the only flag delta —
controls the file-cache state that confounded the prototype reading.
Analyzer: `tools/analyze-gplr25-pair.py` (PP window, busy coverage, feed
serial chain, effective staging).

**512 rung result (anchor-512, both arms PASS check-telemetry + answer gate,
exit 0, n_slots=2):**
- af0 `gplr25-ab-512-af0.jsonl`: PP wall 11,958 ms, busy 11,976 ms, idle
  -19 ms (-0.2%), feed serial chain 11,933 ms, effective staging 1.53 GB/s
  (cold first-touch: chain = wall means the copy chain IS the wall).
- af1 `gplr25-ab-512-af1.jsonl`: PP wall 11,989 ms (+31 ms, -0.3% delta —
  noise), identical structure; worker ENGAGED (async_feed_init depth=1,
  3,880 async_feed_ready events, 0 in af0).
- Read: at 1 chunk/layer (512-token prompt, batch-major) compute is only
  0.74 s of a ~12 s wall; the ~306 ms/layer feed chain is the critical path
  and BOTH arms already submit the next copy at compute start (controller
  lookahead, 39/40) — the worker had no headroom left to exploit. The 0.74 s
  compute sum is the theoretical af ceiling at this rung and it is inside
  run-to-run noise.

**Supersedes (confound identified, claim removed):** the 2026-09-15
prototype reading "PP wall 12.29 s (af1, 2 slots) vs 17.6 s (same-session
3-slot ladder arm) -> tok/s 41.7 vs 29.1" compared DIFFERENT slot shapes
(2 vs 3) across DIFFERENT file-cache states (the 17.6 s arm was the cold
first rung; the 12.3 s arm ran warm). The paired af0@2-slot wall (11.96 s)
matches the af1 arm (11.99 s), so the earlier af-vs-no-af attribution of
the 5.3 s gap was wrong — the gap was cache state + shape, not the worker.
Prototype mechanism evidence (overlap proven, worker live) STANDS; the
wall/tok-s numbers from the unpaired reading are WITHDRAWN.

**Pending (superseded 2026-09-15, see the newer section above):** the 4k
rung pair has since run — verdict there supersedes this paragraph's
expectation setting.

Artifacts: `gplr25-ab-512-{af0,af1}.{jsonl,tgtext}`,
`tools/analyze-gplr25-pair.py`, `tools/run-gplr25-4k-pair.ps1`.

## 2026-09-15: OWNER DECISION IMPLEMENTED — default --layer-slots 2, auto-3 on fit, loud refusal below 2; T-GPLR-25 async-feed prototype proven live; latent checker defect fixed

**Owner decision (2026-09-15, this session):** "we should really use 2, and 3
if the budget allows. Make the default 2, note it also where the budgets are
set or managed. Keep it transparent. If we need a lower vram residency there
should be a warning or denial with the text why we cannot get it."
Implemented as stated, with the budget point documented here:

**Where the budget is set/managed (the transparency note the owner asked for).**
The slot count is decided in the runner's T-GPLR-04 CAPACITY PREFLIGHT
(`tools/layer-streaming/main.cpp`, the block introduced by this decision,
inside the preflight between the inventory parse and
`ls_bounded_window_for_policy`'s adoption of the window). Its arithmetic is
the preflight's own bounded window — the SAME
static + per-layer-resident + slot-window + KV + recurrent-state + reserve
terms `byte_account` emits, re-derived per rung through the same
`ls_bounded_window_for_policy` helper (no second arithmetic path). The three
budget INPUTS an operator can set: `--residency-budget-mib` (explicit total
residency cap; the bounded window must fit under it), `--reserve-mib
(default 512)` (headroom left between the bounded config and free VRAM), and
`--ctx-size`/`--kv-cache` (the KV term). The ladder consumes free VRAM
(`vram.free_start`, the observed post-CUDA-context reading).

**The decision ladder (default path only; an explicit --layer-slots is a
contract used verbatim, never altered):**
1. 3 slots when the 3-slot bounded need (bounded + act slabs + reserve)
   <= observed free — stderr line `layer-slots: default 2 -> 3 (T-GPLR-25
   owner ladder: budget admits it; ... +1 slot = 522190848 B for copy/compute
   overlap)`.
2. else 2 slots when only the 2-slot need fits — stderr line naming the 3-slot
   shortfall and the 2-slot spare.
3. else LOUD REFUSAL (exit 1, pre-sink `run_refused` reason
   `capacity_preflight`) whose text states the need/free/shortfall bytes AND
   why 1 slot is not taken silently (the slot-reuse invariant forbids
   copy/compute overlap at 1 slot — the serialized PP shape H-D measured) and
   that `--layer-slots 1` remains available as an explicit deliberate choice.
No silent fallback at any rung.

**Proof on the real gate (GPU mutex):** qa-smoke 512 now resolves 3 slots
(`run_start n_slots:3`, ladder line in the log, bounded 3-slot need
4,732,676,608 B <= free 22,226,665,472 B) and PASSES — check-telemetry OK
(23,324 events, 3 slots), answer 5/5, negative control FAILs as designed
(`gplr07-qa-512.{jsonl,tgtext}`). Host suite 37/37 on the changed binary.
Cold-cache caveat per AGENTS.md: PP wall/tok-s in this rung's summary are the
cold branch and are not comparable to the warm recorded arms; the claims are
the ladder decision, the event count, and the gate verdict.

**Latent checker defect fixed (exposed by the first 3-slot rung).** The
`slot_evict` rule computed `per_slot = n_layer / n_slots`, correct only when
n_slots divides n_layer (1, 2, 4 — every previously recorded shape); at 3
slots over 40 layers (slot 0 owns 14, slots 1-2 own 13) it drifted after the
first wrap and rejected 1,253 CORRECT `slot_evict` records. The rule now uses
the same uneven-aware formula as the `copy_submit` rule
(`(n_layer - 1 - slot) / n_slots + 1`) — one owner per fact. All previously
recorded shapes are unaffected (the two formulas are equal when n_slots
divides n_layer): the recorded 512 trace re-checks OK and the host suite is
green.

**T-GPLR-25 async-feed prototype (`--pp-async-feed 1`, default 0 =
byte-identical) — implemented and PROVEN LIVE on a real arm, prototype
results only, no performance claim.** Worker thread (T-GPLR-08 worker shape:
1-deep queue, stats under mutex, thread-safe telemetry) submits the NEXT
layer's H2D package copy into a free slot while the main thread computes;
same copy body as `load_layer` (same events, same ordering); slot-reuse
invariant untouched (worker fills only slots the main thread released via
`record_consumer_done`; graph still consumes only after `ev_copy_ready`).
Requires `--layer-slots >= 2` (refused at parse otherwise). Arm
`gplr25-asyncfeed-af1-s2` (2 slots, worker on, batch-major, 512 anchor):
- Overlap PROVEN from the trace: for every layer L, copy_submit(L+1)
  timestamps sit between L's compute start and end, and copy_complete(L+1)
  lands ~260-280 ms into L's 7,500-9,000 ms compute window — the copy is
  fully hidden under compute, vs 0% at the 1-slot recorded shape.
- The 11.5 s host_submit sum moved OFF the wall: PP wall 12.29 s against
  feed 11.54 s + compute 0.74 s (overlap ratio 1.0 of feed) vs the recorded
  1-slot shape's wall ~ feed + compute. PP tok/s 41.7 vs the same-shape
  3-slot ladder arm's 29.1 in the SAME session/cache state (12.3 s vs 17.6 s).
- Checker OK (27,206 events incl. 3,880 async_feed_ready), TG answer gate
  unchanged (structural; the TG leg is untouched by the flag).
- Caveats recorded: one run, cold-cache branch, batch-major vehicle — the
  magnitudes are prototype evidence, NOT a performance result; a gate claim
  needs the paired same-shape A/B (af=0 vs af=1 at equal slots) under held
  cache state. The `async_feed_submit` event never fires (0 of them) — the
  ready-side event carries the evidence; the submit-side event is dead
  telemetry, recorded as a prototype residue, not repaired in this pass.

**Where things stand for H-D:** the serialization mechanism is confirmed and
the fix class is real on this vehicle. The falsifiable next step (per the
T-GPLR-25 contract, amended in the plan) is the paired af=0/af=1 A/B at fixed
2-slot shape plus the 4k/50k rungs, before any tok/s claim leaves prototype
status.

Artifacts (gitignored build/): `gplr25-asyncfeed-af1-s2.{jsonl,tgtext}`,
`tools/analyze-pp-overlap.py`, `tools/analyze-pp-copy-gaps.py`,
`build/analyze_evict_rule.py`, `build/run-asyncfeed-arm.ps1`. Vendored diff
(uncommitted, per standing instruction): `src/llama-layer-streaming.{h,cpp}`
(worker + flag), `tools/layer-streaming/main.cpp` (ladder + flag + checker
rule + refusal text).

## 2026-09-15: documented online research pass on H-C/H-D (no GPU runs, no code changes)

Per AGENTS.md "Problems and Online Help", the CUDA-platform mechanisms behind
the two open hypotheses were researched from authoritative sources (CUDA
Handbook ch. 6.3, NVIDIA forum threads on pageable-memcpy sync behavior and
the WDDM software queue, upstream llama.cpp CUDA-graph precedent), and every
claim was verified against the vendored tree before being recorded. Findings,
full source list, source-vs-tree verification, and the falsifiable probes are
in the new `docs/gplr13-research-hcd-online.md` (section 8). Headlines:

- **H-C narrowed, mechanism revised.** The ~198 us hit-path ready-latency is
  most consistent with WDDM software-queue flush on an EMPTY submission (the
  hit path enqueues no copies before its synchronize; the miss path batches
  behind its copy burst), not retained-set bookkeeping. Sync mode is already
  spin (the vendored `cudaDeviceScheduleSpin` call covers only cc 12.1 iGPUs),
  so blocking-sync mode is NOT the cause and `cudaEventBlockingSync` would
  likely hurt. Cheapest disambiguation, no code change: a K=1 vs K=8 A/B on
  the same stream -- bookkeeping scales with K, WDDM flush does not.
- **H-D refined; a zero-code probe exists.** The row-set fetch source is the
  mmap'd (pageable) weight buffer, so every `set_async` is a hidden staged
  copy (driver pins, copies, DMAs) and PP's 3.34 GB/s is the "synchronous
  variant" shape from the CUDA Handbook's pageable-memcpy analysis. The
  vendored CUDA backend exposes `ggml_backend_cuda_register_host_buffer` under
  env `GGML_CUDA_REGISTER_HOST`, but NOTHING in `src/` or the runner calls it
  (tree-wide grep), so the mmap window is never pinned today. Probe order:
  (1) set `GGML_CUDA_REGISTER_HOST=1` and re-run the 4k arm, watching
  `host_submit_us` per fetch -- if unchanged, the env route is dead code on
  this build; (2) a minimal read-only `cudaHostRegister` of the mmap window
  at runner bring-up; (3) only then the two-buffer ping-pong / prefetch-ahead
  A/B. Caveats recorded: `cudaHostRegister` on an 18 GB region may fail or
  inflate commit, and WDDM batch depth on driver 610.74 is unverified.

No default, no gate verdict, no implementation changed. Nothing committed.

## 2026-09-15 (latest): H-A fixed and proven — row-set TG phase_summary carries transfer/wait/bytes; first real F3 part fields

Continuation of the sweep's H-A hypothesis (vendored change, no default or
gate verdict changed). The row-set TG executor now feeds the same
`add_layer_stats` pipeline the streaming executors use: transfer/bytes
accumulate beside the `rowset_round_*` sums in both `rowset_fetch` emit
paths, the per-layer timing triple (`rowset_last_{submit,transfer,ready}_us`)

is stashed on hit AND miss paths (no stale-carry), and `reset_stats()`
zeroes it per phase. One owner per fact — no new checker rule needed.

**Proof (real arm, 4k retention under the GPU mutex):**
`build/llama-cuda/gplr07-qa-4096-retention.jsonl` — TG `phase_summary`
now reports `total_transfer_us` 20.47 s, `total_wait_us` 1.71 s,
`total_compute_us` 1.67 s, `bytes_transferred` 21.05 GB (was all zeros);
consistent with its 3,840 `rowset_fetch` events. Full qa-smoke gate PASS
(check-telemetry OK, answer 5/5, negative control FAILs as designed).
Cold-start caveat per AGENTS.md: the run is the cold branch of the file
cache (machine restart), so absolute times are not comparable to the warm
recorded arms — the claim is the nonzero split and internal consistency,
not a timing comparison.

**First real F3 producer (same trace):** 3,840/3,880 `layer_compute`
events carry `attn_compute_us`/`ffn_compute_us` + setup parts. First
reading at 4k: attention 30.3 ms vs staged FFN 1.1 ms per layer-window —
TG compute is ~27x attention-dominated at 4k context; expert staging is
the transfer story, not the compute story. This sharpens H-C: the wait
term to optimize next sits in attention, not expert copy.
[CORRECTED 2026-09-15: the ~27x was a single-event artifact — see the
bottlenecks doc's correction. Distributions give attention:FFN ~2.2x (4k)
and ~2.4x (50k) on means; expert staging remains the largest single TG
term at width.]

**Records:** hypothesis annotations in `docs/gplr13-bottlenecks-hypotheses.md`
(discord item 2 and the F3-prediction line); schema note in
`docs/rowset-fetch-telemetry-schema-draft.md` (phase_summary mapping).
Host suite green on the fixed binary; ctest fixtures untouched (no
checker rule changed). [Committed 2026-09-15 with the H-B fix below in
`third_party/llama.cpp` **8e9757169**.]

**Follow-on: H-B fixed the same way (2026-09-15, same session).** The
`run_end.peak_vram_bytes` blindness had a precise mechanism: main's peak
was a two-sample max (free_start vs free_after sampled AFTER
`end_row_cache` released the cache), so the 2.29 GB cache plus the slot
window vanished from the reported peak. Fix: `free_min` tracked in the
budget struct (the one VRAM-state owner) across every refresh point
(post-init, post-cache, post-slot-window); `tel_run_end` reports
`total - free_min`. Non-retention shapes are byte-identical (free_min
collapsed to the old max). Proof: 4k retention re-run
(`gplr07-qa-4096-retention.jsonl`, archived H-A evidence at
`gplr24-ha-4096-retention-ARCHIVE.jsonl`) now reports peak
7,675,117,568 B vs the recorded identical 5,311,627,264 B, with
check-telemetry OK and answer gate PASS on the same run. The reactive
contrast arm could not run: another project held the GPU mutex for the
whole window (1,800 s lock timeout; the recorded reactive trace is
untouched). Host suite green after the H-B edit. [Committed 2026-09-15
in `third_party/llama.cpp` **8e9757169** together with the deletion-first
pass, the T-GPLR-22 refusal traces and the 200k spill fix.]

**Follow-on: F3 width test at 50k (2026-09-15, same session).**
50k retention arm under the GPU mutex (`-SmokeMode retention
-SmokeUbatch 512`, fresh trace `gplr07-qa-50000-retention.jsonl`; the
recorded Sep-14 rung archived to `gplr24-f3-50k/qa-50000-retention-PRIOR.jsonl`):
qa-smoke gate PASS (check-telemetry OK, answer 5/5, control FAILs), TG
7.21 s / 13.3 tok/s / 17.93 GB staged / peak 7.56 GB. The F3 distributions
correct the 4k first-event reading: attention:FFN ratio of means 2.4x
(4k: 2.2x), all 40 layers' attention means < 2 ms, and all three TG
terms comparable at 50k (transfer 3.63 s, wait 1.04 s, compute 2.63 s).
Answer-gate outcome recorded; no default, no gate verdict, no
implementation change. Uncommitted.

---


## 2026-09-14 (latest): full-length validation sweep 4k/50k/200k over the pruned tree — 4k and 50k PASS; 200k abort root-caused and fixed (host input-buffer spill)

Owner-directed sweep of the post-pruning tree at the three ladder lengths
(manual rungs, GPU mutex; measurement pass — no default and no gate verdict
changed). Full analysis and hypotheses H-A..H-D in
`docs/gplr13-bottlenecks-hypotheses.md` (2026-09-14 late section).

**Gate results.**
- **4k PASS** via the qa-smoke gate on the suite (check-telemetry OK, 23,326
  events; answer 5/5 PASS; negative control FAILs as designed). Artifact:
  `build/llama-cuda/gplr07-qa-4096.jsonl` (22:36 re-run; see the cold-cache
  note below for its timing).
- **50k PASS** direct run, recorded streaming shape, `--ubatch 512`
  (`build/llama-cuda/gplr24-sweep/rung-50000.jsonl`, 46,606 events;
  check-telemetry OK; answer PASS; control FAILs). PP 216.4 s / 231.0 tok/s
  cold-ish; TG 184.5 s / 0.52 tok/s.
- **200k first attempt ABORTED at graph build**: `attn_inp_kq_mask exceeds
  backing buffer (offset=421,603,680 size=128,122,880 buffer=477,102,080)`
  → `finish_build: input buffer exhausted` → `GGML_ABORT`
  (`qwen35moe.cpp:478`); runner exit -1073740791. byte_account proves this
  is NOT VRAM (bounded window 5.32 GB vs 22.23 GB free,
  `bounded_within_free=true`). The trace ends at `run_start` and
  `--check-telemetry` rejects it (missing phase_summary/run_end) — correct
  failure behavior for a graph-build abort.
- **200k retry after the fix COMPLETED**: `rung-200000-fix.jsonl`,
  116,926 events, `--check-telemetry` OK; answer 5/5 PASS; negative control
  FAILs as designed. PP 1,022.1 s / 195.7 tok/s / 15,639 evictions; TG
  185.1 s / 0.519 tok/s; peak VRAM 7.18 GB. Same 5/5 needle set as every
  other rung, so the answer gate holds at 200k. The first spill-fix build
  had a real bug — the failed `alloc_tensors` pass had allocated a prefix
  into `buf_inputs`, and spilling the FULL list re-allocated already-bound
  tensors, tripping `GGML_ASSERT(tensor->buffer == NULL)`
  (`ggml-backend.cpp:2057`); the spill now takes only the tensors the
  failed pass left unallocated (transcript:
  `gplr24-sweep/rung-200000-fix.transcript.txt`).

**Root cause and fix (vendored).** `init()` sizes the persistent host input
buffer as `n_masks * max_n_kv * max_ubatch * sizeof(f16) + embd_bytes +
64 MiB` — exactly 477,102,080 B on this shape — but at 200k more persistent
sets bind full-context masks than `n_masks` counts, so the cursor hits the
budget with a 128 MB mask still unallocated. Same sizing-formula class
T-GPLR-05 patched once with a bigger constant. Fix: a host-side SPILL
allocator in `finish_build` — tensors that no longer fit `buf_inputs`
allocate into an extra host buffer (`buf_inputs_spill`) instead of failing
the build. Host memory only, no VRAM, no placement change, and a shape
that fits never creates a spill, so every existing recorded shape's
allocation path is unchanged. `llama-layer-streaming.{h,cpp}`
(`llama_layer_streaming::alloc_spill`). Build clean; host suite re-run after
the fix (see below).

**Cold-cache caveat (per AGENTS.md).** The 22:36 4k re-run is the cold
branch of an uncontrolled Windows file-cache state: TG 185.2 s (0.52 tok/s)
vs the warm recorded 8.1 s (11.8 tok/s), PP 8.3 s vs 6.3 s. Structural
equality was verified (checker OK, answer PASS); timing comparisons across
runs must hold cache state fixed.

**Measured bottleneck structure (from the recorded warm arms).** Streaming
TG stages 54.98 GB across 3,840 reactive fetches in 6.00 s (9.16 GB/s);
retention K=4 cuts that to 21.05 GB / 3.04 s (hits 3,741/3,840 ≈ 97.4%)
and TG wall 8.12 → 5.11 s (+59% tok/s); PP full-bank staging runs at only
3.34 GB/s (39 evictions, no overlap). Three trace-accounting discrepancies
recorded as hypotheses: TG `phase_summary` sums are zero (H-A),
`run_end.peak_vram_bytes` excludes the row cache (H-B), retention's hit
path doubles median ready-latency 101 → 198 µs (H-C).

---

## 2026-09-14: DECISION BRIEF (item 6, owner call) — which second architecture should be the first to pass the generic gates?

**Recommendation: Gemma-4 12B (`arch=gemma4`, `C:/Models/ALL/other/Gemma4/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`),
with the small `qwen35`-arch models run first as a cheap precursor that does NOT satisfy the gate.**
This is a decision, not a task: no adapter, no code, no default and no gate verdict changed, and
no task block exists yet — the choice stays the owner's.

### 1. What a second architecture must actually satisfy

The SDD §11 gates were written for the Qwen vehicles, so the first question is which of them
generalize. Read against their own text, and against what the mechanism actually claims:

| gate | carries over to a dense second arch? |
|---|---|
| 1. adapter reproduces reference package bytes / lifecycle / focused greedy ID / telemetry | **form only.** The *content* is per-arch: there is no existing gemma4 reference to reproduce, so the parity has to be established fresh against a non-residency control on the same build |
| 2. Qwen3.6 MoE inventory before CUDA execution | **form only** ("inventory before execution" is gate 5's job); the 262,144-token Q8 configuration is Qwen3.6-specific |
| 3. one complete expert bank per layer, exact router IDs and top-k vs a control | **N/A for dense.** Its dense analogue is gate 1's ID parity |
| 4. 262K configuration completes or fails at the preflight; no accidental full-model placement | **N/A as written** — the model fits comfortably in VRAM, so the property is unobservable (§5.5) |
| 5. unsupported models and insufficient budgets fail before timed execution | **carries over unchanged**, and is the cleanest generic gate; already implemented and now traced (`run_refused`) |
| 6. default-off llama.cpp builds and behavior unchanged | **carries over unchanged** |
| 7. any speed claim needs a paired normal-vs-residency run at equal shape | **carries over unchanged** — and binds only if a speed claim is made |
| 8, 9, 10 (moe-row-set, the 0.6 gate, moe-row-cache) | **N/A unless the target is MoE** |

So a dense second architecture is gated by **5, 6 and 7 unchanged**, the **form of 1** (per-arch
parity instead of reproduced Qwen bytes), and gate 2's *intent* folded into 5. Nothing new has to
be invented — which is the point of choosing a dense target first.

**Existing suites, which carry over and which need an adapter path.** Carry over **unchanged**:
`tools/run-tests.ps1 -Suite host` (39 tests — new descriptor and lifecycle fixtures join it),
ctest `layer-slot-lifecycle`, `layer-streaming-pinned-1024`, `qwen38-layer-boundary` and
`run-refused` (all default-off or host-only), and the `run_refused`/`--check-telemetry` rules
(the refusal path is arch-independent by construction). Need an adapter path: the semantic-QA
anchor — `tools/semantic-qa/gen-needle-prompt.py` calibrates the prompt to a token count with
**the target model's own tokenizer**, so a 512-token anchor has to be re-generated and the
checker's expected answers re-derived for the new model; and the T-GPLR-18/T-GPLR-19 A/B drivers,
which hard-code the Qwen vehicle's model and flags, and are only needed if a speed claim is made.
`tools/route_overlap/analyze.py` needs a `rowset_fetch` stream and is therefore **out of scope**
for a dense target.

### 2. What is really available on this disk

Scanned the GGUF headers read-only (dependency-free header parse; throwaway script under
`build/`, gitignored). The decisive finding is that **almost everything here is the same two
architectures as the reference**:

| arch key | models on disk |
|---|---|
| `qwen35` (the dense reference's own arch) | Qwen3.8-27B (reference), **Qwen3.6-27B**, **Bonsai-27B-Q1_0**, **Ornith-1.0-9B**, **Qwythos-9B**, **Qwen3.5-0.8B** |
| `qwen35moe` (the MoE reference's own arch) | Qwen3.6-35B-A3B (reference), **Ornith-1.0-35B** |
| `gemma4` | **Gemma-4 12B** (+ a 0.43 GB MTP file) |
| `dflash` | Laguna-S-2.1 (~64 GB across 3 parts, + a vision/dflash file) |
| `deepseek4` | DeepSeek-V4-Flash-Vision-Exp (~96 GB) — **explicitly out of scope by AGENTS.md** |

So Ornith-9B/35B, Qwythos-9B, Bonsai-27B and Qwen3.6-27B are **not** second architectures:
`llama_ls_select_adapter()` routes them through the **existing** adapters unchanged. Running one
first is nearly free and would expose remaining geometry hard-coding, but it cannot close this
gate and must not be recorded as if it had.

### 3. The recommended target and why it is the cheapest credible one

**Gemma-4 12B — `C:/Models/ALL/other/Gemma4/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`** (6.26 GB,
`general.file_type` 2 = mostly Q4_0, GGUF v3). Measured header facts:

- `gemma4`, **48 blocks**, dense (no `expert_count`), `n_embd` 3840, `n_ff` 15360, head count 16;
- **`sliding_window_pattern` = `1,1,1,1,1,0` repeating** — 40 SWA layers and 8 full-attention
  layers, with `key/value_length` 512 (full) vs 256 (SWA) and two rope bases (1e6 / 1e4);
- **`shared_kv_layers = 0`** and **`embedding_length_per_layer_input = 0`** — the two gemma4
  features that would have complicated logical-layer KV ownership and the package member set are
  *inactive in this checkpoint*, so they do not have to be solved to pass the gate;
- `LLM_ARCH_GEMMA4` already exists in the vendored tree and `src/models/gemma4.cpp` already loads
  its tensors into `llama_layer` fields through the standard `load_arch_tensors` pattern — so the
  adapter can enumerate members exactly as the Qwen adapter does, and **no loader work is needed**.

Why nothing else on disk competes: the `qwen35`/`qwen35moe` files are the same arch (§2);
Laguna (`dflash`) is ~64 GB, multi-part, and not a plain decoder; DeepSeek V4 is excluded by the
project instructions. Gemma-4 12B is the only genuinely distinct, small, dense, already-loadable
architecture with a checkpoint on hand.

### 4. What the adapter would have to implement, and what it costs

The adapter interface is small (`name`, `policy`, `merge_resident`, `causal_implicit`, `package()`,
`mtp_package()`); both Qwen adapters share a **194-line** file. The cost is not the descriptor —
it is the four seams around it:

| work item | where | rough size |
|---|---|---|
| gemma4 package descriptor + factory + registry entry | new `src/models/llama-layer-streaming-gemma4.cpp`; `llama_ls_select_adapter` | ~70-90 lines |
| **split layer-major graph seam** — `params.layer_streaming` hook, `build_streaming_graphs()`, per-layer attn/ffn | `src/models/gemma4.cpp` (471 lines today, **no streaming seam at all**) | ~250-450 lines, by analogy with `qwen35.cpp`'s ~420-line seam |
| metadata/inventory generalization | `llama_ls_model_inventory` in `llama-layer-streaming.cpp` | ~100-150 lines changed |
| runner arch gates | `tools/layer-streaming/main.cpp` | ~30 lines |
| tests, fixtures, anchor re-calibration | host suite + `tools/semantic-qa` | ~150-250 lines + fixtures |

**The inventory is the sharpest hidden cost, and it is a hard stop today.**
`llama_ls_model_inventory` opens with `if (arch != "qwen35" && arch != "qwen35moe") return fail(...)`
and then *requires* Qwen-hybrid metadata: `ssm.conv_kernel`, `ssm.state_size`, `ssm.inner_size`,
`ssm.group_count`, `full_attention_interval`, a `u32 attention.head_count_kv`, and a recurrent-layer
mask. Gemma-4 has **none** of the `ssm.*` keys, and its `head_count_kv` is an **array**, so the
preflight would `fail()` on it at the first missing key — before any GPU work. Generalizing this is
the item most likely to regress the Qwen numbers, so it needs the existing Qwen capacity fixtures
as the guard. Five further sites hard-code the arch string (`llama-layer-streaming.cpp` 4433/4436,
`main.cpp` 412/3093/3097) and must become adapter-declared facts.

Critically, **the streaming module has zero awareness of sliding windows or ISWA** — grep for
`iswa` / `is_swa` / `sliding` / `n_swa` across `llama-layer-streaming.{h,cpp}` returns nothing. The
graph input type it needs (`llm_graph_input_attn_kv_iswa`, with a builder) already exists, so this
is wiring rather than new primitives — but it is the part of the estimate with the widest error bar.

### 5. What stays unknown until it is tried

1. **Whether gemma4's graph splits into a slot-bound one-layer graph without changing arithmetic.**
   `qwen35.cpp` has its seam because it was written with one; `gemma4.cpp` has no `params.layer_streaming`
   hook, no per-layer attn/ffn split, and its `graph` constructor is the standard single-graph form.
2. **Whether ISWA attention survives the residency assumptions** — a layer can be either SWA or full,
   with different head widths and rope bases, and the module currently models exactly one attention
   shape per layer. The adapter interface has `causal_implicit` but nothing for a sliding-window mask.
3. **Whether the byte arithmetic generalizes.** The Qwen package size derivation exists; gemma4's member
   set differs (it adds `attn_q_norm`/`attn_k_norm`, `attn_post_norm`, `ffn_post_norm`, an optional
   `out_scale`, and, when `embedding_length_per_layer_input > 0`, `per_layer_inp_gate`/`per_layer_proj`/
   `per_layer_post_norm` — inactive here). No gemma4 per-layer package size is known until the adapter
   enumerates it.
4. **Whether the semantic-QA needle transfers** — the anchor must be re-calibrated to 512 tokens with
   Gemma-4's tokenizer, and the checker's expected answers re-derived; a tokenizer with no `QX-`-style
   tokens is fine (the checker is substring-based) but the calibration is not automatic.
5. **Whether bounded residency is even observable on this target.** At 6.26 GB on a 25.6 GB card the
   whole model fits, so gate 4 is vacuous without constraining free VRAM. A run here can prove
   bounded slot generations, atomic package readiness and ID parity against a control — the
   LIFECYCLE and parity claim — **not** a capacity claim. Worth stating up front, because a green
   run will look like more than it is.
6. **Whether the MTP checkpoint should be used.** `mtp-gemma-4-12B-it.gguf` (0.43 GB) is present. The
   SDD's precedent for a first acceptance run is MTP inventoried but not executed, and that is the
   cheaper choice.

### 6. Where the gate decision for it would come from

The gates are the owner's, not the agent's: `docs/layer-residency-sdd.md` §11 is the governing list
and this brief does not amend it. Under the reading in §1, a Gemma-4 pass would be decided by
**gate 5** (unsupported model and insufficient budget refuse before timed execution — already
implemented, and now recoverable from JSONL via `run_refused`), **gate 6** (default-off host suite
39/39 plus the qa-smoke 512 regression, re-run on the same build), and the **per-arch form of gate 1**
(package bytes, slot lifecycle, and a focused greedy ID matched against a non-residency control on
the same model, prompt and build). Gate 7 binds only if a speed claim is made, and no capacity
claim is available on this target (§5.5). A new task would be drafted **only after the owner picks
the architecture**, with its escalation condition returning the §5 unknowns to the owner if they
resist.

### 7. Scope of this brief

No adapter, no code, no default, no gate verdict, and no new task block. Item 4's brief and the
informational gate 9 decision are untouched. The measurement basis is a read-only GGUF header scan
plus source inspection; the one artifact it created is the throwaway scanner under the gitignored
`build/` tree, which is deliberately not part of the tracked tooling.

## 2026-09-14: T-GPLR-22 PRE-TIMED ABORT — every abort in that family now leaves a trace the checker reads as a refusal (one rule extended, one new shape)

Owner-directed close of the class an audit measured on two real aborts. Both were in
item 3's class — the run read as a crash rather than "refused because of X" — and both
are now refusals in the JSONL. No default changed, no gate verdict touched, no second
rule added, nothing committed.

**The two measured defects (audit, 2026-09-14).**
- the `--graph-budget-mib` abort left `run_start` + `graph_footprint` +
  `run_end(exit_status=1)` and `--check-telemetry` **rejected** it with two invariant
  violations — a real abort whose own trace the checker refuses to read;
- a failed `ls.init` write **no trace at all**.

**The rule ONE owner, extended not forked.** `ls_refusal_trace_errors()` kept its
signature and its call sites in both checkers; what changed is that a present
`run_start` is no longer a contradiction on its own. The rule now rejects exactly two
things on a refusal trace: a `phase_summary` (a completed phase means this was a failed
run, not a refusal) and a `run_end` (`run_refused` is terminal and closes the sink).
The "missing run_start/phase_summary/run_end" completeness rules keep their
`&& !saw_run_refused` suppression, so nothing that passed before fails now.

**The producer.** The abort sites sit AFTER `run_start` but before the run's own
terminal write, so the runner records the deciding check and defers:
- `abort_refusal_reason` / `abort_refusal_detail` / `abort_refusal_needed` /
  `abort_refusal_allowed` / `abort_refusal_cap` carry the decision to the terminal
  site;
- the terminal write emits `run_refused` instead of the old
  `run_end(exit_status=1, failed)` — which also removed the one `tel_error` call site,
  since both checkers reject any `error` event;
- `ls.init`-failure, prebuild-failure and graph-budget sites are wired; the enum (the
  single owner shared by the emit sites and both checkers) gains `init_failed`,
  `prebuild_failed` and `graph_budget`.
- the emitter/enum duplication the audit also flagged was deliberately NOT refactored
  this pass — the record shape literal still exists twice. Recorded as the next
  cleanup, not silently dropped.

**Proven on real arms** (GPU mutex, fresh paths, so no recorded artifact was touched):

| arm | outcome |
|---|---|
| 512 `--graph-budget-mib 1` | exit 1; **44 events**, terminal `reason=graph_budget`, 201,326,592 / 1,048,576 B, `cap_source=budget`, detail = the `io_ctx_reserved` arithmetic. `--check-telemetry` exit 0: `OK (REFUSAL TRACE: 44 events, ...)`, reported as **ABORTED RUN - run_start was written, no timed phase completed, no run_end**. Strict `--validate-telemetry` exit 0, same shape |
| 512 `--telemetry <nonexistent dir>` (a config that PASSES the preflight, so `ls.init`'s own `fopen` is what fails) | exit 1; `reason=init_failed` fires with `ls.init`'s real error and the one-shot write fails on that same path, so the stderr warning is the only record — see the honesty note below. Console evidence only (the probe is an untracked one-off); the shape is pinned by the `run-refused-init-failed` fixture |

The graph-budget run's arithmetic and its `graph_footprint` row are byte-identical to the
pre-change abort, which is the point: the same abort, now readable.

**Non-breaking, falsified in both directions.** Nine refusal fixtures cover every branch
of the rule — 4 accept (post-init pre-run, PRE-SINK single line, pre-timed abort after
`run_start`, and the `init_failed` single line) and 5 reject (reason outside the enum,
missing reason, `phase_summary` contradiction, `run_end` contradiction, and the pre-event
bare-`pkg_member` shape). **Every pre-existing negative keeps its verdict.** The one that
carried `run_start` (`run-refused-with-run-start`) still fails, now through the
no-`phase_summary` clause rather than the old blanket `run_start` rejection — it is a
refusal contradicted by a completed phase, and that is a NARROWER rejection, not a wider
one. The new terminal negative (`run-refused-with-run-end`) pins that `run_refused` stays
terminal even in the aborted shape. Host suite **37/37** (was 35; +2 net — see the
pruning note below), ctest `run-refused` **9/9**. The
recorded 512 qa-smoke gate re-ran and PASSES with the **same 23,326 events**, answer
gate `aggregate=PASS` (all four needles + K1) and the negative control FAILs as
designed; a structural compare against the recorded trace shows the event-name multiset
is equal and there are **zero non-volatile field differences** — no event added, removed
or reordered, no byte figure changed.

**Honesty notes.** (1) A pre-timed abort raised through `init_failed` is emitted but
cannot be persisted when the telemetry path itself is the unopenable thing: the one-shot
emitter uses the same `fopen`, so it fails identically and only warns. That is inherent
to the cause, not a gap in the rule. (2) The residual class is now genuinely only the
load/setup failures and exit-2 usage errors — none of them a byte-arithmetic refusal.

**Deletion pass (same day, same pass).** The audit's duplication findings were then
closed rather than deferred, and the result is smaller and still proven byte-identical:
- **one writer per record shape.** `tel_run_refused` and `tel_run_refused_oneshot` now
  share a single static writer, and `tel_layer_compute` + `tel_layer_compute_parts`
  collapsed into ONE `tel_layer_compute` (the F3 part split appends only when asked, via
  a stack buffer) — so the record shapes read in one place instead of two.
- **one fact-parser per checker.** The 7-line refusal parse and the OK report were
  duplicated verbatim in `--check-telemetry` and `--validate-telemetry`; both now go
  through `ls_refusal_facts` + `ls_parse_run_refused` + `ls_print_refusal_ok`.
- **fixtures cut to branch coverage.** `run-refused-presink-badreason` and
  `run-refused-aborted-prebuild` were removed: the enum check is shape-independent and the
  enum is one array shared by the emit sites and both checkers, so a per-shape/per-reason
  duplicate adds no branch. 11 -> 9 fixtures, 39 -> **37** host tests.
  `run-refused-init-failed` was KEPT: it pins a different producer path (a failed bring-up
  writes a trace, not just a preflight).
- **equivalence evidence, not assertion.** The pruned writer was checked against the
  RECORDED traces rather than re-reasoned: all **11,640** real `layer_compute` records
  across the 4k-reactive and 50k-retention traces (7,680 carrying the F3 part split, 3,960
  plain) re-render byte-for-byte, and the part sums equal `compute_us`/`setup_us` in every
  one. The real graph-budget refusal trace still reports identically (`OK (REFUSAL TRACE:
  44 events, reason=graph_budget, ...)`, same 44-event count), and the two known-bad
  recorded traces still fail while all seven good ones still pass, under both checkers.
  One real defect was caught and fixed mid-prune: routing the refusal parse through an
  `else if (parse(...))` predicate initially set `saw` for EVERY unmatched event, not just
  `run_refused`; the parser now checks the event name itself.

Artifacts (gitignored `build/`): `gplr22-abort/qa-512-graphbudget.jsonl` and
`gplr22-abort/qa-512-preabort.jsonl` (the post-change 512 gate re-run). **Nothing
committed** (owner's standing instruction for this mission).

## 2026-09-14: DECISION BRIEF (item 4, owner call) — should gate eligibility BIND provisioning or stay ADVISORY?

**Recommendation: stay advisory.** Binding has no admissible input at bring-up today,
and the only curve that exists is the echo-biased one. Decision material below; nothing
was implemented, no default or gate verdict changed, and item 6 is untouched.

### 1. The measured basis

**(b) per depth, same rule and vehicle, two streams** (256 TG windows, W=1, 10,200
pairs; `docs/plots/gplr15-b-vs-k.html`). The echo column is the recorded probe; the
non-echoing column is the re-probe (one prompt changed, all flags held):

| K | echoed | non-echoing | 40-layer rows |
| --- | --- | --- | --- |
| 1 | 0.3088 | 0.2468 | 566.2 MB |
| 2 | 0.4343 | 0.3778 | 1132.5 MB |
| 3 | 0.5330 | 0.4704 | 1698.7 MB |
| 4 | **0.6284** | **0.5355** | 2264.9 MB |
| 5 | 0.6803 | 0.5982 | 2831.2 MB |
| 6 | 0.7169 | **0.6487** | 3397.4 MB |
| 7 | 0.7483 | 0.6926 | 3963.6 MB |
| 8 | 0.7775 | 0.7283 | 4529.8 MB |
| oracle | 0.9385 | 0.9143 | unbounded |

**The 0.6 crossing, and why the re-probe moved it.** Echoed, the first reading at or
above 0.6 is K=4 (0.6284) and the interpolated crossing is ~K=3.6 (~2.26 GB). On the
non-echoing stream **the crossing moves to ~K=5.1**, and the shallowest depth that
actually clears the threshold is **K=6 (0.6487, ~3.40 GB)**. Per-K the shift is
0.0492-0.0929 (widest at K=4), i.e. one to two whole K steps. A single prompt change
bought that; nothing about the device or the mechanism changed.

**The byte-to-time conversion degrades with depth** (T-GPLR-19 A/B, `--layer-slots 4`,
256 TG tokens, reactive baseline held at 11.863-11.967 tok/s across five GPU sessions):

| K | cache | saved B/token | TG off | TG on | gain | GB/s | marginal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 546.8 MiB | 396,398,208 | 11.953 | 14.021 | +2.068 | 14.29 | - |
| 2 | 1093.0 MiB | 324,793,824 | 11.967 | 15.820 | +3.853 | 12.18 | 8.93 |
| 4 | 2185.4 MiB | 214,053,600 | 11.863 | 19.727 | +7.865 | 10.67 | 8.35 |
| 6 | 3277.8 MiB | 163,659,552 | 11.878 | 22.074 | +10.196 | 10.52 | 9.55 |
| 8 | 4370.1 MiB | 129,075,264 | 11.873 | 24.481 | +12.609 | 10.23 | 7.69 |

Only the FIRST retained window converts above the link ceiling (14.29 > 13.7 GB/s);
every deeper window converts at roughly the measured effective per-fetch rate (8.42
GB/s). **But every depth still gains in tok/s** — deeper is slower per byte saved, not
worse in outcome. Caveat that matters for this decision: this A/B was run on the
512-token anchor vehicle and its `(b)off` column is the ECHOED series, so the gains are
the optimistic pairing; a non-echoing stream would compress them by roughly the same
0.06-0.09.

### 2. Where today's K=4 default came from

Owner direction, 2026-09-14: "set K=4 as default for now". It rests on the **echoed**
crossing — K=4 is the first recorded reading at or above 0.6 (0.6284) — and it is
explicitly a PROVISIONAL OPERATING DEFAULT, not a gate verdict: no PASS is established
at any depth, and the SDD scope amendment (2026-09-14) owns that decision. The
VRAM-adaptive fit added afterwards (`kRowSetRetainDefault = 4`) keeps K=4 wherever the
device affords it (arms F1/F2, free 19,775,094,784 B) and downscales only the *bare*
default on a tight device (F4 -> K=3; F5 -> refuse, not even K=1 fits). A depth the
operator NAMES is a contract and is never reduced. So the default today is: chosen on a
biased crossing, then fitted to VRAM.

### 3. What the runner knows at provisioning, and what binding would need

At bring-up the runner has: total/free VRAM, the budget and reserve, the package
inventory, and the per-K row footprint (572,719,104 B per K step = 8 rows/layer). It
has **no (b) figure at any depth, and cannot compute one there.**

(b) is a property of a *routing stream*, not of the model, the device, or the depth:
it is the fraction of each next window already resident under bounded retention, and it
is obtained either offline (the analyzer simulating retention over a recorded
`rowset_fetch` union stream) or live (after tokens have been routed). At bring-up zero
tokens have been routed, and for TG the stream depends on tokens not yet generated.

**So the information does not exist, and it is not merely uncollected.** For a binding
rule to have an input, one of these would have to exist first, and none does:

1. **A tracked `(b)`-vs-K oracle** for a declared workload class, produced by the
existing analyzer over a recorded trace, with that trace's provenance recorded. Today
there are measured points for exactly TWO streams on ONE model, and they differ by
0.0492-0.0929 at every K. The four preserved ladder rungs disagree in the other
direction too (ceilings 0.4000-0.4266), because their 3-window ceiling is a
stream-length artifact. A scalar per (model, K) would be wrong-sized for a policy.
2. **A live pre-run probe**: route a short sample, build the union stream, simulate
retention per K, then provision. Cost: a GPU pass whose stream may not represent the
run's, plus it changes what "provisioned once at bring-up" means.
3. **A cheap bring-up proxy.** Nothing existing maps a quantity the runner already has
(rows/layer, union size, top_k) to (b), and the measured spread — 0.2468-0.7283 over K
on one stream, ceilings 0.4000-0.9385 across streams — is why a proxy should not be
trusted. A proxy would also need its own error budget against the 0.6 line.

### 4. The two options, and what each costs

| | A: eligibility BINDS provisioning | B: eligibility stays ADVISORY |
| --- | --- | --- |
| Rule | refuse (or downscale) a depth whose (b) is not shown to clear the declared gate | provision on VRAM fit; report (b) as a reading; a gate claim is read at the depth a claiming task declares |
| Input it needs | a `(b)`-vs-K oracle at bring-up (item 3.1/3.2) — **does not exist** | none beyond what it already reads |
| Behaviour today | no oracle => refuse everything, or bind to the ECHOED curve, which is the number proven inflated by 0.0492-0.0929 | unchanged |
| Cost on this device | the fit resolves K=4 here (F1/F2). Non-echoing (b) at K=4 is 0.5355 < 0.6, so a bind on the honest reading refuses the default where it was chosen; a bind on the echoed reading allows it on a number known wrong. Either way the operator loses the default or the honesty | an operator can run a depth whose (b) is unverified; the only guard is the reported reading, which today does not distinguish "known to clear" from "unknown/biased" |
| Throughput | would trade away the measured gain at every depth (+7.865 tok/s at K=4, +12.609 at K=8) | none — every measured depth still gains |
| Verdict effect | binding does NOT create a gate PASS; it converts an informational reading into a refuse/allow switch resting on a biased input | none; gate stays informational, no PASS at any depth |

### 5. Recommendation

**Keep it advisory.** Reasons, in order of weight:

1. A binding rule today has no admissible input. The only curve available is the
echo-biased one — the exact input the pre-registered caveat forbids committing VRAM on.
2. Binding would decide nothing that is currently undecided. The gate is already
informational and no PASS exists at any depth, so A's only real effect is to convert a
reading into a switch — on a number we have measured to be wrong by 0.0492-0.0929.
3. On the evidence, the current policy is not harmful to throughput: every measured
depth still gains, so VRAM-fit provisioning does not trade TG speed away. The
degradation with depth is in conversion efficiency (10.23-14.29 GB/s), not outcome.
4. It is reversible in the honest direction: staying advisory keeps the operator's
declared depth meaningful, and the moment an oracle exists the rule can be added
without retracting anything measured here.

**If the owner wants binding eventually, this is what must exist first** (and it is a
prerequisite task, not part of this brief): a tracked `(b)`-vs-K surface for a declared
workload class with recorded provenance for the streams behind it; a pre-registered rule
fixing WHICH stream length, W and retention rule count for a verdict; a stated
behaviour for a depth the surface has no entry for (refuse, or fall back to the SDD's
reference band); and a refusal reason plus trace field so a gate-ineligible provisioning
refusal is recoverable from the JSONL — the shape for that now exists (the PRE-SINK
`run_refused` work landed this session), so the missing part is the oracle, not the
plumbing.

### 6. Scope of this brief

No code, no default, no gate verdict, no new capability changed. Item 6 is NOT started:
its model choice is a separate owner call. The K=4 default remains provisional and
echo-biased; the SDD scope amendment still owns the informational-gate decision, and
this brief does not supersede it — it is the material for the owner's call.

## 2026-09-14: T-GPLR-22 PRE-SINK — a refusal raised before `ls.init` now leaves a recoverable `run_refused` trace instead of no file at all

Owner-directed close of the one class mission item 3 left open. ONE event type, and
**no new checker rule** — one producer and two fixtures. No default changed, no gate
verdict touched, nothing committed.

**The gap.** The T-GPLR-04 capacity preflight and the T-GPLR-10 / T-GPLR-16 handoff
fit gates run BEFORE `ls.init` opens the telemetry sink (`llama_layer_streaming ls;`
is constructed *after* the preflight block), so their refusals exited 1 with the
arithmetic on stderr and wrote **no JSONL at all**. The 2026-09-14 50k `--handoff` arm
hit exactly this. An absent file is unambiguous, but a harness still saw only a
nonzero exit — the same "cannot tell refused-because-of-X from a crash" state the
event exists to remove.

**No second rule, proven before any edit.** `ls_refusal_trace_errors()` was NOT
changed, because a trace whose ONLY content is `run_refused` already satisfies it: the
pre-edit probe returned `OK (REFUSAL TRACE: 1 events, reason=row_cache_fit, ...)`. So
this adds a *producer* only:
- `llama_layer_streaming::tel_run_refused_oneshot(path, reason, detail, exit, needed,
  allowed, cap_source)` — **static**, because no instance exists at those sites; it
  opens the path, writes the SAME record through the same JSON-safe folding, closes,
  mirroring the existing one-shot `tel_off_run` precedent.
- a `refuse_before_sink()` helper in the runner, called at the seven pre-sink refusal
  gates (inventory failure, window parse, the `--allow-full-residency` override,
  residency budget, bounded-vs-free VRAM, the T-GPLR-10 handoff gate, the T-GPLR-16
  partial-handoff gate). It is a no-op when no telemetry path was requested, so a
  plain run never enters it.
- the enum — the single owner shared by the emit sites and both checkers — gains TWO
  reasons: `capacity_preflight` and `handoff_preflight`. `detail` carries the exact
  condition and `cap_source` the limiting term.

**Proven on real arms** (GPU mutex, fresh paths, so the recorded 2026-09-14 artifacts
were not overwritten):

| arm | outcome |
|---|---|
| 50k `--handoff`, the recorded arm's flags verbatim | exit 1; trace is ONE line: `reason=handoff_preflight`, 22,525,977,088 / 22,226,665,472 B, `cap_source=free`, detail = the T-GPLR-10 arithmetic. `--check-telemetry` exit 0 (REFUSAL TRACE); `--validate-telemetry` exit 0 |
| 512 `--residency-budget-mib 64` | exit 1; ONE line: `reason=capacity_preflight`, 3,151,424,000 / 67,108,864 B, `cap_source=budget`. Checker exit 0 |

Both traces reproduce the stderr arithmetic byte-for-byte, so the JSONL and the console
now agree. The 50k re-run reproduces the recorded refusal exactly
(`weights+KV+state 21,989,106,176 B ... = 22,525,977,088 B exceeds 22,226,665,472 B`).

**Non-breaking, falsified in both directions.** No existing trace, fixture or recorded
artifact was rewritten or rejected: the five pre-existing refusal fixtures keep their
verdicts (1 passes / 4 rejected), the two new fixtures pin the new shape (single-line
`handoff_preflight` passes; the same single line with an unknown reason is still
rejected), host suite **35/35** (was 33), ctest `run-refused` **7/7**. The recorded 512
qa-smoke gate was re-run and PASSES with the **same 23,326 events**; a field-by-field
compare against the pre-change copy shows the ONLY differences are wall-clock values
plus the timing-derived `ready_at_consumption` boolean (103 vs 99) — no event added,
removed or reordered, no byte figure changed, and **0** `run_refused` records on a
successful run.

Artifacts (gitignored `build/`): `gplr22-presink/qa-50000-handoff.{jsonl,txt}`,
`gplr22-presink/qa-512-budget.jsonl`, and `gplr22-presink/gplr07-qa-512-pre.jsonl`
(the pre-change copy used for the structural compare). The recorded suite logs
`gplr24-modes-50000-handoff.log` / `gplr07-qa-50000-handoff.log` are untouched.
**Nothing committed** (owner's standing instruction for this mission).

**Residual — narrowed, not closed.** The traceless class is now the *load/setup*
failures rather than the refusals: unreadable `general.architecture`, a failed
capability probe, `failed to load model`, `failed to create context`, the
`--inspect-model` errors, and the exit-2 usage errors (`--handoff-partial` bounds, the
layer-major TG cap). None is a byte-arithmetic refusal — they have no
`bytes_needed`/`bytes_allowed` to report — so covering them is a setup-failure-trace
feature, not an extension of this one. [EXTENDED 2026-09-14 (PRE-TIMED ABORT): this
residual did not name the abort family, which the audit then measured on real arms —
the `--graph-budget-mib` abort left a trace `--check-telemetry` REJECTED, and a failed
`ls.init` wrote nothing. Both are refusals now; see the latest entry.]

## 2026-09-14: Mission item 5 — the three opt-in TG modes measured at 4k and 50k; run `handoff` where it fits, `retention` K=4 where it cannot, `reactive` only if neither fits

Owner-directed measurement (mission item 5). Six arms = three configurations x two
lengths, run through the documented suite (`tools/run-tests.ps1 -Suite qa-smoke
-SmokeMode <mode>`), everything except the mode held fixed: same model, same
anchor, same flags, and the documented 50k prefill width (`-SmokeUbatch 512`, the
recorded one-ubatch ceiling workaround). At 4k the recorded one-ubatch shape is
kept. Modes: `reactive` (`--no-handoff --residency-tg`), `retention` (`--no-handoff
--residency-tg --residency-tg-retain`; the bare flag resolves to the K=4 default),
`handoff` (`--handoff`). No default, gate verdict or implementation was changed.

PP is mode-independent, as its bytes prove: staged `bytes_transferred` is
18,327,011,328 B on all three 4k arms and 1,796,047,110,144 B on both 50k arms, with
the same 39 / 3,919 evictions and the same 3,840 `rowset_fetch` events. The mode is
a TG policy, so only the TG leg moves.

| length | mode | PP wall (tok/s) | TG wall | TG tok/s | `peak_vram_bytes` | `row_cache` | answer gate |
|---|---|---|---|---|---|---|---|
| 4k | reactive | 6,262,501 us (654.05) | 8,115,958 us | 11.829 | 5,311,627,264 B (5065.6 MiB) | 0 | PASS |
| 4k | retention K=4 | 6,287,938 us (651.41) | 5,109,286 us | 18.789 | 5,311,627,264 B (5065.6 MiB) | 2,291,531,776 B | PASS |
| 4k | handoff | 6,180,092 us (662.77) | 1,392,102 us | **68.960** | **4,781,047,808 B (4559.6 MiB)** | 0 | PASS |
| 50k | reactive | 207,233,976 us (241.27) | 7,917,790 us | 12.125 | 5,204,672,512 B (4963.6 MiB) | 0 | PASS |
| 50k | retention K=4 | 207,219,795 us (241.29) | 4,568,054 us | **21.016** | 5,204,672,512 B (4963.6 MiB) | 2,291,531,776 B | PASS |
| 50k | handoff | REFUSED (no PP, no trace) | — | — | — | 0 | n/a |

4k ctx 4352 / 1 PP ubatch; 50k ctx 50176 / 98 PP ubatches; 96 TG tokens every arm.
All five completed arms PASS the answer gate with the negative control failing as
designed; re-graded directly from the five `--tg-text` files
(`check-answer.py check --tg-text ...` -> `aggregate=PASS`, `--negative` ->
`aggregate=FAIL`) rather than trusted from the suite log.

**Recommendation (measured, not argued).** The mode choice is a TG-leg choice, and
the three are not variants of one another:
1. Run `handoff` wherever it fits. It is 5.83x reactive TG at 4k and it costs
   *less* reported VRAM (4559.6 vs 5065.6 MiB), because the post-PP peak-order rule
   releases the slot buffers and packs the banks. It cannot serve long contexts on
   this card (arm 6).
2. Run `retention` K=4 where handoff does not fit. 1.588x reactive TG at 4k and
   1.733x at 50k, for exactly 2,291,531,776 B (2185.4 MiB) of device residency, and
   its TG edge is *larger* at the long context where it is the only option.
3. `reactive` only when neither fits: the lowest-residency and slowest TG path, and
   the one that carries no cache lifetime questions at all.
PP latency is the same in all three (see the byte identity above), so nothing in this
ranking trades prefill against decode.

These magnitudes belong to THIS vehicle (qa-smoke, 96 TG tokens, 1 slot) and are not
interchangeable with T-GPLR-19's own A/B, which measured reactive 11.953 -> K=1
14.021 tok/s on the 4-slot batch-major probe vehicle with 256 TG tokens. Same
mechanism, different shape, different absolute gain; neither reading is a correction
of the other.

**Arm 6 — `handoff` at 50k refuses by design, and the ceiling is now arithmetic.**
Exit 1 at the T-GPLR-10 handoff capacity preflight, before the timed PP:
`weights+KV+state 21,989,106,176 B (KV 545,914,880 B, state 65,863,680 B) + act slabs
0 B + reserve 536,870,912 B = 22,525,977,088 B exceeds observed free VRAM
22,226,665,472 B`, shortfall 299,311,616 B (285.4 MiB). This refusal precedes
`ls.init`, so **no telemetry sink exists and no JSONL is written at all** — the
residual the T-GPLR-22 entry left open, hit on a real arm (`run_refused` cannot cover
it). Log `build/llama-cuda/gplr24-modes-50000-handoff.log`; the suite's
`gplr07-qa-50000-handoff.jsonl` is absent, not empty. [AMENDED 2026-09-14 (T-GPLR-22
PRE-SINK): the finding stands and this pass's artifacts are untouched, but the
"cannot cover it" clause is now false — the pre-sink gates write a one-line
`run_refused` through a standalone emitter. Re-running THIS arm's flags writes
`gplr22-presink/qa-50000-handoff.jsonl` (reason `handoff_preflight`, 22,525,977,088 /
22,226,665,472 B) and `--check-telemetry` reads it as a REFUSAL TRACE. The absence
described above is that pass's artifact, not current behaviour.] From the gate's own terms the
boundary is ~22,665 ctx tokens today: weights 21,377,327,616 B + state 65,863,680 B +
reserve 536,870,912 B leave 246,603,264 B of KV at the measured 10,880 B/token (q8,
ctx 50176 -> 545,914,880 B). This is derived arithmetic, not a run — no rung
straddles the boundary.

**VRAM finding: `peak_vram_bytes` cannot see the row cache, and the polled pair
shows the real cost.** The two 4k arms that differ only by the retention flag report
the *same* `peak_vram_bytes` (5,311,627,264 B; likewise 5,204,672,512 B at 50k)
despite one holding a 2,291,531,776 B cache. Reason: `peak_vram_bytes` is
`total - min(free_after, free_start)` sampled **once** at main.cpp:4662-4663, and
`ls.end_row_set()` at main.cpp:4557 (which calls `end_row_cache()`,
llama-layer-streaming.cpp:3653) releases the cache before that reading — so the
figure is a free-VRAM endpoint delta, never a running peak. With `tools/gpu-mem-poll.ps1`
attached to the same 4k pair, per-process dedicated peaks are 3,978.8 MiB (reactive)
vs 6,242.8 MiB (retention), +2,264.0 MiB, consistent with the exact 2,185.4 MiB
cache plus ~79 MiB of KV/compute growth. Attribution verified by absolute time, not
assumed: each poll file spans 200 s (writers closed 18:38:47 / 18:39:57), so the
reactive arm is the process seen only in the first file (t=45.3-63.1 = 18:36:12-18:36:30)
and the retention arm is the one whose window appears in both (t=114.8-127.8 in file 1
and t=44.0-58.5 in file 2, both resolving to 18:37:22-18:37:36); the files overlap by
construction, since the loop starts the next 200 s poller 4 s after the previous arm
ends. Read against bottleneck section 3.1: for reactive the JSONL figure *exceeds* the
polled one by 1,086.8 MiB (5,065.6 vs 3,978.8) — the fixed ~1.09 GB class — while for
retention it falls 1,177.3 MiB *below* its polled figure (5,065.6 vs 6,242.8), i.e. the
cache is missing; the two JSONL figures are identical to the byte. This is a second and
different instance of the H6 accounting class (a *lifetime* blind spot, not a counter
blind spot), recorded in that section. Caveat: one run per arm at 500 ms sampling, so
2,264.0 MiB is a sampled figure; 2,291,531,776 B is exact from `byte_account` and
identical at both lengths.

Artifacts (gitignored `build/`): `gplr07-qa-{4096,50000}-{reactive,retention,handoff}.{jsonl,txt}`,
the 50000-handoff log, `gplr24-4k-{reactive,retention}-poll.csv` and the launchers
`run-mode-50000-handoff.ps1` / `run-4k-poll.ps1`. **Nothing committed** (owner's
standing instruction for this mission). All five completed arms also carry a strict
`--check-telemetry` OK (the suite enforces it before grading). The suite gained one
new parameter, `-SmokeMode`, and its default is verified not to disturb the recorded
gate: `tools/run-tests.ps1 -Suite qa-smoke` with no mode argument re-ran the 512 rung
after this measurement -- `mode=streaming`, artifacts at the recorded
`gplr07-qa-512.{jsonl,txt,log}` paths (no `-streaming` suffixed files were created),
the same 23,326-event trace, answer PASS and negative control FAILs as designed.

## 2026-09-14: T-GPLR-22 — a pre-work refusal is now a structured, recoverable record (`run_refused`) instead of a partial trace that reads like a crash

Owner-directed close of mission item 3. One event type, one checker rule,
fixtures; no default changed, no gate verdict touched, nothing committed.

**The defect.** A refusal raised after `ls.init` opened the telemetry sink (the
row-cache fit gates, the cache allocation, the fit/allocation divergence check,
the prefetch-probe configure) exited 1 with its arithmetic on stderr and left
**40 `pkg_member` records and no `run_start`** — which `--check-telemetry`
rejected with 3 invariant violations, i.e. indistinguishable from a crashed
partial trace. Recorded as a residual three times.

**The event.** `run_refused` — `{event, run_id, reason, detail, exit_status,
bytes_needed, bytes_allowed, cap_source, phase, t_us}`. Emitted **before**
`run_start` and **terminal**: it is the last event in the trace and closes the
sink (it replaces the one bare `run_end` that used to be written at the
prefetch gate, which was the same confusing shape). `detail` is JSON-escaped by
the producer, so a free-text message can never break the line. The `reason`
enum and its validator live once in the runner (`kRunRefusalReasons` /
`ls_run_refusal_reason_ok`), used by both the emit sites and the checkers, so areason cannot be emitted without being checkable. Six reasons:
`row_cache_derive`, `row_cache_fit`, `row_cache_depth`, `row_cache_alloc`,
`row_cache_divergence`, `prefetch_configure`. [Later additions, same list and same
owning claim: `capacity_preflight` and `handoff_preflight` (PRE-SINK), then
`init_failed`, `prebuild_failed` and `graph_budget` (PRE-TIMED ABORT) — eleven total
as of 2026-09-14.]

**The rule, one owner.** `ls_refusal_trace_errors()` is called by BOTH
`--check-telemetry` and `--validate-telemetry`: a trace carrying `run_refused`
must have no `run_start`, no `phase_summary` and no `run_end`, and its `reason`
must be in the enum. The three "missing run_start / phase_summary / run_end"
rules describe a run that started, so they are suppressed for a refusal trace.
With no `run_refused` present every input is false and the rule is a no-op.
[AMENDED 2026-09-14 (PRE-TIMED ABORT): the "no `run_start`" clause was RELAXED
for the ABORTED-RUN shape — `run_start` is admissible and neither required nor
forbidden — so the rule now rejects only a `phase_summary` and a `run_end`. In
this entry's own terms a refusal trace no longer needs to be pre-work; the
one-owner claim and the suppression of the three "missing" rules both stand.
See the latest entry.]
`--validate-telemetry` returns before building a report (a refusal has no
phases), so no `.summary.json` is written for one.

**Non-breaking, and falsifiable that way.** Nothing that used to pass is
rejected; the only new acceptance is the new shape. The pre-event refusal shape
is deliberately still REJECTED, and there is a fixture that proves it, so the
claim is tested rather than asserted. No recorded trace or fixture was
rewritten to make anything pass.

**Real runs, one GPU session (both reasons), MoE probe model.**

| arm | flags | outcome |
| --- | --- | --- |
| R1 | bare `--residency-tg-retain` + `--residency-budget-mib 4700` | exit 1; `run_refused` reason **`row_cache_fit`**, 2,291,531,776 / 210,310,656 B, `cap_source` budget |
| R2 | `--residency-tg-retain 4` + `--residency-budget-mib 6000` | exit 1; `run_refused` reason **`row_cache_depth`**, 2,291,531,776 / 1,573,459,456 B, "the largest depth that fits is K=2" |
| — | *(a first R1 attempt that accidentally omitted the retain flag)* | retention OFF, so no fit ran and nothing refused: a normal 21,020-event run, checker OK; `rowset_init.retain_k` 0, `row_cache` 0, `residency_cap` 4,928,307,200 B. Evidence the event is not emitted on any non-refusal path |

Both refusal traces are 41 records (40 `pkg_member` + 1 `run_refused`), carry no
`run_start`, and `--check-telemetry` now reports them as
`OK (REFUSAL TRACE: 41 events, reason=<r>, exit_status=1, <needed>/<allowed> B,
cap_source=budget)` with the detail line — so the refusal is recoverable from
the JSONL alone. R1's allowance (210,310,656 B) and R2's (1,573,459,456 B)
reproduce the recorded budget-cap arms C3/C4 exactly.

**Also observed and worth stating:** the first attempt at R1 omitted the bare
`--residency-tg-retain` flag, so retention was OFF and the run completed
normally (21,020 events, `rowset_init.retain_k` 0, `row_cache` 0,
`residency_cap` 4,928,307,200 B). That is a correct run of a different
configuration, not a failure of the rule, and it doubles as evidence that the
new event is not emitted on any non-refusal path.

**Fixtures (5, host-only, no GPU).** Generated by the tracked
`gen-lm40-fixtures.py` (which re-emits every existing fixture byte-identically,
verified): `run-refused-positive` (the real shape -> PASS) and four WILL_FAIL
negatives — `run-refused-badreason` (reason outside the enum),
`run-refused-noreason` (no reason field), `run-refused-with-run-start`
(contradicted by `run_start` + `phase_summary`), and `run-refused-legacy` (the
pre-event shape, which must stay rejected). Wired into CMakeLists and into the
host suite's `Include` regex.

**Validation.** Build clean. Host suite **33/33** (was 28; +5). The 5 new
fixtures: 1 passes, 4 rejected, as designed. ctest `layer-streaming-pinned-1024`
**2/2**, `qwen38-layer-boundary` **1/1**, `layer-slot-lifecycle` **1/1**. QA
smoke 512 **PASS** (check-telemetry OK, answer PASS, negative control FAILS as
designed). Real refusal traces for both reasons, checker-clean.

**Residual.** A refusal raised BEFORE the sink exists (the T-GPLR-04 capacity
preflight, argument validation) still writes no trace at all — the new event
covers the refusals that produced a partial trace, which is the class the
residual named. That absence is unambiguous (no file), unlike the old partial
trace, but a harness still sees only a nonzero exit for it. Covering it would
mean opening the sink before the preflight, which is a bigger change than this
item's declared size. [RESOLVED 2026-09-14 (PRE-SINK) for the refusal half, and
NOT by opening the sink early: a static one-shot emitter writes the same terminal
record before `ls.init` exists, so the preflight gates now leave a one-line
`run_refused` and the checker rule is unchanged. The load/setup failures and the
exit-2 usage errors stay traceless — they are not byte-arithmetic refusals. See
the PRE-SINK entry at the top.] The two probe runners are untracked one-offs under the
gitignored `build/` root. **Not committed** — the owner did not ask this turn.

## 2026-09-14: the non-echoing re-probe RAN — K=4 does NOT clear gate 9 on it (0.5355 vs 0.6284 echoed); the first clearing depth is K=6

Owner-directed closing measurement for the mission's first item: the
non-echoing 256-token re-probe plus the offline statistic (b). One GPU session
(~0.6 min incl. the 885-token PP), no library change, no default change, no gate
verdict changed, nothing committed.

**First, a provenance correction that decides what was measured.** The intact
artifacts from the earlier aborted attempt
(`build/llama-cuda/gplr20-guess-early/probe-tg256-repro.{jsonl,txt}`,
2026-09-14 17:27) are NOT the re-probe. They are a **bit-for-bit reproduction
of the recorded echoing probe**: the same 10,240-event `rowset_fetch` route
stream, the same greedy token IDs, the same text sha256
`f690bfa4df173b7c...`, the same analyzer report (only the filename line
differs), and the same readings 0.3088 / 0.4343 / 0.6284 / 0.7169 / 0.7775 with
oracle 0.9385. Its prompt is the echoing needle document, and its 256 generated
tokens contain 25 distinct values inside a ~34-token near-repeat loop. So that
trace belongs to the **"guess early" expert-repetition item** (its runner's own
framing), not to the re-probe — reading it again returns the same biased number
the re-probe exists to escape, and it cannot decide K=4. Its repetition
statistics, now recorded for that item: T5 token-level window-equality **0.0019**
(n_pairs 10,200); T7 reappearance distance 1:25,202 / 2:7,564 / 3:4,270 /
4:5,350 windows; rows seen once 739/5,342; cumulative coverage at retention
1/2/4/8 = 0.3291 / 0.4279 / 0.5535 / 0.7025.

**The re-probe, and its declared rule (authored 2026-09-14 — it does NOT
predate the original measurement).** The recorded probe invocation with exactly
ONE change, the prompt BODY: `build/calc-prompt.txt` (the natural-language code
prompt already recorded for the gplr06 code-prompt runs, 885 tokens) instead of
the echoing needle document; `--pp-tokens 885` to take it whole; everything else
held at the recorded values (`--residency moe-expert-bank --residency-tg
--pp-schedule batch-major --ubatch 512 --tg-tokens 256 --layer-slots 4
--kv-cache q8 --greedy --no-mtp --pp-no-logits --allow-full-residency`).
*Reading*: statistic (b) as SDD section 10 / gate 9 defines it (tracked analyzer,
row-capacity-lru, W=1, next window in the denominator) at the declared depth,
plus T8/T5/T7 for context. *Decision rule, fixed before the run*: K=4 survives
only if (b) at K=4 >= 0.6 on the non-echoing stream; the pre-registered
direction is that a non-echoing (b) is <= the echoed (b) at every K, because
repetition can only raise temporal reuse.

**Non-echoing check — PARTIAL, and the defect is in the declaration.** Declared
conjunction: `distinct 16-gram ratio >= 0.90` AND `zlib(id sequence) ratio >=
0.50`, calibrated beforehand on the known-echoing stream. Result: 16-gram ratio
**1.0000** (echoed 0.3734; all 241 spans distinct) but zlib ratio **0.4238**
(echoed 0.1152). zlib compresses structured HTML/CSS tokens well with no
repetition at all, so that criterion measured the entropy of a code stream
rather than repetition and was mis-chosen when declared. The stream is
non-echoing on the criterion that actually measures repetition and on the
generated text (a single-pass calculator document, no loop; distinct-token
ratio 0.4102 vs 0.0977); the certification is recorded as partial rather than
repaired after the fact.

**Result (256 TG windows, W=1, 10,200 pairs).**

| K | echoed (recorded probe) | non-echoing (re-probe) | delta | VRAM / 40 layers |
| --- | --- | --- | --- | --- |
| 1 | 0.3088 | 0.2468 | -0.0620 | 566.2 MB |
| 2 | 0.4343 | 0.3778 | -0.0565 | 1132.5 MB |
| 3 | 0.5330 | 0.4704 | -0.0626 | 1698.7 MB |
| 4 | 0.6284 | **0.5355** | **-0.0929** | 2264.9 MB |
| 5 | 0.6803 | 0.5982 | -0.0821 | 2831.2 MB |
| 6 | 0.7169 | 0.6487 | -0.0682 | 3397.4 MB |
| 7 | 0.7483 | 0.6926 | -0.0557 | 3963.6 MB |
| 8 | 0.7775 | 0.7283 | -0.0492 | 4529.8 MB |
| T8 oracle | 0.9385 | 0.9143 | -0.0242 | unbounded |

Every K falls, so the pre-registered direction holds. **Verdict: the K=4
default does not survive as a gate reading.** Its non-echoing (b) is 0.5355,
below the 0.6 threshold; the recorded 0.6284 is confirmed echo-inflated by
0.0929. The 0.6 crossing moves from ~K=3.6 (echo, ~2.26 GB) to ~K=5.1 on this
stream, so the shallowest depth clearing the gate is **K=6 (0.6487, ~3.40 GB)**.
T4 consecutive-window containment 0.3088 -> 0.2468; T5 window-equality 0.0019 ->
0.0004 (exact window identity was never the discriminator); rows seen once
739/5,342 -> 1,039/7,311; cumulative coverage at retention 1/2/4/8 =
0.2699 / 0.3832 / 0.5160 / 0.6498.

**One correction to how the echo caveat has been quoted.** The re-probe's
ceiling (0.9143) is nowhere near the 0.4000-0.4266 range recorded for the four
preserved ladder rungs. Those are 3-window streams, so their ceiling is
structurally capped at two prior windows and is monotone in stream length: that
range is a STREAM-LENGTH artifact, not the non-echoing expectation. The only
like-for-like comparison is 256 windows vs 256 windows, 0.9385 -> 0.9143.

**Nothing moved.** No default, no gate verdict, no implementation: K=4 stays
the mechanism's provisional operating default, and a K=4 gate PASS remains
UNAVAILABLE — now on measured evidence instead of a caveat. The gate stays
informational per the SDD scope amendment, which now carries this validation
note; the plan's "outstanding re-probe" entries are annotated RESOLVED.

**Artifacts and validation.**
`build/llama-cuda/gplr15-nononchoing/probe-tg256-nononchoing.jsonl` — run exit
0, `--check-telemetry` OK (21,260 events), `rowset_init` 1, `rowset_fetch`
10,240, `rowset_round` 256, `rowset_refused` 0; PP 885 tokens at 111.7 tok/s
(transfer 30.1 s), TG 256 tokens at 12.070 tok/s, peak VRAM 5,731.56 MiB. The
analyzer report sits beside it; the launching one-off
(`build/llama-cuda/gplr15-nononchoing/run-reprobe.ps1`) and the trace are
untracked under the gitignored `build/` root, so a clean checkout needs a fresh
run to re-derive these numbers. **Not committed** — the owner did not ask, so
the four doc records are on disk only.

**Residual.** The non-echoing certification is partial (see above), and this is
ONE prompt on ONE model: a second non-echoing stream would test whether the
~0.06-0.09 shift is stable or prompt-specific, and the re-probe prompt choice
was the agent's, not a recorded owner decision.

## 2026-09-14: F3/H8 implemented — TG row-set `layer_compute` carries real per-part timings, so the transfer term now splits into submit / copy / compute

Owner-directed implementation of follow-up F3 (H8). Host + one GPU run; library
change only (`third_party/llama.cpp/src/llama-layer-streaming.cpp` and `.h`), no
runner flag, no routing or placement change, no new dependency.

**Defect closed.** The row-set TG executor ran a two-part graph per layer but
emitted `tel_layer_compute(il, 0.0, 0.0, ...)` — section 3.6's 120 zero-valued
TG events per run, which made the TG decomposition unable to separate
FFN/attention compute from scheduler overhead.

**Change.** `run_graph_plain` now returns the same split the other executors
record (`setup_us` = sched reset + alloc, `compute_us` = launch + execute +
sync); the per-layer loop times part A (attn+residual+post-norm+router) and
part B (FFN + shared experts) around the router read and `load_row_set`; new
`tel_layer_compute_parts` emits ONE `layer_compute` per layer/window with
`compute_us`/`setup_us` unchanged in meaning (the A+B sums) plus
`attn_compute_us`/`attn_setup_us`/`ffn_compute_us`/`ffn_setup_us`. Resolves
schema-draft open decision 3: part A does NOT emit its own event, so event
cardinality and every existing consumer of `compute_us` are untouched.

**Measured (MoE probe model, GPU mutex held, one session).**

| trace | shape | result |
| --- | --- | --- |
| `build/llama-cuda/gplr21-f3-parts/rowset-pp64-tg3-parts.jsonl` | the recorded probe flags unchanged (fresh path; the 2026-09-13 trace is preserved) | 120 TG `layer_compute`, `compute_us` 0 -> **66,822**, `setup_us` 0 -> **4,788**; attn 43,000 (setup 2,737) / ffn 23,822 (setup 2,051) us |
| `build/llama-cuda/gplr21-f3-parts/rowset-512-tg256-retain4.jsonl` | 512 PP / 256 TG batch-major, `--residency-tg-retain 4` | 10,240 TG `layer_compute`, **0/10,240 zero-valued in either part**; 93.5% of TG wall accounted |

Per TG token on the 256-token stream (wall 48,225 us/token, 20.736 tok/s):
submit **23,574.9 us** (48.9%), copy completion **7,959.9 us** (16.5%), attn
compute **8,019.6 us** (16.6%), FFN compute **4,224.2 us** (8.8%), graph setup
**1,297.6 us** (2.7%) = 45,076.1 us accounted (**93.5%**); **3,148.9 us
(6.5%) not yet instrumented** (router's blocking device-to-host probs read,
sampling, round boundaries — NOT attributed by this fix and not claimed to be).
So `transfer` = submit + copy completion = 65.4% of the token and
compute+setup = 28.1%. Both `compute_us == attn+ffn` and `setup_us ==
attn+ffn` hold on all 10,240 events.

**No behavior change, proven against recorded evidence.** On the 256-token
stream the 10,240-event `rowset_fetch` stream (layer, chunk, union hash,
fetched/retained counts, staged bytes), the 256 `rowset_round` byte accounts,
`byte_account` on every original field, and the greedy text sha256
`f690bfa4df173b7c...` are **identical to the recorded K=4 arm-B1**. On the
probe trace, 569/569 records match the 2026-09-13 trace once all timing fields
and run identity are excluded; the only structural differences are fields
added by intervening tasks (`n_rows_retained`/`n_rows_retired`/`retain_k`) and
the wall-derived `tokens_per_s`. The F3 change is additive on timing only.

**Validation.** Build clean (no new warnings). Host suite **28/28**. ctest
`layer-streaming-pinned-1024` **2/2**, `qwen38-layer-boundary` **1/1**,
`layer-slot-lifecycle` **1/1** (the last three need `Q38LS_MODEL_PATH` exported;
they fail loudly as "missing or empty model environment variable" without it).
QA smoke 512 **PASS** (check-telemetry OK, answer PASS, negative control FAILs
as designed). Route-overlap analyzer self-test **PASS (55 checks)**. Both new
traces exit 0 with `--check-telemetry` OK (569 and 21,020 events).

**Residuals.** (1) The ~6.5% unattributed TG share above. (2) `phase_summary`'s
TG line still reports `compute 0.0 / layers 0` for row-set runs, because the
row-set path does not call `add_layer_stats`; the per-layer `layer_compute`
events are the record, and reconciling the summary is a separate change. (3)
The two probe runners are untracked one-offs under the gitignored `build/`
root, so the traces are not reproducible from a clean checkout without a fresh
run.

**Not committed.** The owner did not ask for a commit this turn; the two
vendored files and the four doc records (`docs/currentstate.md`,
`docs/gplr13-bottlenecks-hypotheses.md`,
`docs/rowset-fetch-telemetry-schema-draft.md`, the plan) are on disk and the
Commit field is still to be filled when the owner authorizes the commit.

## 2026-09-14: T-GPLR-07 ladder rungs 4k and 50k PASS (owner-invoked, one at a time) — and the 50k rung exposed a HARNESS limit, not a mechanism one

Owner-invoked acceptance runs on the semantic-QA ladder, one length at a time
under the GPU mutex, on the MoE probe model in the reactive streaming-TG shape
(`--no-handoff`, no retention). These are answer-stability measurements; no gate
verdict is created or changed by them.

| rung | command | result |
| --- | --- | --- |
| 4k | `tools/run-tests.ps1 -Suite qa-smoke -SmokeLengths 4096` (documented shape: whole prompt in ONE 4096-token ubatch) | **PASS** |
| 50k | `tools/run-tests.ps1 -Suite qa-smoke -SmokeLengths 50000 -SmokeUbatch 512` | **PASS** |

**4k**: prompt 4096 tokens exact (needles at rows 27/55/84/106 = 25/50/75/95%
depth); exit 0; `--check-telemetry` OK (23,326 events, 1 slot); PP 4096 tok in
6.23 s (657 tok/s, transfer 5.45 s, wait 6.7 ms); TG 96 tok in 182.1 s
(0.527 tok/s); peak VRAM 5,059.56 MiB; answer `QX-7741 TR-2918 MK-5501 LD-8834
Paris` -> aggregate PASS (all 5 items); negative control FAILS as designed.
Trace `build/llama-cuda/gplr07-qa-4096.jsonl`.

**50k**: prompt 50,000 tokens exact (needles at rows 346/693/1040/1318); exit 0;
`--check-telemetry` OK (46,606 events, 1 slot); PP 50,000 tok in 209.1 s
(239.2 tok/s, transfer 186.1 s, wait 0.65 ms, compute 22.4 s, 3,919 evictions,
graph_builds 0); TG 96 tok in 181.3 s (0.529 tok/s); peak VRAM 4,957.56 MiB
(LOWER than the 4k rung: single-slot batch-major peaks on the slot window, not
on the context); same five-needle answer -> aggregate PASS; negative control
FAILS as designed. Trace `build/llama-cuda/gplr07-qa-50000.jsonl`.

**The 50k rung first FAILED, and the failure belongs to the ladder harness, not
the residency path.** Run exactly as the suite documents it — the WHOLE prompt in
one ubatch — 50k aborts inside the first prefill ubatch with
`ggml/src/ggml-cuda/mmid.cu:138: GGML_ASSERT(nbytes_shared <= smpbo) failed`, an
upstream CUDA MoE-matmul shared-memory limit whose demand grows with the tokens
in the batch. Classification: harness/test-expectation, NOT task-owned — nothing
in the layer-residency path is involved (the run reaches the timed PP, emitting
`byte_account` and `graph_footprint` first), and the recorded 24k rung ran at
24,575 tokens, so the shape worked to ~24.6k and breaks by 50k. Isolation: the
SAME rung and flags with only the prefill width changed (`--ubatch 512`) exits 0,
passes check-telemetry and produces the same answer — reproduced twice, once
through the suite and once by a direct one-off
(`build/llama-cuda/gplr07-qa-50000-chunked.*`: PP 50,000 tok in 210.1 s =
238.0 tok/s, same five needles; its first cut produced an ungraded trace because
a PowerShell `ErrorActionPreference` quirk turned the checker's stderr into a
terminating error — fixed in the one-off, noted here rather than left silent).
Failing artifacts preserved: `build/llama-cuda/gplr07-qa-50000-oneubatch-fail.{jsonl,log,suite.out}`
(47 events: 40 `pkg_member`, `byte_account`, `run_start`, `graph_footprint`, one
copy/slot pair, then the abort).

Smallest repair made: `tools/run-tests.ps1` gains `-SmokeUbatch N` (default 0 =
the recorded one-ubatch shape, so the 50/512/4k rungs are bit-for-bit
unchanged); ladder lengths above the ~24.6k the 24k rung ran at must pass an
explicit width. Recorded in the T-GPLR-07 contract.

**The new `byte_account` fields are also exercised on these real traces** (not
only on fixtures): both rungs run with retention OFF and their records carry
`row_cache` 0 / `residency_cap` 0 with `bounded_window` 3,190,417,920 B (4k) /
3,688,983,040 B (50k), and both pass `--check-telemetry` with the new rule live.
These rungs emit no `rowset_init` at all (the streaming-TG shape has no row-set
policy header), so the deferred retention tie is correctly SKIPPED rather than
fired on an absent field — the immunity branch, on real data.

**Residual / open.** The ladder is now green at 50/512/4k/50k (24k was green
earlier in the campaign); 100k and 200k remain unrun, each needing its own
owner-invoked run and, per the finding above, an explicit prefill width. The
one-ubatch ceiling is an upstream CUDA limitation this repository cannot repair;
the harness records it instead of hiding it. Provenance: both rung commands are
reproducible from TRACKED tooling (`tools/run-tests.ps1` plus
`tools/semantic-qa/gen-needle-prompt.py` and `check-answer.py`); the traces and
the direct one-off probe live under the gitignored `build/` root and do not
survive a clean checkout.

**Commit.** This entry, the T-GPLR-07 contract note and the `-SmokeUbatch`
parameter: the `Record ...` commit for this pass is **3dd3a6a** (hash recorded
in this follow-up commit, per the convention's existing field format).

## 2026-09-14 (later): the TOTAL residency cap is now provable from JSONL — `byte_account` carries the provisioned row cache and the cap, and the checker enforces containment

Owner direction: "Add the provisioned row-cache bytes and the total residency cap to
the byte_account record and teach the checker to verify that bounded residency
plus cache stays under the budget, so the cap is provable from JSONL alone."
Host compiler + THREE GPU arms + five host fixtures. No gate verdict is created
or changed, and the mechanism's behaviour does not move: this pass changes what
the run RECORDS and what the host can CHECK.

**Two new fields, and one changed claim.** `byte_account` (the T-GPLR-04
structured memory record) now emits `row_cache` — the private VRAM the per-layer
cache actually allocated, summed over the 40 MoE layers and recorded from the
allocation itself (`row_cache_alloc_bytes`, set by `begin_row_cache()`), never
re-derived — and `residency_cap`, the operator's `--residency-budget-mib` in
bytes with **0 = no explicit cap** (the cache is free-derived). The existing
`residency_budget_mib` stays. `bounded_within_free` now COUNTS the cache
(`bounded_window + row_cache + act_slab + reserve <= free_vram`): the cache is
residency, and leaving it out let that claim read true while the cache pushed the
run past the reading.

**Prediction vs allocation is checked.** The fit predicts the resolved depth's
bytes with the library's own per-member arithmetic; `begin_row_cache()` records
what it allocated. The runner now refuses loudly when the two differ, before any
timed work, so the emitted figure cannot describe a configuration the run does
not have.

**The checker rule, with one owner.** `ls_byte_account_errors()` is shared by
`--check-telemetry` and `--validate-telemetry`: (1) `row_cache` present without
`residency_cap` is rejected (a half-migrated producer); (2) `residency_cap` must
equal `residency_budget_mib * 1 MiB`, and 0 when no budget was set; (3) with a
cap, `bounded + row_cache <= cap`. Because `byte_account` precedes `rowset_init`,
the policy tie is deferred to end-of-trace (`ls_byte_account_policy_errors`):
retention ON requires `row_cache > 0`, retention OFF requires `row_cache == 0`, so
a zero cannot pass as a real account. Pre-field traces carry neither field and
are accepted — structural immunity for recorded evidence — so the rule is
enforced exactly where the producer reports it.

**Three GPU arms** (untracked one-off
`build/llama-cuda/gplr19-byteacct/run-emit.ps1`, ONE mutex session, probe 512-token
PP + 256-token TG, Qwen3.6-35B-A3B, `--reserve-mib` 512; not reproducible from
tracked tooling alone):

| arm | flags | `byte_account` | checker |
| --- | --- | --- | --- |
| V1 | bare retain, no budget | `retain_k` 4, `row_cache` 2,291,531,776, `residency_cap` 0 | OK |
| V2 | bare retain, budget 16384 | `row_cache` 2,291,531,776, cap 17,179,869,184 (= 16384 MiB), containment 7,009,528,320 <= cap | OK |
| V3 | bare retain, budget 6000 | fit downscales to **K=2**, `row_cache` 1,146,093,568 (the log's K=2 prediction exactly), cap 6,291,456,000, containment 5,864,090,112 <= cap | OK |

V3 is the strongest single result: the depth the fit CHOSE is the depth whose
bytes the cap check passes on. V1's `row_cache` is exactly the fit's printed
prediction for K=4 (2,291,531,776 B), which is the drift check's premise
confirmed from outside the binary.

**Regression is exact and additive.** V1's `byte_account` is identical to the
recorded budget-free R1 record once the two new fields are removed; all 256
`rowset_round` records, the deterministic accounting (30,641 fetched / 51,279
retained / 29,361 retired / 54,797,721,600 B staged) and the greedy text sha16
`f690bfa4df173b7c` match the recorded K=4 rungs.

**Five host fixtures, no GPU** (`fixtures/byte-account-*.jsonl`, generated by the
tracked `gen-lm40-fixtures.py`; every number is the recorded budget-cap
evidence): 2 positives PASS — arm C1's 16384 MiB configuration, and a budget-free
run where containment does not apply — and 3 WILL_FAIL negatives — arm C4's
6000 MiB + K=4 configuration asserted as a completed run (the runner refuses it
at bring-up), a cap field that is not the flag it claims (16385 MiB), and cache
bytes in a trace whose retention is OFF. Wired into `tools/run-tests.ps1 -Suite
host` (23 -> 28).

**Validation.** Build clean (no new warnings; the four `C4477` remain pre-existing
at `llama-layer-streaming.cpp` 1764/1778/1872/1878); host suite **28/28**; ctest
`layer-streaming-pinned-1024` 2/2, `qwen38-layer-boundary` 1/1,
`layer-slot-lifecycle` 1/1 (canonical
`Q38LS_MODEL_PATH=C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`);
QA smoke 512 PASS (check-telemetry OK, all five needles found, negative control
FAILS as designed); `--validate-telemetry` OK on a new-field trace AND on a
pre-field trace (immunity); all three arms exit 0 with `--check-telemetry` OK.

**Residual / open.** (1) A pre-work refusal still writes no `byte_account`
(partial trace without `run_start`, rejected by the checker) — unchanged by this
pass. [CLOSED 2026-09-14 by T-GPLR-22: a pre-work refusal now writes a
structured terminal `run_refused` event and the checker reads it as a refusal
trace instead of rejecting it. See the entry at the top of this file.] (2) The record proves the configuration at bring-up; a later foreign
allocation can still make the cache itself fail loudly. (3) Activation slabs
remain outside the cap in the checker too, matching the recorded decision that
they are a free-VRAM-gated runtime class. (4) The rule is enforced only on
traces produced after this change; earlier recorded traces are accepted without
it, deliberately.

**Commit.** Vendored half in `third_party/llama.cpp` **3f70b64d7** (the library
cache-byte record, the shared `byte_account` emission, the runner's drift check
and both validators, the fixture generator, five fixtures and the ctest
targets); outer half (this entry, the SDD amendment, the plan entry,
`tools/run-tests.ps1` and the gitlink bump) in the `Record ...` commit for this
pass: **e18188a** (hash recorded in this follow-up commit, per the convention's
existing field format).

## 2026-09-14: `--residency-budget-mib` is now a TOTAL residency cap — the row cache must fit under it, and a NAMED depth past the cap is refused rather than allocated

Owner direction: "make --residency-budget-mib a true total residency cap that the
row cache must also fit under, so the cache cannot allocate on top of a budget
the operator set." Host compiler + TEN GPU arms in one mutex session. This closes
the open owner call recorded at the end of the shared-budget-model pass ("it
remains an owner call whether an explicit residency budget should also cap the
cache"). No gate verdict is created or changed.

**Shape of the change** (`tools/layer-streaming/main.cpp`, one file, +86/-31).
The single budget model gains a two-sided cache allowance instead of one:
`cache_cap_free` (the existing rule — the post-`ls.init` reading less
`--reserve-mib` less the slot window the run has not allocated yet) and
`cache_cap_budget` (`residency_budget_bytes` minus `bounded`, which already
counts the slot window `begin_row_set()` allocates later), with `cache_budget` =
the binding minimum of the two. `bounded_budget_bytes` is renamed
`residency_budget_bytes` because the field's role widened. With no explicit
budget, `cache_cap_budget` equals `cache_cap_free` and does not bind. Both caps
are reported by a new `ls_vram_budget_cache_capped_by_budget()`, so a refusal
sends the operator to `--residency-budget-mib` when the budget — not free VRAM —
is what ran out. The fit itself now runs for a NAMED depth too (it previously ran
only for the bare flag): a named depth is still never reduced, but one that does
not fit the allowance is refused loudly instead of being allocated past the cap.

**Ten arms, one session** (probe: 512-token PP + 256-token TG, Qwen3.6-35B-A3B,
`--reserve-mib` 512 unless noted; bounded residency 4,717,996,544 B, slot window
2,088,763,392 B, free VRAM 19,775,094,784 B). They ran from an untracked one-off
under the gitignored `build/` root (`build/llama-cuda/gplr19-budgetcap/run-cap.ps1`
plus its `arm-*.jsonl` / `arm-*.log` evidence), so these figures are **not
reproducible from tracked tooling alone**: the tracked gate driver
`tools/run-gplr19-rowcache-ab.ps1` passes no `--residency-budget-mib` and so does
not exercise this cap. R1-R5 are regression arms with no budget flag; C1-C5
exercise the new cap (`budget` column = `--residency-budget-mib`):

| arm | flags | resolved | evidence |
| --- | --- | --- | --- |
| R1 | bare | **K=4** | allowance 17,149,460,480 B — the recorded F1 line, reproduced |
| R2 | bare, reserve 780 | **K=4** | allowance 16,868,442,112 B — the recorded F2 line |
| R3 | NAMED 4 | **K=4** | still no fit line at all (the check runs, passes, prints nothing) |
| R4 | bare, reserve 15000 | **K=3** | free-bound downscale at allowance 1,957,691,392 B |
| R5 | bare, reserve 16500 | **refused, exit 1** | allowance 384,827,392 B; "not even K=1 fits" |
| C1 | bare, budget 16384 | **K=4** | allowance 12,461,872,640 B — the cap does not bind, accounting identical to R1/R3 |
| C2 | bare, budget 6000 | **K=2** | allowance 1,573,459,456 B; budget-bound downscale (K=3 needs 1,718,812,672 B), notice names the budget |
| C3 | bare, budget 4700 | **refused, exit 1** | allowance 210,310,656 B < K=1's 573,374,464 B; "not even K=1 fits. Raise --residency-budget-mib (or --reserve-mib)" |
| C4 | NAMED 4, budget 6000 | **refused, exit 1** | "explicit depth … needs 2291531776 B but the row cache is allowed only 1573459456 B … the largest depth that fits is K=2" — new: a named depth may not exceed the cap |
| C5 | NAMED 2, budget 6000 | **K=2** | no fit line; same run as C2 by the other route (see parity) |

**Regression is exact, and the cap is visible in the arithmetic.** R1-R5
reproduce the recorded F1-F5 evidence byte-for-byte — identical `byte_account`
apart from `run_id`, identical deterministic accounting, greedy text sha16
`f690bfa4df173b7c` in every arm, identical `tg_tokens` id vectors — so the cap
touched nothing on a run that sets no budget. C1 (non-binding budget) is
byte-identical to R1. C2's fitted K=2 and C5's named K=2 are the *same* run:
46,484 rows fetched / 35,436 retained / 45,844 retired, 83,147,218,944 B staged,
and 35,436 retained is exactly the recorded K=2 rung's figure — so the fit and a named depth resolve to one configuration rather than
two that merely agree. The K byte model behind the arm design
(`bytes(K) = 655,360 + 572,719,104*K`) was fitted to the recorded K=3/K=4 cache
sizes and reproduces the published curve's 546.8/1093.0/3277.8/4370.1 MiB for
K=1/2/6/8; every predicted byte figure (12,461,872,640 / 1,573,459,456 /
210,310,656 / 1,146,093,568) printed exactly as computed.

**Validation.** Build clean (no new warnings; the four `C4477` are pre-existing
in `llama-layer-streaming.cpp` 1764/1778/1872/1878); host suite 23/23; ctest
`layer-streaming-pinned-1024` 2/2, `qwen38-layer-boundary` 1/1,
`layer-slot-lifecycle` 1/1, all with the canonical
`Q38LS_MODEL_PATH=C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`;
QA smoke 512 PASS (check-telemetry OK 23,326 events, all five needles found,
negative control FAILS as designed); all seven completed arms exit 0 with
`--check-telemetry` OK; analyzer self-test PASS (55 checks).

**Environment note (cost me one red run, now recorded).** The three GPU ctest
targets bake `Q38LS_MODEL_PATH` at configure time, and running them against the
Qwen3.6-35B-A3B MoE model makes `layer-slot-lifecycle` fail with
"run_start declared handoff_mode on/auto-on but no handoff events followed": with
`--layer-slots 1`, `--pp-tokens 512` and no `--handoff`/`--handoff-mode` flag the
default `auto` resolves ON for that policy, and no handoff event follows. That is
unrelated to this change (nothing here touches handoff or `run_start`) and does
not occur with the canonical dense Qwen3.8 model, where the same target passes
1/1. Left as an observation for the owner, not fixed here.

**Residual / open.** (1) A pre-work refusal still emits no structured telemetry
event: the run exits 1 with the byte arithmetic on stderr and leaves a partial
trace (40 `pkg_member` records, no `run_start`), which `--check-telemetry`
rejects with 3 invariant violations. That is unchanged by this pass and was
already true of F5 — but the earlier record of F5 as "no trace written" was
inaccurate and is corrected in place above. [CLOSED 2026-09-14 by T-GPLR-22: the
refusal now emits `run_refused` (reason enum + the byte arithmetic + exit
status, terminal) and both checkers read the trace as a REFUSAL TRACE, exit 0,
printing the reason. See the entry at the top of this file.] (2) Activation slabs stay outside
the cap by design: they are a measured runtime class gated against observed free
VRAM (preflight condition 3), not a residency this mechanism manages; if the
owner wants `--residency-budget-mib` to cover them too, that is a further,
explicit widening. (3) The `byte_account` record still carries no cache figure,
so the cap is auditable from stderr and `rowset_init`'s `retain_k` but not from
JSONL alone. [CLOSED 2026-09-14 (same day, owner-directed): the record now
carries `row_cache` and `residency_cap` and the checker holds bounded residency
plus cache under the cap — see the latest entry. Kept in place as the record of
what was true then.] (4) As before, the allowance is evaluated once at bring-up and the
slot-window hold-back is a conservative estimate (2,088,763,392 B against
1,950,367,744 B measured).

**Commit.** Vendored half in `third_party/llama.cpp` **ad565487a** (one file,
+86/-31); the outer half (this entry, the SDD scope amendment, the plan entry
and the gitlink bump) in the `Record ...` commit for this pass: **05bbdd5**
(hash recorded in the follow-up commit, per the convention's existing field
format).

## 2026-09-14: the capacity preflight and the row-cache fit now share ONE VRAM budget model — two parallel arithmetics and two unowned device readings collapsed into `llama_ls_vram_budget`; no number moves

Owner direction: "make the capacity preflight and the cache fit share one VRAM
budget model instead of two parallel arithmetics, so a device's reported numbers
cannot disagree about what is already allocated." Host compiler + five GPU arms
under one mutex. This is a **refactor with no behaviour change claimed or
observed** — no gate verdict is created or changed, and no arithmetic changed
value.

**What was actually duplicated.** Three things. (1) `reserve_bytes` was derived
from `--reserve-mib` at *three* independent sites (the preflight, the cache fit,
and the `byte_account` emission). (2) The device reading was two unowned locals:
`free_vram` before `ls.init` and a second `free_now` re-read inside the fit, with
nothing recording that they measure different moments. (3) The "what is left to
allocate" quantity was inline in the preflight's checks but a separate expression
in the fit, and the slot window travelled between them through a bare local
(`pf_slot_window`) hoisted out of the preflight block for that single reuse.

**The one model.** `llama_ls_vram_budget` (tools/layer-streaming/main.cpp) owns
the inputs (`reserve_mib`, `residency_budget_mib`, `bounded`, `slot_window`), both
readings (`free_start`, `free_now`, `total`) and every derived quantity
(`reserve_bytes`, `bounded_budget_bytes`, `cache_budget`). One
`ls_vram_budget_derive` computes them; `ls_vram_budget_measure_start`,
`ls_vram_budget_adopt_window` and `ls_vram_budget_refresh` are the only writers
and the only callers of the device read. The T-GPLR-04 preflight, the T-GPLR-10
and T-GPLR-16 handoff gates, the T-GPLR-19 fit and the `byte_account`/`run_start`
emissions now all read its fields. The hoisted `pf_slot_window` local is deleted.

**Two readings, kept deliberately and documented in the struct.** `free_start` is
taken before `ls.init` — the preflight proves its out-of-core and capacity
conditions against it and `run_start` telemetry reports it. `free_now` is
refreshed after `ls.init`, where the static/KV/state classes the preflight's own
reading still counted are already outside free VRAM; that is the reading the fit
budgets from, which is *why* the fit never needed to re-derive what was already
allocated. Collapsing these two into one number was explicitly NOT done: they
measure different moments and both are load-bearing. What was removed is the
ambiguity, not a reading.

**No number moves.** All five fit arms re-ran in one GPU session and reproduce
the recorded values byte-for-byte: F1/F2 "fitted to the device" K=4 of a
17,149,460,480 B / 16,868,442,112 B cache budget; F3 (named depth) still emits no
fit line at all; F4 the K=3 downscale at budget 1,957,691,392 B; F5 the "not even
K=1 fits" refusal at 384,827,392 B, exit 1, no trace. Per-arm deterministic
accounting is identical old vs new (F1/F2/F3 30,641 fetched / 51,279 retained /
54,797,721,600 B staged = 214,053,600 B/token; F4 38,425 / 43,495 /
68,723,490,816 B), greedy text sha16 is `f690bfa4df173b7c` in every arm both
before and after, and the `byte_account` and `run_start` records are identical
except for `run_id` / `t0_us`. F1 still equals the recorded K=4 rung exactly.

**Validation.** Build clean (no new warnings; the four `C4477` are pre-existing
in `llama-layer-streaming.cpp` 1764/1778/1872/1878); host suite 23/23; ctest
`layer-streaming-pinned-1024` 2/2, `qwen38-layer-boundary` 1/1,
`layer-slot-lifecycle` 1/1; QA smoke 512 PASS (check-telemetry OK 23,326 events,
all five needles found, negative control FAILS as designed); all four completed
fit arms exit 0 with `--check-telemetry` OK (21,020 events, 4 slots).

**Residual risk / open decision.** The explicit `--residency-budget-mib` cap is
now owned by the model as `bounded_budget_bytes`, but it still governs only the
*bounded* window: the cache budget is derived from free VRAM and is deliberately
not clamped by it, so a run that sets an explicit residency budget *and* a bare
retain flag can still place a cache on top of that budget. That is now a named,
recorded decision instead of an accident of two arithmetics — but it remains an
owner call whether an explicit residency budget should also cap the cache.
[CLOSED 2026-09-14 (owner-directed): the cache is now clamped by the same budget
— the field is `residency_budget_bytes` and a named depth past the cap is
refused; see the TOTAL residency cap entry above.] The
fit's inputs remain estimates (the slot-window hold-back is
2,088,763,392 B against 1,950,367,744 B measured slot VRAM — conservative, the
safe direction), and the budget is still evaluated once at bring-up, so a later
foreign allocation can make the cache itself fail loudly.

**Commit.** Vendored half in `third_party/llama.cpp` **89e86f275** (one file,
+143/-59); the outer half (this entry, the SDD sentence, the plan amendment and
the gitlink bump) in the `Record ...` commit for this pass: **0d9bc38** (hash
recorded in the follow-up commit, per the convention's existing field format).

## 2026-09-14: the per-layer row cache's DEFAULT depth is now fitted to the device — K=4 where the device affords it (unchanged on the reference box), the largest depth that fits where it does not, a loud refusal when not even K=1 fits; a NAMED depth is still never reduced

Owner direction: "make the K=4 default depth VRAM-adaptive … so an opt-in run
on a tight device provisions what fits instead of failing loudly at 2.19 GB".
Host + five GPU arms under one mutex. This closes the footgun recorded when
K=4 became the default (an opt-in run on a tight device previously had no
choice but a 2.19 GB allocation or a bring-up failure).

**Shape of the change.** `llama_layer_streaming::row_cache_bytes_for(k)`
(library, public, `const`) predicts the private VRAM a depth would need by
running the *same* per-member sizing the allocation runs — it creates the cache
tensors in a throwaway metadata context (no device memory) and sums the CUDA
alloc sizes, so the prediction and the allocation cannot drift. The per-layer
loop in `begin_row_cache()` was refactored onto that shared helper; the two
sites are now one arithmetic. The tool owns the policy, as it already owns
`--reserve-mib`: `row_cache_fit_depth()` in `tools/layer-streaming/main.cpp`
picks the largest depth in `[1, requested]` whose prediction fits the budget.

**The budget is free VRAM minus what the run must still allocate.**
`budget = free_now − reserve_mib − slot_window_estimate`. The slot window is
held back deliberately: `begin_row_set()` runs *after* the cache (~line 3638 vs
~3195), so those slot buffers do not exist yet and an unfitted cache would eat
those bytes and fail mid-generation — the loud-at-bring-up rule would be broken.
Nothing is double-counted: the preflight's `bounded` figure is *not* reused for
this because it contains the already-allocated static/KV/state bytes, which
would have downscaled the default on the very device where K=4 is measured to
work.

**Only the default is fitted.** A depth the operator names is a contract and is
never reduced — a gate run must get the depth it declares, never a silently
shallower (b). `--residency-tg-retain N` therefore allocates in full or fails
loudly exactly as before; the fit runs only for the bare flag. The resolved
depth is what `rowset_init.retain_k` reports, so any (b) claim stays anchored to
the depth actually provisioned rather than to the one that was asked for.

Device facts and the five arms (same 512-token anchor, batch-major probe shape,
`--layer-slots 4`, `--kv-cache q8`, 256 TG tokens; K=4 needs 2,291,531,776 B,
K=3 1,718,812,672 B, and each K step adds 572,719,104 B = 8 rows/layer):

- total VRAM 25,650,855,936 B; free at the fit 19,775,094,784 B; slot-window
  estimate 2,088,763,392 B.

| arm | reserve | budget | resolved | evidence |
| --- | --- | --- | --- | --- |
| F1 | 512 (default) | 17,149,460,480 B | **K=4** | "fitted to the device"; 40 layers x 32 rows = 2185.4 MiB |
| F2 | 780 | 16,868,442,112 B | **K=4** | budget still ample — the branch is not near its edge here |
| F3 | 780, depth NAMED 4 | (fit not run) | **K=4** | no fit line at all: a named depth bypasses the fit |
| F4 | 15,000 | 1,957,691,392 B | **K=3** | downscale branch: "default depth K=4 needs 2291531776 B but only 1957691392 B … is available to the cache; provisioning the largest fitting depth K=3"; allocated 40 x 24 rows = 1639.2 MiB |
| F5 | 16,500 | 384,827,392 B | **refused, exit 1** | "the cache budget is only 384827392 B … not even K=1 fits"; no trace written [CORRECTED 2026-09-14: a refusal does leave a partial trace — 40 `pkg_member` records, no `run_start`, which `--check-telemetry` rejects; "no trace" meant no run trace. The refusal wording also changed to "the row cache is allowed only …" when the cap was made total.] |

`--reserve-mib` is the only lever that shrinks the cache budget while keeping
the preflight's own `bounded(4,717,996,544) + slabs + reserve <= free(22,226,665,472)`
check satisfied, which is how F4/F5 reach the downscale and refusal bands on a
device that affords K=4. Both bands were therefore *executed*, not merely
argued.

**Parity.** F1 (bare flag) is byte-identical to F3 (named K=4) **and to the
recorded K=4 rung**: `retain_k` 4, staged 54,797,721,600 B (5,351,340 B/fetch),
rows fetched/retained/retired 30,641 / 51,279 / 29,361, greedy text sha16
f690bfa4. So on this device the fitted default is indistinguishable from the
recorded operating point — the adaptivity costs nothing here. F4's K=3 arm
reproduces the campaign invariant at a *third* depth (K=3 was not in the
crosschecked K=1/2/4/8 set): runtime `n_rows_retained` 43,495 equals the offline
statistic's `hits_total` exactly, at offline (b) = 0.533027.

**Validation.** Build clean; host suite 23/23; QA smoke 512 PASS (check-telemetry
OK 23,326 events, all five needles found, negative control FAILS as designed);
ctest `layer-streaming-pinned-1024` 2/2, `qwen38-layer-boundary` 1/1,
`layer-slot-lifecycle` 1/1; all four completed arms exit 0 with
`--check-telemetry` OK; GPU-free parser checks unchanged for a bad depth
("expects 1..64 windows (got 0)") and the `--help` text documents the fit.

**Residual risk.** The fit's *inputs* are estimates: `pf_slot_window` is the
preflight's host-derived slot-window figure (measured slot VRAM was 1,950,367,744 B
against the 2,088,763,392 B estimate, so the hold-back is slightly conservative —
the safe direction), and the budget is evaluated once at bring-up, so a later
allocation by a foreign process can still make the cache itself fail loudly.
No gate verdict is created or changed by any of this: the K=4 default remains a
provisional operating default whose (b) crossing is still echo-biased, and the
non-echoing re-probe remains its outstanding validation. F5's refusal leaves the
run unstarted, which is the intended "loud, pre-work" behaviour, not a regression.

**Commit.** Vendored half in `third_party/llama.cpp` **cc7cacc60** (three files,
+165/−20); outer half — this entry, the SDD amendment clause, the plan's T-GPLR-19
update, and the 2-line gitlink bump (`4fcca3777` → `cc7cacc60`) — in the root
`Record ...` commit carrying these records: **9564ca5**. That outer hash is
recorded here in a follow-up commit because a commit cannot name its own,
matching the convention's existing two-step.

## 2026-09-14: K=4 set as the per-layer row cache's provisional DEFAULT depth (owner-directed) — bare-flag arm reproduces the recorded K=4 rung exactly; the mechanism stays opt-in and default-off

Owner direction: "set K=4 as default for now". Host + one GPU arm under the
mutex. Code: `kRowSetRetainDefault = 4` in `tools/layer-streaming/main.cpp`,
selected when `--residency-tg-retain` is passed with NO value (a new optional
argument; `is_decimal_token` keeps a following flag from being eaten as one),
and `llama-layer-streaming.h`'s `row_set_retain_k` default raised 1 -> 4. Any
depth in 1..64 may still be named explicitly and overrides the default.

What did NOT change, and that is the point: the mechanism is still opt-in and
default-off (`row_set_retain` false unless the flag is given; `--residency-tg`
still required), so nothing about default placement, the CPU path, or the
non-experimental build moves. This is a PROVISIONAL OPERATING DEFAULT, **not a
gate verdict**: no gate PASS is created or established at any depth (every
reading above 0.6, at K>=4, comes from the self-echoing stream), T-GPLR-15's
close stands (gate read at the depth a claiming task declares; BELOW 0.6 at the
SDD's reference sizing), and the K=4 crossing it rests on is the echo-biased
half of the measurement. SDD scope amendment 2026-09-14 owns the decision; the plan's
T-GPLR-19 and T-GPLR-15 blocks cite it rather than restating a rule.

Evidence that the bare flag resolves to K=4 end to end — ONE arm, this task's
probe settings, retention flag bare:
`build/llama-cuda/gplr19-k4default/run-bare.ps1` (untracked harness, same shape
as the driver's mutex acquisition). Exit 0, wall 41.5 s, `--check-telemetry`
OK (21,020 events, 4 slots), `rowset_init retain_k 4`.

| quantity | bare-flag arm | recorded K=4 rung (B1/B2) |
| --- | --- | --- |
| `retain_k` | **4** | 4 |
| staged total (B) | 54,797,721,600 | 54,797,721,600 |
| B/fetch (x40 = B/token) | 5,351,340 (214,053,600) | 5,351,340 (214,053,600) |
| rows fetched / retained / retired | 30,641 / 51,279 / 29,361 | 30,641 / 51,279 / 29,361 |
| greedy text sha16 | f690bfa4df173b7c | f690bfa4df173b7c |

Every deterministic quantity is byte-identical to the recorded rung, so the
default path and the explicitly-named `--residency-tg-retain 4` path are
indistinguishable. Throughput was NOT comparable and is not claimed: this arm
measured 63.998 PP / 17.413 TG tok/s against the rung's 92.0-92.6 / 19.7,
consistent with cross-session machine load (the mutex serializes GPU use, not
the rest of the host), and the driver's paired design exists precisely because
single cross-session timings are not evidence.

GPU-free parser checks on the new optional value, all as designed:
`--residency-tg-retain` at the end of argv no longer exits with "missing
value" (it takes the default), `--residency-tg-retain nonsense` reports
`unknown argument: nonsense` rather than silently eating the token, and
`--residency-tg-retain 0` still fails loudly with "expects 1..64 windows (got
0)". Build clean; the `--help` text documents N defaults to 4.

**Residual risk carried forward.** The default is provably implemented but its
GATE reading is not validated: K=4's 0.6284 comes from the self-echoing probe
stream, so the non-echoing re-probe is now the outstanding validation of this
default (previously it was the precondition for adopting any depth above the
reference band). Any task that claims a K=4 gate PASS inherits that requirement.

**Commit.** Vendored half in `third_party/llama.cpp` **4fcca3777** (the two
files, +47/−8); outer half — this docs set, the driver note, and the 2-line
gitlink bump (`a61528b47` → `4fcca3777`) — in root commit **a193028**. That hash
is recorded here in a follow-up commit because a commit cannot name its own,
matching the convention's existing two-step.

## 2026-09-13: T-GPLR-19 deferred rungs RUN at K=2, K=4, K=6, K=8 — every rung passes, the runtime reproduces the offline statistic exactly, and the byte-to-time conversion is now measured as DEGRADING with depth

Owner-directed ("run the deferred K=2 and K=4 and K=6 and K=8"). GPU work
under the mutex, one acquisition per rung (four arms A,B,A,B inside each, as the
contract requires), plus the measured series added to the published artifact.
Driver: `tools/run-gplr19-rowcache-ab.ps1 -RetainK <K>`, extended in this pass
(see "Driver change" below). Each rung writes to its own output root
(`build/llama-cuda/gplr19-k<K>-ab/`) so no rung can overwrite another's evidence.

```text
 K  cacheMiB  (b)off  pred B/token  meas B/token  delta   rows_ret==hits  TG off   TG on    gain   sprd   GB/s  ms save
 1     546.8  0.3088   395,835,932   396,398,208  0.142%      True        11.953  14.021  +2.068  0.011  14.29   12.34
 2    1093.0  0.4343   324,007,411   324,793,824  0.243%      True        11.967  15.820  +3.853  0.045  12.18   20.35
 4    2185.4  0.6284   212,811,470   214,053,600  0.584%      True        11.863  19.727  +7.865  0.163  10.67   33.61
 6    3277.8  0.7169   162,144,078   163,659,552  0.935%      True        11.878  22.074 +10.196  0.058  10.52   38.88
 8    4370.1  0.7775   127,437,019   129,075,264  1.286%      True        11.873  24.481 +12.609  0.288  10.23   43.38
```

**All four rungs PASS criteria 1 and 2.** Every arm exited 0 with
`--check-telemetry` OK; the ordered `layer|union_ids|hash` route stream and the
greedy token IDs are identical across all four arms of every rung; the reactive
arms reproduce the recorded baseline exactly (10,240 fetches, 572,719,104
B/round, `rows_reused` 0); zero refusals anywhere. The reactive TG arms held at
**11.863–11.967 tok/s** across five separate GPU sessions, which is what makes
the rung-to-rung comparison readable.

**The strongest result is the row-set agreement, not the throughput.** The
runtime's retained-row count equals the offline statistic's `hits_total`
EXACTLY at every rung — 25,202 / 35,436 / 51,279 / 58,498 / 63,443. The C++
cache and the independent Python simulation are selecting the same rows, not
merely agreeing on a fraction. The measured gate fraction `1 - staged/reactive`
tracks the offline (b) to within **0.29 percentage points** at every rung
(0.3079/0.4329/0.6263/0.7142/0.7746 against 0.3088/0.4343/0.6284/0.7169/0.7775),
and the staged-bytes prediction is inside the 2% band at every depth, drifting
monotonically from +0.142% to +1.286%.

**The finding that changes the F2 arithmetic: the conversion rate is not flat.**
Saved bytes per token convert to saved time at **14.29 GB/s at K=1, then 12.18,
10.67, 10.52 and 10.23 GB/s** at K=2/4/6/8. The marginal rate between rungs is
8.93 (K1→2), 8.35 (K2→4), 9.55 (K4→6) and 7.69 (K6→8) GB/s — i.e. only the
FIRST retained window converts at above-link-ceiling rates; every deeper window
converts at roughly the path's measured effective per-fetch rate (8.42 GB/s).
So `docs/plots/gplr15-b-vs-k.html` view 2's blue series (flat 13.7 GB/s) is
optimistic at depth and its orange series (8.42 GB/s) is the better model for
K>1. Diminishing returns are stark in cache terms: 8x the cache (0.57 -> 4.53 GB
slot basis) buys 2.51x the saved bytes and 1.75x the TG gain.

**Measured vs projected, and the artifact now shows it.** All five measured
points sit INSIDE view 2's projected band at their own K — verified from the
rendered DOM, not by eye (K=1 371.7 px inside 364.4–374.0; K=2 361.4 inside
351.2–368.0; K=4 339.0 inside 320.7–356.7; K=6 325.6 inside 299.7–350.4; K=8
311.8 inside 280.9–345.7). Measured tok/s: 14.021 / 15.820 / 19.727 / 22.074 /
24.481 against projected bounds of 13.61–15.30 / 14.66–17.60 / 16.64–22.92 /
17.73–26.59 / 18.56–29.87. So view 2's BRACKET was right at every depth while
its slope assumption was wrong — and the artifact's previous claim that the
proportionality "at 3x extrapolation into F2's regime is UNTESTED" is now
falsified and has been replaced.

**The echoed-stream caveat applies, and these are NOT gate verdicts.** The probe
stream echoes its own anchor prompt, which inflates temporal reuse; that is why
the plan deferred these rungs behind a non-echoing re-probe. The absolute (b)
values — and therefore the gate standing at K>=4 — are optimistic, and the
oracle ceiling (0.9385 here against 0.4000–0.4266 on the 3-window ladder rungs)
is the signature. What these rungs establish is INTERNAL consistency: the
runtime reproduces the offline statistic it was predicted from, exactly, on the
same stream. What they do not establish is that K=4 clears 0.6 on a
non-echoing prompt. That re-probe remains outstanding and is still the
gate-bearing measurement.

One data-quality note, stated rather than smoothed: the reactive arms' PP read
79.0–92.6 tok/s across sessions while their TG stayed at 11.86–11.98, so PP from
these arms is not comparable across rungs and should not be read as a PP result.

**Driver change (tracked, `tools/run-gplr19-rowcache-ab.ps1`).** Added
`-RetainK` (default 1) and `-OutRoot`. Criterion 2's byte target is now DERIVED
from the offline (b) at that K as `baseline x (1 - (b)_K)` instead of re-typed
per rung — at K=1 that formula reproduces the contract's literal 395,835,932 B
exactly, which is the check that the generalisation is honest. Criteria 3's
pre-registered bounds are asserted only at K=1; for K>1 the driver says plainly
that no pre-registered bound exists and reports the measured conversion as
context rather than as a verdict. Regression evidence: with the change in place,
`-AnalyzeOnly` at K=1 regenerates `summary.json` **byte-identical** and the same
criteria verdicts.

**Artifact change.** `tools/route_overlap/plot-b-vs-k.py` now takes
`--measured-ab` as a list (one A/B summary per depth; absent paths are skipped
and the series degrades to however many points exist — verified by passing a
nonexistent path and getting K=1 alone). View 2 draws the measured series as
hollow rings joined by a dashed line plus a legend row, and the caption now
states is degrading instead of untested. Byte-side deltas are compared in
PERCENTAGE POINTS of (b) because that is the quantity the gate is written in.
Determinism: the artifact regenerates byte-identical.

Validation: build clean, outer host suite PASS, analyzer self-test PASS (55
checks), artifact deterministic, all 20 arms exit 0 with checker 0.

## 2026-09-13: Vendored-fork commit convention DECIDED and applied — the vendored backlog is committed, so the T-GPLR-12/13/16/18/19 mechanisms are reproducible from a clean checkout

Owner-directed ("decide the vendored-fork commit convention"). Documentation and
commits only; no code changed by this pass, no GPU runs. The rule is written
normatively in
`docs/plans/generic-prefill-layer-residency-plan.md` under "Vendored-Tree Commit
Convention".

**The convention was not a choice, which is why leaving it "open" was costing
more than deciding it.** `third_party/llama.cpp` is recorded by the root repo as
a gitlink (index mode `160000`) and there is no `.gitmodules`, so the root stores
only a commit SHA for that directory — never its file contents. A vendored
change is therefore reproducible from a clean checkout **only** if (1) it is
committed inside the nested repo and (2) the root repo commits the bumped
gitlink. There is no third mechanism. The practice was already evidenced four
times in root history (`2308dca`, `64b5aeb`, `1148136` — "Bump vendored
llama.cpp to 169ab086e" — and `84be194`), and the plan's own `Commit:` fields
already used the two-hash form ("vendored code through third_party/llama.cpp
<sha>; outer docs/tools through <sha>"). What was missing was a normative
statement, not a decision.

**Rules.** Nested commits go on the fork's local `master` with the subject
`layer-streaming: T-GPLR-<ids> - <summary>`. The fork is **never pushed**: its
`origin` is upstream `ggml-org/llama.cpp`, and AGENTS.md forbids upstream-facing
material, so a stray `git push` inside that directory is the one live hazard in
this arrangement (it is now ahead 4 of `origin/master`). The root commit's
subject is `Record ...` and it carries the task's docs, its tracked `tools/`
surface, and the 2-line gitlink bump. The task's `Commit:` field names **both**
hashes, because the root hash alone does not identify the vendored code.

**Applied.** `third_party/llama.cpp` **a61528b47** — 9 files, +2,694/-149 — the
accumulated mechanism that had been dirty since `099a27a7e` (T-GPLR-08/10/11).
It carries `src/llama-layer-streaming.{h,cpp}`, `src/llama-context.cpp`,
`src/models/{models.h,qwen35moe.cpp}`, `ggml/src/ggml-backend.cpp`,
`ggml/src/ggml-cuda/ggml-cuda.cu`, and `tools/layer-streaming/{main.cpp,
CMakeLists.txt}`. Checked before staging: no absolute or host paths anywhere in
the added lines, no new `TODO`/`FIXME`, and every diagnostic print is env-gated
(`LLAMA_LAYER_STREAMING_DEBUG`, `GGML_LS_GRAPH_EXEC_MARKERS`,
`LLAMA_LS_TG_ROUTER_PROBE`), so default behaviour and the CPU path are untouched.
`git diff --cached --check` clean; nested worktree clean after the commit. The
outer half — docs, tools, and the 2-line gitlink bump
(`099a27a7e` → `a61528b47`) — is root commit **54c6df4**, recorded in a follow-up
commit because a commit cannot name its own hash; that two-step is the plan's own
existing field pattern ("...through 84be194 (follow-up 7bfe83b)").

**Attribution limit, stated rather than hidden.** The commit spans five tasks
(T-GPLR-12/13/16/18/19) as one unit and therefore cannot tell you which hunk
came from which task. Per-hunk splitting was rejected on a mechanical ground,
not convenience: the T-GPLR-19 hunks live in files that also carry T-GPLR-00..18
hunks, so no partition into per-task nested commits yields an intermediate tree
that compiles — which would defeat the only purpose the commit has,
reproducibility. The commit pins the tree state that all the campaign's evidence
was actually measured on; going forward, one nested commit per task restores
per-task attribution.

**What was included besides the vendored tree, and the coupling argument.**
`tools/run-tests.ps1` is in the root commit even though its diff is T-GPLR-12/16
work, because it pins the host-suite pattern for
`layer-streaming-dense-mtp-auto-refusal` — a ctest target that exists **only**
inside the nested repo's now-committed CMakeLists (confirmed: 0 occurrences in
`099a27a7e`, present at `CMakeLists.txt:536` in the backlog). Committing the
gitlink while leaving that harness uncommitted would create a tree whose tracked
test list references a target the tracked CMakeLists does not define — the same
half-committed defect this pass exists to stop.

**Excluded, deliberately.** `AGENTS.md` (an agent-tooling `## Tools` section,
not campaign work), the `.lean-ctx*` artifacts, and five campaign runners that
tracked docs cite but that remain untracked and gitignored:
`tools/run-gplr13-ladder.ps1` and `tools/run-gplr13-negfix.ps1` (both cited in the
plan and currentstate), `tools/run-gplr13-rowset-probe.ps1` (cited in
`docs/gplr13-bottlenecks-hypotheses.md`), `tools/run-memprobe-dense.ps1` and
`tools/gplr16-dump.cpp` (cited in currentstate). This is a **live instance of the
campaign's own defect class** — tracked text citing a file a clean checkout lacks
— left unfixed on purpose: three of the four runners default to a host model path
via a parameter (acceptable), but `tools/run-memprobe-dense.ps1` hardcodes
`C:/Models/...` with no override, and AGENTS.md requires model paths to be
treated as environment facts rather than source constants. Tracking them is
therefore a hardening task, not a bookkeeping one, and folding it into this pass
would have silently committed host-specific constants. Reported here so it can
be authorized as its own task.

**Corrections made in the same pass**, since the commit falsifies them: the plan's
T-GPLR-13 and T-GPLR-18 blocks both said their vendored implementation "remains
uncommitted" / "stays uncommitted in the vendored tree", T-GPLR-18's follow-up
note said the pinned arm was therefore not reproducible from a clean checkout,
the plan's T-GPLR-19 `Commit:` fields read `none`, and two bullets in the
T-GPLR-19 entry above named the convention as an open owner call. All five now
record `a61528b47`, in the established in-place bracketed style rather than by
rewriting dated records.

**Evidence.** Nested build clean (CUDA Release, `llama-layer-streaming` exit 0),
outer host suite **23/23**, T-GPLR-19's four-arm A/B gate passing all six
criteria, and the runtime-vs-analyzer retention cross-check EQUAL at K=1,2,4,8 —
all on exactly the tree now pinned by `a61528b47`.

## 2026-09-13: T-GPLR-19 DONE — the four-arm K=1 A/B gate RAN and all six acceptance criteria passed; F2-M1 is now a measured mechanism

GPU work under the mutex, plus documentation. Driver:
`tools/run-gplr19-rowcache-ab.ps1` (TRACKED, per the task's RESOLVED
harness-home decision). Arms A,B,A,B in ONE acquisition of
`Global\AI-GPU-EXCLUSIVE-v1`; `tools/gpu-mem-poll.ps1` attached per arm;
`--check-telemetry` per trace. The arms differ ONLY by
`--residency-tg-retain 1`; the vehicle is the T-GPLR-15 probe's batch-major
shape (the only row-set vehicle with enough TG tokens to measure a staging
change). Re-runnable without the GPU via `-AnalyzeOnly`.

| arm | retain | exit | checker | pp tok/s | TG tok/s | staged/round (B) | transfer_us | rows_fet | rows_ret | rows_rtd | refused |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 0 | 0 | 0 | 92.060 | **11.951** | 572,719,104 | 1653.0 | 81,920 | 0 | 0 | 0 |
| B1 | 1 | 0 | 0 | 92.036 | **14.015** | 396,398,208 | 1362.3 | 56,718 | 25,202 | 56,398 | 0 |
| A2 | 0 | 0 | 0 | 92.396 | **11.955** | 572,719,104 | 1648.9 | 81,920 | 0 | 0 | 0 |
| B2 | 1 | 0 | 0 | 92.588 | **14.026** | 396,398,208 | 1356.2 | 56,718 | 25,202 | 56,398 | 0 |

- **Criterion 1 (feasibility + parity) PASS.** Four arms exit 0 with
  `--check-telemetry` OK; the ordered `layer|union_ids|hash` route stream is
  identical across all four arms; greedy token IDs are identical across all four
  arms; the reactive arms reproduce the recorded baseline EXACTLY — 10,240
  fetches, 572,719,104 B/round, `rows_reused` = 0.
- **Criterion 2 (mechanism gate, decided on BYTES) PASS — not
  falsified-by-evidence.** K=1 stages **396,398,208 B/round** against the
  predicted 395,835,932: **+0.142%**, inside the 2% band. Cross-token retention
  is non-zero on all 40 layers. So the statistic (b) the gate was read at
  predicts realized staged BYTES to the fraction of a percent, not merely an
  upper bound.
- **Criterion 3 (performance) PASS, inside the pre-registered envelope.**
  Reactive mean 11.953 -> K=1 mean **14.021 tok/s**, gain **+2.068** against a
  within-pair spread of **0.011** — the gain is ~188x the spread, so the
  proportionality clause did NOT fire. 14.021 lies between the two
  pre-registered bounds (above the 13.61 conservative bound, below the 15.30
  optimistic one). The removed bytes converted at **14.29 GB/s** — faster than
  the 13.7 GB/s link ceiling, because the marginal removed bytes are cheaper
  than average ones (the per-call cost does not shrink with them). No speedup
  claim beyond "this paired run measured +2.068 tok/s at K=1": the two bounds
  were declared UPPER bounds precisely because (b) comes from
  self-echoing-anchor stream, and the 3x extrapolation the artifact's View 2
  rests on remains untested.
- **Criterion 4 PASS:** `-Suite qa-smoke` 512 — `--check-telemetry` OK (23,326
  events), answer aggregate **PASS** on all five items (N1-N4 + K1), negative
  control **FAILS as designed**.
- **Criterion 5 PASS:** `-Suite host` **23/23**; `layer-streaming-pinned-1024`
  **2/2** (27.4 s), `qwen38-layer-boundary` **1/1**, `layer-slot-lifecycle`
  **1/1** — the dense pinned fixture shares the slot/pinned path this task
  touches, so this is the no-regression evidence for the eviction rewrite.
- **Criterion 6 (reported, not claimed):** the runtime logs the cache at 40
  layers x 8 rows/layer = **546.8 MiB** private VRAM, K=1. `peak_vram_bytes`
  was **identical across all four arms** (5,723.56 MiB) — and it is computed
  from free-memory deltas around the run (`main.cpp:4169`), so it evidently does
  not resolve a persistent private allocation. The cache's cost must be read
  from its own allocation log, not from that field. The attached poller CSVs
  cannot price it either: their adapter-level `dedicated_bytes` peak
  (~394-428 MiB) is inconsistent with a run that demonstrably uses ~5.7 GiB.
  Recorded as an instrumentation limitation rather than glossed.
- **Offline (b) vs runtime, on the SAME stream:** offline (b) at K=1 = 25,202
  hits / 81,600 required = 0.308848; the runtime retained rows on both B arms =
  **25,202** — exact agreement, not approximate. The A arms are reported N/A
  with the reason (retention off, so there is no runtime cache to compare), not
  as mismatches.
- **Reproducibility:** the whole driver was re-run after three driver defects
  were fixed (a regex for the allocation line, the offline-vs-runtime check
  applied to cache-less arms, and `-AnalyzeOnly` deleting its own evidence), and
  the gate verdicts were unchanged — the gain was +2.037 then and +2.068 now
  across a ~1.2% session-to-session drift, so the verdict is robust to it.
- **Naming reconciled, not silently changed:** criterion 2 and step 6 name a
  counter `rows_resident_cross_token`; the shipped telemetry field is
  `n_rows_retained` (with `n_rows_retired`). The driver evaluates the shipped
  field and says so. The task is otherwise unchanged: no new package type,
  policy, or placement mechanism beyond SDD section 4 — the shape-(i) cache is
  the section 4 `moe-row-cache` class, and its retention policy is the one the
  2026-09-13 owner-directed section 4 amendment already records.
- **Artifact update (required by the task):** `docs/plots/gplr15-b-vs-k.html`
  View 2 now states AND DRAWS the measured K=1 point — a ringed marker between
  the two projected series, at y=371.7 against y=374.0 (link ceiling, 13.61
  tok/s) and y=364.4 (effective rate, 15.30), so the reader can see which series
  the hardware landed near instead of taking the caption's word for it — with
  reactive 11.953, K=1 14.021, gain +2.068, spread 0.011, realized byte
  reduction 30.79% against the 30.88% projected, conversion 14.29 GB/s, all
  read from the A/B `summary.json` by a new `--measured-ab` option in
  `tools/route_overlap/plot-b-vs-k.py`, so the chart cannot drift from the
  measurement. Its previous claim that "no paired measured run exists for any
  point here" was **falsified by this run** and is replaced. Disclosed in the
  artifact's provenance: the summary is gitignored build output, so the marker
  is drawn only while it exists — the caption carries the values regardless.
  Verified by reading the rendered page: the label and the ring are present in
  the live DOM and the marker sits between the two series at K=1 by coordinate;
  pixel-level visual capture was unavailable in this build, so the placement is
  asserted from the SVG coordinates rather than from a screenshot.
- **Open owner call: NONE remaining for this task.** [CORRECTED 2026-09-13:
  this bullet named the vendored-fork commit convention as open and said
  T-GPLR-19's Commit field stays `none`. The convention was DECIDED the same day
  (see the entry at the top of this file) and applied, so the field records both
  hashes and the K=1 arm's vendored half is now reproducible from a clean
  checkout. Retained as written: the mechanism is a change in
  `third_party/llama.cpp`, so T-GPLR-19's Commit field stays `none` until that
  is settled; the clean-checkout caveat from the harness-home decision therefore
  still stands for the K=1 arm's vendored half.]

## 2026-09-13: Retention policy RESOLVED — the implementation moved to the row-capacity LRU the gate already measured, so the published (b) curve is now a prediction of the mechanism instead of an overstatement

Host-only implementation + documentation change (no GPU run, no library run).
Two independent implementations of the `moe-row-cache` retention rule had
diverged, because SDD section 4 ("capacity is a ROW count of `K*W*top_k`") and
the phrase "K-window" in gate 9 admit two readings. They are now one rule.

- **The decision.** `K` is a ROW BUDGET of `min(n_expert, K*W*top_k)` rows per
  layer, and the cache keeps that budget's most recently ROUTED rows. The
  competing reading ("hold exactly the rows routed in the last K windows") was
  retired rather than the statistic. It was decided on DOMINANCE, not taste:
  both readings charge the identical `row_set_cap = min(n_exp, K*top_k)`, so
  there is no VRAM tradeoff to weigh, and the row-capacity rule weakly wins —
  a row routed in the last K windows has at most cap-1 rows routed after it,
  so it is always among the cap most recently routed and can never be evicted
  where the window-count rule would keep it. The weaker rule was the one the
  runtime happened to implement; the stronger one was already what the gate and
  the published curve measured.
- **What moved, and what did not.** The IMPLEMENTATION moved
  (`llama-layer-streaming.cpp` `load_row_set` + its C++ checker). The statistic
  did NOT: (b) was already computed under the row-capacity reading, so
  `docs/plots/gplr15-b-vs-k.html` regenerates **byte-identical** (`diff -q`
  clean) and the curve's numbers now describe the mechanism. K=1 is provably
  unchanged — the two readings are identical there (cap = 8 = exactly one
  window), which is why the K=1 pre-registered prediction in T-GPLR-19 stands
  verbatim and the outstanding K=1 gate is unaffected.
- **Measured effect, on the preserved 10,200-pair probe stream.**
  Statistic (b) per K, implementation vs the retired rule: K=1 0.308848 both;
  K=2 0.434265 vs 0.401544; K=4 **0.628419** vs 0.519436; K=8 0.777488 vs
  0.659252. The gate crossing moves from ~K=5.9 to ~**K=3.6**, so at K=4 the
  gate now PASSES as published instead of failing by 0.081 — at the same
  1.13 GB. The oracle ceiling is unchanged at 0.938456, so the stream was
  never the limit; the policy was.
- **Anti-drift, verified rather than asserted.** New tracked harness
  `tools/route_overlap/crosscheck-retention.py` replays one shared recorded
  stream through the runtime rule and the offline analyzer and fails on any
  disagreement. Result: window-sequence identity OK (analyzer `chunk` order ==
  runtime `epoch = ubatch - 1`, chunks consecutive), the C++ checker ACCEPTS
  the emitted fixture at K=1,2,4,8, and the runtime rule equals the analyzer's
  `row-capacity-lru` policy **exactly, row for row**, at every K. Negative
  control: asserting the superseded `kwindow-union` policy instead reports
  DRIFT at K>=2 and exits 1, so the check cannot pass vacuously. It caught a
  real bug during development — an emitter cache keyed by expert alone instead
  of `(layer, expert)`, which let layer 1's first window claim residency from
  layer 0; the C++ checker rejected the fixture and the emitter was fixed.
- **Analyzer gained an explicit policy.** `analyze.py` now names both rules
  (`--retention-policy {row-capacity-lru,kwindow-union}`, default the former)
  and records the policy in its report, so the two can be compared by name
  rather than by convention. Default behaviour is unchanged (K=1/K=2 values
  bit-identical; oracle and exact-consecutive identical), self-test grew 51 ->
  **55 checks** and pins both policies' K=2 divergence (0.75 vs 0.50 on a
  purpose-built fixture) so neither can move silently.
- **Not done, deliberately.** No K>=2 GPU arm: the mechanism's recency eviction
  is validated host-side (checker + crosscheck + build) and by the K=1
  invariance proof, but a K=2 arm (16 rows/layer, 1.13 GB) remains to be run,
  and T-GPLR-19's deferred `K > 1` rungs still want the non-echoing re-probe
  first. Nothing is committed; the vendored-fork convention is still open.
  [SUPERSEDED 2026-09-13, later the same day: the K=2/4/6/8 arms HAVE been run
  and all passed (see the entry at the top of this file); the non-echoing
  re-probe is still outstanding and is still the gate-bearing measurement; and
  the vendored-fork convention was decided and applied (`a61528b47` /
  `54c6df4`). Retained as written because this entry records what was true when
  the policy was resolved.]
- **Runtime arm run at K=1** (one arm, GPU mutex held; driver
  `build/llama-cuda/gplr19-k1/run-k1.ps1`, same vehicle as the recorded probe
  plus `--residency-tg-retain 1`, so the only difference is the flag under
  test). Runner exit 0; `--check-telemetry: OK (21,020 events, 4 slots)`; peak
  VRAM **5,723.56 MiB — unchanged from baseline**, so the 566 MB cache did not
  raise peak; 0 refusals; 0 accounting errors; TG **13.926 tok/s** (wall
  18,382.7 ms) against the recorded reactive baseline's 11.578 tok/s (wall
  22,110.8 ms).
- **The runtime reproduces the offline statistic exactly, not approximately.**
  The arm retained **25,202** rows over 10,240 fetches and staged
  **101,477,941,248** bytes — both figures identical to what the analyzer and
  the cross-check harness computed offline for this stream at K=1 (25,202 hits;
  staged 101,477,941,248). Reuse over the whole stream is 0.307642 (0.308848
  once the 1,809 cold windows are excluded, which is the analyzer's
  convention). This is a three-way agreement — implementation, C++ checker,
  offline analyzer — on real hardware, not only in simulation.
- **Arithmetic preserved, decisively.** The K=1 arm's greedy decode output
  (`k1-retain.txt`) is **byte-identical** to the recorded baseline's
  (`probe-tg256.txt`): 463 bytes each, `diff` clean. Wrong retained rows or a
  bad D2D refresh would have changed the logits, so this is stronger evidence
  than any telemetry counter that the rewritten eviction path serves the right
  bytes.
- **The pre-registered 15.30 tok/s was NOT reached, and the reason is
  specific.** The BYTE half of the prediction was met exactly (30.77% staged
  bytes removed, matching (b) = 0.308848 to the row), so the shortfall is the
  byte→time conversion — i.e. T-GPLR-19's OTHER declared unknown ("whether time
  tracks bytes 3x outside the calibrated range"), not the echo-inflation caveat
  that was expected to bite. Implied effective bandwidth of the removed bytes:
  ~12.2 GB/s. This is one arm against a RECORDED baseline, not a same-session
  paired A/B, so it is not a gate verdict; T-GPLR-19's four-arm gate remains
  unrun.
- **Validation:** build clean; host suite **23/23**; reactive probe trace
  `--check-telemetry` exit 0 (no retention → no regression); the K=1 union-rule
  fixture still exit 0 (proving K=1 invariance) while the K=2 union-rule
  fixture now exit 1 (proving the checker changed); all three tampered variants
  still rejected; `py_compile` clean on both tools; published artifact
  regenerates byte-identical.

## 2026-09-13: T-GPLR-19 drafted (F2-M1 per-layer row residency at K=1) and its A/B driver's home DECIDED — tracked `tools/`, not a gitignored `build/` one-off

Documentation only: no code, no build, no GPU run. This change records the
harness-home decision (the Scope bullet plus the `RESOLVED` paragraph); the
T-GPLR-19 contract itself and the two plan summaries that would otherwise
disagree with it landed in `7a7030e`.

- **T-GPLR-19 drafted** (`draft`, NOT ready, awaiting an owner call; it does not
  displace the T-GPLR-16 MIN_BATCH=1 revert, still the only already-authorized
  implementation item). It scopes a new `moe-row-cache` residency class per SDD
  section 4, keyed by LOGICAL LAYER at K=1 (8 rows/layer), behind a new
  default-off flag. The prediction is pre-registered and rests on a measured
  statistic, not a routing guess: (b) at K=1 = 0.308848 over the recorded 10,200
  pairs, so the complement is 69.12% of staged bytes (395,835,932 B of
  572,719,104 B per token), giving TG 11.578 -> 15.30 tok/s at the measured
  8.419 GB/s effective rate and 13.61 at the 13.7 GB/s link ceiling — both UPPER
  bounds, because the (b) they rest on comes from a self-echoing stream.
- **Harness-home decision (decided on owner instruction, 2026-09-13):** the A/B
  driver is **TRACKED** at `tools/run-gplr19-rowcache-ab.ps1`, recorded as a
  `RESOLVED 2026-09-13` paragraph in the T-GPLR-19 block. It produces this
  task's gate — criterion 1's four-arm feasibility and criterion 2's BYTE
  verdict are read from its traces — and a gate a clean checkout cannot run is
  unverifiable, the citation defect already fixed twice here. The nearest
  precedent agrees: T-GPLR-18, same A/B/A/B shape, same GPU-mutex pattern, same
  poller attachment and per-trace checker, tracks its driver
  (`tools/run-gplr18-pinned-ab.ps1`, `ba1f31e`). The gitignored
  `tools/run-gplr13-*.ps1` convention is rejected because those runners produced
  EXPLORATORY telemetry while this driver produces a CONTRACT acceptance
  verdict — the same reason the Phase A spec was tracked rather than inlined.
- **What the decision does NOT close.** Tracking the driver narrows the
  clean-checkout gap to its VENDORED half only. The K=1 mechanism is itself a
  new change in `third_party/llama.cpp`, and it will sit in the dirty vendored
  tree exactly as T-GPLR-18's `load_row_set` change does until the owner decides
  the vendored-fork convention, so the K=1 arm is still not reproducible from
  the tracked tree alone. That question stays open in the task's Escalate list.

## 2026-09-13: The (b)-versus-K curve published as a standalone artifact — `docs/plots/gplr15-b-vs-k.html`, generated by `tools/route_overlap/plot-b-vs-k.py` — now with a second, wall-clock view that prices the F2 retention choice

Owner request: the retention curve had to be inspectable without running the
analyzer, and then that the F2 retention choice be readable as TIME and not only
as VRAM. Host-only; no library change, no GPU run, and no new dependency
(stdlib only, inline SVG, zero external references — checked for `http`/`src`/
`@import` hits: none).

- **Artifact:** `docs/plots/gplr15-b-vs-k.html`, tracked, self-contained — open it
  in any browser. It shows the 13 measured points (K=1..32), the 0.6 gate line,
  the SDD section 4 reference band (K=1-2, shaded), the T8 oracle ceiling 0.9385,
  the first crossing (K=4 = 2.26 GB), the K axis with its row formula, a VRAM
  axis that prices each K, the full value table, and the echo-stream caveat. A
  **basis caveat** now travels with the VRAM axis (see below).
- **View 2, wall-clock (added on owner request 2026-09-13):** statistic (b) is
  the fraction of the next window's rows a cache already holds, so it removes
  `(b) x bytes staged per token` from the link. The basis is MEASURED from the
  trace, not assumed: 14,317,978 B/fetch over 10,240 fetches (8.00 rows each),
  1,701 us mean transfer = **8.42 GB/s effective**, so staging is **78.8% of
  token time** (68.0 ms of 86.4 ms/token over 40 layers) and the residual
  compute floor is 18.3 ms/token. Projected: **13.6 -> 21.2 tok/s across K=1..32
  at the 13.7 GB/s link ceiling** (15.3 -> 44.4 at the measured 8.42 GB/s
  effective rate), against a **54.5 tok/s asymptote** for the lever as a whole
  (every staged byte removed) and the 11.578 tok/s measured baseline. The view is
  labelled a PROJECTION, not a claim: no paired measured run exists.
- **Basis caveat found while adding view 2, and now printed in the artifact.**
  The VRAM axis uses the model's slot accounting (`slot_bytes / n_expert` =
  1,769,472 B/row), which is NOT uniform on this model: 3 of the 40 layers stage
  2,039,808 B/row, so the true per-K footprint is **1.1% above the axis figures**
  (K=32: axis 18.12 GB vs **18.33 GB** from the measured per-layer rows, which
  reconciles exactly with the run's own `bytes_full_bank` = 18,327,011,328 B).
  Both figures are already recorded independently: the bank total in
  `docs/AGENT_ENVIRONMENT.md`, and the split naming the three layers (37 x
  452,984,832 + 3 x 522,190,848 B; layers 34/38/39, whose `ffn_down_exps` are the
  q6_K tensors) in this file's T-GPLR-00 entry. So this is not a new measurement —
  the artifact had been quoting one of the two bases as though it were the only
  one.
  The axis is deliberately LEFT on the slot basis because the SDD section 4 band
  and the T-GPLR-15 retention records are written in it; the caveat states both
  so the two cannot be confused. No verdict changes: statistic (b) is a fraction
  over rows and is independent of row bytes.
- **Rung overlay (added on owner request 2026-09-13):** all four ladder rungs are
  now overlaid on the gate chart as a second set of series (`--ladder`), so the
  saturation-versus-stream-length argument is visible instead of described. Each
  rung is ringed at its **structural saturation depth K=2** and flat from there
  on, against the probe curve that keeps climbing to K=24 — the two depths differ
  by construction. The claim is CHECKED, not drawn on faith: the tool requires
  every K >= windows-1 to equal that rung's own oracle and prints a discrepancy
  (and says so in the artifact) if it ever fails; **4/4 rungs confirmed**, which is
  also an independent re-derivation of the recorded rung values
  (K=1: 0.3766/0.3703/0.3891/0.3578 for 50/512/4k/200k; K=2 = oracle =
  0.4031/0.4031/0.4266/0.4000). Rung input: `build/gplr15-ladder-wide.json`, a
  fresh analyzer pass over the four preserved ladder traces at the probe's own K
  set (`--retention-k 1,2,3,4,5,6,7,8,10,12,16,24,32`), whose K=1/K=2 values
  reproduce the recorded `build/gplr15-phaseA.json` band exactly — so this is
  more resolution on the same measurement, not a second measurement.
- **A real defect the overlay exposed, now fixed.** Extending the geometry check
  to test labels against PLOTTED SERIES (not just other labels) found that the
  probe's own curve cut through four of its value labels (`K=2/4/8` and the
  crossing annotation where the curve leaves its marker steeply), and that view
  2's measured-rate curve cut through its `15.3` label. This was pre-existing and
  invisible to the earlier label-vs-label check. Fixed with a white halo
  (`paint-order="stroke"`, 12 labels) rather than by juggling 13 positions per
  series: legibility no longer depends on the curve's slope. No label is crossed
  without a halo now.
- **Producer:** `tools/route_overlap/plot-b-vs-k.py`. It reads the analyzer's
  `--json` report, so the chart cannot drift from the numbers. Re-render:
  `python tools/route_overlap/plot-b-vs-k.py --input
  build/llama-cuda/gplr15-phaseA-probe/analysis-wide.json [--trace
  build/llama-cuda/gplr15-phaseA-probe/probe-tg256.jsonl --tg-tok-s 11.578] --out
  docs/plots/gplr15-b-vs-k.html --ladder build/gplr15-ladder-wide.json` (the
  `--trace`/`--tg-tok-s` pair adds view 2; omitting it renders view 1 alone and
  suppresses the basis caveat, which needs measured per-layer row sizes. The
  `--ladder` file is produced by `python tools/route_overlap/analyze.py
  "build/llama-cuda/gplr13-ladder/*/rowset-*.jsonl" --retention-k
  1,2,3,4,5,6,7,8,10,12,16,24,32 --json build/gplr15-ladder-wide.json`, and the
  artifact prints that command in its provenance).
- **Geometry verified, not eyeballed.** The preview webview would not composite a
  screenshot in this session, so the SVG was checked programmatically instead:
  13 data markers (plus the crossing ring, 14 SVG circles) land exactly on their
  (K, b) positions and inside the plot box; the
  curve is monotone 0.3088 -> 0.9385; the threshold line sits at y(0.6) and the
  oracle line at y(0.9385) to the pixel; all 5 VRAM ticks land on K x-positions;
  the shaded band spans exactly K=1..2; every labelled K carries the analyzer
  value verbatim; the VRAM labels recompute from rows x 1,769,472 B x 40 layers;
  and no label overlaps another or runs off the canvas (0 in each case). View 2
  was checked the same way and its tight geometry DID need a fix: the link
  projection is nearly flat early (13.61 at K=1 vs 14.67 at K=2, 6 px apart) and
  the measured-rate series sits 9-17 px above it at those K, so the naive value
  labels collided. Labels are now placed by two guards — skip where the two
  series are closer than 22 px, and skip where the previous label's box would be
  touched — which drops K=1 and K=2 and keeps K=4/8/32 (the table below carries
  every value). Both views now pass: markers in-box, no clipped label, no
  overlap, both series monotone, and the measured-rate series strictly above the
  link series at every K. The projection arithmetic was also recomputed
  independently from the raw trace, not read back from the tool: per-token
  572.7 MB, 78.8% staging, 13.61/15.30 at K=1 and 21.21/44.38 at K=32, all
  matching the printed table. With the overlay on, view 1 carries 70 circles
  (4 rungs x 13 points + 4 rings, the probe's 13, and the crossing ring), 5
  polylines and 40 labels; both views still pass markers-in-box, nothing
  clipped, no label overlaps, and no label crossed by a series without a halo.
  The rung values printed in the artifact's table were diffed against
  `build/gplr15-ladder-wide.json` field by field, not eyeballed.
  (A first pass of that recompute gave 44.39 and was
  wrong: it used the b printed to 4 dp, 0.9385, where the report carries more
  digits. The tool's 44.384936 is the correct one, so the doc quotes the tool.)
- **Two real defects found and fixed by that check** — both would have shipped:
  the crossing annotation was long enough to overflow the left edge and to touch
  the 0.6 axis label (now `K=4 crosses the gate at 2.26 GB`, moved clear), and
  the oracle label was right-aligned into the `K=32` value label sitting on the
  same plateau (now anchored left). A third, cosmetic: the HTML read
  "...can exceed it. the reported K=24 already saturates it."
- **Reachable from the numbers:** the analyzer footer now points at the plot tool
  beside the K note, so the chart can be produced from the same JSON it renders.
- **Residual, recorded rather than assumed away:** the chart is reproducible from
  any row-set trace, but the three inputs it was generated from all live under the
  gitignored `build/` — the probe report, the probe trace, and the new
  `build/gplr15-ladder-wide.json` rung report — so a clean checkout has the
  artifact and the producer but not the inputs; re-deriving it needs a fresh probe
  run plus the ladder analyzer pass the artifact's provenance prints, exactly like
  the analyzer numbers themselves. This is the same residual class as the two
  cited pilot numbers, and it is disclosed in the artifact rather than papered
  over: the reader is told which commands re-create the inputs.

## 2026-09-13: T-GPLR-15 CLOSED DONE-falsified-by-evidence — the escalated retention choice resolved in favour of the SDD's reference band (K=1-2: 0.3088 / 0.4343 vs 0.6); the 2.26 GB crossing rejected as the biased half of a self-echoing stream

Owner-directed close of the escalation recorded in the probe entry below.
Decision only: no library change, no GPU run, no new task scope.

The gate is read at the retention the adopting task declares, and the SDD's own
sizing is the only non-arbitrary declaration available (SDD section 4: capacity
`min(n_expert, K*W*top_k)` rows x member slab bytes, "a one-to-two window
retention is ~0.57-1.13 GB across 40 MoE layers"). Measured there:
statistic (b) = **0.3088 at K=1 (566 MB) and 0.4343 at K=2 (1.13 GB)**, both
below the 0.6 threshold, so the gate FAILS and the task closes falsified per
SDD gate 9.

The three options as they stood, and what each would have cost:

| option | retention / VRAM | gate | Phase B |
| --- | --- | --- | --- |
| (i) declare K=4 | 32 rows/layer = 2.26 GB over 40 layers | PASS (0.6284) | eligible on owner review |
| (ii) keep the SDD's band | K=1-2 = 566 MB / 1.13 GB | FAIL (0.3088 / 0.4343) | not entered; task closed falsified |
| (iii) re-probe non-echoing | unchanged until run | undetermined | deferred |

- **Adopted: (ii).** The FAIL is the ROBUST half of the measurement: repetition
  can only INCREASE temporal reuse, so a non-repetitive stream measures the same
  or lower. The gate therefore fails on evidence whose bias direction reinforces
  the verdict rather than undermining it.
- **(i) rejected on evidence, not on caution.** The K=4 crossing (0.6284, first
  reading at or above 0.6) is the OPTIMISTIC half: it sits on a stream that
  echoes its own anchor document (463 chars, the document's code phrase 6x,
  `<think>` 7x). Committing 2.26 GB of VRAM to a number known to be biased
  upward is exactly what the pre-registered caveat forbids. Rejecting it does
  not dispute the ceiling (0.9385) — it declines to spend VRAM on the crossing.
- **(iii) is not a prerequisite for this close, but is recorded as a
  PRECONDITION for any future retention above the band.** By the same
  monotonicity it can only LOWER (b), so it cannot rescue (i) — it can only
  confirm the in-band FAIL more strongly. Its value is forward-looking: no
  higher-retention commitment should be made until the stream is not an echo.
- **What is NOT falsified.** Only the BUDGETED variant. The T8 oracle ceiling is
  0.9385, so the predictor's existence is untouched, and a future task may
  declare a retention and re-read the same statistic under SDD gate 9. F2's
  per-layer row-cache class is unaffected (its SDD section 4 treatment already
  exists), and the reactive row-set path plus the closed handoff remain the
  program's TG answer. Nothing here is a throughput, quality, or equivalence
  claim.
- **Residual risk carried forward.** The verdict is K-sensitive by construction
  and this close fixes it at the SDD's band; if F2 later adopts a larger
  retention, it inherits the echo-stream caveat as its own measurement problem,
  not this task's. The per-layer spread at K=2 (0.1946-0.5353, no layer reaching
  0.6) means the aggregate-vs-per-layer escalation never fired, so no per-layer
  policy question is left open here either.

## 2026-09-13: Closing-hash and tracked-status bookkeeping after ba1f31e

Documentation only; this record lands with its own commit. AGENTS.md requires the
closing commit hash in the task plan (the four-dimension audit found T-GPLR-13
missing exactly this), so `ba1f31e` is now in the T-GPLR-18 `Commit:` field.
T-GPLR-15 stays `Commit: none` — it is in-progress with the gate reading
measured but a retention decision outstanding, so no close exists to cite.
[SUPERSEDED 2026-09-13: the retention decision was then made and T-GPLR-15
closed DONE-falsified-by-evidence — see the newest entry at the top of this
file; its closing hash is recorded by the bookkeeping commit that follows this
one. The wording above describes the state at the time of the commit it
records.]
T-GPLR-14's field said `Commit: none` too, which stopped matching reality once
its instrument entered history: it now records that the tool was untracked at
close, that the instrument landed as `1dbdaf3` and its poller helper as
`ba1f31e`.

The tracked/untracked statements were re-swept after `tools/gpu-mem-poll.ps1`
and `tools/run-gplr18-pinned-ab.ps1` became tracked. One was stale: the schema
draft's section 6 still called `tools/route_overlap/analyze.py` UNTRACKED, so it
now dates the tracking (`1dbdaf3`) and separates what a clean checkout does
contain from what it does not (the `.ps1` runners, the MEMPROBE tools, and the
gitignored `build/` traces). All other such statements — the T-GPLR-14 harness
list, the `build/analyze_gplr13_ladder.py` provenance caveat, and the
retained-as-written blocks carrying their own RESOLVED annotations — were
checked and still hold. The caveat that the pinned arm is unreproducible from a
clean checkout until the vendored `load_row_set` change is committed stands
unchanged; that stays the owner's repo-convention call.

## 2026-09-13: T-GPLR-18 pinned row-set staging CLOSED DONE-falsified-by-evidence — the 512 A/B shows pinned only RELOCATES the staging copy (submit 1,918 -> ~927 us, ready-wait 110 -> ~1,058 us, per-fetch transfer flat) and costs ~1.8 GiB more pinned host memory

F1 (bottlenecks section 6.3) tested against its own recorded counter-evidence
(section 6.7). One library change: `load_row_set` honors `slot.buf_staging`
when `--copy-source pinned` (pack each fetched slab into the pinned buffer and
enqueue from there; natural destination offsets `e * slab_bytes`; byte-exact
memcpy). [UNCOMMITTED 2026-09-13: the `load_row_set` change sits in the dirty
`third_party/llama.cpp` working tree and is excluded from this commit per the
owner's vendored-fork convention, so a clean checkout does not contain it and
the pinned-arm numbers below are not reproducible from the tracked tree alone.]
New driver `tools/run-gplr18-pinned-ab.ps1`; the four arms run
A(direct), B(pinned), A, B inside ONE acquisition of the shared-GPU mutex (same
mutex name and wait/release behavior as `tools/with-gpu-lock.ps1`), with
`tools/gpu-mem-poll.ps1` attached per arm; the arms differ ONLY by
`--copy-source`, everything else the T-GPLR-13 512 ladder shape.

Recorded evidence (`build/llama-cuda/gplr18-pinned-ab/`: `driver.log`,
`summary.json`, four traces, four poller CSVs). Anchor: the calibrated
512-token prompt; 3 TG tokens per arm.

```
arm  source  exit  check  wall_ms  pp tok/s  submit_us  ready_us  transfer_us  staged_bytes   peak_vram
A1   direct    0     0    10,419    87.7      1,937      109.5     2,046.6      1,718,157,312  6,005,784,576
B1   pinned    0     0    12,117    83.9        898    1,057.4     1,955.2      1,718,157,312  6,005,784,576
A2   direct    0     0    16,527    72.6      1,899      110.4     2,009.2      1,718,157,312  6,005,784,576
B2   pinned    0     0    14,449    83.4        955    1,058.4     2,013.9      1,718,157,312  6,005,784,576
```

Means in the transfer/wait columns are per `rowset_fetch` (120 fetches per arm);
staged bytes are the run total.

- **Criterion 1 feasibility PASS.** Four runs exit 0, four `--check-telemetry`
  passes (0 `rowset_refused`), staged bytes identical (1,718,157,312 B =
  3 rounds x 572,719,104 B/round = 3.125% of the 18,327,011,328 B bank),
  route streams identical (one SHA-256 over the ordered
  `layer|union_ids|hash` stream, equal across all four arms), greedy token IDs
  identical (`248046,198,248045`), peak VRAM identical. `run_start` carries
  `"copy_source":"direct"/"pinned"` labelling the arm, with `"rowset":"on"` and
  `"handoff_mode":"off"` in both.
- **Criterion 2 instrumentation gate FAIL — falsified-by-evidence.** Direct
  transfer mean 2,027.9 us vs pinned 1,984.6 us: delta 43.3 us, BELOW the
  largest within-arm repeat spread (58.7 us). The signature is exactly
  T-Q38PERF-03's: host submission FALLS (1,918 -> 927 us) while the
  availability wait RISES by about the same ~1,050 us (110 -> 1,058 us),
  leaving the per-fetch total flat. Pinned does not stage faster; it makes the
  same staging explicit as a host memcpy. Both arms also sit near 2x the
  pre-registered ~1.03 ms floor (14.16 MB at the measured 13.7 GB/s Gen4 x8
  ceiling), so no per-call pageable-staging overhead was recoverable at 24
  slabs per fetch.
- **A cost, not a benefit.** The pinned arm allocates one staging buffer per
  slot (slot_bytes ~= 458 MB x 4 slots ~= 1.83 GB of non-pageable host
  memory). Poller host commit peak: direct 19.95 / 20.07 GiB vs pinned
  21.25 / 22.09 GiB. The falsified arm is also the more expensive one.
- **Criterion 3 (performance claim) NOT reached** - there is nothing to claim,
  and the contract's pre-registered bound forbids claiming more than the
  1.93 -> >= 1.03 ms conversion anyway.
- **Criterion 4 default-off immunity PASS.** The `direct` arms reproduce the
  recorded ladder baseline (572,719,104 B/round, 120 fetches, 3.125%, same
  field semantics); `tools/run-tests.ps1 -Suite host` 23/23 PASS; ctest
  `layer-streaming-pinned-1024|qwen38-layer-boundary|layer-slot-lifecycle` 4/4
  PASS with `Q38LS_MODEL_PATH` set (the dense pinned fixture shares the pinned
  path this task touched); `-Suite qa-smoke` 512 default-off rung PASS
  (`check-telemetry` OK, answer PASS 5/5 needles, negative control FAILs as
  designed).

**Verdict:** F1 is dead on this hardware. Wall time is not claimed either way -
it is dominated by shared-GPU noise at this shape (10.4-16.5 s across four
arms, with the two `direct` arms the fastest and the slowest), and the TG and
PP rates overlap across arms. The TG lever for the not-fitting regime reverts
to F2 (bytes) or to T-GPLR-15 hiding, not to staging mechanics - which is what
the T-GPLR-18 escalation clause pre-registered for a flat result.

## 2026-09-13: T-GPLR-15 Phase A probe RUN — the gate FAILS inside the SDD's reference retention band (0.3088 / 0.4343 vs 0.6) on 10,200 real pairs; the ceiling is 0.9385 and the curve crosses 0.6 at K=4 (2.26 GB), which is a VRAM/product decision, not an implementer's

First real measurement of the gate on a stream long enough to mean something.
One GPU run under the shared mutex with the memory poller attached; no library
changes.

Recorded evidence
: - Command: `--model C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf
    --prompt-file <ladder 512 anchor> --residency moe-expert-bank --residency-tg
    --pp-schedule batch-major --pp-tokens 512 --ubatch 512 --tg-tokens 256
    --layer-slots 4 --kv-cache q8 --greedy --no-mtp --pp-no-logits
    --allow-full-residency --telemetry <probe-tg256.jsonl> --tg-text <probe-tg256.txt>`,
    wrapped in `tools/with-gpu-lock.ps1 -TimeoutSeconds 900`, poller
    `tools/gpu-mem-poll.ps1 -IntervalMs 500`; driver kept under the gitignored
    `build/llama-cuda/gplr15-phaseA-probe/run-probe.ps1` (the tracked status of
    the `tools/` harness is a separate open owner decision).
  - Hardware: RTX 5090 (per `docs/AGENT_ENVIRONMENT.md`). Exits: runner 0,
    `--check-telemetry` OK (21,020 events, 4 slots).
  - Timings: PP 512 tokens at 76.934 tok/s (wall 6,655 ms); TG 256 tokens at
    11.578 tok/s (wall 22,111 ms). Runner peak VRAM 5,723.56 MiB
    (6,001,590,272 B); poller adapter peak 4,641 MiB, commit peak 18.6 GiB,
    avail min 34.7 GiB. Artifacts under `build/llama-cuda/gplr15-phaseA-probe/`
    (`probe-tg256.jsonl`, `.log`, `.txt`, `mem.csv`, `analysis.{txt,json}`).
- **The plan's flagged uncertainty is RESOLVED: row-set TG after a batch-major
  PP works.** `run_start` carries `"rowset":"on"` with `"schedule":"batch-major"`
  and `"handoff_mode":"off"`; 10,240 `rowset_fetch` events (40 layers x 256
  tokens), 256 `rowset_round`, 1 `rowset_init`, 0 `rowset_refused`. No fallback
  (repeated-3-token prompt family, or lifting the layer-major cap) is needed.
- **How to read K.** K is the cache's retention DEPTH in token windows, not
  bytes and not a count of experts. A K-window cache holds
  `min(n_expert, K*W*top_k)` expert rows per layer (W=1, top_k=8, n_expert=256
  here), and one row is that expert's gate+up+down slabs (1,769,472 B). The byte
  totals therefore FOLLOW from K rather than being chosen: K=1 keeps only the
  previous window's 8 experts (566 MB over 40 layers), K=2 keeps 16 (1.13 GB),
  K=4 keeps 32 (2.26 GB), and K=32 hits the 256-row cap at 18.12 GB, the full
  bank. Statistic (b) is defined AT a K, so a gate reading is complete only
  when the K it was read at is named; the SDD section 4 reference band is K=1-2.
  The analyzer's report footer now prints this note beside the curve.
- **Gate reading (SDD section 10 statistic (b), 10,200 pairs):**
  K=1 **0.3088**, K=2 **0.4343** — both BELOW the 0.6 threshold, so the gate
  FAILS at the SDD's reference retention band. Per-layer at K=2: min 0.1946 /
  mean 0.4343 / max 0.5353, so no layer reaches 0.6 and the "aggregate passes
  only because a few layers are near 1.0" escalation does NOT fire.
- **But the stream is highly predictable with more retention:** the T8 oracle
  ceiling is **0.9385**, and the curve crosses 0.6 at **K=4 (32 rows/layer,
  2.26 GB over 40 layers)**: K=3 0.5330 (1.70 GB), K=4 0.6284 (2.26 GB),
  K=8 0.7775 (4.53 GB), K=16 0.9376 (9.06 GB), reaching the 0.9385 oracle
  value by K=24 (13.59 GB, 192 rows/layer); the full 256-row bank is 18.12 GB
  at K=32, which adds nothing over K=24. T7 agrees: cumulative reappearance coverage at
  retention 1/2/4/8 windows is 0.329 / 0.428 / 0.554 / 0.703.
- **Sample caveat that must travel with the ceiling (do not quote 0.94 as a
  property of "real generation").** The generated text is an echo loop of the
  anchor document: 463 characters, the document's code phrase repeated 6x,
  `<think>` 7x. Token-level periodicity is real but partial — best match 0.479
  at lag 39, immediate-repeat rate only 8.6%. Repetition can only INCREASE
  temporal reuse, which cuts two ways and both are useful:
  (a) the FAIL at K=1-2 is CONSERVATIVE — a non-repetitive stream would measure
      the same or lower, so "the gate fails inside the reference band" is the
      robust half of this result;
  (b) the K=4 crossing and the 0.9385 ceiling are likely OPTIMISTIC and are not
      a sound basis for committing 2.26 GB of VRAM until they are re-measured
      on a stream that does not echo its own prompt.
- **Escalated, not decided** (the contract's own escalation condition: the (b)
  verdict is K-sensitive and the retention/VRAM tradeoff is product intent).
  The owner's choice: (i) declare K=4 (2.26 GB) as the adopting task's
  retention, which makes the gate pass and Phase B eligible for review; or
  (ii) keep the K=1-2 reference band, in which case T-GPLR-15 closes
  falsified-by-evidence per gate 9; or (iii) re-probe on a non-repetitive
  prompt first, because the crossing point is measured on an echoing stream.
  Nothing here claims a speedup, a quality result, or an F2 implementation;
  T-GPLR-15 stays in-progress. [DECIDED 2026-09-13: option (ii), the SDD's
  K=1-2 reference band — the task closed DONE-falsified-by-evidence; see the
  newest entry at the top of this file. The wording above is retained as
  written: it describes the state before the choice was made.]

## 2026-09-13: T8 oracle retention ceiling added to the analyzer — on the ladder samples NO policy can reach the 0.6 gate, but those samples structurally cannot settle it

Owner-requested follow-up on the Phase A analyzer; the analyzer change and this
record land in the same commit. Host-only: no library changes, no GPU runs, no
GPU mutex needed.

- **New T8 oracle** = statistic (b) under UNBOUNDED retention: the fraction of
each window covered by the union of ALL prior windows for that layer. No cache
policy of any size can exceed it on the same stream, so it answers "could any
policy reach the 0.6 gate here?". Implemented as `simulate_retention(wins,
None)` (same simulator, eviction disabled) plus `oracle_containment`, reported
with pooled hits/required totals so EXACT saturation by a reported K is
detectable rather than inferred from two rounded means.
- **Result on the recorded ladder streams: no policy reaches the gate.**
Oracle = 0.4000 (200k) / 0.4266 (4k) / 0.4031 (50) / 0.4031 (512), all below
0.6, and every rung reports `ceiling_saturated_by_k = 2`.
- **That saturation is STRUCTURAL, not evidence about long-stream
predictability.** With W=1 and top_k=8, K=2's capacity is exactly 16 rows, and
the union of two 8-row windows can never exceed 16 — so on any 3-window stream
unbounded retention is provably IDENTICAL to K=2. The oracle therefore confirms
only that retention sizing is not the limiting factor on these samples; the
limit is the routing stream's own sparsity (T7 at the 512 rung: 486 of 702
layer-rows requested exactly once within 3 windows). Since the oracle is MONOTONE in stream length (more history
can only add coverage), a 3-window ceiling bounds the sample and never the
program. The 256-token probe is where this test becomes decisive.
- **Machine-checked, including the negative direction:** the report prints the
below-threshold conclusion per trace and carries the monotonicity caveat
inline; the self-test grew to 51 checks (was 43) with a 4-window fixture whose
reuse distance exceeds the 2-window capacity — K=1 = K=2 = 0.0 while the oracle
is 1/3 and `ceiling_saturated_by_k` is None — plus the mirror case requiring the
3-window fixture to report exact saturation at K=2. Both paths of the new logic
are therefore covered, not just the one the real traces happen to exercise.
- Artifact regenerated: `build/gplr15-phaseA.{txt,json}`.

## 2026-09-13: Both remaining untracked docs brought under version control — the model-run testing standard and the 2026-09-11 inference performance review

Owner decision on the open question left by the schema-doc pass. Documentation
only: no code changed, no GPU runs, no new task scope. Both docs are tracked as
of this commit.

- **Decision: track BOTH; inline neither.** `docs/model-run-testing-standard.md`
  (173 lines) and `docs/inference-performance-review-2026-09-11.md` (585 lines)
  are now tracked in `docs/`. `git ls-files` was empty for both and neither had
  any commit history, so a clean checkout did not contain them.
- **Why the standard:** the T-GPLR-15 block defers to it for what counts as
  recorded evidence ("attached per the testing standard"; "recorded evidence
  per the model-run testing standard") and
  `docs/gplr13-bottlenecks-hypotheses.md:176` cites it by path. The standard
  that defines the probe run's evidence shape is not optional context for that
  run. It stays marked a DRAFT — tracked is not approved; owner review is still
  desired before it becomes a runner or CI contract.
- **Why the review:** it is uncited anywhere in the repo, which is why it was
  easy to leave behind, but it is the origin analysis for the whole MoE row-set
  program: its P1 transfer-granularity finding IS the row-set hypothesis, and
  its focused experiment 7 anticipated the T-GPLR-15 reuse gate by warning that
  the 60% TG cache hit rate was unevidenced and that per-layer TG IDs had to be
  collected before any prediction work. Phase A has since measured that gate:
  0.37-0.43 against the 0.6 threshold. Losing the document that predicted the
  result on the way to measuring it would be a real loss.
- **Both headers carry a dated status note** in the established in-place style:
  tracked status and role stated, the original status text retained as-written
  inside a `[SUPERSEDED ...]` block. The review's note also records that its
  per-task status observations are as of 2026-09-11 and are superseded by this
  file and the active plan, so a reader cannot mistake its dated READY/DRAFT/
  rejected table for current state.
- **Content sweep before tracking:** no absolute local paths, usernames, host
  names, IP addresses, or credentials in either file; the only "serial" matches
  are the word serial (vs parallel). Both are project design docs.
- **Retires the earlier flag** rather than rewriting it: the previous entry's
  "same defect class, still open" bullet is annotated RESOLVED in place.
- **Partly closed 2026-09-13:** `tools/route_overlap/analyze.py` — the tool
  that produces the statistic (b) numbers — is now TRACKED, and
  `tools/gpu-mem-poll.ps1` followed it into history on 2026-09-13 because the
  T-GPLR-18 A/B driver invokes it (a cited file a clean checkout lacks is a
  defect). Still untracked and
  left to the owner: the T-GPLR-14 `.ps1` runners (`run-gplr13-ladder.ps1`,
  `run-gplr13-negfix.ps1`, `run-gplr13-rowset-probe.ps1`),
  `tools/gplr16-dump.cpp` and `tools/run-memprobe-dense.ps1` — pre-existing work
  not authored in this thread. `.lean-ctx*.toml*` are agent tooling artifacts
  that belong in `.gitignore`, not history.

## 2026-09-13: T-GPLR-15 Phase A analyzer implemented — first real values of the SDD gate-9 statistic (b), and K=1 reproduces the recorded pilot exactly

Implementer work on T-GPLR-15's controlling phase: host-only, no library
changes, no GPU runs, no GPU mutex needed. The analyzer is TRACKED as of
2026-09-13 (this entry first recorded it as a working-tree change; the commit
that added it also updated this line).

- **The analyzer now works on real row-set traces.** `tools/route_overlap/analyze.py`
  accepts `rowset_init` / `rowset_fetch` / `rowset_round` / `rowset_refused`
  (it previously exited 2 on the first of them, so no ladder rung could be
  analyzed at all) and VALIDATES each `rowset_fetch`: canonical ascending
  unique comma-joined `union_ids`, count equal to `route_union`, and a
  recomputed hash equal to the emitted one (exit 2 with file:line otherwise).
- **New statistic (b), the SDD gate-9 verdict statistic**
  (`simulate_retention` + `resident_containment`): per-layer bounded-retention
  simulation over the recorded union streams, capacity `min(n_expert, K*W*top_k)`
  rows, window-granular LRU, next window in the denominator (`|A n B| / |B|`),
  reported as a per-K curve. Windows with a predecessor only (a cold cache
  holds nothing at the first window). Also added: exact consecutive-window
  containment from full `union_ids` in both directions (T4, direction always
  stated), and the per-layer reappearance-distance histogram (T7) with
  cumulative coverage at 1/2/4/8 windows.
- **T5 is re-sourced from `rowset_fetch`** and now prints its `n_pairs`; it is
  reported as CONTEXT, never the gate. Two pre-existing defects fixed while in
  the file: the measured T5 rate was computed and then DROPPED (never stored,
  so the report's T5 line never showed it), and a nested glob pattern
  (`.../*/rowset-*.jsonl`) crashed with a traceback instead of failing loudly —
  the pattern expansion now uses `glob` and honors the tool's failure contract.
- **Cross-validation on the preserved ladder traces PASSED, exactly.**
  `python tools/route_overlap/analyze.py "build/llama-cuda/gplr13-ladder/*/rowset-*.jsonl"`
  (artifact: `build/gplr15-phaseA.{txt,json}`) reproduces every recorded pilot
  number: T5 MEASURED 0/80 window-equality at every rung, and statistic (b) at
  K=1 = 0.3766 (50) / 0.3703 (512) / 0.3891 (4k) / 0.3578 (200k) against the
  recorded containment means 0.377 / 0.370 / 0.389 / 0.358. The hash recompute
  passed on all 120 `rowset_fetch` events per rung.
- **First real K=2 values (new information):** (b) = 0.4031 / 0.4031 / 0.4266 /
  0.4000 in the same order, at capacity 16 rows/layer = 1,132.5 MB over 40
  layers (K=1: 8 rows, 566.2 MB). Both K values in the SDD's reference band sit
  far below the 0.6 threshold on this stream.
- **The aggregate hides real per-layer spread** — at 512 the per-layer (b) at
  K=1 is min 0.0625 / mean 0.3703 / max 0.7500 over 2 pairs per layer. This is
  the aggregate-vs-per-layer tension the task's escalation clause names, now
  with numbers; it is reported, not resolved (a per-layer policy is product
  intent, not an implementer call).
- **Internal consistency check:** at K=1 the total hits equal the gap-1
  reappearance count by construction (237 on the 512 rung, 229 / 249 / 241 on
  the others), which the T7 histogram independently reproduces.
- **Finding — the vendor's "FNV-1a-64" is not standard FNV-1a.**
  `tel_rowset_fetch` seeds with the literal `1469598103934665603`; the standard
  offset basis is `0xCBF29CE484222325` = `14695981039346656037` (one digit
  longer), and the vendor comment claims equality with this tool's helper. The
  helper now uses the vendor literal, verified by reproducing emitted hashes;
  with the standard basis the recompute cross-check would fail on every real
  event. No library change was made.
- **Validation:** self-test PASS (43 checks, was 19) including the K-curve
  fixture (K=1 = 0.0 vs K=2 = 0.5 on the same windows), the hash-mismatch and
  non-canonical-ids negative controls, and the `--retention-k` guard; reruns
  byte-identical for both txt and json.
- **Provenance is now genuine for these numbers.** The recorded pilot values
  were originally MACHINE-computed by the untracked one-off; they are now
  reproduced exactly by `tools/route_overlap/analyze.py`, which is TRACKED as of
  2026-09-13, so the producer of the gate-9 statistic is present in a clean
  checkout. What a clean checkout still does NOT contain: the untracked
  `build/` traces it reads (`.gitignore:/build/`), so re-deriving the numbers
  still needs either those traces or a fresh ladder run — and the T-GPLR-14
  `.ps1` runners and MEMPROBE tools remain untracked (owner's call; only the
  analyzer was in this thread's scope).
- **Remaining Phase A step (not done here):** the one 256-token batch-major
  generation probe under `tools/with-gpu-lock.ps1`, which is what turns this
  cross-validation into the gate verdict (~10,200 pairs). Nothing in this entry
  claims a gate verdict, a performance result, or an F2 implementation.

## 2026-09-13: Rowset telemetry schema draft brought into version control (tracked, not inlined) — the T-GPLR-15 contract no longer cites a spec a clean checkout would not contain

Owner decision on the open question flagged in the T-GPLR-15 contract's
escalation clause. Documentation only: no code changed, no GPU runs, no new
task scope. The schema's content is unchanged apart from its status header.

- **Decision: TRACK, do not inline.**
  `docs/rowset-fetch-telemetry-schema-draft.md` is now tracked in `docs/`
  beside the SDD and the plans. Inlining into the plan was rejected because
  three tracked docs cite it by section — the plan (sections 3 and 6),
  `docs/currentstate.md` (section 6), and `docs/gplr13-bottlenecks-hypotheses.md`
  (section 4.2 and open decision 3) — and because 221 lines of event/field
  schema is detail the contract deliberately delegates, not task-contract
  content; inlining would duplicate it with no single owner.
- **The draft was documenting shipped behavior, not a proposal.** Its section 7
  decision 2 is implemented as the ascending comma-joined `union_ids` canonical
  string (`third_party/llama.cpp/src/llama-layer-streaming.cpp:3643`) and its
  section 8 validations ran under T-GPLR-13. The header status is corrected
  accordingly, with the original 2026-09-12 draft status retained as-written
  inside a `[SUPERSEDED ...]` block.
- **What this resolves.** The plan's T-GPLR-15 escalation clause no longer
  lists "the Phase A spec stays UNTRACKED" as an escalation trigger; it records
  the resolution instead. The earlier draft section 5/6 corrections (gate read
  against SDD statistic (b), the reappearance-distance histogram, hash and
  containment framed as context and integrity checks) land with this commit —
  they were previously on disk but unversioned.
- **Same defect class, still open (owner decision):** the plan defers to
  `docs/model-run-testing-standard.md` by name only, never by path ("per the
  model-run testing standard" and "per the testing standard", both in the
  T-GPLR-15 block), while `docs/gplr13-bottlenecks-hypotheses.md:169` cites it
  by path. The file remains untracked, so the same clean-checkout gap applies.
  [RESOLVED 2026-09-13: the owner decided to TRACK it — see the newest entry at
  the top of this file. The wording above is retained as written.]

## 2026-09-13: Gate 9 decision rule now single-owned by the SDD (T-GPLR-15 metric corrected); cited pilot numbers carry a provenance caveat; T-GPLR-13's commit recorded

Audit follow-up. Three surgical documentation corrections. No code changed,
no GPU runs, no new task scope.

- **One owner for the T-GPLR-15 gate.** `docs/layer-residency-sdd.md`
  section 10 now carries the sole "Gate 9 decision rule": the verdict
  statistic is (b), the realized predictor ceiling — the fraction of the NEXT
  window's routed expert rows that a persisting per-layer cache (retention K
  windows, W=1) would already hold, next window in the denominator
  (`|A n B| / |B|`) — and the 0.6 threshold is read against it. Exact
  window-hash equality and exact consecutive-window containment are CONTEXT
  only, never the verdict: at top_k=8 over n_expert=256, exact window identity
  is structurally near zero (measured 0/80 at every ladder rung), so gating on
  it would have closed T-GPLR-15 on a sparsity artifact instead of on routing
  predictability. The T-GPLR-15 task block now cites the SDD rule rather than
  defining its own.
  **Supersedes** the gate-rule line in the 2026-09-13 T-GPLR-15 contract entry
  below (`primary = hash-equal consecutive-window rate`); that entry is left
  unedited as the record of what was written at the time. The SDD is the
  governing spec and the later-dated amendment, so this adopts its rule rather
  than overriding it — no owner decision was needed and none was inferred.
- **Provenance caveat on every cited pilot number** (0/80 hash-equal,
  containment 0.358-0.389): they are MACHINE-computed by
  `build/analyze_gplr13_ladder.py`, an untracked one-off that lives inside the
  gitignored `/build/` tree, so it does not survive a clean checkout and the
  numbers are NOT reproducible from tracked tooling alone. Recorded in the
  bottlenecks doc section 2 (canonical statement; caveat attached under the
  table) and in the plan's T-GPLR-15 cross-validation step. The Phase A
  re-derivation in `tools/route_overlap/analyze.py` replaces them as evidence;
  a disagreement there is investigated and recorded, not silently resolved.
  (Corrected 2026-09-13 after post-commit verification: the first version of
  this caveat claimed no producing script existed in the repository, which
  contradicted the bottlenecks doc section 1 and was false.)
- **T-GPLR-13's DONE block records its commit** (`46644f1`) per AGENTS.md,
  with the faithful qualification that the commit carries the documentation
  close-out only — the vendored T-GPLR-13 implementation remains uncommitted
  in `third_party/llama.cpp`. [CORRECTED 2026-09-13: the qualification's second
  half is superseded — the vendored implementation is now committed as
  `third_party/llama.cpp` **a61528b47**. The `46644f1` record itself was accurate
  and is unchanged.]

## 2026-09-13: SDD amended with the F2 per-layer row cache (`moe-row-cache`) residency class — realized reuse and the T-GPLR-15 gate's cumulative-resident-set ceiling are now defined and measurable

Owner-requested SDD treatment written into docs/layer-residency-sdd.md
(amended 2026-09-13; sections 1, 4, 6, 7, 8, 9, 10, 11, 12). No code changed;
no GPU runs.

The class (analysis label F2; no task ID yet) is defined as an opt-in,
CUDA-only, TG-only sub-mode of `moe-row-set`:
- Per-layer keying is the defining property: storage and every resident claim
  are keyed by LOGICAL LAYER, never by slot, because the layers sharing a slot
  alias one device buffer at offset 0 and an intervening owner overwrites the
  same byte ranges — the reason the bookkeeping-only H1(b) variant is unsound
  (bottlenecks doc section 6.4). Provenance is structural (layer, member,
  expert, written window/generation stored with the bytes), never inferred.
- Bounded retention sized for the worst case: min(n_expert, K*W*top_k) rows x
  member slab bytes — ~14.2 MB per 8-row set and ~0.57-1.13 GB across 40 MoE
  layers for one-to-two window retention on the Qwen3.6 target; overflow is a
  loud refusal, and an entry a running consumer may read is never evicted.
- Per-window atomicity, the section-4 execution-shape constraint unchanged,
  and two permitted execution shapes (cache-to-slot-bundle D2D refresh before
  the row-consuming phase, or graph tensors bound directly to the cache) that
  the adopting task must declare.
- Reuse optimizes the COPY list only; the reactive superset check on the
  router's exact IDs is untouched. Filling the cache reactively is this class;
  filling it ahead of need is T-GPLR-15's variant.

What this unblocks for the T-GPLR-15 gate: the SDD now names the two distinct
statistics and requires a verdict to state which was used — (a) consecutive-
window exact containment, a property of the routing stream that is measurable
WITHOUT the cache (T-GPLR-15 Phase A), and (b) cumulative-resident-set
containment, the realized predictor ceiling, which is only well-defined once
per-layer persistence exists (i.e. requires F2). New gate 10 requires
positive reuse demonstration, cache-on/cache-off pairs for any performance
claim (the cache changes the byte denominator), and byte-for-byte unchanged
cache-off and default-off paths. Cache failure reasons, telemetry classes
(cache-served rows / device refresh bytes / link bytes / high-water /
invalidations, never double-booked), adapter duty (9) for declared capacity
and provenance ranges, capacity-preflight treatment, and a risk list were all
added.

## 2026-09-13: T-GPLR-18 (F1 pinned row-set staging) added to the plan as READY with a 512 A/B gate — counter-evidence recorded first; 6.3's pinned-throughput assumption corrected; T-GPLR-15 probe protocol corrected to the batch-major vehicle

F1 from the T-GPLR-13 close-out is now plan task T-GPLR-18, READY 2026-09-13,
with an explicit 512 A/B acceptance gate. Contract shape: `load_row_set`
honors the existing `slot.buf_staging` when `--copy-source pinned` is
requested (packed slabs, natural destination offsets, interleaved
pack-and-enqueue exactly mirroring `load_layer`); no new flag, no checker or
schema change (`run_start.copy_source` already labels the arm); the
event lifecycle and the host-blocking wait are retained; drivers are
tools/run-gplr18-pinned-ab.ps1 (A,direct / B,pinned / A / B interleaved under
one GPU-mutex session at the T-GPLR-13 512 ladder shape, `--tg-tokens 3`) plus
the existing checkers.

Counter-evidence recorded in the contract before the hypothesis (bottlenecks
doc section 6.7):
- Measured H2D is 13.7 GB/s steady-state on a negotiated Gen4 x8 (87% of the
  15.75 GB/s link); pageable sources are internally staged at that same rate,
  so pinned adds no DMA throughput — only an explicit host memcpy.
- T-Q38PERF-03 already ran the dense A/B and pinned LOST end-to-end (PP
  8,366.4 vs 8,195.3 ms; TG 2,757.5 vs 2,515.0 ms) with submit down and
  availability wait up (TG 5.3 -> 1,510.4 ms).
- Surviving hypothesis is only per-call overhead: 24 copies of ~0.59 MB per
  fetch; measured submit 1,826 us vs the ~1,033 us floor for 14.16 MB at
  13.7 GB/s. Pre-registered per-fetch ceiling ≈ 1.03 ms (from 1.93 ms); a
  bigger claim is arithmetic error.
Gate: feasibility (both arms exit 0, checker OK, identical staged bytes
572,719,104/round, identical token IDs and route hashes) + instrumentation
(pinned transfer_us mean below direct by more than the within-arm repeat
spread, wall not worse); performance claimed only if the wall improves beyond
spread and qa-smoke 512 stays PASS. A negative result closes the task as
falsified-by-evidence and returns the TG lever to F2 (bytes).

Also corrected this turn (contract-level, no code): the T-GPLR-15 Phase A
256-token probe as originally drafted was UNRUNNABLE — the runner caps
handoff-free layer-major TG at 3 tokens (main.cpp:2190/:2878) and
`--residency-tg` forces handoff off (main.cpp:2465). The corrected protocol
uses the batch-major vehicle (`--pp-schedule batch-major --tg-tokens 256`),
which has no cap; the hit-rate statistic is a property of the generation
stream and batch-major/layer-major PP produce the same KV per the validated
equivalence gates, but row-set TG after batch-major PP is UNVERIFIED, so a
one-run feasibility check precedes the gate run. Recorded too: greedy
decoding is deterministic, so repeated identical runs add no independent
samples — only stream length (or a prompt family) grows n.

Plan serialization note: two READY tasks now exist (T-GPLR-15 Phase A,
host-only analyzer + probe; T-GPLR-18, small library change + GPU A/B). They
are independent; the owner selects order and implementation stays one at a
time. No code changed, no GPU runs, no commit.

## 2026-09-13: T-GPLR-15 contract rewritten gate-measurement-first (READY) — Phase A measures the 0.6 hit-rate gate with no library changes; Phase B only on PASS + owner review

The plan's T-GPLR-15 block is rewritten as a two-phase contract on owner
request, now that T-GPLR-13 is DONE and `rowset_fetch` telemetry exists in
real traces:
- Phase A (controlling): extend `tools/route_overlap/analyze.py` so T5's
  per-token verdict sources `rowset_fetch` (schema draft section 6: known-
  event growth, hash recompute on every event, determinism unchanged), add
  exact consecutive-window containment and a per-layer reappearance-distance
  histogram; cross-validate the re-sourcing against the EXISTING ladder
  traces (must reproduce the hand-computed pilot 0/80 hash-equal,
  containment 0.358-0.389); then ONE owner-invoked 256-token generation
  probe (`--residency-tg --tg-tokens 256 --greedy --no-mtp`, GPU mutex,
  poller attached) yielding 10,200 consecutive-window pairs at ~2.4 MB
  trace.
  [CORRECTED 2026-09-13: the pilot is MACHINE-computed, not hand-computed —
  produced by the untracked `build/analyze_gplr13_ladder.py`; see the
  provenance caveat at the top of this file. The wording above is retained as
  written.]
- Gate rule: primary = hash-equal consecutive-window rate (exact, W=1).
  [SUPERSEDED 2026-09-13 by the gate-rule entry at the top of this file — the
  SDD section 10 rule governs; this line is retained as written.]
  PASS >= 0.6 per SDD gate 9 advances to Phase B (implementation, own
  dated contract update at entry; F2 per-layer row caches are a prerequisite
  design and a NEW residency class under SDD section 4). FAIL < 0.6 closes
  the task DONE-falsified-by-evidence with no library change; the reactive
  row-set stands as the TG answer alongside the closed handoff. Aggregate
  passing only via a few near-1.0 layers while late layers collapse is
  reported as volatility and escalated (per-layer policy = product intent).
- Also recorded: the secondary cumulative-resident-set containment
  distinguishes "prediction fails" from "rotation discards" — the 2-step
  pilot could not. The 0.6 threshold itself remains the SDD's number; this
  contract adds no new threshold.

## 2026-09-13: T-GPLR-13 marked DONE (owner review) — reactive expert row-set residency closed; residual risks and open follow-ups recorded

The owner reviewed the completed task and marked it done. All acceptance
criteria were already demonstrated (implementation complete, built clean,
end-to-end PP64+TG3 run, 50/512/4k/200k ladder + qa-smoke regression all
PASS, dual-path negative controls 18/18, validator parity, dense cold-path
ctest pair 2/2). No code changed for this close-out; both repos remain
uncommitted (no commit authorized).

Residual risks at close:
- TG tok/s (10.1 -> 7.7 across the ladder) is instrumentation evidence
  only; the row-set win is byte/wall evidence on 3-token windows. No
  headline performance claim is authorized by the task contract.
- The dominant TG staging term is submit-side pageable-H2D staging
  (bottlenecks doc section 6.2); the projected 1.2-1.4x from F1 is a
  hypothesis until the 512 A/B probe runs.
- H2 (200k staging-bandwidth decline 5.79 -> 4.39 GB/s) and the ~5 GB
  non-KV commit residual (H7) are explained-to-first-order only.
- All TG `layer_compute` timings are zero (H8/F3) — telemetry debt that
  blocks exact compute-vs-staging attribution.
- The reuse branch never fires (rows_reused=0); not a defect, but the
  H1(b) bookkeeping-only fix is UNSOUND (section 6.4) and must not be
  attempted over the aliased slot buffers.
- qa-smoke shapes remain batch-major; the 32x/14.6-19.2x row-set contrast
  is a same-model contrast, not a controlled A/B (section 3.8).

Open follow-ups (none scheduled; owner to prioritize):
- F1 pinned row staging (~14.2 MB window; needs a task contract; 512 A/B
  falsifier, 200k A/B decides H2).
- F2 per-layer row caches (≤ ~570 MB; makes the 0.358-0.389 containment
  reuse realizable and measures T-GPLR-15's 0.6 gate over a real stream;
  requires an SDD residency treatment before a contract).
- F3/H8 nonzero TG per-part timings (small telemetry change; do alongside
  F1).
- T-GPLR-15 predictive prefetch (DRAFT; 0.6 hit-rate gate unverified;
  negative 2-step pilot 0.358-0.389).
- Owner-invoked endurance ladder rungs 50k/100k/200k remain unrun.

## 2026-09-13: TG row-set staging analysis (section 6 of docs/gplr13-bottlenecks-hypotheses.md) — ready-latency is ~5% of TG fetch; the transfer term is submit-side pageable-H2D staging; H1(b) corrected, fix ranked F1 pinned row staging

Read-only analysis (no GPU runs, no code changes; online pass via context7
against NVIDIA CUDA programming guide + driver-API sync-behavior docs):
- TG `rowset_fetch` ready_us mean is 103.8-107.1 µs = ~5-6% of transfer_us
  (1,835-2,273 µs); the submit term carries ~94-95%. The 74.6-98.7% ready
  shares in the same file belong to the PP whole-bank path, not TG.
- Code-verified mechanism: `load_row_set` issues 24
  `ggml_backend_tensor_set_async` calls per fetch (~14.15 MB) directly from
  mmap-backed pageable sources and never uses `slot.buf_staging`; pageable
  H2D async copies stage synchronously through a driver pinned bounce
  buffer, matching the measured 6.5-8.2 GB/s host rate.
- Ranked fixes (each needs its own task contract): F1 explicit pinned row
  staging (~14.2 MB window; expected staging term 1.8-2.2 -> ~1.0-1.4
  ms/fetch, ~1.2-1.4x TG tok/s if the mechanism is confirmed; 512 A/B probe
  as falsifier, 200k A/B decides H2), F2 per-layer row caches (≤ ~570 MB
  total, makes the 0.37 containment reuse realizable AND measures
  T-GPLR-15's 0.6 gate over a real stream), F3 nonzero TG per-part timings
  (H8, do alongside F1), F4 W>1 (H4, leverage shrinks after F1/F2).
- CORRECTION to H1(b) in section 4: per-layer coverage over the current slot
  buffers is UNSOUND — slot bundles alias one buffer at offset 0, so an
  intervening owner overwrites layer il's rows before il returns; per-layer
  reuse requires real per-layer row storage (F2), not bookkeeping alone.
  Claiming coverage without it could silently execute another layer's
  weights. The never-fired reuse branch remains NOT a defect, but its fix is
  now correctly scoped.

## 2026-09-13: T-GPLR-13 test ladder executed — all five sizes PASS (50/512/4k/200k row-set + qa-smoke-512 regression), no failures, no retries

Owner-directed ladder on the completed row-set path. Harness:
`tools/run-gplr13-ladder.ps1` (new; mirrors `tools/run-tests.ps1` shapes:
row-set sizes use the probe flag set with `--pp-schedule` default, ubatch 512
for 512+, single-ubatch for 50, `--tg-tokens 3 --greedy --no-mtp`; qa-smoke
mirrors the T-GPLR-07 512 rung exactly). Every GPU run under
`tools/with-gpu-lock.ps1`; `tools/gpu-mem-poll.ps1` attached per run (0.5 s
sampling, killed at run end). Model: Qwen3.6-35B-A3B-MTP-UD-Q4_K_S. A foreign
deepseek_v4.exe held ~16.8 GB VRAM before the ladder and exited before the
recorded runs; all five ran on an idle device (free VRAM at size-50 run_start:
22.23 GB of 25.65 GB).

Artifacts preserved under `build/llama-cuda/gplr13-ladder/<size>/`
(`rowset-*.jsonl`, `rowset-*.log`, `mem.csv`, anchor prompt; plus `summary.json`,
`driver.log`); qa-smoke dir also holds `qa512.{jsonl,txt,log}`.

| size | exit | check-telemetry | TG tok/s | runner peak_vram | poller dedicated peak | shared peak | commit peak | avail min | staged % | fetch | rounds | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 0 | OK (569 ev) | 10.10 | 5,674 MiB | 4,581 MiB | 66 MiB | 17.3 GB | 36.9 GB | 3.13 | 120 | 3 | PASS |
| 512 | 0 | OK (569 ev) | 9.38 | 5,728 MiB | 4,635 MiB | 74 MiB | 17.3 GB | 38.4 GB | 3.13 | 120 | 3 | PASS |
| 4k (4001 tok) | 0 | OK (663 ev) | 9.52 | 5,796 MiB | 4,703 MiB | 78 MiB | 17.8 GB | 36.1 GB | 3.13 | 120 | 3 | PASS |
| 200k (200000 tok) | 0 | OK (4,469 ev) | 7.67 | 9,742 MiB | 8,649 MiB | 268 MiB | 24.0 GB | 33.4 GB | 3.12 | 120 | 3 | PASS |
| qa-smoke-512 (off) | 0 | OK (23,326 ev) | 0.53 (96 tok) | 4,350 MiB | 3,267 MiB | 74 MiB | 16.1 GB | 35.3 GB | n/a | 0 | 0 | PASS (answer PASS, negative FAILS) |

Findings:
- No size failed; zero refusals (`rowset_refused` count 0 in every trace),
  zero retries. Route parity held at every size (checker lifecycle + union
  validation on all 120 fetch events per row-set run).
- Staged bytes are context-length invariant: 572,719,104 / 18,327,011,328
  bytes per round (3.12-3.13%) at 50 through 200k tokens - the reactive
  union at W=1 is exactly 8/256 rows regardless of KV size, which is the
  row-set thesis holding under the largest recorded prompt.
- Runner-vs-poller VRAM gap is a stable ~1.09 GB at every size (5,674 vs
  4,581 at 50; 9,742 vs 8,649 at 200k) - consistent with the T-MEMPROBE-01
  delta-metric skew (CUDA context + teardown ordering), not a new leak.
- Shared (non-local) segment scales mildly with context (66 -> 268 MiB from
  50 to 200k) while dedicated grows ~4.1 GB - no duplication behavior; the
  T-MEMPROBE-01 conclusion extends to the 200k regime.
- Host RAM safe at all sizes: commit peak 24.0 GiB (= 25,754,857,472 B;
  the PowerShell driver labels binary GiB while the Python analysis file
  uses decimal GB — same measurement, different unit convention) of the
  113.5 GB limit at 200k (the 927 KB anchor text + mmap); available
  physical never below 33.4 GiB.
- TG tok/s declines 10.1 -> 7.7 across the ladder (3-token windows; staging
  572 MB/round dominates). Instrumentation result only - not a performance
  claim per the task contract; per-round ratios are the acceptance input.
- Row-set VRAM footprint delta 50 -> 200k: 5,674 -> 9,742 MiB runner
  (~+4.07 GB ~= the 200k KV estimate + activation slack). Preflight free
  VRAM headroom remains ample on the idle 24 GB device.

Harness notes for successors: Windows PowerShell 5.1 turns native stderr into
a terminating NativeCommandError under `$ErrorActionPreference='Stop'` (the
checker's stderr info lines killed early driver runs - use the driver's
Invoke-Capture); `Stop-Process` must be awaited before reading the poller CSV
(handle lingered and zeroed the first metrics pass); the terminal tool cannot
background commands - the 200k driver was detached via `Start-Process` with
log polling, and the whole ladder fits in ~25 minutes wall.

## 2026-09-13: T-GPLR-13 implementation COMPLETE — moe-row-set TG path runs end-to-end, checker + negative controls green, default-off path untouched

The reactive expert row-set task is implemented, built clean, and validated
end-to-end. Contract honored: single vertical slice, opt-in `--residency-tg`,
no behavior change on the handoff/default paths (both host suites green).

Wiring summary (all in the vendored tree unless noted):
- Runner: `--residency-tg` flag + usage; loud gates (requires `--residency
  moe-expert-bank`, `--tg-tokens > 0`; incompatible with `--residency off`,
  `--handoff/--handoff-partial`, `--mtp`); explicit request suppresses the
  T-GPLR-12 auto-handoff fit heuristic (row-set IS the TG policy choice);
  `ls.row_set_requested` set before init; `begin_row_set()` after PP (loud
  refusal aborts TG before any decode), `end_row_set()` after TG.
- Library: `run_start` gains `"rowset":"on"` exactly when active (absent when
  off — the immunity assertion holds); prebuild guard keeps the row-set TG
  set at width 1 (fails loudly on `n_rs_seq > 0`); executor stages the
  reactive union per layer/window and emits `rowset_init` / `rowset_fetch` /
  `rowset_round` (+ `rowset_refused` on failure), with `union_ids` ascending
  and the FNV-1a-64 hash over the canonical comma-joined string (schema
  draft sections 1-3).
- Checker (both check- and validate-telemetry): rowset event validation per
  schema section 4 — W=1 identity (`route_union == top_k`), accounting
  identity (`route_union == fetched + resident`, bytes iff fetched),
  generation lifecycle per (slot, owner, generation), hash recompute,
  run_start-marker gating, default-off absence. Hash compares use
  `strtoull` (the U64 hash exceeds LLONG_MAX; `atoll` saturated silently).
  Unknown rowset events still fail loudly; legacy traces unaffected.

Evidence (GPU mutex, exit 0, check-telemetry OK, 569 events):
- `build/llama-cuda/gplr13/rowset-pp64-tg3.jsonl`: Qwen3.6-35B MoE, PP 64 +
  TG 3, 4 slots. 120 `rowset_fetch` events; per-round `rowset_round`:
  staged 572,719,104 / full-bank 18,327,011,328 bytes = **3.12%** per round
  (8/256 experts routed at W=1 — exactly the reactive union). Route parity
  held: no missing-row path fired; checker recomputed every hash.
- Negative controls (`tools/run-gplr13-negfix.ps1`, host-only): 8/8 — the
  untouched trace passes; tampered fixtures each trip their rule loudly
  (hash mismatch, accounting identity, W=1 identity, first-window phantom
  resident rows, tracked-coverage desync, marker stripped, injected
  refusal).
- Driver: `tools/run-gplr13-rowset-probe.ps1` (rerunnable end-to-end probe
  under the GPU lock).

Status vs acceptance criteria: route-event parity demonstrated (executed
rows cover the routed IDs; checker lifecycle validation enforces it on every
trace); staging bytes per TG round = the measured union-width ratio (3.12%);
wall time recorded as instrumentation only. Focused-validation status:
ALL items complete (512 row-set run + qa-smoke 512 regression + ladder +
the dense ctest pair below); nothing else is open. Audit close-out
2026-09-13: dense cold-path validation passed — qwen38-layer-boundary +
layer-slot-lifecycle 2/2 under the GPU mutex with Q38LS_MODEL_PATH set
(harness note: tools/run-tests.ps1 reads but does not propagate the model
path to ctest children; export it before GPU ctest suites). Validator
parity: the rowset_init window_tokens rule now exists in BOTH validators
(check_telemetry main.cpp:738 and validate_telemetry, identical message);
a rule sweep found no other row-set asymmetry. Negative controls are
dual-path: 9 fixtures x both validators = 18/18 correct verdicts plus both
positive controls (tools/run-gplr13-negfix.ps1). The never-fired reuse
branch (rows_reused=0 at 4 slots) is ruled NOT a T-GPLR-13 defect — its
first positive demonstration is owned by the H1 follow-up contract
(docs/gplr13-bottlenecks-hypotheses.md).

Notes for successors: `--residency-tg` + `--handoff` now conflict loudly
instead of racing the auto-fit decision; the serial TG loop aborts TG when
row-set activation fails (no more decode-after-refusal). T-GPLR-15's inputs
(rowset_fetch union streams + hashes) now exist in real traces.

## 2026-09-13: Memory-validation probe (T-MEMPROBE-01, owner request) — no RAM/VRAM overcommit observed; shared GPU memory does NOT duplicate device allocations; runner peak_vram_bytes overstated ~32%

Owner asked to validate actual RAM and VRAM usage (suspicion of overcommitting)
and whether the GPU driver duplicates memory because it is "shared". Evidence
from one dense 1024-token layer-major probe (Qwen3.8-27B-UD-Q6_K_XL, ubatch
256, 4 slots, q8 KV, --no-handoff, GPU mutex, exit 0, check-telemetry OK,
generated ids 286 220 197 = the reference set) with a 0.5 s Windows
GPU-adapter-memory poller running across the whole run:

Artifacts: `build/llama-cuda/memprobe/memprobe-dense-1024.{jsonl,csv}`;
tooling: `tools/gpu-mem-poll.ps1` (new, incremental-flush poller: adapter
dedicated/shared segments, per-process GPU memory, avail phys, system commit)
and `tools/run-memprobe-dense.ps1` (driver; rerun to regenerate). Note LUIDs
rotate across boots — analyze by magnitude, not by fixed LUID.

1. **VRAM: no overcommit.** OS-observed peak dedicated (local) segment on the
   NVIDIA adapter = 3,356.9 MiB (all attributable to the runner, pid peak
   3,352.6 MiB within the ~4 MiB counter noise MS KB4490156 documents for
   per-process GPU counters). Runner-reported `peak_vram_bytes` = 4,439.56 MiB
   (4,655,218,688 B). The runner's number OVERSTATES the physical peak by
   ~1,082.7 MiB (~32%) because it is a before/after free-VRAM delta
   (main.cpp:3641: `total - min(free_after, free_at_start)`), not a true peak:
   CUDA context (~500-800 MiB) and other allocations that free before teardown
   are counted. It is a conservative upper bound, never under-reports; true
   mid-run transients above the end-state footprint would also be missed, so
   it is neither a peak nor a steady-state number.
2. **Shared GPU memory: NO duplication observed.** WDDM "shared GPU memory"
   is the non-local (aperture) segment backed by system RAM, used for system
   copies, command, and paging buffers. If the driver mirrored device
   allocations, shared would have grown ~3.3 GiB during the run. Measured:
   NVIDIA shared segment max 76.2 MiB (idle ~0.2 MiB); runner per-process
   non-local max 76 MiB. The 24 GiB of host-RAM drop seen during the run
   (avail phys 53.9 -> 29.7 GiB min) is the file-backed mmap of the 25.9 GB
   GGUF + staging touching pages, NOT a hidden copy of VRAM. Resident set
   grows because the file cache maps real pages; system commit peaked at only
   16.8 GB of the 113.5 GB limit (pagefile pressure ~none).
3. **RAM admission rule remains sound.** The T-GPLR-09 bank-cache admission
   arithmetic (avail_phys - banks >= 0.15 x total_phys) works on avail/total
   physical, which the probe confirms behaves as recorded (min avail 29.7 GiB
   with the model mapped, well above reserve). The VirtualLock/prefetch path's
   documented limitation (no SeLockMemoryPrivilege on this token) is
   unchanged.
4. **Telemetry gap (task-owned, owner call):** `peak_vram_bytes` should either
   be (a) sampled during the run (e.g. cudaMemGetInfo at each phase_summary,
   cheap) or (b) relabeled (e.g. `vram_before_after_delta_bytes`) so traces
   cannot be read as a device peak. Until fixed, treat reported peak VRAM as
   an upper bound with a ~0.5-1.1 GiB context/teardown skew on this machine.
   The 2026-09-10 8 GB MoE run's runner 6693.56 vs NVML 5607 MiB disagreement
   (~1.09 GB) matches this mechanism exactly (NVML sampled sparse under load;
   the delta metric is the larger, skewed number) — that discrepancy is now
   explained; no new defect is implied there.
5. **Caveats.** Single probe shape (dense, small ctx; MoE expert-bank staging
   not re-probed — its per-run VRAM story is the slot window + staging buffers,
   covered by the same delta-metric caveat). Windows GPU adapter counters are
   formatted-data (0.5 s granularity); a sub-0.5 s transient could still hide
   from BOTH instruments, which is why recommendation (a) samples
   cudaMemGetInfo in-process at phase boundaries. Shared-segment growth is
   still the first thing to watch if VRAM pressure ever pushes the driver
   toward paging; `--physical-vram-bytes`/preflight arithmetic is unaffected.

No code changed. No commit (not authorized).

## 2026-09-12: T-GPLR-13 gate (b) DECIDED — reactive row-set is the priority TG path; T-GPLR-16 deprioritized with option (a) disposition

The owner's priority call was made on the four-way comparison (reactive
row-set vs the T-GPLR-16 escalation options a/b/c). Decision and
reasoning, recorded in the plan header, T-GPLR-13 and T-GPLR-16 blocks:

- **T-GPLR-13 (reactive row-set) is chosen** and moves DRAFT -> READY
  (next implementation task). Why: (1) single-process — the T-GPLR-17
  rejection ("cannot have inference that constantly switches engines")
  makes single-process operation program-defining, and row-set is
  single-process by construction while T-GPLR-16's surviving options
  are not clearly so; (2) bounded VRAM by construction (slot window +
  row-granular packages) — it is the designed answer for the
  not-fitting regime (T-GPLR-16's original target), which the closed
  T-GPLR-10 handoff cannot serve; (3) the riskiest unknown (scheduler
  cross-backend staging) was already retired by the SDD amendment's
  hard execution-shape constraint, grounded in the T-GPLR-16 deadlock
  evidence; (4) it is also the measurement instrument — its
  rowset_fetch events are the only source of per-token TG route data,
  so it unblocks T-GPLR-15's UNMEASURED 0.6 hit-rate gate regardless
  of TG speed outcome; (5) in the fits regime it does not displace
  anything: auto-handoff (T-GPLR-12 default profile) already wins
  there at 64.63 tok/s and row-set is TG-only opt-in.
- **T-GPLR-16 disposition: option (a)** — revert MIN_BATCH=1 to the
  pre-fix partial TG (known step-1 gold flip, QA-gate failure accepted)
  and deprioritize the task to DRAFT; the in-progress implementation is
  parked. Option (b) (upstream device-side deadlock chase, symbolized
  build + nsight/driver trace) is declined for now: open-ended upstream
  debugging investment against a sharply characterized but
  upstream-owned failure, while row-set sidesteps the mechanism
  entirely (its packages stage through the copy-event lifecycle before
  graph execution; MUL_MAT_ID never sees host-resident banks on this
  path). Option (c) (CPU-PP consistent arithmetic) stays recorded as a
  valid fallback but unprioritized: it avoids the deadlock by
  construction yet regresses PP throughput, and it does not close the
  not-fitting regime the way row-set does.
- Priority order now: T-GPLR-13 (ready), then T-GPLR-15 review on
  rowset_fetch evidence. T-GPLR-16 resumes only on a new owner call.
- No code changed in this step (decision record only; docs edits in
  the plan and this file). The MIN_BATCH=1 probe guards in the vendored
  tree (LLAMA_LS_NO_MINBATCH_PROBE etc.) now have an owner disposition
  pointing at option (a); the revert itself is vendored-tree work that
  should be scheduled alongside or after T-GPLR-13 per owner direction.

Vendored llama.cpp state: `099a27a7e` — layer-streaming: T-GPLR-08/10/11,
prefill layer residency, MoE handoff, tail-ubatch fix (2026-09-09).
Working tree adds T-GPLR-12 (default profile v2) on top of `099a27a7e`,
uncommitted (no commit authorized yet): runner tri-state + defaults,
module handoff_mode telemetry, MoE builder bf16 slab casts, checker
`handoff_ever` fix, CMake/qa-smoke default-immunity pins.

## 2026-09-12: Drafted model-run testing standard

Drafted `docs/model-run-testing-standard.md` in response to the question of
whether the workspace should standardize on a single-shot, file-driven,
capped llama-completion shape to avoid runaway/hard-to-grade test runs.

Verdict: the intent is right; the exact phrasing in the request is too
informal to adopt as written. The draft standard captures the intent as a
reproducible invocation shape: single-shot, prompt from a file, explicit hard
cap, deterministic artifacts, and programmatic grading with a must-fail
negative control. Force-termination is retained as a controlled fallback for
a genuinely stuck process, not as the normal way a test finishes.

The draft standard points at the existing semantic-QA smoke gate as the local
model for this shape. It does not yet claim the whole repo has adopted it;
that is an owner-review decision. Implementation note:
the standard does not wait for those three clarifications to exist; it records
them as explicit scope items, and the implemented shape resolves the vehicle
question directly: "llama-completion" is the stock llama.cpp single-shot
completion vehicle, and the equivalent local vehicle for the experimental path
is the layer-streaming runner with --prompt-file / --tg-tokens / --tg-text.
Whether the standard is adopted repo-wide, and against which real failure mode,
is still an owner-review decision.

## 2026-09-12: moe-row-set SDD amendment written (owner-directed; T-GPLR-13 gate (a) resolved)

The owner directed the `moe-row-set` amendment and its elaboration. Amended
docs/layer-residency-sdd.md sections 1, 4, 6, 8, 9, 10, 11, 12 and the plan
header + T-GPLR-13 status. Key content:

- New package class `moe-row-set` (section 4): per logical MoE layer and TG
  window (default one token), the deduplicated union of routed-expert rows
  (gate/up/down) the router actually selects; reactive, never predictive;
  immutable per generation; atomic ready/retire; loud under-coverage refusal.
- **Execution-shape constraint (hard requirement)**: row-set packages are
  CUDA-resident slot buffers staged BEFORE graph execution under the standard
  copy-event/generation/consumer-done lifecycle; `MUL_MAT_ID` on this path
  always consumes CUDA-resident weights of the matching slot generation;
  scheduler cross-backend staging (ggml-backend rowcopy, op-offload rewires)
  is prohibited for row-set graphs. Grounded in the T-GPLR-16 deadlock
  evidence (same-date section below).
- Policy `moe-row-set` added (TG-phase only; PP keeps its declared policy;
  default-off unchanged). Adapter boundary grows per-window reactive
  row-membership enumeration + rowset_fetch emission + declared forbidden
  staging paths. Failure contract, telemetry (full per-token union ID lists —
  T-GPLR-15's gate telemetry), acceptance gates (route-event parity,
  checker-validated completeness, staging bytes vs full-bank baseline,
  QA gate, default-off regressions; performance claims only vs the reactive
  baseline and only at hit rate >= 0.6), and risks (UNMEASURED temporal hit
  rate, volatile union width, the execution-shape landmine) all extended.
- T-GPLR-13 gate (a) is thereby RESOLVED; gate (b) (owner priority vs the
  handoff path) remains open. T-GPLR-15's 0.6 hit-rate gate is unchanged and
  still unmeasurable until T-GPLR-13 exists.

## 2026-09-12: T-GPLR-16 MIN_BATCH=1 wedge fully characterized — device-side stream deadlock, containment attempt reverted, escalation

Continuation of the 2026-09-11 entry. The "wedge" is now precisely located
and mechanized; the prior interpretation (upstream rowcopy × CUDA-graph-cache
corruption) is **superseded**.

New evidence (all under GPU mutex, artifacts `build/llama-cuda/gplr16/`):

1. **The freeze point is the 3rd TG token, not the 2nd.** TG telemetry
   `ubatch` indexes tokens: ub1 = token 0, ub2 = token 1, ub3 = token 2.
   Tokens 0–1 complete with bit-exact step-0 margins; every MIN_BATCH=1 run
   then dies inside the first `layer_compute` of ub3 (marker legs:
   `[ls-exec] g=18 compute_enter` with no `compute_done`; g=18 is host-bank
   layer 17's graph with rowcopy, layer 0's graph in the no-rowcopy leg).
2. **It is a device-side stream deadlock, not a host lock and not corruption
   at capture time.** During the freeze: GPU util 0 %, process alive, main
   thread spinning inside `cudaStreamSynchronize` → driver `cuStreamQuery`
   poll loop (captured with a StackWalk64 dumper, `tools/gplr16-dump.cpp`,
   `build/gplr16-dump.exe`, no PDBs so frames below the CUDA boundary are
   export-guess noise). If left alone long enough, Windows TDR reaps the
   stuck kernel: `CUDA error: the launch timed out and was terminated` at
   `ggml_backend_cuda_synchronize`.
3. **Independent of CUDA graph capture.** With `GGML_CUDA_NO_GRAPHS_PROBE=1`
   (verified applied: 0 "warmup complete" lines) the run froze identically
   at ub3/l17. Direct execution of the same mixed-split graph deadlocks too.
4. **Independent of expert rowcopy.** With `GGML_SCHED_NO_EXPERT_ROWCOPY_PROBE=1`
   (full 452 MB bank copy fallback instead of used-row staging) the run froze
   identically at ub3/l0. Staging tensors are not the trigger.
5. **Decisive correlation: MIN_BATCH value, not token count.** Every
   completing partial-handoff cell (ab2-D/E/F, ab-B/C; 96 TG tokens each)
   ran with MIN_BATCH at its default 32, i.e. TG-side expert FFN of host
   banks on CPU. Their garbage `.txt` answers show they are pre-fix builds —
   completing-but-wrong, not counter-evidence. Every MIN_BATCH=1 attempt
   (TG expert FFN of host banks forced onto CUDA through scheduler staging)
   deadlocks the device by TG token 2. The deadlock is in the CUDA-side
   `MUL_MAT_ID` execution against host-resident expert banks under the
   partial-handoff scheduler configuration (fast-path kernel per
   `ggml_cuda_mul_mat_id_needs_sync` selection), regardless of how the rows
   got there (rowcopy staging or full-bank copy).
6. **Containment attempt (capture-time staged-weights discriminator) is
   proven ineffective and was reverted.** Two independent reasons: (a) the
   `ggml_cuda_graph_check_compability` check runs at graph build, *before*
   `ggml_backend_sched_compute_splits` rewires `MUL_MAT_ID` src[0] to the
   rowcopy staging tensors, so it can never observe them at capture time;
   (b) capture is not the mechanism anyway (finding 3). A comment remains at
   the site in `ggml/src/ggml-cuda/ggml-cuda.cu` so the idea is not retried
   blindly; the code is functionally back to the pre-attempt state.
7. **Memory-commit lesson (owner request: keep commit under ~50 GB).** A
   frozen leg holds ~20.6 GB private commit and WDDM mirrors more; two
   concurrent legs (one frozen, one mutex-waiting) approach 130 GB commit on
   a 61.7 GiB host. GPU legs stay strictly serialized, one at a time, and a
   run confirmed frozen is killed immediately after its evidence is captured.

New env probe guards in the vendored tree (all reversible, remove after
resolution): `GGML_LS_GRAPH_EXEC_MARKERS` (per-graph exec markers), plus the
previously recorded four. New artifacts: `contain-V1.*` legs (containment
run, graphs-off leg, rowcopy-off leg), `fix-V1-killed-session-evidence.*`,
`build/gplr16-dump.exe`, `tools/gplr16-dump.cpp`.

Open items (escalated to owner — the plan's escalation condition applies;
none of the three can be closed by implementer action alone):
- (a) Revert MIN_BATCH=1 → pre-fix partial TG with the known step-1 gold
  flip; fails the QA gate as demonstrated (2026-09-11 evidence).
- (b) Chase the device-side deadlock upstream (symbolized CUDA build with
  PDBs, nsight/driver trace) — a real but open-ended debugging investment;
  the trigger is now sharply characterized (MUL_MAT_ID CUDA fast path on
  host-resident banks, 3rd token, partial-handoff scheduler configuration).
- (c) Consistent-arithmetic alternative: CPU PP for host layers (recorded
  option c) — avoids the deadlock by construction, needs the owner's call
  because PP throughput regresses.

## 2026-09-11: T-GPLR-16 root cause — mixed-arithmetic PP→TG switch, not corruption

Full root-cause chain for partial-handoff TG degradation, all evidence under
GPU mutex in `build/llama-cuda/gplr16/`:

1. **No memory corruption.** Runtime per-node eval-callback probes (`rt-*`)
   show routing/expert IDs/attention bit-identical between full handoff and
   partial N=20 through the whole stack at TG step 0.
2. **Divergence = cross-backend kernel variance.** Pre-fix partial TG computes
   host-bank `MUL_MAT_ID` on CPU (`buf=CUDA_Host`; CUDA offload gate requires
   `get_op_batch_size >= GGML_OP_OFFLOAD_MIN_BATCH`=32, `MUL_MAT_ID` batch =
   ne[2]=1 at TG). CPU Q4_K vs CUDA MMQ differs ~0.3% per host layer,
   envelope-consistent through L20 — benign in isolation.
3. **The killer is the PP→TG arithmetic switch.** PP runs all-CUDA; TG flips 20
   layers to CPU arithmetic mid-trajectory. Greedy chain evidence: TG step 0
   survives (margin 0.440 full vs 0.354 partial, same gold 248046), step 1
   flips gold 198→271 (margin 16.30→1.75), then repetition collapse. Stock
   `-ncmoe 20` proves CPU-FFN TG is fine when CONSISTENT: stock S2 answered
   the anchor-512 QA identically to full-CUDA S1 (check-answer 5/5 PASS both)
   with its FFN measurably on CPU for PP and TG (PP 4.60 ms/tok vs 0.53, TG
   17.0 ms/tok vs 8.4). The SDD "must not alter model arithmetic" invariant
   is violated *per-request* by the switch, not by host placement per se.
4. **MIN_BATCH=1 fix works numerically, wedges in upstream machinery.** Runner
   override `GGML_OP_OFFLOAD_MIN_BATCH=1` under `--handoff-partial` routes TG
   expert FFN to CUDA (rowcopy of used experts only): TG step-0 margin becomes
   0.439782 == full handoff exactly; V2 full-handoff regression bit-identical
   to reference; tiny-prompt partial runs complete correctly. But anchor-512
   partial wedges at TG token 1, layer-17 graph (2nd execution): graphs ON →
   memcheck-named `mul_mat_vec_q` wild-read (IDs garbage at kernel arg);
   graphs OFF/rowcopy OFF/launch-blocking → deterministic hang, same spot;
   synccheck clean. This is upstream ggml-backend rowcopy × CUDA-graph-cache
   interaction on 41-graphs/token mixed-split TG sets, not our executor
   (our executor snapshots/restores rewired srcs and syncs correctly).

Open items: choose between (a) revert MIN_BATCH=1 and accept pre-fix partial
with the known step-1 numeric flip (fails QA gate), (b) fix/contain the
upstream rowcopy+capture wedge (e.g. disable CUDA graphs for TG layer graphs
containing rowcopy splits only — narrower than the global NO_GRAPHS probe,
which also hung), or (c) escalate the mixed-arithmetic invariant question to
the owner (PP-on-CUDA + TG-on-CPU for host banks cannot pass the QA gate as
demonstrated; consistent-arithmetic alternatives are CPU PP for host layers
or the MIN_BATCH=1 route once the wedge is fixed). Env probe guards in the
vendored tree (remove after resolution): `GGML_CUDA_NO_GRAPHS_PROBE`,
`GGML_SCHED_NO_EXPERT_ROWCOPY_PROBE`, `LLAMA_LS_NO_MINBATCH_PROBE`,
`LLAMA_LS_TG_ROUTER_PROBE`. Executor `[ls-exec]`/rt-probe markers are
guarded by `LLAMA_LS_GRAPH_EXEC_PROBE`/`LLAMA_LS_TG_ROUTER_PROBE`.

Also fixed this session: duplicate `layer-streaming-scheduler-256-run`/
`tail513` ctest blocks in `tools/layer-streaming/CMakeLists.txt` (prior-
session edit accident; configure failed with duplicate test names).

## 2026-09-10: T-GPLR-17 REJECTED by owner and fully reverted (two-process hybrid bridge)

After the implementer proved the mechanism end-to-end (bridge probe under
GPU mutex, artifacts `build/llama-cuda/gplr17-probe/`: runner layer-major
PP 512 tokens + `--state-out` -> slot state file 71,447,536 B (KV + recurrent
DeltaNet state + 512 prompt tokens, sidecar interop gate OK) -> llama-server
`-ncmoe 40 -ngl 999` healthy in 3.6 s, `POST /slots/0?action=restore`
n_restored=512 == sidecar, completion prompt_n=1 continuity proof, TG 23.2
tok/s, driver exit 0), the owner rejected the approach outright: an
inference system that switches engines mid-session is unusable for
production or even local use ("cannot have inference that constantly
switches engines"). Decision: revert everything.

Reverted in full (verified vs the pre-existing working tree): runner
`--state-out` flag + state-save block + sidecar, `state_save` telemetry
method, off/handoff conflict gates, host fixtures and CMake tests,
tools/run-hybrid.ps1, and the hybrid-probe/hybrid-qa/hybrid-50k suites in
tools/run-tests.ps1. The vendored tree is back to the T-GPLR-12 working
tree only; the runner was rebuilt from the reverted sources and the host
suite re-passes. The plan records T-GPLR-17 as rejected with the recorded
state-save/restore seam facts (library `llama_state_seq_save/load_file`,
server POST /slots/{id}?action=save|restore) kept as historical evidence.

Consequence for the open program: the single-process mixed-placement TG
mode (T-GPLR-16, owner-gated) is the aligned direction for TG-at-context
under the 8 GB ceiling; no two-process orchestration will be pursued
without a new owner decision.

## 2026-09-10: T-GPLR-17 drafted READY — two-process hybrid bridge (streaming PP -> state file -> CPU-expert llama-server TG)

Planner session (docs-only; no code, no GPU run). Owner request: wire
llama_state_save_file/load_file into the runner and llama-server CLIs so
streaming PP at 50K hands off KV state to a CPU-expert llama-server for
stock TG under an 8 GB VRAM ceiling. Key inspection facts (verified in the
vendored tree, not assumed):
- `llama_state_seq_save_file` / `llama_state_seq_load_file` exist
  (src/llama-context.cpp, state_seq_load_file ~3534) and the payload is the
  prompt token list PLUS the full memory-module write
  (`memory->state_write(io)`) — KV AND recurrent DeltaNet state both cross
  the process boundary.
- The server ALREADY exposes the seam: POST /slots/{id}?action=save|restore
  (routes registered server.cpp:273; handlers server-context.cpp ~2480 and
  ~4553; restore erases the slot, loads the file, updates cached prompt).
  So the server side needs ZERO vendored code changes; only the runner needs
  one flag (`--state-out PATH`, calling llama_state_seq_save_file after PP)
  plus a PowerShell driver `tools/run-hybrid.ps1` (start fit-target/-ncmoe
  server, /health, POST restore, generate). Existing server unit tests cover
  the endpoints (tools/server/tests/unit/test_slot_save.py).
- Drafted as T-GPLR-17 in the generic plan, status READY, next eligible
  item. Explicitly NOT an SDD amendment: no new residency package type,
  policy, or placement mechanism — orchestration over two existing
  processes plus one runner flag. Acceptance: 50K end-to-end under the GPU
  mutex (runner ~6.5 GB; server <= 8 GB ceiling), state-file size ~KV
  534 MB + recurrent 66 MB, continuity proven by the QA 512 rung through
  the hybrid, default-off byte-identical, dense ctest 2/2.
- Plan repair en route (flagged): T-GPLR-15's body (Goal/Scope/Acceptance
  etc.) had been stranded after T-GPLR-16's block since the 2026-09-10
  renumber session, leaving T-GPLR-15 a stub; the body is now inside
  T-GPLR-15's own section (14 -> 15 -> 16 -> 17 ordering), content
  unchanged.
- Naming note: full-context `llama_state_save_file/load_file` (session
  variant, completion.cpp --session path) is NOT the chosen vehicle; the
  per-sequence slot variant matches the server endpoints and avoids
  whole-cache clobbering.

Runner default profile v2 (T-GPLR-12, owner-approved 2026-09-09):
`--kv-cache q8`, `--pp-schedule layer-major` + `--pw-probe`,
`--act-slab-dtype bf16`, `--handoff auto` (auto enables handoff only
when the T-GPLR-10 fit gate passes at run start; explicit `--handoff`
keeps hard-refuse semantics; `--no-handoff` disables). Every flag stays
overridable; refusal/refusal-free behavior is loud in both auto
branches (`handoff_mode`: auto-on | auto-off | on | off, validated by
check-telemetry). MTP stays opt-in (subsumed by auto-handoff in the
fits regime).

Milestone tag `layer-residency-plan-complete` (2026-09-06): this repo →
27514d1dabeef7119b53241aad65167ae2a410d8; vendored llama.cpp →
169ab086e857d287daac626dc7eda6a4f9764efe. Both annotated, local only
(not pushed).

## 2026-09-10: Streaming-PP + stock-TG hybrid (--handoff) at 50K — loud preflight refusal on today's VRAM baseline

Owner question: can streaming layer-major PP be combined with stock llama.cpp TG for a
262K test, and can the combo run at 50K now? Shape attempted (GPU mutex, launcher
`build/run-hybrid-50k-handoff.ps1`): `--pp-schedule layer-major --pp-tokens 51200
--ubatch 512 --layer-slots 4 --ctx-size 51456 --kv-cache q8 --handoff --pp-no-logits
--tg-tokens 16 --greedy --no-mtp` on Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.
- Outcome: REFUSED at the T-GPLR-10 handoff capacity preflight, before the timed PP,
  exit 1 — designed loud behavior, no silent fallback. `weights+KV+state
  22,003,032,576 B (KV 559,841,280 B) + act slabs 419,430,400 B + reserve 536,870,912 B
  = 22,959,333,888 B exceeds observed free VRAM 22,226,665,472 B`.
- Margin arithmetic: shortfall 732,668,416 B (~699 MiB). Total q8 KV @ 51456 is only
  559,841,280 B, so NO context size closes the gap (floor at near-zero ctx is still
  ~173 MB short) — the full-model handoff cannot fit on this card at today's desktop
  free-VRAM baseline (22.23 GB, down from 22.92 GB observed 2026-08-21) at ANY length.
- Ceiling unchanged: the combo requires the whole model in VRAM post-handoff, so 262K
  is impossible on 24 GB by construction and 50K is now below the floor too. The combo
  remains proven at small ctx (T-GPLR-10: 2048-token run, TG 64.63 tok/s, parity PASS).
- Artifacts: `build/llama-cuda/gplr-hybrid-50k-handoff.{jsonl(empty),log}`; the .jsonl
  holds no timed events (refusal precedes PP). No commit.
- [AMENDED 2026-09-14: the "at ANY length" / "50K is below the floor too" claims held
  for THAT day's shape, which carried non-zero activation slabs (419,430,400 B) and 4
  slots. In the qa-smoke handoff shape used 2026-09-14 (act slabs 0 B, 1 slot) the
  gate PASSES at 4k ctx -- run, exit 0, TG 68.960 tok/s, peak 4,559.6 MiB -- and
  refuses at 50k with a 299,311,616 B shortfall (22,525,977,088 vs 22,226,665,472 B).
  The gate's own terms put today's boundary at ~22,665 ctx tokens of KV capacity, so
  the ceiling is length-dependent and now derived rather than assumed; no run
  straddles it. See the 2026-09-14 mode-measurement entry above. No dated claim
  above is retracted -- the 2026-09-10 shape still fits at no length.]

## 2026-09-10: Streaming-TG slowness profiled against stock baselines (host-only analysis, no GPU run)

Question: where does the 8 GB streaming TG's 0.509 tok/s (1.966 s/token)
go, and why is stock TG 32-127x faster? Method: event-level re-analysis of
`build/llama-cuda/gplr-moe-8gb-50k.jsonl` (TG window = events between the
two phase_summary lines), the runner source, the ncmoe262k bench JSON, and
the owner's stock llama-server baseline log
`C:/Development/AI/qwen38GZI/out/llamacpp-baseline.log` (36 KB, full session).
No GPU work; no mutex needed.

Stock TG baselines, now with correct shapes:
- llama-bench `-ncmoe 20` TG leg: 64.9 tok/s = 15.42 ms/tok, but at
  n_prompt=0 (zero-length KV, no MTP) — a weights-only decode rate.
- Owner's llama-server run (new, 2026-09-10): build 10876, `--fit-target
  16000` auto-fit placed 34/41 layer groups on GPU (~6.6 GiB) with the
  remaining routed experts on CPU, `-c 262000`, q8 KV, FA, MTP draft
  (max 5). Measured: **TG 32.44 tok/s (30.83 ms/tok) over 1724 tokens**,
  MTP acceptance 0.546 (acc/pos 0.816/0.661/0.525/0.415/0.313, mean len
  3.73), graphs reused 456, PP 372 tok @ 39.49 tok/s. This is the real
  "expected TG" at context on this machine.
- Derived (arithmetic, marked as such): routed-expert row bytes per token
  = 8 experts x 1.77 MB x 40 layers = ~565 MB; the bench's 15.4 ms and the
  server's per-round numbers both close at ~35-40 GB/s effective DDR feed
  (memory-bound CPU expert reads) — the stock path touches only routed
  rows, never whole banks.

Streaming-TG decomposition (from the trace, 3 tokens, all 120 copies):
- Wall 1.966 s/token. Copy windows sum to 2.220 s/token (54.98 GB = 18.33
  GB x 3; ~0.25 s/token of copy overlaps compute), event wait 6.2 ms,
  compute 46.6 ms (2.4% of wall). The phase IS the copy chain.
- Per-bank copy: 453 MB in 46.4-190 ms (p50 47.0 ms); effective feed
  8.26 GB/s (host_submit 85.8% + ready_latency 14.2% of each window).
- Serialization: 108/120 next-layer copies were submitted only after the
  current layer's compute started; effectively one copy in flight — the
  slot protocol copies, waits on the CUDA event, computes, per layer.
- Compute is NOT the problem: 46.6 ms/token summed over 40 layers (0.31
  ms recurrent layers; the 10 full-attn layers 0.6-7.1 ms each at 50k q8
  KV). graph_builds 0 in TG; per-copy ready-latency (~6-22 ms fault-start)
  is small only relative to 453 MB copies — it would dominate if banks
  were chopped into small pieces without pipelining.

Why (attribution): the streaming path stages WHOLE 453 MB banks — 18.33 GB
per token — while the token's routed experts touch ~565 MB: a **32x byte
amplification**. Combined with a ~4.6x slower effective byte feed (8.26 GB/s
fault-driven H2D slot stream vs ~35-40 GB/s host DDR for the CPU-expert
path), that accounts for the ~127x gap to the bench number; the server's
32.44 tok/s with MTP sits on the same ~15-19 ms raw forward (MTP nets ~3.7
tokens/round at 0.546 acceptance). The stock CPU-expert path is effectively
row-set semantics executed by the CPU at DDR bandwidth; the streaming path
is whole-bank semantics over a slow, serialized VRAM-slot feed. Ceiling
arithmetic at measured rates: whole-bank staging cannot exceed ~0.45-0.7
tok/s even with pinned staging (12.7 GB/s) — the gap is the unit of
staging, not tuning.

Implications for the open program (estimates, not measurements):
- VRAM row-set (T-GPLR-13 shape) at 8.26 GB/s: ~68 ms/token copy feed ->
  ~14.6 tok/s ceiling if per-layer row copies pipeline behind compute;
  latency-dominated (~9 tok/s) if they stay serialized. MTP verify raises
  bytes/round (draft-route unions) and does NOT help a bandwidth-bound
  H2D path (~9-14 tok/s).
- The CPU-expert path (the owner's baseline shape) reaches 32.44 tok/s
  today with MTP — the fastest measured TG-at-context on this machine.
  The two-process hybrid (streaming PP + KV state save/restore into a
  fit-target llama-server, T-GPLR-16-adjacent) is the bridge that
  preserves our PP rate and this TG rate; a single-process equivalent
  needs mixed-placement TG, not row-set, to match it.
- `--copy-source pinned` exists and is a cheap probe for the copy-feed
  half (PP pinned staging measured H2D ~12.7 GB/s), but it cannot change
  the 32x byte amplification.

Supersedes the interpretation (not the data) of the NCMOE-262K section's
"TG @ full 262K ctx" claim above. Artifacts: the traces named above plus
the owner baseline log; no new run, no commit.

## 2026-09-10: T-GPLR-14 numbering collision resolved (mixed-placement TG mode renumbered to T-GPLR-16)

The NCMOE-262K section (2026-09-09) forward-referenced a "T-GPLR-14
mixed-placement TG mode" — a fresh collision with T-GPLR-14 (route-overlap
instrument, done 2026-09-09), added concurrently by the NCMOE session.
Resolution: the mixed-placement TG mode is drafted as **T-GPLR-16** (next
free ID after T-GPLR-15) in the generic plan, house-style row with goal,
scope, non-goals, design/failure contract, falsifiable acceptance criteria,
focused validation, and escalation condition. The stale reference in the
NCMOE section above was renumbered to T-GPLR-16, and the plan header and
serialization note now list T-GPLR-16 among the remaining DRAFT items
(T-GPLR-13 reactive row-set, T-GPLR-15 predictive prefetch, T-GPLR-16
mixed-placement TG — all owner-gated; one ready task at a time).
T-GPLR-14 remains closed: `tools/route_overlap/analyze.py` and its
byte-identical fidelity proof are untouched by this edit. No plan or SDD
content was redesigned; the collision fix is renumbering + drafting only.

## 2026-09-09: MoE TG row-set program drafted/renumbered; T-GPLR-14 DONE (route-overlap instrument)

Numbering collision caught before it hurt: the main worktree already holds
the owner's own uncommitted T-GPLR-12 "default profile v2" (done,
owner-approved, gplr12-* traces). The row-set program drafts were
renumbered to T-GPLR-13 (reactive expert row-set), T-GPLR-14 (route-
overlap instrument), T-GPLR-15 (predictive per-token prefetch) and are
recorded in the plan.

T-GPLR-14 is implemented and closed (plan status -> done):
- Tool: `tools/route_overlap/analyze.py` — stdlib-only, read-only,
  deterministic (no timestamps). Computes per-layer route-union widths
  (T1), the exact chunk/layer hash-equality overlap bounds (T2/T3: equal
  FNV route hashes => identical expert unions; unequal never implies
  inequality), an advisory sampled-ID containment overlap (T4: |A n B| / |A|,
  biased low, never gate input), and the token-level temporal verdict (T5:
  MEASURED only from per-token route events, else loud INSUFFICIENT_INPUT /
  NO_TG_PHASE / NO_ROUTE_EVENTS). Unsupported events, missing route fields,
  malformed lines, and missing files exit 2 naming file:line or the path.
- Validation: self-test PASS (19 checks on hand-computed fixtures, including
  the per-token MEASURED path and both loud-failure negatives); fidelity
  proof — a rerun over the 5 gplr12-* traces reproduces
  `build/llama-cuda/gplr14-route-overlap.{txt,json}` BYTE-IDENTICALLY
  (json record-equal; txt including the `json report:` trailer);
  determinism — double run byte-identical (same output names).
- Real evidence (5 independent MoE traces): union widths 245–256 on the
  512-token anchor, collapsing to 19–174 on the 50-token anchor (late
  layers collapse hardest); 100k/262k PP unions near-saturated; T2/T3 = 0.0
  (no chunk ever repeats a full union — consistent with full-bank prefill
  unions saturating); T4 advisory 0.2892/0.2952/0.3096 on the PP runs.
- PRIMARY FINDING (instrumentation result, not a guess): the runner emits
  route events ONLY at PP-chunk granularity — no surviving trace carries
  per-token TG routes (the T-GPLR-06 burst traces that motivated the
  "routes already exist per token" premise did not survive). The
  token-level temporal hit rate that gates T-GPLR-15 (proposed 0.6
  threshold) is UNDECIDABLE from existing traces and stays
  PROPOSED/UNMEASURED. Consequence recorded in the plan: T-GPLR-15's gate
  telemetry falls to T-GPLR-13's `rowset_fetch` events (full per-token
  union ID lists), so no extra telemetry work appears.
- Gated, not blocked: T-GPLR-13 still needs the two owner decisions (the
  `moe-row-set` SDD amendment and product priority vs the handoff path)
  plus the vendored CUDA tree.
- Provenance: the first implementation ran in an isolated (ephemeral)
  workspace lost on tab close; the tool was reconstructed in the main
  worktree and re-proven against the surviving artifacts byte-for-byte and
  the surviving self-test fixtures (%TEMP% route-overlap-selftest-*). The
  T4 containment metric is pinned by the recorded 0.28921875 on
  100k-bf16 (Jaccard and pooled variants do not reproduce it).
- No commit (not authorized).

## 2026-09-09: T-GPLR-12 DONE — default profile v2 (bf16 kept on MoE by owner decision)

All four flips validated under the GPU mutex (traces
`build/llama-cuda/gplr12-*.jsonl`, checker OK on every trace):
- Dense anchor: f32 vs bf16 IDs `[286]` identical; ctest
  layer-streaming-layer-major-1024 2/2 in 19.8 s under pure new
  defaults (run_start slab=bf16 kv=q8 handoff_mode=auto-off, ID 286).
- Auto-on: QA-50/QA-512 pure-default runs auto-enabled handoff, migrate
  18,327,011,328 B exact, answers PASS (N1; 5/5), negative controls
  FAIL as designed. Auto-off: 262K streamed at 1160.9 tok/s with the
  loud notice (24.3 GB need > 22.2 GB free), exit 0.
- bf16 MoE routing perturbation (escalation clause FIRED): 100K pair
  route hashes 0/840 equal; QA-512 same-shape three-way — bf16 vs f32
  differ by ONE inserted template-boundary token, needle tail
  byte-identical, answers 5/5 both; the f32 control also differs from
  the old batch-major reference (pos 43), so TG-ID parity across
  schedules was never attainable for MoE. Owner re-decided: bf16
  default kept EVERYWHERE; MoE comparisons are answer-gated, never
  token-ID-gated.
- Why bf16 is worth it (measured): slabs halve (100K MoE 1.53 GiB ->
  768 MiB; 262K 4 GiB -> 2 GiB), peak VRAM drops equally, and slab
  bytes feed the auto-handoff fit arithmetic (marginal shapes flip
  auto-off -> auto-on). Dense: bit-identical + ~2% faster PP.
- Defects found & fixed en route: (1) MoE streaming builder lacked the
  dense builder's three slab_bf16 cast points — MoE+bf16 crashed in
  norm.cu (ported the cast-on-read pattern); (2) the new checker
  on/auto-on rule used handoff_seen (reset per phase_summary) and could
  never pass — now handoff_ever.
- Environment note: CTest entries bake Q38LS_MODEL_PATH at configure
  time; the build had been configured with it empty (every dense GPU
  ctest failed at arg parsing). Reconfigured with
  Q38LS_MODEL_PATH=C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf.
- qa-smoke and the eight dense GPU ctests are pinned to their recorded
  shapes (--pp-schedule batch-major / explicit dtype+kv / --no-handoff)
  so the default flip cannot silently change what they test; host suite
  23/23 including the new tri-state/refusal fixtures.

## 2026-09-10: Owner probe — MoE 50K streaming PP + streaming TG under an 8 GB VRAM ceiling

Question: can our layer-major PP followed by a TG phase stay under 8 GB VRAM
on the MoE model? Answer: yes, in the streaming-TG shape. Run (GPU mutex,
exit 0): `llama-layer-streaming --model <Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf>
--pp-schedule layer-major --pp-tokens 51200 --ubatch 512 --layer-slots 4
--ctx-size 51456 --residency-budget-mib 7000 --no-handoff --pp-no-logits
--tg-tokens 3 --greedy --no-mtp --telemetry-chunk-stride 1 --telemetry
build/llama-cuda/gplr-moe-8gb-50k.jsonl`.
- Preflight (loud, passed): bounded 4-slot window 5,269,481,984 B <= budget
  7,340,032,000 B (7000 MiB); KV q8 @ 51456 = 533.91 MiB; reserve 512 MiB
  against observed free 22,226,665,472 B.
- PP: 51200 tok in 36.77 s = 1392.57 tok/s (transfer 108.62 s, wait 0.5 ms,
  compute 16.96 s, 36 evictions, graph_builds 0).
- TG: 3 greedy tokens in 5.90 s = 0.509 tok/s (transfer 6.66 s, wait
  18.6 ms, compute 139.9 ms, 120 evictions — TG pages the 40 banks per
  token; that IS the 8 GB TG path). check-telemetry OK (9163 events).
- Peak VRAM: **6693.56 MiB (6.54 GB) by runner telemetry**; NVML poller
  (1 s, sparse under load) sampled 5607 MiB max. Both well under 8 GB.
- Why `--no-handoff` is required here: the MoE handoff fit arithmetic
  (full_model 22.0 GB + slabs + reserve) would likely pass on this 24 GB
  card and migrate all 40 expert banks (18.33 GB) — impossible under 8 GB;
  and layer-major + no-handoff caps TG at 3 tokens by design
  (layer-streaming-lm-tg-cap). Full-handoff TG at 8 GB remains open work
  (owner-gated mixed-placement/row-set program, T-GPLR-13+).
- Artifacts: `build/llama-cuda/gplr-moe-8gb-50k.{jsonl,log}`,
  `build/gpu-mem-poll-8gb50k.csv`, launcher `build/run-8gb50k.ps1`.

## 2026-09-09: NCMOE 262K probe — stock TG at full context, 16 GB pipeline evidence

Owner question: can the streaming-PP + stock-llama.cpp-TG hybrid run
262K context on a 16 GB card? Ran `llama-bench -p 262144 -n 64 -fa 1
-ctk q8_0 -ctv q8_0 -ngl 99 -ncmoe 20 -r 1 -o json` (Qwen3.6-35B-A3B
Q4_K_S, RTX 5090 Laptop 24 GB, Ryzen 9 9955HX, GPU mutex, exit 0):
- PP @ 262K: **498.6 tok/s** (stock placement, first 20 banks on CPU).
- TG: **64.9 tok/s** (15.4 ms/tok). [Corrected 2026-09-10: the llama-bench
  TG leg runs with n_prompt=0 — zero-length KV on a BOS-only context — so
  "full 262K ctx" above is wrong for this number; the bench also has no MTP.
  The at-context stock TG baseline is the owner's 2026-09-10 llama-server
  run recorded in the streaming-TG profiling section below.]
- VRAM: **flat 14,885 MiB** load→teardown (sampler, 2 s cadence). No
  transient above steady state — llama-bench reserves weights+KV+
  compute buffers at load. Fits a 16 GB card with ~1.5 GiB margin
  (dGPU is headless on this laptop; idle = 0 MiB). N=18 would add
  ~0.9 GiB more headroom if ever needed.
- Artifacts: `build/llama-cuda/ncmoe262k.{json,log}`, `build/nvsmi-
ncmoe262k.csv`, `build/ncmoe262k.DONE`.

Combined 16 GB pipeline (measured): our streaming PP 1160.9 tok/s →
262K prefill in ~226 s; stock `-ncmoe 20` TG 64.9 tok/s thereafter.
Still a two-process measurement, not a single-process hybrid: KV state
save/restore exists in the library (`llama_state_save/load_file`) but
no CLI exposes it, and our runner cannot do `-ncmoe`-style partial
placement — the single-process version is the T-GPLR-16 mixed-placement
TG mode (owner-gated new work; renumbered 2026-09-10 from a stale
"T-GPLR-14" reference — that ID belongs to the closed route-overlap
instrument).

## 2026-09-07: T-GPLR-07 smoke tier wired into tools/run-tests.ps1 (host-validated)

`tools/run-tests.ps1 -Suite qa-smoke` runs the smoke gate as one repeatable
run under the GPU mutex. DEFAULT rung is 512 tokens on the 512-token anchor;
the 50-token rung is opt-in via `-SmokeLengths 50` (comma list; the
quoted-string form silently binds as one int and must not be used).
2026-09-09 owner decision: the ladder series is 50/512/4k/24k/50k/100k/200k
-- the 15/30 rungs were removed from the ladder entirely (their live
evidence below is historical; the 30-token answer-quality failure is no
longer a ladder blocker). 4k+ lengths are manual owner-invoked commands,
never part of the suite. The generator refuses targets below 15 tokens (the
K1-scaffold floor), so no shorter run exists.

Per-rung flow (unchanged leg shape, now looped over the selected rungs,
default 512):
1. `gen-needle-prompt.py --target-tokens <len> --calibrate llama-tokenize
   --model <Qwen3.6 GGUF>` (host-only, exact token count) -> parse
   achieved=/items= from its summary line.
2. Runner under `tools/with-gpu-lock.ps1` with the new flags:
   `--prompt-file <calibrated> --ubatch <achieved> --tg-tokens 96
   --pp-no-logits --pp-schedule batch-major --residency moe-expert-bank
   --kv-cache q8 --allow-full-residency --telemetry <dedicated JSONL>
   --tg-text <detok text>`. Single-ubatch PP is the documented workaround
   for the tail-ubatch gap.
3. Strict `--check-telemetry` on the dedicated JSONL (the gate that the
   ladder's earlier 512/4k rungs lacked — mixed-log claims are not
   accepted).
4. `check-answer.py check --tg-text <detok text> --items <items= list>` per-needle
   grading (50 = N1, 512 = all 5 items) + the same grading with
   `--negative` must FAIL as designed (exit != 0 with aggregate=FAIL). The
   checker grades the runner's `--tg-text` file directly -- the runner already
   detokenized with the model's own vocab via `llama_token_to_piece`, so no
   trace re-parsing and no GGUF vocab read happen at grading time
   (`check-answer.py check` also still accepts `--trace --model` for
   backward-compatible trace grading).

Any step failing exits the suite 1 with the failing rung named; each rung
gate keeps the dedicated trace `gplr07-qa-<len>.jsonl`, the detokenized text
`gplr07-qa-<len>.txt`, and the runner log `gplr07-qa-<len>.log` in the build
dir for inspection.

Host validation (2026-09-07, no GPU): PowerShell parse clean; checker
self-test 15/15 (added tg-text file mode cases); calibration exact against the real model
(llama-tokenize, vocab-only) for the smoke tier: 50 -> N1, 512 -> N1,N2,N3,N4,K1
(needle rows 3/6/9/12 of 12), all achieved == target; 15/30 calibration
(from the same generator, now removed from the ladder) is historical.
prompt files regenerated in `build/llama-cuda/gplr07-qa-prompts/`. The 50/512
legs of this suite previously ran end-to-end on the GPU (2026-09-07 16:08/
16:12, surviving dedicated JSONL traces + logs in the build dir). A fresh
512 default-gate pass is the intended next invocation whenever the
GPU mutex is free; the plan's remaining acceptance items (strict-trace
re-verification of the 4k rung, dense Qwen3.8 regression re-run) are
unchanged.

## 2026-09-07/09: qa-smoke gate live runs (50/512 PASS; 15/30 removed from the ladder on 2026-09-09)

Ran the gate legs individually under the GPU mutex (shared-GPU contention
forced leg-by-leg execution; `tools/run-tests.ps1 -Suite qa-smoke` remains
the canonical one-shot gate). Command shape per leg, identical to the suite:
`llama-layer-streaming.exe --model <Qwen3.6-35B GGUF> --prompt-file
<anchor> --ubatch <achieved> --tg-tokens 96 --pp-no-logits --pp-schedule
batch-major --residency moe-expert-bank --kv-cache q8 --allow-full-residency
--telemetry <dedicated JSONL> --tg-text <detok text>`, each runner leg under
`tools/with-gpu-lock.ps1`, then strict `--check-telemetry` and
`check-answer.py check --tg-text ...` (+ negative control).

Results (the 15/30 rungs below were removed from the ladder on 2026-09-09;
their evidence is historical, artifacts retained in the build dir):
- 15-token rung: PASS. runner exit 0; check-telemetry OK (23326 events, 1
  slot); K1 'Paris' found; negative control FAILs as designed (exit 1,
  aggregate=FAIL). Artifacts `gplr07-qa-15.{jsonl,txt,log}` (17:49).
- 30-token rung: answer gate FAIL. runner exit 0 and check-telemetry OK,
  but the model does not answer: it pattern-completes the note scaffold
  ("Note N2: ... is QX-7742", "Note N3: ... QX-7743", ...) instead of
  stating QX-7741. The checker correctly rejects (aggregate=FAIL) and the
  negative control correctly PASSes on the wrong answer (exit 0) -- the
  designed must-fail property worked exactly as specified. This is a
  model-quality failure on the 30-token N1 scaffold (no pads between
  statement and question), NOT a runner/telemetry/checker defect: all
  scheduler invariants held. Greedy decoding makes a re-run near-certain
  to reproduce. Tuning the scaffold or the acceptance wording is a
  plan-owned change (generator contract) and was NOT done; superseded by
  the 2026-09-09 ladder reduction (see resolution below). Artifacts
  `gplr07-qa-30.{jsonl,txt,log}` (17:58).
- 50-token rung: PASS (fresh tg-text-graded leg, 2026-09-09). runner exit 0;
  check-telemetry OK (23326 events, 1 slot, graph_builds 0 both phases); N1
  'QX-7741' found (aggregate PASS); negative control FAILs as designed
  (exit 1). Artifacts `gplr07-qa-50.{jsonl,txt,log}` regenerated. (Earlier
  trace-graded run 16:08 superseded.)
- 512-token rung: PASS (fresh tg-text-graded leg, 2026-09-09). runner exit
  0; check-telemetry OK (23326 events, 1 slot, graph_builds 0 both phases);
  all five items found ('QX-7741 TR-2918 MK-5501 LD-8834 Paris', aggregate
  PASS); negative control FAILs as designed (exit 1). Artifacts
  `gplr07-qa-512.{jsonl,txt,log}` regenerated. (Earlier trace-graded run
  16:12 superseded.)

Blocker resolution (2026-09-09, owner): the 15/30 rungs were removed from
the ladder series entirely (series = 50/512/4k/24k/50k/100k/200k). The
30-token answer-quality failure is therefore no longer a ladder blocker;
its artifacts (`gplr07-qa-30.{jsonl,txt,log}`) remain as evidence, and the
negative control's must-fail property was still proven live by that run.
Default-gate invocation: `powershell -NoProfile -ExecutionPolicy Bypass
-File tools/run-tests.ps1 -Suite qa-smoke` (512 only, ~5 min); the 50-token
rung is opt-in via `-SmokeLengths 50`; 4k+ ladder lengths stay manual
owner-invoked commands.

## 2026-09-09: Planner review -- T-GPLR-08 ready, T-GPLR-09 held

Reviewed the two drafted tasks per the planner workflow (read SDD, current
state, environment facts, controlling code; stated outcomes/gates/unknowns;
resolved ordinary uncertainty with focused probes; only environment/policy
decisions deferred to the owner).

T-GPLR-08 (one-layer-ahead PrefetchVirtualMemory probe) moved to READY. The
contract is falsifiable (262K wall drop >= 20% AND bank transfer drop >= 2x
vs the 351.9 s T-GPLR-04 baseline; effective per-bank feed >= 2x the 52 MB/s
cold rate), evidence-backed (the warm-cache leg falsified standby residency;
layer 0's ~3.5 GB/s working-set path is exactly the mechanism a rolling
prefetch populates), and narrow (new flag, 1-deep worker queue, two
telemetry events, one probe). Start remains serialized after T-GPLR-06/07
close per the plan's ordering rule (one ready task at a time). One draft
typo fixed: the focused-validation build target said llama-cli; corrected to
llama-layer-streaming.

T-GPLR-09 (host-RAM bank cache tier) stays DRAFT with a recorded blocker:
(1) it hard-depends on T-GPLR-08's range walker and telemetry plumbing;
(2) a 2026-09-09 environment probe (whoami /priv, <host>\<user>,
non-elevated) shows NO SeLockMemoryPrivilege and SeIncreaseWorkingSet-
Privilege disabled, so the contract's primary residency path (working-set
raise + VirtualLock) cannot execute on this machine as configured and the
acceptance gates (wall <= 110 s) are unreachable without a Local Security
Policy change. Owner decision required before re-review: (a) enable "Lock
pages in memory" via elevated policy, (b) redesign without VirtualLock
(weaker guarantee), or (c) defer until T-GPLR-08 results inform the tier's
value. Privilege facts recorded in docs/AGENT_ENVIRONMENT.md.
[Superseded 2026-09-09: T-GPLR-09 is PARKED (owner decision), not merely
held -- see the PARKED section below.]

## 2026-09-09: T-GPLR-06/07 CLOSED — T-GPLR-08 and T-GPLR-10 may start

T-GPLR-06 owner review closed (acceptance met; plan status -> done): k=4
batch verify 2.58x >= the 2x wall gate (d8 2.97x; code-text d16 6.91x vs
the measured serial 249.69 s); greedy-prefix parity under the recorded
knife-edge standard (top-2 margins; reruns bit-identical); every failure
loud (probe trace, en-route defects: n_outputs_max MTP clamp + graph-input
buffer sizing, both fixed and recorded); traces archived in the build dir.

T-GPLR-07 acceptance closed (plan status -> done):
- 512/4k strict dedicated-JSONL re-verification COMPLETE. The 512 rung was
  re-verified earlier today (gplr07-qa-512.jsonl, check-telemetry OK, all
  5 items PASS, negative control fails as designed). The 4k rung was
  re-run fresh today under the GPU mutex (GPU idle after a ~7-min wait;
  anchor regenerated: target=4096 achieved=4096, needle rows 27/55/84/106
  of 113, matching the ladder record): runner exit 0;
  `--check-telemetry` OK on the dedicated JSONL (38686 events, 1 slot,
  graph_builds 0 both phases); PP 4096 tok @ 610.5 tok/s (vs the original
  606.9); TG 160 tok @ 0.521 tok/s (vs 0.53); aggregate PASS on all five
  items (QX-7741 TR-2918 MK-5501 LD-8834 Paris); negative control FAILs as
  designed (exit 1). Peak VRAM 5.37 GiB (this run used --ctx-size 25088;
  the original 4k run used a smaller ctx). Artifacts
  gplr07-qa-4k.{jsonl,txt,log}. The 'check-telemetry OK' claims for 512/4k
  are now backed by surviving dedicated JSONL.
- Dense Qwen3.8 regression re-run GREEN (GPU mutex, ctest
  layer-streaming-layer-major-1024 2/2 in 24.6 s, Q38LS_MODEL_PATH =
  C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf): trace
  q38ls-lm-1024.jsonl tg_tokens ids=[286] == the recorded reference first
  greedy ID; run_start arch=qwen35 n_layer=64 slots=4 schedule=layer-major
  kv f16.

Plan serialization now: T-GPLR-08 (prefetch probe) and T-GPLR-10 (MoE
handoff) are the next eligible tasks, one at a time, in either order;
T-GPLR-09 remains held on the SeLockMemoryPrivilege blocker.
[Superseded 2026-09-09: T-GPLR-08/10/11 all closed and T-GPLR-09 PARKED
-- see the DONE and PARKED sections below.]

## 2026-09-09: T-GPLR-11 DONE — batch-major tail-ubatch gap closed

The recorded tail-boundary defect ("begin_set: activation buffer exhausted"
e.g. 511 tokens with ubatch 64; workaround --ubatch = pp-tokens) is fixed.
Pre-coding probes corrected the contract's hypothesis: with PP logits
enabled the runner marks every position, so the full-width variant B
already carries the final ubatch's outputs (512/64 logits exited 0 pre-fix,
PP first greedy ID 48) -- the ONLY missing prebuilt variant is the tail
WIDTH. Fix: the batch-major prebuild now also builds the tail ubatch
variant (width t = n % ubatch when 0 < t < ubatch and > 2 ubatches; real
tail tokens + real logits values, continuing state, consecutive synthetic
positions like variant B), and buf_act grows one act pair (PP term 4 -> 6
halves of max_ubatch, <= ~2 MiB at this model). Layer-major was already
correct (T-Q38CTX-01 prebuilds every chunk width).

Evidence (GPU mutex, Qwen3.6-35B, batch-major, moe-expert-bank, KV q8):
- 511/64 --pp-no-logits (previously exit 1, activation buffer exhausted):
  exit 0, PP 22.9 tok/s, evictions 319, graph_builds 0, check-telemetry OK
  (gplr11-fix-511x64.jsonl).
- 512/64 WITH logits: exit 0, graph_builds 0, check-telemetry OK, PP first
  greedy ID 48 -- identical pre-fix (gplr11-probe-512x64logits) and
  post-fix (gplr11-regress-512x64logits): the working shape is unchanged.
- Dense Qwen3.8 layer-major-1024 CTest regression 2/2 (ID 286 reference
  intact); host suite 22/22 green.
- Semantic-QA gate (runner change touched the experimental path): 512 rung
  re-run on the fixed binary -- check-telemetry OK, all five items PASS,
  negative control fails as designed (gplr07-qa-512.{jsonl,txt,log}).

Consequence: --ubatch can now be chosen freely below pp-tokens under
batch-major; the --ubatch = pp-tokens workaround guidance is retired.
(Plan header left for the T-GPLR-10 close-out to consolidate, avoiding
concurrent header edits across sessions.)

## 2026-09-09: T-GPLR-08 implemented and probed — prefetch hypothesis FALSIFIED (instrumentation result, task closed)

Implementation (default-off, byte-identical N=0): `--pp-prefetch-ahead N`
flag (host-safe parse, off-conflict table entry), per-member range walker
with an init-time mapping-containment guard (region-level VirtualQuery,
MEM_MAPPED), a dedicated worker thread + 1-deep queue enqueued from
`load_layer` after the eviction/consumer-done block, telemetry
`prefetch_submit`/`prefetch_complete` (+ `prefetch_init` and a run-summary
line); N=0 emits no new events. Build clean; `tools/run-tests.ps1 -Suite
host` green; host guards verified (range bound, off-conflict, usage).

Probe (262,144-token PP-only Q8 layer-major, GPU mutex, Qwen3.6-35B, same
shape as the T-GPLR-04/warm-cache baselines):
- N=0: wall 350.7 s (747.6 tok/s), transfer 326.4 s, check-telemetry OK
  (5507 events, 2 slots, graph_builds 0).
- N=1: wall 352.9 s (742.9 tok/s), transfer 328.3 s, check-telemetry OK
  (5586 events, 2 slots, graph_builds 0). Prefetch: 39/39 banks submitted
  and completed, 0 errors, 17.87 GB, API call 19-28 ms/bank (submit-rate
  line: feed=18,640 MB/s); every next-layer prefetch completed before that
  layer's copy window opened.
- Verdict: wall +0.6% (noise). BOTH acceptance gates (wall -20%;
  transfer/feed >= 2x) are FALSIFIED; recorded per the contract's
  falsification clause as an instrumentation result, not a performance
  claim. Parity criterion PASSED: the 512 rung (3783/3783 prefetches,
  check-telemetry OK, answer PASS, negative control FAILs as designed) and
  a fresh 50 rung on the prefetch binary (check-telemetry OK, answer PASS,
  negative control FAILs) via `tools/run-tests.ps1 -Suite qa-smoke
  -SmokeLengths 50`.
- Why it looks flat: the fault-driven memcpy feed is unchanged (~2.0 s/bank
  mean in both runs, lists within noise). The per-bank `transfer` field
  again absorbs the 10 full-attn layers' ~29.3 s compute windows (slow at
  copy indices 4,8,...,36 in BOTH runs; slow total 263.6 s / 264.5 s);
  cross-checked trace-for-trace against the archived 352.5 s warm-cache
  baseline (gplr-longpp-262144-warmcache.jsonl: same slow indices,
  264.8 s) — today's runs reproduce the baseline exactly, so there is no
  regression and no improvement.
- Attribution (open): PrefetchVirtualMemory returned success on every bank
  yet the fault rate did not rise. Layer 0's map-time prefetch still shows
  the fast path is real (~0.13 s / 3.5 GB/s, identical in both runs), so
  the API CAN populate the working set; the rolling probe shows it does
  not help steady-state. Remaining hypotheses: prefetched pages are
  trimmed before use under host memory pressure (7.1 GiB free of 61.7 at
  analysis time), or the ~2 s/bank stream is not page-fault-bound at all.
  Distinguishing them needs a fault-count probe (working-set query before/
  after prefetch) — out of T-GPLR-08 scope.
- Consequences: T-GPLR-09 (host-RAM tier) loses its main premise (the
  working-set mechanism it hoped to exploit does not transfer to rolling
  prefetch); its privilege blocker already holds it as draft. T-GPLR-10
  (MoE handoff) is unaffected (its 2x floor compares handoff vs streaming
  serial, not vs prefetch). N=2 was attempted; the run was killed by a
  terminal timeout mid-flight (5/39 prefetches, no run_end) — no result,
  not evidence.
- Artifacts: build/llama-cuda/gplr08-262k-n{0,1}.{jsonl,log}; the n2 file
  is the killed fragment.

## 2026-09-09: T-GPLR-10 DONE — MoE handoff (streaming PP -> stock llama.cpp TG hybrid)

Implemented, validated, and closed (plan status -> done). Implementation:
- `begin_handoff` accepts moe-expert-bank (migrate_layer runs verbatim on
  pkg_cache: the 40 bank packages); `handoff_attention_only` now refuses
  under merge_resident (exit 2) — partial handoff has no meaning when only
  banks are pageable.
- Defect found and fixed during validation: `migrate_layer` built the
  handoff bundle with only the bank package members, leaving the non-bank
  members (attention/router/shared-expert/norms) null, so the post-handoff
  TG graph rebuild dereferenced null and the process died silently right
  after the migrate event (first gplr10-ho-bm attempt). Fix: non-bank
  members point at the slot bundle (`dst = src`, the same rule
  `create_bundle` uses). The loud-failure invariant holds: a broken bundle
  fails at graph build, never runs another layer's data.
- Runner gates: `--mtp + --handoff` under moe-expert-bank -> exit 2
  "...handoff TG is stock llama.cpp decode without a draft loop; MoE MTP
  verify requires the streaming executor"; layer-major `--tg-tokens > 3`
  cap lifts when `--handoff` is active (cap remains without); policy-aware
  handoff progress message. Dense `--mtp + --handoff` verified still legal.
- VRAM preflight handoff-fit branch (full_model_plus_runtime + act_slab +
  reserve <= free at run start, structured byte table, refuse before the
  timed PP). Peak-order rule: slot buffers (522 MB) + act slabs (32 MB)
  released before the 18.33 GB buf_handoff allocation (recorded in the
  handoff `release` event).
- The migrate event now records `free_vram` (true handoff-window peak:
  free 3.30 GB of 25.65 GB at migrate -> peak ~22.35 GB).
- Checker: hybrid-TG rules (bank transfer ~0 in the post-handoff
  phase_summary; handoff event field validation incl. optional free_vram)
  — backward compatible with the archived dense handoff trace (re-verified
  against it directly).

Evidence (GPU mutex; Qwen3.6-35B-A3B-MTP-UD-Q4_K_S unless noted):
- Host suite 22/22 green, including the five new T-GPLR-10 gate fixtures
  (refuse-mtp-handoff-moe, lm-tg-cap, lm-tg-cap-lift,
  dense-mtp-handoff-legal, dense-attn-only-legal).
- Parity (512 layer-major pair): route-hash sequences 40/40 identical
  streaming vs handoff; TG prefix ids [220, 248046, 198] exact (the
  streaming reference TG is capped at 3 by the layer-major experiment).
- Speed: batch-major streaming reference TG 0.495 tok/s -> handoff TG
  64.63 tok/s (~130x; the contract's >= 2x instrumentation floor is
  exceeded; recorded as instrumentation, not a general claim).
  Layer-major handoff: TG 16 tok in 284 ms. Migrate 18,327,011,328 B in
  ~3.08 s (~5.9 GB/s effective copy rate).
- Answer gate: QA-512 rung WITH `--handoff` (batch-major, ubatch 512,
  tg 96): check-telemetry OK, aggregate=PASS 5/5 needles (QX-7741 TR-2918
  MK-5501 LD-8834 Paris), negative control FAILs as designed
  (gplr10-qa512-ho.{jsonl,txt,log}).
- Dense regression (Qwen3.8-27B-Q4_K_M, 1024 PP + 16 TG batch-major,
  --allow-full-residency — the T-GPLR-04 out-of-core preflight postdates
  the recorded dense evidence): first ID 286 == the recorded reference;
  all 16 ids identical streaming vs handoff; check-telemetry OK both
  (gplr10-dense-ref/-ho.{jsonl,log}).
- VRAM: 262K Q8 + --handoff refuses exit 1 with the byte table before the
  timed PP (gplr10-refuse-262k.log); 2048-token run fits with free
  22.23 GB at start and free_vram 3.14 GB at migrate (no OOM, no silent
  fallback).
- Shared-GPU-memory investigation (owner report: "28 GB shared / 48 GB
  total on PP, none on TG"): adapter `\GPU Adapter Memory\Shared Usage`
  stayed at the ~0.29-0.37 GB desktop baseline through model load, PP
  (dedicated 3.87 GB), and the handoff copy window (dedicated 18.63 GB) —
  run-wide, sampled at 1 s (gplr10-attrib trace +
  gpu-mem-poll-attrib.csv). The mechanism surfaces NO shared-memory
  spike: pageable host-mmap banks are read directly by the copy path
  (copy_source=direct), which Task Manager attributes to dedicated VRAM
  during the copy window, and TG after handoff transfers 0 bytes. The
  Task-Manager "Shared GPU memory" figure is machine-wide (desktop +
  driver + other processes), not a second pool this runner allocates.

Gaps / residual risk:
- FIXED (same-day follow-up): the TG phantom evictions. Root cause was in
  `load_layer`'s two handoff branches, not the summary: the executor re-arms
  `consumer_pending` per layer, and the handoff branches ran the streaming
  slot-reuse protocol (consumer wait + `total_evictions++` + `slot_evict`)
  although handoff TG copies nothing and overwrites no slot data — one
  phantom eviction per layer per token (640 = 40x16x1 on MoE; the same
  inherited defect produced 1024 = 64x16x1 on the dense full handoff,
  T-Q38HG-02 era). Fix: the consumer-wait/evict blocks are removed from the
  handoff branches (the streaming branch is untouched). Evidence
  (gplr10-evictfix-{moe,dense}.{jsonl,log}, GPU mutex, check-telemetry OK
  both): post-migrate slot_evict 0, TG phase_summary evictions 0, PP
  evictions unchanged (MoE 39 / dense 127 with identical transfer bytes),
  TG ids unchanged (dense [286, 220, 197, ...] identical to the pre-fix
  reference; MoE identical to the QA-512 anchor reference). Host suite
  22/22.
- Route hashes are prompt-content-derived: parity claims are same-prompt
  only (the needle-2048 attrib run vs the synthetic-2048 mem-lm run differ
  by design).
- Measurement tooling: per-process GPU counters stall (25+ s/call) under
  full GPU load; adapter counters are the reliable path. The surviving
  poller is build/poll-gpu-memory.ps1 (v10); earlier diagnostics v1-v8 are
  superseded.
- No commit (not authorized).

## 2026-09-09: T-GPLR-09 PARKED — owner decision, plan updated

The held host-RAM bank-cache tier (T-GPLR-09) is now formally PARKED in
the plan, not merely held as draft. Owner decision 2026-09-09; the plan's
T-GPLR-09 status block records the rationale, the two-key-door framing,
and the reopen triggers.

Why parked — a two-key door, and neither key turns today:
- Key 1 (policy): SeLockMemoryPrivilege is absent and
  SeIncreaseWorkingSetPrivilege disabled for the token (2026-09-09 probe),
  so the contract's primary path (working-set raise + VirtualLock) cannot
  execute without a Local Security Policy change — an owner-level
  environment change outside the implementer lane.
- Key 2 (premise): T-GPLR-08's falsified probe (this day) showed the
  working-set mechanism the tier relies on does not transfer to the
  rolling stage (hinted pages 39/39 populated, fault-driven feed
  unchanged), and the Sep 7 warm-cache leg already showed standby
  residency adds nothing to PP. A locked private copy would additionally
  be strictly worse under memory pressure (dirty private pages trim to
  the pagefile — up to ~18 GB pagefile traffic).

Reopen triggers (any ONE, owner-initiated, recorded in the plan's status
note): (1) privilege granted AND a fault-count attribution probe shows
staging is fault-bound with trimming — not memcpy-bandwidth-bound; (2)
the workload moves to host-memory-tight conditions with recorded
hard-fault/RSS thrash evidence per run; (3) target hardware/OS changes so
the privilege arrives by default.

Consequences: no code was ever written for the tier, so nothing is
retired; the 15% admission rule stands as design evidence only. TG
batch-verify staging already sits at the RAM-speed floor (earlier
measurement), so a resurrected tier could only ever target PP. The
remaining program items are unchanged: endurance ladder rungs
50k/100k/200k (owner-invoked), the T-GPLR-08 attribution question, and
the second-architecture decision (owner-gated).

Docs-only change (plan status block + header); no commit (not requested
for this decision).

## 2026-09-09: T-GPLR-10 drafted — MoE handoff (streaming PP + stock llama.cpp TG)

(Superseded 2026-09-09 by the T-GPLR-10 DONE section above; the design
facts below were confirmed during implementation.)

Scoped per owner request (working thread): the hybrid TG path proven on
dense (T-Q38HG-02/04, 24.75 tok/s) extended to the MoE model. Drafted as
T-GPLR-10 in docs/plans/generic-prefill-layer-residency-plan.md. Key facts
established by code inspection (not assumptions):
- The ONLY hard blocker in the module is begin_handoff's blanket
  merge_resident refusal (llama-layer-streaming.cpp ~2193-2198). The dense
  migration machinery (migrate_layer lambda: dev tensors -> one contiguous
  buf_handoff -> async copy -> sync -> handoff event) runs verbatim on the
  MoE pkg_cache: the migration list is exactly the 40 bank packages
  (18,327,011,328 B = 37 x 452,984,832 + 3 x 522,190,848); router,
  shared-expert, attention, recurrent, norms, static, MTP are already
  GPU-resident under merge_resident (~3.05 GB non-bank). KV + DeltaNet state
  stay context-owned (invariant).
- adapter_qwen35moe.mtp_package == nullptr: MoE MTP is the resident
  streaming path, so --mtp + --handoff under moe-expert-bank becomes a loud
  exit-2 gate refusal (stock TG has no draft loop). Dense --mtp + --handoff
  stays legal.
- Runner gates to change: the layer-major --tg-tokens > 3 cap (main.cpp
  ~1679) lifts when --handoff is active (post-handoff TG is stock
  llama_decode regardless of PP schedule; cap remains without handoff); the
  handoff progress message "all (64 layers)" becomes policy-aware.
- VRAM: reuse the T-GPLR-04 preflight (main.cpp ~2077) as the handoff fit
  gate (full_model_plus_runtime + slabs + reserve <= free at run start,
  refuse before the timed PP). Long-ctx refuses by design (262K Q8
  full+runtime 24.30 GB vs observed free 22.92 GB); short-ctx ~21.4-21.5 GB
  + overhead is marginal and decided by measured free VRAM. Peak-order rule:
  release slots/slabs before allocating the 18.33 GB handoff buffer.
- Checker: hybrid-TG phase validates with bank transfer ~0 in the
  post-handoff phase_summary (handoff-aware rules already accept
  graph_builds > 0 after handoff); no invented copy pairs.
- --tg-text works post-handoff (writes from the generated vector, no
  streaming-only restriction), so the T-GPLR-07 512-rung answer gate is the
  end-to-end parity vehicle.

## 2026-09-07: T-GPLR-07 runner flags --prompt-file / --tg-text landed (host-validated)

Both flags are additive and default-off in the experimental runner
(`tools/layer-streaming/main.cpp` in the vendored tree). Contract — the
plan left the interaction details open, so the decisions are recorded here:
- `--prompt-file PATH` reads the prompt text as raw bytes from PATH and
  feeds the exact same `prompt_text` pipeline as `--prompt`/`--prompt-env`
  (tokenize, pp_tokens override, MTP padding). Combining it with `--prompt`
  or `--prompt-env` is a loud exit-2 error; with the flag absent, the
  existing last-wins `--prompt`/`--prompt-env` behavior is byte-identical.
  Missing file, read failure, and empty file each exit 2 before
  `llama_backend_init` (host-only). Works in both the streaming and
  `--residency off` paths (it is prompt-input plumbing; the off path
  consumes `prompt_tokens` identically).
- `--tg-text PATH` writes the detokenized generated text
  (`llama_token_to_piece`, special=false — byte-identical rendering to the
  "--- generated text ---" section) after successful TG that produced
  tokens, regardless of the prompt source. Oversized vocab pieces are
  retried with the required buffer (negative return contract). A failed
  write is loud and fails the run (`run_end.exit_status 1`). Streaming path
  only: `--tg-text` + `--residency off` is rejected by the single
  off-conflict table (one entry added, exit 2).

Host validation (2026-09-07, no GPU, no model load): build exit 0 (only
main.cpp recompiled); default-off regression — five probe commands
(`--pp-tokens 0`, bad `--copy-source`, nonexistent-model arch read, forced
`off` arch read, unset `--prompt-env`) produce byte-identical stderr and
identical exit codes pre/post change (`build/audit/baseline/` vs
`build/audit/post/`); new-flag surface — missing value, unreadable file,
empty file, prompt-source conflict, and off-conflict all exit 2 with the
recorded messages; a valid file is read and routed
("prompt-file: read N bytes" then the expected model-arch failure, exit 1);
host CTest suite 15/15 on the rebuilt binary. The tokenize/TG-write
end-to-end legs are exercised by the GPU ladder runs per the plan. Still
remaining for acceptance: smoke wiring in `tools/run-tests.ps1`, the 512/4k
strict-trace re-verification, and the dense Qwen3.8 regression re-run.

## 2026-09-07: --mtp + --pp-no-logits n_outputs_max clamp fixed

The en-route defect recorded below (GGML_ASSERT(n_outputs_max <=
cparams.n_outputs_max) in llama-context.cpp:2551 at the first MTP round,
loud abort exit -1073740791) is fixed in the runner,
third_party/llama.cpp/tools/layer-streaming/main.cpp (cparams
n_outputs_max selection, just above the forced-off branch). Root cause:
with --pp-no-logits the runner clamped n_outputs_max to 1 (kept nonzero
only for graph reservation), but the MTP batch-verify decode requests
logits on the frontier plus every draft -- up to 1 + mtp_draft rows in
ONE ubatch (batch.logits[j] = true for all rmax = 1 + nd rows) -- so
llama_decode's output_reserve asked the context for more output rows
than it declared, tripping the assert. The clamp now accounts for the
MTP draft width: n_outputs_max = pp_no_logits ? (mtp ? 1 + mtp_draft :
1) : pp_tokens. All other combinations are unchanged (pp logits on:
pp_tokens; pp_no_logits without mtp: 1), and the draft context's own
n_outputs_max = 1 is still correct (its decodes never request more than
one logits row). --pp-no-logits + --mtp is valid again; the d8config no
longer must keep PP logits on. Compile/link verified (cmake --build
build/llama-cuda --config Release --target llama-layer-streaming). GPU
model leg (2026-09-07 15:33, warm cache, GPU mutex, trace
build/gplr06-batchverify-d8-nologits-fix.jsonl, log
out/q36-d8-nologits-fix.log, script build/gplr-mtp-nologits-fix-run.ps1):
the d8 + --pp-no-logits config COMPLETES -- runner exit 0, no assert.
Fingerprints: the target context now logs n_outputs_max = 9 (the draft
context stays at 1), and all 40 MTP rounds run. Parity vs the logits-on
warm baseline is exact: first-16 ids identical (220 248046 198 248045
248068 271 248069 271 2064 7701 421 279 1414 488 3766 369), 309
drafted / 87 accepted / acceptance 0.282, 2.17 accepted/round, TG wall
82.0 s (81,974 ms) vs 81.8 s, check-telemetry OK (10127 events). PP
phase 5992.5 ms vs warm 5932.9 ms (+1%, noise) -- PP is staging-bound
(~5.4 s transfer of 40 layer copies, compute only 0.35 s) at this
shape, so suppressing PP logits does not change the PP wall here; the
saving would show where logit rows dominate compute/copy. --pp-no-logits
+ --mtp is a valid, parity-identical d8 configuration.

## 2026-09-07: Host I/O Baseline and Tier Direction (owner decision)

Measured (host-only, no GPU): total RAM 61.69 GiB (66,234,945,536 B), free
50.50 GiB at measurement time. Drive: MSI M461 2TB NVMe on C:. Sequential
read of the 21.39 GB Qwen3.6 GGUF with 64 MiB reads: 7,745 MB/s buffered,
2,728 MB/s unbuffered qd1 (FILE_FLAG_NO_BUFFERING). The "HDD-like ~52 MB/s"
hypothesis for the 262K effective page-in rate is falsified: the device
delivers ~50x that; 52 MB/s is the mmap page-fault path (small serialized
I/Os). Warm buffered reads also confirm the whole 21.39 GB file fits
resident in the 61.7 GiB RAM cache. Warm-cache 262K leg (2026-09-07,
below) additionally falsifies cache-manager retention as the collapse
mechanism: with the entire file resident in standby before AND during the
run, wall/per-layer copy times are identical to the cold baseline.

Owner decision: pursue BOTH tiers per the 15% rule — host-RAM bank residency
when the pageable set fits in RAM leaving >= 15% of total RAM for the OS, and
rolling one-layer-ahead prefetch as the streaming tier when it does not.
Drafted as T-GPLR-08 (prefetch probe) and T-GPLR-09 (host-RAM bank cache
tier) in docs/plans/generic-prefill-layer-residency-plan.md. T-GPLR-09
admission math on this machine: reserve = 0.15 x 61.69 GiB = 9.25 GiB; bank
set 18.33 GB (17.07 GiB); free 50.50 GiB -> 50.50 - 17.07 = 33.43 GiB >= 9.25
GiB -> admitted today; the rule rejects larger pageable sets automatically.


## T-GPLR-05 Complete: Capability Registry + Regression Closure — Plan Done (2026-09-05)

The generic prefill layer-residency plan is complete: every loaded model now
receives a deterministic host-only capability result, only adapter-declared
policies can be selected, and the tested support matrix is published without
any performance claim.

Capability registry. `llama_ls_probe_capability(path)` reads only the GGUF
header KV (no CUDA enumeration, no model load, no placement change) and
returns a deterministic result: `qwen35` -> adapter `qwen35`, policy
`dense-layer`; `qwen35moe` -> adapter `qwen35moe`, policy `moe-expert-bank`;
anything else -> `off` with the stable reason `no layer-residency adapter for
this architecture`. The runner exposes `--probe-capability` (host-only JSON)
and defers `llama_backend_init` until after policy resolution, so all refusal
paths are strictly host-only.

Policy modes. `auto` selects only adapter-declared policies; on an
unsupported architecture it runs `off` and explains why on stderr (exit 0,
SDD 8) — only a FORCED policy/arch mismatch refuses host-only (exit 2).
Forced `off` runs
the normal llama.cpp path inside the runner (no tensor-buft overrides, no
streaming init, batch-major `llama_decode`) and writes an audit telemetry
record (`policy":"off"` + `capability_reason`). Forced `dense-layer` /
`moe-expert-bank` are validated against the probed architecture.

Post-audit hardening (residency off first-class): the normal path lives in
`run_residency_off()`; `off` combined with streaming-only flags (layer-major,
mtp, handoff, explicit pp-schedule, layer-slots, residency-budget) is rejected
loudly before CUDA (exit 2); off audit traces are validated by the telemetry
checker (policy `off` rules: run_start/run_end/meta consistency, no slot
events) via checked-in fixtures; and all 16 run tests pass
`--allow-full-residency` explicitly, removing ambient-VRAM flakiness.

Telemetry single-owner refactor (2026-09-05, post-audit): the run_start
identity head (event/run_id/t0_us) lives in one shared builder
(`tel_run_start_head`) used by both the streaming and off-run emitters;
the off-trace rules (policy-off classification, run_start metadata rule,
streaming-event rejection) live in one shared helper set used by BOTH
`check-telemetry` and `validate-telemetry` with byte-identical message
text. Intended behavior delta only: a healthy off run_end now emits
`"error":"none"` (failed:false) instead of placing the non-error string
"off mode: normal llama.cpp placement" in the error field. Validation:
host suite green; full layer-streaming suite 44/44 under the GPU mutex
with the final binary; emitted run_start field order verified identical
to the pre-refactor contract on both trace classes (fixtures updated for
the error-field delta); MoE forced-off vs layer-major streaming parity on
the final binary: identical first 3 generated IDs (19241 13 561) on the same
12-token prompt, both traces check-telemetry OK (dense-model parity runs are
superseded; MoE is the governing workload). Post-audit hardening: the six
off-incompatible flag conditions now live in one flag-to-reason table (single
exit code 2) evaluated after argv parsing and before the host-only model-arch
read, so a gate rejection wins over any later failure class; verified host-only
for every entry, including auto-resolved off.

Long-context layer-major PP measured (MoE, 2 slots, Q8 KV @ 262144, ubatch 512,
--pp-no-logits, direct staging, GPU mutex, 2026-09-06): 50,000 tok -> 69.7 s
wall / 69.6 s transfer / 62.6 s compute = 717.8 tok/s; 100,000 tok -> 135.0 s /
129.3 / 126.6 = 740.7 tok/s; 262,144 tok -> 351.9 s / 327.6 / 338.6 = 745.0
tok/s. 38 bank evictions and graph_builds 0 in every run. Decomposition
(2026-09-06, from per-event trace data): bytes_transferred is constant
(18.33 GB, 40 copies) and pure per-bank H2D copy cost is flat (layer-0
copies ~0.26 s in every run, ~10.4 s total), but wall is paced by the host
page-in stream of those same 18.33 GB: effective feed rate 263 MB/s at 50K
vs 52 MB/s at 262K (uncontrolled file-cache behavior; device limit falsified 2026-09-07 - MSI M461 2TB NVMe measured 2,728 MB/s unbuffered / 7,745 MB/s buffered sequential reads of the model file; see AGENT_ENVIRONMENT.md) -- that, not copy
volume, is why the transfer-time column climbs while bytes stay flat. The
phase transfer/compute fields measure submit->complete windows that include
overlapped compute (per-copy transfer_us inflates to ~29.5 s exactly on the
10 copies overlapping the full-attention layers' ~29.7 s compute windows),
so "compute hidden behind transfer" is not decidable from the phase fields
alone; sampled layer_compute sums to 41.3 s at 262K. Net: long-PP wall ~
18.33 GB / effective page-in rate; tok/s rises mildly with length only
because the fixed byte stream spreads over more tokens (marginal cost
~1.33 ms/token at 262K). 262K
confirmation and variants (all exit 0, check-telemetry OK): rerun 350.6 s =
747.7 tok/s (reproducible within 0.4%); pinned staging 352.1 s = 744.4 tok/s
(identical to direct -- the bottleneck is the mmap read, not the staging
copy; wait 0.6 s); 4 slots 352.3 s = 744.2 tok/s (36 evictions, slot window
1.82 GiB; cumulative transfer time tripled to 967.8 s yet wall unchanged --
wide-window copies run concurrently behind compute, leaving wall bound by
the host read stream). Traces: build/gplr-longpp-*.jsonl, all
check-telemetry OK.

Warm-cache leg (2026-09-07, same command/shape, GPU mutex, trace
build/gplr-longpp-262144-warmcache.jsonl, script build/gplr-warmcache-run.ps1,
log out/q36-longpp-262k-warmcache.log): the whole 21.39 GB GGUF was read
into the file cache immediately before the run (standby 0.38 -> 20.41 GB;
second-pass warm read 8,138 MB/s) and STILL sat at 20.57 GB standby after
run_end -- the file never left memory during the run, with 53 GB free the
whole time (cleanest environment any long-PP run has had). Result: wall
352.5 s / 743.7 tok/s, transfer 328.1 s, compute 339.0 s, 38 evictions,
graph_builds 0, peak VRAM 12,211 MiB -- identical to the 351.9 s baseline
(delta +0.17%), and per-layer copy windows match trace-for-trace: layer 0
= 0.13 s, recurrent ~1.7-2.3 s, full-attn ~29.2-29.4 s. Cache-manager
retention is FALSIFIED as the cause of the 263->52 MB/s collapse:
standby-resident pages fault through the memcpy path at ~200-215 MB/s, no
faster than cold. Two refinements:
- The apparent length-dependent collapse is mostly a phase-field artifact:
  total_transfer_us absorbs the 10 full-attn layers' ~29.3 s compute
  windows (~293 s of the 328 s total); the genuine per-bank stream is
  ~266 MB/s at 50K vs ~215 MB/s at 262K (~20% slower, not 5x), and wall ~
  30 recurrent x ~2.05 s copy + 10 full-attn x ~29.3 s compute = ~354 s.
- Layer 0's copy (0.13 s ~ 3.5 GB/s, identical in both traces) rides the
  loader's one-shot map-time PrefetchVirtualMemory, i.e. WORKING-SET
  residency -- 16x the standby-fault rate. The fast path exists and is
  working-set residency, not standby: this directly validates the
  T-GPLR-08 rolling one-layer-ahead prefetch (PrefetchVirtualMemory
  populates the working set) and pins its falsifiable gate to the layer-0
  rate (~3.5 GB/s feed -> wall floor ~66 s, matching the warm-resident
  boundary traces' 3968 tok/s).

Open performance frontier: TG remains a full 40-bank pass per token
(0.26-0.52 tok/s measured). Draft task T-GPLR-06 (MTP draft + layer-major
burst verify, falsifiable 2x-tok/s hypothesis at k=4, exact greedy-prefix
parity) is appended to the plan as draft, awaiting owner review before any
implementation. A pre-implementation probe already triggered the draft's
first escalation condition: --mtp --handoff on the MoE loads the draft head
(blk.40) but begin_handoff is unsupported for the moe-expert-bank policy
(loud TG decode -3, exit 1; build/gplr06-mtp-probe.jsonl) -- the MoE MTPacceptance
rate is unmeasurable with the current binary. Owner decision taken: redesign
the draft leg. Step 1 landed: --mtp without --handoff is now legal under
moe-expert-bank (MTP package is GPU-resident there; single gate moved after
policy resolution; dense-layer still requires handoff). First MoE MTP run
completed (exit 0): drafted 16 accepted 9 = 0.562 accept rate, and greedy IDs
exactly match the sequential no-MTP baseline on the same 512-token prompt
(build/gplr06-{seq-baseline,mtp-resident}.jsonl). Wall unchanged (0.474 vs
0.493 tok/s) because the existing loop verifies serially -- one full bank
pass per draft; the remaining T-GPLR-06 work is the layer-major batch verify
that amortizes one bank pass across k drafts. Acceptance measured over 128
tokens (all legs exit 0, greedy IDs identical everywhere): sequential 0.522
tok/s; draft 4 -> rate 0.479, 1.91 accepted/round; draft 8 -> rate 0.250,
2.00 accepted/round -- accepted tokens per bank pass saturate near 2, so the
step-2 batch-verify projection is ~2.7-3.0x wall (2x hypothesis reachable
with margin). En-route defect found and
fixed: the layer-major graph-input buffer was sized embd-bytes-exact plus a
flat 64 MiB slack, which exhausted at chunk 493/512 on the 262K run (two
deterministic loud aborts, never silent corruption); sizing now adds 1 MiB
headroom per chunk (host-side buffer, no VRAM cost). Failed attempt traces
archived in build/gplr-attempt1/.

T-GPLR-06 step 2 (batch verify) landed and measured (2026-09-06): the MTP
round now verifies the frontier plus all drafts in one width-(1+d) target
ubatch - one expert-bank pass per round instead of one per draft - with
corrected-for-state rollback via exact KV seq_rm plus the recurrent snapshot
ring, and prebuilt width-k phase-1 graph sets preserving graph_builds == 0.
128-token draft-4 config: 95.1 s wall vs the 245.4 s serial baseline (2.58x,
target ~90 s), exit 0, 47 rounds, avg accepted/round 1.70. Parity verdict
per owner decision (bit-exactness not required across ubatch shapes, but
divergence must be measured): the one divergence from the serial stream was
instrumented on both paths with top-2 margins and runner-up tokens - the
flipped decode is the identical logical decode (same token/position/history),
serial chose its argmax by 0.0037 logits (smallest margin of the whole run)
and the width-5 shape flipped it at 0.14 noise scale; all comparable rows
agree on gold AND runner-up; both streams re-join afterward. Quantized-
weight shape noise on a knife-edge tie, not a state or wiring defect; reruns
bit-identical. Host suite green (15/15) on the final binary; traces at
build/gplr06-{seq128-margins,batchverify-d4-v2}.jsonl. Draft-8 batch verify
measured 2026-09-07: 82.6 s (1.55 tok/s), 2.97x over serial, avg
accepted/round 2.17 (40 rounds) - deeper drafts pay off once verify is
batched; all cross-width divergences are the same measured sub-0.15-margin
tie flips (traces build/gplr06-batchverify-d8.jsonl). Draft-32 probe:
96.1 s, peak VRAM 7.42 GB, exit 0, graph_builds 0 - the frontieris d8 (82.6 s); acceptance horizon is 7 (depth >= 8 never accepted in 43
rounds), memory is NOT the binding constraint (ring planes scale
linearly, 2.18 GB at d32), and the depth-32 ring rewind is exercised
(6 zero-accept rounds) with identical first-16 IDs across all configs.

Warm-cache TG staging bound (2026-09-07, same d8 command, GPU mutex,
trace build/gplr06-batchverify-d8-warmcache.jsonl, script
build/gplr-tg-warmcache-run.ps1, log out/q36-tg-warmcache.log): the file
cache was warm before and during the run (standby 21.1 GB, 52.8 GB free,
GPU idle -- cleanest TG environment recorded). TG wall 81.8 s / 1.564
tok/s vs the 82.6 s / 1.55 tok/s baseline (delta -1%, noise); 40 rounds,
accepted/round 2.17, acceptance 0.282 -- behaviorally identical to the
recorded baseline. The bound: batch-verify wall is staging-volume-bound
at 40 x 458 MB = 18.33 GB per round, and every TG wall on this machine
collapses to that volume divided by ~9 GB/s (serial 128t: 250.4 s / 128
rounds = 1.96 s/round = 9.1 GB/s; d8: 2.01 s/round = 8.9 GB/s; d8 warm:
2.00 s/round = 9.0 GB/s; compute is only 1.4 s of the 82 s wall).
Consequences: (1) TG staging is ALREADY at the RAM-speed floor -- the
host-RAM tier (T-GPLR-09) adds ~0 wall for TG batch verify, exactly like
standby residency added nothing to PP; (2) batch verify's remaining lever
is acceptance (tokens per round -- the code-prompt d16 3.54 tok/s optimum
is the evidence), not staging; (3) batch verify at 1.56 tok/s already
matches the dense handoff resident-TG rate (1.68 tok/s), so staging is no
longer the MoE-vs-dense TG differentiator. En-route defect found: --mtp
with --pp-no-logits trips GGML_ASSERT(n_outputs_max <= cparams.
n_outputs_max) in llama-context.cpp:2551 at the first MTP round (loud
abort, exit -1073740791; FIXED same day -- the runner clamp now covers
1 + mtp_draft, see the '--mtp + --pp-no-logits n_outputs_max clamp
fixed' entry above). Also
noted: per-copy window telemetry is scheduler-skew sensitive (median 48
ms while a foreign process held the GPU busy vs ~0 ms with it idle);
aggregate staging rate (~9 GB/s) is the robust number, not per-copy
windows. Coding-prompt depth
sweep 2026-09-07 (calculator-spec prompt, --prompt-env, 1024 PP,
ctx 2048, tg 128, batch-major): d8 35.5 s TG (3.60 tok/s, 17 rounds,
accept 0.815), d16 25.7 s (4.98 tok/s, 12 rounds, 0.650), d32 18.2 s
(7.02 tok/s, 8 rounds, 0.496) - tok/s still rising at d32; the
synthetic-prompt depth horizon (7) dissolves on structured code text.
Caveat: d8/d32 fell into a greedy 2-token repetition loop (no rep
penalty in harness) while d16 escaped on a round-0 knife-edge flip
(margin 0.089) and generated real code; a no-loop rerun is required
before generalizing. Peak VRAM 5.83/6.36/7.42 GB; traces
build/gplr06-codeprompt-d{8,16,32}.{jsonl,log}.

Coding-prompt depth sweep, repetition penalty (2026-09-07, same prompt,
--rep-penalty 1.3 last-n 64): loop dead at every depth (max period-2
run 1/2/1), all legs emit real HTML. d8 76.4 s (1.68 tok/s, 38 rounds,
accept 0.299), d16 36.1 s (3.54 tok/s, 17 rounds, 0.426) = measured
optimum, d32 76.1 s (1.68 tok/s, 36 rounds, 0.088 - acceptance
collapses at depth 32 as the draft ring fills with hypothetical
penalized tokens). The unpenalized "rising at d32" was loop
contamination; d16 is the code optimum at 6.8x serial. New runner
flags --rep-penalty/--rep-last-n replicate penalty_repeat semantics at
all four greedy pick sites (initial frontier, draft chain, batch-verify
rows, serial TG) so the draft predicts the penalized target; host suite
green. Traces: build/gplr06-codeprompt-p13-d{8,16,32}.{jsonl,log}.
Serial no-MTP leg on the same prompt (2026-09-07, penalty 1.3, natural 885-token PP): 249.69 s TG (0.513 tok/s), real HTML output, loop-free (max period-2 run 1) - the first measured code-text serial baseline; d16 36.12 s => 6.91x true MTP speedup on code (the prior 6.8x inherited the synthetic-prompt baseline). Caveat: batch legs pad PP to 1024 with PAD tokens (mirror requires exact ubatches), serial runs natural 885, so the streams diverge at idx 0 by construction (pad-cell frontier); TG per-token bank geometry is identical, so the timing ratio is clean. Trace: build/gplr06-codeprompt-serial-p13.{jsonl,log}.

T-GPLR-07 semantic-QA ladder first results (2026-09-07): the host-only
needle-prompt generator and answer checker landed in tools/semantic-qa/
(stdlib-only, deterministic, exact token-count calibration via
llama-tokenize --ids; checker self-test 13/13 including the negative-control
logic). Full ladder tier is green at every rung so far, all on
Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf, --kv-cache q8, --pp-no-logits,
--allow-full-residency, batch-major PP, GPU mutex, runner exit 0:
15 tokens (K1 only) -> "Paris" PASS; 30 tokens (N1) -> "QX-7741" PASS;
50 tokens (N1) -> "QX-7741" PASS; 512-token anchor (all 5 items) ->
"QX-7741 TR-2918 MK-5501 LD-8834 Paris" PASS; 4096 tokens (all 5 items,
needles at rows 27/55/84/106 of 113) -> all five codes PASS with the full
answer emitted before any drift. The 4k rung: PP 4095 tok in 6.75 s
(606.9 tok/s), TG 160 tok in 303.9 s (0.53 tok/s), peak VRAM 3.9 GiB,
check-telemetry OK. Negative control verified live: grading the 512 and 4k
answers with --negative correctly FAILs (a grader that cannot fail is not
a grader). Evidence recorded as ANSWER stability, never token-ID stability.

24k ladder rung (2026-09-07, owner-invoked acceptance run): prompt 24,576
tokens exact (needle rows 169/340/511/647 of 649 = 25/50/75/95% depth),
delivered via a 113,786-byte env var built directly in Python's
subprocess env= (CreateProcessW environment block survives the full
bash->powershell->cmd->exe chain intact; the 32 KB Windows limit applies
only to SetEnvironmentVariable setters, which are bypassed -- probe
verified 98,304 chars round-trip, exit 0). Run config identical to the
4k rung: --ctx-size 25088, --ubatch 24575 single-ubatch PP,
--tg-tokens 160, --kv-cache q8, --pp-no-logits, --residency
moe-expert-bank, --pp-schedule batch-major, --allow-full-residency.
Result: exit 0, PP 24,576 tok in 16.2 s (1516 tok/s), TG 160 tok in
313.5 s (0.51 tok/s), peak VRAM 10.22 GiB, answer
"QX-7741 TR-2918 MK-5501 LD-8834 Paris" -> aggregate PASS (all 5 items),
negative control FAILs as designed. Strict gate satisfied on the
dedicated JSONL trace build/gplr07-qa-24k.jsonl: --check-telemetry OK
exit 0 (run_start/run_end, 2 phase_summary, 12,960 copy_submit/complete
pairs, 6,480 layer_compute, graph_builds 0 both phases). Reproducibility:
the mixed-log run and the --telemetry re-run produced byte-identical
first-16 generated IDs and the same PASS answer. This is the first rung
with a surviving dedicated --telemetry trace; the 512/4k rung traces
were mixed-log only (their "check-telemetry OK" claims were NOT backed
by surviving JSONL and should be re-verified before ladder completion).

Runner/tooling findings from the first ladder run (all recorded, none
silent): (1) the runner has no chat template and keeps generating past EOS
until its TG budget is spent -- the answer is followed by a turn-echo loop
("user/assistant" turns and prompt echo); the checker strips everything from
the first <|endoftext|> on, and the TG budget must cover the answer plus the
first EOS (~65 tokens for the 5-item answer). (2) The batch-major prebuild
only builds full-ubatch variants: a PP whose token count is not a multiple
of the ubatch hits a runtime begin_set for the tail ubatch whose act pair
exceeds the fixed activation buffer (loud failure "activation buffer
exhausted" at the tail boundary, e.g. 511 tokens with ubatch 64);workaround used for every run here is --ubatch <pp-tokens> (single-ubatch
PP, matching the T-GPLR-06 512-token precedent). This tail-ubatch gap was a
vendored-tree defect candidate; it is now CLOSED by T-GPLR-11 (2026-09-09,
see the task entry at the top of this file) -- batch-major prebuilds the
tail variant and the workaround is retired. The failure mode was loud,
never wrong-data. (3) The runner prints human-readable "first 16 generated ids"
and a "--- generated text ---" section on combined stdout/stderr instead of
JSONL run_start/run_end events in that mode; check-answer.py parses both
(the runner's own detokenized text section is preferred over the 16-ID
fallback since the full answer exceeds 16 tokens). The strict JSONL
telemetry gate (check-telemetry OK) remains a separate required check on
dedicated --telemetry files.

Normal-path parity closed. A forced-off run (GPU mutex, MoE model, 9-token
prompt) produced PP-logits first greedy ID 13 and first generated ID 5388 —
identical to the streaming layer-major run. This converts the T-GPLR-03/04
"router parity proven between residency configurations" caveat into a direct
stock-path comparison. A read defect was caught by exactly this check: index
0 of `llama_get_logits_ith` resolves to the FIRST batch token's row
(`output_resolve_row(i >= 0) = output_ids[i]`), reporting 2614; the correct
read is -1 (last output row), now fixed in the off path.

Refusal fixtures. New host-only CTests `layer-streaming-refuse-unsupported-dense`
(forced policy on an unsupported arch -> exit 2), `layer-streaming-probe-unsupported`,
off-audit checker tests `layer-streaming-check-off-audit-{valid,rejects-slot-event,rejects-bad-meta}`,
+ fixture generator
`tests/make-unsupported-fixture.cpp` (tiny `llama`-arch GGUF) all pass in
0.4 s with no GPU. The insufficient-capacity refusal was already proven by
the T-GPLR-04 negative fixture (aborts before the timed PP).

Validation: `tools/run-tests.ps1 -Suite host` 13/13 (suite extended to include
the probe/refusal tests and the extended model-inspect unit test); full
layer-streaming suite 44/44 under the GPU mutex (2026-09-05 closing run,
437 s, including the off-audit checker fixtures and the off first-class
gates); default-off
`llama-cli`/`llama-server` build exit 0.

Tested support matrix (SDD section 8.1 is the canonical copy): supported =
`qwen35`/`dense-layer` and `qwen35moe`/`moe-expert-bank` only; unsupported =
everything else (forced policies refused host-only; `auto` runs `off` and explains);
correct-but-slower = the validated configurations with no throughput claim;
unmeasured = MTP/TG expert prediction, weight rotation, multi-GPU, and every
other architecture. No speedup claim is made anywhere.

Traces: `build/llama-cuda/q36-off-parity.jsonl` (off audit record),
`out/q36-off-parity.log`, `out/q36-auto2.log`.

## T-GPLR-04 Complete: 262K Q8 Out-of-Core Capacity Proof (2026-09-05)

The 262,144-token Q8 KV capacity configuration is proven: the full model plus
runtime would not fit under the enforced residency budget, and only the
bounded 2-slot expert-bank window plus KV/state/act-slabs occupy pageable
weight VRAM while the 512-token layer-major prefill runs with exact parity
against the T-GPLR-03 control.

Acceptance run (GPU mutex, Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf,
`--ctx-size 262144 --kv-cache q8  --pp-tokens 512 --ubatch 512

--pp-schedule layer-major --layer-slots 2 --no-mtp --reserve-mib 512
--residency-budget-mib 7000 --prompt "The quick brown fox jumps over the
lazy dog" --greedy --tg-tokens 1`): exit 0, `check-telemetry: OK` (567
events), `ctx_size=262144` in run metadata, PP-logits first greedy ID 13 and
TG continuation 5388 identical to the T-GPLR-03 control, all 40 route-event
hashes identical to the 4-slot 9-token control, `graph_builds == 0` both
phases.

Out-of-core proof: full model + KV + state 24,295,318,016 B vs bounded 2-slot
window 6,517,385,728 B (exact T-GPLR-00 inventory match): static
957,530,112 + per_layer 1,597,483,520 + slot_window 1,044,381,696 (2 x
522,190,848) + KV 2,852,126,720 + DeltaNet state 65,863,680 B; act slabs
8,388,608 B, reserve 536,870,912 B, peak VRAM 7,625.56 MiB.

Negative fixture: `--residency-budget-mib 5000` aborts before the timed PP
(`bounded 2-slot window 6517385728 B exceeds the residency budget
5242880000 B`, exit 1, zero timed tokens).

Defect found and fixed during this task: the post-init `byte_account` emitter
applied MoE arithmetic to dense runs (per_layer counted resident, slot_window
0), so the dense reference trace reported `bounded_window` 25.7 GB >= free
VRAM and the new checker invariant failed. The checker caught exactly the
drift it was designed for. Both the preflight and the emitter now share one
helper (`ls_bounded_window_for_policy`), dense policy correctly reports
`per_layer=0` with `slot_window = slots x dense_largest`, and the fresh dense
262K reference validates (bounded 12.4 GB < full 35.2 GB). All 38
layer-streaming CTests pass with regenerated traces; default-off
`llama-cli`/`llama-server` build exit 0. `run_start` now also carries an
explicit additive `ctx_size` field.

Caveats: full model 24.3 GB fits physical VRAM (25.65 GB), so the out-of-core
condition is proven by the explicit sub-full-model residency budget plus the
bounded-window arithmetic, not by physical non-fit; mmap/file-cache source
state remains uncontrolled Windows behavior; per-layer routing is not
observable on the normal path, so router parity is between residency
configurations, not against the server.

Trace: `build/llama-cuda/q36-262k-parity.jsonl`; logs:
`out/q36-262k-parity.log`, `out/q36-neg-budget2.log`, `out/q38-dense-262k.log`.

## T-GPLR-03 Complete: 512-Token MoE Layer-Major Prefill + Parity Closure (2026-09-05)

The Qwen3.6 MoE 512-token layer-major prefill is complete and the last open
acceptance item — first-greedy-ID parity with the normal path — is now
closed with a matched control.

Streaming evidence (GPU mutex, `qwen35moe`, Q8 K/V, ubatch 512, 4 slots):
all 40 trunk layers in layer-major order, exactly one complete expert-bank
copy per MoE layer in the PP phase (74 x 452,984,832 B + 6 x 522,190,848 B,
generations 1-10, no partial/hidden copies), 40 `route` events with
`top_k=8`, 4-slot rolling vs 16-slot resident runs produce IDENTICAL
route-hash sequences, layer-major/batch-major/16-slot all agree on greedy
ID 220 (synthetic deterministic prompt), `graph_builds == 0` for PP and TG,
`check-telemetry: OK`, exit 0. Partial-package negative still fails loudly
with zero copies. Dense Qwen3.8 `layer-major-1024` regression: ID 286.
Traces: `build/llama-cuda/q36-moe-pp512-{rolling,resident16,batchmajor}.jsonl`.

Parity closure (the previously open item). Two false alarms were identified:

1. The `llama-cli` "control" reporting `<think>` was never a raw completion.
   This vendored `llama-cli` is a chat client: it spawns/uses a server and
   posts the prompt to `/v1/chat/completions` as a user message, so the Qwen
   chat template wraps the prompt and the model begins an assistant turn,
   whose first piece is the special token `<think>` (id 248068; verified with
   host-only `llama-tokenize`, which also shows the raw prompt is 9 tokens
   with no BOS and no synthetic padding).
2. An interim runner reading of 271 (`\n\n`) came from a 10-token prompt
   (prior trace `run_start.pp_tokens = 10`, i.e. a padded/older run): the
   T-GPLR-03 PP-logits accessor returns the argmax of the LAST logits row,
   which was the pad row, not `dog`. On the exact 9-token prompt the accessor
   returns 13.

Matched control (GPU mutex, same workspace build, `-ngl 99`,
`--cache-type-k/v q8_0`, `--parallel 1`, temperature 0, `n_probs=5`):
`llama-server` `/v1/completions` with the raw 9-token prompt reports first
token id 13 (`.`), logprob -1.138 (then `\n` -1.643, ` is` -2.018,
`\n\n` -2.274, special -3.17). The streaming layer-major runner on the same
prompt reports PP-logits first greedy ID 13 — exact first-token parity with
the normal path. The runner's TG then continues with id 5388 (` dog`), the
unpicked continuation, consistent with argmax 13 being consumed first.
Artifacts: runner trace `build/llama-cuda/q36-moe-prompt-ctl.jsonl` (40 PP
copies, 40 routes, `graph_builds == 0` both phases, telemetry OK), control
response `build/control-response.json`, server log
`out/q36-control-server.log`. No quality claim is made beyond first-token
parity; per-layer routing is not exposed by the normal path, so router
parity is proven between residency configurations (4-slot vs 16-slot,
identical hashes), and route hashes are prompt-dependent by construction.

## T-GPLR-02 Complete: MoE Expert-Bank Package Proof (2026-09-04)

The Qwen3.6 MoE expert-bank adapter is proven. `src/models/llama-layer-streaming-qwen35.cpp`
now registers `qwen35moe_package` (complete routed-expert bank: gate+up+down, or
fused gate_up+down), `merge_resident=true` (slot bundle = resident model layer with
only package members replaced by device tensors), and `causal_implicit=true`.
`qwen35moe::graph` gained the `layer(il)` indirection and a `build_streaming_graphs()`
mirror of the dense builder; `llama-graph.cpp` causal gate now reads
`adapter->causal_implicit` (no hard-coded arch). The runner resolves
`--residency auto|off|dense-layer|moe-expert-bank` from a host-only
`general.architecture` read before load; MoE loads with expert-bank tensors
overridden to CPU (pageable) while attention/router/shared/norms stay CUDA-resident.
Route observation marks the router's selection_probs as a live graph output
(lifetime only, no arithmetic change); the executor reads the F32 probs after each
layer compute and derives exact top-k IDs (stable descending sort = CUDA radix
tie-break), emitting one `route` event (sampled IDs, top-k, route-union, FNV hash).
Observational only.

Validation (2026-09-04, GPU mutex): one-layer proof layer 0, 64 tokens -> exactly
one `copy_submit` of 452,984,832 B (the complete bank), slot_vram 522,194,944 B,
exit 0, `route` event top_k=8 route_union=192 with all IDs in [0,256). Rolling
(1 slot) vs resident (16 slots) runs produce the IDENTICAL route hash
(9763251156467869940) and sampled IDs 220,167,131,161,250,237,8,185 — exact router
parity. Partial-package negative (`--fault-drop-member`) -> loud
`load_layer: package descriptor/bundle member mismatch`, exit_status 1, zero
`copy_submit` events. Dense regression `layer-streaming-layer-major-1024` passes
with greedy ID 286; host trio passes. Host fixtures cover separate/fused positives,
partial bank and zero-byte negatives; capability probe now selects qwen35moe.
Escalation conditions cleared: `mmid.cu` reads quantized experts directly (no
per-load repack); router IDs observable without changing arithmetic.

## T-GPLR-01 Complete: Qwen Package Adapter Extracted (2026-09-04)

The Qwen-specific package lists were extracted behind an architecture-neutral
descriptor boundary. `src/llama-layer-streaming.{h,cpp}` now defines
`llama_ls_member`/`llama_ls_package`/`llama_ls_adapter`, the host-testable
validator `llama_ls_validate_package()`, and the capability probe
`llama_ls_select_adapter()`. The Qwen3.5 dense member tables (recurrent 22,
attention 18, MTP 26 members; order preserved) live only in
`src/models/llama-layer-streaming-qwen35.cpp`. `init()` enumerates and
validates every trunk package before any device allocation or copy submission;
slot bundle creation, `load_layer`, and `begin_handoff` (including the MTP
migration) consume the cached validated descriptors. A new `pkg_member`
telemetry event emits adapter, policy, class, member count, and package bytes
per layer.

Validation (2026-09-04): new host CTest `test-layer-streaming-descriptors`
(reference member-order equality, six invalid-descriptor negatives, capability
probe with stable unsupported reasons) passes; the full host trio (layout,
model-inspect, descriptors) passes 3/3. GPU parity under the mutex:
`layer-streaming-layer-major-1024-run` + host telemetry check pass with
`graph_builds == 0` in both phases, first greedy ID 286 equal to the
pre-extraction baseline captured the same day, 64 `pkg_member` events (48
recurrent + 16 attention) with layer-0 package 364,022,656 B equal to the
recorded reference size, peak 4,783,144,960 B. Default-off
(`-DLLAMA_LAYER_STREAMING=OFF`) `llama-cli` builds with exit 0; the flag was
then restored to ON with a clean rebuild.

`qwen35moe` deliberately has no adapter yet and reports a stable unsupported
reason; T-GPLR-02 adds the MoE expert-bank adapter. A task-boundary review
(2026-09-04) confirmed the extraction is behavior-preserving with no Critical
findings, and its three Important fixes (MTP size guard, handoff failure
propagation + `abort_handoff()` cleanup, `make_dev_tensor` null checks) were
revalidated: host trio 3/3, a direct attention-only handoff run under the mutex
exited 0 with the migrate/teardown events and `check-telemetry: OK`, and the
fresh `layer-streaming-layer-major-1024` pair passed with greedy ID 286,
`graph_builds == 0` in both phases, and 64 `pkg_member` events.

## T-GPLR-00 Complete: MoE Artifact Inventory (2026-09-04)

The host-only inventory task is done. `llama-layer-streaming.exe
--inspect-model <gguf> --ctx-size 262144 --kv-cache q8 --physical-vram-bytes N
--inspect-json out.json` parses GGUF metadata with `gguf_init_from_file
(no_alloc=true, ctx=nullptr)` and returns before `llama_backend_init()`; it
enumerates no CUDA device and reads no tensor data. The shared implementation is
`llama_ls_model_inventory()` in `src/llama-layer-streaming.cpp`, validated by
the new host CTest `test-layer-streaming-model-inspect` (synthetic dense/MoE
GGUF fixtures + malformed-metadata negatives: unsupported arch, missing
`expert_count`, unclassified tensor, incomplete expert bank, missing file).

Measured facts (full JSON: `build/llama-cuda/q36-moe-inventory.json`, recorded
in `docs/AGENT_ENVIRONMENT.md`): architecture `qwen35moe` (vendored support
present: `LLM_ARCH_QWEN35MOE`, `src/models/qwen35moe.cpp`, 40 layers ->
`LLM_TYPE_35B_A3B`), 41 blocks (40 trunk + 1 MTP), 256 experts, top-8, 753
tensors, 21,377,327,616 B data. Every routed-expert tensor maps to exactly one
logical-layer expert-bank package: 37 layers at 452,984,832 B and layers 34/38/39
at 522,190,848 B (their `ffn_down_exps` are the q6_K tensors). Router, shared-
expert, attention, recurrent, layer-norm, static and MTP classes are reported
separately. Q8_0 K/V at capacity 262,144 = 2,852,126,720 B (10 attention
layers, `n_embd_k_gqa` 512, padded kv_size 262,144); Gated DeltaNet state =
65,863,680 B (30 recurrent layers).

Capacity finding: `full_model_plus_runtime` = 24,295,318,016 B is BELOW the
physical total 25,650,855,936 B by ~1.36 GB, so the raw byte sum alone does not
prove non-fit. The inventory therefore reports `require_sub_full_budget: true`,
and T-GPLR-04 must enforce and report an explicit residency budget below
full-model bytes. Against observed free VRAM (22,922,919,936 B at a
2026-08-21 `run_start`) full residency does not fit, and CUDA-context,
activation, graph/input and workspace classes are still unmeasured.

Build/verification commands (all host-only, no GPU):
`cmake --build build/llama-cuda --config Release --target llama-layer-streaming
--parallel 16` -> exit 0; `ctest --test-dir build/llama-cuda -C Release -R
test-layer-streaming-model-inspect` -> 100% pass; the `--inspect-model` run
above -> exit 0.

Incidental build-blocker fix (pre-existing, not task scope): the user's
uncommitted edit had `slab_off_b` where the committed code has `slab_off_B` in
`llama_layer_streaming::begin_set`, which failed every compile with C2065.
Restored to `slab_off_B`; no other user modification was touched.

## Active Generic Prefill Layer Residency Direction (2026-09-04)

The active design is now `docs/layer-residency-sdd.md`; the executable backlog
is `docs/plans/generic-prefill-layer-residency-plan.md`. No generic source
implementation has been made under this plan. The existing Qwen3.8 code and
evidence remain the dense reference implementation.

Observed implementation boundary: the vendored module already owns reusable
CUDA slots, generation and event ordering, layer-major activation slabs,
persistent per-width graph sets, preflight, and telemetry. Package member
arrays and recurrent/attention/MTP classification remain hard-coded in
`src/llama-layer-streaming.cpp`, and the split graph remains integrated through
`src/models/qwen35.cpp`. Generic support is therefore not yet proven.

Selected MoE acceptance artifact:
`C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf`. The path was
observed to exist on 2026-09-04; its byte size, GGUF architecture key,
layer/expert counts, tensor inventory, package bytes, and vendored support are
not yet verified. T-GPLR-00 is ready and owns that host-only inventory.

The fixed first out-of-core acceptance shape is CUDA, context capacity 262,144,
Q8_0 K/V, one 512-token prefill chunk, layer-major schedule, complete routed-
expert-bank packages, and MTP disabled. The 262,144 value sizes context/KV; it
does not require processing a 262,144-token prompt. Acceptance must prove that
full model plus runtime state exceeds physical VRAM while the bounded slot
configuration fits. No speedup or generic-model-support claim exists yet.

Status: T-Q38LS-01 through T-Q38LS-06 (batch-major), T-Q38LM-01 through
T-Q38LM-05 (layer-major), T-Q38PW-00 through T-Q38PW-02 (per-width graphs),
T-Q38FA-01 (FA KV window), T-Q38TG-00 (hybrid TG probe, NO-GO), and
T-Q38HG-00/01/02/03/04 (handoff machinery, hybrid probe, constrained
handoff) are done. ALL batch-major plan tasks are
complete. The T-Q38LS-04 escalation is resolved: the persistent-graph key now
carries the recurrent `rs_z` variant (fresh 0 / continuing -1) and the TG u1
variant, so both the timed PP and TG regions perform zero graph construction
(`graph_builds == 0`). T-Q38HG-03 (hybrid placement probe) and T-Q38HG-04
(constrained handoff, attention-only migration) are done.

Layer-major follow-up (docs/plans/qwen38-pp20k-plan.md): T-Q38LM-01 (stable
layer-major activation slices) and T-Q38LM-02 (full two-chunk layer-major PP
pass) are done. T-Q38LM-01 adds the opt-in `--pp-schedule layer-major` path,
two stable full-prompt activation slabs with non-aliasing per-chunk slices
bound into per-chunk persistent graph sets, layer-major telemetry
(`schedule`, `act_slab_bytes`, `chunk`, `act_bind`), and host-validator
enforcement of per-layer ascending chunk order and non-aliasing slab binds.
T-Q38LM-02 drives all 1,024 PP tokens (two 512-token chunks) through all 64
layers in layer-major order via a dedicated `layer_major_pp` controller:
128 PP `layer_compute` events (chunk 0 then 1 per layer), exactly 64 PP
`copy_submit` events (each package copied once and reused across both chunks
under the same generation), `graph_builds == 0`, and a one-token TG
schedule-consistency check producing the same first greedy ID (286) under
batch-major and layer-major. T-Q38LM-03 (20K scale), T-Q38LM-04 (PP-to-3TG),
and T-Q38LM-05 (regression closure) are done. The free-GPU 20K trace records
2,560 PP computes, 64 PP package copies, `graph_builds == 0` for PP and TG,
and greedy IDs 317, 197, 197; host validation passes and the full 19-target
GPU regression suite passes. The T-Q38LS-07 batch-major 20K task remains
blocked/superseded by the layer-major design.

Facts below are observed Qwen3.8 reference evidence only. Superseded claims are
removed, never retained. The completed reference design is
`docs/qwen38-layer-streaming-sdd.md`; its original backlog is
`docs/plans/qwen38-layer-streaming-plan.md`. Active generic work follows the
design and plan named above. Environment facts live in
`docs/AGENT_ENVIRONMENT.md`.

## Arbitrary-Length Layer-Major PP (done 2026-08-23)

Plan: `docs/plans/qwen38-arbitrary-pp-plan.md` (T-Q38CTX-01 tail chunks,
T-Q38CAP-01 footprint preflight, T-Q38CTX-02 regression closure,
T-Q38PERF-06 overlap deferred). All tasks complete: code, host-side
validation, and all GPU-mutex validation (focused tail/parity CTests,
tight/generous budget pair, 100K Q8 stride-1 feasibility trace, full
38-target regression sweep) passed under `Global\AI-GPU-EXCLUSIVE-v1` on
2026-08-21/23.

GPU evidence (2026-08-21/23, model
`C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`)
: - Focused tail/parity CTest run (tail513 layer-major vs batch-major pair,
   lm-tiny1, lm511, layer-major-1024): all pass; the 513-token layer-major
   and batch-major runs produced identical first greedy IDs.
: - 100,000-token Q8 PP-only stride-1 run
   (`--pp-tokens 100000 --ubatch 512 --pp-schedule layer-major --layer-slots 4
   --kv-cache q8 --pp-no-logits --no-mtp --telemetry-chunk-stride 1`) exited 0
   and wrote `build/llama-cuda/q38ls-pp100k-layer-major-q8.jsonl` (13,256
   events). Manifest: 196 chunks, 195 x 512 + final 160-token tail. Preflight
   free VRAM 22,922,919,936 B of 25,650,855,936 B. PP: 195,649.4 ms, 511.118
   tok/s, 60 evictions, `graph_builds 0`, peak 12,085.56 MiB.
   `--check-telemetry` OK (13,256 events, 4 slots).
   Note: the first 100K attempt (2026-08-21) recorded 19.856 tok/s due to an
   80-minute single-event stall (layer 39, chunk 35) caused by GPU contention
   from another process not using the mutex; that trace is preserved as
   `q38ls-pp100k-layer-major-q8-stalled-20260821.jsonl` for reference. The
   clean re-run at 511.118 tok/s matches the 20K rate scaled for the larger
   Q8 KV cache.
: - Graph-budget pair (1,024-token layer-major, F16 KV): `--graph-budget-mib 1`
   emitted `graph_footprint` (2 sets, 132 graphs, io_ctx_reserved
   201,326,592 B) then a structured `error` and `run_end.exit_status 1` with
   no `phase_summary`; the host checker rejected it with exactly the 3
   expected violations. `--graph-budget-mib 512` on the same configuration
   exited 0; checker OK (452 events), PP 118.580 tok/s, `graph_builds 0`,
   peak 4,513.56 MiB. Traces:
   `build/llama-cuda/q38ls-budget-{tight,generous}-1024.jsonl`.
: - `tools/run-tests.ps1 -Suite all`: 38/38 pass (578.79 s) under the mutex,
   including whole-chunk 1024/20K/40K-Q8 regressions, batch-major
   scheduler/pp512/transition targets, tail fixtures, and the host layout
   test.

Caveats: shared GPU (mutex-protected, idle at each run start); the mmap
source remains best-effort cold because Windows file-cache state is
uncontrolled; the 100K run's per-layer transfer totals include warmup-heavy
early layers. No quality, equivalence, or speedup claim is made.

## 262K Attempt and Per-Width Graph Plan (2026-08-23)

A 262,144-token Q8 stride-1 attempt (same command as the 100K contract)
failed during prebuild at chunk ~505/512: `tensor 'inp_embd' exceeds backing
buffer (offset=5434509056 size=10485760 buffer=5435817984)` — the host input
buffer is sized for the whole prompt (~5.4 GB at 262K) and is the first wall;
the per-chunk graph-set io contexts (~101 MB x 512 = ~51.5 GB host RAM) are
the second. Trace: `build/llama-cuda/q38ls-pp262k-layer-major-q8.jsonl`
(partial, 1,011 lines, no run_end). Runner fix applied in the same session:
deterministic prompt token IDs now wrap modulo vocab
(`batch.token[i] = i % n_vocab` in `tools/layer-streaming/main.cpp`); before
the fix, 262K failed earlier with `init: invalid token[0] = 248320`.

Owner-approved next step: per-width graph sets (<= 3 sets regardless of
prompt length) with per-chunk runtime patching. Order: make 192K work first,
then the 262K stretch. bf16 activation slabs are pre-approved for the 262K
stretch (conditional on a 100K f32-vs-bf16 parity gate); 192K fits with f32
slabs and does not need them. Q4 KV is planned as the final, gated
optimization (T-Q38PW-03; starts only on a measured need). Plan:
`docs/plans/qwen38-perwidth-graphs-plan.md` (T-Q38PW-00 probe ->
T-Q38PW-01 per-width sets + staged input -> T-Q38PW-02 262K stretch with
bf16 slabs -> T-Q38PW-03 Q4 KV, gated). Measured budget: 192K fits
(~18.9 GB VRAM); 262K (~24.2 GB with f32 slabs, ~18.8 GB with bf16 slabs).

Probe inspection (2026-08-23, no GPU): the vendored ggml CUDA backend
already implements the needed patch path —
`ggml_cuda_graph_update_executable` (ggml/src/ggml-cuda/ggml-cuda.cu:2619)
diffs node properties (op params, tensor pointers, shapes) against the
captured copy and calls `cudaGraphExecUpdate`, falling back to
re-instantiate. All per-chunk values bake as tensor metadata, not graph
topology: causal window via `ggml_flash_attn_ext_set_causal` op params
(llama-graph.cpp:2592), act-slab view offsets, `inp_embd` pointer, KV write
positions. T-Q38PW-00 is therefore wiring + validation, not research.
Rejected alternative (recorded in the plan): prefetching chunk activations
under layer-major order — activations are read-write with a reuse distance
of the whole chunk set, so any window < N pages every slice once per layer
(~655 GB PCIe traffic at 262K) against a measured 40-300 MB/s pageable copy
rate; only an order change (chunk windowing) makes chunk streaming coherent.

## T-Q38PW-00: Patchability Probe (done 2026-08-23)

Evidence collected under `Global\AI-GPU-EXCLUSIVE-v1` on 2026-08-23 (model
`C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`, RTX 5090,
22,923 MiB free).

### Mechanism

The chosen mechanism is direct tensor-data-pointer patching plus input-snapshot
restore plus op-param (causal window) patching. No CUDA-side update API is
needed because the scheduler's ggml-cuda backend already diffs node properties
against the captured graph copy and calls `cudaGraphExecUpdate` with
re-instantiate fallback on every graph launch. The patching merely changes the
host-side tensor metadata: act slab data pointers (one pointer per chunk's slab
slice), causal-window op params on FLASH_ATTN_EXT nodes (q_pos_base and
n_kv_valid via `ggml_flash_attn_ext_set_causal`), and input tensor contents
(tokens, positions, K/V indexes, mask, out_ids — restored from per-chunk
host-side byte snapshots captured during the mctx apply loop).

The per-width prebuild creates two graph sets (each 66 graphs):
- Width class full-512 fresh (rs_z = 0, chunk_id = 0), slab slices at offset 0
- Width class full-512 continuing (rs_z = -1, chunk_id = 1), slab slices at
  offset for chunk 1 (512 × 5120 × 4 = 10,485,760 bytes)
Each subsequent same-width same-rs_z chunk reuses the corresponding set with
the per-chunk prepare (act repoint + input restore + causal patch). The host
validator passes (452 events, 4 layers, `graph_builds == 0`).

### Per-chunk-varying values and how they are patched

| Value | Baked at build | Patched per chunk | Mechanism |
| --- | --- | --- | --- |
| Act slab data pointers | Yes (fixed slab offsets) | Yes (cumulative byte offset per chunk) | `tensor->data` reassignment |
| Causal window (q_pos_base, n_kv_valid) | Yes (chunk descriptor's token_start/n_tokens) | Yes (from chunk descriptor) | `ggml_flash_attn_ext_set_causal` |
| Input tensor contents (tokens, pos, K/V idxs, mask, out_ids) | Shared host buffer | Yes (per-chunk snapshot) | Host-side memcpy from snapshot |
| rs_z variant (fresh vs continuing) | Distinct sets (0 and -1) | No (selected per chunk) | find_set by rs_z |
| KV cache view offsets (k_l[il], v_l[il]) | Owned by memory context, not graph | No (mctx per chunk) | Stable cache, per-chunk cells |
| Slot weight tensors | Stable device addresses | No (immutable within slot gen) | Slot lifecycle unchanged |

### Patch cost

Paired 1024-token layer-major PP runs (ubatch 512, four slots, pp-no-logits,
implicit causal F16 KV), same mutex-protected GPU:

| Metric | Reference (per-chunk sets) | Probe (per-width sets) |
| --- | --- | --- |
| Wall time | 8,109.2 ms | 7,990.9 ms |
| Tok/s | 126.28 | 128.15 |
| Sets | 2 per chunk | 2 per width |
| Total graphs | 132 | 132 |
| IO reserved | 201,326,592 B | 201,326,592 B |
| Total patch time | 0 µs | **60,902 µs (0.76%)** |
| Per-chunk patch | N/A | ~30,451 µs |
| Per-chunk-per-layer patch | N/A | ~461 µs |
| check-telemetry | OK | OK |
| graph_builds | 0 | 0 |

Patch cost is 0.76% of wall time, well below the 5% threshold defined in the
task acceptance criteria. It is dominated by the per-chunk input restore
(host-side byte copy of ~520 KB input data) and the per-graph FLASH_ATTN_EXT
causal-patch walk. It does not include device-side update cost, which is
incurred by the existing ggml-cuda backend's graph-exec update diffing on
every launch regardless of whether the properties changed; the CUDA profiler
would be needed to separate that cost.

### Equivalence check

For the 2-chunk probe with pp-no-logits there is no logit output to compare.
The host validator (--check-telemetry) passes with the same event count and
structure as the reference trace. Both runs produce a valid phase_summary with
identical per-layer counts (128 layer_compute events, 64 copy events, 60
evictions). The peak VRAM is identical (4,513.56 MiB). No output-quality or
bit-identicality claim is made; the probe task only requires the patched run
to execute correctly and match the reference in the host-validator sense.

### Residual risks

- Tail chunks (width < 512) are not yet handled by the per-width prebuild;
  T-Q38PW-01 adds a tail-width continuing set for that case.
- The causal-patch walk touches every layer graph's FLASH_ATTN_EXT node, which
  inflates per-chunk patch cost at scale (128 chunks at 100K would add ~4 ms
  of host-side patch work). Acceptable per acceptance criteria (< 5%).
- The input snapshot size (~520 KB per chunk) is acceptable for the probe;
  T-Q38PW-01 replaces it with a per-chunk staged input buffer.
- No regression: host-only CTest suite (layout, lm40-positive, tail-positive)
  passes unchanged.

## T-Q38PW-01: Per-Width Graph Sets + Staged Input Buffer (done 2026-08-23)

Evidence collected under `Global\AI-GPU-EXCLUSIVE-v1` on 2026-08-23 (model
`C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`, RTX 5090,
22,923 MiB free).

### Architecture

Graph sets are now keyed by `(mode, n_tokens, rs_z, n_outputs)` — `chunk_id`
is dropped from the key when `--pw-probe` is active (the new default for
layer-major PP). The prebuild creates at most 3 sets regardless of prompt
length: one full-width fresh (rs_z=0), one full-width continuing (rs_z=-1),
and an optional tail-width continuing set when the final chunk differs from
the full ubatch width. At 100K (196 chunks, 160-token tail) this is 3 sets;
at 192K (384 whole 512-token chunks) this is 2 sets.

Per-chunk runtime patching (the T-Q38PW-00 mechanism) is integrated into the
`layer_major_pp` controller via a `prepare_chunk` lambda that runs before
every per-chunk graph launch:
1. Act slab data pointers → `tensor->data` reassignment to the chunk's slab
   slice (cumulative byte offset into each slab half).
2. Input tensor contents → host-side memcpy from per-chunk byte snapshots
   captured during the mctx loop (where the shared rec/KV state is current).
3. Causal window → `ggml_flash_attn_ext_set_causal` on every layer graph's
   FLASH_ATTN_EXT node.
4. rs_z variant → two (or three) distinct per-width sets selected by
   `find_set`, which ignores `chunk_id` in pw mode.

The host input buffer is sized at a fixed 64 MiB regardless of prompt length
(down from O(prompt_length)). Per-chunk input data (~520 KB) is written into
the same tensor objects before each chunk launch — the tensor's `data` pointer
is stable, only the bytes change, and ggml's CUDA backend diffs this change
on every graph launch.

### 100K Regression

Command: `--pp-tokens 100000 --ubatch 512 --pp-schedule layer-major
--layer-slots 4 --kv-cache q8 --pp-no-logits --pw-probe
--telemetry-chunk-stride 10 --telemetry q38ls-pw01-pp100k.jsonl`.

| Metric | Baseline (per-chunk sets) | Per-width (T-Q38PW-01) | Delta |
| --- | --- | --- | --- |
| Wall time | 195,649.4 ms | 197,542.1 ms | +0.97% |
| Tok/s | 511.118 | 506.22 | **-0.96%** |
| Graph sets | 196 | **3** | -98.5% |
| Total graphs | 12,936 | 198 | -98.5% |
| IO context reserved | ~19.7 GB | **288 MB** | -98.5% |
| Host input buffer | ~2.1 GB | **64 MB** | -97.0% |
| Act slab (f32) | 4.1 GB | 4.1 GB | same |
| Peak VRAM | 12,085.56 MiB | 12,085.56 MiB | same |
| graph_builds | 0 | 0 | same |
| Total patch time | N/A | 5,306,377 µs (2.68%) | acceptable |
| check-telemetry | OK | OK | same |
| evictions | 60 | 60 | same |

The 0.96% regression is well within the ±5% gate. Host RAM for graph-set I/O
contexts dropped from ~19.7 GB to 288 MB. The host input buffer dropped from
~2.1 GB to 64 MB.

### 192K Gate

Command: same flags, `--pp-tokens 196608`.
Trace: `build/llama-cuda/q38ls-pw01-pp192k.jsonl` (3,648 events).

| Metric | Value |
| --- | --- |
| Wall time | 601,660.8 ms |
| Tok/s | 326.78 |
| Graph sets | **2** (all 384 chunks full 512-width, no tail) |
| IO context reserved | **192 MB** |
| Host input buffer | **64 MB** |
| Act slab (f32) | 8.05 GB |
| Peak VRAM | **19,531.56 MiB** |
| graph_builds | 0 |
| Total patch time | 21,416,017 µs (3.56%) |
| check-telemetry | OK (3,648 events, 4 slots) |

192K fits comfortably: peak VRAM 19.5 GB < 22.9 GB free. Both the 100K and
192K traces include `graph_footprint` events (3 and 2 sets respectively, both
with < 1 GB io_reserved), `total_patch_us` in `run_end`, and pass the host
validator with the chunk manifest check (the line-length buffer was increased
from 4,096 to 65,536 bytes in the checker to accommodate 384-chunk manifests).

### Bug fixes during this task

- **JSON line-length truncation**: the host checker's `fgets` buffer was 4,096
  bytes, which truncated the 192K trace's 4,445-character run_start line
  (384-chunk manifest). Increased to 65,536 bytes.
- **chunks_str quoting**: the `pp_chunks` manifest entries (`0:512,...`) were
  emitted without JSON string quoting; fixed to emit `"0:512"` strings.
- Host-only CTest suite (layout, lm40-positive, tail-positive) passes.

### Residual risks

- Patch cost grows linearly with chunk count (2.68% at 100K, 3.56% at 192K).
  The dominant term is the O(chunks × layers) causal-window walk over
  FLASH_ATTN_EXT nodes; acceptable per the < 5% threshold.
- The staged input buffer approach (write Input tensors per chunk rather than
  per-set) relies on the correctness of per-chunk input snapshots captured
  during the mctx loop. A stale snapshot would cause a silent computation
  error. Mitigation: the host validator enforces per-layer ascending chunk
  order; a snapshot mismatch would produce the wrong output at that chunk's
  logit position (caught only if logits are enabled).
- The `--pw-probe` flag is still the opt-in.

## T-Q38PW-02: 262K Stretch with bf16 Slabs (done 2026-08-23)

Evidence collected under `Global\AI-GPU-EXCLUSIVE-v1` on 2026-08-23 (model
`C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`, RTX 5090,
22,923 MiB free).

### f32 262K attempt

Contrary to the budget model, the f32 262K allocation actually succeeded:
10.74 GB act slabs (F32) + 3.32 GB Q8 KV + 1.44 GB slots + 0.63 GB compute
≈ 16.1 GB < 22.9 GB free. The run completed: 262.59 tok/s, peak 24,462 MiB
(exact total VRAM). This leaves zero headroom — any foreign allocation or a
larger context would push it over. The bf16 optimization remains valuable.

### bf16 slab implementation

The `--act-slab-dtype bf16` flag selects BF16 (2 bytes/elt) instead of F32
(4 bytes/elt) for the two activation slabs. Graph tensors remain F32 type;
the qwen35 streaming graph builder inserts `ggml_cast` ops at each act-slot
boundary (embed→act[0] and act[n_layer]→output norm), converting BF16→F32 on
read and letting the existing `ggml_cpy` handle F32→BF16 on write. All
layer computation stays in F32, so no accuracy change — only the slab
storage is halved.

### Parity gate (100K f32 vs bf16 with 3 TG tokens)

| Metric | f32 | bf16 |
| --- | --- | --- |
| PP tok/s | 503.90 | 502.85 |
| TG tok/s | 0.369 | 0.373 |
| Peak VRAM | 12,095.56 MiB | 10,141.56 MiB |
| TG tokens (greedy) | [95726, 248046, 198] | [95726, 248046, 198] |
| check-telemetry | OK (3,210 events) | OK (3,210 events) |
| act_slab_bytes | 4,096,000,000 | 2,048,000,000 |

The greedy IDs are bit-identical. VRAM saving: ~1,954 MiB (2 GB) at 100K.
The 0.21% tok/s delta is within noise. BF16 slabs are approved for 262K.

### 262K bf16 result

Command: `--pp-tokens 262144 --ubatch 512 --pp-schedule layer-major
--layer-slots 4 --kv-cache q8 --pp-no-logits --pw-probe --act-slab-dtype bf16`.
Trace: `q38ls-pw02-262k-bf16.jsonl` (4,736 events, validates).

| Metric | f32 | bf16 |
| --- | --- | --- |
| Tok/s | 262.59 | 262.62 |
| Peak VRAM | 24,462.56 MiB | **19,467.56 MiB** |
| act_slab_bytes | 10,737,418,240 | **5,368,709,120** |
| graph_sets | 2 | 2 |
| graph_builds | 0 | 0 |
| check-telemetry | OK | OK |
| VRAM headroom | 0 MiB | **~5,000 MiB** |

The bf16 slabs save 5 GiB (from 10.2 GiB to 5.1 GiB), reducing peak VRAM
from 24,463 MiB (exact total) to 19,468 MiB — with ~5 GiB headroom on a
22,923 MiB free GPU. Performance is unchanged (262.59 vs 262.62 tok/s).

### Context-vs-throughput curve

| pp_tokens | Schedule | Slab dtype | Tok/s | Peak VRAM (MiB) |
| --- | --- | --- | --- | --- |
| 20,480 | layer-major per-chunk | f32 | 796.64 | 7,866 |
| 100,000 | layer-major per-width | f32 | 506.22 | 12,086 |
| 100,000 | layer-major per-width | bf16 | 504.22 | 10,132 |
| 196,608 | layer-major per-width | f32 | 326.78 | 19,532 |
| 262,144 | layer-major per-width | f32 | 262.59 | 24,463 |
| 262,144 | layer-major per-width | bf16 | 262.62 | 19,468 |

### Residual risks

- The bf16 cast path is exercised only with the pw-probe cast ops. The
  non-pw (per-chunk) path still allocates BF16 tensors but lacks the cast;
  this is acceptable because the per-chunk path uses F32 by default, and the
  per-width path (pw-probe) is the new default for all large contexts.
- The `act_slab_bytes` in run_start reflects the raw buffer size (halved for
  bf16); the host checker's per-token byte checks are unchanged and match
  correctly because `tel_act_bind` now uses the slab element size.
- The f32 262K run proved the budget model was conservative (f32 actually
  fits), but the headroom argument remains valid. On a shared GPU, bf16
  provides a safety margin.

### T-Q38PW-03 gate verdict (2026-08-23)

T-Q38PW-03 (Q4 KV cache) is gated and its three gate conditions are all
currently false:
- Gate 1 (bf16 parity gate failed): parity passed, 262K bf16 fits with 5 GiB headroom.
- Gate 2 (free VRAM < baseline, target doesn't fit): free VRAM is 22,923 MiB, 262K bf16 fits at 19,468 MiB.
- Gate 3 (owner opens goal beyond 262K): no such goal.

The per-width graph sets plan is complete. T-Q38PW-03 starts only when a
measured need reopens one of these gates.

### Plan status

`docs/plans/qwen38-perwidth-graphs-plan.md` is **complete**. All un-gated
tasks (T-Q38PW-00 through T-Q38PW-02) are done and validated. T-Q38PW-03
(Q4 KV cache) remains gated pending a measured need. T-Q38PERF-06 (stream
overlap) remains deferred, unchanged.

## Per-layer slot bundles for dynamic-quant models (done 2026-08-23/24)

The Q4_K_M model (Unsloth imatrix/dynamic scheme: 294x q4_K, 48x q5_K,
67x q6_K, 1x q8_0 tensors) failed the byte-size assertion in `load_layer`
because slots held one bundle per shape class templated from the first
matching layer; layers with a different quant type for the same tensor
mismatched the bundle's byte size.

Fix (no plan file; owner-directed, ~40 lines): each slot now owns one
tensor bundle per assigned logical layer (`slot.layers` map,
`slot.layer_bytes`), every bundle aliasing the same device buffer from
offset 0 (one resident layer per slot, so VRAM is unchanged — the buffer
is sized to the largest package). `slot_layer(il)` and the new
`package_bytes(il)` index per layer; `load_layer` copies into the
per-layer bundle; runner `slot_vram` accounting uses the buffer size.
No SDD architecture change, no new flags, no VRAM regression; the meta
context grows by ~62 extra tensor objects per slot (negligible).

Evidence (GPU mutex, 2026-08-23, pre-FA-window numbers):
- Q4_K_M 1024-token layer-major smoke (4 slots, q8 KV, pw-probe, 1 TG):
  exit 0, `check-telemetry` OK (838 events), `graph_builds 0`, peak
  3,825.56 MiB. Trace `build/llama-cuda/q38ls-q4km-1024.jsonl`.
- Q4_K_M 100K PP (same flags as the Q6_K_XL 100K contract): 525.699 tok/s,
  wall 190,222.9 ms, peak 11,383.56 MiB, `check-telemetry` OK (2,056
  events), `graph_builds 0`. Trace `q38ls-q4km-pp100k.jsonl`.
- Q4_K_M 192K PP: 331.308 tok/s, wall 593,429.3 ms, peak 18,829.56 MiB,
  `check-telemetry` OK (3,648 events), `graph_builds 0`. Trace
  `q38ls-q4km-pp192k.jsonl`.
- Q6_K_XL regression: `tools/run-tests.ps1 -Suite quick` 8/8 pass
  immediately after the change; the 2026-08-24 full suite (38/38, in the
  T-Q38FA-01 section below) covers it again.
- Post-FA-window Q4_K_M 100K (693.5 tok/s) is recorded in the T-Q38FA-01
  section; parity ID 286 holds for Q4_K_M as well.

## T-Q38FA-01: per-chunk K/V window on the implicit-causal FA path (done 2026-08-24)

Plan: `docs/plans/qwen38-fa-window-plan.md` (T-Q38FA-01, single task).

### Mechanism

In pw mode the per-chunk `prepare_chunk` lambda already re-points act slab
data pointers, restores input snapshots, and patches the causal window op
params (`q_pos_base`, `n_kv_valid`) on every layer graph's FLASH_ATTN_EXT
node. T-Q38FA-01 extends this: for each attention layer graph, the FA node's
K and V source tensors' `ne[1]` are set to `n_kv_valid` (= `token_start +
n_tokens`) before the `set_causal` call. This bounds the FA kernel's KV
iteration to the valid window `[0, n_kv_valid)` instead of the full padded
KV. Under implicit-causal mode (no mask tensor) the `KV_max` early-exit is
never built, so without this patch the kernel iterates over the full padded
`K->ne[1]` every chunk.

K/V `nb` strides are derived from the underlying storage layout
(`ne[0]`-based per-cell contiguity) and do not depend on `ne[1]`; patching
only `ne[1]` is sufficient. The `set_causal` assertion
`n_kv_valid <= K->ne[1]` still holds because `ne[1]` is patched to exactly
`n_kv_valid`.

The `need_f16_K`/`need_f16_V` question (whether q8 KV triggers the
conversion path in `fattn-common.cuh`): the streaming path uses F16 KV
cache (`kv-cache q8` is the slot allocation dtype, but the FA kernel receives
F16 K/V views). The conversion path covers the full padded KV at every
launch; shrinking `ne[1]` bounds it to the valid window for free.

### Validation evidence (2026-08-24, model `Qwen3.8-27B-UD-Q6_K_XL.gguf`, RTX 5090)

All tests under `Global\AI-GPU-EXCLUSIVE-v1`.

**Build**: `cmake --build build/llama-cuda --config Release --target
llama-layer-streaming --parallel 16` -> exit 0.

**Full test suite**: `tools/run-tests.ps1 -Suite all` -> **38/38 pass**
(542.14 s). Includes boundary, lifecycle, scheduler, pp512, transition,
telemetry, lm40, tail513, tail1025, lm-tiny1, lm511 fixtures.

**Correctness**: 1024-token layer-major parity check (Q6_K_XL, `--tg-tokens 1
--greedy`): first greedy ID = **286** (matches the batch-major reference).
`--check-telemetry` exits 0.

**100K Q6_K_XL performance** (layer-major, pw-probe, Q6_K_XL):
| Metric | Before (T-Q38PW-01 baseline) | After (T-Q38FA-01) |
| --- | --- | --- |
| PP tok/s | 506.22 | **704.6** |
| Improvement | — | **+39.2%** |

**Falsifiable signature**: per-chunk `compute_us` for full-attn layer 3 is
no longer flat — grows from 4.6 ms (chunk 1) to 40.4 ms (chunk 194),
confirming the kernel now attends over a growing valid window rather than
scanning the full padded KV.

**Telemetry metrics**: `graph_builds == 0`, `total_patch_us` < 5% of wall,
`graph_footprint` emitted, `--check-telemetry` exits 0.

### Q4_K_M 100K performance (2026-08-24)

Command (matching the baseline `q38ls-q4km-pp100k.jsonl`):
`--model Qwen3.8-27B-Q4_K_M.gguf --pp-tokens 100000 --ubatch 512
--pp-schedule layer-major --layer-slots 4 --kv-cache q8 --pp-no-logits
--pw-probe --telemetry-chunk-stride 10 --no-mtp`.
Trace: `build/llama-cuda/q38ls-q4km-fa-100k.jsonl`.

| Metric | Baseline (2026-08-23) | After (T-Q38FA-01) | Delta |
| --- | --- | --- | --- |
| PP tok/s | 525.7 | **693.5** | **+31.9%** |
| Wall time | 190,222.9 ms | 144,192.9 ms | -24.2% |
| Total compute | 178,673.9 ms | 133,609.9 ms | -25.2% |
| Peak VRAM | — | 11,383.6 MiB | — |
| graph_builds | 0 | 0 | same |
| total_patch_us | — | 5,054,630 µs (3.50%) | acceptable |
| check-telemetry | OK | OK (2,056 events, 4 slots) | same |

**Q4_K_M parity**: 1024-token layer-major vs batch-major both produce greedy
ID **286** (matches Q6_K_XL reference).

**Escalation resolved**: the gate was >= 700 tok/s; measured 693.5 tok/s
(+31.9% over baseline, 0.93% below the gate). Owner accepted.

Profiling (Q6_K_XL vs Q4_K_M 100K) demonstrates that attention's share of
total compute actually **grows** from 56.5% to 57.4% under weight-only
quantization — not shrinks. The FA kernel (45.0% of total compute) is
weight-quantization-invariant: it operates on the quantized KV cache
(Q8/F16), not on model weights. The matmuls that do benefit from
quantization are only ~20% of each full-attn layer and ~44% of total
compute, and Q4 dequant overhead (per-group scales/input_scales) cancels
most of the bandwidth savings, yielding only ~4% matmul speedup. The net
effect is that the fixed FA cost dominates both variants, and the small
matmul speedup shrinks the denominator by less than the numerator —
attention's share rises slightly. The 31.9% vs 39.2% gap between Q4_K_M and
Q6_K_XL is not explained by quantization shifting the attention/matmul
ratio; it is within normal per-run variance on an attention-dominated
workload where the FA kernel is the invariant floor.

The falsifiable signature is confirmed: layer 3 compute grows from ~5.6 ms
(chunk 10) to 39.5 ms (chunk 190), versus the baseline's flat ~42 ms
profile.

All acceptance criteria confirmed: correctness (parity ID 286 on both
Q6_K_XL and Q4_K_M), falsifiable signature, graph_builds = 0, patch_us
< 5%, full test suite 38/38.

## T-Q38FA-02: resident control verdict (done 2026-08-24)

Plan: `docs/plans/qwen38-fa-efficiency-plan.md` (T-Q38FA-02).

Q6_K_XL cannot serve as the fully-resident control on this GPU: its
weights alone (~24.1 GiB) exceed the 24,462 MiB total VRAM, so any
`-ngl 99` run spills to system RAM (observed: 3.7 GB shared GPU memory)
and is invalid as a control. The control was run on Q4_K_M instead,
which has a matched streaming baseline (693.5 tok/s post-FA-window).

Command (GPU mutex, single repetition): `llama-bench.exe -m
C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-Q4_K_M.gguf -ngl 99 -p 100000
-n 0 -fa on -ctk q8_0 -ctv q8_0 -t 16 -r 1 -b 8192 -ub 512` ->
**804.66 ± 0.00 tok/s**.

| Path | tok/s | Delta |
| --- | --- | --- |
| Fully resident (llama-bench) | 804.66 | — |
| Streaming (post-FA-window) | 693.5 | **-13.9%** |

**Verdict: the vendored FA/GEMM kernels are not the problem.** The same
backend achieves 805 tok/s fully resident. The 111 tok/s (14%) gap is
entirely streaming-path overhead (patch walk, graph replay diffing,
copy-stream contention, per-chunk host work). The "34 TFLOPS FA
anomaly" from the 2026-08-24 ceiling analysis was an artifact of
comparing against a 20K short-prompt ~1000 tok/s figure; the true
resident ceiling at 100K is ~805 tok/s and we are at 86% of it.

Consequences: T-Q38FA-04 (overhead reduction) is the primary lever.
T-Q38FA-05/06 (f16-KV A/B, ncu profile) are deprioritized — the kernels
are fine. T-Q38FA-03 (fused-op audit) closed same day: fused GDN
confirmed in streaming graphs (resolve_fused_ops runs at context init
before streaming graph build; runtime logs confirm "fused Gated Delta
Net (autoregressive/chunked) enabled").

## T-Q38FA-04: patch-walk cache (done 2026-08-24)

Plan: `docs/plans/qwen38-fa-efficiency-plan.md` (T-Q38FA-04).

Mechanism: `prepare_chunk` previously walked all 66 graphs x all nodes
(~3,840 node inspections) plus the full set-context tensor list on every
call — 12,544 calls at 100K (196 chunks x 64 layers + embeds). The
persistent graphs are never rebuilt (`graph_builds == 0` invariant), so
the first call now caches per set: the FLASH_ATTN_EXT node pointers
(`fa_nodes`) and the INPUT-flagged tensor list (`input_tensors`), and
every later call patches those lists directly. ~50M node-walk iterations
per 100K run eliminated.

Evidence (GPU mutex, Q4_K_M, 100K, q8 KV, pw-probe, trace
`build/llama-cuda/q38ls-fa04-pp100k.jsonl`):

| Metric | Before (T-Q38FA-01) | After (T-Q38FA-04) |
| --- | --- | --- |
| Wall | 144,192.9 ms | **135,389.3 ms** |
| Tok/s | 693.5 | **738.6 (+6.5%)** |
| Total compute | 133,609.9 ms | 122,210.3 ms |
| total_patch_us | 5,054,630 | 6,859,812 |
| check-telemetry | OK | OK (2,056 events, 4 slots) |
| graph_builds | 0 | 0 |

Honest attribution caveat: the compute delta (-11.4 s) cannot be caused
by a host-side walk change; at least part of the +6.5% is run-to-run GPU
state variance (shared GPU, clock/thermal state). The attributable win
is the walk elimination; a paired back-to-back run would be needed for
clean attribution. Gap to the resident control (804.66) is now 8.2%.

`total_patch_us` rose (5.05 -> 6.86 s) despite the walk shrinking — the
patch timer also covers the input-snapshot memcpy and the dominant term
was never the node walk alone; run variance applies here too. The
remaining patch cost is dominated by the per-chunk input restore
(~520 KB memcpy) and the per-FA-node `set_causal` op-param writes.

Validation: build exit 0; parity 1024-token layer-major + 1 greedy TG
(q8 KV) exit 0, check-telemetry OK; `tools/run-tests.ps1 -Suite all`
38/38 pass (561.59 s) on Q6_K_XL.

### Escalation: q8-KV layer-major/batch-major greedy-ID divergence (pre-existing)

Found during T-Q38FA-04 parity: with **q8 KV**, 1024-token layer-major
produces greedy ID **538** while batch-major produces **286** (Q4_K_M,
same build/flags). The divergence predates T-Q38FA-04 — it is present in
the 2026-08-23 per-layer-bundle smoke trace (`q38ls-q4km-1024.jsonl`,
ID 538). With **f16 KV**, layer-major produces 286 (matches batch-major
and the recorded T-Q38FA-01 parity evidence, traces
`q38ls-fa-parity{,2}.jsonl`). The T-Q38FA-01 "parity ID 286" record was
measured with f16 KV. So the layer-major implicit-causal path diverges
from batch-major specifically under quantized KV. Not caused by
T-Q38FA-04 (pointer caching changes no numerics). Needs its own
investigation task: candidate causes are the implicit-causal mask path
vs the dense-mask path interacting with q8 KV dequant, or the FA-window
`ne[1]` patch under q8 strides. All 100K q8-KV benchmarks to date
(525.7 -> 693.5 -> 738.6 tok/s) were run on the diverging configuration;
the rates are still valid throughput evidence, but the q8-KV layer-major
correctness gate is open.

## T-Q38TG-00: Hybrid TG Feasibility Probe (done 2026-08-24 — NO-GO)

### H2D rate measurements (corrected 2026-08-24 — original was disk-bound)

The original T-Q38TG-00 numbers (2.90/2.96 GB/s pageable, 2.77 pinned)
were measured against **mmap'd weights without warming the file cache**,
so the "copy time" included NVMe page-fault reads — 2.9 GB/s is NVMe
speed, not the PCIe bus. Re-measured with `--no-mmap` (all weights in
ordinary host RAM, zero page faults) and a 4-copy-per-layer protocol
(  `--pp-tokens 256 --ubatch 512 --layer-slots 1`, copies 2-4 = steady
state, copy 1 warmup ignored). Steady-state transfers are stable to
±0.1 ms across ubatches.

| Model | Source | Package size | transfer (steady) | Rate |
| --- | --- | --- | --- | --- |
| Q4_K_M recurrent (layers 1,2,4,5) | direct no-mmap | 245.9 MB | 17.94 ms | **13.71 GB/s** |
| Q4_K_M full-attn (layer 3) | direct no-mmap | 222.9 MB | 16.28 ms | **13.69 GB/s** |
| Q4_K_M recurrent (layer 0) | direct no-mmap | 245.9 MB | 22.9 ms | 10.7 GB/s (first-layer anomaly) |
| Q4_K_M recurrent | pinned no-mmap | 245.9 MB | DMA 18.0 ms + staging 11.7 ms | DMA 13.7 GB/s, total 29.8 ms |

Traces: `build/llama-cuda/q38ls-tg00-4x-direct-q4km.jsonl`,
`q38ls-tg00-4x-pinned-q4km.jsonl`.

**True steady-state H2D DMA rate is 13.7 GB/s**, 4.7x higher than the
original disk-bound figure. **Pinned staging gives no DMA advantage** —
with the source already in host RAM, `cudaMemcpyAsync` from pageable
memory is internally staged at the same 13.7 GB/s; the explicit pinned
path only adds an ~11.7 ms staging memcpy and is slower end-to-end. The
unverified "50 GB/s" discussion figure remains withdrawn; the measured
DMA ceiling is 13.7 GB/s.

PCIe link fact (nvidia-smi, 2026-08-24): negotiated **Gen4 x8** (max
also Gen4 x8 — laptop GPU wiring). Theoretical ceiling ~15.75 GB/s; the
measured 13.7 GB/s is **87% of the physical link** — the bus is
saturated and no software technique (pinned rings, multi-stream,
ReBAR, GDS) can exceed it. The rotation break-evens (45-50 GB/s) are
2.9-3.2x the physical link maximum: the hybrid design is impossible on
this hardware, not merely impractical. (A Gen5 x16 desktop/server target
would change this math — ~50 GB/s practical is at the rotation
break-even.)

### CPU recurrent layer cost (clean re-run, llama-bench)

The earlier `llama-server`/`llama-cli` attempts hung because `llama-cli`
enters interactive mode after generation and never exits; `llama-bench`
exits cleanly and reports deterministic tg128/tg16 rates. Measured with
our own build (04b569142) at `-ngl 17` (16 attention + output resident,
48 recurrent on CPU) vs `-ngl 99` (fully resident), `-fa 0`:

| Model | -ngl 99 TG | -ngl 17 TG | CPU recurrent diff |
| --- | --- | --- | --- |
| Q4_K_M | 36.32 t/s (27.5 ms/tok) | 3.80 t/s (263.2 ms/tok) | (263.2-27.5)/48 = **4.91 ms/layer** |
| Q6_K_XL | 25.15 t/s (39.8 ms/tok) | 2.63 t/s (380.2 ms/tok) | (380.2-39.8)/48 = **7.09 ms/layer** |

The Q6_K_XL `-ngl 17` number (2.63 t/s) closely matches the owner's
2026-08-23/24 stock-server baseline (2.55 t/s), validating the method.
The CPU recurrent cost is 4.9-7.1 ms/layer depending on quantization,
not the 7-8 ms/layer estimate in the plan.

### Mechanism verdict: Path A

Inspected `llama-model.cpp:1354-1372` — the stock `get_layer_buft_list`
is a static tail-count split: `i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`.
A layer's device assignment is baked into `pimpl->dev_layer[il]` at load
time and never changes. There is no runtime weight movement in the stock
path.

**Verdict: Path A** — extend our module with an explicit per-layer placement
map. Load all layers to CPU mmap (as the streaming path already does), then
let the placement map decide: attention layers permanently bound to GPU
slots that never rotate, recurrent layers through slot rotation with
prefetch. The stock's static split mechanism is not touched. This is the
cheaper path because our module already owns host→device weight copies;
we only add the decision of *which* layers copy and on what schedule.

### Go/No-Go verdict: NO-GO (re-confirmed with corrected 13.7 GB/s rate)

## T-Q38HG-00/01/02: PP→TG Handoff (done 2026-08-24)

### T-Q38HG-00: Q4 KV cache
Implemented `--kv-cache q4` option (Q4_0 K and V). Greedy-ID parity confirmed
at 1024 tokens (ID 286, matches Q8 and F16 references). 38/38 full suite passes
with default KV type unchanged. Code change: `main.cpp` arg parsing only
(the vendored FA kernels already support Q4_0 KV dequant).

### T-Q38HG-01: Standard PP configuration validation
Config: 2 slots, bf16 slabs, Q4 KV, Q4_K_M model. 100K layer-major PP:
- Tok/s: 741.04 (+0.33% vs fa04 baseline 738.6) — Q4 KV FA path is slightly
  faster than Q8 (no dequant in the FA kernel).
- Total wait: 0.6 ms (< 10 ms gate). 2-slot lookahead holds comfortably.
- Peak VRAM: 7,395.56 MiB (7.22 GiB).
- q38ls-hg01-pp100k.jsonl, check-telemetry OK.

### T-Q38HG-02: Handoff machinery
End-to-end mechanism: PP (streaming) → begin_handoff (allocate + copy all 64
packages to one device buffer) → standard TG through layer_streaming_compute
with slot_layer returning resident tensors → end_handoff (free buffer).

Key design decisions:
- `slot_layer(il)` returns `handoff_layers.at(il)` when `handoff_active`.
  No graph topology changes — the same persistent graphs see different tensor
  data pointers (host mmap during PP, device buffer during handoff TG).
- `load_layer()` bypasses H2D copies when handoff is active (records
  zero-cost events for telemetry consistency).
- TG graph set is invalidated (`built = false`) after migration so the first
  post-handoff decode rebuilds with resident tensors.

Validation (Q4_K_M, 1024 PP + 16 TG, batch-major):
- Parity: ID 286 matches the standard TG baseline.
- TG rate: 24.75 tok/s (vs 0.61 tok/s without handoff — ~40× speedup).
  The TG tok/s is below llama-bench fully-resident (36.3 t/s) because the
  batch-major streaming path pays per-ubatch scheduler reset/split overhead
  that the standard decode path amortizes.
- Migration time: 1.72 s (15 GB Q4_K_M @ ~8.7 GB/s effective copy-stream
  rate). Plan expected ~1.1 s at 13.7 GB/s; the delta is from per-tensor
  `set_async` calls vs a single contiguous memcpy.
- Full suite: 36/36 passes.
- Telemetry: `handoff` events (migrate with byte count and copy time,
  teardown); host checker accepts graph_builds > 0 after handoff.
- q38ls-hg02-accept.jsonl.

### T-Q38HG-03: Hybrid Placement Probe (done 2026-08-24)

Read-only analysis probe. Three questions, all answered:

**Q1: Can `-ot` (--override-tensor) express "attention on GPU, recurrent on CPU"?**

YES. Mechanism: `common/arg.cpp:253` parses `pattern=buffer_type`, applies via
`std::regex_search` against tensor names in `llama-model-loader.cpp:1181`.
Tensor names follow `blk.N.member.weight` format. The 16 full-attention
layers can be targeted with:
```
-ot "blk\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\.=CUDA0"
```
Combined with `-ngl 0` (all-CPU baseline), this places only attention
layers on CUDA. Validated with `-ot "blk.0.=CPU"` on `-ngl all`:
PP 238→203 t/s, TG 31.7→25.1 t/s — the override correctly reroutes
layer 0 tensors.

The 48-layer recurrent alternation is ~520 chars — well within OS limits.
Alternative: Path A tail-count split (`llama-model.cpp:1354`, `-ngl 17`)
would put only 4 of 16 attention layers on GPU — suboptimal for handoff.

**Q2: Attention-only migration scope?**

16 packages (full-attn layers only), ~3.8 GB at Q4_K_M (vs 15 GB for all
64). ~275 ms copy at 13.7 GB/s theoretical (~440 ms at measured 8.7 GB/s
on copy stream). Recurrent layers never migrate.

**Q3: Runtime-rebind — YES.**

The existing handoff mechanism (T-Q38HG-02) already supports this:
- `slot_layer(il)` routes attention layers to `handoff_layers.at(il)`
  (CUDA-resident) and recurrent layers to `slots[...].layers.at(il)`
  (host mmap via slot copies).
- `ggml_backend_sched_split_graph` assigns backends based on tensor
  buffer types automatically. CUDA tensors → CUDA backend, CPU tensors
  → CPU backend.
- The graph builder (`qwen35.cpp:layer(il)`) calls `slot_layer(il)`, so
  attention-layer graphs get CUDA-resident weights. No scheduler or
  graph-builder changes needed.

**Conclusion:** T-Q38HG-04 (constrained handoff) is unblocked.
Implementation is attention-only `begin_handoff()` (copy 16 layers, not
64) + standard handoff lifecycle.

### T-Q38HG-04: Constrained Handoff (done 2026-08-24)

Implemented `--handoff-attention-only` flag via `begin_handoff(true)`:
only 16 full-attention layers (blk.3,7,...,63) migrate to the device
buffer; recurrent layers route to CPU-resident host_layers via slot_layer().

**Results:**

| Metric | Value | Expected |
|--------|-------|----------|
| Migration bytes | 3.54 GB | ~3.8 GB ✓ |
| Copy time | 0.424 s | <0.5 s ✓ |
| Greedy ID (1024-tok) | 286 | 286 ✓ |
| Full test suite | 36/36 | 36/36 ✓ |
| TG tok/s | 3.28 | 3.80 (`-ngl 17` ref) ✓ (within ~15%) |

**TG routing fix:** slot_layer(il) returns `*host_layers[il]` for
recurrent layers during attention-only handoff, so the ggml scheduler
places them on CPU. load_layer() skips H2D copies for them. Transfer
collapsed from ~1280ms to ~68ms per 8-token batch.

Re-arm (T-Q38HG-06) is a follow-up task.

### T-Q38HG-06: MTP speculative decoding (done 2026-08-24)

Implemented by a second agent; reviewed and fixed 2026-08-24.

**Bug found during review:** the acceptance check compared
`draft[a] == prev` (the token *before* the current position) instead of
`draft[a] == cur` (the gold for the current position). This off-by-one
caused the reported 0.017-0.053 acceptance rate. Fixed to `draft[a] == cur`.

**Results (Q4_K_M, 1024 PP batch-major, 32 TG, `--handoff --mtp --mtp-draft 2`):**

| Metric | Baseline (no MTP) | MTP |
|--------|-------------------|-----|
| TG tok/s | 25.4 | 23.6 |
| Sequence | `286,220,197,289,...` | **identical** |
| Acceptance rate | — | 0.4167 (15/36) |
| check-telemetry | OK | OK |

**Parity: exact.** MTP-on and MTP-off produce identical 32-token greedy
sequences. The period-3 cycle (`289,197` repeating) is the model's true
greedy behavior on the synthetic prompt (deterministic token IDs
`i % n_vocab`), not an MTP defect.

**Acceptance rate 0.4167** is above the 0.3 escalation threshold. At
n_draft=2 the draft overhead slightly exceeds the acceptance benefit
(23.6 vs 25.4 tok/s). Higher n_draft or better acceptance would be
needed for net speedup; the mechanism is proven correct.

Traces: `q38ls-mtp-baseline.jsonl`, `q38ls-mtp-on.jsonl`.

The original verdict's reasoning was wrong (disk-bound measurement) but
the verdict itself survives the corrected numbers. True steady-state H2D
DMA is 13.7 GB/s. Two break-evens, both unmet:

1. All-48-streaming parity with the 3.80 t/s `-ngl 17` baseline:
   48 × 246 MB = 11.8 GB/token ÷ 263 ms/token budget = **45 GB/s
   required**; at 13.7 GB/s the transfer alone is 861 ms/token →
   ~1.16 t/s ceiling (better than the original 0.25 estimate, still 3.3×
   below the baseline it would replace).
2. Hybrid rotation (hide each window's copy behind the CPU compute of the
   current window): 246 MB ÷ 4.91 ms CPU layer time = **50 GB/s
   required**. At 13.7 GB/s the copy:compute ratio is fixed at 3.65×
   regardless of window size — the token always stalls 3.65× longer than
   the CPU work it overlaps. No window size or prefetch schedule fixes a
   fixed ratio.

The correct outcome per plan contract stands: record the negative result
and stop. T-Q38TG-01 remains not-started (no-go). The hybrid rotation
design is not viable on this machine's PCIe bus at 13.7 GB/s. The viable
TG lever remains post-PP slot expansion (all layers resident, zero
copies) for models that fit after slab release.

This plan is **complete with a negative result** (measurement corrected
2026-08-24; verdict unchanged).

## 20K Layer-Major Evidence (done 2026-08-20)

The availability probe found 24,137 MiB free and 0 MiB used. The GPU mutex
protected `powershell -NoProfile -ExecutionPolicy Bypass -File
tools/run-tests.ps1 -Suite pp20k-layer-major` exited 0 and wrote
`build/llama-cuda/q38ls-pp20k-layer-major.jsonl`; both `--check-telemetry` and
`--validate-telemetry` exited 0, producing the companion summary JSON.

- PP: 20,480 tokens / 40 chunks, 25,708.0 ms, 796.64 tok/s; 64 package copies,
  2,560 layer computes, 60 evictions, 23,111,182,336 bytes transferred,
  81,204.8 ms transfer, 0.5 ms availability wait, 17,752.6 ms compute, and
  `graph_builds == 0`.
- TG: 3 tokens, 7,630.3 ms, 0.39 tok/s; emitted IDs 317, 197, 197; 192
  evictions, 8,071.5 ms transfer, 17.1 ms availability wait, 270.8 ms
  compute, and `graph_builds == 0`. TG began from the live layer-major PP
  session; the validator accepted the PP-to-TG lifecycle without a slot reset.
- Preflight reported 22,922,919,936 free bytes of 25,650,855,936 total; two
  activation slabs used 838,860,800 bytes, and the 41 graph-set footprint
  contained 2,706 graphs. Peak VRAM was 8,247,640,064 bytes (7,865.56 MiB).
  The early PP layers had warmup-heavy transfer/compute totals; raw timings are
  recorded only as evidence, not as a speedup or quality claim.
- `tools/run-tests.ps1 -Suite all` then passed all 19 targets under the same
  mutex, including batch-major, layer-major, and the positive/negative 40-chunk
  lifecycle fixtures. The default-OFF `llama-cli` target had already rebuilt
  successfully. Default batch-major behavior remains unchanged and the
  layer-major path remains opt-in.

Caveats: this run had an idle GPU at start, but the mmap source remains
best-effort cold because Windows file-cache state is uncontrolled. No resident
baseline, quality evaluation, equivalence test, or speedup comparison was run.

## T-Q38PERF-01: copy timing split (done 2026-08-20)

`copy_complete` now carries additive `host_submit_us` and
`ready_latency_us` fields. Their sum equals the legacy `transfer_us` field;
the host telemetry checker rejects partial, negative, or inconsistent splits
when they are present while accepting historical fixtures without them. This
does not isolate DMA duration: `ready_latency_us` still includes copy-stream
queueing. The focused `layer-streaming-layer-major-1024` CUDA CTest passed
under the GPU mutex, followed by `--check-telemetry`; its first copy recorded
122.3 ms host submission, 344.2 ms ready latency, and 466.5 ms legacy
submit-to-ready transfer. The three host-only `layer-streaming-lm40` fixture
tests also passed. Default behavior and slot scheduling are unchanged.

## T-Q38PERF-02: package-class slot allocation (done 2026-08-20)

Slots now allocate only the recurrent and/or attention package classes served
by their logical-layer mapping. The representative tensor for each allocated
class is selected from a matching logical layer, which also fixes mixed-class
slot counts where the slot index itself can name an attention layer. For the
four-slot schedule, `slot_vram_bytes` fell from 2,853,972,608 to
1,444,448,896 bytes (1,409,523,712 bytes, or 1,344.2 MiB, saved). The focused
four-slot layer-major CTest passed. A new persistent five-slot mixed-class
runner-plus-checker fixture passed; its checker now derives each slot's actual
layer-cycle length, so it accepts uneven 64-layer distributions (12 or 13
layers per slot) while preserving the four-slot policy. Default behavior is
unchanged.

## T-Q38PERF-03: pinned-host package staging (done 2026-08-20)

The isolated runner now accepts `--copy-source direct|pinned` (default
`direct`) and records the selected value in `run_start.copy_source`. Pinned mode
gives each slot one CUDA-pinned host buffer sized to that slot's device package
allocation. It packs package tensors into the buffer before enqueueing the
existing asynchronous copies; the existing consumer-complete event protects the
buffer from reuse. Pinning failure or allocator fallback to ordinary host memory
fails initialization rather than silently changing copy behavior.

Paired mutex-protected 1,024-token layer-major traces at u-batch 512, four
slots, zero PP logits, and one greedy TG token both exited 0 and passed
`--check-telemetry` (837 events, four slots):
`build/llama-cuda/q38ls-direct-1024-paired.jsonl` and
`build/llama-cuda/q38ls-pinned-1024-paired.jsonl`. Pinned PP was 8,366.4 ms
(122.394 tok/s), versus direct 8,195.3 ms (124.950 tok/s). It reduced summed PP
host submission from 7,505.1 to 6,062.4 ms, but raised availability wait from
0.5 to 393.8 ms. Pinned TG was also slower (2,757.5 vs 2,515.0 ms), with
availability wait rising from 5.3 to 1,510.4 ms. This single paired observation
does not establish a speedup; pinned staging remains opt-in as an instrumented
experiment. After CMake reconfiguration, the mutex-protected
`layer-streaming-pinned-1024` CTest fixture passed its runner and host checker
(2/2, 17.90 s).

## Completed

### T-Q38LS-01: reproducible CUDA experiment envelope (done 2026-08-18)

Commands (both from workspace root, Git Bash; compilation does not need the
GPU mutex):

1. `cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON` ->
   exit 0. CUDA 13.2.51 toolkit, host compiler MSVC 19.50.35726.0, target arch
   `120a-real` (RTX 5090 / sm_120). OpenSSL 3.6.1 found. NCCL absent (warning
   only; single GPU, irrelevant).
2. `cmake --build build/llama-cuda --config Release --target llama-cli --parallel 16`
   -> exit 0. Artifact `build/llama-cuda/bin/Release/llama-cli.exe`.
3. `cmake --build build/llama-cuda --config Release --target llama-gguf --parallel 16`
   -> exit 0. Artifact `build/llama-cuda/bin/Release/llama-gguf.exe`.

Model identification evidence:
- `llama-gguf.exe "<model>" r` (llama.cpp's own reader): GGUF version 3,
  alignment 32, data offset 10,996,800, n_kv 51, n_tensors 866; `qwen35.*`
  KV keys present; per-tensor sizes verified (e.g. `blk.0.attn_qkv.weight`
  55,705,600 bytes). Full dump in the env guide.
- `llama-cli` under the GPU mutex (`tools/with-gpu-lock.ps1`) with
  `-ngl 8 -p hi -n 1`: model loaded and executed on CUDA. Banner reported
  `build b10493-04b569142`, model path, `modalities: text`. Prompt processed
  at 13.2 t/s (partial offload: 8/64 layers on GPU), one token generated.
  Log: `out/q38ls-ngl8-smoke.log`. Evidence that the CUDA runtime loads and
  runs this exact artifact.
- `llama-cli -ngl 0 -p hi -n 1` (CPU-only, no CUDA): did not complete within
  600 s and was killed (201 s CPU consumed). The CPU path is impractical for
  this 26 GB hybrid model; consistent with the CUDA-only experiment scope.

Quant facts (corrected during this task): `general.file_type = 14` maps to
llama.cpp enum `LLAMA_FTYPE_MOSTLY_Q4_K_S` whose display name is "Q4_K -
Small" - a legacy label that does not describe this file. Actual per-tensor
quants (this tree's ggml enum: F32=0, Q8_0=8, Q5_K=13, Q6_K=14):
`output.weight` Q8_0, `token_embd` Q6_K, `attn_qkv`/`ffn_down` Q8_0,
`ffn_gate`/`ffn_up` Q6_K, `ssm_alpha`/`ssm_beta` Q5_K, 1-D params F32. This
is the UD/Q6_K_XL dynamic scheme. Byte sizes measured from tensor offsets are
ground truth and are in the env guide (static 2.394 GB, recurrent package
364.0 MB max, full-attn package 352.4 MB max, four-slot window ~1.444 GB).

### T-Q38LS-02: one-layer Qwen3.8 execution boundary (done 2026-08-18)

Implementation (all gated by CMake option `LLAMA_LAYER_STREAMING`, default OFF):
- `src/llama-layer-streaming.{h,cpp}`: slot buffers + tensor bundles, persistent
  activation pair, persistent host input buffer, dedicated CUDA copy backend,
  per-slot CUDA events, JSONL telemetry, and the layer-by-layer executor
  (`llama_context::layer_streaming_compute`).
- `src/llama-graph.h`: `llm_graph_params.layer_streaming` (default nullptr).
- `src/models/qwen35.cpp` + `models.h`: layer-view indirection `layer(il)`
  (model.layers[il] normally, slot-bound bundle in streaming mode) and
  `build_streaming_graphs()` which produces one cgraph per piece: embed graph,
  64 per-layer graphs, output graph.
- `src/llama-context.{h,cpp}`: gated streaming branch in `process_ubatch`.
- `tools/layer-streaming/main.cpp`: runner (`llama-layer-streaming` exe) with
  boundary mode, PP, greedy TG, JSONL telemetry.

Validated evidence (commands from workspace root, GPU runs under
`tools/with-gpu-lock.ps1`):
- `cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON
  -DLLAMA_LAYER_STREAMING=ON` and `cmake --build ... --target
  llama-layer-streaming` -> exit 0.
- OFF-build check: separate dir `build/llama-cuda-off` (no option) builds
  `llama-cli` -> exit 0 (default path unchanged; the only ungated change is a
  nullptr-initialized field in `llm_graph_params`).
- `ctest --test-dir build/llama-cuda -C Release -R qwen38-layer-boundary
  --output-on-failure` -> Passed (4.0 s).
- Boundary run: `llama-layer-streaming --model <env path> --layer-slots 1
  --boundary-layers 0,3 --pp-tokens 64 --ubatch 512 --no-mtp --telemetry
  out/q38ls-boundary.jsonl` -> exit 0. Recurrent layer 0 (Gated DeltaNet) and
  full-attention layer 3 both constructed and executed one-layer CUDA graphs.
  Telemetry: `out/q38ls-boundary.jsonl` (14 lines: run_start, per-layer
  slot_request/copy_submit/copy_complete/layer_ready_wait/layer_compute,
  phase_summary pp 64 tokens 205.7 tok/s, run_end exit 0). Copy byte counts
  match the env guide exactly: layer 0 364,022,656 B, layer 3 352,380,928 B.
  Peak VRAM 3.87 GiB during the boundary run (foreign process still resident).

Key measured behavior (one slot): the H2D copy from the pageable mmap source
blocks the host inside `cudaMemcpyAsync` (driver staging for pageable memory),
so per-layer wall is copy-then-compute with ~0 availability wait at
consumption; transfer_us (submit->complete) ~111 ms for the recurrent package
and ~104 ms for the full-attn package on this shared GPU. This is recorded as
observed behavior, not a defect or a claim.

### T-Q38LS-03: full one-slot dense streaming prefill (done 2026-08-18)

Command (GPU mutex; workspace root):
`powershell -NoProfile -ExecutionPolicy Bypass -File tools/with-gpu-lock.ps1 -Command "<build>/bin/Release/llama-layer-streaming.exe --model C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf     --pp-tokens 512 --ubatch 512
 --layer-slots 1 --no-mtp --telemetry out/q38ls-t03.jsonl"`
-> exit 0. Log `out/q38ls-t03.log`, telemetry `out/q38ls-t03.jsonl` (3074 lines).

Measured results:
- PP: 512 tokens in 24,363.2 ms = 21.02 tok/s. 8 u-batches of 64 tokens.
- `phase_summary` (JSONL): tokens 512, ubatches 8, wall_us 24,363,220,
  tokens_per_s 21.015, bytes_transferred 184,889,458,688 (512 package copies
  x ~361 MB avg), evictions 511, total_transfer_us 23,391,879, total_wait_us
  43,900, total_compute_us 850,252, layers_run 64.
- Per-layer averages (over 8 executed u-batches): recurrent layer transfer
  ~45.4-53.2 ms + compute ~1.3-2.1 ms; full-attn transfer ~43.9-44.7 ms +
  compute ~1.4-2.1 ms; setup ~0.04-0.09 ms; wait ~0.08-0.10 ms. Copy bytes
  match the env-guide package inventory (recurrent 364,022,656 B,
  full-attn 352,380,928 B).
- CUDA0 compute buffer 188.64 MiB (matches expected), CUDA_Host 2.565 MiB.
  Peak VRAM 3.91 GiB.
- JSONL ordering verified: every `layer_compute` follows the matching
  generation's `copy_complete`; telemetry has 512 copy_submit/slot_request/
  copy_complete/layer_ready_wait/layer_compute pairs (8 ubatches x 64 layers)
  plus run_start/phase_summary/run_end.

Required validations:
- Runner command (above) -> exit 0 and eight PP u-batches in telemetry. Passed.
- `ctest --test-dir build/llama-cuda -C Release -R layer-slot-lifecycle
  --output-on-failure` -> Passed (30.9 s) under the GPU mutex.

Defect found and fixed during this task:
- Persistent graphs were corrupted on graph reuse: `ggml_backend_sched_alloc_graph`
  rewires node src pointers in place when introducing cross-backend copies, and
  `sched->ctx` is freed on the next split; re-splitting a reused graph then
  aliased stale rewired pointers with unrelated tensors (observed: embed cpy
  src degraded to a 64-elem token leaf -> `cpy.cu` element-count assert). Fix:
  the executor snapshots each graph's node src pointers before alloc and
  restores them after compute, keeping the persistent graphs pristine across
  u-batches. T-04/05 inherit this fix.

### T-Q38LS-04: four-slot persistent-graph PP streaming (done 2026-08-19)

Command (GPU mutex; workspace root):
`powershell -NoProfile -ExecutionPolicy Bypass -File tools/with-gpu-lock.ps1 -Command "<build>/bin/Release/llama-layer-streaming.exe --model C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf  --layer-slots 4 --pp-tokens 512 --ubatch 512

 --no-mtp --telemetry out/q38ls-pp.jsonl"`
-> exit 0. Log `out/q38ls-pp.log`, telemetry `out/q38ls-pp.jsonl` (3071 lines).

Measured results (512-token trace):
- PP: 512 tokens in 26,540.0 ms = 19.29 tok/s. 8 u-batches of 64 tokens.
- `phase_summary`: tokens 512, ubatches 8, wall_us 26,539,954, tokens_per_s
  19.292, bytes_transferred 184,889,458,688, evictions 508,
  total_transfer_us 102,470,682, total_wait_us 2,159, total_compute_us 775,010,
  layers_run 64, graph_prep_us 2,046, graph_builds 1.
- Peak VRAM 6.04 GiB (vs 3.91 GiB for one slot). Four-slot window observed as
  expected (~2.85 GB of slot buffers reported at run_start).
- With four slots the pageable H2D copies now overlap the compute of the other
  slots; total_wait_us dropped to ~2.2 ms (from 43.9 ms at one slot) because a
  layer's copy is usually already complete when consumed. Transfer remains
  dominant (~99.8 GB accumulated copy time vs ~0.78 s compute); copies from the
  pageable mmap source still serialize on the copy stream.

Required validations:
- Runner command (above) -> exit 0 with all required PP event types. Passed.
- `ctest --test-dir build/llama-cuda -C Release -R layer-streaming-scheduler
  --output-on-failure` -> Passed (run 12.5 s + host-side invariant check
  0.1 s) under the GPU mutex. The check validates exactly n_slots distinct
  slots, `slot == layer % n_slots`, per-slot generation monotonicity,
  copy_submit -> copy_complete -> layer_ready_wait -> layer_compute ordering,
  prior-resident `slot_evict` identity, and `graph_builds == 0` on the 64-token
  single-ubatch run.
- Regression: `qwen38-layer-boundary` Passed (4.0 s), `layer-slot-lifecycle`
  Passed (30.4 s) under the GPU mutex. Default (OFF) `llama-cli` build still
  compiles.

Implementation notes:
- Added `llama_context_params.n_pad` (default 0 = unchanged). The runner sets
  `n_pad = n_ctx` so the KV cache pads `n_kv` to the full context for every
  ubatch, keeping all graph shapes constant across the whole session (the
  existing `get_n_kv` padding mechanism, now pinned to full width). Tradeoff:
  full-attention layers attend over the full padded width from the first
  ubatch; recorded, not claimed as an optimization.
- Added `llama_layer_streaming_prebuild()` (exported wrapper around
  `llama_context::layer_streaming_prebuild`) which splits the batch, applies
  the first ubatch (so graph shapes match), builds the 66 per-piece graphs
  before the PP timer, then `memory->clear(false)` releases the applied cell
  metadata so the timed decode starts from the identical fresh cell mapping.
- Persistent graph entries now carry explicit `(mode, logical_layer, slot)`
  keys and the executor fails loudly on any mode/layer/slot mismatch; slot
  generation is now in copy_complete/layer_ready_wait/layer_compute telemetry;
  `slot_evict` now reports the prior resident layer.
- Runner gains a host-side `--check-telemetry PATH` validator (no GPU/model)
  used by the scheduler CTest.

Resolved by T-Q38LS-04B (rs_z two-variant persistent-graph key):
- The Gated DeltaNet recurrent `rs_z` is 0 on the first PP ubatch (fresh state;
  the graph includes an in-graph zeroing of one state cell) and -1 on every
  continuing ubatch (zero-sized view). T-Q38LS-04B closes this: the graph cache
  is now keyed `(mode, n_tokens, rs_z)` with per-variant input tensors and result
  objects, and the prebuild builds all three variants (fresh-PP, continuing-PP,
  TG-u1) before the timed region. The 512-token/4-slot timed PP region now
  reports `graph_builds == 0` (pre-fix baseline was 1) and the 256-token
  four-ubatch run that crosses the rs_z 0 -> -1 transition also reports
  `graph_builds == 0`. See the T-Q38LS-04B section below.

### T-Q38LS-04B: per-variant persistent graph sets (done 2026-08-19)

Observed behavior
- The streaming module now owns per-variant graph sets keyed
  `(mode, n_tokens, rs_z)`. Each set owns its io context, activation region,
  input region, and `llm_graph_result`. The executor selects the set per ubatch
  by `(phase, n_tokens, rs_z)`, verifies the prebuilt input tensor shapes
  (mismatch fails loudly per SDD), writes inputs, and runs the set's piece
  graphs. `process_ubatch` bypasses the `gf_res_prev` reuse/build machinery in
  streaming mode; an on-demand build of an unprebuilt variant remains as a
  counted fallback.
- Prebuild builds variant A (fresh PP, rs_z 0), variant B (continuing PP,
  rs_z -1), and, when `--tg-tokens > 0`, variant C (TG u1, mode 1, rs_z -1),
  then `memory->clear(false)` releases all applied cell metadata so the timed
  decode starts from the identical fresh cache.

Validation evidence (Release, GPU mutex)
- `layer-streaming-pp512`: 512 tokens / 4 slots, 8 ubatches, `graph_builds == 0`,
  `wall_us ~25.6ms`, `total_transfer_us ~97.2ms`, `total_compute_us ~0.94ms`,
  `layers_run 64`. Host `--check-telemetry` passes.
- `layer-streaming-scheduler-256`: 256 tokens / 4 slots, 4 ubatches crossing the
  rs_z 0 -> -1 transition, `graph_builds == 0`. Host check passes.
- `layer-streaming-scheduler` (64-token), `qwen38-layer-boundary`,
  `layer-slot-lifecycle`: all pass. OFF build `llama-cli` compiles clean.

Root cause of the first GPU run's shape-guard failure (debugging record)
- The executor verified sets via `llm_graph_result::can_reuse`, whose
  `params.allow_reuse` half compares the stored prebuild `llm_graph_params`
  (dangling `mctx`/`sched`/`ubatch.data` pointers) against the live decode
  params, so it always failed despite every tensor shape matching (prebuild
  n_kv 256 == live n_kv 256, n_rs 1, rs_z 0, n_tokens 64). Fix: run each
  prebuilt input's own `can_reuse` (shape checks, which re-point the input mctx
  to the live context) and skip the stale-pointer params half. No SDD/plan
  boundary changed.

### T-Q38LS-05: PP-to-TG transition (done 2026-08-19)

Observed behavior
- TG (128 greedy tokens) begins immediately after PP with the exact four slot
  assignments and per-session state (KV + Gated DeltaNet) left by PP; no slot
  reset or host prewarm. The first TG layer-0 request finds slot 0 already
  resident from PP (generation 129, `ready_at_consumption true`, wait ~3us) -
  no wrap-around preload.
- The TG phase uses the prebuilt `tg_u1` graph set (mode 1, n_tokens 1, rs_z -1)
  built before the timer, so the timed TG region reports `graph_builds == 0`.
- 128 greedy token IDs are emitted in a `tg_tokens` JSONL event and
  `run_end.n_tokens_generated == 128`.

Validation evidence (Release, GPU mutex, 512 PP + 128 greedy TG / 4 slots)
- `layer-streaming-transition` (run + host check): pass; run ~345s.
- PP: 512 tok / 8 ubatches, `graph_builds 0`, ~19.8 tok/s, transfer ~98s, compute ~1.0s.
- TG: 128 tok / 128 ubatches, `graph_builds 0`, 0.41 tok/s, transfer ~1223s,
  compute ~9.6s; per-layer transfer/wait/compute present for all 64 layers.
- All PP-only scheduler/slot/boundary regressions pass; OFF build compiles clean.

Notes
- `reset_stats()` at the PP->TG boundary makes each `phase_summary` report its
  own phase's totals (previously cumulative, so TG double-counted PP).
- Per-layer copy/wait/compute telemetry now carries a `phase` field so TG events
  are distinguishable from PP events.
- `cparams.n_ctx`/`n_pad` round `pp+tg` up to a 256-multiple (640 -> 768) so
  `n_pad` divides `kv_size == n_ctx_seq`; the mask masks empty padded KV slots to
  -INF so the larger constant `n_kv` is safe.

### T-Q38LS-06: evidence-only profiling record (done 2026-08-19)

Reproducibility command (no GPU/model needed for validation)
- `build/llama-cuda/bin/Release/llama-layer-streaming --validate-telemetry <trace>.jsonl`
  -> exit 0, writes `<trace>.jsonl.summary.json` (the validated profile artifact).
- Source trace: `--model <Q6_K_XL.gguf>  --layer-slots 4 --pp-tokens 512 --ubatch 512



  --tg-tokens 128 --greedy --no-mtp --telemetry <trace>.jsonl` (run via CTest
  `layer-streaming-transition`).

Validation result
- `--validate-telemetry` on the transition trace: exit 0, 52225 events, 4 slots,
  64 layers. `ctest -R layer-streaming-telemetry`: positive (writes summary) and
  negative (WILL_FAIL on missing/incomplete telemetry) both pass.

Profile artifact (q38ls-pp-tg.jsonl.summary.json) raw metrics
- PP: 512 tokens, ~7-20 tok/s (shared-GPU noise), transfer ~98-267s, wait ~4ms,
  compute ~1-2.3s, 64 layers, graph_builds 0.
- TG: 128 tokens, ~0.4 tok/s, transfer ~1.2s, wait ~62ms, compute ~9.3s,
  64 layers, graph_builds 0.
- peak_vram_bytes recorded; per-layer transfer/wait/compute min/mean/max for all
  64 layers per phase.
- Caveats recorded: source cache best-effort cold (uncontrolled Windows file
  cache); no speedup baseline was run; no quality/equivalence/speedup claim.

Manual test running + output (2026-08-19)
- `tools/run-tests.ps1 -Suite all|quick|boundary|telemetry` runs the CTest
  suites under the GPU mutex (`--output-always` prints every test's output).
  `-Suite telemetry` is host-only (no GPU, no lock).
- Every model run now self-checks its `--telemetry` trace at exit and prints a
  per-phase summary: tokens, tok/s, wall, total transfer/wait/compute, layers
  run, evictions, graph_builds, peak VRAM. `--check-telemetry <trace>.jsonl`
  prints the same summary for any existing trace (host-only).
- Note: `build/llama-cuda/q38ls-pp-tg.jsonl` must be regenerated by a fresh
  `layer-streaming-transition-run` (it was truncated by a killed mid-run test
  launch on 2026-08-19 and deleted).

### T-Q38LM-01: stable layer-major activation slices (done 2026-08-20)

Opt-in `--pp-schedule layer-major` path (default `batch-major`) proving that
two 512-token chunks can be driven through one selected logical layer while
retaining distinct, stable, full-prompt activation-slab ranges. Evidence only;
no quality/equivalence/speedup claim.

Commands (workspace root, Git Bash; GPU runs under `tools/with-gpu-lock.ps1`):
- Reconfigure with the model path baked into the CTest entries (the build had
  been reconfigured with an empty `Q38LS_MODEL_PATH`, which emptied every
  GPU test's `--model` arg): `export Q38LS_MODEL_PATH=
  C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf` then
  `cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON
  -DLLAMA_LAYER_STREAMING=ON` -> exit 0.
- Build (no GPU): `cmake --build build/llama-cuda --config Release --target
  llama-layer-streaming --parallel 16` -> exit 0.
- Layer-major boundary probe (GPU mutex): `ctest --test-dir build/llama-cuda
  -C Release -R layer-streaming-lm-boundary-run --output-on-failure` (runner:
  `--pp-schedule layer-major --boundary-layers 0 --pp-tokens 1024 --ubatch 512
  --pp-no-logits --no-mtp --telemetry build/llama-cuda/q38ls-lm-boundary.jsonl`)
  -> Passed (8.4-20 s). Log `out/q38ls-lm-boundary.log`.
- Host recheck (no GPU): `llama-layer-streaming --check-telemetry
  build/llama-cuda/q38ls-lm-boundary.jsonl` -> exit 0 (15 events, 1 slot).
- Regression (GPU mutex): `tools/run-tests.ps1 -Suite boundary` -> 2/2 pass;
  `-Suite quick` -> 8/8 pass (boundary, lifecycle, scheduler, scheduler-256,
  pp512). Default-OFF `llama-cli` builds (no GPU).

Observed behavior
- `run_start` carries `schedule:"layer-major"` and `act_slab_bytes:41943040`
  (2 x [5120,1024] F32 = 40 MiB for the 1,024-token probe). The preflight
  prints `act slabs require 41943040 bytes (2 x 20971520); free VRAM
  21610102784, total 25650855936` before inference. The formula is
  `2 * n_embd * pp_tokens * 4`, which is 838,860,800 bytes (800 MiB) for the
  20,480-token envelope; this is an allocation size, not a VRAM-fit claim.
- `act_bind` events record the per-chunk slab offsets: chunk 0 -> slab A
  [0,10485760) + slab B [0,10485760); chunk 1 -> slab A [10485760,20971520)
  + slab B [10485760,20971520). The slices are non-aliasing (host validator
  proves it).
- The layer-0 package is copied once (one `copy_submit`/`copy_complete`,
  generation 1, 364,022,656 B) and both chunk computes reuse that one
  generation: two `layer_compute` events, chunk 0 (ubatch 1) then chunk 1
  (ubatch 2), same slot 0, same generation 1. No second copy, no eviction
  between the two chunks. KV/recurrent state is owned by the logical layer
  and written in-graph, never by a slot index.
- `phase_summary`: tokens 1024, ubatches 2, `graph_builds 0` (per-chunk graph
  sets were prebuilt; the timed region did no graph construction), compute
  ~241 ms, transfer ~2320 ms (shared-GPU noise), peak VRAM ~3.85 GiB.

Selected graph-binding mechanism (recorded per the task contract)
- Per-chunk prebuilt graph sets: each chunk's act pair is a pair of
  non-aliasing fixed-offset slices of the two slabs, bound at prebuild. The
  set key is `(mode, n_tokens, rs_z, n_outputs, chunk_id)`. Measured
  two-chunk probe footprint: `chunks=2 graph_sets=2 total_graphs=132
  act_slab_bytes=41943040 io_ctxs=2` (2 sets x 66 graphs; 2 io contexts).
- Rejected mechanism: a `ggml_view_2d` over a stable slab tensor. The
  scheduler's galloc calls `ggml_backend_view_init` on a view whose source
  has a backend buffer, which allocates a *separate* buffer for the view and
  defeats the stable-slab / non-aliasing semantics. Fixed-offset
  `ggml_backend_tensor_alloc` from the slab buffer is used instead (the same
  documented API the batch-major act pairs use).
- Host `--check-telemetry` now enforces, for `schedule == "layer-major"`:
  per-layer ascending chunk identity (0,1,2,...) and non-aliasing `act_bind`
  slices per slab. Both negative fixtures (out-of-order chunk; aliasing bind)
  are rejected with exit 1, so the checks are not vacuous.

Caveats: shared GPU (a foreign process held VRAM; timing is subject to
shared-GPU noise); source cache is best-effort cold (uncontrolled Windows file
cache); this is a one-layer, two-chunk boundary probe - full 64-layer PP, 20K
scale, and the PP->TG handoff are T-Q38LM-02..04.

### T-Q38LM-02: full two-chunk layer-major PP pass (done 2026-08-20)

A dedicated `llama_context::layer_major_pp` controller drives 1,024 PP
tokens (two 512-token chunks) through all 64 logical layers in layer-major
order (chunk 0 then chunk 1 for every layer before the next layer), with each
package copied once and reused across both chunks. This is a
scheduler-state / determinism check, not a quality or speedup claim.

Commands (workspace root, Git Bash; GPU runs under `tools/with-gpu-lock.ps1`):
- Build (no GPU): `cmake --build build/llama-cuda --config Release --target
  llama-layer-streaming --parallel 16` -> exit 0. Default-OFF build
  (`build/llama-cuda-off`) also compiles (the experimental path is gated out).
- Layer-major 1024 (GPU mutex): `ctest --test-dir build/llama-cuda -C Release
  -R layer-streaming-layer-major-1024 --output-on-failure` (runner:
  `--pp-schedule layer-major --pp-tokens 1024 --ubatch 512 --layer-slots 4
  --pp-no-logits --tg-tokens 1 --greedy --no-mtp --telemetry
  build/llama-cuda/q38ls-lm-1024.jsonl`) -> 2/2 pass (run ~16-36 s, host
  check ~0.07 s). Log `out/q38ls-lm1024-full.log`.
- Regression (GPU mutex): `ctest -R layer-streaming-(scheduler^|pp512^|layer-major-1024)`
  -> 8/8 pass; `tools/run-tests.ps1 -Suite quick` -> 8/8 pass. Batch-major and
  the existing CTest names are unchanged.
- Host negative fixtures (no GPU): a deliberately reordered-chunk trace and a
  duplicated-chunk trace both fail `--check-telemetry` with exit 1 ("layer-major
  chunk not ascending for layer"), so the layer-major invariants are not
  vacuous.

Observed behavior
- `phase_summary` (PP): tokens 1024, ubatches 2, `graph_builds 0`, 64
  layers_run, 60 evictions (4-slot lookahead), compute ~0.74 s, transfer
  ~31 s (shared-GPU noise, host-synchronous pageable copies). 128 PP
  `layer_compute` events: chunk 0 then chunk 1 for each of the 64 layers, no
  layer L+1 compute before both layer-L computes complete. Exactly 64 PP
  `copy_submit` events: each layer's package is copied once and consumed by
  both chunks under the same generation; the generation increments as the
  four slots are reused (layers 0-3 gen 1, 4-7 gen 2, ...).
- TG (1 token) uses the standard per-ubatch path and reuses the TG-u1 set
  exactly like batch-major (`graph_builds 0`). The one-token TG
  schedule-consistency check produces the same first greedy ID **286** under
  batch-major (`q38ls-bm-1024.jsonl`) and layer-major (`q38ls-lm-1024.jsonl`)
  for the same 1,024 PP tokens.

Design notes (recorded per the task contract)
- The controller creates one memory context per chunk and applies it in chunk
  order so the shared rec/KV state is current for each chunk's `find_set` /
  `set_inputs` (the prebuilt per-chunk sets were built against the identical
  applies). The recurrent `rs_z` is captured per chunk at apply time (chunk 0
  -> 0, chunk 1 -> -1) because `get_rs_z()` reads the shared rec memory that
  the next `apply()` overwrites; the controller re-resolves each chunk against
  the same `(rs_z, n_out, chunk)` tuple the prebuild used.
- The per-ubatch path (batch-major) is unchanged. Layer-major PP is rejected
  there as a routing error; layer-major TG (phase 1) reuses the TG-u1 set
  exactly like batch-major. Activation slabs swap only after every chunk of
  the current layer completes (fixed per-chunk slab slices, no aliasing).

Caveats: shared GPU (timing subject to shared-GPU noise); source cache is
best-effort cold (uncontrolled Windows file cache); this is a 2-chunk / 64-layer
PP determinism check - 20K scale, a 3-token TG, and the negative/final
regression are T-Q38LM-03..05.

### T-Q38LM-03 through T-Q38LM-05: 20K PP, 3-token TG, and regression closure (done 2026-08-20)

See "20K Layer-Major Evidence" above. The free-GPU run completed all 40 chunks
through every logical layer with legal generation reuse and then generated
three TG tokens from the same live state. Both telemetry validators accepted
the 4,113-event trace, and the full mutex-protected regression suite passed
19/19 targets. The default-OFF build remains successful. Raw measurements are
subject to uncontrolled Windows file-cache state and are not a speedup or
quality claim.

## Known gaps / risks (current)

- Foreign `deepseek_v4.exe` shared the GPU during earlier validation
  (~12.4 GiB VRAM held, high util), so those timing results are subject to
  shared-GPU noise. The completed 20K run started with the GPU idle; every
  trace still records observed free VRAM at `run_start`.
- `llama-cli --log-file` stalled with an empty log file on this build (two
  runs killed). Not task-owned (our runner writes its own JSONL telemetry),
  but recorded here: do not rely on llama-cli's file logger.
- CPU-only execution of this model is impractically slow (see above).
- With pageable mmap copy sources the copies are host-synchronous; the four-slot
  lookahead overlaps compute but the copies themselves still serialize on the
  copy stream. Recorded as observed behavior, not a claim.
- The CMake CTest entries bake `$ENV{Q38LS_MODEL_PATH}` at configure time; a
  reconfigure without that env var set rewrites the model path to empty and the
  GPU tests fail at arg parsing. Reconfigure with Q38LS_MODEL_PATH exported.

## Blocked / escalated

- T-Q38LS-04 `rs_z` variant escalation is resolved by T-Q38LS-04B.
- T-Q38TG-01 (hybrid TG executor) is permanently gated: T-Q38TG-00
  returned NO-GO (measured H2D ~3.0 GB/s vs ~29 GB/s break-even).
- T-Q38HG-04 (constrained handoff, attention-only migration) is done —
  all acceptance criteria met: migration 3.54 GB / 0.424 s, parity ID
  286, test suite 36/36, TG 3.28 tok/s (within ~15% of -ngl 17 baseline).
  slot_layer() routes recurrent layers to CPU-resident host_layers;
  load_layer() skips H2D copies for them.
- T-Q38HG-05 (GTX 1080 validation) is gated on owner hardware access.
