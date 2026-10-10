# Does the VRAM expert path earn its place at token generation?

**A reconciliation of the stock CPU-expert TG baseline against our own row-set
ceiling. Written 2026-09-29, host-only analysis, no new GPU run.**

> **UPDATE 2026-09-29 (later the same day).** The owner supplied a fresh stock
> baseline, `docs/llamacpp 8gb vram base.md` (build 10892), which **reproduces
> and beats** the anchor this document originally leaned on: **40.96 tok/s**
> against the recorded 32.44. It also (a) narrows, but does not close, the
> 7x bracket on CPU expert cost, and (b) exposed a **correction I owe**: the
> hit rate I led with, 0.4518, was the wrong statistic. Sections 6 and 8 are
> new; section 5 supersedes the original section 4's figure. The verdict
> stands and is now better supported, and the decisive next experiment turned
> out to cost one run and no code.

> **UPDATE 2026-09-30. The question in the title is now answered
> unconditionally, and the answer is no.** *(Partly retracted later the same
> day — read the CORRECTION block below this one first.)* Fitting the T-GPLR-19 A/B arms' wall
> time against their own staged bytes gives `ms/token = 28.92 + bytes/10.35
> GB/s` to ±1.9% over 8 matched arms. The **28.92 ms intercept — what the path
> costs with zero bytes on the link — already exceeds the incumbent's entire
> 24.41 ms token.** At a perfect cache the VRAM expert path delivers 34.58
> tok/s against 40.96. Every earlier closure in this document argued that the
> hit rate could not get high enough; the correct statement is that no hit rate
> is high enough, and §15.6 locates most of the loss in staging machinery
> rather than in the link or the GPU. **§15** is that measurement, **§13** is
> what survives (the per-expert split, whose overlapped ceiling is 1.39x; the
> best measured placement reaches 1.37x), and **§14** corrects a mislabelled
> retention figure that three sections had been reasoning from. §15.5 adds the
> other half of the predictor's closure: at three capacities it evaluated
> 81,424 proposals, placed none, and cost up to 2.5% of TG to do so. **§13.4
> carries a second correction of the same kind as §14** — the model was being
> evaluated at the mean of two prompts' hit rates, which overstates a concave
> function; fixing it inverts the placement ranking, putting LRU-K ahead of
> warm-then-pin. **§12.1b carries a third**: `h_stock`, the last unmeasured
> input to the cost model, was guessed at 9 of 40 banks and is actually 7.5,
> readable straight off the anchor run's own fitter log. That one moved every
> multiplier here up by 1.3%, and made the best placement *cheaper in VRAM than
> the baseline it beats*. **§13.6 then closes the capacity question the
> throughput sections had left open**: the anchor's VRAM accounting reconciles
> exactly, so it projects to any context, and at the settled 236,000 only the
> **48-row** cache fits — the same policy §13.4 picked on throughput, by an
> unrelated criterion. It also surfaces a dependency nobody had stated: that fit
> needs a bare card, because the anchor's real budget was the fitter's own
> **7,055 MiB**, not the 8,462 the `--fit-target` flag implies — a figure §10.1
> had right and four later paragraphs had drifted off, and correcting it
> *strengthens* §15.5's closure (nothing in the kprobe family fits, not even
> K=4). **§12.1a carries a seventh**, found by the same method: it had
> *validated* the model's constant host cost and declared the real host "never
> more expensive" than charged — false at the stock anchor, where the host runs
> 6.5 experts and the measured curve charges 2.50 us/layer more. The conclusion
> survives (the model is conservative, ceiling 1.388x against the curve's
> 1.399x, ranking unchanged) but the reason it gave was the opposite of the true
> one. **§12.1c carries an eighth, and it is the first that is also an
> optimization.** Both of those curves are **8-thread** curves; the anchor they
> are solved against ran at **16** — a unit mismatch no amount of re-reading an
> 8-thread sweep could find. The cross-product of threads x expert count had
> never been measured, and in the 2–3 expert regime the split actually runs in,
> the optimum is **12 threads**, where 8 is 15–17% slower. §12.1's "saturates at
> 8 threads, and past 8 it gets worse" is true only at k=8, where it measured.
> Correcting the anchor's thread count takes `other` to **14.083 ms**; running
> the host leg at 12 takes `h*` from 0.6825 to **0.6662** — buying residency
> with threads rather than VRAM, which after §13.6 is the trade that is actually
> available — and **flips §13.4's top two**, though by less than the measurement
> spread, so the right reading is that 48 and 64 rows are tied on throughput and
> capacity breaks the tie. It also yields **+1.36% on stock with no split, no
> cache and no new code** — one flag. **§12.2a is the same mistake on the
> prefill side, and it is the largest free win in this document.** §12.2
> compared 8 threads against 16, found 16 better, and stopped; the box is
> 16c/32t, prefill is compute-bound, and §12.3 shows prefill is **79.4% host
> expert FFN**. Sweeping the rest, over three invocations that agree to 1%,
> **16 is a local pessimum**: 12 beats it by 13.7% and 26/28/30 form a plateau
> **20% below it** (32 is worse again, so "use every thread" is also wrong).
> Carried into §12.3's own accounting that is **prefill 152.13 s → 127.49 s,
> 347 → 414 tok/s, 1.19x, for one flag**. The recommendation becomes
> **`--threads 12 --threads-batch 28`**; §12.2's `8 / 16` was the wrong pair in
> both phases. The same sweep **closes the ubatch question**, which §11.7 had
> only ever priced downward: ubatch 1024 does reach the same compute floor, but
> `--threads-batch 28` gets there at ubatch 512 and **the two do not stack** —
> so the compute-buffer envelope the §11.7 family spends five subsections
> fighting over was defending a gain available free from a flag. All of it cost
> zero GPU time.

> **CORRECTION 2026-09-30 (later the same day): findings 9-18 retract the
> following claims in the UPDATE above.** Each section carries its evidence. All of it
> is host-only re-derivation from recorded logs, traces and source, with no
> new GPU run.
>
> **Retracted from the UPDATE:**
>
> 1. **"Answered unconditionally… no hit rate is high enough", the 28.92 ms
>    intercept and 34.58 tok/s at a perfect cache** — eighteenth finding, §15.
>    - **Why the fit fails.** It pooled cache-off and cache-on arms, which run
>      different execution shapes, so its intercept is biased up.
>    - **The path's own fits.** The cache-on intercept is 26.47 ms (prompt A,
>      ctx 768) and 23.69 ms (kprobe, ctx 1280). That is 0.92-1.03x at a
>      perfect cache (arithmetic on measured fits), or ≈0.88-1.01x after the
>      expected 0.6-1.0 ms context term.
>    - **The answer is still no, for a different reason.** The 7,055 MiB
>      budget admits K ≤ 2 at short context (0.39x, measured) and nothing at
>      236k (expected). Break-even needs `h` ≥ 0.984 (kprobe only), and `h` = 1
>      needs K ≥ 32. VRAM and `h` bind, not the fixed cost.
>    - **"Most of the loss is staging machinery" is withdrawn.** `M` is 19-35%
>      of a stock token. At the operating point the link carries 58% of the
>      token (§15.6).
> 2. **"The largest free win in this document": `--threads-batch 28`, 1.19x,
>    prefill 152.13 s → 127.49 s** — ninth finding, §12.5.
>    - From 32 tokens per call, stock copies every host expert bank to the GPU
>      and computes it there. The copy is 86-100% of a steady ubatch.
>    - The 1.19x prices `--no-op-offload` only. `--threads-batch` is expected
>      to move stock prefill by nothing measurable.
>    - §12.3's 79.4% host-FFN share falls with it.
> 3. **"Closes the ubatch question"** — ninth finding, §12.5. It is inverted:
>    the PP levers are ubatch and pinned memory.
>    - From u1024 the copy beats the CPU cell, and its lead grows with the
>      ubatch.
>    - Stock at `-ub 2048` has not been run.
> 4. **"+1.36% on stock with no split, no cache and no new code — one flag"**
>    — twelfth finding, §12.1c correction.
>    - It priced stock's host leg at an interpolated 6.5 experts per layer.
>      Stock's host layers run all 8.
>    - At k=8 it is +0.92% at face value, and unproven. Four invocations favour
>      fewer threads; one favours 16.
>    - The same error moves `other` to ~13.94 ms, not 14.083 (−1.0%, expected).
>      That is inside the cross-invocation spread.
> 5. **The recommendation `--threads 12 --threads-batch 28`** — items 2 and 4.
>    No thread flag is recommended until a TG thread A/B runs. §12.2a's sweep
>    is real, but it measures a path stock does not run.
>
> **The other findings, each corrected in place:**
>
> - **Tenth (§12.5).** The fitter and the loader count blocks differently, so
>   the anchor's block 0 ran wholly on the CPU against the fitter's intent:
>   32.79 host bank-equivalents.
> - **Eleventh (§5, §12.5).** The 32.44-vs-40.96 comparison differs in four ways
>   (context, binary, placement, draft) and cannot price MTP. Priced at the
>   anchor's placement with δ = 0, drafts of 3-4 give 1.42-1.49x TG — expected,
>   not measured.
> - **Thirteenth (§9.1, §9.4).** Every run without `--residency-tg` reports its
>   load-time usage as `peak_vram_bytes`. Every retention peak recorded before
>   nested `8e9757169` excludes the row cache.
> - **Fourteenth to sixteenth (§9.5).**
>   - The byte account charges 4 slots at 1,992.0 MiB against 1,860.0 MiB
>     allocated.
>   - That over-charge is what refused the 4-slot K=2 arm. Under the budget's
>     own definition it fits by 16.3 MiB (expected).
>   - The free-VRAM fit subtracts the already-allocated slot window a second
>     time.
> - **Seventeenth (§9.6).** T-GPLR-62 S2's compact slot frees memory only after
>   prefill, so its peak is a full-bank peak. Its preflight prices the whole run
>   at the narrow TG size.
>
> **What stands from the UPDATE:**
>
> - §15.5's predictor result: 81,424 proposals, none placed, up to 2.5% of TG.
> - §13.6's 7,055 MiB realised budget.
> - The §12.1b `h_stock` measurement.
>
> Findings 9-18 do not re-audit §13's split or §14.

> **UPDATE 2026-10-01. Findings 22-30: the owner's two real configurations,
> measured head to head.** This is the first time the two recorded server
> configurations in `docs/llamacpp 8gb vram base*.md` have been run against
> layer-major in the same mutex hold, with VRAM matched rather than assumed.
> Full numbers, per-arm layer counts and KV placement in **§12.6**. Every
> LM/stock ratio is same-session; cross-session LM variance reaches 16%. VRAM is
> the WDDM per-process dedicated peak, never the `peak VRAM:` stderr line,
> which is a pre-run baseline identical in every arm.
>
> - **Twenty-second.** LM's +22-23% compute over same-engine stock is the
>   **causal-mode flash-attention dispatch**, not padding, mask upload or host
>   gaps: a causal op with `mask == nullptr` fails the `use_gqa_opt` /
>   `gqa_opt_applies` gates (`ggml-cuda/fattn.cu:58,386`) and takes
>   `<256,256,64,1>`. The **V1** side-copy fix restores the gate behind
>   `FAFIX_CAUSAL_GQA=1` and is **VRAM-neutral** — ON and OFF peak at 10.056 GiB
>   either way at ctx 262,144, and likewise at ctx 128,000/f16 and
>   512,000/q8. At a 50k prompt its contribution is **context-independent**:
>   **1.186x / 1.182x / 1.182x** at ctx 262,144 / 128,000-f16 / 512,000, a 0.4%
>   spread across a 4x context range. **Measured** (ab9). V1 is **not** adopted
>   into the vendored tree; that needs a Planner contract and a QA gate.
> - **Twenty-third.** **V1's own contribution GROWS with prompt length**:
>   1.186x at 50k, **1.342x at 144,001**, **1.377x at 200,000**. The longer the
>   attention window, the more the un-optimised K/V re-read costs. **Measured**
>   (ab6, ab7).
> - **Twenty-fourth.** **The win is PP-only.** LM TG is **0.640x / 0.687x /
>   0.741x** stock while spending **1.32x / 1.65x / 1.90x** stock's VRAM, and at
>   **both** owner contexts LM cannot generate sustained tokens at all — the
>   handoff preflight is short by **1.93 GiB** at ctx 262,144 and **5.34 GiB** at
>   ctx 512,000, and refuses loudly. **Measured** (ab12, ab11cap, ab5tg).
>   Charter reading: for prefill-then-generate at these contexts, stock wins.
> - **Twenty-fifth.** **The LM/stock multiplier DECAYS with prompt length**:
>   **1.943x -> 1.651x -> 1.533x** at 50,000 / 144,001 / 200,000 tokens, about
>   **11% of the advantage lost per doubling**. LM keeps 0.582x of its 50k
>   throughput at 200k; stock keeps 0.737x. The mechanism — stock copies only
>   routed experts, so it improves as the route union saturates — is
>   **UNPROVEN**: no route telemetry was collected at 144k/200k. LM also pays
>   ~8 KiB of VRAM per prompt token (measured slope 7.95 / 8.09 KiB/token;
>   **not attributed** — the act slab is sized by ubatch and is the same 409,600,000 B
>   in all three runs, so it cannot explain the slope). **Measured** (ab9, ab10, ab6, ab7).
> - **Twenty-sixth.** **The largest free owner lever found is a stock flag.**
>   At doc2, `--fit-target 16000 -> 8000` gives **PP 425.9 -> 2,228.1 (5.23x)**
>   and **TG 12.13 -> 42.47 (3.50x)**, residency 26/42 -> 41/42 layers with the
>   full 5,312.50 MiB KV moved onto the GPU. No code. `--ubatch-size` is the
>   opposite trap on the same config: 512 -> 8192 buys 4.58x PP by collapsing
>   residency to **2/42** layers and pushing the **entire** KV cache to the CPU,
>   at 0.44x TG. **Measured** (ab5).
> - **Twenty-seventh.** **The anchor prompt files are not named by token
>   count.** `anchor-100000.txt` is **144,001** tokens; the other two are exact
>   (50,000 and 200,000), and byte density is a consistent 4.634-4.638 B/token.
>   `--pp-tokens` does **not** truncate — it selects the file, and the tool
>   reports the file's own length. A planned context checked against the file
>   *name* fails on both engines identically (`--ctx-size 131072 is smaller
>   than pp+tg (144004 tokens)` / HTTP 400). **Measured** (ab6 attempt 1).
> - **Twenty-eighth.** **A real defect in the vendored tool, recorded not
>   fixed.** `tools/layer-streaming/main.cpp:3454` puts
>   `handoff_mode_str = "on";` **inside a `//` comment**, so `--handoff-partial N`
>   parses `N` but leaves handoff at `auto`: the comment's own "Implies an
>   explicit handoff-on request" is unreachable, and the `--help` text
>   (`main.cpp:67-68`) never warns that the flag alone is inert.
>   The flag is therefore ineffective without also passing `--handoff`. One-token
>   fix; **not made** — the vendored tree is another workstream's and adoption
>   needs a Planner task contract. **Code, read 2026-10-01.**
> - **Twenty-ninth.** **"254 of 256 experts" is a 512-token figure**, not a
>   property of a real prompt. At the 50k anchor the route union is
>   **0.717 / 0.815 / 0.855** (72% / 82% / 86%) at ubatch 512 / 2048 / 4096.
>   §15.3's conclusion stands and strengthens; the figure does not travel.
>   **Measured** (`build/stock-levers/route-union-report.txt`).
> - **Thirtieth.** **The `byte_account.kv` under-report suspicion is WITHDRAWN.**
>   Raised during this session from a WDDM VRAM slope of 20.65 KiB per context
>   token against a reported 10.63. It is wrong: the preflight reports
>   **`KV (q8 @ 262144) : 2852126720 B`**, which is exactly **2,720.00 MiB**,
>   byte-identical to stock's independently reported `CUDA0 KV buffer size =
>   2720.00 MiB` at the same context (`ab10-s13u4.log`, `ab5-s5f12k.log`). Both
>   engines agree and both are linear in context. The residual unexplained item
>   is narrower and is **not** a KV miscount: LM's *unaccounted* VRAM grows
>   3.115 -> 5.507 GiB from ctx 262,144 to 512,000, about 9.8 KiB per context
>   token — a second context-scaled allocation `byte_account` does not name.
>   **Measured; cause UNPROVEN.**

Sources, and what kind of claim each one is:

| Figure | Source | Kind |
| --- | --- | --- |
| **40.96 tok/s TG (24.41 ms/tok), 2551 tokens at 52,819 ctx, no MTP** | owner, build 10892, `docs/llamacpp 8gb vram base.md` | **measured**, today |
| 347.19 tok/s PP (2.88 ms/tok) over 52,819 tokens | same | **measured**, today |
| 41 layers fitted, **33 overflowing**, 7008 MiB CUDA / 16046 MiB free | same, fit log | **measured** — and `id=0, target=7055 MiB` on the line above fixes the *realised* budget at **7,055 MiB**, not the 8,462 the flag implies (§10.1, §13.6). *The loader realised one block less than the fitter chose: block 0 ran wholly on the CPU, 32.79 host bank-equivalents (tenth finding, §12.5).* |
| **0.010376 MiB of KV per context token** (10 of 40 layers are attention, 30 recurrent) | same, `llama_kv_cache` line | **measured**, and reproduces exactly in the 512k/2-slot run |
| 70 graph splits at bs=1, 3907 graph nodes, graphs reused 2540/2551 | same | **measured** |
| 32.44 tok/s TG (30.83 ms/tok), MTP 0.546 | owner, build 10876, 2026-09-10 | **measured**, superseded as the anchor. *Not an MTP comparison: it differs from the anchor in context, binary, placement and draft (eleventh finding, §5).* |
| ~565 MB of routed-expert rows per token | derived | **derived**: 8 x 1.77 MB x 40 layers = 566,231,040 B |
| 46.6 ms/token compute, 40 layers, 50k ctx | streaming trace | **measured**, whole-bank path |
| 4.4 / 6.5-8.2 / 8.26 GB/s achieved feeds | gplr29 / TG row fetch / streaming trace | **measured**, instrumented whole-bank copies — *not* the end-to-end marginal, see §15.2 |
| ~~**10.35 GB/s marginal feed**~~ | fit of ms/token vs staged bytes, 8 ~~matched~~ A/B arms | ~~**measured, end-to-end**, ±1.9%~~ — **withdrawn 2026-09-30 (eighteenth finding, §15)**: pools cache-off and cache-on arms, which run different execution shapes. Cache-on band **8.8-12.4 GB/s** |
| ~~**28.92 ms/token fixed cost of the streaming path**~~ (everything that is not transfer) | same fit | ~~**measured** — exceeds the whole 24.41 ms incumbent token, see §15.1~~ — **withdrawn 2026-09-30**: biased up by the same pooling. The path's own intercepts are in the next row |
| **26.47 ± 0.31 ms + bytes / 8.83 GB/s** (prompt A, ctx 768, 8 legs, ≤ 1.2%); **23.69 ± 0.17 ms + bytes / 12.43 GB/s** (kprobe predictor-off, ctx 1280, 3 legs) | cache-on legs only: `build/llama-cuda/gplr19-k{2,4,6,8}-ab/summary.json` (`retain` > 0), `build/llama-cuda/gplr58-kprobe-summary.json` (role `off`) | **measured fits**, 1 SE: 0.92x / 1.03x of stock at `h` = 1 before the expected 0.6-1.0 ms context term (§15 correction) |
| ~~14.6 GB/s~~ | recon `copy_gpu_ms` | **NOT a feed rate - see §3. Withdrawn.** |
| 0.4518 demand-hit share | T-GPLR-59 D-gate | **measured — but see §6, wrong statistic for this claim** |
| ~~0.6487 (K=8) retention hit share~~ | T-GPLR-19 honest ladder | **MISLABELLED 2026-09-30 — it is rung **6**'s `honest_b`, not K=8. K=8 is 0.7283. See §14.** |
| **0.6461 (K=6) / 0.7255 (K=8) / 0.8319 (K=12) retention hit share** | arm-B `n_rows_resident`, re-read 2026-09-30 | **measured, reps identical** |
| **0.583-0.598 retention-only hit share** | T-GPLR-60 G1 | **measured** |
| ~~108.9 us per layer, 8 experts, 1 token~~ | `tools/expert-cpu-bench` 2026-09-29 | **WITHDRAWN 2026-09-30 — cache-contaminated, understated 2.62x. See §2.** |
| **288.6 us per layer, 8 experts, 1 token** | `tools/expert-cpu-bench` 2026-09-30 arm C (`--pool 256 --id-stride 8`) | **measured, clean** — at **8 threads**; the anchor ran at **16** and the split's optimum is **12**, see §12.1c |
| **56.2 us / 2 experts, 90.0 us / 3 experts at 12 threads** (the split's own operating point) | `threads-x-experts`, 1000 iters, 2026-09-30 | **measured** — 15–17% under the 8-thread figure §12.1 called a bandwidth floor |
| **anchor ran `n_threads = 16`** | `docs/llamacpp 8gb vram base.md` lines 24, 309 | **measured** — the thread count `other` must be solved at |
| **28.7 ms/layer expert FFN at ubatch 512, 28–30 threads** (vs 36.6 at the recorded 16) | `pp-threads{,-fine,-top}`, 3 invocations, 2026-09-30 | **measured**, 1% cross-invocation — ~~implies prefill 152.13 s → 127.49 s, **1.19x**, §12.2a~~ **retracted as a stock lever 2026-09-30 (ninth finding, §12.5)**: stock prefill copies host banks to the GPU from 32 tokens per call, so this CPU cell prices `--no-op-offload` only |
| ~35-40 GB/s effective DDR feed | 2026-09-10 entry | **derived**, arithmetic closure |
| Anchor placement and buffers: offloaded 41/42, CUDA0 model 5,192.44 MiB; **KV on CUDA0**, 1,328.12 MiB for 128,000 q8_0 cells; CUDA0 compute 427.53 MiB; 119 / 70 graph splits | `docs/llamacpp 8gb vram base.md` lines 268-270, 292-293, 304-307 | **measured**. The KV being device-resident puts the anchor's extra attention at ≈575 MB of KV read per token at 52,819 tokens, so the context term is **expected** at ≈0.6-1.0 ms (§15) |
| Prefill 152,132.47 ms / 52,819 tokens; eval 62,258.22 ms / 2,551 tokens | same, lines 403-404 | **measured** — the anchor's two timing lines |
| `peak_vram_bytes` = `total − free_min`, refreshed only at the retention fit/allocation and `--residency-tg` activation | `tools/layer-streaming/main.cpp:4478, 4956, 5036, 5742, 6424`; T-GPLR-24 H-B nested `8e9757169` removed the second sample | **code** — non-`--residency-tg` peaks are load-time usage; retention peaks before `8e9757169` exclude the row cache (§9.4) |
| Row cache allocated at bring-up, only under retention, read only by TG `load_row_set` | `main.cpp:5026-5037`; `src/llama-layer-streaming.cpp:8958` (`begin_row_cache`), `7517-7547` (per-layer cost), `9668` (`if (row_set_retain)`) | **code** — cache-off arms allocate no cache; K × 572,719,104 + 655,360 B, 546.2 MiB per K (§15 correction) |
| Retention and non-retention TG run different execution shapes | `src/llama-layer-streaming.cpp:9658-9667` | **code** — cache-on: miss H2D into the per-layer cache, then a device refresh of the slot bundle for the whole union; cache-off: slot-keyed H2D (§15 correction, item 2) |
| gplr19 arms: A = `retain 0`, B = `retain K`; summary peak = last `peak_vram_bytes` in the trace | `tools/run-gplr19-rowcache-ab.ps1:244-249, 358-390` | **code** — the "differ ONLY by the retention flag" design intent does not hold at the execution level |
| Slot buffers sized per rotation; byte account charges `n_slots × moe_largest_bytes` | `src/llama-layer-streaming.cpp:3224-3241`; `main.cpp:573` | **code + trace** — 4 slots allocate 1,860.0 MiB and are charged 1,992.0 MiB (§9.5) |
| S2 compact-slot swap after prefill: compact buffers allocated first, full-bank freed last | `main.cpp:5734`; `src/llama-layer-streaming.cpp:9180, 9463, 9510` | **code** — an S2 run's peak is a full-bank peak (§9.6) |
| **LM 4,079.2 tok/s @ 10.056 GiB vs stock 2,099.6 @ 10.193 GiB, ctx 262,144, 50k, 41/42 layers, KV 2,720.00 MiB on CUDA0 — 1.943x** | `build/stock-levers/ab10.wrapper.log`, `ab10-s13u4.out`, `build/llama-cuda/logs/ab10-lmv4*-a1-u4096.mem.csv` | **measured**, same session (§12.6.1) |
| **LM 4,080.4 @ 14.976 GiB vs stock 2,228.1 @ 15.039 GiB, ctx 512,000 `--parallel 2`, 41/42 layers, KV 5,312.50 MiB on CUDA0 — 1.831x** | `build/stock-levers/ab5.wrapper.log`, `ab5-s5f8k.out` | **measured**, same session (§12.6.1) |
| **LM TG 25.074 / 26.934 / 29.045 tok/s @ 13.434 / 16.809 / 19.341 GiB vs stock 39.204 @ 10.193 GiB — 0.640 / 0.687 / 0.741x** | `build/llama-cuda/logs/ab12-lmh*-a1-u4096.jsonl` (`phase_summary`, `phase: tg`), `ab12-s13.out`, `ab12-s13b.out` | **measured**, same session (§12.6.2) |
| **Handoff refusal: needs 24,295,318,016 B at ctx 262,144 and 27,960,222,208 B at ctx 512,000 against 22,226,665,472 B observed free** | `build/stock-levers/ab11cap-lmaut.tail.txt`, `ab5tg.tail.txt` | **measured** refusals, 1.93 GiB and 5.34 GiB short (§12.6.2) |
| **LM/stock PP 1.943x / 1.651x / 1.533x and V1 ON/OFF 1.186x / 1.342x / 1.377x at 50,000 / 144,001 / 200,000 tokens** | `build/stock-levers/ab{6,7,13}.wrapper.log`, `ab13-s6u4k.out`, `ab7-s7u4k.out`, `ab9.wrapper.log` | **measured**; the 144,001 stock term is ab13's six-clean-request mean (was ab6 request 0 alone, 1,734.05; now 1,735.35, ratio 1.652x -> 1.651x); the decay *mechanism* is **unproven** (§12.6.4) |
| **doc2 `--fit-target 16000 -> 8000`: PP 425.9 -> 2,228.1, TG 12.13 -> 42.47, 26/42 -> 41/42 layers** | `build/stock-levers/ab5.wrapper.log` arms `s5u512` / `s5f8k` | **measured** (§12.6.5) |
| **KV q8 @ 262,144 = 2,852,126,720 B = 2,720.00 MiB, identical on both engines** | `ab5tg.tail.txt` preflight; `ab10-s13u4.log` `llama_kv_cache` | **measured** — withdraws the `byte_account.kv` suspicion (§12.6, thirtieth finding) |
| `handoff_mode_str = "on";` sits inside a `//` comment, so `--handoff-partial N` never enables handoff | `tools/layer-streaming/main.cpp:3454`; help text at `:67-68` promises no handoff | **code, read 2026-10-01** — defect recorded, **not fixed** (§12.6.8) |

## 1. Both paths move the same bytes

    8 experts x 40 layers x 1,769,472 B = 566,231,040 B/token  (~565.6 MB)

Fixed by the router and identical for both paths. The only question is
**whether those bytes are read from DDR or cross PCIe.**

## 2. The CPU path, rebuilt against the new anchor

The 2026-09-29 run is the cleanest baseline in the record: **no MTP at all**,
52,819 tokens of real context, 2,551 generated tokens, 24.41 ms/token.

The 2026-09-10 run that this document originally used was **32.44 tok/s with
MTP draft 5 at 0.546 acceptance**. The fresh run reaches **40.96 tok/s with no
speculative decoding**, a 26% improvement on a 16-builds-newer binary. MTP was
not carrying that baseline; it was likely costing it.

> **RETRACTED 2026-09-30 (eleventh finding, §12.5).** The comparison cannot
> price MTP, because the two runs differ in four ways at once:
>
> - **Context:** a 372-token prompt against 52,819 tokens.
> - **Binary:** build 10876 against 10892.
> - **Placement:** with MTP loaded, the fitter counted the nextn block and
>   reserved an f16 draft context. It put **blocks 0..7 wholly on the CPU**,
>   against the anchor's block 0.
> - **Draft:** n_max 5, and every round drafted all five.
>
> The MTP log does measure the draft cost (2.153 ms per drafted token) and the
> acceptance at each draft position (0.816, 0.661, 0.525, 0.415, 0.313).
> Priced at the anchor's placement with δ = 0, drafts of 3–4 tokens would give
> **1.42–1.49x** TG. That is expected, not measured, and it breaks even at
> δ = 6.2–8.0 ms per extra verify token. Nothing measures δ yet. The
> "costing it" reading and the 26% attribution are both withdrawn; the GPU A/B
> in §12.5 decides.

Against the new anchor, the expert FFN bracket becomes:

| Method | Per layer, 8 experts | 40 layers/token | Share of 24.41 ms |
| --- | --- | --- | --- |
| ~~`expert-cpu-bench` (2026-09-29)~~ | ~~108.9 us~~ | ~~4.35 ms~~ | ~~18%~~ |
| **`expert-cpu-bench` (2026-09-30, clean)** | **288.6 us** | **11.55 ms** | **47%** |
| `-ncmoe 20` whole-model differential | ~430 us | ~17.2 ms | 70% |

**CORRECTED 2026-09-30, and the correction changes which end of the bracket
survives.** The 2026-09-29 microbenchmark advanced expert ids by 1 per
iteration, so k-1 of every k slabs were still L2/L3-warm and the marginal cost
it reported was a cache cost, not a DRAM cost. Its own `gbps` column rises
41.9 -> 128.6 GB/s as k goes 1 -> 8, which cold DRAM cannot do. Re-measured with
iteration-disjoint expert sets over a full 256-expert pool, the figure is
**288.6 us**, understated **2.62x**. Root cause, the fix (`--id-stride`) and the
full clean table are in `docs/plans/qwen36-cpu-expert-placement-draft.md` item 3.

**The elimination argument still runs, but it now lands in the middle rather
than at the bottom.** If the high end were true - 17.2 ms of expert work - then
everything else in a 24.41 ms token would have to fit in 7.2 ms, at 52.8k
context, across **70 graph splits per token**, plus attention, router, shared
expert, sampling and the MTP-free verify path. That is still implausibly tight,
so the differential's face value is still excluded. The clean 11.55 ms leaves
12.86 ms for all of that, which is comfortable — and it is the *right* kind of
number for a run whose experts are mostly on the host.

**What the correction actually bought: the bracket closed.** The draft flagged
the differential as "~4x the microbenchmark and unexplained". Against the clean
figure the differential is **430 / 288.6 = 1.49x** — exactly the
scheduler-overhead-and-no-overlap gap the draft already attributed it to, and
exactly what you would expect once `-ncmoe 20`'s partial GPU residency is
accounted for. **The whole-model differential was never wrong; the
microbenchmark was.** Two independent measurements that disagreed by 4x now
agree to within the overhead that separates a microbenchmark from a whole model.
That reconciliation is the reason to trust 288.6 and not merely to prefer it.

## 3. The link rate: a correction the record already made, and I broke

I previously tabulated "14.6 GB/s" as the optimistic end of the same quantity as
the other rows. **It is not the same quantity**, and the record says so in as
many words (`AGENT_ENVIRONMENT.md:1213-1218`):

> "The DMA is ~1.0-1.1% of the host-observed copy cost - the other ~99% is
> queueing behind compute. **Copy-queue-vs-DMA confusion is over: cite
> `copy_gpu_ms` for wire time, `transfer_us` for host cost, and never mix them
> in one statistic.**"

`copy_gpu_ms` is the CUDA-event clock **on the stream that carries the DMA** -
the bus's own capability, with the host's submit cost excluded. I put it beside
end-to-end feeds and read it as a faster feed. That is precisely the confusion
the record names. **14.6 GB/s is withdrawn from every ceiling argument here.**

The owner's own read - that 14.6 GB/s has never actually been seen working - is
correct, and the reason is that **no path in this repo achieves it end to end.**
The achievable figures, all host-observed, are:

| Achieved feed | Measurement | What it covers |
| --- | --- | --- |
| **4.4 GB/s** | gplr29, warm pageable, bank-sized | the same copy with a cold queue |
| 6.5-8.2 GB/s | `rowset_fetch` `transfer_us`, 2026-09-13 | TG row unions, pageable, submit-bound (94-95% of the term) |
| **8.26 GB/s** | streaming trace, per-bank feed | whole-bank H2D, `host_submit` 85.8% |
| 13.7 GB/s | T-Q38TG-00, **previous machine**, Gen4 x8, `--no-mmap`, one copy with the GPU idle and nothing queued | a clean submit->complete measurement, but on hardware this box is not |

**8.26 GB/s is the best *instrumented whole-bank* feed on this box**, and 4.4
GB/s is what the same copy costs when the queue is cold.

**SUPERSEDED 2026-09-30 as "the best feed ever measured end to end".** It is not.
Differencing wall time against staged bytes across the A/B arms — which is an
end-to-end measurement by construction, since it uses only throughput and a byte
count — gives a marginal feed of **8.9 to 11.1 GB/s, 10.35 GB/s on the matched
pair** (§15). The instrumented `transfer_us` timers undercount because they
charge submit overhead that the surrounding token partly absorbs. Use 8.26 GB/s
for what a single bank-sized copy costs when you are timing that copy; use
10.35 GB/s for what an extra megabyte of staging costs a real token.

## 3a. The VRAM path at the rates we actually achieve

**Scope correction 2026-09-30: this table is the ceiling at ZERO retention, not
a ceiling on the path.** Every row below feeds the full 565.6 MB/token, i.e. it
assumes every demanded row is fetched. That is the whole-bank/cold-cache case.
Read as written it was contradicted by our own runs, which reach **19.7-24.6
tok/s** with the row cache on (`gplr19-k4/k6/k8-ab`) — so the sentence "the
unassisted row-set path does not exceed 14.6 tok/s on this hardware" is
**WITHDRAWN as a bound on the path**, and retained only as what it actually
measures: the h=0 floor. Even that floor is optimistic against the arms, which
measure 11.87 tok/s at h=0 — the table's 46.6 ms compute binding is too small.

| Achieved feed | Copy feed for 565.6 MB (h=0) | Ceiling with 46.6 ms compute binding |
| --- | --- | --- |
| 4.4 GB/s | 128.6 ms | **7.8 tok/s** |
| 6.5 GB/s | 87.0 ms | **11.5 tok/s** |
| 8.26 GB/s (instrumented whole-bank) | 68.5 ms | **14.6 tok/s** |
| **10.35 GB/s (marginal, §15.2)** | **54.7 ms** | **11.96 tok/s** at the fitted 28.92 ms binding |

The last row is the only one that reproduces a measurement: the h=0 arms
measure **11.87 tok/s** (four reps, `gplr19-k6-ab` and `gplr19-k8-ab`). The
first three rows are wrong in both terms and the errors partly cancelled — the
feed was understated (8.26 against a marginal 10.35 GB/s) and the non-transfer
binding was overstated (46.6 ms against a fitted 28.92 ms).
*(Corrected 2026-09-30, eighteenth finding: the reproduction is circular. The
28.92 / 10.35 line was fitted to these same h=0 arms. The h=0 arms are the
cache-off shape, which is cheaper at full fetch: they average 84.07 ms over all
eight prompt-A reps, where the path's own cache-on line, `26.47 + bytes / 8.83
GB/s`, gives 91.33 ms at 572.7 MB. That is a 7.26 ms gap between the two
shapes. See §15, item 2 of the correction.)*

**REPLACED 2026-09-30: this paragraph no longer needs a model at all.** It
previously read "at h = 0.6487 only 35.13% of the rows cross the link, 0.3513 x
68.5 ms = 24.1 ms of copy, landing at ~27.1 tok/s against 24.06 measured,
agreement to within 13%", and called that the strongest validation in the
document. Two things were wrong with it: `0.6487` is not the retention of the
arm it was compared against (§14), and the whole exercise was unnecessary,
because the A/B runs measure the line directly. Regressing each run's ms/token
on its own staged bytes per token — 8 arms, 3 byte levels, identical
`route_digest`, identical `peak_vram_bytes` — fits to **±1.9%** with no fitted
physical parameter, and yields the two constants of the path (§15):

    ms_per_token = 28.92 + staged_bytes / 10.35 GB/s

**The conclusion is unchanged and the reason is now much stronger.** The path's
*non-transfer* floor, 28.92 ms/token, already exceeds the incumbent's **entire**
token of 24.41 ms. Break-even against host compute at `h > 0.832` was never the
binding constraint: even at `h = 1`, with a perfect cache and zero bytes
crossing the link, this path delivers **34.58 tok/s against 40.96**. See §15.

> **Corrected 2026-09-30 (eighteenth finding, §15).** The conclusion stands; the
> reason given for it does not.
>
> - **The fit's premises are wrong.** "Identical `peak_vram_bytes`" is pre-H-B
>   telemetry that never saw the row cache (§9.4). The 8 arms run two
>   execution shapes, so pooling them biases the intercept up.
> - **The path's own line is `26.47 + bytes / 8.83 GB/s`** (cache-on legs only,
>   ≤ 1.2%). At `h` = 1 that is 37.79 tok/s, 0.92x (arithmetic on a measured
>   fit). The kprobe family gives 1.03x. Both are about parity, not 0.84x.
> - **What loses is the operating point.** The 7,055 MiB budget admits K ≤ 2 at
>   short context: 63.21 ms, 0.39x, measured. Break-even needs `h` ≥ 0.984 and
>   `h` = 1 needs K ≥ 32 (§15.7).

**CORRECTED 2026-09-30: the "18-point gap" does not exist.** This paragraph
previously read "best measured retention is 0.6487, and going K=6 -> K=8 moved
it by 0.0026. An 18-point gap that depth tuning does not close." Re-reading the
arms' own `n_rows_resident` telemetry gives K=6 = 0.6461, K=8 = **0.7255**,
K=12 = **0.8319** steady state. The K=6 -> K=8 step is **0.0794**, thirty times
the stated 0.0026, and **K=12 sits at 0.8319 against a 0.832 break-even — at
it, not 18 points below it.** Depth tuning closes the gap completely. See §14.

**What actually closes this path is that the split strictly dominates it.** On
a miss, streaming pays `transfer + device compute` = 214.2 + 14.8 = 229.0 us
per expert; the split pays host compute = 35.7 us. That is 6.4x in the split's
favour **at every hit rate and every VRAM budget**, with no break-even to argue
about. The 0.832 threshold was never the load-bearing argument and should not
be quoted as one. §7b had this right; §3a and §7a did not.

The earlier claim of "a factor of 2.8" compared the wrong pair of numbers
(h=0 ceiling against the stock anchor); the honest figure is 1.70 against a real
measurement. Neither the ~21.5 tok/s figure from the earliest version of this
document nor the 2.8x factor should be quoted.

## 4. The structural argument, unchanged

**The VRAM expert path is a token-generation instrument; it cannot help
prefill.** In layer-major PP the whole bank streams regardless of any cache,
and that is measured, not assumed: T-GPLR-26 recorded "no depth changes
prefill". The charter's first priority is PP speed. An instrument that cannot
move priority #1, and whose TG ceiling is half the incumbent's, does not earn
its place. This holds even if every noisy number above is wrong in our favour.

## 5. What the fitter actually did, and why it matters

The fit log is the most interesting new evidence in the document:

    "cannot meet free memory target of 16000 MiB, need to reduce device memory by 14162 MiB"
    ...
    "- CUDA0: 41 layers (33 overflowing), 7008 MiB used, 16046 MiB free"
    "CPU_Mapped model buffer = 19515.72 MiB, CUDA0 model buffer = 5192.44 MiB"

`--fit-target` is documented upstream as the **free-margin target per device**
(`common/arg.cpp:2852`). The owner asked for 16,000 MiB of headroom and the
fitter honoured it exactly - 16,046 MiB free. So the 7008 MiB is not waste and
the 16 GB is not an oversight; it is a deliberate reserve. The filename suggests
an 8 GB-class scenario was the intent.

**But that is exactly the lever.** 16 GB of VRAM is sitting idle while the CPU
does expert work, and `--fit-target` is a single flag that hands more of it to
the fitter. The question "should expert weights live in VRAM" is therefore
**directly answerable with stock llama.cpp, no code, one run** - see §8.
*(Superseded 2026-09-29, §10.1: the 16,000 MiB margin is not idle. It is the
reserve that makes this run an 8 GB-class configuration, which is the session's
target condition. Spending it leaves that condition, so it is not a lever here.
§8 is superseded for the same reason.)*

## 6. Correction: I led with the wrong hit rate

I previously wrote that the row cache "hits 0.4518, so even a perfect cache
leaves 21.3 ms/token of link". **0.4518 is the wrong statistic for that claim.**
It is the T-GPLR-59 D-gate's *demand-hit share*, measured in a deliberately
capacity-constrained configuration (8.66 GB peak, zero speculative rows
admitted, the run that failed its gate). The *retention* hit share - the number
that describes what a resident cache actually delivers - is measured elsewhere
and is higher:

- **0.6461 at K=6, 0.7255 at K=8, 0.8319 at K=12** (arm-B `n_rows_resident`,
  the statistic the SDD's gate 9 rule actually defines; the 0.6487 this line
  previously carried for K=8 is rung **6**'s `honest_b` — see §14)
- **0.583-0.598** retention-only at K+1 (T-GPLR-60 G1)

At 0.65, only **35% of 565.6 MB = 198 MB/token** crosses the link, which at the
measured marginal feed (**10.35 GB/s**, §15.2) is **19.1 ms/token** of transfer
rather than 54.6 ms.
*(Corrected twice on 2026-09-30: this line first divided by the withdrawn
14.6 GB/s wire-time figure and reported 13.6 ms, then by 8.26 GB/s for 24.0 ms.
The fitted marginal feed is 10.35 GB/s — see §15.2.)*

**The conclusion drawn from it is withdrawn.** This paragraph used to end "that
materially strengthens the cache side of the ledger, and it is the reason §8 is
worth running rather than skipping". It does not strengthen the ledger, because
transfer was never what decided it: the path's **non-transfer** cost is
28.92 ms/token against a 24.41 ms incumbent token, so it loses at every hit rate
including `h = 1` (§15.1). A cheaper transfer term shrinks a term that was not
binding. *(Corrected 2026-09-30, eighteenth finding: the path's own
non-transfer cost is 23.69-26.47 ms, about parity at `h` = 1. Transfer **is** the
binding term at the operating point: it is 58% of the K=2 token (§15.6
correction). The withdrawal still holds for a different reason. Even at the
fastest measured cache-on feed, 12.43 GB/s, prompt A's K=2 token would be
23.69 + 26.1 ≈ 49.8 ms = 0.49x (arithmetic). The VRAM budget, which caps K and
so `h`, decides it.)*

What does *not* change: the four failed attempts were not about the retention
rate. T-GPLR-58 placed **zero** speculative rows (fill room 0.00% at K=2),
T-GPLR-59 D1 admitted **zero** rows at runtime, and T-GPLR-60 lost to
equal-row retention. Those were about *prediction*, not about what a
retention cache delivers. The correction moves the number; it does not rescue
the predictor family.

## 7. Verdict

**REVISED 2026-09-30.** The verdict below split into two once the CPU constant
was corrected, because the correction separates two mechanisms this document had
been treating as one.

**7a. Still do not build the VRAM expert path that WORKS BY TRANSFER**, now on
three independent grounds, the last of which admits no counterargument:

- it is a TG-only instrument and cannot move the charter's first priority
  (prefill). **Measured, not argued**: across the matched A/B arms, turning the
  cache on moves prefill by 0.46% (K=6) and 1.03% (K=8) on identical routes
  (§15.3);
- on a miss it pays `transfer + device compute` = 214.2 + 14.8 = 229.0 us per
  expert where a per-expert split pays host compute = 35.7 us: **6.4x in the
  split's favour at every hit rate and every VRAM budget**; and
- **its non-transfer floor alone exceeds the incumbent's whole token.** Fitting
  wall time against staged bytes over 8 matched arms gives `ms = 28.92 +
  bytes/10.35 GB/s` to ±1.9%. At `h = 1` — perfect cache, zero bytes on the
  link — that is **34.58 tok/s against 40.96 stock, 0.84x** (§15.1). Measured
  at K=8 it reaches 24.34-24.62 tok/s, a factor of 1.68.

  **This third ground is the one that ends the discussion.** The first two say
  the path is worse; this one says no hit rate can make it better. Nothing in
  the retention/prediction family needs evaluating against `h` again.
  *(Corrected 2026-09-30, eighteenth finding, §15: the third ground is withdrawn
  as stated.*
  - *The 28.92 ms floor pools two execution shapes. The cache-on line puts
    `h` = 1 at 0.92-1.03x, about parity.*
  - *The replacement ground is the operating point. The 7,055 MiB budget admits
    K ≤ 2 (0.39x, measured).*
  - *Break-even needs `h` ≥ 0.984, which needs K ≥ 32, ≈18.3 GB of cache.*
  - *The conclusion, no further retention/prediction work, stands. The feed
    figure in the second bullet is 8.26 GB/s. At the cache-on band 8.8-12.4 GB/s
    the miss costs 157-216 us, 4.4-6.0x the split's, so the direction holds.)*

  **CORRECTED 2026-09-30.** This bullet previously argued the closure from
  "break-even requires h > 0.832; the best measured is 0.6487, and the K=6 ->
  K=8 step moved it by 0.0026. The gap is ~18 points and depth does not close
  it." All three numbers were wrong: measured K=8 is **0.7255**, the K=6 -> K=8
  step is **0.0794**, K=12 reaches **0.8319** — at break-even, not 18 points
  below it — and the 0.0026 was rung 6's `honest_b` differenced against rung 6's
  own residency rate, i.e. one run against itself. Depth does close that gap.
  See §14. The closure never needed it.

The previously quoted "unassisted ceiling of 14.6 tok/s" is withdrawn as a bound
(see §3a) — it is the h=0 floor. Using the real measurement makes the case
weaker in magnitude (1.68x, not 2.8x) and much stronger in kind, since it is no
longer an argument from an arithmetic ceiling.

**7b. The per-expert SPLIT is a different mechanism and is NOT closed by any of
this.** `docs/plans/qwen36-cpu-expert-placement-draft.md` (T-GPLR-61) computes
non-resident experts **on the host and never fetches them**, which deletes the
link term that §7a turns on. Under that mechanism a cache hit trades ~36.5 us of
CPU for ~14.8 us of GPU instead of avoiding a 214 us transfer, so it pays at
**every** hit rate above zero and the 0.832 break-even is irrelevant to it.
Estimated prize: **~1.2-1.45x** over stock. Two of its four inputs are still
estimates. Its blocking execution-shape question was **resolved OUTSIDE the SDD
§4 prohibition on 2026-09-30** and is recorded in that draft's item 5.

**What this still does not say.** It does not say a *cache* can never help. §6
corrected the number in the cache's favour, and §5 found 16 GB of idle VRAM
plus a one-flag way to use it. Neither of those is settled by a ceiling
argument. They are settled by the run in §8 — which is now also the **prerequisite
for 7b**, because §8 moves stock's own static residency share, and a split
mechanism must be credited only against a stock baseline that has already been
tuned with free flags.
*(Superseded 2026-09-30, §10.1 and §12.5: the baseline is the 8 GB-class anchor,
so there is no sweep to run first. The free flags left inside that budget are
the prefill ubatch and pinned memory (§12.5) and the TG thread count (§12.1c).
None of them has had a GPU A/B yet.)*

## 8. The decisive experiment, which costs one run and no code

> **SUPERSEDED 2026-09-30 (§10.1, §12.5). Do not run this sweep as a verdict
> on the VRAM path.**
>
> - **It leaves the target condition.** The anchor's 16,000 MiB margin is what
>   makes it an 8 GB-class configuration (§10.1). Arms B-D spend that reserve,
>   so each of them measures a larger card.
> - **Its predicted outcome rests on a wrong model of stock TG.** At batch 1
>   stock computes host-resident experts on the CPU. The CUDA backend's op
>   offload claims a host-weight op only from 32 tokens up
>   (`ggml/src/ggml-cuda/ggml-cuda.cu:5474`, threshold at 5645; ninth
>   finding, §12.5). A bank moved into VRAM therefore swaps host compute,
>   `c` = 37.1 us per expert at k = 8 and the anchor's 16 threads (§12.1), for
>   device compute `d`, with no link term at all.
>   - "More experts in VRAM = slower" would need `d > c`.
>   - `d` is unmeasured. §15.6 bounds it only from above, at 33-48 us
>     depending on the intercept and on ΔA.
>   - So the sweep cannot prove or refute the reconciliation.
> - **What it would measure instead.** At a fixed context, the TG slope
>   against the number of VRAM-resident banks is `8 × (c − d)` per bank per
>   token. That pins `d`, the input §13's ranking and §15.6's split both hinge
>   on. So it is a calibration run, not a verdict. The fitter re-places
>   whole layers, and the KV and compute buffers move with them, so each arm's
>   bank count must be read off its own fitter log, as §12.1b did.

**Sweep `--fit-target` on stock `llama-server`, same model, same prompt, same
context, one mutex hold.** The fitter then places progressively more expert
weight in VRAM, and TG tok/s is a direct read on whether that helps.

| arm | `--fit-target` | expected placement |
| --- | --- | --- |
| A (recorded) | 16000 | 7008 MiB CUDA, 33/41 layers overflow - 40.96 tok/s |
| B | 8000 | more expert weight resident |
| C | 4000 | substantially more |
| D | 2000 | near-maximal expert residency |

- **If TG tok/s falls as the target falls** (more experts in VRAM = slower),
  the reconciliation is proven directly, by the incumbent itself, and the
  VRAM expert path is closed. This is the predicted outcome given §2 and §3.
- **If it rises**, the reconciliation is wrong, the 0.65 hit share is carrying
  real weight, and the cache becomes worth building - with the strongest
  possible evidence behind it, because the mechanism is stock's own.
- **If it is flat**, the placement is not the lever at all, and the answer is
  that neither path is bandwidth-bound at this context.

Either outcome is decisive, and **none of them requires a line of new engine
code, an SDD amendment, or the T-GPLR-61 mechanism.** The 7x bracket stays open
until then, because the baseline carries no per-layer decomposition - it is a
fresh end-to-end number, and it is worth saying plainly that a better end-to-end
number is not a decomposition.

## 9. The 50k comparison, answered - and it is asymmetric

The owner asked whether a ~50k post-prefill measurement exists. **For PP: yes,
and it is decisive. For TG: no, and it cannot be produced at 8 GB by the current
design.**

### 9.1 PP at 50,000 tokens - 7.1-7.9x the recorded stock baseline, on an uncapped card *(corrected 2026-09-30, thirteenth finding)*

27 recorded traces sit in the 45k-55k band. The best is
`build/llama-cuda/gplr53-a3-50k.jsonl`, and its `run_start` is fully
self-describing:

    pp_tokens 50000, schedule layer-major, n_slots 3, ctx_size 50176
    kv_cache f16, copy_source mmap-registered, staging_source mmap-registered
    act_slab bf16 409,600,000 B, n_layer 40, arch qwen35moe
    free_vram 22,226,665,472 B, total_vram 25,650,855,936 B

| 50,000-token PP | pass 1 | steady state (best pass) |
| --- | --- | --- |
| `gplr53-a3-50k` (mmap-registered) | 2,469 tok/s | **2,759 tok/s** |
| `gplr52-a2-50k` (mmap-registered) | 2,420 tok/s | 2,741 tok/s |
| `gplr50-50k-f16-default` (pinned) | 2,374 tok/s | 2,614 tok/s |
| `gplr-moe-8gb-50k` (q8, direct) | 1,393 tok/s | - |
| **stock `--fit-target 16000`, 52,819 tokens** | **347.19 tok/s** | - |

**7.9x on the best arm's steady state, 7.1x on its cold pass** (pass 1 of
`gplr53-a3-50k`, the cold-file-cache pass). The pinned arm's cold pass is
the low end, at 6.8x. *(This sentence first read "6.8x on the best arm, 7.1x
on the most conservative one", which pairs each ratio with the wrong arm.)*
On the charter's first priority, at the prompt length the owner chose, the
margin is not close.

> **Corrected 2026-09-30 (thirteenth finding).** Four qualifications, each
> read off the traces:
>
> - **That arm did not run at 8 GB.** Its `byte_account` records
>   `residency_budget_mib 0` and 22,226,665,472 B free on the 24 GB card, so
>   nothing capped it. Its own byte account puts the bounded window at
>   4,973.5 MiB: static 913.2, per-layer 1,523.5, three slots 1,494.0, f16 KV
>   980.0 and recurrent state 62.8. The activation slab adds 390.6 MiB. Compute
>   buffers and the CUDA context are not in that account. So it is *expected* to
>   fit the anchor's 7,055 MiB budget (§10.1). That is unproven.
> - **Its `peak_vram_bytes` cannot settle it.** The runner reads free VRAM
>   once, after loading the device-resident weights and before creating the
>   context (`tools/layer-streaming/main.cpp:4478`). It refreshes that reading
>   only on row-cache and `--residency-tg` runs (4956, 5036, 5742). A plain PP
>   run therefore reports `total − free` from before its KV, slots, slab and
>   compute buffers exist. Here that is 3,424,190,464 B = 3,265.56 MiB, the same
>   figure every such run reports. Nested commit `8e9757169` (2026-09-15,
>   T-GPLR-24) deleted the run-end sample that used to catch them. The fix is
>   one line, `ls_vram_budget_refresh(vram, device);` at the summary block
>   (6389). §9.4 lists the claims this voids.
> - **The one 50k-band run under an explicit cap measured 4.0x.**
>   `gplr-moe-8gb-50k` ran with `--residency-budget-mib 7000`, 4 slots, q8 KV,
>   the `direct` copy source and 51,200 tokens (trace of 2026-09-10). It
>   reached 1,392.6 tok/s at a peak of 6,693.6 MiB, taken by the run-end sample
>   that `8e9757169` later deleted. Its deficit is most likely the copy source,
>   not the cap: 108.6 s of summed transfer against 34.5 s for
>   `gplr53-a3-50k`, at nearly the same compute (17.0 s against 18.0 s). That
>   is inferred, and unproven. Nobody has run the current arm under the cap.
> - **Against tuned stock the margin is expected at ~2–2.5x.** §12.5 prices
>   stock at `-ub 2048` at 1.97–2.53x below layer-major, steady state against
>   steady state. Unproven.

Honest differences between the two: our arms use **f16 KV** where stock used
**q8_0** (so ours carries roughly double the KV bytes); our context is 50,176
where stock allocated 128,000; the binaries differ; and stock ran its default
`-ub 512`. ~~stock's PP is itself doing CPU expert work (33/41 layers
overflowing) while ours streams whole banks through three slots. Different
mechanisms, same length, and ours is 7x.~~ **Corrected 2026-09-30: stock's PP
does no CPU expert work.** From 32 tokens per call, stock copies each
host-resident expert bank to the GPU and computes it there (ninth finding,
§12.5). That is 32.79 bank-equivalents, ~14.8 GB per 512-token ubatch (tenth
finding). Both engines are PCIe streamers. Ours moves each bank once per pass,
18.3 GB for the whole 50k prompt (`phase_summary.bytes_transferred`). Stock
moves ~14.8 GB for every ubatch, about 1.5 TB for its 52,819 tokens. That is
where the 7x comes from. It is also why stock's own lever, a larger ubatch, is
expected to close most of it (§12.5).

### 9.2 TG at 50,000 tokens - the data does not exist, and cannot yet

**Every one of the 27 traces in the band has `tg tokens = 3`.** Not one has a
long generation. The recorded TG rates there are 0.456-0.769 tok/s.

That is not a data gap, it is a **gate**, and the gate is deliberate:

    third_party/llama.cpp/tools/layer-streaming/main.cpp:4644
      if (schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR && !handoff && tg_tokens > 3)

`AGENTS.md:179` records the same rule: "layer-major refuses `--tg-tokens > 3`
when handoff auto does not activate (exit 2). Long-id identity arms at the 50k
shape are impossible without forcing handoff."

And handoff is **structurally impossible at 8 GB**: the T-GPLR-10 fit gate
requires `full_model_plus_runtime` = 24,295,318,016 B, which is 3.4x the
~7,100 MiB budget. It is not a tuning problem. The cap cannot be lifted by any
flag combination available today.

**And the 3-token numbers must not be read as a rate.** They are the whole-bank
streaming path, transfer-bound at 32x byte amplification - the path §3 already
priced at 0.45-0.7 tok/s by its own ceiling arithmetic. Quoting 0.769 tok/s as
"our TG at 50k" would be quoting a three-token smoke of the worst-performing
path in the program.

### 9.3 What this means for the session

The 8 GB comparison is now half-answered and the other half is a **capability
gap, not a measurement gap**:

- ~~**PP at 8 GB: we win 7x, and the data is already recorded.** Done.~~
  **PP at 8 GB: not done** *(corrected 2026-09-30, §9.1)*. The 7x arm ran
  uncapped. The one capped run measured 4.0x, on an older copy source. Tuned
  stock is expected to cut the margin to ~2–2.5x (§12.5). Settling it takes
  two GPU runs: the current arm under `--residency-budget-mib 7000`, and stock
  at `-ub 2048` (§12.5 A/B vehicle).
- **TG at 8 GB: we have no long-generation path at all.** Every 50k trace stops
  at 3 tokens, and lifting that requires the full model resident, which 8 GB
  forbids.

This is the strongest argument the session has produced, and it is not the one
the reconciliation led with. It reframes the CPU-expert split (T-GPLR-61) from
"a latency optimisation" to **the only identified route to a long-TG path at
8 GB**: a split needs the demanded rows on the host plus whatever is resident -
no full-model handoff, no 3-token cap, and a host cost measured at ~35.7 us
per expert (§12) against 121-241 us for fetching it, 3.4-6.8x cheaper.
*(Corrected 2026-09-30: this first read "a marginal cost measured at ~9 us per
expert". The clean bench refutes that. `audit-C` runs 33.10 us at one expert
and 288.65 us at eight (§12.1a), so the marginal expert costs
(288.65 − 33.10)/7 = 36.5 us. The ~9 us was the k=4→8 slope of the withdrawn
cache-contaminated bench, (108.9 − 74.1)/4 = 8.7 us
(`docs/plans/qwen36-cpu-expert-placement-draft.md:99-100`), which §2
withdrew.)*

### 9.4 Claims the peak-VRAM defect voids *(thirteenth finding, 2026-09-30)*

**Mechanism.** `run_end.peak_vram_bytes` is `total − free_min`
(`tools/layer-streaming/main.cpp:6424`). `free_min` starts at one free-VRAM
reading, taken after the device-resident weights load and before the context
exists (4478). It moves only at three refresh points: the row-cache fit and
allocation (4956, 5036, both under `if (ls.row_set_retain)`) and row-set
activation after prefill (5742, under `if (residency_tg)`). Retention requires
`--residency-tg` (4173). **So every run without `--residency-tg` reports its
load-time usage as its peak.** Its KV, slots, activation slab and compute
buffers are never seen.

Until nested commit `8e9757169` (2026-09-15, T-GPLR-24 H-B) the summary block
took a second sample, `free_after`, and reported
`total − min(free_after, free_start)`. H-B deleted that sample while fixing a
real under-count on retention runs, whose row cache is released before the
summary. Its comment says the sample "moved into the free_min tracker"
(6389), but no call was added there. Its record claims "Non-retention shapes
are byte-identical" (`docs/currentstate/2026-09-15..2026-09-16.md:1150`). No
run tested that. The one contrast arm it planned, the reactive one, never ran,
and that arm sets `--residency-tg`, so it would have refreshed at 5742 anyway.

**The traces prove it.** On every such trace, `run_end.peak_vram_bytes`
equals `byte_account.total_vram − byte_account.free_vram` to the byte
(25,650,855,936 − 22,226,665,472 = 3,424,190,464):

| trace | shape | slots (`slot_vram_bytes`) | KV | reported peak |
| --- | --- | --- | --- | --- |
| `tg60-architecture-smoke` | batch-major, q8, 64 tokens | 1, 498.0 MiB | 2.7 MiB | 3,265.56 MiB |
| `gplr07-qa-50`, `gplr07-qa-512` | batch-major, q8, 96 TG tokens | 3, 1,494.0 MiB | 2.7 / 8.0 MiB | 3,265.56 MiB |
| `ubatch-pp-ab-a1..a4` (§11.7.2) | layer-major 50k, f16, u512 / u256 | 3, 1,494.0 MiB | 980.0 MiB | 3,265.56 MiB |
| `gplr53-a3-50k` (§9.1) | layer-major 50k, f16 | 3, 1,494.0 MiB | 980.0 MiB | 3,265.56 MiB |
| `gplr35-s4-200k` (§11.7.8) | layer-major 200k, f16 | 4, 1,860.0 MiB | **3,910.0 MiB** | 3,265.56 MiB |
| `gplr51-a3` (cited in §11.7.8 as "gplr35/51/58") | layer-major 200k, f16 | 4, 1,860.0 MiB | **3,920.0 MiB** | 3,265.56 MiB |
| `gplr-moe-8gb-50k`, pre-`8e9757169` (2026-09-10) | layer-major 51,200, q8, 7,000 MiB cap | 4, 1,860.0 MiB | 533.9 MiB | **6,693.56 MiB** |

One slot and three slots report the same figure. The 200k runs' KV alone,
3,910.0 and 3,920.0 MiB, is larger than the peak each reports.
`gplr35-s4-200k`'s own byte account puts
the bounded device window at 8,401.5 MiB before its 1,562.5 MiB slab. The
reported figure is the load-time usage: device-resident weights (static
913.2 + per-layer 1,523.5 MiB) plus ~829 MiB of CUDA context and runtime. The
one pre-`8e9757169` run reports a figure its load reading cannot produce, so
the run-end sample was still live then.

**First seen on 2026-09-16, but the cause was never found.** T-GPLR-35
addendum 2 recorded that "the `peak VRAM` telemetry line reports 3265.56 MiB
IDENTICALLY at 2, 3 and 4 slots" and switched to the bounded-window figures
(`docs/currentstate/2026-09-15..2026-09-16.md:9`). Nobody traced why. Later
sections then cited the same figure as a trust control.

**Voided** (each annotated in place):

| site | claim | what stands |
| --- | --- | --- |
| §9.1 | the 50k arm's VRAM at 8 GB | corrected in place |
| §11.7.2 | "peak VRAM is identical at 3,265.56 MiB in all four arms - so the session is trustworthy" | the throughput drift control (0.25% / 3.63%). The VRAM half is structural: it reads the same for any arms |
| §11.7.4 | "with peak VRAM identical at 3,265.56 MiB", beside the 122 / 82 split counts | the split counts, which are log lines. The runner's u512 and u256 compute buffers are *expected* to differ (§11.7's per-token term), and this instrument cannot see it |
| §11.7.8 | "corroborated by peak VRAM (3,265.56 MiB in `gplr35-s4-200k.log` …)" | the slot allocation, on `slot_vram_bytes` alone: the sum of `ggml_backend_buffer_get_size` over the slot buffers (main.cpp:5216-5225). The corroboration is false, because 3,265.56 MiB cannot hold that run's KV |
| §11.9 items 11, 12 | "peak VRAM identical"; "identical peak VRAM while PP moved 30.58%" | the PP measurements |
| `docs/currentstate.md`: the tg60 smoke and both T-GPLR-07 smoke entries | "peak VRAM was 3265.56 MiB"; "3,424,190,464 bytes (3,265.56 MiB)" | load-time usage only |

The dated archives (`docs/currentstate/2026-09-1*.md`,
`docs/plans/generic-prefill-layer-residency-archive.md`) carry the same figure.
They are history and are left as written.

**Not affected: every `--residency-tg` run made after `8e9757169`, except S2 (§9.6).** The 5742 refresh fires after
prefill, with KV, slots, slab and compute buffers still alive. On an S2 run
(`--residency-tg-window W > 0`) it fires after the full-bank slots are
released, so its reported peak is the pre-prefill cache sample. The rest are covered:

- 5,957.56 / 7,089.56 MiB (§11.7.8, §11.9);
- the honest gplr19 legs (2026-09-18) and the gplr58 kprobe legs (2026-09-23) (§3a, §15);
- the SDD's 10,161.56 MiB (2026-09-18);
- `docs/currentstate.md`'s 8,658,681,856 B D-gate, its 10,169.56 MiB gplr58 null and its 7.80 / 9.93 / 12.06 GiB retention ladder.

> **Corrected 2026-09-30 (eighteenth finding).** This list first read "the
> gplr19 A/B and kprobe legs". The retention runs made before `8e9757169` are
> voided the other way: their row cache is missing from the reported peak.
>
> - **Mechanism.** Before the fix the run-end peak was a two-sample max, and on
>   a retention run the second sample was taken after `end_row_cache` had
>   released the cache (`docs/currentstate/2026-09-15..2026-09-16.md:1143-1150`;
>   first diagnosed in `docs/currentstate/2026-09-14.md:605-613`).
> - **Which runs.** All the gplr19 families from 2026-09-13 and 2026-09-14,
>   including the prompt-A capacity families `gplr19-k2/k4/k6/k8-ab`
>   (2026-09-13, 22:01-22:22). Their 42 finished legs all report
>   6,001,590,272 B: 10 cache-off legs and 32 cache-on legs at K = 1 to 8.
>   (`gplr19-k4default`'s single leg reports 6,022,561,792 B.) That is the
>   "identical `peak_vram_bytes`" §3a and §15 relied on.
> - **What those runs actually held (measured).** Each leg ran with
>   `tools/gpu-mem-poll.ps1` attached. Its `mem-*.csv` shows exactly one
>   runner process above 1 GiB, and none of the files overlap. The per-process
>   dedicated high-water marks are:
>
>   | K | cache off | cache on | difference |
>   | --- | --- | --- | --- |
>   | 1 (`rowcache-ab`) | 4,636.7 MiB | 5,202.7 MiB | +566.0 MiB |
>   | 2 | 4,636.7 | 5,768.7 | +1,132.0 |
>   | 4 | 4,636.7 | 6,900.7 | +2,264.0 |
>   | 6 | 4,636.7 | 7,952.7 | +3,316.0 |
>   | 8 | 4,636.7 | 9,084.7 | +4,448.0 |
>
>   Both repeats of every leg agree to 0.1 MiB. The cache-on arms hold the
>   whole cache plus 20-79 MiB. At K=8 the log records
>   `40 layers x 64 rows/layer = 4370.1 MiB`.
> - **The fix reads the same thing (measured).** On the post-fix honest legs,
>   `run_end.peak_vram_bytes` sits exactly 1,073.3 MiB above the poller's
>   figure on cache-off, K=6 and K=8 alike:
>   - cache off: 5,721.56 vs 4,648.3 MiB;
>   - K=6: 9,037.56 vs 7,964.3 MiB;
>   - K=8: 10,169.56 vs 9,096.3 MiB.
>
>   So the peaks differ by the same +3,316.0 and +4,448.0 MiB on both
>   instruments. That is a retention-shape validation of the H-B fix that
>   its own record lacked. Script: `build/mem-poll-peaks.py` (scratch,
>   host-only).
>
> The throughput those families measured stands. Their VRAM figures do not.
> Read cache-on VRAM from the poller table above, not from `summary.json`.

Those runs can still miss allocations made after 5742, such as graphs first
built during TG. That gap is unmeasured. §11.7.14's stock-server peaks come
from their own `max(load, post)` harness, not from this runner.

**The fix is proposed, not applied.** It is one line at the summary block,
before `tel_run_end` (main.cpp:6389):

    ls_vram_budget_refresh(vram, device);

It restores the deleted run-end sample inside the tracker, so plain runs
again report `total − min(free at load, free at the summary)`. It is still
two samples, not a high-water mark. Allocations released before the summary
stay invisible, as they did before 2026-09-15.

- *Not applied* because the nested tree carries another workstream's
  uncommitted change (`src/models/qwen35moe.cpp`), and a rebuild would ship
  it into every validation binary.
- *Validation* needs an owner-authorized GPU run of the `gplr07-qa-512`
  shape. Its reported peak must exceed `total_vram − free_vram` by at least
  its slots plus KV, 1,494.0 + 8.0 MiB.
- *Until the fix lands, read such a peak as not measured.* A peak equal to
  `total_vram − free_vram` on a trace with `slot_vram_bytes > 0` is the
  load-time figure.

### 9.5 The slot window is charged at the wrong size, and the free-VRAM fit charges it twice *(fourteenth to sixteenth findings, 2026-09-30)*

**What the runner allocates.** Each slot is one device buffer, sized to the
largest package among the layers that rotate through it (`il ≡ s mod
n_slots`), plus a 4,096 B pad (`src/llama-layer-streaming.cpp:3224-3241`).
This model has two bank sizes: 432.0 MiB (all Q4_K) and 498.0 MiB (Q6_K down
projection). At 3 slots every slot's rotation contains a 498.0 MiB layer. At
4 slots only two do.

| slots | allocated: `graph_footprint.slot_vram_bytes` | charged: `byte_account.slot_window` = `n_slots × moe_largest_bytes` (main.cpp:573) | charge − allocation |
| --- | --- | --- | --- |
| 3 | 3 × 522,194,944 = 1,566,584,832 B (**1,494.0 MiB**) | 3 × 522,190,848 = 1,566,572,544 B | −12,288 B (the pads) |
| 4 | 2 × 522,194,944 + 2 × 452,988,928 = 1,950,367,744 B (**1,860.0 MiB**) | 4 × 522,190,848 = 2,088,763,392 B (1,992.0 MiB) | **+138,395,648 B (132.0 MiB)** |

Measured: both columns are fields of recorded traces. The 3-slot allocation
is on `gplr53-a3-50k`, `ubatch-pp-ab-a*` and `tg-k2ab-a*`; the 4-slot one on
`gplr35-s4-200k`, `gplr51-a3`, the gplr19 legs and the gplr58 legs. The
charge is on the same traces (`gplr35-s4-200k`: 2,088,763,392). The 4-slot
sum has exactly one decomposition into the two slot sizes, so "two big, two
small" is arithmetic, not inference.

**Fourteenth finding: this document used the charge as the allocation.**
§11.7.8 printed 1,492.6 MiB for 1,566,584,832 B (it is 1,494.0). §11.7.7,
§11.7.8, §11.7.9 and §11.9 items 15-17 used 1,992.0 MiB as the 4-slot
allocation; the device holds 1,860.0 MiB. Every derived figure moves:

| claim | as written | corrected | status |
| --- | --- | --- | --- |
| a fourth slot's cost (§11.7.7) | 498.0 MiB, 175.9 MiB per tok/s, 3.4x the cache | 1,860.0 − 1,494.0 = **366.0 MiB**, **129.3 MiB per tok/s, 4.6x** | bytes measured; the tok/s inputs are single arms (below) |
| 4 slots vs 3 slots + K=2 (§11.7.7) | 595.0 MiB lighter (1,992.0 vs 2,587.0) | **727.0 MiB lighter** (1,860.0 vs 2,587.0) | measured bytes |
| 4-slot row-set utilisation (§11.7.8) | 54.0 / 1,992.0 = 2.7% | 54.0 / 1,860.0 = **2.9%** | measured bytes |
| slots sized to the row-set working set would free (§11.7.8) | ~1,452 / ~1,938 MiB | **1,453.5 / 1,806.0 MiB** at 3 / 4 slots | derived |
| the compact S2 slot frees, 4 slots (§11.7.9, item 17) | 1,929.8 (W=8) / 1,867.5 MiB (W=16), 1.22x the 2 × 236k envelope | **1,801.9 / 1,743.8 MiB, 1.14x** | measured: the S2 runs' own byte report, `audit-w8.log` and `a9-w16-final2.log` (2026-09-30): "all-slots 1950351360 -> 121896960 (1743.8 MiB freed total)" |

The direction of every conclusion survives. The prizes are ~7% smaller; the
slot's byte-efficiency edge over the cache is larger. That edge still rests on
single 256-token arms that §11.7.10 item 3 puts inside run-to-run noise, so it
remains unproven.

**Fifteenth finding: the 132.0 MiB over-charge is what refused §11.7.6's
4-slot K=2 arm.** Under `--residency-budget-mib` the cache gets `budget −
bounded` (main.cpp:659-661), and `bounded` carries the 1,992.0 MiB charge:

    as coded:      6,291,456,000 − 5,266,696,704 = 1,024,759,296 B  <  1,146,093,568 B needed  (short 115.7 MiB)
    as allocated:  bounded falls by 138,395,648 to 5,128,301,056 B
                   6,291,456,000 − 5,128,301,056 = 1,163,154,944 B  >= 1,146,093,568 B         (16.3 MiB spare)

So under the budget's own definition, 4 slots + K=2 fits. *Expected, not
measured*: the arm has never run, and the margin is thin. The budget excludes
the activation slab and compute buffers by design (main.cpp:607-613), so
those are not in the 16.3 MiB.

This retracts the *structural* reading of that refusal in §11.7.6, §11.7.7
and §11.9 items 8(c) and 14. The refusal was honest to the code's arithmetic,
and the arithmetic is 132.0 MiB pessimistic at 4 slots. It does not retract
"slots and cache draw on one budget". It does promote the never-run
**4 slots + K=2** to the obvious TG candidate at this budget: each half of it
has the better single-arm number against the 3-slot K=0 control, though both
gaps sit inside §11.7.10's noise band.

The same charge enters the preflight's capacity conditions (main.cpp:4743,
`bounded + act_slab + reserve > free_start`), so a 4-slot run within 132.0
MiB of its limit is refused for memory it would never allocate. At 3 slots
the charge is 12 KiB *under*, which is noise.

**Sixteenth finding: the free-VRAM fit holds the slot window back a second
time.** `ls_vram_budget_derive` leaves the cache `free_now − (reserve +
slot_window)` (main.cpp:652-653). The stated premise is that the slot window
is one "this run has not allocated yet" (main.cpp:619, 649, 4943-4944). The
SDD says the same: "less the slot window the run has not allocated yet"
(`docs/layer-residency-sdd.md:194-195`). **On the full-bank
path that premise is false.** `ls.init` (main.cpp:4881) allocates every slot
buffer (src:3241), the activation buffers (src:3330) and the slab (src:3389)
before the refresh at 4956 reads `free_now`. `begin_row_set` allocates slot
memory only on the S2 compact path (src:9463, freeing the full-bank buffers
after, src:9510). That is the one case where the term is right.

The recorded refusal proves it to the byte (`build/llama-cuda/explicit-refuse-B.*`,
2026-09-18, 4 slots, `--reserve-mib 14000`, explicit `--residency-tg-retain 8`):

| quantity | bytes | source |
| --- | --- | --- |
| free at the preflight | 22,226,665,472 | log: "observed free VRAM : 22226665472 B of 25650855936 B" |
| free at the fit | 19,775,094,784 | refusal text |
| drop between them | 2,451,570,688 (2,338.0 MiB) | contains the 1,860.0 MiB of slots |
| allowance as coded: − 14,680,064,000 reserve − 2,088,763,392 slot window | 3,006,267,392 | refusal text: K=8 "needs 4582408192 B" → refused |
| allowance without the second charge | **5,095,030,784** | K=8 fits, 512,622,592 B (488.9 MiB) spare |

The bare-flag run of the same day reads the same `free_now`: its "4582408192
B of a 17149460480 B cache allowance" is 19,775,094,784 − 536,870,912 −
2,088,763,392. So the reading is stable across runs, not one bad sample.

What moves:

- The 2026-09-18 validation (SDD: the bare flag "fitted down to **K=5**" and an
  explicit K=8 "refused loudly" under `--reserve-mib 14000`) still proves the
  fit-down and refusal paths fire. It no longer proves they fire *at the right
  depth*. With the second charge removed, both runs resolve to K=8 at that
  reserve (derived from recorded bytes).
- This bites where charter priority 3 lives. The free branch binds on a real
  8 GB or 16 GB card and under `--reserve-mib` emulation. There it withholds
  1,494.0 MiB (3 slots) or 1,992.0 MiB (4 slots) from the cache: 2.7 or 3.6
  depth steps of ~546 MiB. On the 24 GB card under `--residency-budget-mib`,
  the budget branch binds instead, so §11.7.6's arms are untouched by this
  finding. They are touched by the fifteenth.
- No host fixture covers `ls_vram_budget_derive`: it is a `static` function in
  the runner, and no test under `tests/` names it. Both slips are outside
  every automated gate.

**Proposed fixes, not applied** (same reason as §9.4's: the nested tree
carries another workstream's uncommitted change, and the budget model is a
mechanism change that needs a Planner task contract):

1. *Sixteenth:* hold back only what is still to be allocated. On the full-bank
   path, that is nothing:

       const uint64_t pending = b.slot_window_live ? 0 : b.slot_window;
       const uint64_t held = b.reserve_bytes + pending;

   `slot_window_live` is set after `ls.init` when `row_set_window_width == 0`.
   The S2 path leaves it false, because its compact window is still pending
   at the fit.
2. *Fifteenth:* after `ls.init`, re-adopt the window at the allocated size
   (the sum main.cpp:5216-5225 already computes), so `bounded` and the
   budget branch price the real slots:

       vram.bounded     = vram.bounded - vram.slot_window + slot_bytes_allocated;
       vram.slot_window = slot_bytes_allocated;
       ls_vram_budget_derive(vram);

   Making the *preflight* exact as well needs per-layer bank bytes in the
   inventory (`Σ_s max_{il ≡ s} bank(il) + 4096`). That is lower value, and
   it is a tool change.

*Validation, owner-authorized GPU runs only:* rerun the `explicit-refuse-B`
command, which must now allocate K=8 and pass `--check-telemetry`; and run the
§11.7.6 4-slot K=2 arm, which must now allocate 1,093.0 MiB and not refuse.
A host fixture for `ls_vram_budget_derive` should land with fix 1, so a third
slip here cannot hide.

### 9.6 The compact slot's saving lands after the peak, and the W-aware preflight prices the whole run at the TG size *(seventeenth finding, 2026-09-30)*

**What the code does.** T-GPLR-62 S2 (`--residency-tg-window W`) narrows each
slot to W experts. It does so once, inside `begin_row_set` after prefill
(main.cpp:5734 → `build_row_set_window`, src:9180). It allocates every compact
buffer first (src:9463) and frees the full-bank buffers last (src:9510). Until
then an S2 run is a full-bank run:

- `ls.init` (main.cpp:4881) allocates the full-bank slots (src:3241);
- `begin_row_cache` (5026) allocates the row cache on top of them;
- prefill runs with both live.

Nothing needs the cache before the swap. Only the TG row-set path reads it
(src:9668).

The preflight does not follow that order. Since T-GPLR-62 §8.11,
`ls_bounded_window_for_policy` narrows the slot term to `n_slots ×
ceil(moe_largest × W / n_expert)` whenever W > 0 (main.cpp:552-571). All three
consumers get the narrow figure: the default slot ladder (4560), the adopted
budget window (4602, 4611) and the byte account (5150). The code comment says a
full-bank figure would refuse "for memory it is not going to use"
(main.cpp:547-548). **The run does use that memory.** It holds it from `ls.init`
to the end of prefill, which includes the whole time the cache is live.

**Measured: the reported S2 peak is a bring-up sample.** The eight S2-audit
traces share one shape:

- 4 slots, ctx 2048, no budget;
- a K=4 cache (`byte_account.row_cache` 2,291,531,776 B, `rowset_init.retain_k` 4);
- `graph_footprint` identical field for field (84 graphs, `slot_vram_bytes` 1,950,367,744).

| W | traces | TG-phase slot bytes (the runner's own report) | `run_end.peak_vram_bytes` |
| --- | --- | --- | --- |
| 0 (full bank) | `audit-w0`, `a9-w0-norecon`, `a9-control-fullbank` | 1,950,367,744 B (1,860.0 MiB), never narrowed | 8,381,857,792 B (7,993.56 MiB) |
| 4 (refused at the first TG layer, after the swap) | `a9-w4-refuse`, `a9-w4-refuse2` | 30,474,240 B (29.1 MiB) | 8,266,514,432 B (7,883.56 MiB) |
| 8 | `audit-w8` | 60,948,480 B (58.1 MiB) | 8,266,514,432 B (7,883.56 MiB) |
| 16 | `a9-w16-final2`, `a9-w16-norecon` | 121,896,960 B (116.3 MiB) | 8,266,514,432 B (7,883.56 MiB) |

Three widths hold compact windows up to 87.2 MiB apart, yet they report the
same peak to the byte. So their peak is not taken after the swap. It is the 5036
sample, taken right after the cache lands, with the full slots live. W=0 has
no swap, so its post-prefill sample (5742) still holds everything, plus what
prefill added. That addition is exactly 115,343,360 B (110.0 MiB). What it is
(CUDA pool or compute buffers) is inferred, not instrumented. Each peak
repeats to the byte across its traces.

**Derived: S2's true high-water is above the full-bank control's, not below
it.** Prefill runs the same code at every W. The width is read only on the
row-set path (src:9180-9335, 9644, 10028) and in preflight arithmetic, which
sizes nothing but the cache (main.cpp:4959). So an S2 run reaches
`begin_row_set` holding what W=0 holds at its 5742 sample. It then allocates
its compact buffers before it frees anything:

    high-water >= W=0's peak + compact buffers (with the 4 x 4,096 B pads)
      W=8:   >= 8,442,822,656 B (8,051.7 MiB); reported 7,883.56, under by >= 168.1 MiB
      W=16:  >= 8,503,771,136 B (8,109.8 MiB); reported 7,883.56, under by >= 226.3 MiB

So §9.4's "not affected: every `--residency-tg` run" does not cover S2 runs.
Their 5742 refresh fires after the full slots are released, so it never sees
prefill with those slots live.

**The release does reach the driver.** T-GPLR-62 §11.4 reads the 110 MiB gap
as a small saving. It says the full-bank buffers are "not returned to the
driver by the CUDA caching allocator". Both halves are wrong:

- A backend buffer is freed with a direct `cudaFree`
  (`ggml-cuda.cu:734-736`). ggml's CUDA pool serves compute temporaries, not
  slot buffers.
- The traces agree. Had the 1,860.0 MiB stayed held, every S2 run's 5742
  sample would sit above W=0's peak by at least its compact window (29.1 to
  116.3 MiB). Instead each S2 run reports 110.0 MiB below it. The gap is
  prefill growth that S2's samples never see, not a saving.

The contract's conclusion, no peak saving, stands. In §16.3, "the run must
still hold full-bank slots to do its own prefill" is right. "A whole-run
high-water mark" is not: the figure is four samples (§9.4), and for S2 the
binding one is taken before prefill.

**Consequence 1: budget emulation.** This is the 24 GB card under
`--residency-budget-mib`. The budget branch gives the cache `budget − bounded`
with the narrow slot term, but the cache is allocated while the full slots are
live. Take §11.7.6's shape (4 slots, 51,200 tokens, q8 KV, 6,000 MiB) at W=16:

    bounded (W=16 preflight) = 3,308,481,024 B  ->  allowance 2,982,974,976 B  ->  accepts K=5 (2,864,250,880 B)
    live through prefill     = 5,128,301,056 (allocated, §9.5) + 2,864,250,880 = 7,992,551,936 B (7,622.3 MiB)
                             = 1,622.3 MiB over the cap the operator set

The runner's own contract says that "no retain flag can place a cache on top
of a budget the operator set" (main.cpp:607-610). With W > 0 it can. The
overrun reaches 1,735.5 MiB at 4 slots (1,400.6 MiB at 3) when the allowance
is used in full. A budget-emulated S2 arm would report a cache depth that a
real card of that size cannot hold. **Expected, not measured:** every S2 run
so far used budget 0 at ctx 2048, so no recorded figure is wrong yet.

**Consequence 2: a real card, the free branch.** The fit reads free VRAM after
`ls.init`, when the full slots are already allocated (§9.5). It holds back
only the compact window still to come, and that is correct here: the swap's
transient needs exactly that room. So the cache it grants is physically there.
But the cache is sized before the swap, so the 1,743.8 MiB the swap frees is
never offered to it, and nothing else allocates after the swap.

*Expected:* on a real 8 GB or 16 GB card, S2 as ordered today buys no cache
depth and no peak reduction. Its TG-phase saving is real (measured, §9.5's
table) and unused.

**Consequence 3: the preflight under-reserves on a crowded card.** The
capacity condition (main.cpp:4743) and the default slot ladder (4560-4578)
price a W>0 run at the compact size, but `ls.init` allocates full-bank slots.
A run can pass the preflight with 1,735.5 MiB (4 slots) or 1,400.6 MiB
(3 slots) less free VRAM than bring-up will take. What happens next:

- If `ls.init` cannot allocate its slots, the run fails loudly, but as
  `init_failed` (main.cpp:4886-4892), not as the preflight's byte-accounted
  refusal.
- If the full-bank 3-slot window does not fit but the compact one does, the
  ladder picks 3 slots, and bring-up then fails where the owner's ladder
  promises 2 slots.
- If `ls.init` fits with too little left, the failure moves into prefill.

Charter priority 3 lives exactly on such a card.

**What this means for S2's value (derived).** In §11.7.6's shape at 6,000 MiB,
an honest full-bank account gives the cache K=2 (§9.5). Allocating the cache
*after* the swap would give it K=5 with 121.5 MiB spare, and every phase would
stay inside the cap:

- prefill: 4,890.7 MiB;
- the swap transient (full + compact): 5,007.0 MiB;
- TG (compact + K=5): 5,878.5 MiB.

That is the whole realizable prize of the compact slot: three cache-depth
steps at this budget. The order has to change to collect it. Whether deeper K
pays in TG tok/s is the separate §11.7.10 question.

**Proposed fixes, not applied** (for the same reason as §9.4 and §9.5; both
need a Planner task contract):

1. *Honesty, at once.* Price bring-up and prefill at the full-bank window: in
   `bounded`, the capacity condition and the slot ladder. Keep the compact
   term only as the free branch's pending hold-back (§9.5, fix 1). While the
   cache is allocated before prefill, S2 then gives the cache nothing, and the
   preflight says so instead of promising room that exists only after the
   swap.
2. *The enabler.* Fit the cache arithmetically at bring-up, which keeps
   T-GPLR-19's refusals loud before any work. Allocate it after
   `build_row_set_window`, and budget each phase against its own residency,
   using the three figures above. Constraints the contract must keep:
   - T-GPLR-58's configure-before-`begin_row_cache` order
     (main.cpp:4909-4928);
   - the generation-boundary cancel in `begin_row_set` (src:9211-9223);
   - the AGENTS.md `spec_reset` gotcha.

   The cost is that an allocation failure surfaces after prefill instead of
   at bring-up. The rationale for bring-up allocation (main.cpp:4898-4901)
   must be amended to say so.

*Validation, owner-authorized GPU runs only:*

- One S2 run under `--residency-budget-mib`. Today it must accept K=5 at
  6,000 MiB. With fix 1 it must refuse or fit down. With fix 2 it must hold
  K=5 inside the cap.
- A phase-resolved VRAM high-water instrument, which T-GPLR-62 §16.3 already
  names. Without it, no S2 peak figure is measured.

---

## 10. Re-anchoring on an 8 GB VRAM budget (session focus, 2026-09-29)

### 10.1 The run the owner supplied already IS the 8 GB configuration

No sweep is needed to obtain the 8 GB-class baseline. From the fit log:

    device total 24,462 MiB, free at start 23,119 MiB
    --fit-target 16000  ->  usable budget = 7,055 MiB
    ("id=0, target=7055 MiB" - the fitter measures free AFTER CUDA init, so it
     is not the 23,119 of the device banner; see §11.2 for the 2-session
     analogue. The 7,119 I first wrote here was wrong.)
    "41 layers (33 overflowing), 7008 MiB used, 16046 MiB free"

The fitter consumed the budget to the MiB. **A ~7.1 GiB VRAM budget on this
card is an 8 GB-class configuration**, and the owner measured it today:

| | 8 GB-class reference (2026-09-29, build 10892) |
| --- | --- |
| PP | **347.19 tok/s** (2.88 ms/token, 52,819 tokens) |
| TG | **40.96 tok/s** (24.41 ms/token, 2,551 tokens, 52,819 ctx, **no MTP**) |
| VRAM | 7,008 MiB = model 5,192 + KV 1,388 + compute 427.5 |
| Placement | 41 layers fitted, **33 overflowing to host**, 19,515 MiB CPU-mapped |
| Splits | 70 graph splits per token at bs=1, 3,907 graph nodes |
| Graphs | reused 2,540 / 2,551 |

This is the anchor for the whole session. Everything below is measured against
it, and no `--fit-target` sweep is required or wanted.

### 10.2 What 8 GB means for OUR path, from the recorded byte classes

`AGENT_ENVIRONMENT.md` already carries the arithmetic:

| Class | Bytes | MiB | In `bounded`? |
| --- | --- | --- | --- |
| **bounded 1-slot — THE SUM** | **5,995,194,880** | **5,717.7** | — (this IS the sum) |
| static weights | 957,564,416 | 913.2 | component |
| per-layer non-slot tensors | 1,597,556,224 | 1,523.5 | component |
| slot window, 1x largest expert bank (layer 34) | 522,190,848 | 498.0 | component |
| Q8 KV at 262,144 | 2,852,126,720 | 2,720.0 | **component — already inside the sum** |
| Gated DeltaNet recurrent state | 65,863,680 | 62.8 | **component — already inside the sum** |
| bounded 2-slot | 6,517,385,728 | 6,215.3 | alternative total |

> **READ THIS BEFORE ADDING ANY ROW (corrected 2026-10-01; the arithmetic here
> was misread once and the error cost a real planning decision).**
> **The rows are NOT additive.** `bounded` already contains KV and state:
> `main.cpp:585` derives it in one line —
> `out.bounded = static_b + out.per_layer_resident + out.slot_window + kv_b + st_b;`
> — so `bounded_1slot + KV + state` = 5,717.7 + 2,720.0 + 62.8 = **8,500.5 MiB**
> is a **double count**, not a footprint. It also double-counts the expert bank,
> which is inside the slot window. Adding every row of the table gives 8,998.5
> MiB, which is likewise meaningless. The only totals in this table are the two
> bolded `bounded` rows.

**Streaming prefill fits at 8 GB, even at 262k context** — **WITHDRAWN
2026-10-01; see the act-slab law below, which is why it was wrong.** The bounded
figures are right (1 slot 5,718 MiB, 2 slots 6,215 MiB); the missing term was
the act slab, which is ~2 GiB at a full 262,144-token prompt, not ~0.4 GiB. That is the regime the charter names ("a lower-end
GPU where the expert banks cannot fit even without KV") and it is reachable.
The 18.3 GB of routed-expert banks can never be resident at this budget - by a
factor of 2.6 - so layer-major streaming is not a preference here, it is the
only option.

**Which check each number actually feeds (verified 2026-10-01 against
`main.cpp`, correcting a derivation that had conflated them).** The capacity
preflight is **three independent refusals with three different right-hand
sides**, and only one of them is the operator's budget:

| # | Test (source) | Compared against | 1-slot @ 262k, q8 | 8 GB verdict |
| --- | --- | --- | --- | --- |
| 1 | `full_model_plus_runtime + act_slab <= free_start` (`main.cpp:4722`) | observed free VRAM | 19.9 GiB model | **needs `--allow-full-residency`**, else refuses |
| 2 | **`bounded > residency_budget_bytes` (`main.cpp:4730`)** | **`--residency-budget-mib`** | 5,717.7 vs 8,192 | **ACCEPT, remainder 2,474.3 MiB** |
| 3 | `bounded + act_slab + reserve > free_start` (`main.cpp:4741`) | **observed free VRAM**, *not* the budget | 6,620.3 vs ~23,000 | passes on a 24 GB card **regardless of the budget** |

So the act slab and the 512 MiB reserve are checked against **physical VRAM only**. They are *not*
subtracted from `--residency-budget-mib`, and a budget arm does not pay for
them. **Consequence for planning:** the earlier derived figure "6,620.3 MiB vs
8,192 -> 1,571.7 MiB slack" was wrong — it added tests 2 and 3 together. The
correct budget arithmetic is `bounded <= budget` alone, and the real slack is
**2,474.3 MiB**, not 1,571.7.

**THE ACT SLAB LAW — measured 2026-10-01 across 15 recorded traces, and it
invalidates the "fits at 262k" claim above.**

> `act_slab = pp_tokens x n_embd(4096) x dtype_bytes`
> **8,192 B/token in bf16. 16,384 B/token in f32.**

Measured points (`act_slab_bytes` in `run_start`, `pp_tokens` in the same event):
1 -> 8,192; 9 -> 147,456; 50 -> 409,600; 511 -> 4,186,112; 513 -> 4,202,496;
2,048 -> 33,554,432; 4,096 -> 33,554,432; **50,000 -> 409,600,000**;
51,200 -> 419,430,400; 100,000 -> 819,200,000; **144,001 -> 1,179,656,192**;
200,000 -> 1,638,400,000. The one apparent outlier, 262,144 -> 4,294,967,296
(= 2^32, twice the law), is **`act_slab_dtype: f32`**, not a different policy —
its own `run_start` says so, and 262,144 x 16,384 = 4,294,967,296 exactly.

**It scales with the PROMPT, not with ubatch and not with slot depth.**
`ab15` measured `act_slab_bytes` = **409,600,000 identical at ubatch 512, 1024,
2048 and 4096**. It is the whole prompt's hidden states — the `inp_embd`
allocation — and layer-major PP must hold it resident because every layer walks
every chunk.

**So the earlier statement "streaming prefill fits at 8 GB, even at 262k
context" is WRONG and is withdrawn.** At the full 262,144-token prompt in bf16
the act slab is **2,048.0 MiB**, not the ~140-390 MiB a ubatch reading
suggested. With the 512 MiB reserve:

| config @ ctx 262,144, KV q8 | bounded | act slab | reserve | total | vs 8,192 |
| --- | --- | --- | --- | --- | --- |
| 1 slot (does NOT pipeline) | 5,717.7 | 2,048.0 | 512.0 | **8,277.7** | over by 85.7 |
| **2 slots (the pipelining floor)** | 6,215.3 | 2,048.0 | 512.0 | **8,775.3** | **over by 583.3** |

**The activation slab, not the KV cache, is what breaks 8 GB at full context** —
2,048.0 MiB and growing linearly, versus KV's 2,720.0 MiB which does not grow
once ctx is fixed. A **50,000**-token prompt at 2 slots does fit
(6,215.3 + 390.6 + 512.0 = 7,117.9 MiB accounted; `ab15` measured a 7.289 GiB
per-process peak at ubatch 512). What does not fit is the **full** context, and
the gap is closed by making the act slab host-resident or recomputed, not by
tuning slots or ubatch.

That remainder is also exactly the row-cache cap, because
`ls_vram_budget_derive` (`main.cpp:655-662`) computes
`cache_cap_budget = budget - bounded` and takes
`cache_budget = min(cache_cap_free, cache_cap_budget)` — again with no act-slab
or reserve term. **On an 8 GB card at 262,144 context the TG row cache reaches
K=4 (2,185.4 MiB), not K=2** (K=6 at 3,277.8 MiB does not fit). This is a
materially better TG picture than the budget-vs-physical conflation implied,
and it is the number §10.3's ladder should be read against.

### 10.3 What 8 GB means for the TG row cache

**K=1 costs 546.8 MiB.** That is not one expert per layer: the capacity rule is
`cap = min(n_expert, K * W * top_k)` with `W == 1` and `top_k = 8`
(`llama-layer-streaming.cpp`, `row_cache_bytes_for`), so **K=1 is one full
8-row window per layer, 40 layers x 8 rows** - i.e. exactly one token's top-8
demand, nothing more. The ladder, measured: 546.8 / 1,093.0 / 2,185.4 /
3,277.8 / 4,370.1 MiB at K=1/2/4/6/8.

Against a 1-slot streaming config already occupying 5,718 MiB of a ~7,100 MiB
budget: K=1 fits with ~836 MiB spare, K=2 is marginal, K=4+ does not fit at
262k context.

> **CORRECTION 2026-09-29 (owner, same day). The paragraph above is wrong and is
> withdrawn.** It compared the row cache against the **PP-phase** bounded peak,
> but `--residency-budget-mib` is a **TOTAL residency cap for the token-generation
> phase** (`main.cpp:570`: "a TOTAL residency cap, not a window-only cap"), and
> the question is the TG-phase peak, after the PP act slabs and layer slots have
> been released. The engine already computes the answer - see the replacement
> text below. The owner was right and the arithmetic was not.
>
> *Citation corrected 2026-10-01.* This block cites `main.cpp:570` for that
> quote. In the current source, line 570 is slot-window arithmetic inside
> `ls_vram_estimate`; the quoted wording is the flag's own help text at
> `main.cpp:221-227` ("explicit experimental TOTAL residency budget ... the
> T-GPLR-19 row cache must fit in what it leaves (bounded residency + cache <=
> budget)"). The owner's correction stands unchanged. §10.2 (corrected
> 2026-10-01) now carries the exact derivation, `main.cpp:655-662`, and the
> consequence that at 262,144 context an 8,192 MiB budget reaches **K=4**
> (2,474.3 MiB of remainder), not the K=1/K=2 picture the withdrawn paragraph
> implied.

**K=1 is not inert, though - it has a measured A/B** (T-GPLR-19, 4 layer slots,
256 TG tokens, reactive baseline held at 11.863-11.967 tok/s across five GPU
sessions):

| K | cache | bytes saved / token | TG off | TG on | gain | effective GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 546.8 MiB | 396,398,208 | 11.953 | **14.021** | +2.068 | 14.29 |
| 2 | 1093.0 MiB | 324,793,824 | 11.967 | 15.820 | +3.853 | 12.18 |
| 4 | 2185.4 MiB | 214,053,600 | 11.863 | 19.727 | +7.865 | 10.67 |
| 6 | 3277.8 MiB | 163,659,552 | 11.878 | 22.074 | +10.196 | 10.52 |
| 8 | 4370.1 MiB | 129,075,264 | 11.873 | 24.481 | +12.609 | 10.23 |

So K=1 buys a real **+2.068 tok/s (+17%)** for 546.8 MiB, and it is the only
depth whose saved bytes convert *above* the link ceiling (14.29 GB/s effective
vs the 13.7 GB/s bus figure - and note that comparison is the wire-time
statistic §3 warned against, so read it as "the first window is the cheapest
one", not as a bandwidth claim). Deeper is slower per byte saved but larger in
outcome; the marginal cost per step is 7.7-9.6 GB/s, close to the measured
8.42 GB/s effective per-fetch rate.

Two caveats that bind this hard:

- The K=1 hit rate is **0.3088** (25,202 / 81,600 on the offline (b)
  statistic, same stream) against 0.6461 at K=6. The 0.65 in §6 is a K=6/K=8
  number, so the budget that forces K=1 also forces the worst hit rate on the
  ladder. **The two facts compound.**
- This A/B ran on the **512-token anchor vehicle** with a 4-slot config, and
  its `TG off` column is the ECHOED series - the optimistic pairing, which a
  non-echoing stream would compress by roughly 0.06-0.09. It is not a 50k
  measurement.

Even the deepest depth measured, K=8 at 4,370.1 MiB, reaches 24.481 tok/s
against the stock 8 GB-class reference of 40.96 - and K=8 does not fit an 8 GB
budget anyway. The cache helps; it does not catch the incumbent.

### 10.3b The capacity preflight already answers this - K=4, not K=1

`--residency-budget-mib` is a **TOTAL residency cap** for the token-generation
phase, not a window-only cap (`main.cpp:570`). The preflight resolves the
largest depth that fits and **refuses loudly** with reason `row_cache_depth`
and the message *"the largest depth that fits is K=N"*. It has already been
exercised, and it was recorded:

| arm | flags | verdict |
| --- | --- | --- |
| R1 | `--residency-budget-mib 4700` | exit 1, `run_refused` reason **`row_cache_fit`**, needed 2,291,531,776 / allowed 210,310,656 B |
| **R2** | `--residency-tg-retain 4` + **`--residency-budget-mib 6000`** | exit 1, reason **`row_cache_depth`**, needed 2,291,531,776 / allowed 1,573,459,456 B, **"the largest depth that fits is K=2"** |

R2's allowance is 1,500.2 MiB at a 6,000 MiB budget, so the non-cache
residency in that shape is ~4,500 MiB. Carrying that to the 8 GB-class budget
of ~7,055 MiB predicts an allowance of ~2,555 MiB - which admits **K=4**
(2,185.4 MiB) and still excludes K=6 (3,277.8 MiB). **Superseded for the
2-session case by §11.6, which is computed from measured class bytes rather
than carried from R2.**

**That prediction is falsifiable in one run, with no code:** set
`--residency-budget-mib` to the 8 GB figure and read the refusal's own
`largest depth that fits is K=N`. The instrument is already built, already
tested, and already emitted into the JSONL (`--check-telemetry` prints the
needed/allowed byte pair and `cap_source`).

Why this matters: the A/B above puts **K=4 at 19.727 tok/s** against K=1's
14.021 and a baseline of 11.863 - so the withdrawn "K=1 is the ceiling" claim
was costing 5.7 tok/s of a lever that was never actually capped there. The
honest statement is that **the 8 GB depth is unknown until the preflight is run
at the 8 GB budget**, and the recorded evidence at 6,000 MiB says K=2, not
K=1. Note also that the R1 arm that omitted `--residency-tg-retain` ran to
completion with `row_cache` 0 - the cache is opt-in and does nothing at all
unless asked for.

This is an 8 GB *budget* on a 5090, not an 8 GB *card*. Two directions of error,
and they do not cancel:

- **Compute is optimistic.** A 5090 is far faster than any 8 GB card. The
  46.6 ms/token compute term and the whole GPU half of the incumbent's 24.41 ms
  would be much worse on real 8 GB hardware. TG-at-8 GB is therefore *harder*
  than the 40.96 tok/s reference suggests.
- **The link may be pessimistic.** T-Q38TG-00 measured **13.7 GB/s end to end**
  on the previous machine's Gen4 x8, which is a *cleaner* number than anything
  this box has produced. A real 8 GB card's link may well beat this laptop's
  achieved 8.26 GB/s.

What survives both caveats is the **byte arithmetic** - §10.2 and §10.3 - and
the structural argument in §4, neither of which depends on a rate at all. The
timings do not transfer and should not be quoted as if they did.

### 10.5 What this session should hold to

1. The 8 GB-class reference is **in hand** and needs no sweep.
2. Our streaming PP fits the budget at 1-2 slots even at 262k, which is the
   regime the charter cares about first.
3. The TG row cache is squeezed to K=1 at that budget, and K=1 is its worst
   measured operating point. This is an argument the reconciliation did not have.
4. Every rate quoted in this document is now an **achieved feed** (4.4 /
   6.5-8.2 / 8.26 GB/s instrumented whole-bank, **10.35 GB/s marginal
   end-to-end**), never `copy_gpu_ms` wire time. Prefer the marginal figure for
   any question of the form "what does one more staged byte cost a token"; see
   §15.2.

---

## 11. The two-session regime (2026-09-29) - where the problem inverts

Owner goal: **more than one inference at maximum context.** Baseline supplied:
`docs/llamacpp 8gb vram base 2x 262k context q8.md` (build 10892).

### 11.1 What the run actually is

`--parallel 2`, `-c 512000` -> n_ctx 512,000, **n_ctx_seq 256,000 per session**,
q8_0 KV, FA, `--fit-target 16000` (usable budget 6,993 MiB - §11.2).

| | 1 session (base) | **2 sessions (this run)** |
| --- | --- | --- |
| layers fitted on GPU | 41 | **26** |
| resident weight buffer | 5,192.44 MiB | **1,311.95 MiB** |
| KV | 1,328.12 MiB (CUDA) | **3,187.50 CUDA + 2,125.00 CPU** |
| compute buffer | 427.53 MiB | **2,099.41 MiB** |
| total device "self" | 7,008 MiB | 6,674 MiB (model 1,311 / context 3,262 / compute 2,099) |
| TG measured | 40.96 tok/s | **23.26 -> 22.39 tok/s** (one session) |

Three things in that log change the picture, and two of them correct premises:

1. **The KV does NOT fit in VRAM.** There is a `CPU KV buffer size = 2125.00
   MiB` - 2,125 MiB spilled to host. The fitter says it outright: *"with only
   dense weights in device memory there is **still a total deficit of 1808
   MiB**"*. At 2 sessions the budget is oversubscribed before a single expert
   weight is placed.
2. **Only one slot actually ran.** Slot 1 took a 372-token prompt and ~300
   generated tokens; slot 0 stayed empty (*"skipping, slot is empty"*,
   *"saving idle slot to prompt cache"*). So this is **2-parallel capacity with
   one session exercised** - concurrent-2 throughput is still unmeasured.
3. **Per-session context is 256,000, not 262,144.** `-c` is split across
   `--parallel 2`. The model's true maximum is only reachable single-session, or
   at `-c 524288` (2,288 more tokens of KV) if it fits.

### 11.2 The 8 GB budget, stated exactly (corrects 7,119 -> 6,993)

The spendable budget is the fitter's own line, not my arithmetic:
`common_params_fit_impl: id=0, target=6993 MiB`, against a free of **22,993 MiB**
measured at that step (not the 23,119 MiB of the initial device query, which was
taken before CUDA init). **Budget = 6,993 MiB, fitted = 6,674 MiB, spare =
319 MiB.** Physical check: 16,297 free at teardown against the 16,000 reserve
= **297 MiB**.

The **1,491 MiB "unaccounted"** is the CUDA context / allocator overhead
(24,462 - 22,993 = 1,469 before load, 1,491 after - the same number). It sits
on the *free* side of the ledger, outside both the budget and the three named
classes, so **it does not reduce headroom**; it is why the 8 GB simulation needs
22,993 + 1,491 = 24,484 MiB of card. An earlier revision of this section
treated it as extra cost and concluded headroom was "at or below zero". That was
wrong: the fitter's model is accurate to ~22 MiB here, and the real headroom is
**319 MiB nominal / 297 MiB physical.**

### 11.3 How many weights fit? Two complete layer packages

Subtract the two untouchable classes from the budget:

| class | MiB | source |
| --- | --- | --- |
| fitter budget | 6,993.00 | measured (`target=6993`) |
| - context (CUDA0 KV 3,187.50 + RS 75.38) | -3,262.88 | measured |
| - compute | -2,099.41 | measured |
| **= weight envelope** | **1,630.71** | derived |
| weight class actually spent | 1,311.95 | measured (`CUDA0 model buffer size`) |
| **weight budget left on the table** | **318.76** | derived |

**The weight envelope is 1,630.71 MiB.** Against it (byte classes from
`AGENT_ENVIRONMENT.md`: static 957,530,112 B; non-expert/layer 39,937,088 B;
smallest bank 452,984,832 B; largest full package 563,962,624 B):

| configuration | MiB | fits 1,630.71? |
| --- | --- | --- |
| output head only (the static that is ever resident) | 398.02 | yes, 1,232.69 left |
| **output head + 1 largest complete layer package** | **935.73** | **yes, 694.98 left** |
| output head + 2 complete packages (both smallest) | 1,338.18 | **yes, 292.53 left** |
| output head + 2 packages, one q6_K bank | 1,404.18 | **yes, 226.53 left** |
| output head + 3 complete packages (smallest) | 1,808.26 | **no, 177.55 over** |

**Answer to "how many weights can we fit": two complete layer packages, with
226-292 MiB over. Three do not fit at any mix.** [CORRECTED 2026-09-29: an
earlier revision of this table charged the full 913.16 MiB static class against
the budget. `token_embd` is never resident - see §11.7.3 - so the static floor
is 398.02 MiB, not 913.16, and the answer moves from "one" to "two."]

**The 318.76 MiB the stock fitter leaves unused buys only non-expert tensors.**
Marginal cost per extra dense layer, read off the fitter's own ladder: 30/29 ->
27/26 = 132 MiB / 3 = **44.0 MiB**; 25/24 -> 26/25 = **44.0 MiB** (the 27 -> 25
step is unusable - context shrank 536 MiB inside it). So 318.76 / 44 = **7
more layers, 26 -> 33**, and the byte table's 38.08 MiB/layer would give 8
(26 -> 34). **Zero expert banks: the smallest is 432.0 MiB, 1.36x the entire
spare.** Routed-expert banks are 85.7% of all tensor bytes, and they are exactly
the 0 bytes that fit.

So the stock answer is "26 shallow layers, no experts"; the layer-residency
answer is "2 whole layers, all 41 in rotation". Different currencies. The stock
plan has breadth of cheap tensors and executes 16 of 41 layers with no GPU
weights at all; the residency plan has two layers and executes all 41, two at a
time, out of a 226-292 MiB margin. [CORRECTED 2026-09-29 - see §11.7.3]

### 11.4 The context-length lever is larger than the whole weight envelope

Measured: 2 x 256k -> KV **5,312.50 MiB** (2,125.00 host + 3,187.50 device).
2 x 128k -> **2,656.25 MiB**, saving **2,656.25 MiB** - **1.63x the entire
1,630.71 MiB weight envelope**, for free, at the cost of half the context.
The other large term, the compute buffer (2,099.41 at 2 sessions vs 427.53 at
1), scales with slots x batch, not context.

Note also the simulation is flattering by ~800 MiB: `--fit-target 16000` on a
24,462 MiB card leaves 8,462 MiB of envelope, of which 1,491 is overhead =
6,993 spendable. A real 8,192 MiB card has **~6,193 MiB spendable**. Every
headroom figure above should be read with that 800 MiB subtracted on real
hardware.

### 11.5 Does K=1 fit in the envelope? Yes - and it still cannot be used

K=1 is 546.8 MiB. The weight envelope is 1,630.71 MiB; the static floor is
**398.02 MiB** (§11.7.3, the embedding is host-side), so K=1 needs 944.82 MiB
and **fits with 685.89 MiB over.** [CORRECTED 2026-09-29: this section
previously charged the full 913.08 MiB static class and concluded K=1 did not
fit. It does.] **Fitting is not the same as working:** §11.6.1 shows K=1
refuses outright with two concurrent sessions.

More importantly: **the row cache is the wrong thing to be sizing.** At 2
sessions the classes invert -

| class | 1 session | 2 sessions |
| --- | --- | --- |
| weights | 5,192 MiB (74% of self) | **1,311 MiB (20%)** |
| context (KV+RS) | 1,403 MiB | **3,262 MiB (49%)** |
| compute | 427 MiB | 2,099 MiB (31%) |

**Layer residency governs weight bytes, and weights are now the smallest class
in the budget.** The 8 GB two-session problem is a KV-footprint problem, and
every tool this program owns points at the 20%. That is the most important
strategic fact in this document.

What does fit at that shape is the configuration the runner already supports:
**the row-set path with the cache OFF** - one 14.2 MB window per layer, no
persistent per-layer retention. The R1 arm in the 2026-09-14 record is the
precedent: with `--residency-tg-retain` omitted it ran to completion,
`row_cache` 0, 21,020 events. Whole-bank streaming is the *borderline* case, not
the impossible one: one 498 MiB bank fits inside the 1,630.71 MiB weight
envelope (§11.3) but not inside the 318.76 MiB the stock fitter leaves spare,
and not at all on a real 8,192 MiB card.

### 11.6 Can we run K=1 at 2 x 256k with q8 KV? No - and the cache is not the reason

Owner question, 2026-09-29, after rejecting q4 KV. Answer: K=1 is
**arithmetically allocatable and practically dead**, and the binding term is
not the cache - it is the non-expert tensors the cache displaces.

The cache sizing rule is measured, not assumed (`llama-layer-streaming.cpp`,
`row_cache_bytes_for`): `cap = min(n_expert, K*W*top_k)` with W == 1 and
top_k == 8. **K=1 is exactly one token's top-8 window per layer** - 8 rows x 40
layers = 320 rows; the recorded allocation is 546.8 MiB (the raw 320 x
1,769,472 B = 539.96 MiB plus per-buffer overhead).

The budget equation at 2 x 256k q8, against the 1,630.71 MiB weight envelope
(§11.3), with **`static_resident = 398.02 MiB` (the output head only - the
embedding is host-side by hard-coded design, §11.7.3)** and per-layer non-expert
cost given three ways (the spread is unreconciled - see the note below):

| K | cache MiB | static + cache | left for non-expert | layers at 38.08 / 44.0 / 71.5 MiB |
| --- | --- | --- | --- | --- |
| 0 | 0 | 398.02 | 1,232.69 | 32 / 28 / 17 |
| **1** | **546.8** | **944.82** | **685.89** | **18 / 15 / 9** |
| 2 | 1,093.0 | 1,491.02 | 139.69 | 3 / 3 / 1 |
| 4 | 2,185.4 | 2,583.42 | -952.71 | impossible |

**K >= 3 is arithmetically impossible at 2 x 256k q8. K=2 fits with 139.69 MiB
to spare, and that 139.69 MiB is 1-3 non-expert layers out of 41 - so K=2 is
allocatable and still useless. K=1 leaves 9-18 resident layers.**

[CORRECTED 2026-09-29: an earlier revision of this table charged the full
913.16 MiB static class, which put K=2 out of reach and K=1 at 2-4 layers. The
embedding is never resident (§11.7.3), so the correct static floor is 398.02
MiB. K=1 moves 2-4 -> 9-18 layers and K=2 becomes allocatable. The conclusion
below survives on corrected numbers.]

Per-layer non-expert, three sources that do not agree:
39,937,088 B = 38.08 MiB (byte table, `AGENT_ENVIRONMENT.md`); 44.0 MiB (the
fitter's own ladder, 30/29 -> 27/26 and 25/24 -> 26/25); 71.5 MiB (the
fitter's measured 3,845 MiB for all 41 dense layers, over 41, with no static
subtracted). The spread is 1.9x and is not yet explained, so every count above
is a range.

**Why 9-18 resident layers is still fatal.** With 23-32 of 41 layers holding no
resident non-expert weights, per-token non-expert traffic becomes 875-1,219
MiB/token, against the 13.5 MiB/layer the row-set path is built around. Scale
the owner's own measured 2-session rate by the traffic ratio - same box, same
link, same model, traffic-dominated:

- stock 2-session measured: 540 MiB/token of expert traffic (40 usable expert
  layers x 8 rows x 1,769,472 B - the MTP bank is not loaded, §11.7.3) at
  1/22.39 tok/s = 44.66 ms/token
- K=1 shape: 1,416-1,759 MiB/token = **2.6-3.2x** the traffic
- **scaled: 117-143 ms/token = 7.0-8.5 tok/s** (derived, linear-scaling
  assumption stated; not measured; the 71.5 MiB/layer end gives 4.5 tok/s)

**So K=1 at 2 x 256k q8 costs 2.6-3.2x the link traffic to buy a cache that is
worth +2.068 tok/s at 1 session.** The two do not compose. The row cache is
not where the two-session problem is won.

**A note on K=1 being the *best* depth per byte, not the worst.** From the
T-GPLR-19 A/B (baseline 11.915 tok/s, 4 layer slots, 256 TG tokens), value per
MiB of cache: K=1 **0.00385**, K=2 0.00357, K=4 0.00358, K=6 0.00310,
K=8 0.00288 tok/s/MiB - monotonically decreasing, so K=1 is the most
byte-efficient depth and K=8 the least. The problem is not that K=1 is a poor
depth; it is that **546.8 MiB cannot be spent on a cache at this shape at all**,
and that its hit rate is measured at 1 session only.

### 11.6.1 MEASURED 2026-09-29: the 2-session K=1 hit rate - the prediction was wrong in kind

The previous revision of this section carried a prediction: *"the 2-session K=1
hit rate is at most ~0.15, plausibly lower."* **It is wrong in both magnitude
and kind.** Measured from two real single-session traces, using the retention
cache's own exact policy read from source.

**Two facts that made this answerable without a runner change.** The retention
cache is a **per-layer** structure of `cap = min(n_expert, K*W*top_k) = 8K`
rows with LRU eviction, and **a row needed by the current window is never
evicted** - `llama-layer-streaming.cpp:9169-9223`; exceeding `cap` after
eviction is a **loud refusal** (`tel_rowset_refused`, reason
`insufficient_vram`, "per-layer cache holds N rows; layer L still needs M
after eviction"), never silent growth. And the runner **cannot** drive two
sessions at all: `cparams.n_seq_max = 1` (`main.cpp:4271`) and every batch row
is `n_seq_id = 1`. So this is answered by **simulating the documented policy
over two measured route streams**, not by a concurrent run.

| provenance | value |
| --- | --- |
| session A | `build/llama-cuda/gplr19-k1/k1-retain.jsonl` (recorded T-GPLR-19 K=1 arm) |
| session B | `build/llama-cuda/gplr19-2sess/sessionB.jsonl` (**new**, 2026-09-29) |
| session B run | same flags as the K=1 control, only `--prompt-file` differs (maintenance-log anchor vs a mathematics prompt, 435 words vs 366) |
| verdict | `rowcache-2sess-B.done` **PASS 0**, `gpu_pre=clean gpu_post=clean`, `digest 3EE2B9E87F42C375`, `llama.dll 2BEE822926D5111F` |
| telemetry check | `check-telemetry: OK (21449 events, 4 slots)`, exit 0 |
| input | `rowset_fetch.union_ids` - the **full** per-token per-layer top-8, 10,240 events = 256 steps x 40 layers |

**Model validation first.** At K=1 the cache holds exactly the previous
window, so the hit rate must equal the mean consecutive-window overlap
`|A_t INTERSECT A_(t-1)| / 8`. Measured on session A: **2.4708 / 8 = 0.3088** -
**exactly the recorded figure**, which also confirms the recorded denominator
81,600 is correct (255 transitions x 8 x 40, cold first step excluded). The
same computation on `n_rows_resident` gives 25,202 / 81,920 = 0.3076, which is
the same number with the cold step in the denominator. **The doc's K-ladder
figures stand.** (An LRU simulator was written first and returned 0.3915; it
models replacement within the window rather than a whole-window swap, so it was
discarded rather than reported.)

**The two-session result:**

| case | per-layer demand | K=1 cap = 8 | outcome |
| --- | --- | --- | --- |
| 1 session (recorded + revalidated) | 8 | fits | **0.3088** |
| 2 sessions **time-sliced** (A,B,A,B...) | 8 | fits | **0.0341** - a **9.1x** collapse |
| 2 sessions **batched** (= what `--parallel 2` does) | mean **15.34**, max 16 | **exceeded** | **refuses on 10,199 of 10,200 steps (99.99%)** |

Cross-session shared experts at the same step, per layer: **0.659 of 8 (8.2%)**
- the two sessions route almost disjointly, as the different domains predict.

**Two conclusions, both stronger than the prediction.** (1) Even in the
forgiving time-sliced case the collapse is **0.0341, not ~0.15** - because
alternating between two sessions destroys the temporal locality that produces
the 1-session 0.3088 in the first place; the within-session overlap does not
transfer across the session boundary. (2) The case llama.cpp actually runs is
batched, and there **K=1 does not serve at all - it refuses.** 60.5% of steps
demand the full 16 rows and 99.99% exceed the 8-row cap.

**Structural floor: K=2 is the minimum depth that can serve two concurrent
sessions** (cap 16 = the maximum union of two top-8 sets). That is arithmetic,
not tuning.

**Which retires the whole 2-session row-cache question at ubatch 512.** From
§11.7.1's envelope arithmetic, at the owner's ubatch 512 the weight envelope is
1,630.71 MiB: K=1 + 2 complete layer packages fits (1,884.98? no - over by
254.27) and K=2 does not (over by 330.39). **There is no viable row-cache
configuration for two concurrent sessions at ubatch 512.** At ubatch 256 the
envelope is 2,040 MiB and **K=2 + 1 complete layer package = 1,961.10 fits with
78.90 MiB spare** - the single point in the whole sweep where a row cache and
two sessions coexist. Two independent measurements (cache capacity, compute
buffer) converge on the same cell.

### 11.7 MEASURED 2026-09-29: the compute buffer scales with ubatch only, and the 800 MiB target is unreachable

Tool: `tools/fit-compute-sweep.ps1` (new, this session), driven by
`llama-fit-params -fitp on` -> `common_fit_print` ->
`common_get_device_memory_data_impl` - the same estimator the fitter uses.
**Validated against the owner's measured run: estimate `compute 2099` vs
measured `CUDA0 compute buffer size = 2099.41`, `context 3262` vs 3,262.88,
`host context 2175` vs 2,175.25.** So this is a memory measurement, not a
speed one, and it costs seconds per cell instead of a 21 GB load.

| provenance | value |
| --- | --- |
| verdict | `computebuf-sweep.done` **PASS 0**, `computebuf-ctx-slots.done` **PASS 0**, `fitngl-u{256,128,64,512}.done` **PASS 0** |
| provenance | `digest 3EE2B9E87F42C375`, `llama.dll 2BEE822926D5111F` on every cell |
| mutex | `Global\AI-GPU-EXCLUSIVE-v1`, `gpu_pre=clean gpu_post=clean` on all 6 holds |
| telemetry | `build/llama-cuda/logs/computebuf-sweep-*.csv`, `fitngl-u*.log` |

**Result 1 - `--batch-size` does nothing. Only `--ubatch-size` moves it.**
At `-c 512000 --parallel 2`, batch 8192 / 4096 / 2048 / 1024 / 512 all give
**compute = 2099 at ubatch 512** and **1691 at ubatch 256**, to the MiB.
`--batch-size` is not a lever; the earlier suggestion of `--batch-size 4096`
in this section was wrong and is withdrawn.

**Result 2 - the compute-buffer ladder (measured):**

| ubatch | compute (MiB) | freed vs 512 | real fitter `-ngl` | fitted used / free |
| --- | --- | --- | --- | --- |
| 512 (owner) | 2,099 | 0 | **26** | 6,674 / 16,318 |
| 256 | 1,691 | **408** | **30** | 6,962 / 16,030 |
| 128 | 1,619 | 480 | **30** | 6,890 / 16,102 |
| 64 | 1,584 | **515 (max)** | **30** | 6,854 / 16,138 |

**The 800 MiB target cannot be reached. The ceiling is 515 MiB** (ubatch 64),
a 285 MiB shortfall, because the ladder saturates: below ubatch 256 only a
ubatch-independent residual (graph overhead, the 70 graph splits, the FA
workspace) remains. And **the fitter saturates too - at `-ngl 30` from ubatch
256 downward.** Only the 512->256 step converts into anything: it buys
**4 more layers (26 -> 30)**. The further 107 MiB freed at ubatch 64 stays as
headroom (used *falls* 6,962 -> 6,854 because compute fell with it).

**The freed memory is spent, not banked** - the fitter immediately reallocates
it to more offloaded layers, which is what the design question needs.

**Result 3 - RETRACTION 3: the compute buffer is context-dominated, not
slot-dominated.** The previous revision of this section claimed *"1,671.88 MiB
is the measured price of the second slot."* That compared the owner's 1-session
run (`-c 128000`) against the 2-session run (`-c 512000`) and so changed two
variables at once. Measured, holding everything else fixed:

| ctx | batch | ubatch | parallel | compute (MiB) |
| --- | --- | --- | --- | --- |
| 128,000 | 8192 | 512 | 1 | 627 |
| 256,000 | 8192 | 512 | 1 | 1,201 |
| 262,144 | 8192 | 512 | 1 | 1,229 |
| 512,000 | 8192 | 512 | 1 | 2,349 |
| 256,000 | 8192 | 512 | 2 | **1,076** |
| 128,000 | 8192 | 512 | 2 | 564 |

**At the same total `-c`, the second slot makes the compute buffer 125 MiB
CHEAPER, not dearer** (1,076 at p=2 vs 1,201 at p=1, c=256,000), because
`n_ctx_seq` halves. The buffer tracks `n_ctx` almost linearly (627 -> 1,201 ->
2,349 for 128k -> 256k -> 512k at p=1). **The second slot is not what costs
the memory; the context length is.** With 256k fixed per session by the
owner's requirement, this lever is capped by ubatch alone.

*Caveat on the 1-session row:* the estimator says 627 at `-c 128000 --parallel 1
-ngl 26`, where the owner's real run measured 427.53 - but that run had
`-ngl 41`, and graph splits depend on the offload count. **The estimator is
validated at `-ngl 26` (exact) and is not claimed valid at other `-ngl`; every
cell in this section holds `-ngl 26`.**

### 11.7.1 What ubatch 256 actually buys (the answer)

Envelope = budget - context - compute. At `-c 512000 --parallel 2`:

| ubatch | envelope (MiB) | best residency that fits | spare |
| --- | --- | --- | --- |
| 512 (owner) | 1,630.71 | static + **2** complete layer packages | 292.53 |
| **256** | **2,040** | **static + K=1 cache (546.8) + 2 complete packages** | **155.02** |
| 256 | 2,040 | static + **K=2** cache (1,093.0) + 1 complete package | 78.90 |
| 128 / 64 | 2,090 / 2,125 | same as 256 - the fitter cannot use the extra | - |

**`--ubatch-size 256` is the setting.** It is the only cell that adds a
capability rather than headroom, and it is what makes **K=1 row cache + 2
complete layer packages** coexist at 2 x 256k q8, with 155 MiB to spare.
**K=2 + 2 packages stays out of reach** - it would need ~391 MiB more than
ubatch 256 provides, and 800 MiB was never available.

**The cost is unmeasured and it lands exactly where the charter is most
sensitive.** Token generation is unaffected - TG runs at 1 token per ubatch
whatever this is set to - so the whole cost falls on **prefill**, which is
charter priority #1. Halving the micro-batch means more launches and less
device parallelism per launch. **The 2,759 tok/s 50k PP reference was measured
at ubatch 512; no recorded arm measures PP at ubatch 256.** Do not adopt
ubatch 256 on the strength of the memory result alone - it needs a PP A/B
against the same trace before it is a recommendation rather than a candidate.

### 11.7.2 MEASURED 2026-09-29: ubatch 256 costs 30.6% of prefill - the one viable 2-session cell is DEAD

> **Confirmed from the host side and extended upward 2026-09-30 (§12.2a).** The
> expert-FFN microbench puts ubatch 256 at **76.0 us/token/layer against 512's
> 70.4** at the 16 threads the recorded runs used — an 8% host-side penalty,
> the same sign as the 30.6% measured end-to-end here. More useful: the sweep
> also priced **upward**, which nothing had. Ubatch 1024 reaches **56.5
> us/token**, a 20% gain over 512 — but `--threads-batch 28` reaches the same
> floor at ubatch 512, and the two **do not stack** (28 threads at 1024 is
> 56.9, unchanged). So the compute-buffer envelope this whole §11.7 family
> fights over was defending a gain recoverable for free from a flag.
>
> *Corrected 2026-09-30 (§12.5).* The "do not stack" reading holds only on the
> CPU path (`--no-op-offload`). Stock prefill copies the host experts once per
> ubatch, so a larger ubatch is the stock lever: u1024 is expected at
> 1.74–1.92x, and `--threads-batch` is expected to move nothing measurable. The
> envelope was defending the real lever, not a free one. The 30.6% cost of ubatch
> 256 measured below is on this repo's layer-major path and stands.

The last unmeasured downside of §11.7.1, now measured. Driver
`tools/run-ubatch-pp-ab.ps1`, 4 arms ABBA (512, 256, 256, 512) inside **one**
mutex hold, `--pp-repeat 3` so each arm also carries its own internal
replication. **The arm is the recorded T-GPLR-53 A3 50k arm with exactly one
token changed (`--ubatch`)** - same prompt, ctx 50176, 3 slots, layer-major,
mmap-registered, f16 KV, bf16 act slab, async feed, recon.

| provenance | value |
| --- | --- |
| verdicts | `ubatch-pp-ab-a{1..4}-*.done` **PASS 0** each; session `ubatch-pp-ab.done` **PASS 0** |
| provenance | `digest 3EE2B9E87F42C375`, `llama.dll 2BEE822926D5111F` on all four arms |
| mutex | one hold, `gpu_pre_run = clean`, `gpu_post_run = clean`, per-arm post-probe clean |
| telemetry | `check-telemetry: OK` on every arm (arm 4: 3589 events, 3 slots) |
| data | `build/llama-cuda/logs/ubatch-pp-ab.csv`, `ubatch-pp-ab-a*.jsonl` |

**The control is this session's own ubatch-512 arm, NOT the recorded
2,759 tok/s.** The 2026-09-22 reference ran `exe FE3B0A108C1D9BC9 /
dll 8A9F3D83F24D0FEC`; today's binary is `exe C12FE2C084EFA339 /
dll 2BEE822926D5111F`. **Both differ**, so that number is a cross-binary
reference and could not have been used as a control - the A1/A4 arms exist
precisely to supply one on a single binary. (Today's own u512 steady,
2,532.5 tok/s, is 8% below the recorded 2,756.5, which is the size of the
cross-binary gap and a reason to distrust the historical figure.)

| arm | ubatch | chunks | pass 1 | steady (pass 2-3 mean) | duration |
| --- | --- | --- | --- | --- | --- |
| A1 | 512 | 98 | 2,060.1 | 2,486.5 | 80.3 s |
| B1 | 256 | 196 | 1,626.0 | 1,760.2 | 103.1 s |
| B2 | 256 | 196 | 1,554.5 | 1,755.8 | 104.3 s |
| A4 | 512 | 98 | 2,044.7 | 2,578.4 | 76.6 s |

**steady u512 = 2,532.5 tok/s; steady u256 = 1,758.0 tok/s.**

**PRE-REGISTERED COST OF ubatch 256 ON PP = 30.58% -> DISQUALIFYING (>15%).**
Drift control: the two u256 arms agree to 0.25% and the two u512 arms to 3.63%,
~~and peak VRAM is identical at 3,265.56 MiB in all four arms~~ - so the session is
trustworthy and the result is not noise. Both A arms are below the recorded
reference, consistently. *(Corrected 2026-09-30, §9.4: the identical peak is
the load-time reading every run without `--residency-tg` reports, whatever its
arms allocate, so it controls nothing. The throughput agreement is the drift
control, and it stands on its own.)*

**Mechanism, from the per-phase telemetry (pass 3):**

| | u512 (98 chunks) | u256 (196 chunks) | per-chunk u512 | per-chunk u256 |
| --- | --- | --- | --- | --- |
| wall | 18,826.7 ms | 28,585.1 ms | 192.1 ms | **145.8 ms** |
| transfer | 35,672.1 ms | 54,281.9 ms | 364.0 ms | **277.0 ms** |
| compute | 18,584.6 ms | 27,290.8 ms | 189.6 ms | 139.2 ms |

Per chunk, u256 is **cheaper** (145.8 vs 192.1 ms wall) - but there are twice as
many chunks, so the total is 1.52x worse. **Halving the micro-batch buys only
24% off the per-chunk transfer cost while doubling the number of transfers.**
The layer-residency transfer bill is therefore dominated by a large
**per-(chunk x layer) fixed cost**, not by tokens staged, so micro-batch size is
close to the wrong knob for this design: it trades a 24% per-chunk saving for a
2x count.

**CONSEQUENCE: the two-session row cache has NO viable configuration in this
regime.** The two candidates are now both closed:
- **ubatch 512** - K=1 **refuses** with two sessions (§11.6.1) and K=2 is over
  the envelope by 330.39 MiB (§11.7.1);
- **ubatch 256** - K=2 + 1 complete layer package fits with 78.90 MiB spare, but
  costs **30.58% of prefill**, which is charter priority #1, under a rule fixed
  before the numbers were seen.

This is a **negative result that closes a question**, not a failed experiment:
the session verdict is PASS and the measurement is sound.

### 11.7.3 MEASURED 2026-09-29: where the tensors actually land, and two retractions

Run (owner request, 2026-09-29): the owner's exact 2-session argument set at
**`-lv 5`**, executed as `llama-fit-params` rather than `llama-server` because
the placement decision is entirely inside the fitter - same code, same
arguments, seconds instead of a 21 GB load.

| provenance | value |
| --- | --- |
| command | `llama-fit-params.exe -m <Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf> -ctk q8_0 -ctv q8_0 --flash-attn on --parallel 2 --batch-size 8192 --ubatch-size 512 -c 512000 -lv 5 --fit-target 16000 --load-mode none` |
| binary | `build/llama-cuda/bin/Release/llama-fit-params.exe` |
| telemetry | `build/llama-cuda/logs/static-place-fitparams.log` |
| verdict | `build/llama-cuda/verdicts/static-place-fitparams.done` - **PASS 0**, `gpu_pre=clean gpu_post=clean`, `llama.dll 2BEE822926D5111F` |
| mutex | `Global\AI-GPU-EXCLUSIVE-v1` acquired, pre-run probe clean |
| exit | 0 |

`LOG_LEVEL_DEBUG = 5` in `common/log.h:24` and the gate is
`verbosity <= common_log_get_verbosity_thold()` (`common/log.h:106`), so
**`-lv 5` is exactly the level that turns on the per-tensor placement lines** -
`-lv 4`, which both owner baselines used, cannot show them.

**Fit reproduced the owner's run exactly**: `-ngl 26`, "26 layers, 6674 MiB
used, 16318 MiB free", `target=6993`. (Deficit 1,785 MiB here vs 1,808 in the
owner's log - a free-memory probe difference between builds, not a fit
difference.)

**The placement, measured:**

| tensor class | bytes | device | evidence |
| --- | --- | --- | --- |
| `token_embd` | 540,344,320 (515.13 MiB) | **CPU, unconditionally** | `llama-model.cpp:1370-1371`: `pimpl->dev_input = { cpu_dev, ... }`, comment *"there is very little benefit to offloading the input layer, so always keep it on the CPU"* |
| `output` head | 417,177,600 (398.02 MiB) | **CUDA0** | index `n_layer_all` = 41, inside the `i_gpu_start`=16 .. 41 offload window, no `-ot` override |
| non-expert, `blk.16`-`blk.39` (24 layers) | ~913.9 MiB | **CUDA0** | 913.9 + 398.02 = 1,311.9 vs measured 1,311.95 |
| non-expert, `blk.0`-`blk.15` (16 layers) | ~609 MiB | CPU | below `i_gpu_start` = `n_layer_all + 1 - n_gpu_layers` = 16 |
| **all 120 expert-bank tensors, `blk.0`-`blk.39`** | 18,327,011,328 | **CUDA_Host** | `-ot "blk.16.ffn_(up\|down\|gate_up\|gate)_(ch\|)exps=CPU,..."`, and `buffer type overridden to CUDA_Host` on every `ffn_*_exps.weight`. **No override targets anything but CUDA_Host.** |
| `blk.40.ffn_{down,gate,up}_exps.weight` (MTP bank) | 452,984,832 (432.00 MiB) | **never loaded** | `W model has unused tensor blk.40.ffn_down_exps.weight (size = 150994944 bytes) -- ignoring`, x3 |

Layered shapes: `blk.34/38/39.ffn_down_exps` log at **210 MiB q6_K**, all
other banks at **144 MiB q4_K** - matching the byte table's Unsloth per-tensor
mix.

**RETRACTION 1 - the static class is 398.02 MiB resident, not 913.08.** §11.3
and §11.6 previously charged the full 957,530,112 B static class against the
weight envelope. It is not available: the embedding is host-side by hard-coded
design, not by budget pressure. Every "static" line in this document is now
**398.02 MiB**. Consequences, both favourable: the envelope holds **2** complete
layer packages, not 1 (§11.3); the K ladder admits **K=2** and puts K=1 at 9-18
resident layers, not 2-4 (§11.6).

**RETRACTION 2 - ">= 745 MiB of static on the host" and its TG hypothesis are
both withdrawn.** The bound was built on 26 layers x 44.0 MiB, which
double-counted the output layer as if it were a trunk layer. The output head is
resident, and it is the tensor read once per generated token; the embedding is
read once per prompt ubatch, ~1 MiB/token amortised at ubatch 512. **Neither
can explain the measured 22.39 tok/s, and the "material unexplained component"
claim is false.** The 22.39 tok/s stands unexplained by placement - the honest
candidate remains the 40 expert banks (540 MiB/token) and the 2,099.41 MiB
compute buffer, neither of which is a placement question.

**New standing fact - 432.00 MiB of the GGUF is dead weight.** The MTP
block's expert bank is reported unused and never loaded, so the effective
offloadable model is **40 expert layers, not 41**, and the 41-layer figures in
this document overstate expert traffic by 2.4%. Any per-layer accounting that
counts 41 expert banks is wrong.

### 11.7.4 MEASURED 2026-09-29: the graph split count is NOT what sets the compute buffer

Owner challenge to §11.7: the 70-splits figure was flagged earlier, so is the
split count the real driver rather than ubatch? **Tested and answered: no.**
Tool `tools/split-count-probe.ps1`, reading `graph splits` and the compute
buffer from the same log (`llama-context.cpp:703-706` emits the split count
from inside `sched_reserve`, immediately before the buffer sizes). All arms
under `tools/with-gpu-lock.ps1`.

| provenance | value |
| --- | --- |
| verdicts | `split-count-probe.done` **PASS 0**, `split-count-probe-lo.done` **PASS 0** |
| provenance | `digest 3EE2B9E87F42C375`, `llama.dll 2BEE822926D5111F` |
| telemetry | `build/llama-cuda/logs/split-count-probe.csv`, `splitsrv-u*.log` |
| shape | llama-server, `-c 512000 --parallel 2`, q8_0 KV, FA, fit-target 16000 (the §11.7 sweep shape) |

| ubatch | **splits (bs)** | splits (bs=1) | **CUDA0 compute (MiB)** | fit layers | fit used |
| --- | --- | --- | --- | --- | --- |
| 512 | **410** | 82 | 2,099.88 | 26 | 6,674 |
| 256 | **338** | 82 | 1,691.25 | 30 | 6,962 |
| 128 | **338** | 82 | 1,619.75 | 30 | 6,890 |
| 64 | **338** | 82 | 1,584.00 | 30 | 6,854 |

**The decoupling is in the saturation region.** From ubatch 256 down to 64 the
split count is **exactly constant at 338** while compute still falls
**107.25 MiB**. A quantity that does not change cannot explain a quantity that
does. Splits are **not** the driver.

**Both terms are real, which is why the first step looked like proof.** Over
512->256 the split count falls 72 and compute falls 408.63 - a tidy
**5.68 MiB per split** that is trivially suggestive with n=2. It is a
coincidence of the first step, not a law.

**The law that does hold is a per-token one, and it is exactly collinear.**
Taking the three points that share splits=338:

    compute = 1,548.25 + 0.5586 x ubatch        (MiB)

u64 -> 1,584.00, u128 -> 1,619.75, u256 -> 1,691.25. **All three lie on that
line to the decimal** - a three-point collinearity test that passes, not a
two-point fit. Predicted u512 from the per-token term alone is 1,834.3 against
a measured 2,099.88, an excess of **265.6 MiB that coincides exactly with the
72 extra splits (3.69 MiB each)**. So the defensible model is two terms:

    compute  ~  1,548 (fixed base)
             +  0.559 x ubatch          (per-token tensor size)
             +  3.69 x (splits - 338)   (per-split reservation, active > 256)

**Caveat, stated because it matters:** the per-split coefficient rests on a
single difference (one pair, 72 splits), so 3.69 MiB/split is a
one-degree-of-freedom estimate, not a fitted law. **Why the split term only
switches on above ubatch 256 is unexplained** - plausibly the scheduler begins
splitting *within* layers once the per-layer tensors no longer fit one
reservation, but no recorded arm isolates that.

**A third, independent shape agrees and is stronger still.** The layer-streaming
A/B (§11.7.2) reports **`graph splits = 122 (bs), 82 (bs=1)` at BOTH ubatch 512
and 256**, ~~with peak VRAM identical at 3,265.56 MiB~~ - while PP throughput moved
**30.58%**. In the runner's shape the split count explains **none** of the
change. `splits (bs=1)` is 82 in all six recorded cells, and the 2026-09-22
reference agrees at 122/82, so the bs=1 figure is a property of the model
shape, not a knob. *(Corrected 2026-09-30, §9.4: the struck VRAM clause is the
runner's load-time reading, identical on every run without `--residency-tg`.
It says nothing about the compute buffer at either ubatch. The split counts
come from the log and stand.)*

**Consequence: §11.7's conclusion stands and is now better grounded.** The
compute buffer is driven by ubatch (per-token tensor size), with a saturating
split term that only bites between 512 and 256. Halving ubatch from 512 to 256
is the *only* step that changes splits, and it is also the only step that buys
fitter capacity (26 -> 30 layers) - which is exactly why §11.7.2's -30.58% PP
result lands on that step and why the 800 MiB target stays out of reach.

### 11.7.5 MEASURED 2026-09-29: context buys the same capacity as ubatch, at ZERO PP cost

Owner question: since 512->256 is the only step that changes splits and the
only one that buys layers, can a context length *between* those buy the
capacity without the -30.58% PP cost? **Yes, and it dominates the ubatch
trade.** Tool `tools/split-count-probe.ps1 -CtxList`, ubatch held at **512**,
parallel 2, q8_0 KV, FA, fit-target 16000.

| `-c` total | per session | splits (bs) | compute (MiB) | **layers** | fit used |
| --- | --- | --- | --- | --- | --- |
| 512,000 (owner) | 256,000 | 410 | 2,099.88 | **26** | 6,674 |
| 480,000 | 240,000 | 357 | 1,972.97 | 29 | 6,969 |
| **472,000** | **236,000** | 338 | 1,940.22 | **30** | 6,921 |
| 468,000 | 234,000 | 338 | 1,925.89 | 30 | 6,880 |
| 456,000 | 228,000 | 338 | 1,876.77 | 30 | 6,742 |
| 444,000 | 222,000 | 285 | 1,599.13 | 33 | 6,961 |
| 432,000 | 216,000 | 266 | 1,556.38 | 34 | 6,860 |

**Largest `-c` that still fits >= 30 layers is 472,000 = 236,000 per session -
a 20,000-token (7.8%) context reduction, with ubatch 512 untouched, so the
prefill cost is zero BY CONSTRUCTION.** Same tool; verdicts
`ctx-for-layers-a.done` / `ctx-for-layers-b.done` **PASS 0**; the fitter logs
"context size set by user to N -> **no change**" in every cell, so none of the
saving is a silent fitter reduction - it is entirely the user's choice.

**The mechanism is not "context shrinks". It is the opposite, and better.**
Full class breakdown at the winning point (`splitsrv-u512-c472000.err.log`):

| class | `-c 512000` | `-c 472000` | change |
| --- | --- | --- | --- |
| CUDA0 model | 1,311.95 | **1,464.29** | +152.34 (4 layers x 38.08, the byte-table figure exactly) |
| CUDA0 KV | 3,187.50 | **3,428.69** | **+241.19** |
| CUDA0 RS | 75.38 | 87.94 | +12.56 |
| **CPU KV (host spill)** | **2,125.00** | **1,469.44** | **-655.56** |
| CUDA0 compute | 2,099.41 | 1,940.22 | -159.19 |
| total self | 6,674 | 6,921 | +247 |

**The context class went UP on the device while total context went down.** The
fitter frees budget by shrinking `-c`, and then spends it two ways: 4 more
resident layers **and 655.56 MiB of KV pulled back from the host onto the
GPU.** So the trade is not "less context" alone - it is "less host spilling",
which is the opposite of what the name suggests and strictly better for a
2-session decode.

**Side effect on the row cache: it does NOT reopen.** Envelope at the winning
point = 6,993 - 3,516.63 (context) - 1,940.22 (compute) = **1,536.15 MiB**.
static 398.02 + K=2 (1,093.0) = 1,491.02 fits with **45.13 MiB spare and no
room for any layer package**; static + K=1 (546.8) leaves 591.33, enough for
one complete package (470.08) with 121.25 spare - but K=1 refuses with two
sessions (§11.6.1). **K=2 is affordable at 236k/session but only barely, and
only if the design gives up layer residency entirely.**

**The trade, stated as the owner should see it:**

| option | per-session ctx | ubatch | layers on GPU | PP tok/s | KV on host |
| --- | --- | --- | --- | --- | --- |
| owner baseline | 256,000 | 512 | 26 | 2,532.5 | 2,125 MiB |
| **context trade** | **236,000 (-7.8%)** | **512** | **30** | **2,532.5 (unchanged)** | **1,469 MiB** |
| ubatch trade | 256,000 | 256 | 30 | 1,758.0 (**-30.6%**) | - |

**The context route gets the same 4 layers, keeps charter priority #1 intact,
and cuts host KV spill by 656 MiB.** Its price is 7.8% of per-session context -
the same class of decision as the rejected q4 KV (§11.8), so it is the owner's
call and is **not** recommended here. What it retires is the *necessity* of the
ubatch route: nothing in this regime now requires paying PP for capacity.

### 11.7.6 MEASURED 2026-09-29: K=2 token generation under an 8 GB-class budget - and the slot beats the cache

Owner direction: 236k/session is settled; run the K=2 row-cache TG test at 50k
scale. Driver `tools/run-tg-k2-budget-ab.ps1`, 2 arms per session (K=0 control
= `--residency-tg-retain` OMITTED, which is the documented off switch; K=2),
`--residency-budget-mib 6000`, 50k PP + 256 TG, `batch-major` (required: the
3-token cap at `main.cpp:4644` fires only for layer-major), 4 layer slots
unless stated, q8 KV, one mutex hold per session.

| slots | K | TG tok/s | PP tok/s | row_cache | outcome |
| --- | --- | --- | --- | --- | --- |
| 4 | 0 | **15.231** | 326.9 | off | ran |
| 4 | 2 | - | - | - | **REFUSED** `row_cache_depth` |
| 3 | 0 | 12.40 | 329.5 | off | ran |
| **3** | **2** | **14.24** | 327.8 | **1,093 MiB** | ran, **+1.84 tok/s (+14.8%)** |

`tg-k2-slots3.done` **PASS 0**, both arms `check-telemetry` clean,
`digest 3EE2B9E87F42C375`, `llama.dll 2BEE822926D5111F`, `gpu_pre/post = clean`.
Allocated cache 1,093 MiB - the recorded K=2 size to the MiB.

**The 4-slot refusal is the informative part, and it is honest.** At 4 slots the
50k prefill's bounded residency is 5,266,696,704 B (~~5,021.9~~ 5,022.7 MiB) of the 6,000
budget, leaving **1,024,759,296 B** for a cache that needs **1,146,093,568 B** -
short by 115.7 MiB. The refusal names the maximum outright: *"the largest depth
that fits is K=1"*. **Layer slots and row cache compete for the same budget**,
so depth is bought by dropping a slot. This also corrects the 2026-09-14 R2
record, which reported K=2 as the maximum at a 6,000 MiB budget: R2 ran a
**512-token** prefill (allowance 1,573,459,456 B), and a 50k prefill consumes far
more bounded residency, leaving less for the cache. **That record is shape-
specific, not a property of the budget.** *(Corrected 2026-09-30, §9.5: the
5,266,696,704 B charges 4 slots at 4 × 498.0 MiB, but the device holds 2 × 498.0
+ 2 × 432.0 = 1,860.0 MiB. Priced at the real allocation, the cache gets
1,163,154,944 B and K=2 fits with 16.3 MiB spare. The refusal is honest to
the code's arithmetic; it is not a structural limit. Expected, not run.)*

**And the result inverts the premise: at this budget the SLOT is worth more
than the CACHE.** 4 slots with no cache = **15.231 tok/s** is the best number in
this space. 3 slots + K=2 = 14.24. Buying the cache costs a slot, and the slot
is worth ~23% (15.23 -> 12.40) while the cache returns 14.8%. **The cache does
not pay for the slot it requires.** *(Corrected 2026-09-30, §9.5: it does not
require the slot. The slot-for-cache trade is forced by the over-charge, so the
comparison this table cannot make is 4 slots vs 4 slots + K=2. That arm is
unrun.)*

**What this still is NOT.** It is **one session**, not two: the runner hard-codes
`n_seq_max = 1` (§11.6.1) and has no session flag, so no recorded arm can give
2 x 236k TG. And the KV pressure is not in it either - `--ctx-size 52224`, not
472,000. The residency budget governs weight bytes only. So the two-session
constellation remains unmeasured. That gap is now the single largest unmeasured
thing in this document.

**The client side of that gap is one JSON field, not a feature (verified
2026-09-29).** The owner baseline's idle slot 0 is **not** a server limitation:
`id_slot` is a documented request field
(`tools/server/README.md:1151`), parsed at `server-context.cpp:4222`
(`json_value(data, "id_slot", -1)`) and honoured by `get_available_slot` at
`server-context.cpp:1494` (`ret = get_slot_by_id(task.id_slot)`). Sending
`id_slot: 0` and `id_slot: 1` on two simultaneous requests deterministically
fills both slots. The 22.39 tok/s owner figure is therefore a **one-active-slot
measurement, not a concurrency limit** - it does not mean two sessions cannot
be served, only that the client asked for one.

Recorded: `build/llama-cuda/logs/tg-k2ab-*.{jsonl,log,txt}`,
`tg-k2-slots3.done`, `tg-k2-budget-ab2.done` (the 4-slot refusal).

### 11.7.7 Three different objects are called "slot" - and the TG one is not optional

Owner challenge: *"we have 2 slots for layers on the prompt prefill, but on the
token generation we shouldn't have any slots out there. We should have just the
K buffer."* The premise is half right and the conclusion is wrong, and the
distinction matters because it decides where the weight bytes should go. Read
from source and `rowset_init` telemetry (2026-09-29).

**Three unrelated objects share the word:**

| object | where | what it is | size (measured) |
| --- | --- | --- | --- |
| **session slot** | `llama-server --parallel N` | one conversation | n/a - a queue entry, no VRAM |
| **layer/residency slot** | `--layer-slots N` | a device buffer the FFN **reads from** | **498.0 MiB** (`slot_window / n_slots` = 522,190,848 B) |
| **row cache, depth K** | `--residency-tg-retain K` | per-**logical-layer** retention store, all 40 layers | 546.8 (K=1) / **1,093.0** (K=2) MiB |

**CORRECTION 2026-09-29: a slot is 498.0 MiB, not the 432.0 MiB I stated in the
first pass of this subsection.** Two different numbers exist and I cited the
wrong one. `rowset_init.slot_bytes` = 452,984,832 B (432.0 MiB) is the
**row-set fetch unit** - a q4_K bank. The **allocation** is `byte_account`'s
`slot_window`, which is 1,566,572,544 B over 3 slots = 522,190,848 B = **498.0
MiB**, sized for the *largest* bank (the q6_K `ffn_down_exps` of layers
34/38/39). The byte account closes exactly:
`static + per_layer + slot_window + kv + state` = 957,530,112 + 1,597,483,520 +
1,566,572,544 + 557,056,000 + 65,863,680 = **4,744,505,856 B**, the reported
`bounded_window`. So `bounded_window` and `row_cache` are the two draws on one
residency budget, and a slot is 498.0 MiB.

And it is **aliased**: the source comment at
`llama-layer-streaming.cpp:9052-9057` states that *"a slot's bundles alias ONE
device buffer across every layer that shares it, so slot coverage can never
survive an intervening owner."* A slot is therefore a **reusable read buffer**,
not a retention store - which is exactly why `load_row_set` bypasses
slot-keyed bookkeeping on the retention path.

**The byte account also reproduces the 4-slot refusal to the byte** - the
strongest available check that the model of these budgets is right:

    4 slots -> static + per_layer + 4x522,190,848 + kv + state = 5,266,696,704 B
    6,000 MiB cap (6,291,456,000) - 5,266,696,704 = 1,024,759,296 B = 977.3 MiB
    K=2 needs 1,146,093,568 B (1,093.0 MiB) -> short 115.7 MiB
    K=1 needs 546.8 MiB -> fits, so "the largest depth that fits is K=1"

*(Corrected 2026-09-30, §9.5: this reproduces the code's arithmetic, not the
device. `4x522,190,848` is the preflight charge. The four slots actually
allocated are 2 × 522,194,944 + 2 × 452,988,928 = 1,950,367,744 B, so the check
confirms that the refusal follows from an over-charge of 132.0 MiB. It does not
confirm that the budget model is right.)*

**Two buffers on TG, yes - the owner was right to ask.** At 3 slots + K=2 both
are live and separate: `slot_window` **1,494.0 MiB** (3 x 498.0, complete banks)
and `row_cache` **1,093.0 MiB** (640 partial rows). The successful run's totals:
`bounded_window` ~~4,523.1~~ 4,524.7 + `row_cache` 1,093.0 = **5,617.7 MiB** against the
6,000 MiB cap, 382.3 MiB spare. `reserve` (512.0 MiB) sits outside the cap.

**So the cache does not replace the slot. They are serial stages.** Per window:
the complement is **H2D'd into the layer's private cache**, then *"the slot
bundle is then refreshed device-side for the whole union"* (D2D), and *"the FFN
still reads the slot with the router's own IDs."* Dropping the slot would leave
the FFN with nothing to read. `window_tokens` = 1 in the same telemetry, so this
is per-token, per-layer.

**What the challenge gets right**, and it is the useful part: on TG the layer
slot is *not* the PP rotation window. In PP a slot is a prefetch pipeline stage
(async H2D overlapped against compute). In TG there is no prefetch - the slot is
just a buffer you refresh. The owner's framing ("no slots on TG") describes the
*scheduling* role correctly, and that role genuinely is absent.

**The reframing that decides the budget: K is denominated in slot-equivalents.**
Pure-row arithmetic, 40 layers x 8K rows x 1,769,472 B, against one 498.0 MiB
slot:

| K | cache (rows only) | **slot-equivalents** |
| --- | --- | --- |
| 1 | 540.0 MiB | **1.08 slots** |
| 2 | 1,080.0 MiB | **2.17 slots** |
| 8 | 4,320.0 MiB | **8.67 slots** |

(Recorded allocations are 546.8 / 1,093.0 MiB - the delta is per-layer
alignment overhead, 13.6 MiB over 40 layers.)

*(Corrected 2026-09-30. Not every row is 1,769,472 B. Layers 34/38/39 carry the
q6_K down projection, whose expert row is 2,039,808 B. The allocation is
`K × 8 × (37 × 1,769,472 + 3 × 2,039,808) + 40 × 16,384` B. The fixed term is
one 4,096 B pad per layer plus one per member (src:7522-7538). That formula
reproduces every recorded size to the byte: K=2 1,146,093,568, K=4
2,291,531,776, and K=8 4,582,408,192, which is the "needs" in the refusal text.
So the delta is 6.19 MiB per K plus 0.625 MiB of pads, 13.0 MiB at K=2, and
none of it is alignment. Against a 498.0 MiB slot the cache costs **1.10 / 2.19
/ 8.78** slots at K=1 / 2 / 8. Against the 432.0 MiB slot a fourth slot adds
(§9.5) it costs **1.27 / 2.53 / 10.12**. The conclusion holds and gets
stronger.)*

**"Just the K buffer" is not the cheaper option - it is 2.17x the price at K=2,
and 8.7x at K=8.** This is the arithmetic behind §11.7.6's measured result
(4 slots + no cache = 15.231 tok/s beats 3 slots + K=2 = 14.24): the cache is
*retention across tokens*, the slot is *breadth across layers*, and at this
budget breadth wins.

**What the cache actually holds, corrected (owner question).** "K x 8 = 16, the
most recent 16 experts" is right *per layer* and wrong in two ways that matter.
(1) It is a **row-capacity LRU**, not a window union: `cap = min(n_expert,
K*W*top_k) = 8K` rows, recency-evicted, and rows the current window needs are
protected from eviction. (2) **There is no cross-layer sharing.** The cache is
keyed by logical layer and the router is per-layer, so layer 0's 16 rows are
*different experts* from layer 39's 16. K=2 is therefore **640 rows total
(16 x 40)**, not 16 experts seen "on all layers", and it gives **16 of 256
experts = 6.25% coverage per layer**.

Compare the two stores at the bytes they cost:

| store | rows | spread | per-layer coverage | hit on its layer |
| --- | --- | --- | --- | --- |
| K=2 cache, 1,093.0 MiB | 640 | 40 layers | **16 / 256 = 6.25%** | **0.439** (measured below) |
| 3 slots, 1,494.0 MiB | 768 | 3 layers | **256 / 256 = 100%** | **1.000** |

**Nearly the same row count and the same order of bytes, radically different
coverage.** And the cache can never close the gap at any affordable K: full
coverage of one layer needs `cap = 256`, i.e. K = 32, i.e. 10,240 rows =
~~**17.6 GiB**~~ **17.07 GiB** (18,327,666,688 B: every expert bank, 18,327,011,328 B
as in §11.7.3's placement table, plus 640 KiB of pads; corrected 2026-09-30). The cache is structurally a *partial* store; the slot is the
*complete* one. They are complementary in kind and competing only in bytes.

**MEASURED 2026-09-29: the K=2 hit rate is 0.439, and it is new.** `rowset_round`
reports `bytes_staged`, so the rate falls out of the traces already on disk. The
K=0 arm is the true-demand control: 139,824.0 MiB over 256 tokens x 40 layers =
**8.09 rows per window**. The K=2 arm staged 78,404.1 MiB = **4.54 rows per
window**, so the retention share is **1 - 4.54/8.09 = 0.439** - i.e. the cache
eliminated 43.9% of row traffic (61,419.9 MiB over 256 tokens). This sits
correctly between the recorded K=1 (0.3088) and K=6 (0.6461), and it is the
first K=2 hit rate measured on this model at this shape.

**The byte-efficiency verdict, which is the whole argument in one line.** Equal
slots isolate the cache: 3 slots + K=0 = 12.40, 3 slots + K=2 = 14.24 tok/s, so
the cache buys **1.84 tok/s for 1,093.0 MiB = 594.0 MiB per tok/s**. A fourth
slot buys **2.831 tok/s (12.40 -> 15.231) for 498.0 MiB = 175.9 MiB per
tok/s**. **The slot is 3.4x more byte-efficient than the cache** - and the
4-slot configuration is simultaneously **6.9% faster on 595.0 MiB less weight
memory** (1,992.0 vs 2,587.0 MiB). That is why the cache loses here, and it is a
coverage argument rather than a bandwidth artifact.

*(Corrected 2026-09-30, §9.5: the fourth slot is a 432.0 MiB slot, so four slots
cost 1,860.0 MiB and the fourth adds **366.0 MiB**. That is **129.3 MiB per
tok/s**, and the slot is **4.6x** more byte-efficient than the cache. Four slots
are **727.0 MiB lighter** than 3 slots + K=2 (1,860.0 vs 2,587.0 MiB). The traces
give 12.401 and 14.238 tok/s (`logs/tg-k2ab-a1-k0.jsonl`, `a2-k2.jsonl`,
`phase_summary` tg), so the gap to 15.231 is **7.0%**, not 6.9%. The verdict's
direction stands and its margin widens. Every rate here is a single arm, per
§11.7.10 item 3. The 4-slot session's own traces were overwritten by the 3-slot
session, which reused the same arm tags. Its 15.231 survives only in this
section.)*

**It also closes the 4-slot refusal exactly.** At 4 slots the 50k prefill's
bounded residency left **1,024,759,296 B (977.3 MiB)** for the cache. K=2 needs
**1,146,093,568 B (1,093.0 MiB)** - short by 115.7 MiB. K=1 needs 546.8 MiB,
which fits. Hence the refusal's *"the largest depth that fits is K=1"*, with no
slack in the reasoning. The budget is a single pool; slots and cache draw on
the same MiB, and at 2 x 236k that pool is 1,536.15 MiB (§11.7.5). *(Corrected
2026-09-30, §9.5: it closes the code's arithmetic exactly, and that arithmetic
over-charges four slots by 132.0 MiB. With the real 1,860.0 MiB the allowance is
1,163,154,944 B, so K=2 fits with 16.3 MiB spare. Expected, not run.)*

### 11.7.8a The whole mechanism in plain terms

Written at owner request, for a reader with no background in this system.

**The problem.** The model has 40 layers. Each layer holds a **bank of 256
"experts"** - think of them as 256 specialist mini-networks. Together that is
**18 GB of expert weights**, and the GPU has ~8 GB. So the weights cannot all be
on the GPU. They sit on the disk, in RAM.

**What a token actually needs.** For each layer, a little **router** looks at the
token and picks **8 of the 256 experts**. Only those 8 do any work. So for one
token you need 40 layers x 8 experts = **320 experts, about 565 MB** - even
though each expert is only 1.77 MB. Every token re-fetches roughly that much.
This is the "32x byte amplification" the project has recorded for years.

**The two buffers, as a workshop.**

    disk/RAM  --(1) PCIe, SLOW, ~8 GB/s-->  CACHE  --(2) on-GPU, FAST-->  SLOT  --(3)--> MATH
    (18 GB)                                 (filing cabinet)              (workbench)

- The **CACHE is a filing cabinet.** It holds expert weights you already fetched,
  filed by (which layer, which expert). It is smart about what to keep and what
  to throw away.
- The **SLOT is the workbench.** This is the only place the actual arithmetic can
  read from, because the compute graph was built pointing at it.
- Step **(2)** is the **device-to-device (D2D) copy**: moving something from one
  part of GPU memory to another part of GPU memory. It never leaves the GPU.

**Which one does the math run on? The SLOT. Always.** The cache is never read by
the arithmetic. It is read only by step (2). This is stated in the source at
`llama-layer-streaming.cpp:9050`: *"The FFN still reads the slot with the
router's own IDs."*

**Why not just point the math at the cache?** Because the two are shaped
differently. The cache is a compact, per-layer, throw-away-oldest-first store.
The slot is a full 256-position array indexed by absolute expert number, which is
the exact layout the compute graph was built to read. Getting from one shape to
the other is step (2).

**Why have both at all?** Because of *lifetime*:
- The **cache remembers.** The same expert often gets picked again on the next
  token. The cache remembers it, so step (1) can be skipped. That is the whole
  point - it is what made the hit rate 0.439, i.e. 43.9% of step (1)'s work
  disappeared.
- The **slot forgets instantly.** It is wiped and refilled for every layer, and
  shared between layers (that is what "n_slots" controls). It cannot remember
  anything, by design - that is what makes it safe to overwrite.

You cannot have only the cache (nothing to run the math on) and you cannot have
only the slot (it would re-fetch from disk 565 MB *per token*, every token).

**Is the D2D wasteful?** No - it is the cheap part. It moves 13.5 MB per layer
where step (1) moves 13.5 MB across PCIe. Same bytes, but D2D runs at hundreds
of GB/s versus ~8 GB/s, so it costs roughly **2% of what step (1) costs**. You
are paying a 2% cost to avoid a 100% cost. That is a good trade.

**So where is the actual waste?** The slot is reserved at **498 MB** because
during **prefill** it genuinely needs to hold a whole 453 MB bank. But during
**token generation** it only ever receives 8 experts = **13.5 MB**. So 97.3% of
the workbench is empty the entire time you are generating tokens. The workshop is
sized for the wrong job.

### 11.7.8 The two buffers are serial in TIME and simultaneous in MEMORY - and the slot is 97% dead

Owner question: *"we have slot buffer and a K buffer, but they're not being used
at the same time, right?"* **In time, yes - strictly serial. In memory, no -
both are occupied the whole time, and that is the finding.**

**The serial order, from source** (`llama-layer-streaming.cpp:9334-9392`),
per (layer, token) window:

1. **4a - H2D the complement into the cache.** Iterates `fetch` (the **missing
   rows only**); `bytes_staged += slab_bytes` here and nowhere else. A cache hit
   contributes **zero** PCIe bytes.
2. **4b - D2D refresh of the slot from the cache for the WHOLE union.**
   Iterates `uniq` (**all 8 rows, hit or miss**), unconditionally, writing
   `v_dst` at `uniq[i] * dst_t->nb[2]`. On-GPU, not PCIe.
3. `ev_copy_end` recorded, then a **host-blocking**
   `ggml_backend_event_synchronize(ev_copy_ready)`.
4. **FFN reads the slot.**

So the cache is written first, the slot second, the slot read third. Never
concurrently. **Their lifetimes differ, and that is why both are needed:** the
cache is keyed by logical layer and **persists across tokens** (that is the
0.439 retention share); the slot is aliased across layers and **overwritten
every layer** (~13.3 reuses per token at 3 slots).

**The prediction this makes, and the measurement confirms it exactly.** If the
cache only ever skips step 4a, its entire benefit must be in transfer and wait,
not compute. From the two 3-slot arms:

| | K=0 (12.401 tok/s) | K=2 (14.238 tok/s) | delta |
| --- | --- | --- | --- |
| transfer | 11,957.5 ms | 9,622.4 ms | **-2,335.1** |
| wait | 11,498.1 ms | 7,804.0 ms | **-3,694.1** |
| compute | 5,534.4 ms | 5,176.6 ms | **-357.8** |
| **peak VRAM** | **5,957.56 MiB** | **7,089.56 MiB** | **+1,132.00** |

**96.2% of the gain is transfer+wait; compute barely moves, exactly as the code
says.** And the cache cost **1,132.00 MiB of peak VRAM** - which is why 4 slots
+ K=2 refuses (§11.7.6).

**RETRACTION of the coverage argument I gave in §11.7.7.** I wrote that a slot
gives "256-of-256 experts (100% coverage) for 3 layers" and that this was why
the slot beat the cache. **That is wrong for the row-set TG path.** Step 4b
writes only the current window's `uniq` rows into the slot; the other 248 expert
rows of that layer's `experts` tensor are never written and never read (the
router selects only the top-8). The slot therefore holds **the same 8 rows the
cache would**, and its remaining capacity is untouched:

| | allocated | live working set | utilisation |
| --- | --- | --- | --- |
| 3 slots | ~~1,492.6~~ **1,494.0 MiB** (`graph_footprint.slot_vram_bytes` = 1,566,584,832) | 3 x 13.5 = **40.5 MiB** | **2.7%** |
| 4 slots | ~~1,992.0~~ **1,860.0 MiB** (1,950,367,744) | 54.0 MiB | ~~2.7%~~ **2.9%** |

*(Corrected 2026-09-30, §9.5: the 4-slot row used the preflight charge, 4 ×
498.0 MiB. The allocation is 2 × 498.0 + 2 × 432.0 MiB. The 3-slot row was a
rounding slip.)*

`slot_vram_bytes` is a **real device allocation**, not an accounting figure -
confirmed at 1,566,584,832 B here and 1,950,367,744 B in the gplr35/51/58 runs,
~~and corroborated by peak VRAM (3,265.56 MiB in `gplr35-s4-200k.log`, which is
consistent with ~1,859.8 MiB of slots on device plus KV and compute).~~
*(Struck 2026-09-30, §9.4: that peak is the load-time figure. It cannot hold the
run's own 3,910.0 MiB of KV, so it corroborates nothing. The allocation stands on
`slot_vram_bytes` alone. That field is the sum of `ggml_backend_buffer_get_size`
over the slot buffers (main.cpp:5216-5225), which is 1,860.0 MiB, not 1,859.8.)*

**So `n_slots` is NOT a capacity knob in row-set TG - it is a pipelining-depth
knob.** More slots means more independent buffers, so layer L's 4a/4b can
overlap layer L+n_slots' compute instead of serialising on
`ev_copy_ready`. That, not coverage, is why 4 slots (15.231) beat 3 (12.401).

**The actionable consequence, and it is large.** The slot is reserved at
full-bank width (498.0 MiB) because **PP's `load_layer` genuinely stages whole
banks** - but the row-set TG path only ever needs 13.5 MiB per slot. Sizing the
slot to its row-set working set when row-set is active would free **~1,452 MiB
at 3 slots** and **~1,938 MiB at 4**, against a 6,000 MiB budget. **The 4-slot
K=2 refusal (short 115.7 MiB) is an artefact of the reservation rule, not a
physical limit** - and at 2 x 236k the llama-server weight envelope is only
1,536.15 MiB (§11.7.5), so ~1,452 MiB is **essentially the entire envelope**.
This is a code change in the vendored experimental path, it reopens the row
cache at 2 sessions, and it is the highest-value untested item this session
produced. **The slot-NARROWING half (S2) is still NOT implemented and NOT
validated; only the correctness hardening around it (S1) has landed - see
§11.7.10.** *(Corrected 2026-09-30, §9.5: sizing the slots to the working set frees
**1,453.5 MiB** at 3 slots (1,494.0 − 40.5) and **1,806.0 MiB** at 4 (1,860.0 −
54.0). The 4-slot refusal has a second, cheaper cause as well: the preflight
charges 132.0 MiB more than the four slots occupy. Charging the real allocation
alone lets K=2 fit, with no slot narrowing (expected, not run). "It reopens the
row cache" also needs an order change: S2 as committed narrows the slots after
prefill, while the cache is sized before it, so the freed memory never reaches
the cache (§9.6).)*

### 11.7.9 The compact-slot fix: authorized and contracted; S1 landed, ~~S2 open~~ S2 committed 2026-09-30

Owner authorized the expert-ID remap direction on 2026-09-29 and directed that
GPU runs be held until the design is settled. Contract:
`docs/plans/gplr62-compact-rowset-slot-draft.md` (T-GPLR-62, `in-progress`).
**Correctness slice S1 is in the tree and validated (§11.7.10); the memory slice
S2 - the compact W-wide slot - is not started.** *(Updated 2026-09-30: S2 is
committed in the nested tree, 2298a7cf9 through 9b90025c4 (eight commits; with
the preceding bd076f145 the nested HEAD is nine commits past the outer pin
de39888). It belongs to the T-GPLR-62 workstream, which reports its
acceptance. Its first traces are measured in §9.6: the compact slot's saving
lands after the reported peak, and the W-aware preflight prices the whole run at
the TG size.)*

**The blocker, with the proof.** The slot cannot be narrowed without remapping
expert IDs, for three structural reasons, all read from the vendored source:
`make_dev_tensor` (`:1132`) copies the source shape so `ne[2] = n_expert = 256`;
`slot_bytes = slot_bytes_max + 4096` (`:3236`) is taken over those full-bank
tensors with **no width knob**; and the row-set write lands at
`uniq[i] * dst_t->nb[2]` (`:9372`) - **absolute expert ID** - which the FFN's
`MUL_MAT_ID` needs. Reducing `ne[2]` makes the router's absolute IDs index out
of range. `AGENTS.md:65` forbids altering expert IDs, so this required an
explicit owner amendment, which was given.

**A detail that made the design cheaper than expected - then partly
un-cheapened (§11.7.10).** PP and TG coexistence looked **already solved**:
`begin_handoff_partial` (`:7092-7120`) and the PP peak-order path (`:6838-6866`)
already free every `slot.buf` and `slot.buf_staging` before the handoff
allocation, and already emit a `handoff`/`release` event carrying `slot_bytes`.
**CORRECTED 2026-09-29: row-set TG suppresses the handoff (recorded
`handoff auto -> off`), so that release never fires and a separate PP->TG
transition IS still required.** The release machinery is a good model to copy,
not a mechanism to reuse. My "two pools vs handoff resize" framing was also
wrong: the answer is a third thing - a compact pool allocated at its own
PP->TG transition.

**What it buys (derived from measured per-expert bytes, NOT yet measured
end-to-end).** Largest per-expert is the q6_K bank:
`522,190,848 / 256 = 2,039,808 B`.

| W | per slot | 4 slots | freed vs ~~1,992.0~~ **1,860.0** MiB |
| --- | --- | --- | --- |
| 8 | 15.56 MiB | ~~62.25~~ **58.1 MiB** | ~~1,929.8~~ **1,801.9 MiB** |
| **16** | **31.12 MiB** | ~~124.50~~ **116.3 MiB** | ~~1,867.5~~ **1,743.8 MiB** |

~~**1,867.5 MiB is 1.22x the entire 2 x 236k weight envelope (1,536.15 MiB).**~~
**1,743.8 MiB is 1.14x the envelope.**
At 4 slots, W=16, 6,000 MiB cap, `bounded_window` falls 5,266,696,704 ->
**3,308,481,024 B**, leaving **2,844.8 MiB** for the cache - so ~~**K=2 fits (it
refuses today) and K=4 (2,160.0 MiB) becomes possible, which was impossible at
any K before.**~~

**CORRECTED 2026-09-30 (§9.5, §9.6).** Four corrections:

- *The table priced every slot at the q6_K bank.* That is the preflight's
  charge. The runner sizes each compact slot to the largest W-wide package in
  its own rotation (src:9434-9457, allocated at 9463). At 4 slots, two slots
  hold a q6_K layer and two do not. So the per-slot figure fits only the two
  big slots, and the 4-slot column is 60,948,480 / 121,896,960 B. Those are the
  S2 runs' own byte reports (`audit-w8`, `a9-w16-final2`). The freed column
  also used the 1,992.0 MiB charge as the full-bank allocation. The device
  holds 1,860.0 MiB (fourteenth finding).
- *"It refuses today" is the 132.0 MiB over-charge, not the slot's size.*
  Priced as allocated, 4 slots + K=2 fits at W=0 with 16.3 MiB spare
  (fifteenth finding).
- *The preflight accepts more than K=4.* The 2,844.8 MiB allowance admits K=5,
  at 2,864,250,880 B. K=4's cache is 2,291,531,776 B (2,185.4 MiB, the S2
  traces' `byte_account.row_cache`). 2,160.0 MiB priced every row at the Q4_K
  per-expert size.
- *None of it is realizable as ordered.* The cache is allocated before
  prefill, with the full-bank slots live, and the compact swap comes after
  prefill. A K=5 run in this shape would hold 7,622.3 MiB through prefill, which
  is 1,622.3 MiB over the cap (expected, §9.6). The allowance is the
  preflight's arithmetic, not memory the run has. Collecting it needs the cache
  to be allocated after the swap (§9.6, fix 2).

**W is the worst-case union, never the mean** - SDD §4 already demands it, and
§4's loud-under-coverage clause already guarantees a too-narrow window *refuses*
rather than reading a wrong expert. Measured worst case: 8 (one session), 16
(two batched, mean 15.34, max 16).

**The two acceptance criteria that actually matter are A3 (route parity) and A4
(answer stability), not the throughput ones.** A1/A2/A5/A6 could all pass on a
build that silently computed the wrong experts. The contract pre-registers that
a failed A3 or A4 stops the task, with no acceptable throughput number buying it.

**Feasibility re-checked against the real graph code 2026-09-29
(§4.1 of the contract). Placement is legal and the reduction order is safe:
the `ggml_add` aggregate is a left-fold over *positions* in `selected_experts`,
not over id values, so a position-preserving remap is bit-identical; and
`MUL_MAT_ID` sorts ids internally and reads `n_experts` from the tensor's
`ne[2]`, so a W-wide slot is a first-class shape. **The escalation condition did
not fire.**

**But the re-check found a real defect in my own design, and it is the sharpest
argument in this document.** `selected_experts` is **overloaded**: the same
tensor is consumed by `ggml_get_rows(ctx0, probs, selected_experts)` -
where `probs` is the full `[1, n_expert, n_tokens]` probability vector indexed by
**absolute** expert id - as well as by every `build_lora_mm_id` on the slot
weights. A single remapped tensor passed to `get_rows` would return
`probs[0..W-1]`, i.e. **the probabilities of experts 0..W-1 instead of the
routed ones**, and because `get_rows` sorts and inverts internally the result
would have the right shape, the right order, and entirely wrong values. Every
expert silently misweighted, with a plausible-looking answer. So the gather must
**split** the index tensor, not replace it, and `build_moe_ffn` needs one gated
parameter (`selected_experts_weight`, defaulting to null). That is an edit to
`src/llama-graph.cpp`, shared by every MoE architecture. **My earlier
description of this task as "a lookup table and a graph edit inside the
experimental path" was too light, and is corrected.**

**Consequence for the gates: A8 was added** - `selected_experts_abs` must be
bit-identical to a non-row-set run's routing weights. Route parity and answer
stability could both survive a small uniform misweighting; A8 is the only
criterion that isolates this specific defect.

**Adversarial review of the T-GPLR-62 gates found three defects in them
(2026-09-29), and the first is serious.** `tools/check-route-parity.py` accepts
only `traces`, `--json` and `--quiet` - **there is no `--ref`, no baseline, no
golden** (its own docstring says "no reference stream"). It checks that under
`--pp-repeat 2` each route tuple appears **exactly twice**, i.e. **pass 1 against
pass 2 of the same build**. **A build that computes consistently wrong experts
produces consistent routes and passes A3 trivially.** Route parity is a
*dETERMINISM* gate; the compact-slot defect class is *deterministic*. So A3,
which my first draft made load-bearing, **cannot detect the failure it was
written for** and is demoted to cheap smoke.

**A9 replaces it: a slot-content recon.** For every (layer, window) and every
union position `i`, byte-compare the staged slot bytes against the checkpoint's
bytes for `uniq[i]`, per member. That is a direct proof of the mapping rather
than a consistency observation, and the recon machinery already exists
(`recon_begin_bank` / `recon_end_bank`); it has simply never been pointed at the
row-set path. **A10** adds the subset assertion described below.

**A1 was checking the wrong field** - `rowset_init.slot_bytes` is
`slots[0].layer_bytes.at(0)`, **layer 0's package bytes**, carrying the comment
*"one row-set slot == one existing full-bank-shaped slot buffer"*
(`:8960-8963`). It would **not change when the implementation is correct**. The
allocation is `graph_footprint.slot_vram_bytes`. This is the second time this
session the `rowset_init`-vs-real-allocation confusion has bitten; it is now
called out in the contract by name.

**The concrete silent-wrong-expert mechanism is a stale LUT entry.**
`union_pos` has 256 entries and only `|uniq|` are written per window, so an
entry for an expert outside this window's union **still holds the previous
window's value - a valid-looking position pointing at a different expert.** The
required hardening, in order: re-initialise the **whole** table to `-1` every
window; any `-1` lookup **refuses loudly** (never clamp, never wrap, never fall
back); and assert the device id set is a **subset** of the host union, because
`read_router_ids` computes the top-k **on the host** from device-read `probs`
while the graph independently computes `ggml_argsort_top_k` **on the device** -
two independent top-k implementations. **A10 firing is a pre-existing
correctness question, not a compact-slot regression, and the contract says stop.**

**A5's throughput floor is a guess, not a prediction** - `ne[2]` feeds
`ggml_cuda_should_use_mmq`, so W=16 may pick a different kernel config than
W=256. And **A4 is necessary but insufficient**: the strict gate is a
substring/regex needle check, and eight wrong experts out of 256 degrade text
without necessarily removing the needle. Recorded honestly rather than papered
over.

**VMM, recorded so it is not re-litigated.** CUDA VMM reserves a large *virtual*
address range and maps *physical* handles only over the slices actually written.
It would keep tensor shapes and expert IDs exactly as they are - no invariant
change at all - and free the same ~1,867 MiB. It was set aside because it means
writing raw `cuMemAddressReserve` / `cuMemCreate` / `cuMemMap` / `cuMemSetAccess`
into the vendored tree, reaching around ggml's buffer abstraction for a context
it does not own, and because it costs one unmap+map per (layer, window) - 10,240
remaps over a 256-token run - whose driver overhead on a WDDM consumer part is
unmeasured and could plausibly exceed the ~2% D2D cost the design protects. The
remap is a lookup table plus a graph edit inside an opt-in experimental path;
VMM is a new memory subsystem.

### 11.7.10 SLICE S1 LANDED: the 256-token TG noise floor is +/-11-13%, and it breaks an earlier conclusion

Owner authorised implementing T-GPLR-62. **The correctness half is in the tree
and validated; the memory half is NOT started.**

**Landed** - `src/llama-layer-streaming.cpp`, 31 lines, two hunks: a
**post-placement containment self-check** (every union member must name a
source, else loud `coverage_gap` refusal - the exact condition under which 4b
would read a position it never wrote, i.e. compute with the WRONG expert), and
the **4b `GGML_ASSERT` converted to a structured refusal** (an assert aborts the
process with no telemetry and no clean exit, which is not what SDD §4's
under-coverage clause asks for). Build clean; the **DLL** changed, not the exe.

**Validation: 3 builds x 2 arms, 6 runs.** All six: exit 0, `check-telemetry: OK`
with an identical 51,598-event count, **zero `rowset_refused`**, and **peak VRAM
byte-identical across all three builds** (5,957.56 / 7,089.56 MiB). So: **no
regression, no spurious refusal, and no memory change - the compact slot that
would move VRAM is not implemented.**

**The finding that matters more than the change: the measurement is underpowered.**

| build | K=0 TG | K=2 TG |
| --- | --- | --- |
| control | 15.10 | 14.60 |
| qwen35moe only | 12.32 | 17.70 |
| S1 (mine) | 15.21 | 18.83 |

**Identical event counts and identical peak VRAM, so the only variable is
timing: that is +/-11-13% run-to-run variance on a single 256-token arm.**
Therefore:

1. **I am not claiming S1 improved throughput.** The apparent +29% sits inside
   single-arm noise. Claiming it would be the exact error this session's
   adversarial review was commissioned to prevent.
2. **A5's ">= 14.24 tok/s" floor is not a usable gate** - it is inside the noise
   band.
3. **This undercuts §11.7.6's headline.** "4 slots 15.231 beats 3 slots 12.40"
   is a +22.8% single-arm gap, the same order as this noise floor. **The claim
   that the layer slot beats the row cache is NOT established by the recorded
   evidence** and needs repeated arms. The 2.7%-utilisation finding is
   unaffected - that is a byte count, not a timing.

**A confound found and named:** `src/models/qwen35moe.cpp` carries a 4-line
uncommitted change from an earlier session (`ggml_set_output` on the router's
selection probs, to keep a diagnostic leaf alive for an AER readback). **Its
mtime is Sep 25 13:11; the recorded binaries are Sep 25 12:29 - so it is in
NONE of the recorded numbers in this document**, and it changes graph structure.
It is preserved untouched and should be committed or reverted before any
performance claim is made against a recorded baseline.

**A retraction of my own doubt:** I said mid-session that the provenance gate
reported an identical DLL hash for two different DLLs. **That was a misread of
the `-ValidateOnly` dry run**, which ignores `-BinDir`. The real runs
distinguish perfectly (control `2BEE822926D5111F` vs S1 `925559792BF3FEE1`), and
the gate **REFUSED** the third build's arms for a real staleness fault I had
introduced. **The gate is sound; the doubt is withdrawn.**

**And one piece of good news for S2:** `layer(il)` returns `ls->slot_layer(il)`
when streaming is active (`src/models/qwen35.cpp:142`), so **the graph is already
built against the slot bundle** - narrowing the bundle's expert tensors to
`ne[2] = W` narrows `MUL_MAT_ID`'s index space with **no arch-builder change**.
That is materially smaller than the contract assumed. The PP->TG re-point is
still needed, and the handoff release at `:6838`/`:7092` does **not** fire under
row-set TG (`handoff auto -> off`), so a separate transition is required - the
"reuse the existing release" claim in §11.7.9 is **corrected**.

### 11.7.11 Cross-thread isolation: real for files, FALSE for measurement (checked 2026-09-29)

Owner asked whether the concurrent **MTP Usage Predictor** worktree
(`5e3cb7b5`) is genuinely isolated from this one, given the UI's "master
isolated" label. **Files: yes. Measurements: no.**

**What IS isolated (verified, not assumed):**

| resource | this worktree | MTP worktree | isolated? |
| --- | --- | --- | --- |
| worktree | `b947db99-...` | `5e3cb7b5-...` | yes |
| `third_party/llama.cpp` | own clone, own `.git` | own clone, own `.git` | **yes** |
| build source (`CMAKE_HOME_DIRECTORY`) | main tree's clone | **its own clone** | **yes** |
| `build/` | main tree's `build/llama-cuda` | its own `build/llama-cuda` | **yes** |
| uncommitted files | 2 | **0** | no overlap |

`third_party/llama.cpp` is a **gitlink (index mode 160000)**, so each worktree
materialises its own clone. The other worktree's `CMakeCache.txt` proves its
build compiles its own source, not the main tree's. **Neither thread can clobber
the other's source or binaries.**

**What is NOT isolated - and this is the part that matters:**

1. **The two clones are on DIVERGED histories.** The parent repo's submodule pin
   is `de39888e` ("layer-streaming: harden D1 observation lifecycle", 2026-09-25),
   which is what this thread builds. **The MTP worktree's clone is at
   `06a892fb`, which does not exist in this clone's history at all** -
   `git merge-base --is-ancestor` says it is not an ancestor, and
   `git log` cannot find it here. **The two threads are not running the same
   engine, so no tok/s figure is comparable across them.**
2. **The submodule pointer is shared state.** The MPT worktree's clone is
   *off-pin*. If that worktree ever commits the gitlink, it moves the pin for
   everyone.
3. **The GPU is the only genuinely shared resource**, serialised solely by the
   `Global\AI-GPU-EXCLUSIVE-v1` mutex. Idle at the time of check. The
   AGENTS.md failure mode - a non-conforming project holding ~16.5 GiB *during*
   a mutex-held run - applies to both threads equally, and neither can detect
   the other except through the wrapper's pre/post `nvidia-smi` probe.
4. **The model file** `C:/Models/...` is shared read-only. No conflict.

**A caveat about MY OWN isolation, which is weaker than the label suggests.**
This worktree's `third_party/llama.cpp` is an **empty directory (0 entries)**
and it has **no `build/`**. That is why the S1 edit landed in the **main
tree's** checkout at `C:/Development/AI/qwen38GZI/third_party/llama.cpp` and not
in the worktree. So my vendored changes sit in the *shared main checkout* - the
least isolated location available. They are safe from the MTP thread (separate
clone), but they are **not** protected by my worktree, and anything run from
the main tree sees them.

**Practical rule for both threads: compare tok/s only within a thread, and only
against a control built from that thread's own clone.** A cross-thread
comparison would differ by an unknown engine delta on top of the +/-11-13%
single-arm noise floor already measured in §11.7.10.

### 11.7.12 How the gitlink actually works here - and three hazards it creates

Owner asked whether editing `third_party/llama.cpp` "would reflect on all the
rest as well". **It does not.** The distinction matters, so it is written down.

**A gitlink is a bookmark, not a copy.** Normal git tracks file *contents*. A
gitlink is the one exception: a single index entry, **mode `160000`**, whose
content is a **commit hash** and nothing else. This repo's parent index records,
in full:

    160000 de39888e3d8d0ed5075ac2d75f1b80518f196658 0  third_party/llama.cpp

**Forty characters. Not the ~10,000 llama.cpp files.** That is why this
worktree's `third_party/llama.cpp` is an **empty directory** - git never
populated it, because the parent has no file contents to give it.

**The lifecycle:**

1. Someone commits in the **submodule**: a new commit in the llama.cpp repo.
2. Someone commits the **gitlink** in the parent: the recorded SHA changes.
3. Only then does anyone else see it - and only when they run
   `git submodule update` (or re-clone with `--recurse-submodules`).

**So: editing files inside `third_party/llama.cpp` is entirely local.** Each
clone has its own working files and its own object store (verified: independent,
no `objects/info/alternates`). Editing never propagates. This session is the
proof - 2 files modified in the main tree's clone, **0** in the MTP worktree's
clone, parent untouched. **The only thing that propagates is the recorded
pointer, and only through an explicit commit in BOTH repos.**

**Hazard 1 - the push remote is UPSTREAM.** The clone's `origin` is
`https://github.com/ggml-org/llama.cpp.git` for **fetch AND push**.
`AGENTS.md` states: *"Do not create an upstream pull request, push, or
write upstream-facing material."* A `git push` run from inside that directory
would aim at **ggml-org**, not at anything of the owner's. This is the sharpest
practical consequence of the tree being an independent repo with a real remote,
and it is not obvious from the path name.

**Hazard 2 - there is no `.gitmodules`.** The parent records a commit but
declares **no URL**, so `git submodule status` returns
`fatal: no submodule mapping found in .gitmodules for path
'third_party/llama.cpp'`, and `git submodule update --init` **cannot
reconstruct the tree from the repo alone.** The clones exist because something
outside the repo materialised them. The pin is recorded; the source is not.

**Hazard 3 - the pin is already inconsistent.** The parent records
`de39888e`; the MTP worktree's clone is at `06a892fb`, which is **not present in
the main clone's history at all**. Two clones of the "same" bookmark, pointing
at commits with no common line of descent, and the parent has no way to know.

**Practical rule: a change to `third_party/llama.cpp` does NOT reflect to other
worktrees, and that is why §11.7.11's caveat exists** - the S1 edit is invisible
to the MTP thread, and equally it is not protected by this worktree, because it
lives in the shared main checkout.

### 11.7.13 Push-to-upstream closed on the ONE clone that had it (2026-09-29) - and a new exposure found

Owner asked to remove the upstream push remote. **Done, and the scope is narrower
than §11.7.12's hazard 1 implied - corrected here.**

| clone | fetch | push |
| --- | --- | --- |
| **MAIN** `third_party/llama.cpp` | `github.com/ggml-org/llama.cpp.git` | **`DISABLED`** (was upstream) |
| MTP `5e3cb7b5` | `C:/Development/AI/qwen38GZI/third_party/llama.cpp` | same local path |
| AFR `b78d8fc7` | same local path | same local path |
| AER `qwen38-aer-worktree` | `.../b78d8fc7.../third_party/llama.cpp` | same local path |

**The upstream push URL existed ONLY in the main clone** - the one this thread
builds and edits. The other three push to **local paths**, a deliberate
Freebuff arrangement where per-worktree clones consolidate back into the main
clone. So §11.7.12's hazard 1 was correct for the clone in question and
**over-generalised to "the vendored tree"**; it is now scoped. **The three local
push remotes are intentional and were left untouched** - disabling them would
break consolidation and they carry no upstream risk.

**What was changed:** `git remote set-url --push origin DISABLED` in the main
clone. This is the right instrument rather than `git remote remove origin`:
**fetch still works** (the fetch URL is untouched, so the vendored tree can
still be updated), and push now fails **before any network I/O**:

    $ git push --dry-run origin research/layer-residency
    fatal: 'DISABLED' does not appear to be a git repository
    exit 128

Pre-checks confirmed no bypass: no explicit `remote.origin.pushurl`, no
`remote.pushDefault`, no `push.default`, no `url.*.pushurl` redirect, and
`origin` is the only remote. Exit 128 (the earlier `exit=0` reading was
`head`'s status, not git's).

**NEW EXPOSURE this reveals, and it is the more interesting half.** The three
local remotes **push INTO the main clone's working tree** - which is where the
uncommitted T-GPLR-62 **S1 edit lives**. So an incoming push from another
thread lands on top of uncommitted local changes there. That is not a
hypothetical: it is the mechanism by which the other threads' work consolidates.
**Practical consequences:** (a) do not leave long-lived uncommitted work in the
main clone if other threads are pushing into it - commit it or move it; (b) S1
is currently exposed, which is a further argument for either committing it or
relocating the edit into a worktree that owns its own clone.

### 11.8 The q4 KV lever - OWNER REJECTED 2026-09-29, do not propose again

**Disposition (owner):** q8_0 KV is the required setting for coding/agent work;
q4_0 is not acceptable. The lever below is retained only as recorded history and
**must not be re-proposed as a way to buy VRAM.** Every later section is
computed at **q8_0 KV fixed**.

The arithmetic, for the record: q4_0 K/V rows are 0.625 B/element against
q8_0's 1.0625 B/element (544 B per 512-element row recorded in this run's own
fit), so a q4 KV cache is ~59% of the q8 one: 3,187.5 -> ~1,875 MiB, freeing
~1,312 MiB. T-Q38HG-01 verified q4 KV through the streaming path (741.04 tok/s
at 100k, all 5 answer items PASS), so the plumbing exists; it is simply not on
the table.

This is not speculative plumbing: `--kv-cache q4` is an existing runner flag,
and T-Q38HG-01 already verified q4 KV through the streaming path (741.04 tok/s
at 100k, all 5 answer items PASS). The interaction to test is therefore
**q4 KV x row cache depth at 2 sessions**, which no recorded arm covers.
**[SUPERSEDED by the owner's 2026-09-29 disposition above: q4 is rejected, so
this cell is not a task.]**

### 11.9 What this session should now hold to

1. At 2 sessions the fitter **cannot fit even the dense weights plus KV**; it
   reports a 1,808 MiB deficit and spills 2,125 MiB of KV to the host.
2. **The budget is 6,993 MiB** (the fitter's own `target=6993` line), not the
   7,119 I first used. Spare is **319 MiB nominal / 297 MiB physical**; the
   1,491 MiB "unaccounted" is CUDA-context overhead outside the ledger, not a
   charge against headroom.
3. **MEASURED 2026-09-29 (§11.6.1): K=1 cannot serve two concurrent sessions.**
   Batched (what `--parallel 2` does) the per-layer demand is the UNION of both
   top-8 sets - mean 15.34, max 16, against a cap of 8 - and the runner
   **refuses loudly on 99.99% of steps**. Time-sliced (which llama.cpp does not
   do) the hit rate collapses to **0.0341**, a 9.1x fall from the validated
   0.3088, not the ~0.15 predicted. **K=2 is the structural minimum** for two
   sessions. **At ubatch 512 no row-cache configuration works for two sessions;
   at ubatch 256, K=2 + 1 complete layer package fits with 78.90 MiB spare.**
   Two independent measurements converge on that one cell.
4. **The weight envelope is 1,630.71 MiB** and the stock fitter spends
   1,311.95 of it. It fits **two complete layer packages** (226-292 MiB spare)
   and **not three at any mix**. [corrected 2026-09-29 - the earlier "one" was
   an artefact of charging the embedding against the budget; see §11.7.3]
   **NOTE: this is the WEIGHTS-ONLY capacity. It is not a 2-session row-cache
   capacity - see §11.6.1, where K=1 is shown to refuse with two sessions.**
5. **The 318.76 MiB of unused weight budget buys 7-8 more non-expert layers
   (44.0 MiB each, measured off the fitter's ladder) and zero expert banks
   (smallest = 432.0 MiB).** So the answer to "how many weights can we fit" is
   26 -> 33 shallow layers, or one whole layer - not both.
6. The row-set path with the cache **off** (14.2 MB transient) is the
   configuration that fits, and it is already implemented.
7. **`q4` KV is REJECTED by the owner (2026-09-29) - q8_0 is required for
   coding/agent work. Do not re-propose it.** Per-session context is **settled
   at 236k** (owner, 2026-09-29), i.e. the 7.8% reduction is accepted, so
   128k is moot.
8. **The two-session row cache is CLOSED, on three independent counts.**
   (a) **K=1 refuses** with two sessions - 99.99% of batched decode steps
   demand more than the 8-row cap (§11.6.1). (b) **K=2 costs a layer slot**,
   and at an 8 GB-class budget the slot is worth more than the cache:
   4 slots + no cache = 15.231 tok/s beats 3 slots + K=2 = 14.24 (§11.7.6).
   (c) **K=2 at 4 slots refuses outright** for want of 115.7 MiB. K=2 is the
   structural minimum depth and it still does not pay here.
   *(Corrected 2026-09-30, §9.5: (c) is the preflight's 132.0 MiB
   over-charge, not a shortage. Priced as allocated, 4 slots + K=2 fits with
   16.3 MiB spare (expected, never run). That also removes (b)'s premise in
   §11.7.6's shape: K=2 need not cost a slot there. So the closure no longer
   holds as stated, because it rested on (b) and (c). What remains is (a), which
   sets the minimum at K=2, and item 13's envelope arithmetic: K=2 fits at 2 x
   236k with 45.13 MiB spare, but leaves no room for a layer package. Whether
   K=2 pays for two sessions is unmeasured, and item 18's harness gap blocks
   measuring it. (b)'s 15.231 vs 14.24 remains two single arms inside
   §11.7.10's noise band.)*
9. **MEASURED 2026-09-29: `--ubatch-size` is the only compute-buffer lever;
   `--batch-size` does nothing.** 2,099 -> 1,691 (256) -> 1,619 (128) -> 1,584
   (64). **Max freed 515 MiB; the 800 MiB target is unreachable (285 short).**
   The fitter saturates at `-ngl 30` below ubatch 256, so only the 512->256
   step converts into anything: **4 more layers, 26 -> 30.** The freed memory
   is spent on layers, not banked as headroom.
10. **RETRACTED: "1,671.88 MiB is the price of the second slot."** The compute
    buffer is **context-dominated**. At fixed `-c 256000`, `--parallel 2` is
    **125 MiB CHEAPER** than `--parallel 1` (1,076 vs 1,201). Compute tracks
    n_ctx near-linearly (627 / 1,201 / 2,349 at 128k / 256k / 512k, p=1).
11. **MEASURED 2026-09-29 (§11.7.2): ubatch 256 costs 30.58% of 50k prefill -
    DISQUALIFYING under the pre-registered band.** 2,532.5 -> 1,758.0 tok/s
    steady, ABBA 4 arms, one mutex hold, all verdicts PASS, drift 3.63%,
    ~~peak VRAM identical~~ (§9.4: on these runs that field is the load-time
    sample, not a peak). The recorded 2,759 tok/s reference is a
    **cross-binary** number (exe FE3B0A108C1D9BC9 / dll 8A9F3D83F24D0FEC vs
    today's C12FE2C084EFA339 / 2BEE822926D5111F) and was not used as the
    control. Mechanism: halving the micro-batch cuts per-chunk transfer only
    24% while doubling the transfer count, so the bill is dominated by a
    per-(chunk x layer) fixed cost.
12. **MEASURED 2026-09-29 (§11.7.4): the graph split count is NOT the compute
    driver.** Splits are constant at 338 from ubatch 256 down to 64 while
    compute still falls 107.25 MiB. The law that holds is
    `compute ~ 1,548 + 0.559*ubatch + 3.69*(splits-338)`, and the per-token
    term is exactly collinear across u64/u128/u256. Splits only change between
    512 and 256 (410 -> 338) - the one step that also buys fitter capacity
    (26 -> 30 layers). The runner shape shows 122/82 splits at BOTH ubatch
    values ~~with identical peak VRAM~~ while PP moved 30.58% (§9.4: the peak
    field on these runs is the load-time sample).
13. **MEASURED 2026-09-29 (§11.7.5): CONTEXT buys the same capacity as ubatch
    at ZERO PP cost.** At ubatch 512, `-c 472000` = **236,000 per session
    (-7.8%)** fits **30 layers** vs the baseline's 26, keeps prefill at
    2,532.5 tok/s, **and pulls 655.56 MiB of KV back from the host to the
    device** (host spill 2,125 -> 1,469 MiB). The fitter logs "no change" to
    `-c` in every cell, so the saving is the user's choice, not a silent
    reduction. **This retires the necessity of the ubatch route - nothing in
    this regime now requires paying PP for capacity.** It does NOT reopen the
    row cache: the envelope there is 1,536.15 MiB, so K=2 fits with 45.13 MiB
    spare and no layer package at all. Whether 7.8% of context is acceptable is
    an owner product decision (same class as the rejected q4).
14. **MEASURED 2026-09-29 (§11.7.6): K=2 TG under an 8 GB-class budget.**
    3 layer slots + K=2 = **14.24 tok/s** vs K=0 at the same slots = 12.40
    (**+14.8%**); cache allocated 1,093 MiB. **But 4 slots + no cache = 15.231
    is better than either** - at this budget the layer slot beats the row
    cache, because the cache has to be bought with a slot. At 4 slots K=2
    **refuses** (needs 1,146,093,568 B, only 1,024,759,296 B left after a 50k
    prefill's ~~5,021.9~~ 5,022.7 MiB of bounded residency), naming K=1 as the maximum.
    This corrects the 2026-09-14 R2 record: K=2-as-maximum at 6,000 MiB was true
    for a 512-token prefill, not for 50k. *(Corrected 2026-09-30, §9.5: the
    refusal follows from the preflight's 132.0 MiB over-charge. Priced as
    allocated, 1,163,154,944 B are left and K=2 fits with 16.3 MiB spare. So at
    4 slots the cache need not be bought with a slot (expected, never run).)*
15. **MEASURED 2026-09-29 (§11.7.7): three unrelated objects are called "slot",
    and a slot is 498.0 MiB, not 432.0.** `rowset_init.slot_bytes` (432.0 MiB)
    is the row-set FETCH unit; the allocation is `byte_account.slot_window` /
    `n_slots` = 522,190,848 B = **498.0 MiB**, sized for the largest (q6_K)
    bank. The byte account closes exactly and reproduces the 4-slot refusal
    (5,266,696,704 B) to the byte. **K is priced in slots: 1.08 / 2.17 / 8.67
    slot-equivalents at K=1 / 2 / 8.** The K=2 cache gives 16-of-256 experts
    (6.25%) per layer with a **measured hit rate of 0.439**; 3 slots give
    256-of-256 (100%) for 3 layers. Full per-layer coverage would need K=32 =
    ~~17.6~~ 17.07 GiB, so the cache is structurally partial and the slot is
    structurally complete. **Byte efficiency: 175.9 MiB per tok/s for a slot
    vs 594.0 for the cache - the slot is 3.4x better, and 4 slots + no cache
    is both 6.9% faster and 595.0 MiB lighter than 3 slots + K=2.**
    *(Corrected 2026-09-30, §9.5 and §11.7.7. `slot_window / n_slots` is the
    preflight's charge, not the allocation. Each slot is sized to the largest
    package in its own rotation: 498.0 MiB where the rotation holds a q6_K
    layer, 432.0 MiB where it does not. So 3 slots take 1,494.0 MiB and 4
    slots take 1,860.0 MiB (two of each). "Reproduces the refusal to the byte"
    confirms the code's arithmetic, which over-charges 4 slots by 132.0 MiB.
    K costs **1.10 / 2.19 / 8.78** slot-equivalents. The fourth slot costs
    **129.3 MiB per tok/s, 4.6x** better than the cache. Four slots are
    **7.0%** faster (12.401 / 14.238 / 15.231 tok/s) and **727.0 MiB**
    lighter. Every rate is a single arm.)*
16. **MEASURED 2026-09-29 (§11.7.8): the slot and the cache are SERIAL in
    time and SIMULTANEOUS in memory - and the slot is 97.3% dead.** Per window:
    4a H2D the missing rows into the cache, 4b D2D the WHOLE union into the
    slot, host-blocking wait, FFN reads the slot. Never concurrent, but
    `slot_vram_bytes` = 1,566,584,832 B (~~1,492.6~~ 1,494.0 MiB) is a real allocation
    while the row-set path writes only **13.5 MiB per (layer, token) = 2.7%
    utilisation**. **RETRACTION: the "100% coverage" explanation in item 15 is
    wrong for row-set TG** - 4b writes only the current union, so the slot holds
    the same 8 rows the cache would. `n_slots` is a **pipelining-depth** knob,
    not a capacity knob. Confirmed by the K=2 gain landing 96.2% in
    transfer+wait (-6,029.2 ms) and 3.6% in compute (-357.8 ms), for
    **+1,132.00 MiB peak VRAM** (5,957.56 -> 7,089.56 MiB). **Sizing slots to
    the row-set working set would free ~1,452 MiB (3 slots) / ~1,938 MiB
    (4 slots) - against a 2 x 236k envelope of only 1,536.15 MiB. The 4-slot
    K=2 refusal is an artefact of the reservation rule, not a physical limit.
    The narrowing itself (S2) is NOT implemented; the correctness hardening
    around it (S1) IS - see item 22.** *(Corrected 2026-09-30, §9.5, §9.6:
    working-set slots would free **1,453.5 / 1,806.0 MiB** at 3 / 4 slots. The
    4-slot K=2 refusal comes from the 132.0 MiB over-charge, and pricing the
    real allocation lifts it with no narrowing. S2 is now committed (item 17).
    It narrows the slots only after prefill, so as ordered its saving reaches
    neither the cache nor the peak.)*
17. **AUTHORIZED + CONTRACTED 2026-09-29 (§11.7.9): T-GPLR-62, the compact
    row-set slot.** Owner approved the expert-ID remap and the AGENTS.md:65
    amendment. **S1 (correctness hardening) built and validated; S2 (the shape
    change that frees the VRAM) not started.**
    Blocking proof: `make_dev_tensor` `:1132` forces `ne[2] = 256`,
    `slot_bytes` `:3236` has no width knob, row-set writes at absolute
    `uniq[i] * dst_t->nb[2]` `:9372`. **Derived prize at W=16, 4 slots: slot
    1,992.0 -> 124.50 MiB, freeing 1,867.5 MiB = 1.22x the entire 2 x 236k
    envelope**; `bounded_window` 5,266,696,704 -> 3,308,481,024 B, leaving
    2,844.8 MiB, so **K=2 stops refusing and K=4 (2,160.0 MiB) becomes
    reachable.** W = worst-case union (8 one session / 16 two batched), never
    the mean. **CORRECTED §11.7.10: PP/TG coexistence is NOT already solved** -
    `:6838-6866`/`:7092-7120` release the slots only on the handoff path, which
    row-set TG suppresses, so a separate PP->TG transition is required. **But
    the graph is already slot-bound** (`src/models/qwen35.cpp:142`, `layer(il)`
    returns `ls->slot_layer(il)`), so the shape change needs **no arch-builder
    edit** - materially smaller than assumed. **A9 slot-content recon and A10 are
    the load-bearing gates** (§8.1: A3 route parity is a DETERMINISM gate and
    cannot catch a deterministic mapping error). *(Corrected 2026-09-30, §9.5,
    §9.6, §11.7.9. S2 is committed in the nested tree (2298a7cf9 through
    9b90025c4). The measured prize is 1,860.0 -> 116.3 MiB, freeing **1,743.8
    MiB = 1.14x** the envelope. The 3,308,481,024 B figure is the W-aware
    preflight's. It admits K=5, not just K=4, and K=4's cache is 2,185.4 MiB.
    None of it is realizable as ordered: the cache is allocated before prefill,
    with the full slots live. A K=5 run at 6,000 MiB would hold 7,622.3 MiB
    through prefill (expected). K=2 stops refusing without S2. The measured S2
    peak (8,266,514,432 B at W=4/8/16) is a bring-up sample. The true
    high-water sits above the full-bank control's.)*
18. **The largest unmeasured thing left: 2 x 236k concurrent TG - and it is a
    HARNESS gap, not a capability gap (verified 2026-09-29).** The runner is
    `n_seq_max = 1` and llama-server has no row cache, so no arm can produce
    the full constellation. But the concurrent half alone is reachable today:
    `id_slot` is a documented llama-server request field
    (`server-context.cpp:4222`/`:1494`), so two simultaneous requests pinned to
    slots 0 and 1 deterministically run 2 sessions. The owner's only 2-session
    TG figure (22.39 tok/s) had **slot 0 idle and a 372-token prompt** - it is
    a one-active-slot measurement and must not be read as a concurrency
    ceiling. K=2 is separately unavailable in that harness, and §11.7.6 shows
    it would not be wanted: at 2 x 236k the K=2 cache leaves **45.13 MiB** -
    no room for any layer package - and a direct K=2-vs-slot A/B put the
    cache **behind** the slot (14.24 vs 15.231 tok/s).
19. **MEASURED 2026-09-29 (§11.7.3): `token_embd` (515.13 MiB) is host-side
    unconditionally** - `llama-model.cpp:1370-1371` hard-codes `dev_input` to
    the CPU. The `output` head (398.02 MiB) **is** on CUDA0. The static class
    is therefore **398.02 MiB resident**, not 913.08, which raises the
    layer-residency answer from 1 to **2 complete packages**. `-lv 5` is the
    minimum level that shows per-tensor placement.
20. **The MTP expert bank (432.00 MiB) is never loaded** - logged as an unused
    tensor. The effective model is **40 expert layers**; per-layer accounting
    that counts 41 overstates expert traffic by 2.4%.
21. Every headroom figure here is from a simulation that is **~800 MiB
    flattering** to a real 8,192 MiB card.
22. **MEASURED 2026-09-29 (§11.7.10): the 256-token single-arm TG noise floor is
    +/-11-13%**, from 3 builds with identical event counts (51,598) and identical
    peak VRAM (5,957.56 / 7,089.56 MiB). **Therefore no single-arm TG comparison
    in this document is attributable**, including §11.7.6's "4 slots 15.231 beats
    3 slots 12.40" (+22.8%, same order as the noise) - **the claim that the slot
    beats the cache is NOT established.** The 2.7%-utilisation finding is
    unaffected (byte count, not timing). S1 landed: containment self-check +
    4b assert-to-refusal, 31 lines, 6 runs, no regression, 0 spurious refusals.
    A `qwen35moe.cpp` uncommitted change from another session is a live
    confound in every comparison against a recorded number.
23. **VERIFIED 2026-09-29 (§11.7.11): cross-thread isolation is real for FILES
    and false for MEASUREMENTS.** Each worktree has its own `third_party/llama.cpp`
    clone (a gitlink, mode 160000) and its own `build/`, and the MTP thread's
    `CMakeCache.txt` confirms it compiles its own source - so neither thread can
    clobber the other. **But the clones are on DIVERGED histories: this thread
    builds the pinned `de39888e`, the MTP worktree is at `06a892fb`, which does
    not exist in this clone's history. No tok/s figure is comparable across the
    two threads.** The GPU is the only shared resource, serialised solely by the
    `Global\AI-GPU-EXCLUSIVE-v1` mutex. **Caveat on my own isolation: this
    worktree's `third_party/llama.cpp` is EMPTY and it has no `build/`, so the
    S1 edit landed in the shared MAIN checkout - safe from the other thread, but
    not protected by my worktree.**
24. **VERIFIED 2026-09-29 (§11.7.12): `third_party/llama.cpp` is a gitlink
    (mode 160000) - the parent records a COMMIT HASH, not files, so editing
    files there is purely local and never propagates.** Only an explicit commit
    in the submodule AND the parent moves the pin. **Three hazards: (1) the
    clone's push remote is `github.com/ggml-org/llama.cpp` - UPSTREAM, which
    `AGENTS.md` forbids pushing to; (2) there is NO `.gitmodules`, so
    `git submodule update --init` cannot reconstruct the tree and git reports
    "no submodule mapping found"; (3) the pin is already inconsistent -
    parent says `de39888e`, the MTP worktree is at `06a892fb`, a commit absent
    from the main clone's history.**
25. **DONE 2026-09-29 (§11.7.13): the main clone's push URL is now `DISABLED`
    (`git remote set-url --push`), push fails with exit 128 before any network
    I/O, and fetch still works.** **Scope corrected: the upstream push URL
    existed ONLY in the main clone**; the other three vendored clones push to
    LOCAL paths (Freebuff consolidation) and were deliberately left alone.
    **New exposure: those local remotes push INTO the main clone's working
    tree, where the uncommitted S1 edit lives** - so long-lived uncommitted
    work should not be left there while other threads push.
26. New tools this session. **All five take a `-ValidateOnly` no-GPU dry run**
    that prints each arm's resolved argv and returns before the first process
    launch: `tools/fit-compute-sweep.ps1` (compute buffer vs
    ctx/batch/ubatch/parallel), `tools/run-ubatch-pp-ab.ps1` (the PP A/B, rule
    pre-registered in its header), `tools/split-count-probe.ps1` (splits vs
    compute, `-CtxList`), `tools/run-tg-k2-budget-ab.ps1` (the K=2 TG test), and
    `tools/run-concurrent-tg-sweep.ps1` (the N=2/3/4 concurrent TG sweep).
    Each needs its `-*Exe` passed explicitly or the wrapper cannot attribute the
    binary. **Correction (2026-09-29 audit):** `fit-compute-sweep.ps1` and
    `split-count-probe.ps1` did NOT have the switch when this line was first
    written - the claim was false for two of the four named tools, found by
    invoking them. Both now have it. Note `-ValidateOnly` is the ARM-DRIVER
    convention only: `tools/with-gpu-lock.ps1` and `tools/arm-verdict.ps1` are
    the wrappers and deliberately have no such switch.

27. **The five new arm drivers no longer hardcode the repo root or the model
    path** (2026-09-29). They adopt the convention the pre-existing shared
    tooling already uses - `tools/with-gpu-lock.ps1:43` derives its root with
    `Split-Path (Split-Path $PSCommandPath -Parent) -Parent` - plus a walk upward
    to the first ancestor that actually contains `build\llama-cuda`, and the
    model path became a `-Model` parameter defaulting to the current value.
    **The walk-up is load-bearing, not decoration:** these scripts live in a
    Freebuff worktree whose `build/` does NOT exist, while the build output,
    prompts and results live in the MAIN tree. A plain `$PSCommandPath` parent
    would therefore resolve to a root with no binaries - which is exactly why
    `run-ubatch-pp-ab.ps1` carried a hardcoded root and a comment saying
    `$PSCommandPath` could not be used. `-DataRoot` overrides the derivation.
    Verified: all five run `-ValidateOnly` correctly from a directory that is
    not the repo root, and launch nothing.

    **Correction to a claim in the audit that prompted this:** `Get-ArmArgs` is
    NOT duplicated between `run-tg-k2-budget-ab.ps1` and `run-ubatch-pp-ab.ps1`.
    Same name, unrelated bodies - the first builds `batch-major` / `q8` KV /
    `--residency-tg` / `--tg-tokens 256` arms, the second builds `layer-major` /
    `f16` KV / `--act-slab-dtype bf16` / `--pp-repeat 3` arms. Only the leading
    `--model`/`--prompt-file` pair is common. Extracting a shared helper would
    have coupled two experiments that are meant to vary independently, so the
    duplication was left alone deliberately. The only genuine repeat is the
    12-line root-derivation block, which is identical by design and commented as
    such in each file.

28. The dead `-PromptTokens` parameter is gone from
    `tools/run-concurrent-tg-sweep.ps1` and left no orphans: nothing else in
    that file referenced it. (`-PromptTokens` in `run-gplr44-ab-50k.ps1`,
    `run-gplr45-arm.ps1` and `run-gplr44-pair.ps1` is UNRELATED - pre-existing,
    tracked, unmodified, and genuinely used to select the anchor prompt.)

### 11.7.14 MEASURED 2026-09-29: concurrent decode at N=2/3/4 - every count holds, and aggregate throughput RISES

**The question this section answers:** the effort exists to run two concurrent
coding agents. We had never measured two concurrent decode sessions. Every TG
number before this was a **one-active-slot** number (the owner's own baseline
had slot 0 idle).

**Harness:** `tools/run-concurrent-tg-sweep.ps1` (new). Two pinned clients via the
`id_slot` JSON field on `/completion` - **no code change**; the field already
existed. Pinning is proven, not assumed: server logs show `launch_slot_: id 0 |
task 0` and `id 1 | task 1`, with `n_processing_slots = 2` logged while slot 0
was already decoding.

**Design against the noise floor (§11.7.10).** Single 256-token arms are
worthless at +/-11-13%, so: 1024 generated tokens per arm, **3 arms per
configuration, interleaved rep-outer/parallel-inner** (n2,n3,n4,n2,n3,n4,...) so
thermal or background drift hits every configuration equally rather than
penalising whichever ran last. Medians and spreads reported, never a single arm.

**Configuration (identical across all 9 cells):** model
`Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf` (qwen35moe, 40 layers, 256 experts, 8 used),
`-c 472000 --parallel N`, `-ctk q8_0 -ctv q8_0`, `--flash-attn on`,
`--batch-size 8192 --ubatch-size 512`, `--fit-target 16000`, `-lv 5`,
`--load-mode none`. Hardware: RTX 5090 Laptop, 24463 MiB total, wrapped in
`Global\AI-GPU-EXCLUSIVE-v1` for the whole sweep - one hold, all 9 cells.
Binary: `llama.dll 4AE45B26E576FC14`, digest `359B8BE37625B537`.

**All 9 cells valid. Zero refusals, zero ERROR-level lines, `gpu_pre=clean
gpu_post=clean` on every arm.** All slots reported `stop_type=limit` at the full
1024 tokens (no early EOS inflating a rate).

| N | ctx/slot | layers on GPU | arms (tok/s per session) | **median** | spread | prefill tps | VRAM at load | **peak VRAM** |
|---|---|---|---|---|---|---|---|---|
| 2 | 236,032 | 30 | 15.389 / 15.560 / 15.675 | **15.560** | 1.8% | 63.5-64.9 | 7287 MiB | **7305 MiB** |
| 3 | 157,440 | 30 | 12.355 / 12.763 / 13.607 | **12.763** | 9.8% | 58.0-59.9 | 7257 MiB | **7275 MiB** |
| 4 | 118,016 | 29 | 10.660 / 10.860 / 10.929 | **10.860** | 2.5% | 51.6-57.3 | 7211 MiB | **7261 MiB** |

**Column sources - every figure is a per-arm reading, not a value copied across
arms.** `ctx/slot`, `layers on GPU` and the buffer figures come from the
server's own load log (`n_ctx_slot`, `fit_layers`, and the `load_tensors`
buffer lines), parsed per cell. The three VRAM columns are three SEPARATE live
`nvidia-smi --query-gpu=memory.used` probes taken inside the per-arm loop at
three points: **pre** (before launch, 0 MiB - confirms no foreign process),
**at load** (server loaded, generation not yet started), and **post** (after all
N slots finished generating). The KV cache grows across generation, so
post >= load in every arm (+18 MiB at N=2 and N=3, +50 MiB at N=4); the true
high-water mark is therefore the post figure, and the harness now records it as
`peak_vram_mib = max(load, post)`.

**CORRECTION (2026-09-29 audit): this table previously had a single VRAM column
labelled "peak VRAM" whose values (7287 / 7257 / 7211) were the LOAD-time probe,
not the peak.** The values were real and per-arm, but the label overstated what
they measured, understating actual peak usage by 18-50 MiB. Both columns are now
shown. The label error originated in the harness's own console line, which
printed `$loadVram` under the caption "peak VRAM"; that line now prints
load/post/peak and writes `peak_vram_mib` into each cell's JSON.

The three reps of each N return byte-identical VRAM (7287/7287/7287 etc.). That
is expected, not a copied constant: the probes are re-read per arm, and VRAM at
a fixed configuration is deterministic because the fitter's placement decision
is. The tok/s figures DO vary across reps, so the loop is genuinely re-measuring.

**THE HEADLINE - and it is the opposite of the framing that motivated this
work.** Per-session throughput falls as expected, but **aggregate throughput
rises**:

| N | per-session | % of N=2 | **aggregate** | % of N=2 aggregate |
|---|---|---|---|---|
| 2 | 15.560 | 100% | **31.12 tok/s** | 100% |
| 3 | 12.763 | 82.0% | **38.29 tok/s** | **+23.0%** |
| 4 | 10.860 | 69.8% | **43.44 tok/s** | **+39.6%** |

**Four sessions produce 40% more tokens per second than two, from the same
memory.** This is ordinary batched decode - a step processes all active slots'
tokens in one pass, so the per-token model cost is amortised across them - and
it is the reason concurrency on this model is nearly free in throughput terms.
The owner's premise that a second session would cost a proportional share of
speed is **not** what the hardware does; the cost is paid in *latency per
session* (each session waits its turn in the batch), not in total work done.

**Concurrency is nearly free in MEMORY too - the second surprise.** Peak VRAM
*moves the wrong way*: **7305 (N=2) -> 7275 (N=3) -> 7261 (N=4) MiB, i.e. 44 MiB
LOWER at N=4** (post-generation probe; the load-time figures fall by 76 MiB over
the same range, 7287 -> 7211, but those understate the arm's high-water mark -
see the column-sources note above). Buffers explain it: `model` 1464.29 ->
1424.46 MiB (N=4 drops to 29 layers on GPU) and `compute` 1940.22 -> 1824.97 MiB.
Total context is held constant at ~472k, so **more slots means less context per
slot** (236k -> 157k -> 118k), and a smaller per-slot context shrinks both the KV
split and the compute buffer. This is §11.7.5's "context is the lever"
reproducing itself on the TG side, and it is the *opposite* of the §11.7
estimator's prediction that a second slot adds memory. The direction and the
conclusion are unchanged whichever probe is used; only the magnitude differs.

**Prefill is not free, but it is cheap and it is a one-off.** Prompt rates fall
64.9 -> 58.0 -> 51.6 tps from N=2 to N=4 (-20%). This matters only for the
long-context prefill that the charter prioritises, and only on the first
request of each session.

**Answers to the three questions asked, plainly:**

- **N=2: HOLDS.** 15.560 tok/s per session, 31.12 aggregate, 236k context each.
  This is the configuration the whole project was designed for, and it works.
- **N=3: HOLDS.** 12.763 tok/s per session (-18%), 157k context each. Degrades
  gracefully, no refusal. Spread was 9.8% - the noisiest cell, so its median
  carries the least weight of the three.
- **N=4: HOLDS.** 10.860 tok/s per session (-30%), 118k context each. Still
  +40% aggregate. Cost is 29 layers instead of 30.
- **No refusals at any N.** No capacity deferrals, no slot drops, no
  `SRV_DBG` defer events in any of the 9 logs. The fitter's
  "cannot be used with preferred buffer type CUDA0, using CUDA_Host instead"
  lines are the expected MoE-to-host placement, not errors.

**What this does NOT establish (limits, stated plainly):**

1. **This is the STOCK fitter path, not the layer-residency experimental path.**
   Zero `layer-stream` / `rowset` telemetry appears in any log; the fitter moved
   MoE tensors to system memory and kept 30 of 40 layers resident. So this
   measures **what plain llama.cpp already does** - which is exactly the right
   control to establish first, and exactly the reason no streaming work was
   needed to answer the concurrency question. It does **not** validate S1 or S2.
2. **Hardware is not the 8 GB-class target.** This is a 24,463 MiB RTX 5090
   Laptop with `--fit-target 16000` emulating the envelope; observed peak is
   7.2 GB. The mechanism has never been exercised on a genuinely 8 GB card.
3. **112k-236k context per session at 1024 generated tokens.** The KV cache
   barely fills, so this measures the *pre-fill-to-capacity* regime. Behaviour
   once a session actually approaches its context ceiling is untested.
4. **Three arms, not many.** Spreads of 1.8-9.8% are honest but the N=3 median
   would firm up with more reps. The between-sweep drift was real: an earlier
   1-rep sweep measured N=2 at 14.878 where this sweep's three arms give
   15.389-15.675, a 4.6% gap between sweeps on the same binary - which is why
   cross-run comparison of single arms is banned here, consistent with §11.7.11.
5. **No quality check.** Rate is not correctness. A semantic-QA pass over
   concurrent answers was not run; this section says nothing about answer
   quality under concurrency.

**Caveat on scope of this result (stated once, deliberately).** These numbers
come from the **stock llama.cpp fitter path, not the layer-residency
experimental path** - zero `layer-stream` / `rowset` telemetry appears in any of
the 9 logs, and the fitter moved MoE tensors to system memory keeping 30 of 40
layers resident. They are also from a 24,463 MiB RTX 5090 Laptop with
`--fit-target 16000` emulating the envelope, not a genuinely 8 GB card. So
"holds at N=2/3/4" is established **for the stock path on emulated hardware**,
and says nothing yet about the streaming path or the 8 GB target. No strategic
conclusion is drawn from that here; limits 1 and 2 above carry the detail.

Evidence: `build/llama-cuda/results/r3-*.json` (9 cells),
`r3-...-summary.json`, server logs `build/llama-cuda/logs/conc-r3/`,
verdicts `conc-r1` / `conc-r3` (both PASS, `gpu_pre=clean gpu_post=clean`).

---

## 12. MEASURED 2026-09-30: the host expert FFN, characterised properly

Host-only, no GPU, no mutex. `tools/expert-cpu-bench` after the `--id-stride`
fix (see §2 and the placement draft's item 3). Every cell below is
`--pool 256 --id-stride 8` — a full 256-expert bank with iteration-disjoint
expert sets, so no slab is cache-warm from the previous iteration.

Evidence: `build/expert-cpu-bench/{audit-A,audit-B,audit-C,threads-k8,pp-shape}.{json,log}`,
and from 2026-09-30 the thread cross-products:
`{threads-x-experts,tg-threads-confirm,pp-threads,pp-threads-fine,pp-threads-top}.{json,log}`.

### 12.1 At TG the host expert FFN is DDR-BANDWIDTH-BOUND and saturates at 8 threads

k=8, 1 token, 300 iterations, medians:

| threads | 1 | 2 | 4 | **8** | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| median us | 461.6 | 355.2 | 302.9 | **285.6** | 296.6 | 333.9 |
| GB/s | 30.66 | 39.85 | 46.74 | **49.56** | 47.73 | 42.40 |

**1 -> 8 threads buys 1.62x for 8x the cores, and past 8 it gets worse** (+3.9%
at 16, +16.9% at 32). 49.56 GB/s is this box's DDR ceiling for this access
pattern, which independently corroborates the "~35-40 GB/s effective DDR feed"
row in the source table and supersedes it upward.

> **SCOPE CORRECTED 2026-09-30 (§12.1c).** Every sentence above is true **at
> k=8**, which is the only expert count this table measured. It is **false in
> the regime the split actually runs in**: at 2–3 experts per layer the optimum
> moves to **12 threads**, where 8 threads is 15–17% slower. "Saturates at 8
> threads" is a property of k=8's 14.2 MB working set, not of the machine. The
> cross-product is in §12.1c; read this table as one row of it.

Two consequences that bear directly on the CPU/GPU split:

- **The host leg of any split cannot be made faster by throwing threads at it.**
  ~35.7 us per expert (285.6/8) is a floor set by memory bandwidth, not by core
  count. Any overlapped-split arithmetic must use that figure and must not
  assume thread scaling will shrink it. — **Retracted 2026-09-30.** It can: at
  the split's own operating point 12 threads takes the per-expert cost to
  **28.1 us** (56.2/2) and **30.0 us** (90.0/3). The bandwidth floor is real but
  it is not reached at 8 threads except at k=8. §12.1c.
- **This run reproduces arm C.** 285.6 us here against 288.6 us in the audit
  sweep — 1.0% apart on separate invocations. The clean constant is stable; the
  2026-09-29 figure it replaced was not a noise problem, it was a bug.

### 12.1a The split's linear host leg is validated, and is conservative

§13's cost model charges the host `(1-h)·8·c` per layer — linear in the number
of experts the host still has to run. At the hit rates the split targets the
host runs **1 to 3 experts per layer, not 8**, so that linearity is an
assumption about a regime the headline `c` was never measured in. `audit-C`
already swept it, at 8 threads, and it holds:

| experts | median us | **us/expert** | GB/s |
| --- | --- | --- | --- |
| 1 | 33.10 | **33.10** | 53.46 |
| 2 | 66.95 | **33.48** | 52.86 |
| 4 | 144.40 | **36.10** | 49.02 |
| 8 | 288.65 | **36.08** | 49.04 |

Per-expert cost is flat to **falling** as the count drops — 33.1 us at one
expert against 36.1 at eight, because 1.77 MB sits better in L3 than 14.2 MB
does.

**CORRECTED 2026-09-30. The conclusion below was right; the reason given for it
was false, and the check behind it was incomplete.** This section originally
said the real host is "up to 8% cheaper per expert than charged, never more
expensive", and concluded the linear leg is conservative. The "never more
expensive" half is **wrong**. `c` = 35.7 us is the *8-expert* rate (285.6/8,
from `threads-k8`), and the curve only falls below it at 1–2 experts. At the
**stock anchor** the host runs `(1−0.188)·8` = **6.50 experts per layer**,
where the curve charges **234.43 us against the constant's 231.93 — 2.50 us
per layer more**, not less.

That regime is not incidental: it is where `other` is solved. The original
check asked what the curve does to the overlapped *ceiling* and correctly found
it negligible (above `h*` the device leg binds, so the host's exact cost is off
the critical path). It never asked what the curve does to **`other`**, which is
common to every policy and therefore reaches every number in §13.4.

Replacing the constant with piecewise-linear interpolation of the four measured
totals and re-solving everything — `tools/host-curve.py`:

| | `other` | `h*` | floor | ceiling |
| --- | --- | --- | --- | --- |
| constant `c` = 35.7 (published) | 14.247 ms | 0.7069 | 83.7 us/layer | 56.83 tok/s = **1.388x** |
| measured curve | 14.147 ms | 0.6990 | 82.8 us/layer | 57.28 tok/s = **1.399x** |

Every policy moves **up**, by +0.001x to +0.011x, and the **ranking is
unchanged** (verified by re-sorting both, not by inspection). The best
placement goes 1.371x → 1.378x. So the published constant-`c` figures are
conservative — but via the opposite mechanism to the one this section claimed:
the curve charges *more* at stock, which *lowers* `other`, which *raises* every
multiplier. Two errors pointing the same way.

The earlier "re-solve with `c` = 33.1" bracket gave 1.394x. That was a bound,
not the answer: flat 33.1 applies the one-expert rate at every count, and it
too left the `other` channel out. The interpolated value, 1.399x, sits just
outside its own bracket.

**The constant is retained as published** rather than propagated through six
scripts for a ≤0.011x effect — one cost model is worth more than a marginally
tighter one, since divergent copies of a constant are precisely what produced
the "8,416 MiB" drift (§13.6). `tools/host-curve.py` is the standing check that
the approximation stays conservative. **No new measurement is needed for this
term.** Uncertainty on any of it is the 1.0% cross-invocation spread between
`audit-C`'s 288.65 us and `threads-k8`'s 285.6 us at 8 experts.

> **"No new measurement is needed for this term" was wrong — §12.1c.** Both
> curves above are **8-thread** curves, and the anchor they are solved against
> ran at **16**. That is a unit mismatch, not a tuning choice, and no amount of
> re-interpolating an 8-thread sweep can find it. The thread dimension was the
> needed measurement. The conclusion — the published constant is conservative —
> survives again and by a wider margin (`other` 14.247 → **14.083 ms**), but
> "this term is closed" was a claim about a sweep's coverage, and the sweep did
> not cover it.

### 12.1c MEASURED 2026-09-30: the thread optimum MOVES with expert count, and the anchor ran at 16

§12.1 swept threads at k=8. §12.1a swept expert count at 8 threads. The
cross-product was never measured — and it is the cell the split lives in. The
overlapped split runs the host at `(1−h)·8` = **2–3 experts**, a **4 MB**
working set; k=8 is **14.2 MB**. A saturation point established on 14.2 MB does
not transfer to 4 MB, and it does not transfer.

Host-only, no GPU, no mutex — §12's footing. `--pool 256 --id-stride 8
--iters 1000 --threads 1,2,4,6,8,12,16,32 --experts 1,2,3,4,8`. Iterations
raised 300 → 1000 because `audit-C`'s 1-expert cell spanned min 20.4 / median
33.1 / p90 47.7 (±44%), too noisy to read a thread optimum from. Evidence:
`build/expert-cpu-bench/threads-x-experts.{json,log}`. Median us, 1 token:

| threads | 1 | 2 | 3 | 4 | 8 |
| --- | --- | --- | --- | --- | --- |
| 1 | 64.4 | 120.9 | 181.3 | 258.3 | 510.6 |
| 2 | 45.9 | 98.2 | 153.2 | 199.4 | 387.5 |
| 4 | 31.3 | 70.4 | 116.3 | 156.8 | 308.7 |
| 6 | 29.2 | 66.4 | 111.5 | 148.7 | 293.2 |
| 8 | 30.0 | **64.8** | 104.9 | 143.7 | **286.2** |
| **12** | 29.7 | **56.2** | **90.0** | **130.3** | 286.4 |
| 16 | **28.7** | 61.0 | 97.7 | 140.9 | 293.3 |
| 32 | 35.6 | 64.9 | 117.3 | 157.1 | 332.4 |

| experts | best threads | best us | 8-thread us | what 8 threads costs |
| --- | --- | --- | --- | --- |
| 1 | 16 | 28.7 | 30.0 | +4.5% |
| **2** | **12** | **56.2** | 64.8 | **+15.2%** |
| **3** | **12** | **90.0** | 104.9 | **+16.6%** |
| 4 | 12 | 130.3 | 143.7 | +10.3% |
| 8 | 8 | 286.2 | 286.2 | — |

The k=8 column reproduces `threads-k8` (286.2 vs 285.65; 332.4 vs 333.85), so
this is not drift — it is a dimension nobody had swept. `--threads` in the bench
is `ggml_backend_cpu_set_n_threads` on the CPU backend, the same call
`llama.cpp` makes for its own `--threads`, so the sweep transfers directly.

**The correction. `other` was solved with an 8-thread constant against a
16-thread anchor.** The anchor log's line 24 reads
`n_threads = 16 (n_threads_batch = 16) / 32`, and line 309
`llama threadpool init, n_threads = 16`. At `h_stock` = 0.188 the host runs 6.50
experts/layer:

| | us/layer |
| --- | --- |
| model's `c` = 35.7 (an **8-thread** figure, 285.6/8) | 231.93 |
| measured at the anchor's **16** threads | **236.04** (+4.11) |
| `other` | 14.247 ms published → **14.083 ms** (−0.164) |

This is the **eighth** record error found by re-deriving published numbers from
their raw artifacts, and the fourth in a row whose correction moves the answer
in our favour: a lower `other` raises every multiplier.

**The optimization.** Running the host leg at its per-regime optimum instead of
the anchor's 16, `tools/thread-optimum.py`:

| host leg threads | `h*` | floor | ceiling |
| --- | --- | --- | --- |
| 16 (anchor's) | 0.6825 | 80.8 us/layer | 1.410x |
| 8 (what the model assumed) | 0.6954 | 82.3 us/layer | 1.405x |
| **12** (optimum at 2.67 experts) | **0.6662** | **78.9 us/layer** | **1.416x** |

The ceiling barely moves, and that is §12.1a's point holding: above `h*` the
device leg binds and the host's cost is off the critical path. What moves is
**`h*` itself, 0.6825 → 0.6662** — the residency required to reach the floor
drops by 1.6 points. Given §13.6, where capacity is the binding constraint at
236k and not throughput, buying residency with threads instead of VRAM is the
trade that is actually available.

Per policy, and this is where it matters, because every policy except the top
two sits **below** `h*` and is host-bound:

| rows | policy | h (B / A) | 16 thr | 12 thr | gain | bound by |
| --- | --- | --- | --- | --- | --- | --- |
| 48 | static build-time pin (held out) | 0.334 / 0.388 | 1.140x | 1.161x | +0.021x | host |
| 48 | warm-then-pin, W=128, frozen | 0.509 / 0.727 | 1.319x | 1.333x | +0.014x | device |
| **48** | **LRU-K K=6 (steady state)** | 0.634 / 0.716 | 1.381x | **1.393x** | +0.012x | device |
| 64 | static build-time pin (held out) | 0.412 / 0.484 | 1.200x | 1.225x | +0.025x | host |
| 64 | warm-then-pin, W=128, frozen | 0.598 / 0.814 | 1.345x | 1.357x | +0.012x | device |
| 64 | LRU-K K=8 (steady state) | 0.720 / 0.786 | 1.383x | 1.383x | +0.000x | device |
| 64 | in-sample oracle (ceiling) | 0.757 / 0.873 | 1.361x | 1.361x | +0.000x | device |

**The §13.4 ranking flips at the top**, re-sorted programmatically, not by
inspection: at 16 threads `K=8 @64 > K=6 @48` by +0.0020x; at 12 threads
`K=6 @48 > K=8 @64` by +0.0096x. **Both margins are inside the 1.0%
cross-invocation spread**, which on a ~1.38x figure is about 0.014x. So the
honest reading is **not** "48 rows wins on throughput" — it is that the two are
**tied**, and the tie-break is capacity: 48 rows is the only one that reaches
236k at all (§13.6). Thread tuning removes the one argument that pointed the
other way; it does not manufacture a new one.

**And a win that needs no split at all.** Stock's own host leg at 16 threads is
the thing being measured above. Moving it alone:

| | ms/token | | tok/s |
| --- | --- | --- | --- |
| stock, 16 threads (the anchor) | 24.414 | — | 40.96 |
| stock, 8 threads | ~~24.278~~ | ~~+0.56%~~ | ~~41.19~~ |
| **stock, 12 threads** | ~~**24.083**~~ | ~~**+1.36%**~~ | ~~**41.52**~~ |

*Struck 2026-09-30 (twelfth finding): these rows price stock at an
interpolated 6.5 experts per layer. At k=8, where stock's host layers run, the
same cells give +0.95% (8 threads) and +0.92% (12), and both are unproven — see
the correction below.*

That is the cheapest measured item in this document: no split, no cache, no new
code, one flag. It also re-scopes the open `--threads 8 --threads-batch 16`
task — at TG the right TG figure is **12**, not 8; 8 was inferred from the k=8
sweep, which is the wrong regime for the split and, at 0.56%, leaves most of the
gain on the table even for stock. **The `--threads-batch` half is wrong by more
— see §12.2a, where 16 turns out to be a local pessimum and the prefill win is
1.19x.** The pair becomes `--threads 12 --threads-batch 28`.

> **Corrected 2026-09-30 (§12.5).**
>
> - **+1.36% was priced in the wrong regime (twelfth finding).**
>   `thread-optimum.py` prices stock's host leg as 40 layers at an interpolated
>   6.5 experts each. Stock computes all 8 of a token's experts on every host
>   layer, and none on a device layer.
>   - The interpolation takes 48% of its per-layer gain from the k=4 cell, a
>     shape stock never runs. The 12-thread lead per expert is 2.66 µs at k=4
>     but 0.86 µs at k=8.
>   - In the stock regime the same cells give 6.85 µs per host layer (k=8,
>     293.3 → 286.45 µs). Over 32.79 host bank-equivalents (the loader's
>     placement, tenth finding) that is 0.22 ms per token: **+0.92%**, about
>     two-thirds of the claim. 8 threads ties 12 at k=8 (286.2 µs).
>   - The same error moves `other`. Solved in the stock regime at 16 threads,
>     it is ~13.94 ms, not 14.083 (−1.0%, expected). That raises the 16-thread
>     split ceiling from 1.410x to about 1.421x, inside the 0.014x
>     cross-invocation spread, so nothing downstream is re-derived.
> - **Even +0.92% is unproven.**
>   - Five invocations measure the k=8, one-token cell at 16 threads and at a
>     lower count. Four favour fewer threads: by 1.2% (`pp-shape`, §12.2's run,
>     8 threads), 3.7% (`threads-k8`, 8), 2.3% (here, 12) and 17.7%
>     (`verify-shape`, 12). Priced in the stock regime, that is +0.5%, +1.5%,
>     +0.9% and +6.4% of the token.
>   - The fifth, `pp-threads-fine` (this section's shape, 60 iterations),
>     favours 16 over every lower count it ran: 289.7 µs against 291.3 (10
>     threads), 292.1 (12) and 290.1 (14). At 12 threads that is −0.3% of the
>     token.
>   - All five ran their thread counts in ascending order. The bench's thread
>     loop is outermost and follows `--threads` (`expert-cpu-bench.cpp:581`),
>     and each JSON records its cells in run order. So thread count is
>     confounded with run order.
>   - At this section's shape (pool 256, id-stride 8) the 16-thread cell alone
>     spans 289.7–296.6 µs over four invocations, 2.4%: as wide as the 2.3%
>     lead it is meant to resolve. `verify-shape` runs id-stride 32, so its
>     17.7% is a different shape as well as a different run.
>   - In `verify-shape` the 12-thread lead fades as the run goes on. It wins
>     its first three cells (−17.7/−19.6/−44.4%), then only 2 of the next 7.
>     Its per-expert cost climbs from 27.7 to 35.6 µs with k, while 16 threads
>     stays at 31–35 µs. One invocation cannot separate box drift from a real
>     k-dependence.
>   - So neither size nor sign is established. `tools/thread-optimum.py`
>     prints this bullet's table, with each invocation's thread order read off
>     its JSON.
> - **The `--threads-batch 28` half is retracted (§12.2a).** Stock prefill does
>   not compute host experts on the CPU.
> - **The pair `--threads 12 --threads-batch 28` is withdrawn.** The TG half
>   is settled by `llama-bench -p 0 -n 128` at `-t 16,12,8` and then at
>   `-t 8,12,16` (§12.5 A/B vehicle, item 3).
> - **The table above stands as a measurement of the host leg.** Its 12-thread
>   column rests on the k=2–3 cells. A second invocation (`tg-threads-confirm`,
>   5,000 iterations) reproduces 12 over 16 there: −8.3% at k=2 and −5.9% at
>   k=3, against −7.9% for both here. That invocation also ran threads in
>   ascending order, so the `h*` shift to 0.6662 is indicated, not
>   established.

**Caveat, stated rather than buried.** The bench gives the CPU backend the whole
box to run expert FFN alone. In a real split the device leg runs concurrently
and the CPU is also doing attention, the router and the shared expert. Thread
counts that win in isolation may contend there, and `--threads-batch` is a
separate knob §12.2 shows inverts at PP. Every figure in this section is an
**upper bound** on the thread-tuning gain. It is also the cheapest thing in the
document to falsify: one stock `llama-server` run at `--threads 12`.

### 12.1b `h_stock` was guessed at 9 banks; the log says 7.5

`other` — everything in a token that is not expert FFN — is solved by making
the model reproduce the 40.96 tok/s anchor exactly, which requires knowing the
anchor's own hit rate. Every revision of this document up to now used
`h_stock` = **9/40 = 0.225**, inferred as "CUDA0 model buffer 5,192.44 MiB,
less **~1,302 MiB** of dense weights, over 432 MiB per bank". That 1,302 was a
guess. It is the last unmeasured input to `other`, and it is wrong.

The anchor ran at `-lv 4`, and the fitter prints a memory breakdown for each
configuration it probes — including the probe with **all MoE tensors moved to
system memory**, whose model column is the dense-only device size by
construction. It is **1,921 MiB**, not 1,302.

That figure validates itself against tensor sizes nobody chose: the
all-on-device probe reports 19,399 MiB of model, so the expert mass is
19,399 − 1,921 = **17,478 MiB**, against 40 × (432 + 3.19) = **17,407 MiB**
predicted from the `blk.40` tensor sizes the loader printed while discarding
them. **0.4% apart** — the residue is `ffn_gate_inp`, norms, and MiB rounding.

It also reproduces **exactly** in a second, unrelated run. `docs/llamacpp 8gb
vram base 2x 262k context q8.md` is the same model at `n_ctx` 512,000 across
two slots, with a different fit outcome (26 layers instead of 41, KV split
across CPU and CUDA0, and only 1,311.95 MiB of model on the device). Its
dense-only probe prints the same **1,921 MiB**. Same number, different context,
different slot count, different layer split — it is a property of the model, as
it should be, and not an artifact of the anchor's configuration.

So the anchor had 5,192.44 − 1,921 = **3,271.44 MiB** of expert weight
resident, and three independent routes agree on what that is in banks:

| route | banks | `h_stock` |
| --- | --- | --- |
| ÷ 432 MiB (bank only) | 7.57 | 0.1893 |
| ÷ 435.19 MiB (bank + shared experts) | 7.52 | 0.1879 |
| the fitter's own `(n_layer, n_part, overflow_type) = (41, 33, GATE)` | 7.67 | 0.1917 |

The third route is the log stating its decision outright: 41 − 33 = 8
non-overflowing layers, one of which is the MTP layer whose expert tensors the
loader discarded, leaving **7 full banks**; and the `GATE`/`UP`/none ladder
cost 5,192 / 5,045 / 4,905 MiB of model, steps of ~144 MiB = exactly one expert
tensor, so `GATE` keeps **2 of 3** tensors of the boundary layer on top of
those 7.

**The assumption was high by 1.4 banks.** Correcting it lowers `other` from
14.49 to **14.25 ms** and *raises* the ceiling from 1.368x to **1.388x** —
every multiplier in §13 moves up about **1.3%**. `tools/hstock-from-log.py`.

Two things this does **not** change. The **ranking** in §13.4 is untouched:
`other` is common to every policy, and `h*` does not depend on it at all. And
the error was in the **conservative** direction — the mechanism has *more*
headroom over stock than the document claimed, not less, because stock is
holding less expert weight on the device than we credited it with.

It also removes a standing caveat rather than adding one. The sensitivity
sweep that bracketed this — ceiling 1.408x at 6 banks down to 1.368x at 9 —
was worth running when the input was a guess. It no longer needs quoting as a
risk, because the input is now read off the same log that produced the anchor.

### 12.2 At PP the same code is COMPUTE-bound, and the thread optimum INVERTS *(on the CPU path only: stock prefill copies the host experts to the GPU, §12.5)*

k=8, 20 iterations, medians, per layer:

| tokens per call | 1 | 32 | 128 | 512 |
| --- | --- | --- | --- | --- |
| 8 threads (us) | 287.3 | 2,673.8 | 11,186.0 | 40,603.8 |
| 16 threads (us) | 290.8 | 2,327.0 | 9,786.8 | **36,594.6** |
| 16-thread delta | +1.2% | **-13.0%** | **-12.5%** | **-9.9%** |

At one token per call the weights are read once and used once, so the cell is
bandwidth-bound and 16 threads lose. At 512 tokens each weight is reused across
the micro-batch, the cell becomes compute-bound, and 16 threads win by ~10%.

**Actionable and free:** llama.cpp exposes `--threads` (generation) and
`--threads-batch` (prefill) separately, and the owner's recorded runs set both
to 16 (`n_threads = 16 (n_threads_batch = 16)`). **`--threads 8
--threads-batch 16`** matches each phase to its own optimum. Expected effect is
small — TG's expert portion improves ~3.9%, and TG expert work is ~47% of the
token budget, so ~1.8% on TG with prefill untouched. It costs nothing, it is a
flag, and it is the only remaining free lever found in this domain. *(True that
it is a free lever; false that it is the only one, and the values are wrong —
§12.2a. The lever is much larger than this paragraph makes it sound.)*
*(Corrected 2026-09-30: §12.2a's larger lever is retracted for stock, and the
TG half is priced again below.)*
**Unverified end-to-end:** the microbench isolates the expert FFN, while
`--threads` also governs attention and sampling on the host, so the whole-model
sign must be confirmed before adoption.

> **CORRECTED 2026-09-30 (§12.1c): the TG figure is 12, not 8, and the ~1.8%
> was 3.2x too high.** Two independent regime errors, both from applying a k=8
> number to a 6.5-expert token:
>
> | | §12.2 used | actually |
> | --- | --- | --- |
> | thread delta, 16 → 8 | 3.9% (the k=8 cell) | **1.4%** (interpolated to 6.5 experts) |
> | expert share of the token | ~47% | **38.7%** (236.04 us x 40 = 9.44 of 24.41 ms) |
> | product | ~1.8% | **0.54%**, and the direct calculation gives 0.56% |
>
> `--threads 12` gives **+1.36%** instead, because 12 beats both 8 and 16 at
> every expert count from 2 to 4 and ties 8 at k=8. Confirmed on a dedicated
> 5,000-iteration TG-only run (2 experts: 8 → 63.4, 10 → 58.9, **12 → 56.3**,
> 14 → 61.4, 16 → 61.4 us; 3 experts: 103.0 / 94.6 / **91.5** / 95.3 / 97.2),
> which reproduces the 1,000-iteration sweep within 2%.
>
> Worth noting where the 16 was hiding: **this section had it all along**, in
> the parenthesis above. It was never carried back to §13's cost model, which
> went on solving `other` at 8. A fact can be in a document and still not be in
> its model.

> **Corrected again 2026-09-30 (twelfth finding, §12.1c correction).** The
> correction above made the regime error it describes, in reverse.
>
> - **Stock's host layers do run k=8.** Stock keeps each expert bank wholly on
>   the host or wholly on the device. So each of its 32.79 host
>   bank-equivalents (tenth finding) computes all 8 of a token's experts on the
>   CPU, and each device layer computes none. The 6.5-expert token is the
>   split's regime, not stock's. §12.2 priced the right cell; the table above
>   re-priced it in the wrong one.
> - **The expert share of the token is ~39.4%**, not 38.7% or ~47%: 32.79 ×
>   293.3 µs = 9.62 of 24.41 ms (`threads-x-experts`, 16 threads).
> - **16 → 8 threads is +0.5% to +1.5% of the token.** Three invocations have
>   an 8-thread k=8 cell: `pp-shape` (this section's run) +0.46%,
>   `threads-x-experts` +0.95%, `threads-k8` +1.47%. So the ~1.8% was high by
>   1.2–3.9x, not 3.2x. All three ran 8 threads before 16, so even the sign is
>   unproven.
> - **`--threads 12`'s +1.36% is +0.92% at face value, and unproven.** At k=8,
>   12 threads ties 8 (286.45 vs 286.2 µs). Its lead at k=2–4 does not reach
>   stock. The 5,000-iteration run measured k=2 and k=3 only, so it confirms
>   nothing about stock. And `pp-threads-fine` favours 16 over 12 at k=8.
> - **The PP columns are CPU-path only.** From 32 tokens per call, stock copies
>   the host experts to the GPU and computes them there (ninth finding, §12.5).
>   The 32/128/512-token columns price `--no-op-offload`, not stock prefill, so
>   `--threads-batch` is expected to move stock prefill by nothing measurable.
> - `tools/thread-optimum.py` prints every figure here from the bench JSONs,
>   with each invocation's thread order read off its file.

### 12.2a MEASURED 2026-09-30, RETRACTED as a stock lever: `--threads-batch 16` is a local pessimum of the CPU expert cell ~~, and this is the largest free win in the document~~

> **RETRACTED 2026-09-30 (ninth finding, §12.5).**
>
> - **What stands:** the sweep. 16 threads is a local pessimum of the CPU
>   expert-FFN cell, and 26–30 threads form a plateau about 20% below it. The
>   ubatch-shape table stands too.
> - **The anchor never runs that cell at prefill.** At a batch of 32 tokens or
>   more, op offload copies the host experts to the GPU. `--threads-batch` then
>   drives only the embedding lookup and block 0's state-bound ops. Its effect
>   on stock prefill is expected to be nothing measurable (unmeasured).
> - **The 1.19x prices `--no-op-offload` only.** On the §12.5 copy model that
>   path is lever (c): 1.25–1.44x at 30 threads, and optimistic. The ubatch
>   lever (a) beats it from u1024 up (1.74–1.92x).
> - **"The thread lever strictly dominates the ubatch lever" is retracted.**
>   Under stock the ubatch is the lever: the copy is paid once per ubatch, so a
>   larger ubatch cuts it directly. The ~56 us/token floor below applies only on
>   the CPU path.
> - **The recommendation `--threads 12 --threads-batch 28` is withdrawn.** The
>   `-tb 28` half moves nothing measurable under stock. The `--threads 12` half
>   rests on a TG sweep whose sign is unsettled (§12.1c correction).
> - **The `REPACK` caveat is closed (§12.4).** Override tensors are never
>   `CPU_REPACK`.
> - **The decisive run is the ubatch A/B** (§12.5 A/B vehicle, item 1). Its
>   `-t 16,28` arm checks that threads move nothing.

§12.2 compared **8 against 16** and stopped. The box is 16c/32t, prefill is
*compute*-bound (§12.2's own finding), and §12.3 shows prefill is **79.4% host
expert FFN** — so the thread count on the compute-bound side of the cheapest
knob available governs four-fifths of the charter's first priority, and nine of
its candidate values had never been tried.

k=8, 512 tokens per call, host-only, no GPU, no mutex. Evidence:
`build/expert-cpu-bench/pp-threads{,-fine,-top}.{json,log}`, three separate
invocations, medians in us:

| threads | runs | per-run medians | spread | vs 16 |
| --- | --- | --- | --- | --- |
| 8 | 1 | 40,422 | — | +11.8% |
| 10 | 1 | 35,185 | — | −2.7% |
| 12 | 2 | 31,103 / 31,326 | 0.7% | **−13.7%** |
| 14 | 1 | 38,237 | — | +5.7% |
| **16** | 3 | 36,118 / 36,362 / 36,000 | 1.0% | — *(the recorded run)* |
| 20 | 2 | 33,148 / 32,970 | 0.5% | −8.6% |
| 24 | 3 | 30,933 / 30,777 / 30,914 | 0.5% | **−14.6%** |
| 26 | 1 | 29,524 | — | −18.4% |
| 28 | 2 | 28,913 / 29,011 | 0.3% | −19.9% |
| **30** | 1 | 28,695 | — | **−20.6%** |
| 32 | 2 | 31,257 / 31,083 | 0.6% | −13.8% |

**16 threads is a local pessimum.** It beats 8, which is all §12.2 asked, but
26/28/30 form a **plateau within 3% of each other** at ~20% below it, and 12
beats it by 13.7%. **32 is 8.6% worse than 30**, so "use every hardware thread"
is also wrong — the last pair contends with the main thread and the OS. The
16-thread cell reproduces §12.2's own 36,594 us to 1.2%, so this is a dimension
that was never swept, not drift.

Carried into §12.3's prefill accounting (`tools/thread-optimum.py`), holding
everything else at §12.3's measured values:

| `--threads-batch` | us/layer | s/ubatch | prefill wall | PP tok/s | |
| --- | --- | --- | --- | --- | --- |
| 16 | 36,160 | 1.157 | 152.13 s | 347.20 | the recorded run |
| 12 | 31,215 | 0.999 | 135.80 s | 388.93 | **1.12x** |
| 24 | 30,874 | 0.988 | 134.68 s | 392.18 | **1.13x** |
| **28–30** | 28,695 | 0.918 | **127.49 s** | **414.31** | **1.19x** |

**+19% on prefill for one flag**, no code, no VRAM, no split. Against every
other lever in this document — the streaming path (closed, §15), the row cache
(1.39x ceiling at TG only, and it does not fit at 236k, §13.6), the predictor
(closed, §15.5) — this is the cheapest thing found in the domain, and it lands
on the phase §12.3 identifies as dominant.
*Retracted 2026-09-30: stock prefill does not run this cell, so the +19% prices
`--no-op-offload` only (§12.5 lever (c)).*

Two consequences elsewhere, pointing opposite ways, both worth having:
*(Superseded 2026-09-30 by §12.3's corrected items 2 and 3: the real comparison
is copy vs CPU per bank, and it crosses over between u512 and u1024.)*

- **§12.3 item 2 strengthens.** A 452.98 MB bank at the 10.35 GB/s marginal
  feed costs 43.8 ms against a host cost of 36.2 ms at 16 threads — host wins by
  only **1.21x**, which §12.3 flagged as close enough to need a `-ncmoe`
  differential rather than arithmetic. At 28–30 threads the host costs 28.7 ms
  and wins by **1.53x**. The "not resident → host" rule at PP is no longer
  marginal.
- **§12.3 item 3 weakens by the same ratio.** A layer moved host→device is worth
  **28.7 ms** per ubatch, not 36.6 — VRAM buys **21% less** at prefill than
  §12.3 credited it with. That is the right direction for coherence with §13.6,
  which finds we cannot afford the VRAM anyway.

~~**Recommendation, superseding §12.2's:** **`--threads 12 --threads-batch 28`**.
Both halves changed; 8/16 was the wrong pair in both phases.~~
*Withdrawn 2026-09-30 (see the retraction at the top of this section).*

**The same sweep closes the ubatch question, which had only ever been priced
downward.** §11.7.2 measured that ubatch **256** costs 30.6% of prefill and
killed it. Nobody priced **1024 and up** — it was ruled out on VRAM (§11.7: the
compute buffer scales with ubatch) without anyone asking what it would have
bought. Per-token-per-layer expert FFN cost, `ubatch-shape.json`:

| tokens per call | 128 | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- | --- |
| **16 threads**, us/token | 75.8 | 76.0 | 70.4 | **56.5** | 56.1 |
| **28 threads**, us/token | 60.1 | 64.6 | **56.9** | 56.9 | 54.8 |

There is a **compute floor near 56 us/token/layer, and threads and ubatch are
two routes to the same floor, not two gains that add.** At 16 threads you need
ubatch 1024 to reach it; at 28 threads ubatch 512 is already there, and going to
1024 buys **nothing** (56.9 → 56.9). One route costs VRAM this budget does not
have (§13.6); the other costs a flag. ~~**The thread lever strictly dominates the
ubatch lever.** Ruling 1024 out on VRAM was the right call for a reason nobody
had checked — and it means §11.7's entire compute-buffer fight was over
something recoverable for free.~~
*Retracted 2026-09-30 (§12.5).* The floor is a CPU-path fact. Stock prefill
copies the host experts instead, so the ubatch is the stock lever: u1024 is
expected at 1.74–1.92x. §11.7's compute-buffer fight was over that lever, and
what u1024/u2048 costs in VRAM stays unknown until `llama-fit-params` runs.

**Caveats, stated rather than buried.** The bench gives the CPU backend the
whole box to run expert FFN alone. In a real prefill the same threads also run
attention, the router and the shared expert, and must drive the 8 device-
resident layers. **At 28–30 of 32 hardware threads that oversubscription risk is
real and sharper than anywhere else in this document** — a plateau measured in
isolation can become a cliff under contention. The 12/24 rows are the
conservative fallbacks at 1.12x/1.13x and sit well clear of SMT saturation. The
projection also holds the **non-expert remainder fixed**: at 16 threads the
accounting leaves 1.474 − 1.157 = **0.317 s per ubatch** that is not host expert
FFN, and the 1.19x assumes none of it moves. Host attention is inside that
remainder and would likely *also* speed up, which makes 1.19x conservative;
thread starvation of the device-driving path would cut the other way. Neither is
modelled. The §12.4 `REPACK` caveat also applies: if the real path repacks Q4_K
weights, all these host costs are upper bounds. **One stock `llama-server`
prefill at `--threads-batch 28` falsifies or confirms the whole section**, and it
is the highest value-per-GPU-minute run outstanding.
*Corrected 2026-09-30:*
- **The `REPACK` caveat is closed (§12.4).** Override tensors are never
  `CPU_REPACK`, so these host costs are not upper bounds on that account.
- **A stock `-tb 28` prefill is now predicted to show nothing measurable.**
  §12.5 explains why, so the run no longer tests this section. The
  highest-value run is the ubatch A/B (§12.5 A/B vehicle, item 1).

### 12.3 RETRACTED 2026-09-30 — ~~PREFILL IS HOST-EXPERT-BOUND~~: stock prefill is COPY-bound (§12.5)

> **RETRACTED 2026-09-30 (ninth finding, §12.5).**
>
> - **The cell is real; the anchor never runs it.** 36.6 ms is a real
>   measurement of the CPU computing one bank over a 512-token ubatch. But at a
>   batch of 32 tokens or more, CUDA op offload copies every host weight to the
>   GPU and computes there.
> - **The 1.474 s per ubatch is a copy.** It is a PCIe copy (C = 1.23–1.43 s)
>   plus device compute (g ≤ 0.20 s). The copy is 86–100% of a steady ubatch.
> - **The 79.4% share is withdrawn.** The "host expert FFN" share below does not
>   describe the anchor.
> - **The charter's reframing is inverted.** PP speed on this box is a
>   *host-to-device copy* problem. Layer-major residency solves exactly that
>   problem: it copies each bank once per prompt instead of once per ubatch.
> - **What stands:** item 1. Items 2 and 3 are corrected in place.

This cell had never been measured. At the owner's `--ubatch-size 512`, one
layer's expert FFN on the host costs **36.6 ms** (16 threads, the thread count
the recorded run used). Against the recorded 52,819-token prefill:

> **36.6 ms is a 16-thread figure, and 16 is a local pessimum — §12.2a.** At
> 28–30 threads the same cell costs **28.7 ms**. Everything below is correct as
> an accounting of *the recorded run*, and the 79.4% share is what makes the
> thread count matter, but the absolute per-layer cost is 21% high and the
> prefill wall it implies is beatable by 1.19x with one flag. The two numbered
> consequences below move in opposite directions under that correction — item 2
> strengthens (host beats the link by 1.53x, not 1.21x), item 3 weakens (a layer
> moved to device is worth 28.7 ms, not 36.6). §12.2a carries both.
>
> *Superseded 2026-09-30: this note, its 28.7 ms and its 1.19x price
> `--no-op-offload` only, a path the anchor did not run (§12.5).*

| quantity | value | source |
| --- | --- | --- |
| prefill wall | 152.13 s | measured, `llamacpp 8gb vram base.md` |
| ubatches (52,819 / 512) | 103.2 | derived |
| wall per ubatch | **1.474 s** | derived |
| host-resident expert layers | 32 of 40 | fit log: 41 fitted, 33 overflowing |
| ~~host expert FFN per ubatch~~ | ~~32 x 36.6 ms = 1.171 s~~ | ~~measured x derived~~ — not run by stock |
| ~~share of prefill wall~~ | ~~79.4%~~ | ~~derived~~ — withdrawn |
| host expert **copy** per steady ubatch | C = 1.23–1.43 s of T = 1.43 s | derived, §12.5 copy model |
| **copy share of a steady ubatch** | **86–100%** | derived, §12.5 |

~~**Roughly four-fifths of prefill is the CPU computing expert FFNs.** Not the
link, not the GPU, not the scheduler. This reframes the charter's first
priority: PP speed on this box is a *host expert compute* problem.~~
**Corrected: most of prefill is the link.** It is the copy of host experts, paid
again on every ubatch. PP speed on this box is a *host-to-device copy* problem
(§12.5).

Three things follow, and the third is the important one.

1. **Per-expert row caching cannot help prefill, and now we know why in
   numbers.** At 512 tokens per micro-batch essentially the whole 256-expert
   bank is demanded, so per-expert residency degenerates to per-layer
   residency. This confirms T-GPLR-26's "no depth changes prefill" from the
   cost side rather than the telemetry side.
2. **Streaming the bank loses at prefill too.** A 452.98 MB bank at the
   **marginal** feed measured on this box (10.35 GB/s, §15.2) costs **43.8 ms**,
   against **36.6 ms** to compute it on the host. The host wins by **1.20x**.
   **The "not resident -> host" rule therefore holds at PP as well as at TG** —
   previously it was argued only at TG, and prefill was left as an open flank.
   It is now closed on measurement, in the same direction.
   *(Corrected 2026-09-30: stated as 54.8 ms and 1.50x against the 8.26 GB/s
   instrumented figure. The faster measured feed cuts the margin from 1.50x to
   1.20x. It does not reverse the sign, but it is close enough that a
   PP-specific `-ncmoe` differential should settle it rather than arithmetic —
   see the §12.4 `REPACK` caveat, which pushes the same direction.)*
   *(Corrected again 2026-09-30, §12.5.)*
   - **Stock never runs the host leg this item prices.** At a batch of 32 or
     more, op offload copies the bank to the GPU instead. The real choice is
     copy vs CPU per non-resident bank per ubatch (§12.5 table).
   - **At u512 the CPU wins, but narrowly.** Against the pageable copy it wins
     by 1.04–1.21x at 16 threads (36.2–36.6 ms) and 1.30–1.52x at 28 threads.
     Against a pinned copy (31.5 ms) it wins by 1.09x at 28 threads, and loses
     by 1.15x at 16 threads.
   - **From u1024 the copy wins, and its lead grows with the ubatch:** 1.33–1.55x
     pageable and 1.85x pinned at u1024; 2.56–2.98x and 3.56x at u2048.
   - **So "not resident → host" at PP holds only at u512**, and only with
     `--no-op-offload`. Above u512 the correct rule is "not resident →
     copy, with a larger ubatch".
   - **The `REPACK` caveat is closed (§12.4).** Override tensors are never
     `CPU_REPACK`, so it pushes nothing.
3. **Every expert layer moved from host to device is worth ~36.6 ms per
   512-token micro-batch.** That is the single highest-value quantity in the
   domain, it is bought with VRAM alone, and it is exactly what the VRAM budget
   denies. It is also why §11.7.5's context lever (26 -> 30 layers at zero PP
   cost) matters more than its modest framing suggests: 4 layers x 36.6 ms is
   ~146 ms per ubatch, ~10% of prefill wall.
   *(Corrected 2026-09-30, §12.5: same size, different mechanism.)*
   - **Stock saves a copy, not a CPU compute.** A bank moved to the device saves
     its per-ubatch copy: 37.6–43.8 ms of a u512 ubatch (2.6–3.1%).
   - **The context lever is unchanged in size.** Four banks save 150–175 ms per
     u512 ubatch, 10.5–12.2% of a steady 1.43 s ubatch. That matches the ~10%
     above.
   - **The value per token falls as the ubatch grows.** The copy is paid once
     per ubatch whatever its size, so at u2048 the same bank saves a quarter as
     much per token.
   - **"The single highest-value quantity" is withdrawn.** The ubatch lever
     (§12.5 (a)) cuts every bank's copy count at once and needs no VRAM for
     weights. Layer-major residency removes the per-ubatch copy altogether.

### 12.4 Fidelity caveat, stated rather than buried

The bench drives the ggml CPU backend against Q4_K bytes read straight from the
GGUF. llama.cpp's build reports `REPACK = 1`, and host-resident quantized
weights may be repacked into an AVX512-friendly layout at load, which this
bench does not do. **If repacking is active on the real path, these host costs
are upper bounds.** The TG side argues against a large effect — 11.55 ms of a
24.41 ms token is 47%, and the `-ncmoe 20` whole-model differential sits 1.49x
above the microbench, which is the expected overhead gap and leaves little room
for a missing 2x. The PP side has no such cross-check, so **§12.3's 79.4% is
the weakest number in this section** and should be confirmed against a real
`-ncmoe` differential at prefill before it is leaned on hard.

> **Closed 2026-09-30 from code — §12.5.** Override tensors are never
> `CPU_REPACK`. Separately, §12.5 shows that stock prefill does not compute host
> experts on the CPU at all.

### 12.5 CORRECTED 2026-09-30: stock prefill is a COPY schedule — the levers are ubatch and pinned memory, not threads

> **Ninth error in this document's record. It retracts §12.2a's recommendation
> and §12.3's headline.** Both read the anchor's 152.13 s prefill as the CPU
> computing the host-resident expert banks, and both priced `--threads-batch`
> against that. Stock llama.cpp does not take that path.
>
> When an op's weight sits in a host buffer and the batch is 32 tokens or more,
> the CUDA backend's *op offload* claims the op. The scheduler copies the weight
> to the GPU and runs the op there. At `--ubatch-size 512` every ubatch copies
> ~14.8 GB of host expert bytes across PCIe. The CPU computes none of them, and
> `--threads-batch` moves no host expert.
>
> Stock prefill is therefore a copy schedule. The copy is paid once per ubatch,
> so the levers are:
>
> - the ubatch size, which amortises the copy;
> - pinned host memory, which speeds it up;
> - not threads.
>
> The host-FFN cells of §12.2a/§12.3 stand as measurements, but they price
> `--no-op-offload` only. Everything below is re-derived from raw logs and code
> by three scripts:
>
> - `tools/thread-optimum.py` for the copy model;
> - `tools/mtp-price.py` for MTP;
> - `tools/bank-bytes.py` for placement.
>
> None of it is a new GPU measurement.

> **Measured 2026-10-01 (nineteenth and twentieth findings, §12.6): the copy
> model holds, but two of its three lever predictions and the byte/rate
> figures below do not.** The model `T = C + g` (used experts only, `sched
> copies = 1`) reproduces the real server's per-ubatch time on the pinned
> (`--load-mode none`) runs to within 4.2% at u512, u2048 and u4096. Lever (a)
> is **confirmed**. Lever (b) is **falsified**: `--load-mode none` gained 1.504x
> on the real 50k prompt (1.585x in the bench), not the "bound" of 1.16-1.33x.
> Lever (c) is **falsified** in the bench (about 1.03x at best, not 1.25-1.47x)
> and is dominated by (a) and (b) either way. `--threads-batch` moved nothing.
> The "254 of 256 experts" and "14.78 GB" figures were wrong for the 50k
> anchor prompt (71.7% of experts, 10.56 GB per u512 ubatch), and the "implied
> 10.3-12.0 GB/s" pageable rate was a consequence of that wrong volume. Each
> claim is struck or annotated in place below; the evidence is §12.6.

**Mechanism, read from code in both trees.** The first column is the vendored
base `04b569142`. The second is `e5a8d439c`, the upstream commit that the
anchor's build 10892 reports (public source, fetched raw).

| step | vendored | `e5a8d439c` |
| --- | --- | --- |
| an op whose weight sits in a host buffer goes to the GPU if the GPU backend asks for it | `ggml-backend.cpp:956-964` | `967-974` |
| a weight from an incompatible backend starts a new split once the split has inputs ("the memory of the previously offloaded weights can be reused") | `1314-1344` | `1331-1338` |
| the copy is a full-size tensor of the weight in the compute buffer, allocated at split start | `1375-1420`, `1486` | `1512` |
| before copying: wait on an event, else `ggml_backend_synchronize` | `1622-1627` | `1683-1688` |
| `MUL_MAT_ID`: read the expert ids back and synchronize, then copy only the used experts' rows | `1629-1716` (ids `1660-1663`) | `1690`, ids `1718-1721` |
| `n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1`; events exist only if `n_copies > 1` | `1807`, `1840-1842` | `1865`, `1900` |
| batch size the CUDA backend sees: `MUL_MAT_ID` → `ne[2]`, `GET_ROWS` → 0, else `ne[1]` | `ggml-cuda.cu:5456-5458` | `5521-5524` |
| offload threshold: batch ≥ 32 | `ggml-cuda.cu:5645` | `5710` |

Three consequences follow from this code without any measurement:

- **No overlap.** All three logs below report `sched copies = 1` (anchor line
  308), so no events exist. Every copy waits on a full synchronize, and every
  `MUL_MAT_ID` synchronizes again to read its ids. Copy and compute run
  serially, so a ubatch costs T = C + g.
- **One weight live at a time.** The copies share the compute buffer. That is
  how the anchor's 427.53 MiB CUDA0 compute buffer (line 304) holds one host
  weight beside the activations. The largest host weight is ≤ 210 MiB (the Q6_K
  downs of layers 34/38/39).
- **More than experts.** The default batch rule reads `ne[1]`, so block 0's
  dense matmuls and norm `MUL`s are offloaded too. The CPU keeps only what
  cannot move:
  - the embedding lookup (`GET_ROWS` → 0);
  - block 0's ops on its CPU-resident recurrent state (anchor line 296).

**Log evidence: the splits op offload adds.** Op offload is the only backend
choice that depends on the batch size. So the gap between the graph splits a log
reserves at a 512-token batch and at one token counts op offload's splits. The
gap grows with the number of host-weight tensors:

| run | whole blocks on the CPU | nodes | splits at 512 | splits at 1 | `sched copies` |
| --- | --- | --- | --- | --- | --- |
| anchor, `llamacpp 8gb vram base.md:306-308` | block 0 | 3907 | 119 | 70 | 1 |
| MTP run, `out/llamacpp-baseline.log:320-322` | blocks 0..7 | 4357 | 263 | 76 | 1 |
| 2x262k, `llamacpp 8gb vram base 2x 262k context q8.md:294-296` | blocks 0..15 | 3927 | 410 | 82 | 1 |

**The copy model, from the anchor's own cadence.** The anchor logs one progress
line per 8,192-token chunk (lines 360-378). The server's checkpoint rule
(`tools/server/server-context.cpp:3465-3484`) splits the prompt `n_ubatch + 4`
tokens and 4 tokens before its end. The script refuses to price anything unless
that rule reproduces the logged chunks, and it does:

| tokens | seconds | ubatches ≥ 32 tokens | s per ubatch | note |
| --- | --- | --- | --- | --- |
| 322 | 4.77 | 1 | 4.770 | first call: mmap page-in |
| 8192 | 21.78 | 16 | 1.361 | |
| 8192 | 21.21 | 16 | 1.326 | |
| 8192 | 21.56 | 16 | 1.347 | |
| 8192 | 24.16 | 16 | 1.510 | |
| 8192 | 24.66 | 16 | 1.541 | |
| 8192 | 24.55 | 16 | 1.534 | |
| 2829 | 7.94 | 6 | 1.323 | |
| 512 | 1.40 | 1 | 1.400 | |
| 4 | 0.10 | 0 | — | below 32 tokens: runs on the CPU |

The page-in call and the 4-token tail are kept as logged. The 103 full ubatches
between them took 147.26 s for 52,493 tokens:

window = 103·C + (52,493/512)·g

where C is the copy per ubatch and g is the device compute per 512 tokens. g is
not logged, but it has an upper bound. The research engine computes the same
layers at 2,532.5 tok/s steady (§11.7.2), so g ≤ 512/2,532.5 = 0.2022 s. g
likely sits near that bound: the research engine's own compute is 189.6 ms per
chunk (§11.7.2).

| g, device compute per 512 tokens | C, copy per ubatch | T = C + g | implied copy rate |
| --- | --- | --- | --- |
| 0 (no compute) | 1.4297 s | 1.4297 s | 10.34 GB/s |
| 0.05 s (ASSUMED, optimistic) | 1.3799 s | 1.4299 s | 10.71 GB/s |
| 0.2022 s (bound) | 1.2285 s | 1.4306 s | 12.03 GB/s |

**Copy bytes per ubatch.** At u512 a ubatch copies two things:

- the used experts' rows of every host bank: ~~254 of 256 on average (route
  unions, §15.3)~~ **Corrected 2026-10-01 (twentieth finding): 71.7% of the 256
  experts on the 50k anchor prompt** (all-layer mean; 73.4% weighted by host
  bank bytes), 81.5% at u2048 and 85.5% at u4096 (§12.6). The 254-of-256 figure
  came from a 512-token QA prompt, and route-union width depends on the text;
- 42,885,888 B copied whole: block 0's dense weights and
  `blk.8.ffn_down_shexp`.

~~That totals **14.78 GB**. From u1024 all 256 experts are used, so 14.90 GB.~~
**Corrected: 10,075 MiB (10.56 GB) per u512 ubatch on the 50k anchor prompt,
12,014 MiB at u2048 and 13,131 MiB at u4096** (§12.6, measured route unions
from the layer-major engine's own route telemetry; the sampled route events of
pass 1 and pass 2 are identical: 400/400, 120/120 and 80/80).

**The copy against the wire.** The measured pinned wire rate is 14.38 GB/s: one
bank in 31.5 ms, `copy_gpu_ms` p50, `AGENT_ENVIRONMENT.md:1214`. ~~A u512 ubatch
therefore needs at least W = 1.028 s on the wire. C ≥ W holds at every g, and the
implied 10.3–12.0 GB/s is consistent with copying from a pageable (mmap)
source.~~ ~~**The 52.8k-token prompt moved ~1.54 TB across PCIe** to use 14.85 GB of
host experts.~~ A layer-major pass copies them once.

> **Corrected 2026-10-01 (twentieth finding).** With the measured 10.56 GB per
> u512 ubatch the wire needs W = 0.734 s. The anchor-prompt server run on the
> recorded configuration (S0 in §12.6) took T = 1.423 s per ubatch. Taking g at
> 0.21 s (the layer-major engine's 0.170 s plus the 0.039 s by which the pinned
> stock run exceeds its own prediction at u512) leaves C of about 1.21 s. That is
> a **derived pageable (mmap) copy rate of about 8.7 GB/s**, against 14.35 GB/s
> pinned. C is not bounded below by the pinned W, because the pageable path is
> far slower than the wire; that gap is the whole of lever (b)'s gain. The
> "implied 10.3–12.0 GB/s" came from dividing by the wrong 14.78 GB. A 50k
> prompt at u512 moves about 1.0 TB (98 ubatches), not 1.54 TB. The 8.7 GB/s
> figure is derived from S0's T and an estimated g, not measured directly.

**The levers.** Speedup is over the recorded prefill, and the range spans g =
0.05–0.2022 s. *Steady* is per full ubatch. *This prompt* keeps the page-in
and the tail as logged. The last column was added 2026-10-01 from the
measurements in §12.6 (server = the real 50k anchor prompt through
`llama-server`; bench = `llama-bench` on an 8,192-token random-token prompt,
whose per-ubatch time runs 0.18–0.25 s below the server's on the same flags, so
the server value is the one to trust where both exist).

| lever | status | steady | this prompt | measured 2026-10-01 |
| --- | --- | --- | --- | --- |
| as recorded: `-ub 512`, mmap | logged | 1.00x | 1.00x | server S0 359.83 tok/s; bench 418.4 / 419.4 |
| (a) `-ub 1024` | expected | 1.74–1.92x | 1.69–1.85x | **confirmed**: bench 1.749x (731.6 vs 418.4) |
| (a) `-ub 2048` | expected | 2.80–3.60x | 2.63–3.30x | **confirmed**: bench 2.983x (1,248.2 vs 418.4) |
| (a) `-ub 4096` | expected | 4.01–6.39x | 3.65–5.42x | **confirmed**: bench 4.717x (1,973.8 vs 418.4) |
| (b) `--load-mode none` | bound | 1.16–1.33x | 1.16–1.31x | **falsified**: server 1.504x (541.2 vs 359.8); bench 1.585x (664.8 vs 419.4) |
| (a)+(b) `-ub 2048 --load-mode none` | bound | 3.10–4.63x | 2.90–4.12x | **confirmed, at the upper edge**: server 4.011x (1,443.2 vs 359.8, request 1 of 2; request 2 was contaminated by neighbour CPU load); bench 4.169x (1,748.8 vs 419.4) |
| (a)+(b) `-ub 4096 --load-mode none` | not predicted | — | — | server **5.733x** (2,063.1 vs 359.8; two requests, 2,065.3 and 2,061.0) |
| (c) `--no-op-offload -tb 30` | optimistic | 1.25–1.44x | 1.25–1.43x | **falsified in the bench**: 1.029x at `-tb 28` (431.6 vs 419.4), 0.936x at `-tb 16`; no server run |
| (c) `--no-op-offload -tb 28 -ub 2048` | optimistic | 1.27–1.47x | 1.27–1.46x | **falsified in the bench**: 1.073x (450.1 vs 419.4), and 0.364x of mmap at u2048 |

- **(a) holds only if the placement holds.** The compute buffer grows with the
  ubatch, and under the anchor's `--fit-target 16000` the fitter re-fits. Every
  bank it pushes to the host costs 37.6–43.8 ms of a u512 ubatch (2.6–3.1%) and
  0.175 ms of a TG token (0.72%). The compute buffer's VRAM at u1024/u2048 is
  **unknown** until `llama-fit-params` runs on the GPU.
  - **Measured 2026-10-01 (S4, server, u4096, q8 KV, `-c 128000`, fit-target
    16000):** the fitter did re-fit and the process peaked at **7.342 GiB** (CUDA0
    model 3,749 MiB, KV 1,328 MiB, compute buffer 1,873 MiB; `CUDA_Host` model
    16,165 MiB). Lever (a) held under that placement at 2,063 tok/s with
    `--load-mode none`. TG was 37 tok/s against 41 at u512.
- ~~**(b) is a ceiling**, because C cannot fall below W.~~ **(b) was not a
  ceiling (twentieth finding).** C was bounded below by the *pinned* wire time
  while the recorded run copies from *pageable* memory at about 8.7 GB/s (see
  "The copy against the wire"). llama.cpp names this lever itself, at anchor
  line 245: `tensor overrides to CPU are used with mmap enabled - consider using
  --load-mode none for better performance`.
  - It page-locks the host model buffer: 14,722 MiB (14.4 GiB) at the anchor's
    placement (bank-bytes), of 63,166 MiB RAM.
  - **Pinning that much is proven feasible on this box.** The 2x262k run pinned
    18,602.70 MiB of `CUDA_Host` model buffer (its line 257) with
    `--load-mode none` (its line 232).
  - That run is also the one pinned data point. Its first 353-token chunk took
    1.5675 s (its lines 350→351) and copies 17.41–18.97 GB. At the 14.38 GB/s
    wire rate that is 1.21–1.32 s, which leaves 0.25–0.36 s for compute and
    the CPU work of 16 whole blocks. ~~At the anchor's pageable 10.34–12.03 GB/s
    the same copy would take 1.45–1.83 s, which fits only at the corner of both
    ranges.~~ It is consistent with (b), but it is not an A/B and bounds
    nothing. (That chunk's 17.41–18.97 GB rests on the same 254-of-256
    assumption as the struck "Copy bytes per ubatch" figure, so it is probably an
    over-count for a real prompt; it was not re-derived.)
- ~~**(c) is optimistic.**~~ **(c) is falsified in the bench and dominated by
  (a)+(b) either way (nineteenth finding).** It used the host-FFN bench cells of
  §12.2a. Those give the expert FFN the whole box and leave out block 0's dense
  ops, the activation traffic and the Q6_K downs. It is dominated by (a) above
  u512.
  - Caveat the bench cannot close: its per-ubatch time sits 0.18–0.25 s below
    the server's on identical flags (u512 mmap 1.221 against 1.423 s; u512
    pinned 0.770 against 0.946 s; u2048 pinned 1.171 against 1.419 s). The
    offset is roughly constant, not proportional to copy volume. Candidate
    causes are attention depth (8,192-token bench against a 50k prompt), server
    per-ubatch overhead and the route width of random tokens; they are not
    separated. Under a constant offset `--no-op-offload` stays near 1.03x of the
    recorded configuration on the real prompt; under a route-width cause it
    could reach about 1.2x. Neither is measured, because no `--no-op-offload`
    server run existed when this was written. It does not change the verdict:
    pinned op-offload at u512 already gives 1.504x (server) and u4096 gives
    5.733x.
- **`--threads-batch` alone moves nothing measurable here.** At a batch of 32
  or more no host expert runs on the CPU. Its only prefill work is the
  embedding lookup and block 0's state-bound ops. ~~That is expected to be small
  but is unmeasured; one `-t` arm checks it.~~ **Measured 2026-10-01 (bench,
  `--load-mode none`):** `-t 16` against `-t 28` is 653.6 against 644.6 tok/s at
  u512 (−1.4%) and 1,756.8 against 1,752.6 at u2048 (−0.2%).

**Copy against CPU, per non-resident bank per ubatch.** This is new, and it is
the rule §12.3 item 2 tried to state. The copy costs about the same at every
ubatch, but CPU compute grows with the ubatch:

| ubatch | copy, pageable / pinned | CPU compute (`--no-op-offload`) | cheaper |
| --- | --- | --- | --- |
| 512 | 37.6–43.8 / 31.5 ms | 36.2 ms (16 threads); 28.9–29.0 ms (28) | CPU. At 28 threads 1.30–1.52x vs pageable and 1.09x vs pinned; at 16 threads 1.04–1.21x vs pageable |
| 1024 | same | 58.3 ms (28 threads) | copy: 1.33–1.55x pageable, 1.85x pinned |
| 2048 | same | 112.2 ms (28 threads) | copy: 2.56–2.98x pageable, 3.56x pinned |

The crossover sits between u512 and u1024:

- At the anchor's u512 the CPU is the cheaper home for a non-resident bank,
  which is the only reason (c) gains anything.
- From u1024 up the copy wins, and its lead grows with the ubatch. So op offload
  should stay on and the ubatch should grow.

The CPU figures are the bench's whole-box cells (`ubatch-shape.json`, the prefill
cell in `build/expert-cpu-bench/`), so they flatter the CPU.

> **Corrected 2026-10-01 (twentieth finding).** The copy columns above price a
> FULL bank. A bank's used rows are 73.4% / 83.1% / 86.5% of it at u512 / u2048 /
> u4096 on the 50k anchor prompt (byte-weighted over host banks, §12.6). A
> pinned copy therefore costs 31.6 ms × f = **23 / 26 / 27 ms** per bank, which
> is already below the CPU's 28.9 ms at u512. A pageable copy at the derived
> 8.7 GB/s costs about 38 / 43 / 45 ms. So the "crossover between u512 and
> u1024" holds only for pageable memory: with pinned memory the copy wins at
> every ubatch from u512, which is why lever (b) beat lever (c) in the
> measurements. The per-bank figures are derived from measured route unions and
> the measured pinned wire rate, not timed per bank.

**What layer-major is worth against tuned stock.** ~~2,532.5 tok/s steady against
stock's steady 357.9 tok/s is **7.08x**, but that stock is untuned. Against stock
at `-ub 2048` it would be **1.97–2.53x**. That figure is unproven, because lever
(a) is expected, not measured. §9.1's 7x stands only as a measurement against the
recorded configuration.~~ **Replaced 2026-10-01 (twenty-first finding).** The
7.08x compares layer-major at u512 (2,532.5 tok/s) with the recorded stock at
u512, mmap (357.9 tok/s): it prices the stock *configuration*, not the stock
*engine*. Tuned stock, measured on the 50k anchor prompt: recorded flags 359.8
tok/s (S0); `-ub 4096 --load-mode none` 2,063 tok/s (5.73x S0). The comparison
that matters is same-session with matched KV. The first version of that,
ab1, q8_0 KV, ctx 128,000, the 50,000-token prompt, u4096 on both sides, put
layer-major at **3,668.6 tok/s against 2,117.3 for tuned stock, reported as
~~1.733x~~**. **That figure is superseded 2026-10-01 (§12.6.6): it is still a
comparison at unequal VRAM, and at ctx 128,000 the right stock cell is the
ubatch that fills the same budget, not u4096.** Against `-ub 8192` at
`--fit-target 16000` — the tuned stock baseline at about 7.44 GiB — the same-session
figures are **1.40x** the anchor (3,657.5 against 2,610.7 tok/s at 7.472
against 7.438 GiB) and **1.43x** the vendored engine (against 2,554.8). The
advantage **inverts** once stock is allowed to fill the card: at
`--fit-target 300` layer-major is **0.756x** the anchor (3,657.5 against
4,837.6 at 22.523 GiB) and **0.795x** vendored. So the honest statement is not
a single multiplier but a budget curve: layer-major is ahead at 7-8 GiB, level
at about 19 GiB, and behind once stock is fully resident. **Measured** (ab8,
ab2; sources and per-arm layer/KV placement in §12.6.6). The 1.97-2.53x
projection was for stock at u2048 and was never tested in one session.
Cross-session layer-major numbers differ by up to 16% (u512: 2,532.5 in
§11.7.2 against 2,937 in the 2026-09-30 ubatch sweep; different days and
binaries, cause not isolated), which is why only same-session ratios are used
in §12.6.

**Tenth finding: the fitter and the loader disagree on the first GPU block.**
The two count layers differently:

- **Fitter.** Without MTP it counts graph layers only, and adds the nextn block
  only when `load_mtp` is set (`common/fit.cpp:139-141`; `139-142` at
  `e5a8d439c`).
- **Loader.** It counts every block: `i_gpu_start = max(n_layer_all + 1 - ngl,
  0)` (`llama-model.cpp:1354`; `1494`).

So the fitter's `ngl` lands one block short:

- **Anchor.** The fitter chose (41, 33, GATE) to put every block on the GPU, but
  the loader put **block 0 wholly on the CPU**. bank-bytes reproduces the logged
  CUDA0 5,192.44 MiB only with block 0 on the CPU; the fitter intended
  5,664.28 MiB.
- **2x262k.** The fitter placed block 15 on the GPU, but it runs on the CPU.
  bank-bytes reproduces 1,311.95 / 18,602.70 MiB exactly.
- **MTP run.** `common.cpp:1689` sets `load_mtp`, so both sides count the nextn
  block and agree: blocks 0..7 are on the CPU as fitted.

Upstream at `e5a8d439c` has the same code. It is not reported upstream
(AGENTS.md).

The cost of the mismatch:

- At prefill, block 0's 41.8 MB of dense weights ride every copy: ~0.28% of a
  ubatch.
- At TG the CPU reads those weights every token instead of the GPU reading them
  from VRAM, and the graph gains a CPU↔GPU transition. **The fix is expected to
  gain ~2–3% TG, unproven.** That is ≈0.5–0.7 ms of 24.41 ms at an ASSUMED
  60–80 GB/s host read rate.

The fix is a set of server flags:

- The server's `-ot` separates overrides with `,`.
- `-fit off` is required, because the fitter refuses a user-set `-ngl`
  (`fit.cpp:377-378` at `e5a8d439c`).

- **Variant B, recommended**: block 0 on the GPU, with its bank on the host:
  `-fit off -ngl all -ot "blk\.0\.ffn_(up|down|gate)_exps=CPU,blk\.8\.ffn_down.*=CPU,blk\.(9|[1-3][0-9])\.ffn_(up|down|gate)_exps=CPU"`.
  It adds block 0's dense weights and state to VRAM: 41,771,776 B + 2.09 MiB ≈
  42 MiB, against the fitter's 46 MiB headroom (anchor line 95). It fits, by
  ~4 MiB.
- **Variant A, diagnostic**: the same without the `blk\.0` term. It is the
  fitter's intended placement, adds ~474 MiB, and does not fit the 8 GB
  emulation.

Both variants make `full_offload` true (`llama-context.cpp:2931-2943`; `2532` at
`e5a8d439c`). That pins every layer's `norm` and `l_last` to the layer's device
at prefill as well as at TG. It is a PP-only side effect the A/B must read
through.

**MTP: §2's comparison is confounded four ways** (eleventh finding). §2 set the
2026-09-10 run (32.44 tok/s, MTP draft 5, acceptance 0.546) beside the anchor
(40.96 tok/s, no MTP) and concluded MTP was "likely costing" the baseline. The
two runs differ in four ways:

1. **Context.** The MTP run decoded after a 372-token prompt, the anchor after
   52,819 tokens.
2. **Binary.** Build 10876 against 10892.
3. **Placement.** Loading MTP makes the fitter count the nextn block. It also
   reserves the draft context (f16 KV at 262k) before fitting the target
   (`server-context.cpp:1030-1076`). It fitted (34, 33, MOE), with **blocks 0..7
   wholly on the CPU**, against the anchor's block 0.
4. **Draft.** `n_max 5`, and every round drafted all five.

The MTP log does measure the draft mechanism itself:

- 463 rounds: 2,315 tokens drafted, 1,264 accepted, 1,724 delivered.
- Acceptance by draft position: 0.816, 0.661, 0.525, 0.415, 0.313.
- 2.153 ms per drafted token.
- 114.73 ms per round. Of that, 103.97 ms is verify plus bookkeeping, which is
  4.26 T1. That is not a verify cost at the anchor placement.

**Pricing MTP at the anchor's placement.** A verify of d+1 tokens is below the
offload threshold. So the host experts run on the CPU over the union of the
batch's routes, and no copy term enters TG. `tools/mtp-price.py` prices this
from two inputs:

- **The host bench** (`verify-shape.json`), with median-of-slopes fits:
  - 1 token over U experts costs −39.7 + 35.15·U µs;
  - 2/4/6 tokens add 31.2 + 3.40, 51.5 + 3.62 and 25.4 + 4.86 µs per extra
    token-expert pair;
  - four outlier cells are left out.
- **Route unions** of the 4 distinct TG streams among 610 recorded traces.

| draft d | tokens per round N(d) | union U(d+1) | host adds, ms | ms/token | speedup | with 2 more host banks | break-even δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.816 | 13.05–13.98 | 7.18–8.14 | 18.58–19.11 | 1.28–1.31x | 1.24–1.28x | 9.62–10.59 ms |
| 2 | 2.477 | 17.48–19.12 | 13.03–14.74 | 16.85–17.54 | 1.39–1.45x | 1.35–1.41x | 8.50–9.35 ms |
| 3 | 3.002 | 21.38–23.66 | 18.38–20.73 | 16.40–17.19 | 1.42–1.49x | 1.37–1.44x | 7.22–8.01 ms |
| 4 | 3.417 | 25.00–27.77 | 22.94–25.75 | 16.38–17.20 | 1.42–1.49x | 1.37–1.44x | 6.16–6.86 ms |
| 5 | 3.730 | 28.31–31.60 | 27.38–30.64 | 16.77–17.64 | 1.38–1.46x | 1.34–1.41x | 5.04–5.70 ms |

**Status: expected at δ = 0, unproven.** δ is everything else an extra verify
token costs: attention, GPU dense and expert reads, the CPU block, and sampling.
Nothing recorded measures δ on this placement. So the table prints the δ at
which each draft length breaks even: 5.0–10.6 ms per extra token. Cautions:

- **Acceptance** was measured on a 372-token prompt. At 52.8k, and on another
  prompt, it is unmeasured.
- **Draft KV** defaults to f16. `-ctkd q8_0 -ctvd q8_0` keeps its VRAM reserve
  from evicting banks. Each evicted bank costs the verify round; the
  "2 more host banks" column prices two.
- **Host adds are conservative.** h1 sits ~10% under the k=8/9 cells, so "host
  adds" overstates the cost by up to ~0.9 ms per round.
- **Extrapolation.** U up to 31.6 extends the bench's widest union of 29 along
  its fitted line.
- **Control arm.** Loading MTP also realigns the fitter and the loader (the
  tenth finding). So an MTP arm must be read against a **block-0-fixed
  control** (variant B without spec flags), not against the anchor.

**`--threads 12` for TG: retracted as a point estimate.** §12.1c's +1.36%
(41.52 tok/s) was priced in the wrong regime. At k=8, where stock's host layers
run, the same cells give +0.92% (twelfth finding, §12.1c correction). Three
invocations have a 12-thread k=8 cell. Two favour 12, by 2.3% and 17.7% (the
second is `verify-shape`, at id-stride 32). The third, `pp-threads-fine`,
favours 16 by 0.8%. All three ran threads in ascending order. In
`verify-shape` the 12-thread lead fades across the run: it wins its first three
cells (−17.7/−19.6/−44.4%), then only 2 of the next 7, and its per-expert cost
climbs from 27.7 to 35.6 µs. Neither size nor sign is settled. Item 3 of the
A/B vehicle below decides.

**§12.4's `REPACK` caveat: closed from code.**

1. Tensors overridden to the CPU pick their buffer type from `buft_list_cpu`.
   Its order is ACCEL → host (`CUDA_Host`) → extra (`CPU_REPACK`) → CPU
   (`llama-model.cpp:1034-1093` at `e5a8d439c`; vendored `914`/`929`/`945`).
2. So the loader takes `CUDA_Host` first (`llama-model-loader.cpp:1234`;
   vendored `1183-1191`).
3. Under mmap the loader swaps that host buffer for plain CPU, "avoid using a
   host buffer when using mmap" (`1262`; vendored `1212-1220`).

Override tensors are therefore `CPU_Mapped` under mmap, or pinned `CUDA_Host`
under `--load-mode none`. They are **never `CPU_REPACK`**, and the three logs
mention no `CPU_REPACK` buffer. The host bench reads un-repacked Q4_K, as the
real TG path does, so the bench's layout is faithful.

Only `--no-op-offload --no-host` would admit REPACK. That is unmeasured. Even a
2x CPU gain would give about 2.1–2.75x at u512, which is below (a) at u2048.

**A/B vehicle — not run.** These rules apply to every arm:

- **Authorization.** Each arm is a GPU run: owner-authorized, one at a time,
  through `tools/with-gpu-lock.ps1`.
- **Binaries.** Use the anchor's binaries in `C:\Development\AI\llamacppCUDA`
  (build 10892).
- **Placement.** `llama-bench` runs no fitter unless `-fitt` is given
  (`llama-bench.cpp:2286-2287` at `e5a8d439c`). So `-ngl 41` plus the anchor's
  overrides reproduces the anchor's placement, block 0 on the CPU included.
- **Separator trap.** `llama-bench`'s `-ot` separates overrides with `;` and
  test variants with `,`. That is the opposite of the server, so the server
  strings above cannot be pasted.
- **Threads.** `llama-bench -t` sets both thread counts (`llama-bench.cpp:2134`).

1. **Prefill levers (a)/(b)/(c)**, one factor per invocation:

   `llama-bench -m <model> -ngl 41 -ot "blk\.8\.ffn_down.*=CPU;blk\.(9|[1-3][0-9])\.ffn_(up|down|gate)_exps=CPU" -fa on -ctk q8_0 -ctv q8_0 -lm mmap -b 8192 -ub 512,2048 -p 8192 -n 0 -r 2 -o jsonl`

   Then run the same command at `-ub 512` three more times, varying one factor
   each: `-lm mmap,none`, `-nopo 0,1`, `-t 16,28`. A depth-0 prompt runs faster
   than the anchor's 52.8k average: its cadence rises from 1.33 to 1.54 s per
   ubatch with depth. So read arms against each other, not against 347 tok/s.
2. **Block-0 fix at TG.** Run `-p 0 -n 128 -r 5` at `-ngl 41` with the overrides
   above. Compare it with `-ngl 99` with `blk\.0\.ffn_(up|down|gate)_exps=CPU;`
   prepended to the overrides.
3. **TG threads.** The anchor placement with `-p 0 -n 128 -r 5 -t 16,12,8`,
   then again with `-t 8,12,16`. Every host bench so far ran the lower thread
   count first, so the reversed pair cancels run-order drift (§12.1c
   correction). llama-bench runs the variants in list order: `parse_int_range`
   keeps the comma order, and `get_cmd_params_instances` nests `n_threads` in
   that order (e5a8d439c `llama-bench.cpp:293-333`, `1330`). Thread count is
   not a model parameter, so all three variants share one model load
   (`equal_mparams` omits `n_threads`, `llama-bench.cpp:1277-1284`, `2321`).
4. **MTP.** `llama-bench` has no speculative decoding, so this uses
   `llama-server`.
   - The arm is variant B with `|40` added to the last pattern, so the nextn
     block's experts go to the host too. Add `-ctkd q8_0 -ctvd q8_0` and
     `--spec-type draft-mtp`, with `--spec-draft-n-max` at 1, 2 and 3.
   - The control is variant B with no spec flags, on the same prompt.
   - Arm and control share one placement, so the pair measures δ and acceptance,
     not the 8 GB budget. At that budget, the MTP reserve evicts banks, which
     the "2 more host banks" column prices.
5. **VRAM for (a).** Run `llama-fit-params` with the anchor's flags at
   `-ub 1024` and `-ub 2048`. It prints the fitted placement and compute buffers.
   It initializes CUDA, so it also goes through the lock.

---

### 12.6 MEASURED 2026-10-01: the owner's two real configurations, head to head

This subsection is the answer to §12.5's open list. It replaces that list's
A/B vehicle: those items were **never run**, and what came back instead is the
comparison the charter actually asks for — the owner's two recorded server
configurations, stock against layer-major, in one mutex hold each, with VRAM
matched rather than assumed.

Throughout, **LM** means the layer-major prefill engine with expert-bank
residency (`--residency moe-expert-bank --pp-schedule layer-major`), and
**stock** means the anchor server in `C:\Development\AI\llamacppCUDA`
(build 10892). The binary is the owner-authorised side copy
`C:\Development\fafix\bin-v1`, whose `ggml-cuda.dll` carries the **V1**
causal-flash-attention dispatch fix (§12.6.3) behind `FAFIX_CAUSAL_GQA=1`.
Stock gains nothing from V1 — it is a layer-major-side change.

| Owner configuration | Source | Recorded by the owner |
| --- | --- | --- |
| **doc1**, 1 x 262k | `docs/llamacpp 8gb vram base.md` | 347.19 tok/s PP |
| **doc2**, 2 x 256k (`--parallel 2 -c 512000`) | `docs/llamacpp 8gb vram base 2x 262k context q8.md` | 26/42 layers offloaded, TG 22.4-23.3 tok/s |

Every LM/stock ratio below is **same-session**. Cross-session LM variance
reached 16% (u512: 2,532.5 in §11.7.2 against 2,937 in the 2026-09-30 ubatch
sweep), so no ratio in this subsection crosses a session boundary. VRAM is the
WDDM per-process dedicated peak polled at 1 s, re-derived from each arm's
`*.mem.csv`; the `peak VRAM:` line in the wrapper's stderr is the pre-run
baseline and is identical in every arm, so it is never cited.

**Provenance.** Every session below passed under `-AllowUnprovenanced`: the
wrapper captured no binary identity (`binary_state=not-supplied`), so the arms
are UNCERTIFIED-by-provenance at session level. The per-arm identity lines
carry `exe 9C8741260C5AE8BA / llama.dll 132B5C3EFB4770B6 / ggml-cuda.dll
CDCD45490CB958FC`, unchanged across ab9-ab12 and identical to the V1 side copy
recorded in ab9. All gates are soft; none of this is a release decision.

#### 12.6.1 The headline: a PP-only win

At doc1's context with stock tuned to LM's footprint, LM wins prefill by
**1.943x** at effectively equal VRAM:

| Side | PP tok/s | WDDM GiB | Layers | KV placement |
| --- | --- | --- | --- | --- |
| LM (V1 on), 3 arms `lmv4a/lmv4b/lmv4c` | **4,079.2** | 10.052 / 10.056 / 10.056 | 3-slot window | ctx-owned |
| stock `s13u4`, `-ub 4096 --fit-target 13000` | 2,099.6 | 10.193 | **41/42** | **2,720.00 MiB on CUDA0** |

The two sides are **0.137 GiB** apart, so this is ISO-VRAM in the sense that
matters. Source: `build/stock-levers/ab10.wrapper.log`, per-arm
`ab10-s13u4.out`, `build/llama-cuda/logs/ab10-lmv4*-a1-u4096.mem.csv`. LM
steady is the mean of passes 2..3 of each arm, then the mean of the three arms;
the anchor drift `lmv4a -> lmv4c` is **-0.35%**.

At doc2's context the same picture holds, at genuinely equal VRAM:

| Side | PP tok/s | WDDM GiB | Layers | KV placement |
| --- | --- | --- | --- | --- |
| LM (V1 on), 3 arms `lmc512a/b/c` | **4,080.4** | 14.976 / 14.976 / 14.972 | 3-slot window | ctx-owned |
| stock `s5f8k`, `--parallel 2 -c 512000 -ub 4096 --fit-target 8000` | 2,228.1 | 15.039 | **41/42** | **5,312.50 MiB on CUDA0** |

**0.063 GiB** apart: **1.831x**. Source: `build/stock-levers/ab5.wrapper.log`,
`ab5-s5f8k.out`, `build/llama-cuda/logs/ab5-lmc512*-a1-u4096.mem.csv`. Drift
`lmc512a -> lmc512c` is **-0.7%**.

Note what is *not* in these tables. The report's own ISO-VRAM rule selects, for
an LM arm, the largest stock arm whose peak fits under it — and twice that was
a starved configuration (ab10 `s14u4` at 9.211 GiB left stock 0.85 GiB
under-used; ab6/ab7 `fit=16000` arms run 30/42 layers with 816.00 MiB of KV on
the CPU). Both ratios above are stated against arms that are **verified well
fed**, with the layer count and KV placement printed next to the number.
§12.6.7 records this as a standing hazard.

#### 12.6.2 Token generation loses, and at two contexts it does not exist

Sustained LM generation requires a PP->TG handoff; without one, layer-major
refuses more than three tokens by construction
(`tools/layer-streaming/main.cpp:4871`, before the timed prefill). The refusal
is the finding:

| ctx | Free VRAM observed | Handoff needs | Shortfall | Verdict |
| --- | --- | --- | --- | --- |
| 262,144 | 22,226,665,472 B of 25,650,855,936 B | weights+KV+state 24,295,318,016 B | **1.93 GiB** | `handoff: auto -> OFF`, refuse |
| 512,000 | 22,226,665,472 B | 27,013,751,296 B + act slabs 409,600,000 B + reserve 536,870,912 B = 27,960,222,208 B | **5.34 GiB** | capacity preflight aborts |

Sources: `build/stock-levers/ab11cap.wrapper.log` and
`build/stock-levers/ab11cap-lmaut.tail.txt` (the preflight block: `full model +
KV + state`, `KV (q8 @ 262144) 2852126720 B`, `observed free VRAM`);
`build/stock-levers/ab5tg.tail.txt`. **Measured**, both refusals, both loud,
neither a silent fallback.

Where LM *can* generate — ctx 262,144 with a partial handoff — it is slower than
stock while using more VRAM. All figures are the binary's own
`phase_summary` (`phase: "tg"`, `tokens`, `wall_us`), mean of two mirrored arms,
against stock's `predicted_per_second` over the same 128 generated tokens:

| LM arm | Banks migrated | LM TG tok/s | WDDM GiB | stock TG tok/s @ 10.193 GiB | LM/stock TG | VRAM vs stock |
| --- | --- | --- | --- | --- | --- | --- |
| `lmh28` | `il >= 28` (12) | 25.074 | 13.434 | 39.204 | **0.640x** | 1.32x |
| `lmh20` | `il >= 20` (20) | 26.934 | 16.809 | 39.204 | **0.687x** | 1.65x |
| `lmh14` | `il >= 14` (26) | 29.045 | 19.341 | 39.204 | **0.741x** | 1.90x |

Sources: `build/stock-levers/ab12.wrapper.log`,
`build/llama-cuda/logs/ab12-lmh*-a1-u4096.jsonl`,
`build/stock-levers/ab12-s13.out` / `ab12-s13b.out`. The TG drift anchor
`lmh20 -> lmh20b` is **+0.6%** (26.854 against 27.014). Once the handoff has
happened the TG phase is compute-bound — `lmh20` spends 4,019,738 us of its
4,766,543 us wall in compute, 417,160 us in transfer, 1,036 us waiting — so
there is no transfer headroom left to win back.

**The charter's own test therefore resolves against LM at both contexts.** "A PP
win that collapses TG is not a win": at doc1's and doc2's contexts stock wins
prefill-then-generate, and LM is the prefill-only case. TG rows are comparable
*within* one length and one session only — the owner's own 22.4-23.3 tok/s came
from a 372-token prompt against our 12.13 on a full 50k prompt, and those two
numbers are not comparable (§12.6.7).

#### 12.6.3 The causal-FA dispatch fix (V1), measured

V1 changes only which kernel `flash_attn_ext` dispatches to for a causal op
with `mask == nullptr`: it restores the `use_gqa_opt` / `gqa_opt_applies` gate
that the null mask had disabled (`ggml-cuda/fattn.cu:58` and `:386`), which
buys the cp.async pipeline, removes 8x K/V re-reads, and keeps the KV_max skip
at `fattn-common.cuh:~1094`. It lives **only** in the `C:\Development\fafix`
side copy, behind an env toggle; the vendored tree is untouched and adoption
needs a Planner task contract plus a QA gate.

Three properties, all **measured**:

1. **VRAM-neutral.** At ctx 262,144 / q8 the ON and OFF arms peak at
   **10.056 GiB both ways**; the same holds at ctx 128,000/f16 (8.611-8.615
   across four arms) and ctx 512,000/q8 (14.976 both ways). Source:
   `build/stock-levers/ab9.wrapper.log`.
2. **Context-independent at 50k.** ON/OFF is **1.186x** at ctx 262,144,
   **1.182x** at ctx 128,000 with f16 KV, **1.182x** at ctx 512,000 — a spread
   of 0.4% across a 4x context range. Re-derived from the raw pass lists: ON
   4,058.05 against OFF 3,422.275 at 262k; ON 4,166.175 against OFF 3,524.450
   at 128k/f16; ON 4,036.8 against OFF 3,414.85 at 512k.
3. **Its own contribution grows with prompt length** (§12.6.4).

Fidelity evidence already on the record stands unchanged: logits within the
documented 0.1-1.3 noise floor, answer-QA PASS, ktest2 332 cases with equal
error against a double reference ON and OFF, and a bench per-op 1.37-2.09x at
long nkv. One caveat is **new** and belongs with them: at 128 generated tokens,
two runs of the *identical* partial-handoff configuration produced different
greedy sequences, diverging at token 1 despite a bit-identical token 0
(margin 1.953196, gold 248046 in both; `ab12-lmh20` steps 1-4 run
0.0093/0.1617/0.1099/0.0595 against `ab12-lmh20b`'s
1.6988/2.0027/1.1112/3.2644). Throughput is unaffected (+0.6%), so the timing
above stands, but **the token-level fidelity of the partial-handoff TG path is
unproven**, and it is consistent with — though not proof of — reading
incomplete expert weights on early TG steps. Nothing in §12.6.1 or §12.6.4
depends on that path.

#### 12.6.4 Prompt-length scaling: the multiplier decays

The load-bearing unknown was whether LM's advantage is a fixed multiplier or
erodes with prompt length. It erodes. All three points are ctx 262,144, q8_0,
`-ub 4096`, V1 on, so context is held fixed and only the prompt moves:

| Prompt tokens | LM ON | LM OFF | **ON/OFF** | stock, retuned | **LM/stock** | LM WDDM |
| --- | --- | --- | --- | --- | --- | --- |
| 50,000 | 4,058.1 *(ab9)* | 3,422.3 *(ab9)* | **1.186x** | 2,099.6 *(ab10)* | **1.943x** *(LM 4,079.2 ab10)* | 10.056 *(ab9)* |
| 144,001 | 2,865.4 *(ab6)* | 2,135.6 *(ab6)* | **1.342x** | 1,735.35 *(ab13, 6 req)* | **1.651x** *(LM 2,865.4 ab6)* | 10.769 *(ab6)* |
| 200,000 | 2,372.5 *(ab7)* | 1,723.5 *(ab7)* | **1.377x** | 1,548.1 *(ab7, req mean)* | **1.533x** *(LM 2,372.5 ab7)* | 11.201 *(ab7)* |

**Two of the three LM/stock ratios are same-session; the 50,000 one is not, and the
cell labels now say so.** The `LM/stock` column needs the LM mean measured in the
SAME mutex hold as the stock arm it is divided by, because cross-session LM
variance reaches 16% - larger than most effects here. For 144,001 and 200,000 that
holds (ab6/ab6 and ab7/ab7). For 50,000 it does not: **ab9 ran no stock arm at all**
(`ab9.manifest.jsonl` is ten LM arms), so the 50,000 stock term necessarily comes
from **ab10**, and the LM term must therefore be ab10's **4,079.2** (three arms
`lmv4a/b/c`), not ab9's **4,058.1**. The ratio is **1.943x** and it is correct; the
preceding cell previously printed ab9's figure beside it, so the row as printed
divided to 1.9328x rather than the 1.943x it asserted. **Corrected 2026-10-01:** each
cell now carries its session, and the ratio column names the LM figure it actually
divides. The `LM ON`/`LM OFF`/`ON/OFF` columns remain ab9 throughout, because the
V1 fix was A/B'd there and that is the only session with both V1 states at 50k; the
V1 series (1.186x / 1.342x / 1.377x) is unaffected, since every one of those ratios
is LM-against-LM within a single session.

Sources: `build/stock-levers/ab6.wrapper.log` and `ab7.wrapper.log` (144,001 and
200,000 tokens, 7 arms each, PASS), `ab6-s6u4k.out` / `ab7-s7u4k.out`, traces
under `build/llama-cuda/logs/ab{6,7}-*.jsonl`; the 50,000 row from ab9/ab10.

**The 144,001 stock figure was re-measured, and the caveat is now retired.**
The row above originally read **1,734.05 tok/s** — ab6 `s6u4k` **request 0
alone**, because that arm's request 1 was flagged `CONTAMINATED` (neighbours
mean 8.87 cores, max 31.38, over 6 cores in **28 of 83** samples), leaving one
clean sample behind the ratio. That arm was re-run on its own, in its own mutex
hold (**ab13**, `build/stock-levers/ab13.plan`), with `req=6` instead of `req=2`
and **nothing else changed**: all eight binary SHA-256 hashes verified identical
to `ab6-s6u4k.ident.txt`, same flags, same prompt file
(`anchor-100000.txt`, sha256 `FAF70D6F...`), same `41/42` layers, same
`CUDA0 KV buffer size = 2720.00 MiB`, and the same **10.193 GiB** WDDM
per-process peak.

| | requests | per-request PP tok/s | neighbour load (mean / max / >6 cores) | steady |
| --- | --- | --- | --- | --- |
| **ab6 `s6u4k`** (before) | 2 | 1,734.05 / 1,660.81 | 1.60-8.87 / 31.38 / **28 of 83 on req 1** | req 0 alone, 1,734.05 |
| **ab13 `s6u4k`** (after) | 6 | 1,743.5 / 1,736.4 / 1,733.2 / 1,733.7 / 1,733.8 / 1,731.5 | 1.47-1.82 / 3.30-5.43 / **0 of 81-82 on every request** | **mean of all 6, 1,735.35** |

**All six requests were clean** — zero samples over 6 cores in every request, and
the worst neighbour reading in the whole arm (5.43 cores) is below the
contamination threshold, against ab6's 31.38. The wrapper verdict is `PASS` with
`gpu_pre_run=clean` and `gpu_post_run=clean`.

**Side by side: the stock figure moves +1.30 tok/s (+0.075%), and the ratio moves
1.652x -> 1.651x (-0.001).** The single-sample figure was not wrong; it was
under-determined, and six clean samples say so. The change is ~45x smaller than
the 5% threshold this project ignores by rule, so **no conclusion in this
document moves**. The decay series becomes **1.943x -> 1.651x -> 1.533x**;the per-doubling decay recomputes to **7.8%** on the first interval (50k -> 144k)
and **3.7%** on the second (144k -> 200k), so the two intervals are NOT
symmetric and the "about 11% per doubling" figure quoted elsewhere in this
subsection is the **end-to-end geometric rate across the whole 4x span**, not
either interval. That end-to-end figure is **11.2%** and is unchanged, because
both endpoints are untouched, as is the 0.789 that survives a 4x increase.
**Measured**, six clean requests, one mutex hold.

**Two trends run in opposite directions, and the net decays.** V1's own
contribution *grows* with length — 1.186x, 1.342x, 1.377x — because the longer
the attention window, the more the un-optimised K/V re-read costs. But LM
degrades faster in relative terms than stock does: LM keeps **0.582x** of its
50k throughput at 200k while stock keeps **0.737x**. The net ratio loses about
**11% of its advantage per doubling** of prompt length (0.789 of it survives a
4x increase). Extrapolating that rate puts LM near 1.3x at ~500k, but that is
**extrapolation, not measurement**, and the context would have to grow with it.

Why stock degrades more gracefully is a **hypothesis, explicitly unproven**.
Stock copies only the routed experts (`ggml/src/ggml-backend.cpp:1629-1716`),
and the route union saturates toward 1.0 as a prompt lengthens — measured at
**0.717 / 0.815 / 0.855** at ubatch 512 / 2048 / 4096 on the 50k prompt. At
200k the union is nearer saturation, so stock copies nearly everything and loses
less ground. **No route telemetry was collected at 144,001 or 200,000 tokens**
(`--route-telemetry` was off, the T-GPLR-39 default), so the saturation is not
measured at those lengths.

LM also pays for length in VRAM: **10.056 -> 10.769 -> 11.201 GiB** across the
three lengths, a slope of **7.95 and 8.09 KiB per prompt token** on the two
intervals. That matches the act-slab reservation the preflight reports
(`act slabs require 409600000 bytes (2 x 204800000, bf16)`, and the same
`act_slab_bytes` 409,600,000 in each 50k `run_start`) — two slabs at
**16.0 KB/token** of *reservation*, against a measured realised slope of ~8
KiB/token, i.e. the reservation is an upper bound roughly twice the realised
cost, which is what a reservation is for. Note the act slab is sized by
**ubatch**, not by prompt, so it is constant across these three runs; it is the
**same term** in all three and cannot by itself explain a slope. The slope is
therefore **measured but not attributed** — the preflight's context-scaled
terms are already accounted for, and the residual ~8 KiB/token is unexplained.
**Cause UNPROVEN.** (The neighbouring §12.6 finding 30 records the same shape on
the *context* axis: ~9.8 KiB per context token of unaccounted VRAM.)
Stock's peak does not move with prompt length at all (10.193 GiB at all three),
because its KV is sized by context, not by prompt. So the comparison drifts
*off* ISO as prompts lengthen: 5.7% VRAM mismatch at 144k, 9.9% at 200k.

#### 12.6.5 The largest free owner lever: retune `--fit-target`

At doc2, changing one flag and nothing else:

| `--fit-target` | PP tok/s | TG tok/s | WDDM GiB | Layers | KV |
| --- | --- | --- | --- | --- | --- |
| **16000** (as recorded) | 425.9 | 12.13 | 6.972 | **26/42** | 3,187.50 MiB CUDA0 + **2,125.00 MiB on CPU** |
| 12000 | 1,950.8 | 36.65 | 11.026 | 41/42 | 5,312.50 MiB CUDA0 |
| **8000** | **2,228.1** | **42.47** | 15.039 | **41/42** | **5,312.50 MiB CUDA0** |

**PP 5.23x and TG 3.50x, for one flag, on the owner's own recorded
configuration.** Source: `build/stock-levers/ab5.wrapper.log`, arms `s5u512`,
`s5f12k`, `s5f8k`. `s5u512` reproduces the owner's recorded placement exactly —
26/42 layers, `CUDA0 model buffer size = 1311.95 MiB`, identical to the digit
in `docs/llamacpp 8gb vram base 2x 262k context q8.md` — so this is the
owner's cell, not an approximation of it. **Measured.**

`--ubatch-size` is the trap on the same configuration, and it fails in the
opposite direction:

| `-ub` at `--fit-target 16000` | PP tok/s | TG tok/s | Layers | KV |
| --- | --- | --- | --- | --- |
| 512 | 425.9 | 12.13 | 26/42 | 2,125.00 MiB on CPU |
| 2048 | 1,181.4 | 10.38 | 22/42 | 2,656.25 MiB on CPU |
| 4096 | 1,645.6 | 6.67 | 14/42 | 3,718.75 MiB on CPU |
| 8192 | 1,951.6 | **5.33** | **2/42** | **all 5,312.50 MiB on CPU** |

**4.58x PP bought by collapsing residency from 26 layers to 2 and pushing the
entire KV cache off the GPU, at 0.44x TG.** The cause is §11.7's compute-buffer
term (2 B x ctx x ubatch) competing with the weights inside the fitter's
budget: the CUDA0 compute buffer runs 2,099.41 -> 6,363.78 MiB across that
ladder. **Measured.** A report that shows this arm as `KV None MiB` is not
reporting a bug — it is reporting that no CUDA0 KV buffer was allocated.

At doc1's context the same pattern, milder: `fit 16000` gives 30/42 layers and
7.157 GiB, `fit 13000` gives 41/42 layers, the full 2,720.00 MiB KV on CUDA0 and
10.193 GiB. The retuned value is context-dependent, which is why ab6 and ab7
both used 13000 and recorded 41/42 layers with full GPU KV at both lengths.

#### 12.6.6 The twenty-first finding, again, at ISO-VRAM

§12.5's twenty-first finding replaced a 7.08x framing with **1.733x**. That
figure is now **superseded**: it was still a comparison at *unequal* VRAM, and
at ctx 128,000 the correct comparison is against stock at the ubatch that
actually fills the same budget. From ab8's own same-session arms (LM `lmA`
3,657.5 tok/s @ 7.472 GiB):

| Stock cell at ctx 128,000 | PP tok/s | WDDM GiB | LM/stock |
| --- | --- | --- | --- |
| `-ub 8192 --fit-target 16000`, anchor | 2,610.7 | 7.438 | **1.40x** |
| `-ub 8192 --fit-target 16000`, vendored | 2,554.8 | 7.445 | **1.43x** |
| `--fit-target 300` (everything resident), anchor | 4,837.6 | 22.523 | **0.756x** |
| `--fit-target 300`, vendored | 4,602.2 | 22.361 | **0.795x** |

Source: `build/stock-levers/ab8.wrapper.log`. ab8's `lmA` arm is
UNCERTIFIED-by-provenance, as recorded. The ab2 curve gives the same shape
independently: 1.416x / 1.402x against `-ub 8192`, 1.513x at `fit 12000`, 1.273x
at `fit 8200`, **1.002x** at `fit 4000` (18.990 GiB), **0.759x** at `fit 300`
(22.523 GiB). So LM is ahead at a 7-8 GiB budget, level at about 19 GiB, and
behind once stock is fully resident. **Measured** (ab2, ab8); the ab2 LM arms
carry a -1.0% internal drift.

#### 12.6.7 Two measurement-hygiene traps, recorded because they will recur

**The prompt files are not named by token count.** Measured token counts:

| File | Tokens |
| --- | --- |
| `anchor-50000.txt` | 50,000 |
| `anchor-100000.txt` | **144,001** |
| `anchor-200000.txt` | 200,000 |

Byte density is a consistent 4.634-4.638 B/token, so the middle file is the
odd one out. Two consequences bit this session's first attempt, which is why
they are written down:

- `--pp-tokens` does **not** truncate. The tool computes the prompt length from
  the **file** and reports `pp_tokens=144001` even when `--pp-tokens 100000`
  was passed; `pp=` is a *file selector*. The refusal is explicit:
  `error: --ctx-size 131072 is smaller than pp+tg (144004 tokens)`, and stock's
  client independently returns HTTP 400 with the same 144,001. Both engines
  fail loudly and identically — but only after a full hold is spent.
- Any planned context must be checked against the **file's** token count, not
  against its name. The old plan for this session specified ctx 131,072 on that
  basis and both engines refused. Corrected to ctx 262,144 for all three
  lengths, which also removed the context confound from §12.6.4.

**The session report's ISO-VRAM rule has twice selected a starved stock arm.**
It picks the largest stock arm fitting under the LM peak, which is a sound rule
applied to arms that are not comparable. Both times the fix was to read the
arm's layer count and KV placement next to the ratio — `offloaded N/42 layers`
and the `llama_kv_cache` buffer lines in `build/stock-levers/<arm>.log`. §12.6.1
and §12.6.5 print them for that reason. Recommend treating any stock arm below
~40/42 layers, or with a non-zero `CPU KV buffer size`, as unfit for an ISO-VRAM
claim.

#### 12.6.8 A real defect in the vendored tool — recorded, not fixed

`tools/layer-streaming/main.cpp:3454` reads, in full:

    // explicit handoff-on request.            handoff_mode_str = "on";

The statement is **inside the comment**. `--handoff-partial N` parses `N` and
sets `handoff_partial_n`, but leaves `handoff_mode_str` at its default `"auto"`.
The "Implies an explicit handoff-on request" sentence is the comment's own
preamble, not the flag's `--help` text: `main.cpp:67-68` describes only which
banks migrate and never says the flag implies a handoff. So `--help` and the
code agree with each other, and what the `//` swallowed is the **documented
intent** — a reader of the flag's help would have no way to learn that passing
it alone does nothing. Because auto resolves OFF whenever the full
model+KV+state does not fit, `--handoff-partial` alone can never activate a
handoff at these contexts, and every `--tg-tokens > 3` run refuses. All four
partial-handoff arms of session **ab11** (`lmh20`, `lmh14`, `lmh28`, `lmh20b`)
failed this way — `ARM EXIT 2 (4.7s)` four times in
`build/stock-levers/ab11.wrapper.log`, with the refusal text itself at
`build/stock-levers/ab11-lmh20.tail.txt:2234-2236`
(`handoff: auto -> OFF`, then the cap). Passing `--handoff` explicitly works
around it and is what §12.6.2's numbers were taken with.

**This is a one-token fix in the vendored tree and it has not been made.** The
vendored `third_party/llama.cpp` is another workstream's, carries an
uncommitted `src/models/qwen35moe.cpp` edit, and adoption needs a Planner task
contract. Recorded as a draft item, not as work.

#### 12.6.9 What survives

- LM wins **prefill** at both owner configurations: **1.943x** at doc1
  (0.137 GiB apart) and **1.831x** at doc2 (0.063 GiB apart). **Measured**,
  same-session both.
- LM **loses token generation** wherever it can generate at all (0.640-0.741x
  at 1.3-1.9x stock's VRAM) and **cannot generate** at either owner context,
  by 1.93 GiB and 5.34 GiB of VRAM. **Measured**, by refusal.
- LM's advantage **decays** with prompt length, 1.943x -> 1.651x -> 1.533x, about
  11% per doubling end-to-end. **Measured** at three points; the mechanism behind
  it is **unproven**, no route telemetry at 144k/200k. The 144,001 stock term was
  re-measured on its own mutex hold with six clean requests after ab6's second
  request came back CONTAMINATED; it moved +0.075% and retired the caveat
  (§12.6.4).
- V1 is **VRAM-neutral** and its contribution **grows** with prompt length.
  **Measured**; fidelity of the *partial-handoff TG path* remains **unproven**.
- The largest free win found is a stock flag: `--fit-target 8000` at doc2, 5.23x
  PP and 3.50x TG, no code. **Measured.**
- Everything previously withdrawn stays withdrawn. Nothing in this subsection
  re-opens §12.5's ninth finding, the 28.92 ms intercept, the 10.35 GB/s feed,
  or the 0.6487 mislabelling.

---

## 13. MEASURED 2026-09-30: what the dynamism is actually worth

§12 established that the host expert FFN dominates both phases. §7b left the
per-expert split as the one live mechanism. Neither asked the prior question:
**of the gain a row cache delivers, how much comes from the fine granularity
and how much from tracking the prompt?** Those have wildly different build
costs, and the record had never separated them.

Measured host-only from existing `rowset_fetch` telemetry, which carries the
per-token, per-layer top-8 expert IDs in `union_ids`. Tool:
`tools/expert-skew.py`. Traces: `build/llama-cuda/a9-w0-norecon.jsonl`
(run_id 3640876) and `build/llama-cuda/gplr19-honest-k6-ab/arm-A1.jsonl`
(run_id 2898024) — two 256-token TG streams from different prompts, 10,240
events and 81,920 demand events each.

A correctness check first: `a9-w16-norecon.jsonl` produces numbers **identical
to the last digit** to `a9-w0-norecon.jsonl` in every cell. Those two runs
differ only in the compact slot window width W (`--residency-tg-window`, §9.6;
both hold a K=4 row cache), so the routes must be identical and they are. The
compact window is route-transparent, and the analysis reads what it claims to
read. *(Wording corrected 2026-09-30: this first said "cache width W". The W=16
run's `rowset_narrow` event carries `window_width` 16; both runs carry
`retain_k` 4.)*

### 13.1 Routing is strongly skewed, and the skew is almost entirely per-prompt

Per-layer Gini over the 256 experts is **0.4784 / 0.6831 / 0.7687**
(min/mean/max across the 40 layers) on one prompt, 0.4100 / 0.6738 / 0.8105 on
the other. Routing is nowhere near uniform, which is already worth knowing —
load-balancing training objectives are usually assumed to flatten it.

But the skew does not transfer between prompts. Rank experts on one trace, pin
the top N per layer, score coverage on the *other* trace:

| N per layer | VRAM over 40 layers | in-sample | held-out | uniform null |
| --- | --- | --- | --- | --- |
| 32 | 2,160 MiB | 0.5441-0.5624 | 0.2125 / 0.2283 | 0.1250 |
| 56 | 3,780 MiB | 0.7149-0.7240 | 0.3195 / 0.3535 | 0.2188 |
| 64 | 4,320 MiB | 0.7572-0.7637 | 0.3541 / 0.3944 | 0.2500 |

In-sample the lift over uniform is 3.0-4.5x and looks decisive. Held-out it
collapses to 1.42-1.83x. **The hot set is a property of the prompt, not of the
model.** A build-time static pin is therefore worth far less than the skew
statistic alone suggests, and any analysis that stops at the in-sample number
is measuring its own sampling noise.

### 13.2 The equal-capacity ladder

Capacity is `cap = min(n_expert, K*W*top_k)`, W=1, so K=6 is 48 rows/layer
(3,240 MiB over 40 layers) and K=8 is 64 rows/layer (4,320 MiB). Stock's 7.5
resident banks cost **3,271 MiB** (§12.1b), which is **31 MiB more than the
48-row cache uses** — so the K=6 row is not merely the cheapest good point, it
is strictly VRAM-negative against the baseline it beats. **That dominance is
specific to the anchor's context and does not survive scaling**: stock's
residency is leftover, so it shrinks as KV grows, and by 236,000 tokens on the
same budget stock holds 2,197 MiB and the 48-row cache costs 1,043 MiB *more*
than it, not less (§13.6). Every LRU-K number
below is the run's **own** `n_rows_resident` telemetry, not a simulation; every
placement number is `tools/expert-skew.py` replaying the same traces' routes.
LRU-K is quoted over the second half of the stream so that it is not charged for
cold-start misses the warm-then-pin figures also exclude.

Two prompts, reported separately because they disagree materially — B is the
`gplr19-honest-*` stream (Gini 0.674), A is the `gplr19-k*-ab` stream (Gini
0.771, visibly more repetitive):

| cap | policy | h (prompt B) | h (prompt A) | machinery |
| --- | --- | --- | --- | --- |
| — | stock whole-bank, 7.5 of 40 L | 0.188 | 0.188 | none, shipped |
| 48 | static build-time pin (held out) | 0.3340 | 0.3884 | split op only |
| 48 | warm-then-pin, W=128, frozen | 0.5087 | 0.7270 | split + 1 decision/session |
| 48 | **LRU-K, K=6** | **0.6345** | **0.7156** | full per-token cache |
| 64 | static build-time pin (held out) | 0.4117 | 0.4839 | split op only |
| 64 | warm-then-pin, W=128, frozen | 0.5979 | 0.8143 | split + 1 decision/session |
| 64 | **LRU-K, K=8** | **0.7198** | **0.7860** | full per-token cache |
| 64 | in-sample oracle (ceiling) | 0.7572 | 0.8727 | unrealizable |

Three readings.

**Granularity alone is worth about a quarter of the gain.** A build-time static
pin has the same fine granularity as the cache and the same byte budget, and
reaches only 0.33-0.48. The rest comes from adapting to the prompt.

**Neither adaptive policy dominates.** On prompt B the cache beats warm-then-pin
by 0.122; on prompt A warm-then-pin **beats the cache** by 0.028. A frozen
frequency set does better than recency on a repetitive stream, worse on a varied
one. The record has been treating the cache as strictly superior; at matched
capacity it is not.

**LRU-K is within a few points of the in-sample oracle.** 0.7198 against 0.7572
and 0.7860 against 0.8727 — 95.1% and 90.1% of the best *time-invariant* set
choosable with full knowledge of the session. There is very little left for a
cleverer frequency-based policy to find.

Warm-then-pin saturates: W=64 -> W=128 moves h by ~0.02-0.09, and W=16/32 are
worse than both. The set is learnable from roughly the first 64-128 generated
tokens. That it saturates at all is the evidence that this is frequency and not
a disguised short-window recency effect.

### 13.3 Non-uniform per-layer capacity is worth nothing

Given per-layer Gini ranging 0.41 to 0.81, spending a shared row budget where
the skew is highest is the obvious next move. Measured: allocate the same
`cap x 40` rows to the globally hottest `(layer, expert)` pairs instead of `cap`
per layer, and coverage changes by **at most 0.3% in any cell**, in either
direction, in-sample and held-out alike. Every layer routes exactly 8 experts
per token, so demand is equal across layers by construction, and the marginal
value of the last row turns out to be near-equal too.

**Variable per-layer cache capacity is closed.** It is a free lever on paper
and it pays nothing.

### 13.4 Throughput, and the objective this project has been optimising wrongly

> **CORRECTED 2026-09-30.** The table below previously evaluated the cost model
> at the **mean of the two prompts' hit rates**. That is a modelling error.
> Throughput is *concave* in `h` (the overlapped per-expert cost is
> `max(h·8·d, (1−h)·8·c)`, a max of two lines with a minimum at `h*`), so by
> Jensen's inequality `tok_s(mean h) ≥ mean tok_s(h)`. The bias is ≈0.00x on
> six of the seven policies and **+0.06x on exactly one: warm-then-pin at 64
> rows**, whose two prompts straddle `h*` almost symmetrically (0.5979 and
> 0.8143 against `h*` = 0.7069) so their mean lands 0.0008 away from the
> optimum. **The "1.37x from warm-then-pin" was the ceiling value produced by a
> coincidence of averaging, not a property of the policy.** The table now
> evaluates each prompt and reports the mean of the throughputs plus the
> per-prompt range. `tools/jensen-check.py` isolates the bias;
> `tools/placement-model.py` is fixed. **The ranking inverts** — see below.

Under §12's validated cost model — `c = 35.7 us` host per expert (measured,
§12.1), `d = 14.8 us` device per expert (still **not** measured), `other`
solved from the 40.96 tok/s anchor at `h_stock` = 0.188 (measured off the
anchor's own fitter log, §12.1b) giving 14.25 ms. Arithmetic in
`tools/placement-model.py`:

| cap | policy | h (B / A) | overlapped, mean | per-prompt range | clears `h*`? |
| --- | --- | --- | --- | --- | --- |
| — | stock | 0.188 | 40.96  1.00x | — | no |
| 48 | static pin | 0.334 / 0.388 | 46.42  1.13x | 1.12–1.15x | neither |
| 48 | warm-then-pin | 0.509 / 0.727 | 53.44  1.30x | 1.23–1.38x | A only |
| 48 | **LRU-K K=6** | 0.634 / 0.716 | **55.49  1.35x** | 1.33–1.38x | A only |
| 64 | static pin | 0.412 / 0.484 | 48.67  1.19x | 1.16–1.21x | neither |
| 64 | warm-then-pin | 0.598 / 0.814 | 54.16  1.32x | 1.30–1.35x | A only |
| 64 | **LRU-K K=8** | 0.720 / 0.786 | **56.14  1.37x** | **1.36–1.38x** | **both** |
| 64 | in-sample oracle | 0.757 / 0.873 | 55.24  1.35x | 1.33–1.37x | both |

Every one of those fourteen hit rates was re-derived from the raw traces on
2026-09-30 and all fourteen reproduce exactly (prompt A =
`gplr19-k8-ab/arm-B1.jsonl`, prompt B =
`gplr19-honest-k6-ab/gplr19-honest-k8-ab/arm-B1.jsonl`). The inputs are sound;
only their aggregation was wrong.

> **THE TOP TWO ROWS ARE A TIE, not a ranking — §12.1c, 2026-09-30.** `c` =
> 35.7 is an **8-thread** constant and the anchor ran at **16**; on the anchor's
> real thread count `other` falls to 14.083 ms, and running the host leg at its
> own optimum (12 threads) puts **K=6 @48 ahead of K=8 @64**, 1.393x to 1.383x,
> reversing the two bolded rows. But the margin either way (+0.0020x at 16
> threads, +0.0096x at 12) is **inside the 1.0% cross-invocation spread on the
> host measurements**, ≈0.014x here. Neither ordering is resolvable by this
> evidence. What *is* resolvable: at 236k only 48 rows fit at all (§13.6), and
> the four rows that sit below `h*` are host-bound and gain +0.012x to +0.025x
> from the thread fix while the two above it gain nothing. **Read this table as
> "48 and 64 tie at ~1.38–1.39x, and capacity decides."**

**Under overlap the optimum is an interior point, `h* = c/(c+d)`.** The
per-layer cost is `max(h*8*d, (1-h)*8*c)`: the host leg falls with `h`, the
device leg rises with it, and they cross at `h* = c/(c+d) = 35.7/50.5 =
0.7069`. Below `h*` the host is the bottleneck; **above it the GPU is**, and
extra hit rate buys nothing while the device leg keeps growing. The floor is
`8*c*d/(c+d)` = 83.7 us/layer = **56.83 tok/s = 1.39x, and that is the ceiling
of the entire mechanism at any hit rate whatsoever.** No policy in the table
reaches it; the best gets 1.37x.

The closed form `h* = c/(c+d)` holds only for a *constant* `c`. On the measured
host curve `h*` is 0.6990 (§12.1a, 8 threads) and **0.6662** at the split's own
12-thread optimum, with the floor at 78.9 us/layer and the ceiling 1.416x
(§12.1c). Lower `h*` is the useful direction: it is residency bought with
threads instead of VRAM, and §13.6 is the section that says VRAM is what we do
not have.

**This inverts the objective the project has been optimising** — but not in the
direction the previous draft claimed. Every gate, every retention sweep, every
predictor attempt in this record maximised `h`. The right target is not maximum
`h` but **`h` held close to `h*` on every prompt**. Those are different
objectives, and they rank policies differently: the in-sample oracle has the
highest `h` of anything measured (0.757 / 0.873) and is *beaten* by LRU-K K=8,
because it overshoots `h*` on both prompts.

**What this makes valuable is stability, and that reverses the earlier
conclusion about per-token machinery.** Because throughput is concave, a policy
that lands both prompts near `h*` beats one that overshoots on one and
undershoots on the other, even when the second has the same *average* `h`:

| | h spread across prompts | worst prompt |
| --- | --- | --- |
| warm-then-pin, 64 rows | **0.216** (0.598 → 0.814) | 1.30x |
| LRU-K K=8, 64 rows | **0.066** (0.720 → 0.786) | **1.36x** |

LRU-K's spread is **3.3x tighter**, and that — not its higher mean — is what
makes it win. Adaptivity's payoff is variance reduction in `h`. **LRU-K K=8 is
also the only non-oracle policy measured that clears `h*` on both prompts.**

Note also that **LRU-K K=6 at 48 rows (1.35x, 3,240 MiB) beats warm-then-pin at
64 rows (1.32x, 4,320 MiB)** — better throughput for 1,080 MiB less VRAM. It
dominates on both axes, which matters at a **7,055 MiB** budget (§10.1 — the
fitter's own `target=7055`, not the 8,462 the flag implies; see §13.6).

The honest engineering form of this is a **tunable device share `alpha`**
rather than "device share = residency". Set `alpha = h*` and let the residency
policy merely guarantee `h >= alpha`; then overshooting is harmless instead of
harmful, and the oracle's and K=12's surplus `h` stops being a liability. Under
the draft's literal "not resident -> host, full stop" rule, overshoot costs real
time — which is a concrete reason to relax that rule. **With `alpha` in place
the stability argument weakens**, because clamping at `alpha` truncates the
overshoot half of the spread; warm-then-pin's undershoot on prompt B survives
it, so LRU-K still leads, but by less. Worth deciding with `d` measured.

**The predictor line is closed on arithmetic, not just on evidence** — and this
survives the correction, though the reasoning narrows. A predictor can only
raise `h`. At 64 rows LRU-K already clears `h*` on both prompts, so raising `h`
there is worthless or harmful. The residual case for a predictor is narrow and
specific: lifting the *undershooting* prompt (prompt B at 48 rows, or
warm-then-pin's 0.598) toward `h*` without pushing the other prompt past it.
That is a variance-reduction job, not a hit-rate-maximisation job, and it is
what four predictor attempts were **not** aiming at. §15.5 closes it from the
cost side independently: the coact predictor placed zero rows at three
capacities while taxing TG up to 2.5%.

**`d` is now the only thing worth measuring, but as a setpoint rather than a
gate.** The ceiling is robust — 1.31x to 1.43x as `d` ranges 8 to 30 us — but
`h*` moves with it (0.817 down to 0.543), so `d` determines *where to aim*, not
*whether to build*. It also decides the policy ranking: at small `d`, `h*` rises
toward 0.82 and the high-`h` policies stop overshooting, which would restore
warm-then-pin's and the oracle's standing. **The ranking above is conditional on
`d` ≈ 14.8 us and should not be treated as settled until `d` is measured.**

### 13.5 What survives

The recommendation changes, and it does **not** simplify as much as the
previous draft claimed. Build the **split op with a tunable device share
`alpha`** — that is the piece that buys the whole 1.39x ceiling, and it needs no
staging, no `rowset_recon` path, and no predictor. `d` falls out of that build
as a direct observable and sets `alpha`.

For placement on top of it, the measured ordering at `d` = 14.8 us is **LRU-K
ahead of warm-then-pin at both capacities**, on the mean and on the worst
prompt, with LRU-K K=6 at 48 rows the best VRAM-for-throughput point measured
(1.35x for 3,240 MiB, which at the anchor's context is *less* VRAM than stock's
own 3,271 MiB — though not at 236k, §13.6). So:

- **Do not** treat warm-then-pin as sufficient. The previous claim that
  per-token machinery "buys 0.00x" was an artifact of the Jensen error; on the
  corrected numbers it buys **+0.05x at 64 rows and +0.05x at 48 rows**, and
  more on the worst prompt (+0.06x and +0.10x).
- **Do** build the split op first regardless. Both policies need it, it is worth
  far more than the gap between them (1.32x vs 1.00x, against 1.37x vs 1.32x),
  and it yields `d`, which decides the placement question properly.
- **Then** choose placement with `d` in hand, `alpha` clamping in place, and
  more than two prompts.

Caveats, stated: two prompts is a small sample, and they disagree by up to 0.22
in `h` at cap 64 — larger than the gap between the policies, so the ordering is
*indicated, not established*; what the correction does establish is that the
earlier ordering was produced by an averaging artifact rather than by evidence.
The overlap model assumes the host and device legs of one layer can genuinely
run concurrently, which is the hard part of T-GPLR-61 and is unbuilt. `d` = 14.8
us is still differenced, not measured, and it sets both `h*` and the ranking.
Every tok/s above inherits §12.4's `REPACK` caveat through `c`.
*Closed 2026-09-30 (§12.4).* Override tensors are never `CPU_REPACK`, so the
bench's un-repacked Q4_K is the format stock TG reads, and `c` needs no
`REPACK` discount.

### 13.6 MEASURED 2026-09-30: at the settled context, capacity is decided by VRAM before throughput ever gets a vote

§13.4 ranked the policies at the anchor's context and stopped there. That is
incomplete, because the row cache and the KV cache compete for the same bytes:
a policy can win on throughput-per-MiB and still be unbuildable at the context
the product actually ships. The anchor makes that checkable without modelling
anything, because its VRAM accounting **reconciles exactly**:

```
model 5192.44 + KV 1328.12 + RS 60.72 + compute 427.53 = 7008.81 MiB
fitter, line 95:  "41 layers (33 overflowing),  7008 MiB used,  16046 MiB free"
```

Every term is a printed observable, so the same five terms can be re-evaluated
at another context with nothing invented. Two of them are context-independent
(dense weight 1,921 MiB from §12.1b; compute buffer 427.53 MiB, set by `ubatch`
512), one scales with context, and the row cache replaces the fourth.

The scaling term is small for this architecture, which is the whole reason long
context is affordable here at all. `llama_kv_cache: 1328.12 MiB (128000 cells,
**10 layers**)` — only 10 of 40 layers carry a KV cache, the other 30 being
recurrent — so context costs **0.010376 MiB/token**, roughly a quarter of what
the layer count suggests. That constant reproduces exactly in the unrelated
512k run (`5312.50 MiB (256000 cells, 10 layers, 2/2 seqs)` = 0.020752/cell =
exactly 2x, i.e. the same per-token-per-sequence figure).

**The budget is where the answer actually turns, and the obvious number is
wrong.** `--fit-target 16000` on a 24,462 MiB card looks like an 8,462 MiB
budget, and four paragraphs of this document said "8,416 MiB" on roughly that
reasoning. The fitter disagrees in writing: line 49 prints **`id=0,
target=7055 MiB`**, because it measures free memory *after* CUDA init, when
1,407 MiB of the card is already spoken for. §10.1 established this; the later
sections had simply drifted off it. On a bare 8 GB card that 1,407 MiB is not
there; on a card also driving a desktop it is.

| budget | 48 rows / 3,240 MiB | 56 rows / 3,780 MiB | 64 rows / 4,320 MiB | no cache |
|---|---|---|---|---|
| **7,055 MiB** realised (§10.1) | 135,482 | 83,438 | 31,395 | 447,743 |
| **8,462 MiB** nominal (bare card) | **271,084** | 219,040 | 166,997 | 583,345 |

Max single-slot context, from `tools/ctx-budget.py`. Against the settled
236,000:

- At the **nominal** budget, **only 48 rows fits** — 364 MiB spare, against
  −176 for 56 rows and −716 for 64. This converges with §13.4's throughput
  ranking, which independently picked LRU-K K=6 at 48 rows, on the same policy.
  Two unrelated criteria agreeing is the strongest result in §13.
- At the **realised** budget, **nothing with a row cache reaches 236k** — 48
  rows tops out at 135k. The cache is not free; it is 3,240 MiB that KV cannot
  have, which is 312,000 tokens of context given away.

So the capacity choice is not "48 or 64" but a stated dependency: **48 rows,
and only if the card is not also driving a display.** If it is, the honest
options are a smaller cache, a smaller context, or accepting 135k. That
dependency has to be surfaced now rather than discovered in a fit failure.

Stock is deliberately absent from the table, because its expert residency is
not a fixed allocation — it is whatever the fitter has left over. That is an
asymmetry, not a detail: at a context it cannot afford, **stock degrades, a
fixed row cache fails to fit.** Reading the same five terms with the cache
replaced by "leftover":

| stock leftover | 128,000 ctx | 236,000 ctx |
|---|---|---|
| **7,055 MiB** budget | 3,318 MiB, h 0.191 | 2,197 MiB, **h 0.126** |
| **8,462 MiB** budget | 4,725 MiB, h 0.271 | 3,604 MiB, h 0.207 |

Read that **down the columns**. Budget is a property of the machine; context is
the thing being scaled. At the anchor's own budget, going 128k → 236k costs
stock 2.6 banks and drops `h_stock` **0.188 → 0.126** — so 0.188 is a best case
for that machine, and §13.4's multipliers are *understated* at the shipping
context, not overstated. On a bare card the spare 1,407 MiB more than pays for
the extra KV and stock instead rises to 0.207, but that is better hardware, not
better scaling. Comparing across the rows conflates the two.

The 128k/7,055 cell is the anchor itself, and it is the model's own check: it
predicts 3,318 MiB against the measured 3,271, off by **46 MiB** — which is
exactly the 46 MiB of headroom the fitter left unspent (7,008.81 used of
7,055). The projection carries no free parameters.

Where the trend ends is on disk. The 512k run got only **1,311.95 MiB** of
model onto the device — *below* the 1,921 MiB of dense weight — i.e. zero
expert banks and dense weight itself spilling to host.

Caveats: this projects the compute buffer and the recurrent-state buffer as
context-independent. The first is set by `ubatch`, the second by sequence count,
so both hold for a single slot at fixed `ubatch` — but neither has been
*measured* at 236k, and a fit run at the target context is the cheap way to
confirm before committing to 48 rows. The table is arithmetic on printed
observables, not a prediction of fitter behaviour; the fitter leaves headroom,
as it did here.

---

## 14. MISLABELLED 2026-09-30: the 0.6487 retention figure is K=6, not K=8

`0.6487` was carried through this document and the placement draft as "the best
measured retention hit share (**K=8**)". **It is a real measured number with the
wrong label on it.** It is `honest_b` of the **rung-6** entry of
`build/llama-cuda/gplr19-honest-summary.json:5` — 0.64865196. The rung-8 entry
four lines below it reads `honest_b: 0.72833`, and that is the figure the record
should have carried.

Re-reading the `n_rows_resident` telemetry of every K arm on disk
(`tools/cache-h-survey.py`) gives the ladder independently:

| cap (K) | prompt A (`gplr19-k*-ab`) | prompt B (`gplr19-honest-*`) | reps |
| --- | --- | --- | --- |
| 8 (K=1) | 0.3076 | — | 1 |
| 16 (K=2) | 0.4326 | — | 2, identical |
| 32 (K=4) | 0.6260 | — | 2, identical |
| 48 (K=6) | 0.7141 | 0.6461 | 3, identical |
| 64 (K=8) | 0.7745 | 0.7255 | 2, identical |
| 96 (K=12) | 0.8218 | — | 2, identical |

`honest_b` and the raw residency rate are near-identical statistics of the same
arm: 0.64865 against 0.6461 at K=6, 0.72833 against 0.7255 at K=8, both about
0.0027 apart. Which matters, because:

**The "K=6 -> K=8 step of 0.0026" was one run differenced against itself.**
0.64865 (rung-6 `honest_b`) minus 0.6461 (the same run's residency rate) is
0.00255. The record read that as a *depth* step and concluded depth was
exhausted. It is a *metric* step, between two ways of counting the same K=6 arm,
and it is necessarily tiny. The real K=6 -> K=8 step is **0.0794** on residency
and **0.0797** on `honest_b` — the two agree, and both are thirty times what was
recorded.

**Three claims depended on it and all three were wrong:**

1. "the K=6 -> K=8 step moved h by 0.0026" — the step is **0.0794**, 30x larger,
   and the 0.0026 is not a step at all.
2. "the best measured retention is 0.6487, an 18-point gap to the 0.832
   break-even that depth tuning does not close" — K=12 measures **0.8218 /
   0.8319 steady state**, i.e. **at** the break-even. Depth closes it entirely.
3. "retention depth is exhausted" — it was not; the ladder was still climbing
   steeply at the last point anyone ran.

**What survives, and why.** The streaming path is still closed, and now on a
ground that no hit rate can touch: its non-transfer floor is 28.92 ms/token
against the incumbent's whole 24.41 ms token, so it loses at `h = 1` (§15).
*(Corrected 2026-09-30, eighteenth finding: the path's own floor is 23.69-26.47
ms, about parity at `h` = 1. It is closed on the operating point, where VRAM
caps K and so `h` (§15.7).)*
Separately, a miss costs `transfer + device` = 229.0 us against `host` = 35.7 us,
so a per-expert split beats a stream 6.4x at every hit rate and every VRAM
budget. And K=12's 96 rows/layer cost 6,480 MiB, which does not fit beside dense
weights (**1,921**, §12.1b), KV (1,328), recurrent state (61) and compute (428)
inside the owner's **7,055 MiB** budget (§10.1) — that is 10,218 MiB against
7,055, so it overshoots by 45% and would still overshoot a bare card's 8,462 by
21%. So even at break-even it was never runnable at the operating point.
**Three independent closures, none of which needs the 0.832 threshold.**

**Process note.** This is the seventh error of this kind found in this record by
re-deriving a published number from its raw artifact rather than from the prose,
and it is the most instructive: nothing was mistyped. A correct number was
copied under a wrong label, and a later section then differenced it against the
correct number for the same run and read the near-zero result as physics. Two
guards: `tools/cache-h-survey.py` regenerates the whole ladder from the traces
in one command, and any step between two rungs should be computed inside one
metric, from the source file, never across a table row.

---

## 15. MEASURED 2026-09-30: the streaming path's two constants, and its ~~unconditional~~ operating-point closure (corrected 2026-09-30, eighteenth finding)

> **CORRECTED 2026-09-30 (eighteenth finding): four of §15's premises are
> wrong, and "unconditional" is withdrawn. The streaming path stays closed, but
> on its operating point, not on "no hit rate is high enough".** Re-derived
> host-only from the same `summary.json` files and the source. No new runs.
>
> 1. **"Identical `peak_vram_bytes`" is a telemetry artifact, not a matched
>    design.** The prompt-A families ran 2026-09-13 22:01-22:22. That is before
>    the H-B fix (nested `8e9757169`, 2026-09-15), whose record says the
>    run-end peak was sampled after `end_row_cache` had freed the row cache
>    (§9.4).
>    - Measured: `gplr19-k8-ab` arm B records `cache_alloc` "40 layers x 64
>      rows/layer = 4370.1 MiB private VRAM", and arm A records `""`. Both
>      report 6,001,590,272 B. The harness takes the last `peak_vram_bytes` in
>      the trace (`tools/run-gplr19-rowcache-ab.ps1:389-390`).
>    - The honest family ran after the fix and shows the real shape. Arm A is
>      5,999,493,120 B, within 2 MiB of prompt A's. Arm B is 9,476,571,136 B
>      at K=6 and 10,663,559,168 B at K=8: the cache plus 38-78 MiB.
>    - The same runs, measured directly: each leg's attached
>      `tools/gpu-mem-poll.ps1` record puts arm B at +1,132.0 / +2,264.0 /
>      +3,316.0 / +4,448.0 MiB over arm A at K = 2 / 4 / 6 / 8 (§9.4).
>    - So in every family the two arms differ by the whole cache,
>      K × 572,719,104 B + 655,360 B. None is matched on VRAM.
> 2. **The A/B contrast varies the execution shape, not just the bytes.**
>    `load_row_set` (`src/llama-layer-streaming.cpp:9658-9667`) runs two
>    different paths:
>    - cache off: slot-keyed H2D straight into the slot;
>    - cache on: H2D of the misses into the layer's private cache, then a
>      device-side refresh of the slot bundle for the whole union, on every
>      token.
>
>    A line through both arms is therefore not one cost function, and the arms
>    show it:
>    - A single cost function would give every capacity family the same A/B
>      intercept. Instead they fall monotonically: **36.54 / 30.63 / 29.75 /
>      28.23 ms** at K = 2 / 4 / 6 / 8.
>    - Pooled over all 16 prompt-A arms, the line runs 3.6-3.9% under the arms
>      at 325 MB/token and 2.8-4.1% over them at 129 MB/token.
>    - The cache-on arms alone lie on a straight line (residuals ≤ 1.2%). The
>      cache-off arms sit ≈7.3 ms *below* that line's extrapolation to 573 MB.
>
>    So at full fetch the cache-off shape is the cheaper one. A two-point line
>    through one arm of each shape overstates the cache-on intercept by
>    Δ·F_B/(320 − F_B), where F_B is the cache-on arm's rows fetched per token.
>    With Δ = 7.26 ms this gives 35.99 / 30.81 / 29.38 / 28.58 ms, within
>    ±0.55 ms of all four measured intercepts. That is a consistency check on
>    the same arms, not an independent measurement. §15.1's 28.92 ms carries
>    this bias.
> 3. **The path's own constants come from its cache-on arms**, which share one
>    execution shape:
>
>    | cache-on arms | ctx | fit (± 1 SE) | h = 1 ceiling |
>    | --- | --- | --- | --- |
>    | prompt A, K = 2/4/6/8, 8 arms | 768 | 26.47 ± 0.31 ms + bytes / 8.83 GB/s | 37.79 tok/s = **0.92x** |
>    | kprobe, predictor off, K = 4/8/12, 3 legs | 1280 | 23.69 ± 0.17 ms + bytes / 12.43 GB/s | 42.22 tok/s = **1.03x** |
>    | honest, K = 6/8, 6 arms incl. `-repro` | 1280 | not fittable: two byte levels 45 MB apart, arm-B replicates up to 9% apart | — |
>
>    The h = 1 ceiling is therefore **about parity (0.92x-1.03x), not "loses by
>    16%"**. Two limits:
>    - Both fits extrapolate to zero bytes over roughly 60% of their measured
>      range.
>    - Each byte level is a different K, so any machinery cost that trends with
>      K lands in the intercept, with unknown sign. This is §15.5 item 2's
>      confound, and it applies to these arms too.
> 4. **Every throughput family is short-context.** Prompt A ran at ctx 768,
>    honest and kprobe at 1280, and the anchor at 52,819. The anchor's KV is on
>    the device (`CUDA0 KV buffer size = 1328.12 MiB`, anchor log line 292).
>    - What the anchor pays and these runs do not is mainly reading ≈575 MB of
>      q8 KV per token. That is expected to cost ≈0.6-1.0 ms and is
>      unmeasured.
>    - It is not "most of the anchor" (§15.5 item 1).
>    - At 1.0 ms it moves each ceiling about 4% against the path: 0.88x for
>      prompt A, 0.99x for kprobe.
>
> **Verdict, restated.**
> - **Measured:** every row-cache configuration in §15's four families loses.
>   The best is kprobe `off-k12` at 31.85 ms = 0.77x, and it peaks at
>   12,354 MiB, 1.75x the realised 7,055 MiB budget.
> - **Measured, plus arithmetic:** inside that budget, at short context, the
>   path holds at most K = 2. The cache-free arm is ≥ 5,722 MiB (the peak is
>   sampled, §9.4) and K = 2 adds 1,093 MiB. K = 2 measures 63.21 ms =
>   **0.39x**.
> - **Expected:** at the settled 236,000 context the cache-free arm alone
>   reaches ≈8,157 MiB. That is 2,435 MiB more KV at 0.010376 MiB/token
>   (§13.6), so the path does not fit the budget at all as built.
> - **Arithmetic on the measured fits:** break-even needs h ≥ 0.984 in the
>   kprobe family. The prompt-A family has no break-even, since 26.47 > 24.41.
>   With the ≈1 ms context term, neither family has one.
> - h → 1 means every routed expert resident (K ≥ 32, ≈18.3 GB of cache). There
>   the fair baseline is stock given the same VRAM, not the 8 GB anchor.
> - **The closure therefore rests on the operating point.** Withdrawn:
>   "unconditionally", "no hit rate is high enough", "a perfect cache still
>   loses by 16%", "the conservative reading", and "the only estimate from arms
>   matched on peak VRAM" (§15.2).
>
> §15.6's bound on `d` and its `M` are re-derived from the cache-on intercepts
> (`tools/d-bound.py`).
>
> Evidence:
> - `build/llama-cuda/gplr19-k{2,4,6,8}-ab/summary.json`;
> - `gplr19-honest-k6-ab/summary.json`, the nested
>   `gplr19-honest-k8-ab/summary.json`, and
>   `gplr19-honest-k6-repro/summary.json`;
> - `gplr58-kprobe-summary.json`;
> - `tools/layer-streaming/main.cpp:5026-5037`: the row cache is allocated at
>   bring-up, and the H-B refresh follows it.

The T-GPLR-19 A/B runs were built to test retention, but they are also a direct
two-point measurement of the path's cost structure and nobody read them that
way. Each run holds arm A (cache off, stages every demanded row) against arm B
(cache on, stages only misses), at **identical `route_digest`** and — for the
`gplr19-k*-ab` family — ~~**identical `peak_vram_bytes`**. So each run varies
staged bytes and nothing else~~ *(struck 2026-09-30, eighteenth finding: the
identical peaks are pre-H-B telemetry that never saw the row cache, and the
two arms run different execution shapes)*, and

    ms_per_token = fixed + staged_bytes_per_token / feed

can be fitted with no physical parameter assumed. `tools/stream-slope.py` does
it. Pooling the ~~matched~~ K=6 and K=8 families (8 arms, 3 byte levels):

| arm | h | staged B/token | ms/token | TG tok/s | PP tok/s |
| --- | --- | --- | --- | --- | --- |
| k6 A1 | 0.0000 | 572,719,104 | 84.15 | 11.88 | 91.15 |
| k6 A2 | 0.0000 | 572,719,104 | 84.22 | 11.87 | 91.42 |
| k6 B1 | 0.7141 | 163,659,552 | 45.24 | 22.10 | 91.44 |
| k6 B2 | 0.7141 | 163,659,552 | 45.36 | 22.05 | 91.57 |
| k8 A1 | 0.0000 | 572,719,104 | 84.28 | 11.87 | 91.47 |
| k8 A2 | 0.0000 | 572,719,104 | 84.18 | 11.88 | 91.44 |
| k8 B1 | 0.7745 | 129,075,264 | 41.09 | 24.34 | 91.18 |
| k8 B2 | 0.7745 | 129,075,264 | 40.61 | 24.62 | 92.12 |

**Fit: `ms = 28.92 + bytes / 10.35 GB/s`, residuals within ±1.9% on every arm.**

*Corrected 2026-09-30 (eighteenth finding): three byte levels cannot show the
curvature. With K=2 and K=4 pooled in, the same A/B form misses by up to 4.1%.
The cache-on arms alone give `ms = 26.47 + bytes / 8.83 GB/s`, with residuals
≤ 1.2%.*

### 15.1 The path loses at h = 1

> **Corrected 2026-09-30 (eighteenth finding).** The decomposition below is
> right for one execution shape, which is why the cache-on-only fit is valid.
> It does not hold across the A/B pair, because the two arms run different
> shapes.
>
> - The pooled intercept is biased *up*. It is not "conservative" in the sense
>   argued below: the cache-on arms put the h = 1 token at 26.47 ms (prompt A)
>   and 23.69 ms (kprobe, predictor off), both below 28.92.
> - So h = 1 gives 0.92x-1.03x, not 0.84x, before the expected ≈4% context
>   correction.
> - Withdrawn: "a perfect cache still loses by 16%" and "*sufficient `h` does
>   not exist*".
> - Break-even needs h ≥ 0.984 in the best family. The closure is re-based on
>   the operating point (top of this section).

`fixed` = **28.92 ms/token** is everything the path costs that is not transfer:
device expert compute, attention, and the rowset machinery itself. The stock
incumbent's *entire* token is **24.41 ms**. So:

    h = 1, zero bytes crossing the link  ->  34.58 tok/s  vs  40.96 stock  = 0.84x

**This closes the streaming path with no appeal to hit rate, to `d`, to a feed
rate, or to a break-even threshold.** Every previous closure in this document
argued that the path could not reach a sufficient `h`. The correct statement is
that *sufficient `h` does not exist*: a perfect cache still loses by 16%. The
0.832 break-even, the 18-point gap, the retention ladder — the whole argument
was conducted on a quantity that could never have decided the question.

**Is the intercept safe, given no arm ran at h = 1?** It is an extrapolation
from the smallest measured byte level (129 MB/token) to zero, i.e. 29% beyond
the edge of the 129–573 MB range — modest, but the structural argument matters
more than the distance. Staged bytes are perfectly collinear with rows fetched
here (`fetched = (1-h)·40·8`, every row the same size), so write the token cost
in full generality as a constant plus a per-fetched-row term plus a
per-resident-row term:

    cost = a + b·F + g·R,  with R = 320 − F
         = (a + 320g) + (b − g)·F

The fit returns the intercept `a + 320g` — which **is** the cost at `F = 0`,
all 320 rows resident, i.e. exactly `h = 1`. So the intercept is the right
quantity under all three cost structures: machinery that is constant per token,
machinery charged per row fetched (vanishes at `h = 1`), and machinery charged
per row *resident* (maximal at `h = 1`, and correctly carried into the
intercept). The collinearity costs us the ability to split the slope into
per-byte and per-fetch parts — which is why §15.6 can only bound `d` rather than
solve it — but it does not bias the intercept. **The h = 1 figure is not an
optimistic extrapolation; if anything it is the conservative reading**, since a
per-resident-row cost would make the real `h = 1` token *more* expensive than
28.92 ms, not less.

Three other capacity families fit the same form at different operating points.
Two reproduce the finding outright; the third needs explaining, and is:

| family | matched VRAM | feed | fixed ms | tok/s at h = 1 |
| --- | --- | --- | --- | --- |
| k2 + k4 (prompt A) | yes | 11.05 GB/s | 32.35 | 30.91 = 0.75x |
| **k6 + k8 (prompt A)** | **yes** | **10.35 GB/s** | **28.92** | **34.58 = 0.84x** |
| honest k6 + k8 (prompt B) | **no** | 8.91 GB/s | 25.90 | 38.62 = 0.94x |
| kprobe k4/k8/k12 (prompt C) | **no** | 12.71 GB/s | 24.20 | 41.33 = 1.01x |

*Struck 2026-09-30 (eighteenth finding): the "matched VRAM" column is wrong in
both "yes" rows. The prompt-A peaks are pre-H-B telemetry that never saw the
row cache, and in every family the arms differ by the whole cache.*

- *The first three rows are A/B fits and carry the intercept bias.*
- *The kprobe row is all cache-on, so it is one shape. It pools predictor-off
  and predictor-on legs, and the off legs alone give 23.69 ms.*
- *The cache-on-only prompt-A intercept, over all four K, is 26.47 ms.*

~~Only the first two are matched comparisons.~~ The honest pair's arms differ in
`peak_vram_bytes` by up to 4.7 GB and the kprobe legs by 4.3 GiB, so neither
`fixed` is measuring the same configuration twice. *(Corrected 2026-09-30:
the prompt-A arms differ by the same cache; their telemetry hid it.)*

**The kprobe row is the one that appears to cross, and it does not** — it is
analysed in §15.5. Its legs also ran at ctx 1280 against the anchor's 52,819,
so its intercept contains almost none of the attention the 24.41 ms anchor is
mostly made of; the matched baseline at that context is far above 40.96 tok/s,
which makes the real gap there wider, not narrower. It is listed here rather
than omitted because omitting the least convenient family is how §14 happened.
*(Corrected 2026-09-30, eighteenth finding: the anchor's KV is on the device,
so the attention it pays beyond ctx 1280 is mainly reading ≈575 MB of q8 KV
per token. That is expected to be ≈0.6-1.0 ms, not "most" of 24.41 ms.*

- *Stock at ctx 1280 is therefore expected at ≈42-43 tok/s, not "far above"
  40.96.*
- *The kprobe predictor-off ceiling moves from 1.03x to ≈0.99-1.01x: about
  parity, neither a clear crossing nor a wider gap.)*

`fixed` does fall as capacity rises, which is the expected direction (a larger
cache does less staging work per token beyond the bytes themselves). *(Corrected
2026-09-30, eighteenth finding: in the prompt-A families the fall is the A/B
shape bias, Δ·F_B/(320 − F_B), shrinking as fewer rows are fetched. Δ = 7.26 ms
reproduces all four family intercepts within ±0.55 ms. It is not a capacity
effect.)* But the
two rungs that would carry the trend past 24.41 ms, K=8 and K=12 in the kprobe
family, peak at **1.44x and 1.75x** the entire 7,055 MiB budget (§10.1). The
trend escapes the budget before it escapes the intercept.

### 15.2 The marginal feed is 10.35 GB/s, not 8.26

The fitted slope is an end-to-end measurement by construction: it uses only a
throughput and a byte count, with no timer inside the copy path. It lands at
**8.9–12.7 GB/s across four families and two harnesses** (the fourth, at
12.71 GB/s, is the T-GPLR-58 K ladder — see §15.5), against the 8.26 GB/s this
document had recorded as "the best feed ever measured end to end". The run
drivers' own `conversion_gbs` agree — 8.59 at rung 6, 9.15 at rung 8. §3 is
corrected: 8.26 GB/s is the best *instrumented whole-bank* figure, and the
instrumented timers undercount because they charge submit overhead that a real
token partly absorbs. ~~**10.35 GB/s remains the number to quote**, because it is
the only estimate from arms matched on peak VRAM; the wider band is the honest
spread across unmatched designs.~~

> **Corrected 2026-09-30 (eighteenth finding).** No family is matched on VRAM
> (top of §15), so no single figure is "the one to quote".
>
> - The A/B slopes overstate the cache-on path's marginal feed, through the same
>   shape bias. A two-point A/B line has slope s − Δ/(320 − F_B), which is
>   shallower than the cache-on path's own s.
> - The cache-on arms alone give **8.83 GB/s** (prompt A) and **12.43 GB/s**
>   (kprobe, predictor off).
> - Each figure is the path's end-to-end cost per staged byte, including its
>   per-row cache machinery. It is not the bare link rate.
> - The honest statement is the band: 8.8-12.4 GB/s cache-on, and 8.9-12.7 GB/s
>   from the biased A/B and mixed fits.
> - The prompt-A cache-on figure is close to the 8.26 GB/s instrumented
>   whole-bank figure. The "faster than recorded" reading survives only in the
>   kprobe family.

This cuts **against** the path's case in the record and is recorded anyway: a
faster link makes streaming better, and it still loses, because the loss is in
`fixed`. *(Corrected 2026-09-30: the path's own `fixed` is at about parity with
stock (§15.1 correction). It loses on the operating point: the VRAM budget caps
it at K = 2, h ≈ 0.43.)*

### 15.3 The row cache does nothing for prefill — measured, not argued

Across the four ~~matched~~ arms of each family, prefill moves by **0.46% (K=6),
1.01% (K=4) and 1.03% (K=8)** between cache-off and cache-on, on identical
routes. §12.3 and §13 argued this from route-union width (~254 of 256 experts
demanded per layer at ubatch 512, `gplr14-route-overlap.txt`); this is the same
conclusion from throughput directly, on runs that already existed. **Row caching
is a TG-only instrument. It cannot move the charter's first priority and never
could.**

> **Corrected 2026-10-01 (twenty-ninth finding). The "254 of 256" figure is a
> 512-token figure and must not be read as a property of a real prompt.** It
> comes from a 512-token prompt, where a 512-token ubatch draws almost every
> expert at least once. Measured on the **50,000-token** anchor prompt, the
> route union is **0.717 / 0.815 / 0.855** at ubatch 512 / 2048 / 4096 — that
> is 72% / 82% / 86% of expert bytes, not 99%. Source:
> `build/stock-levers/route-union-report.txt`. The conclusion of §15.3 is
> unchanged and in fact strengthened: the smaller the union, the *less* there is
> for a row cache to avoid copying, which is the same conclusion from the other
> direction. But any reasoning that treats 254/256 as a real-prompt hit rate is
> reading a synthetic figure.

> **Corrected 2026-09-30 (eighteenth finding).** The conclusion holds, and more
> strongly: it holds by construction.
>
> - The cache is allocated at bring-up (`tools/layer-streaming/main.cpp:5026`).
>   It is consulted only by the TG row-set executor (`load_row_set`,
>   `src/llama-layer-streaming.cpp:9668`), so prefill runs the same code in both
>   arms.
> - What these numbers measure is narrower: holding K × 546.2 MiB of idle cache
>   through prefill costs nothing measurable at ctx 768.
> - The honest family's exclusion reason below is also wrong. Its arms differ in
>   VRAM exactly as prompt A's do; prompt A's telemetry hid it.
> - The real reason is replicate noise. Two identical-configuration K=6 cache-on
>   runs report 126.31 and 115.28 tok/s of prefill, 9.6% apart, and the `-repro`
>   pair adds 117.64 and 121.20. Any A/B difference inside that spread is
>   unreadable.

Two families are excluded from that statement and both are excluded for cause:
`gplr19-k2-ab` arm A1 reports 79.02 tok/s PP against 89.98-92.42 for its three
siblings, a single cold outlier that inflates that family's spread to 17%; and
the `honest` family spreads 8-10% but ~~is not a matched comparison at all (§15.1)~~
*(corrected: its identical-configuration replicates spread as much)*.
Neither is evidence that the cache moves prefill — one is a cold run and the
other ~~varies VRAM by 4.7 GB~~ is replicate noise.

### 15.5 A fourth family, a faster link, and the predictor's measured tax

`gplr58-kprobe-summary.json` is a different harness, a different prompt
(`calc-prompt.txt`, 885 PP tokens) and the only K=12 throughput on disk. Six
legs, three capacities, each run twice — once with the coact predictor off and
once with it on:

| leg | h | staged B/token | ms/token | tok/s | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| off-k4 | 0.5349 | 266,588,256 | 45.11 | 22.17 | 7,986 MiB |
| arm-k4 | 0.5349 | 266,588,256 | 45.26 | 22.10 | 7,986 MiB |
| off-k8 | 0.7273 | 156,355,680 | 36.35 | 27.51 | 10,170 MiB |
| arm-k8 | 0.7273 | 156,355,680 | 36.58 | 27.34 | 10,170 MiB |
| off-k12 | 0.8218 | 102,193,056 | 31.85 | 31.40 | 12,354 MiB |
| arm-k12 | 0.8218 | 102,193,056 | 32.67 | 30.61 | 12,354 MiB |

**Fit: `ms = 24.20 + bytes / 12.71 GB/s`, residuals within ±1.3%.**
*(2026-09-30: the three predictor-off legs alone give `ms = 23.69 ± 0.17 +
bytes / 12.43 GB/s`. The on legs carry the predictor tax measured below.)*

> **Corrected 2026-09-30 (eighteenth finding).** Neither reason below survives
> as written.
>
> - **Item 1.** The anchor's KV is device-resident, so the attention it pays
>   beyond ctx 1280 is mainly reading ≈575 MB of q8 KV per token. That is
>   expected ≈0.6-1.0 ms, unmeasured, and far from "most" of the anchor.
> - **Item 2.** Its premise is false on all three counts:
>   - the `gplr19` A arms allocate *no* cache (`cache_alloc ""`);
>   - the two arms run different execution shapes (src:9658-9667);
>   - the byte-exact peaks are pre-H-B telemetry that never saw the cache.
>
>   The A/B design did not escape this family's confound. It traded it for a
>   worse one. Its cache-on arms have the same K collinearity as these legs.
> - **What survives is an about-parity ceiling.** Charged ≈1 ms of context,
>   this family's intercept gives 0.99-1.01x at h = 1: neither a crossing nor
>   the wider gap item 1 predicted.
> - Every leg here is over the budget anyway (next paragraph), which is where
>   the closure now rests.

**Its intercept must not be compared to the 24.41 ms anchor, and this is
deliberate to state because the naive reading inverts the conclusion.** Taken at
face value, 24.20 ms would say the streaming path reaches 41.33 tok/s at `h = 1`
and *edges past* stock. It does not, for two reasons that are both about the
comparison rather than the path:

1. **Context.** These legs ran at `ctx_size 1280`. The 24.41 ms anchor is at
   **52,819** tokens of context. Attention per token scales with context, so
   this intercept contains almost none of the attention the anchor is mostly
   made of. The matched baseline would be stock at ctx 1280, which is
   considerably faster than 40.96 tok/s — so the real gap at *this* operating
   point is **wider** than at the anchor's, not narrower. *(Corrected
   2026-09-30: ≈0.6-1.0 ms of attention, expected; stock at ctx 1280 ≈42-43
   tok/s.)*
2. **Confound.** Each K leg is its own configuration: peak VRAM rises 7,986 ->
   10,170 -> 12,354 MiB as staged bytes fall. The regressor is collinear with
   cache size, so the intercept absorbs any K-dependent machinery trend. ~~The
   `gplr19` A/B design does not have this problem — its A and B arms allocate
   the same cache and differ only in whether they retain into it, at byte-exact
   identical `peak_vram_bytes`. That is why §15.1's intercept is the usable one.~~
   *(Struck 2026-09-30, eighteenth finding — see the correction above.)*

Also note K=8 and K=12 here peak at 10,170 and 12,354 MiB, i.e. **1.44x and
1.75x the owner's entire 7,055 MiB budget** (§10.1 — an earlier revision of this
paragraph used 8,416 MiB, which is neither the realised budget nor a bare card's
8,462). Those two rungs are not runnable at the operating point regardless of
what they measure — **and neither is K=4**, whose 7,986 MiB is itself 1.13x the
budget. The earlier claim that "only K=4 fits" was an artifact of the inflated
budget figure; on the fitter's own number **nothing in this family fits**, which
strengthens the closure rather than weakening it. (K=4 would fit a bare card's
8,462 MiB, and it is the slowest leg in the family regardless.)

**What this family does contribute, cleanly:**

- **A fourth slope, and the fastest: 12.71 GB/s.** The measured marginal feed
  now spans **8.9 to 12.7 GB/s** across four independent families and two
  harnesses. Every one of them is above the 8.26 GB/s this document had been
  quoting as the end-to-end best (§15.2). *(2026-09-30: 12.43 GB/s from the
  predictor-off legs alone. §15.2's correction gives the cache-on band,
  8.8-12.4 GB/s.)*
- **An independent check on `tools/cache-h-survey.py`.** Running the survey over
  all six traces reproduces the harness's own `leg_share` exactly at every rung
  and in both roles: 0.5349 / 0.7273 / 0.8218 against 0.53486 / 0.72726 /
  0.82175. Two readers — the survey summing `n_rows_resident` out of
  `rowset_fetch` events, the harness accounting during the run — agree to four
  places. This validates the survey's arithmetic, which is the tool the §14 h
  ladder rests on. (It is the *same traces* read twice, so it is a reader
  cross-check, not a second workload.) The survey's second-half figure at K=12
  is **0.8319**, which is the retention number quoted in the placement draft.
- **The predictor's cost, measured, with its benefit measured at exactly
  zero.** At every capacity the arm leg evaluated **81,424 proposals** and
  submitted, consumed and placed **none** of them — `"submitted": 0`,
  `"consumed": 0`, `"placed_stages": 0` at K=4, 8 and 12 — while
  `fill_room_admittable` rose 189 -> 771 -> **1,588** stages out of 10,240
  (1.85% -> 7.53% -> 15.51%). So it is not that there was no room. And the
  residency streams confirm it did nothing: the arm and off traces at each K
  report **bit-identical h**, both overall and on the second half. Running it is
  not free, though — the arm legs are slower than their off twins by **0.33%
  (K=4), 0.62% (K=8) and 2.50% (K=12)**. The coact predictor costs up to 2.5% of
  TG to place nothing, and the tax grows with the room it declines to use.

That last point closes the predictor line from the cost side. §13.4 closes most
of it from the value side too, though the corrected numbers leave a narrow
residual: at 64 rows LRU-K already clears `h*` on both prompts so extra `h` is
worthless there, but at 48 rows, and for warm-then-pin's weak prompt, `h` still
undershoots. What a predictor would have to do is lift the *undershooting*
prompt without pushing the other past `h*` — variance reduction, not hit-rate
maximisation. **Measured against that narrower target it still fails: it placed
nothing at any capacity, including the two where there was room and headroom.**

### 15.6 Where the 28.92 ms actually goes — and the first bound on `d`

> **Corrected 2026-09-30 (eighteenth finding): the bound tightens, `M` shrinks
> by a third to a half, and the "not the link" conclusion is inverted at the
> operating point.** The 28.92 ms decomposed below is the withdrawn pooled A/B
> intercept (§15 correction, item 2). Here is the same arithmetic against the
> path's own cache-on intercepts. It is rerun by `tools/d-bound.py` with the same
> `h_stock` 0.188 and `c` = 35.7 us:
>
> | intercept | ctx | `dA` = 0 | `dA` = 1.0 ms (expected context term) |
> | --- | --- | --- | --- |
> | ~~A/B pooled, 28.92 ms~~ (withdrawn) | 768 | `d` ≤ 53.0 us, `M` 9.94 ms (41%) | — |
> | cache-on prompt A, 26.47 ms | 768 | `d` ≤ 43.6 us, `M` 7.49 ms (31%) | `d` ≤ 47.5 us, `M` 8.49 ms (35%) |
> | cache-on kprobe off, 23.69 ms | 1280 | `d` ≤ 32.9 us, `M` 4.71 ms (19%) | `d` ≤ 36.8 us, `M` 5.71 ms (23%) |
>
> Terms:
> - `M` is taken at `d` = 14.8 us. The percentages are of the 24.41 ms stock token.
> - `dA` is the attention the anchor pays at 52,819 tokens of context and these
>   short-context runs do not. It is expected, not measured.
> - Arithmetic labels: the bounds are arithmetic on measured intercepts; `dA` is
>   expected.
>
> What changes:
>
> - **The bound on `d` tightens, from 53.0 to 32.9-47.5 us.** It still excludes
>   nothing in §13.4's 8-30 us sweep, so the ceiling-band reading below stands.
> - **`M` is 4.7-8.5 ms, 19-35% of a stock token, not 41%.** The kprobe
>   decomposition of its 23.69 ms `fixed` is:
>   - `O` ≈ 14.2 ms, 60%;
>   - device expert FFN ≈ 4.7 ms;
>   - `M` ≈ 4.7 ms, 20%.
>
>   So `M` is not "most of that `fixed`".
> - **At the operating point the problem is the link, not `M`.** At K = 2, the
>   largest cache the 7,055 MiB budget admits at the measured short context (at
>   236k even K = 0 does not fit, expected), prompt A measures 63.21 ms =
>   0.39x. The cache-on fit splits it:
>   - 36.8 ms (58%) is 324.8 MB staged per token at 8.83 GB/s;
>   - 26.5 ms is `fixed`, of which `M` is 7.5 ms (12%).
>
>   Deleting `M` entirely would leave 55.7 ms = 0.44x. What binds is how many
>   bytes the budget forces across the link — VRAM, then `h` — so a successor
>   has to cut staged bytes first.
> - **The "not the link (10.35 GB/s…)" figure is the biased pooled slope.** The
>   cache-on band is 8.8-12.4 GB/s (§15.2 correction).

The intercept can be decomposed against the stock anchor, because both paths
share the same non-expert work. Write `O` for that shared remainder, `M` for the
rowset machinery, `d` for the device expert FFN:

    stock:      O + 320*(0.188*d + 0.812*c) = 24.41 ms       (c = 35.7 us)
    streaming:  O + 320*d + M               = 28.92 ms       (h = 0, all on device)

Two equations, three unknowns, but `M >= 0` closes it from one side:

    0.260*d + M = 13.783 ms   ->   **d <= 53.0 us**

That is the **first bound on `d` derived from data rather than assumed**, and it
is weak — it excludes nothing in §13.4's 8-30 us sensitivity sweep, which is the
useful thing it tells us: the sweep spans the admissible range and the 1.32x to
1.46x ceiling band is not hiding a worse case outside it.

Run it the other way, at the working estimate `d = 14.8 us`:

    M = 13.783 - 0.260*14.8 = **9.94 ms/token of rowset machinery**

**41% of a whole stock token, spent on cache bookkeeping, with no bytes moving
and no arithmetic done.** If that number is even roughly right it relocates the
streaming path's problem entirely: not the link (10.35 GB/s, faster than
recorded), not the GPU (`d` is small), but the staging, descriptor-building and
synchronisation around the copy. Any future proposal in this family has to
attack `M` first, and `M` is exactly what the per-expert split (§13) does not
have — it stages nothing, so its `O` is the stock path's own 14.25 ms.
Arithmetic in `tools/d-bound.py`, which also shows that measuring `h_stock`
(§12.1b) moved both terms *against* the streaming path: the bound on `d`
tightens from 53.9 to 53.0 us, and `M` rises from 9.69 to 9.94 ms.

### 15.7 What this changes

> **Corrected 2026-09-30 (eighteenth finding).** Four of the bullets below are
> amended in place: the first, second, fourth and last. The split and
> placement bullets are §13's claims and are not re-audited here.

- §3a's model-based validation is withdrawn and replaced by the direct fit;
  the "agreement to within 13%" is now agreement to within 1.9%. *(Corrected
  2026-09-30: the 1.9% belonged to the pooled A/B fit on three byte levels,
  where the shape bias hid the curvature — with K2/K4 added it misses by up to
  4.1%. The cache-on fit, `26.47 ms + bytes / 8.83 GB/s`, reproduces all eight
  prompt-A cache-on legs within 1.2%.)*
- ~~The streaming/transfer path is closed **unconditionally**. No successor
  proposal in this family needs evaluating against a hit rate again; it needs
  to show a `fixed` below 24.41 ms/token first, and §15.6 says most of that
  `fixed` is machinery — a property of the implementation, not of the policy.~~
  **Corrected 2026-09-30: closed at the operating point, not unconditionally.**
  - **The path's own `fixed` is about parity with stock.** At `h` = 1 it gives
    0.92x (prompt A) and 1.03x (kprobe), arithmetic on measured fits. Charged
    the expected 0.6-1.0 ms context term, that becomes 0.88-0.90x and
    0.99-1.01x.
  - **So the hit rate matters again, but break-even needs `h` ≥ 0.984.** That
    is reachable in kprobe only, and only if the context term is below
    ≈0.72 ms.
  - **Break-even needs a cache that cannot be allocated.** `h` = 1 needs
    K ≥ 32, ≈18.3 GB of cache. The 7,055 MiB budget admits K ≤ 2 at short
    context (63.21 ms = 0.39x, measured) and nothing at 236k (expected).
  - **What binds is VRAM, then `h`, not `fixed`.** Machinery is 19-35% of a
    stock token (§15.6 correction), not most of `fixed`.
  - **The best measured leg is off-k12:** 31.85 ms = 0.77x, at 12,354 MiB
    peak, over budget.
- Every remaining lever is in the **split** (§13), whose `fixed` is the stock
  path's own 14.25 ms because it stages nothing.
- ~~§12.3's prefill closure survives but with a narrower margin: 1.20x, not
  1.50x, once the faster marginal feed is used. That is close enough to want
  the `-ncmoe` prefill differential it was already flagged for.~~
  **Superseded 2026-09-30 (ninth finding, §12.5): §12.3 is retracted.**
  - **Stock prefill never computes host experts on the CPU.** At a batch of 32
    or more, op offload copies each host bank to the GPU, so prefill is
    copy-bound.
  - **"Not resident → host" holds only at u512, and only with
    `--no-op-offload`.** There the CPU beats the pageable copy by 1.04-1.21x
    at 16 threads and by 1.30-1.52x at 28 threads.
  - **From u1024 the copy wins, and its lead grows with the ubatch:**
    1.33-1.55x at u1024 and 2.56-2.98x at u2048, pageable.
  - **The PP levers are ubatch and pinned memory** (§12.5).
- The **predictor line is closed from both sides**, with a narrowed value-side
  argument: §13.4 shows the useful target is `h` held near `h*`, not `h`
  maximised, and LRU-K K=8 already clears `h*` on both prompts at 64 rows;
  §15.5 shows the coact predictor placed zero rows at three capacities while
  taxing TG by up to 2.5%. No further predictor experiment is warranted.
- **§13.4's placement ranking inverted** under the same re-derivation
  discipline that produced §14: LRU-K now leads warm-then-pin at both
  capacities, and LRU-K K=6 at 48 rows dominates warm-then-pin at 64 rows on
  both throughput and VRAM. The ranking is conditional on `d` ≈ 14.8 us.
- Cost: **zero GPU time, zero new runs.** Everything in §15 was extracted from
  `summary.json` files and traces that have been on disk since the T-GPLR-19
  and T-GPLR-58 ladders ran. The four-arm A/B design that produced them was
  built to answer a retention question ~~and happens to be a controlled two-point
  measurement of the cost structure; nobody had read it that way~~.
  *(Corrected 2026-09-30: the design is not a controlled measurement of the
  cost structure.*
  - *Its arms differ in execution shape (src:9658-9667).*
  - *They also differ in VRAM: arm A allocates no cache.*
  - *The byte-identical peaks are pre-H-B telemetry that excludes the row
    cache (§9.4).*
  - *Its cache-on arms alone give the path's constants, and those are what this
    correction uses.)*
