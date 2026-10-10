# Layer residency vs stock llama.cpp — owner summary

**Campaign: hybrid CPU/GPU context inference optimization. Written 2026-10-01.**
Standalone. Everything needed to read this document is inside it; no other file must be opened.

Hardware and model: NVIDIA GeForce RTX 5090 Laptop GPU (24 GB), AMD Ryzen 9 9955HX (16 cores /
32 threads). Model `Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf` — a mixture-of-experts model, 40 decoder
layers, 19.91 GiB of quantized weights. Stock = the unmodified anchor server
`C:\Development\AI\llamacppCUDA` (build 10892). LM = this project's layer-major engine with
expert-bank residency.

**Every gate in this campaign is soft evidence, not release authorization.** Nothing here is
"ready" or "approved". Acceptance is the owner's decision, made from this summary rather than
from any checker exit code. Nothing has been staged, committed, pushed or branched, and no such
action is authorized.

---

## 1. Goal

**What was asked.** Exhaust every optimization available in hybrid CPU/GPU context inference on
this machine, re-check previous work on the topic, redo it where the implementation or the
arithmetic was wrong, and make decisions by common sense rather than by deference to prior claims.

**What was actually settled.** Two questions, answered with measurements on the owner's own two
recorded server configurations rather than on synthetic shapes:

1. **Does layer-major prefill with expert-bank residency beat stock llama.cpp on the charter's
   first priority (prefill speed), at equal VRAM?** Yes, by roughly 1.8–1.9x at the 50,000-token
   prompt, and the margin shrinks as prompts get longer.
2. **Does it beat stock overall, including token generation?** No. Token generation is *worse*
   wherever it works at all, and it does not work at all at the two large contexts the owner
   actually runs.

The campaign's most useful output is not the layer-residency result — it is that **two free
llama.cpp flags beat the entire custom engine on the owner's recorded configuration, at zero
development cost** (see §4).

Two further things were settled by inspection rather than by experiment: one real defect in the
vendored tool that silently disables a documented flag, and one kernel-selection bug that is
worth ~1.19–1.38x of prefill on its own.

---

## 2. Plain mechanism

**The problem.** This model is a *mixture-of-experts* (MoE) model. For every token it processes, a
small "router" picks 8 of 256 expert sub-networks per layer, and only those 8 are computed. The
whole model is 19.91 GiB, but the part needed for any one token is a small fraction of that. The
temptation is to keep everything in VRAM and be done — but 19.91 GiB does not fit in 24 GB once
you add the **KV cache** (the per-token attention history, which grows with context length) and
the compute buffers.

**What stock does.** Stock keeps as many of the 42 decoder layers in VRAM as its `--fit-target`
budget allows, and computes the remaining layers on the CPU. The CPU is not idle during prefill —
it is doing real expert math — and that CPU work is the bottleneck.

**What layer-major does (LM).** Instead of splitting by *layer*, LM splits by *time*. It holds
only **3 layer-sized weight packages** in VRAM at once and walks the model front to back: load
layer 1's package, process layer 1 for the *entire* prompt, then swap in layer 2's package,
process layer 2 for the entire prompt, and so on. Because only three packages are ever resident,
total VRAM use collapses to roughly 10–15 GiB regardless of the model's 19.91 GiB size. The
trade is explicit: you pay a host-to-device copy per layer, and you buy VRAM headroom.

**Why that makes prefill faster.** At the owner's context, VRAM headroom is what lets the copy
overlap with compute instead of queueing behind it. Stock, squeezed for space, copies a layer's
experts to the GPU *and then computes them on the GPU* — the copy is serialized with the compute
that follows it. LM's deeper pipeline overlaps the two. The result is the ~1.8–1.9x measured on
prefill.

**Why it is only faster at prefill.** Prefill is the one phase where the entire prompt flows
through every layer, so LM's "load a package once, use it for the whole prompt" advantage is
maximal. Token generation is the opposite: it produces *one token at a time*, and each new token
must pass through all 40 layers again. So LM re-streams the same packages over and over for what
is, in total, very little arithmetic per copy — the mechanism's strength inverts into its
weakness.

Worse, token generation needs a different memory layout. For generation, LM must **migrate the
whole model to VRAM** (a "handoff") so the single generated token can touch every layer without
streaming. At the owner's contexts there is not enough VRAM for the model *plus* the KV cache, so
the handoff cannot happen — and layer-major then refuses to generate at all. Where a partial
handoff does fit, generation runs but is slower than stock while using more VRAM, because the
partially-migrated layout mixes host and device expert work.

**A separate, smaller finding.** A flash-attention kernel-selection bug was found along the way:
when the attention operation is causal with no explicit mask, the GPU falls back to a slow kernel
because an internal optimization gate does not fire. Fixing the gate is worth **1.19x at 50k
prompts, rising to 1.38x at 200k** — the longer the attention window, the more the slow kernel
costs. This fix is currently *not* adopted; it lives in a side copy only.

---

## 3. Glossary

Every term used above, defined once.

- **PP (prefill / prompt processing)** — the phase that reads the whole input prompt and fills the KV cache. Measured in tokens per second. This is the campaign's first priority.
- **TG (token generation)** — the phase that produces output tokens one at a time, each attending over everything before it.
- **MoE (mixture-of-experts)** — a model whose per-layer computation is split into many small "experts", of which only a few are used per token.
- **Router** — the tiny network that decides which experts a token uses.
- **top-8 of 256** — this model routes each token to 8 of 256 experts per layer, so it stores 256 experts' worth of weights but computes only 8.
- **Expert bank** — the set of a layer's expert weights, treated as one unit for VRAM residency. Roughly 1.77 MB per expert per layer.
- **Layer residency** — how many of the model's 42 decoder layers are held in VRAM. Stock reports this as `offloaded N/42 layers`.
- **Layer-major** — the scheduling order: process layer 1 for the whole prompt, then layer 2, and so on. The alternative, "batch-major", processes a chunk of the prompt through all layers before moving to the next chunk.
- **Residency window / slots** — how many layer-sized packages LM keeps in VRAM simultaneously. This campaign used 3.
- **Handoff** — the PP→TG transition where the whole model is migrated to VRAM so generation can run. LM cannot generate sustained tokens without one.
- **Partial handoff** — migrating only some layers' expert banks to VRAM and computing the rest on the CPU. Measured at `--handoff-partial 28 / 20 / 14`.
- **KV cache** — the stored attention history, one entry per token processed. Its size is set by **context length** (the maximum the model is allowed to remember), not by prompt length.
- **q8_0 vs f16 KV** — two precisions for that cache: 8-bit (half the memory, slightly lossy) and 16-bit. This campaign used q8_0 throughout; the owner rejected q4.
- **Ubatch** — the chunk size prefill is processed in. Larger chunks are more efficient but need more VRAM for compute buffers.
- **`--fit-target`** — stock's flag: the free-VRAM margin, in MiB, that the fitter must leave per device. It is the single knob that decides how many layers stay in VRAM. **Its effect is context-dependent**, because a bigger context means a bigger KV cache, leaving less room for weights.
- **ISO-VRAM** — a comparison in which both sides were measured at the same peak device memory. Without it, a comparison is really "more VRAM versus less VRAM" and proves nothing about speed.
- **WDDM peak** — Windows' per-process device-memory high-water mark, polled once per second during a run. This is the only VRAM figure cited anywhere in this summary.
- **ABBA** — an arm-ordering pattern (A, B, B, A) used so that a slow drift in machine performance cannot masquerade as a difference between the two things being compared.
- **Steady state** — for LM, the mean of passes 2 and 3 of a repeated prefill, discarding the first (cold) pass.
- **Same-session** — both sides measured inside one mutex hold. Cross-session comparisons on this machine drift up to 16%, which is larger than most effects being measured.
- **Mutex / GPU lock** — the named Windows mutex `Global\AI-GPU-EXCLUSIVE-v1` that serializes GPU work on this shared machine.
- **CONTAMINATED** — the session's own flag for an arm where a neighbouring process used more than 6 CPU cores during the timed window, which invalidates that arm.
- **A slot vs device memory** — "resident" always means "held in VRAM"; "host-resident" means held in system RAM.

---

## 4. Pros and cons

Every claim is labeled **measured** (read from a raw artifact), **expected** (arithmetic or
inference from measured quantities), or **unproven** (not established by this campaign).

### Measured — LM prefill, the campaign's headline

| Owner config | LM PP | LM VRAM | stock PP | stock VRAM | stock placement | LM/stock |
| --- | --- | --- | --- | --- | --- | --- |
| doc1: ctx 262,144, 50k prompt | **4,079.2** tok/s | 10.052–10.056 GiB | 2,099.6 tok/s | 10.193 GiB | 41/42 layers, KV 2,720.00 MiB on GPU | **1.943x** |
| doc2: ctx 512,000 `--parallel 2`, 50k | **4,080.4** tok/s | 14.972–14.976 GiB | 2,228.1 tok/s | 15.039 GiB | 41/42 layers, KV 5,312.50 MiB on GPU | **1.831x** |

The two sides are **0.137 GiB** and **0.063 GiB** apart — genuinely ISO-VRAM, which is what makes
these ratios mean anything. Artifacts: `build/stock-levers/ab10.wrapper.log`,
`build/stock-levers/ab5.wrapper.log`, per-arm `*.out` / `*.log` / `*.mem.csv`, LM traces under
`build/llama-cuda/logs/`.

### Measured — the margin decays with prompt length

| Prompt tokens | LM/stock | stock term |
| --- | --- | --- |
| 50,000 | **1.943x** | 2,099.6 tok/s |
| 144,001 | **1.651x** | 1,735.35 tok/s |
| 200,000 | **1.533x** | 1,548.1 tok/s |

Roughly **11% of the advantage lost per doubling** end-to-end (11.2% geometric across the whole
4x span), with 0.789 surviving a 4x increase. The per-interval rates are 7.8% and 3.7%, so the
two intervals are not symmetric; the 11% figure is the end-to-end rate, not either interval.
Artifacts: `build/stock-levers/ab{6,7,13}.wrapper.log`.

**One honest caveat on the middle row.** The 144,001-token stock figure originally rested on a
single request because its companion request came back CPU-contaminated (>6 neighbour cores in 28
of 83 samples, max 31.38). That arm was re-run in its own mutex hold with six requests instead of
two, and **all six came back clean** (0 of 81–82 samples over threshold each; worst neighbour 5.43
cores), giving 1,735.35 tok/s. Side by side: stock moved **+1.30 tok/s (+0.075%)** and the ratio
moved **1.652x → 1.651x**. The single-sample figure was under-determined, not wrong, and the
change is far below the 5% threshold this project ignores by rule. However, that re-run is a
**deliberate cross-session exception**: LM's side of the ratio comes from ab6 and stock's from
ab13, because the contamination forced a separate hold. Every other ratio in this document is
same-session. Artifacts: `build/stock-levers/ab13.plan`, `ab13.wrapper.log`,
`ab13-s6u4k.{out,log,mem.csv,cpu.csv,ident.txt}` (verdict `PASS`, `gpu_pre_run=clean`,
`gpu_post_run=clean`).

### Measured — token generation loses, and does not exist at the owner's contexts

| ctx | Handoff requires | Free VRAM observed | Shortfall | Result |
| --- | --- | --- | --- | --- |
| 262,144 | 24,295,318,016 B | 22,226,665,472 B | **1.93 GiB** | refuses (exit 2) |
| 512,000 | 27,960,222,208 B | 22,226,665,472 B | **5.34 GiB** | refuses before timing |

Both refusals are loud, with no silent fallback. Where a partial handoff *does* fit (ctx 262,144,
128 generated tokens), LM runs **0.640x / 0.687x / 0.741x** stock — 25.074 / 26.934 / 29.045 tok/s
against stock's 39.204 tok/s — while using **1.32x / 1.65x / 1.90x** stock's VRAM
(13.434 / 16.809 / 19.341 GiB against stock's 10.193 GiB). Once the handoff has happened that
phase is **compute-bound**, not transfer-bound: `lmh20` spends 4,019,738 µs of its 4,766,543 µs
wall in compute against 417,160 µs transferring and 1,036 µs waiting, so there is no transfer
headroom left to recover. LM's prefill is unaffected by the handoff (4,072–4,092 tok/s with it).
Artifacts: `build/stock-levers/ab12.wrapper.log`,
`build/llama-cuda/logs/ab12-lmh*-a1-u4096.jsonl`, `ab11cap-lmaut.tail.txt`, `ab5tg.tail.txt`.

### Measured — the most actionable result in the campaign, and it is a stock flag

At the owner's doc2 configuration, changing **one flag and nothing else**:

| `--fit-target` | PP tok/s | TG tok/s | VRAM | Layers resident | KV cache |
| --- | --- | --- | --- | --- | --- |
| **16000** (as the owner recorded it) | 426.00 | 12.14 | 6.972 GiB | **26/42** | 2,125.00 MiB **on CPU** |
| **8000** | **2,227.62** | **42.47** | 15.039 GiB | **41/42** | **5,312.50 MiB on GPU** |

**PP 5.23x and TG 3.50x, for one flag, with no code.** The `16000` arm reproduces the owner's
recorded placement exactly — 26/42 layers and a CUDA0 model buffer of 1,311.95 MiB, matching the
digit in the owner's own notes — so this is the owner's cell, not an approximation of it.
Artifacts: `build/stock-levers/ab5.wrapper.log`, arms `s5u512` and `s5f8k`.

**And the same configuration punishes the obvious "more is better" move.** Raising
`--ubatch-size` from 512 to 8192 buys 4.58x prefill — but collapses residency from 26/42 to
**2/42** layers and pushes the **entire** 5,312.50 MiB KV cache onto the CPU, at **0.44x** token
generation. This is the single most counter-intuitive result in the campaign and it is measured.

Note also that `--fit-target` is **context-dependent**: at ctx 131,072 a target of 16000 is well
fed (41/42 layers), but at ctx 262,144 the same value starves to 30/42 with KV on the CPU. A
tuned value does not transfer between contexts.

### Measured — the flash-attention fix's own contribution

Its share of LM's throughput, measured by toggling it on and off with everything else fixed:
**1.186x** at a 50k prompt (and 1.182x at two other contexts, so it is context-independent at
that length), rising to **1.342x** at 144,001 tokens and **1.377x** at 200,000. It is
**VRAM-neutral** — on and off both peak at 10.056 GiB. Artifacts:
`build/llama-cuda/verdicts/ab9.verdict.json`, traces `build/llama-cuda/logs/ab9-*.jsonl`, ab6/ab7.

### Unproven — carried forward, not smoothed over

- **Why stock degrades more gracefully than LM as prompts lengthen.** The hypothesis is that
  stock copies only the *routed* experts to the GPU, so as a prompt lengthens the set of experts
  it touches saturates toward "everything" and its handicap shrinks. The route union is measured
  at **0.717 / 0.815 / 0.855** at ubatch 512 / 2048 / 4096 on the 50k prompt
  (`build/stock-levers/route-union-report.txt`), but **no route telemetry was collected at
  144k or 200k**, so the saturation is not measured where it matters. Unproven.
- **Token-level fidelity of the partial-handoff generation path.** Two arms with identical flags
  (`lmh20`, `lmh20b`) produced divergent sequences at 128 generated tokens despite a bit-identical
  first token, and the LM driver emits no detokenized text, so no coherence check is possible.
  Unproven.
- **An unexplained VRAM slope.** LM's device memory grows ~7.95–8.09 KiB per *prompt* token and
  ~9.8 KiB per *context* token beyond what its own byte account names. The KV account is **not**
  the cause: LM's preflight reports `KV (q8 @ 262144) = 2,852,126,720 B`, byte-identical to
  stock's independently reported `CUDA0 KV buffer size = 2720.00 MiB`. The slope is measured; its
  cause is unproven.
- **The `alloc_spill` crash.** LM at ubatch 2048 with a large context aborts with
  `input buffer exhausted and spill allocation failed` and exit `0xC0000409`. The suspected cause
  is a 32-byte alignment assumption (`src/llama-layer-streaming.cpp:1173-1202`). It was **seen
  once and never re-exercised** — every LM arm in this campaign ran ubatch 4096 — so this is a
  hypothesis, not a diagnosis.
- **Provenance.** Every session in this campaign ran under a logged `-AllowUnprovenanced` waiver:
  the wrapper captured no binary identity, so the arms are uncertified at session level. Per-arm
  binary hashes were recorded and verified stable. This is a soft gap, stated rather than hidden.

### Known defects — recorded, deliberately not fixed

`third_party/llama.cpp/tools/layer-streaming/main.cpp:3454` contains
`handoff_mode_str = "on";` **inside a `//` comment**. The effect is that `--handoff-partial N`
parses its argument but leaves handoff at `auto`, making the flag **inert on its own**; its own
comment's promise that it "implies an explicit handoff-on request" is unreachable, and the
`--help` text never warns of this. Because `auto` resolves OFF when the model plus KV does not
fit, four arms in session ab11 refused for this reason alone. The workaround is to pass
`--handoff` explicitly. This is a one-token fix and **has not been made**: the vendored tree is a
separate workstream's, it carries another workstream's uncommitted change, and adoption needs a
Planner task contract.

### Everything previously withdrawn stays withdrawn

These were struck earlier in the campaign and must not be revived by this summary:

- "Bench random tokens route to ~69% of expert bytes."
- All `llama-fit-params` standalone-fitter verdicts (the real server fits where the standalone
  fitter declared infeasible).
- The "1.733x vs tuned stock" framing — superseded by the ISO-VRAM figures in §4 above.
- The `n_pad = n_ctx_total` overhead hypothesis (profiling falsified it).
- The u2048 `T_stock` of 1.437 in the route-union report (mixes in a contaminated request; use
  1.419).
- The "6,971 vs about 7,020 actual" fitter figure (unverified; do not cite).
- The `byte_account.kv` under-report suspicion — **withdrawn as incorrect**; the KV figures are
  byte-identical across both engines.

---

## 5. Recommendation

In priority order. The free stock levers come first, because they cost nothing, they are already
measured, and they are worth more than the custom engine on the configuration the owner actually
runs.

**1. Retune stock `--fit-target` at doc2, from 16000 to 8000. (measured, immediate)**
Payoff: prefill **426.00 → 2,227.62 tok/s (5.23x)** and token generation **12.14 → 42.47 tok/s
(3.50x)**, residency 26/42 → 41/42 layers with the full KV cache moved onto the GPU. Cost: one
flag, one restart. This is the highest-value action in the campaign by a wide margin.
Artifacts: `build/stock-levers/ab5.wrapper.log`.

**2. Do NOT raise `--ubatch-size` at doc2. (measured, immediate)**
Payoff of the mistake: 4.58x prefill bought by dropping residency to **2/42** layers, pushing the
entire KV cache to the CPU, and cutting token generation to **0.44x**. The prefill number alone
would look like a win on a dashboard; it is not.

**3. Re-tune `--fit-target` per context, never carry a value across. (measured)**
At ctx 131,072 a target of 16000 is correct; at ctx 262,144 the same value starves the model to
30/42 layers. Read `offloaded N/42 layers` from the server log after every change and confirm the
KV buffer line shows the cache on the GPU.

**4. For prefill-then-generate at 262k and 512k on this machine: stay on stock. (measured)**
This is the blunt one. The charter's own test is that a prefill win which collapses token
generation is not a win. LM cannot generate sustained tokens at either owner context (short
1.93 GiB and 5.34 GiB) and generates at 0.640–0.741x stock with 1.3–1.9x the VRAM where it can.
Item 1 alone closes most of that gap, for free.

**5. Consider LM only if the workload becomes prefill-only — a batch job, an embedding or
classification pass, a scoring sweep — where no generation follows. (measured for PP, unproven
beyond the measured lengths)**
Payoff: **1.943x** at ctx 262,144 / 50k, decaying to **1.533x** at 200k prompt tokens. Before
committing, note the decay is measured at only three points and the mechanism behind it is
unproven, so project the advantage shrinking as prompts lengthen. Also note LM pays ~8 KiB of VRAM
per prompt token, so a long-prompt batch job costs more VRAM than the 50k figures suggest.

**6. Do not adopt the flash-attention fix or the `alloc_spill` fix without a Planner task
contract. (process)**
The flash-attention fix is worth 1.19x–1.38x of prefill and is VRAM-neutral, so it is genuinely
attractive — but it exists only in a side copy (`C:\Development\fafix`), not in the vendored tree,
and moving it requires a task contract plus the semantic-QA answer gate. Drafted as **T-GPLR-65**.
The `alloc_spill` fix is drafted as **T-GPLR-66** and its first step is to *reproduce* the crash,
not to fix it. The `--handoff-partial` defect is drafted as **T-GPLR-67**; the workaround
(pass `--handoff` explicitly) is sufficient until then. Full contracts are in
`docs/plans/generic-prefill-layer-residency-plan.md`.

**7. If more evidence is wanted, buy it in this order.** (judgment)
Re-run the 200,000-token point with route telemetry enabled — that is the single measurement that
would convert the decay mechanism from unproven to explained, and it costs one hold. Then resolve
the partial-handoff token-fidelity divergence, because it is the only open question that would
block any future LM generation work. Everything else on the list is either answered or cheap
enough to re-answer casually.

---

**Bottom line.** Layer-major prefill is a genuine and reproducible prefill win — about 1.9x at
equal VRAM on the 50k prompt — and it is not a win for the workload the owner actually runs,
because token generation is worse everywhere it exists and impossible at both owner contexts. The
free stock `--fit-target` retune is worth 5.23x prefill and 3.50x generation on the owner's own
recorded configuration, costs one flag, and should be done first. All of this is evidence for the
owner's decision, not an approval.

**Nothing in this campaign has been staged, committed or pushed.**