# T-GPLR-71 Handover — what we were trying to do, and what it actually measured

**Date:** 2026-10-02 · **Branch:** `research/layer-residency` · **Status: measured, not accepted.**
Nothing here is committed. Nothing here is "ready" or "approved" — every gate in this repo is advisory, and
final acceptance is the owner's decision.

**Read this file if you want the short version.** The full chronological record is
[`docs/currentstate.md`](currentstate.md) (~393 KB); you do not need it unless you are auditing a claim.

---

## 1. The WHY

**The goal.** This branch runs a 35B mixture-of-experts model on a machine with a **small GPU**
(8 GB class) while supporting **262,144 tokens** of context. Weights are too big to fit, so the branch's
core idea — *layer residency* — stages **one layer's weights at a time** through a small fixed window of
VRAM "slots" instead of holding the whole model on the card. KV cache and recurrent state stay owned by
the session, not by a slot.

**The charter, in priority order:** (1) maximise **PP** (prefill) speed, (2) minimise **VRAM**, (3) make
sure PP holds when VRAM is crowded. **TG** (token generation) must be reported alongside PP always,
because PP and TG are different problems.

**The specific problem T-GPLR-71 attacked.** The hypothesis, formed before any fix:

> During token generation, the decode attention scanned the **whole allocated KV cache** instead of the
> **live token count**. A run holding 50,000 real tokens still paid for a 262,144-slot cache. That scan
> was sized at **9,000.4 µs/token — about 20.8% of the TG wall.** Sizing the attention view from the
> live position instead should recover it **without spending PP and without spending VRAM**, because
> the view gets *narrower*, never wider.

**Why it mattered.** Charter clause 3 says PP must hold when VRAM is crowded. A TG leg that degrades with
the *allocated* context is exactly that failure, and it was the one place paying for capacity nobody used.

**The root cause was a single constant.** Upstream llama.cpp already sizes the view from the live
position. This branch's runner overrode it: `tools/layer-streaming/main.cpp:4470` sets
`cparams.n_pad = n_ctx_total`, forcing `n_kv = 262144` for every graph. One line.

**The trap this created — and why the task was not a one-liner.** If the view tracks the live count and
generation keeps adding tokens, the view eventually becomes **too narrow**. A too-narrow view does not
crash — it silently reads a truncated history, a quietly *wrong* answer. So a **coverage rule** was
added: before each generated token, check the built view against the live position; if it does not cover
it — **or if the engine cannot tell** — rebuild. The fail-safe is deliberate: **unknown counts as
not-covered**, because guessing safe costs a rebuild and guessing fast costs correctness.

---

## 2. How it works, in plain language

**Glossary — every term, defined at first use.**

| term | meaning |
|---|---|
| **PP / prefill** | Consuming the whole prompt at once, every token in large parallel batches. Measured in tok/s. |
| **TG / token generation** | Producing the answer one token at a time. Each step is a *single* token read against the entire accumulated history — far slower per token, and the leg this task targeted. |
| **KV cache** | The per-token keys and values each layer keeps so later tokens can attend to earlier ones. Grows with context. |
| **n_kv** | How many KV positions the attention math actually walks over. |
| **n_pad** | The rounding step: a view width is padded up to a multiple (256 here) so GPU kernels see a shape they like. |
| **view / view width** | The slice of the KV cache one prebuilt graph reads. The whole lever of this task. |
| **graph set** | The prebuilt CUDA execution plan, compiled for specific shapes. Costs milliseconds to build; must be rebuilt if a shape changes. |
| **slot** | One bounded VRAM staging place where a complete layer's weights live during prefill. |
| **layer residency** | Staging one layer at a time through a small slot window rather than holding the model on the GPU. |
| **handoff** | The PP→TG transition, where session state and live slot assignments are retained. |
| **VRAM** | On-board GPU memory — the scarce resource being rationed. |
| **µs/token** | Microseconds per generated token; the unit TG speed is really made of. |
| **tok/s** | Tokens per second — the headline throughput number. |
| **mutex hold** | An exclusive named lock on the single shared GPU, held for a measurement so no other project corrupts it. Every timing number here was taken under one. |
| **n=1 per cell** | One run per experimental cell. A cell measured once is a hypothesis, not a result — see §6. |

**In plain language.** Normally the decode step is handed a 262,144-slot menu and told to read all of
it, even though only 50,000 slots hold anything. The fix hands it a menu sized to what is actually on
it — 50,176 slots, rounded to the 256 boundary. Same arithmetic over the tokens that exist, far less
work over the empty ones, and **nothing has to be added to VRAM** because the menu is smaller.

That creates the under-read hazard described above, and the coverage rule is the answer: check before
each token, rebuild when the view no longer covers the live position, and treat *unknown* as unsafe.

**How the work was validated.** The guard's condition was extracted into a named predicate and the
build's width stamp into a named recorder, so a **host test drives the production symbols** rather than
a copy — a test that re-spelled the guard's `if` would prove nothing about the engine.

---

## 3. Performance results

All figures below were **re-derived from the run artefacts on disk**, not copied forward from earlier
notes. Every claim is labelled **measured** / **expected** / **unproven**.

### 3.1 The headline: token generation, same-hold A/B at ctx 262,144

| quantity | control | candidate | change | label |
|---|---|---|---|---|
| **TG throughput** | 23.407 tok/s | **29.609 tok/s** | **+26.5%** | measured |
| TG compute per token | 39,281.8 µs | 30,255.6 µs | **9,026.2 µs recovered** | measured |
| TG wall per token | 42,722.8 µs | 33,773.4 µs | 8,949.4 µs recovered | measured |
| **PP throughput** (steady pass) | 3,089.0 tok/s | 3,096.7 tok/s | **+0.25%** | measured |
| Graph rebuilds during PP | 0 | 0 | **no cost** | measured |
| **VRAM peak** (poller) | 8,160,768,000 B | 8,160,768,000 B | **byte-identical** | measured |

Also **+28.2%** against the 23.096 tok/s pre-experiment baseline. The measured recovery (9,026.2 µs)
lands within 0.3% of the **9,000.4 µs predicted** before the fix — the estimate was a prediction, not a
measurement, but it is the strongest single sign the diagnosed cause was the real one.

**Charter reading: PP — priority 1 — neither improved nor regressed. The win is entirely in TG.**

### 3.2 Flatness across a 4.00× context sweep — the property that matters most

The candidate's view is capped at 50,176 regardless of how much context is allocated, so TG cost should
not scale with `--ctx-size`. The control's view is the whole allocation, so it should.

| session | control | candidate |
|---|---|---|
| ab20/ab21 (n=1 per cell) | **0.827×** (17.3% degradation) | **1.000×** |
| `dip` replication (n=2, interleaved) | **0.790×** tok/s (1.289× compute) | **1.001×** tok/s (0.998× compute) |

Across a 4.00× context span the candidate holds TG flat to within **0.8%**, while the control loses ~21%.
All three candidate contexts land in a 3.54% band, which is the ordinary replicate spread.

### 3.3 VRAM

**Byte-identical between control and candidate at all three contexts, n=2, 6/6 arms exact**
(5,284,761,600 / 6,197,293,056 / 8,160,768,000 B). At the 262,144 operating point that is
**7,782.71 MiB = 7.600 GiB**.

Two corrections the owner should know about, both found by re-deriving:
- An earlier figure of **7,778.4 MiB** appears in **no artefact** and was wrong; it is withdrawn.
- An earlier claim that VRAM differed by **+270,336 B at ctx 131,072** was an **ab21 transient** — it
  does not reproduce.

### 3.4 The coverage guard actually fires (this was the open question)

| observable | predicted | measured |
|---|---|---|
| `tg_view_rebuild` events | 1 | **1** |
| `built_n_kv` → `required_n_kv` | 50,176 → 50,432 | **50,176 → 50,432** |
| firing at generated token | 177 | **177** |
| `view_rebuilds_total` (PP/PP/TG) | 0, 0, 1 | **0, 0, 1** |
| `kv_view_built` = `kv_view_live` after | 50,432 | **50,432** |
| graph builds (PP/PP/TG) | — | **0, 0, 2** |
| throughput cost of the rebuild | ~none | **29.513** tok/s at tg=256 vs 29.609 at tg=128 |
| VRAM cost | none | **unchanged, byte-identical** |

**The rebuilt set had been built at 50,176 and was genuinely too narrow at token 177 — the guard caught a
real under-read that would otherwise have silently dropped tokens 50,177–50,431 from the decode view.**
That is the silent-wrong-answer class the whole mechanism exists to prevent, now observed rather than
argued.

**Correction worth flagging:** ab20/ab21 read `view_rebuilds_total: 0` and this branch concluded the guard
was "dead in this shape." **That was wrong.** `--tg-tokens 128` stopped **49 tokens short** of the first
pad boundary. `view_rebuilds_total: 0` below ~177 generated tokens is the guard correctly declining to
act, not evidence of unreachability.

### 3.5 Answers and routing are unchanged

| check | result | label |
|---|---|---|
| Semantic QA, 512 tokens, default path | **5/5 needles**, `aggregate=PASS` | measured |
| Semantic QA, 512 tokens, `--kv-graph-pad 256` | **5/5 needles**, `aggregate=PASS` | measured |
| Negative control | `aggregate=FAIL` as designed — the checker can fail | measured |
| Telemetry check on both QA runs | `check-telemetry: OK (30,547 events, 3 slots)` | measured |
| Route parity | **clean on all 19 arms**: 480 events, 240 distinct `(layer, chunk)`, each exactly twice, 0 hash mismatches | measured |

### 3.6 Host-side proof

| check | result | label |
|---|---|---|
| Host suite | **77/77**, exit 0 | measured |
| Narrow ctest + new test | **2/2** and **1/1** | measured |
| Mutation 1 — fail-safe reverted to "unknown is safe" | **5 assertions fail**, ctest `***Failed`, host suite `exited with code 8` | measured |
| Mutation 2 — `cur_build_n_kv` reset deleted | **3 assertions fail**, same propagation | measured |

The two mutations are the test's justification: they prove the test fails when the two decisions that
matter are broken, and that the failure reaches the gate an engineer actually runs.

---

## 4. What is NOT proven — read this before trusting any number

- **MTP is never exercised.** `--mtp` has not been run. Two un-instrumented build sites
  (`llama-context.cpp:1392`, `:3845`) remain unmeasured — this is the live risk behind the
  convention-threaded `cur_build_n_kv` handshake.
- **No arm attests the binary it ran.** The wrapper's provenance gate **REFUSES**
  (`binary_state=not-supplied`) on every session including the latest; hashes recorded in the logs are
  **directory-level**, not the loaded process. Every timing number here is tied to a *named directory*,
  not a proven executable.
- **One card, one hold per result.** No multi-GPU, no second machine, no cross-session replication of
  the headline A/B.
- **Nothing below ~1% is resolvable.** At n=2 the replicate spread is 0.15–2.92%; read the +0.25% PP
  delta and any small TG delta against that floor.
- **Token ids are not reproducible even between two identical control runs** (64/128 positions agree
  across two holds) — pre-existing nondeterminism, which is why QA grades *answers*, never ids.
- **Long QA rungs unrun**: 4k, 24k, 50k, 100k, 200k are owner-invoked one at a time.
- **The 9,000.4 µs/token figure was a prediction.** The biggest remaining TG term — host layer
  execution — was **not re-derived this pass, but has since been**: it is **52.2% of *this* run's wall**,
  not 42.6%. That older 42.6% was measured on control run `ab19-ctl38x`, whose wall was 28% longer; the
  `--kv-graph-pad` fix shrank the denominator without touching the host penalty. The 4.0% `setup_us`
  figure seen alongside it is a **different, disjoint term** (host graph setup, outside `compute_us`),
  not a contradiction. See the 42.6%-vs-4.0% reconciliation entry at the top of
  [`docs/currentstate.md`](currentstate.md).

---

## 5. Pros and cons

**Pros (measured).** Large TG win at the small-card operating point with PP, VRAM and answers unchanged;
TG cost now flat in context where the control degrades ~21%; the safety guard is demonstrated on
hardware and mutation-proven host-side; routing is bit-identical across 19 arms; VRAM byte-identical.

**Cons / risks.** The win is **TG-only** and the charter's priority 1 is PP. Provenance is uncertified
for every measurement. The host-execution term (**52.2%** of this run's TG wall) is larger than the term
this task recovered and is untouched; it is closed on the capacity side (a layer needs 498.0 MiB against
409.3 MiB of headroom), so the only remaining lever is host execution *speed*, not placement. Two structural debts are filed, not fixed: the **~9,900-line
`llama_layer_streaming.cpp` monolith** (its size is *why* a divide-by-zero hid in it through four
reading passes) and the **convention-threaded `cur_build_n_kv` handshake** (only 2 of 5 `model.build_graph`
sites stamp it; the fail-safe mitigates but does not eliminate this). A **core upstream interface,
`llama_memory_i::set_view_pad`, was widened for one experimental flag** — held rather than replaced
because **no test covers it**, so equivalence could not be proven.

---

## 6. The lesson this branch paid for

An earlier sweep ran its arms in **blocks, one per context, n=1**. That produced a convincing-looking
**6.5% dip at ctx 131,072** which three separate checks failed to explain. A replicated, **interleaved**
hold (12 arms, n=2 per cell) showed the dip **does not reproduce** — ctx 131,072 is the *fastest*
candidate cell — and that in the original session **each engine's outlier arm sat at a different
context**, which no property of a context can do. The +270,336 B VRAM delta at that same cell vanished
too.

**Any cell measured once is a hypothesis.** The dip sat only ~1.5× above the replicate spread later
measured. Interleaving and replicating is not ceremony; it is what separates a result from a coincidence.

---

## 7. Recommendation

**Adopt the mechanism. Keep experimenting on the open items. Do not read any gate as authorisation.**

1. **Accept the TG view-sizing change** on this evidence — the multi-instrument case is coherent.
2. **Investigate the 42.6% host-execution share of the TG wall.** It is now the largest term and this
   task's lever does not touch it.
3. **Close the provenance gap** so future arms can attest the running binary.
4. **Make interleaving and n≥2 the default** in the sweep harness, so a transient cannot again read as a
   context effect.
5. **Decide** whether to fund the small test that would make the `set_view_pad` decision provable, or
   accept the widened interface.
6. **Treat the monolith split and the `cur_build_n_kv` handshake as real but deferred** — a redesign is
   an explicit non-goal of this mission.

---

## 8. Where the evidence lives

| what | path |
|---|---|
| Canonical chronological record (start here if auditing) | [`docs/currentstate.md`](currentstate.md) |
| This handover | `docs/t-gplr71-handover-2026-10-02.md` |
| Design + task contracts (near its size cap) | `docs/plans/generic-prefill-layer-residency-plan.md` |
| Headline A/B traces (ab20, ctx 262,144) | `build/llama-cuda/logs/ab20-{ctl38x,g71x}-a1-u1024.jsonl` |
| Context sweep (ab21, blocked layout) | `build/llama-cuda/logs/ab21-*.jsonl` |
| Guard arm (tg=256) | `build/llama-cuda/logs/ab22-g71t256-a1-u1024.jsonl` |
| Replication (12 interleaved arms) | `build/llama-cuda/logs/dip-*.jsonl` + `build/stock-levers/dip.plan` |
| VRAM poller records | `build/llama-cuda/logs/*.mem.csv` (`max dedicated_bytes`) |
| CPU/machine-load records | `build/llama-cuda/logs/*.cpu.csv` |
| Arm + session verdicts | `build/llama-cuda/verdicts/` |
| Host regression test (39 assertions) | `third_party/llama.cpp/tests/test-layer-streaming-tg-view.cpp` |
| The one-line root cause | `third_party/llama.cpp/tools/layer-streaming/main.cpp:4470` |
| Coverage guard predicate | `third_party/llama.cpp/src/llama-layer-streaming.h:763` |
| Width stamp + reset | `third_party/llama.cpp/src/llama-layer-streaming.cpp:3869` |
| Widened upstream interface | `third_party/llama.cpp/src/llama-memory.h:140` |

**Gates re-run at the time of writing:** `resplit --verify` PASS · `check-doc-sizes` PASS ·
`git diff --check` clean in both repos · nothing committed.