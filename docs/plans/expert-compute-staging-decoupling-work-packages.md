# T-GPLR-82 work packages, explained for a newcomer

Companion to `docs/plans/expert-compute-staging-decoupling.md` (the contract). This file
explains the same work in plain language for a reader who does not know the codebase and has
never seen the mechanism. Nothing here changes the contract; where the two disagree, the
contract wins.

---

## 1. What the program is trying to do

We are running a **35-billion-parameter mixture-of-experts model** (Qwen3.6-35B-A3B) on a
**laptop GPU with 8 GB of VRAM**, at a **262,144-token context**. All three constraints are
hostile at once.

**The vocabulary you need once:**

- **MoE (mixture of experts)** — instead of one big matrix per layer, the model has a *bank*
  of 256 smaller matrices called *experts*. A small *router* picks the best few of them for
  each token.
- **top-k** — how many experts the router picks per token. For our model, **8 of 256**.
- **Layer** — the model is 40 layers deep; each layer has its own expert bank.
- **Expert row** — the slice of one expert that one token needs. This is the unit everything
  moves around in.
- **PP / TG** — *prefill* (reading the prompt) and *token generation* (writing the answer).
- **Slot** — a reserved region of VRAM that holds expert rows for whichever layers currently
  own it.
- **Staging / DMA** — the hardware copy that moves a row from the checkpoint on disk into a
  slot.
- **KV cache** — the running memory of the conversation. It belongs to the model, not to a slot.

**The arithmetic that forces the whole design.** One expert bank is 452,984,832 bytes — about
**432 MiB**. Forty of them is about **17 GB**, three times the GPU. So the banks cannot live on
the GPU. They stay in the model file on disk, which the program maps into
ordinary computer memory at startup, and individual rows are copied from there into a
slot when the router asks for one.

## 2. How it works today

VRAM is divided into a few **slots**. Each slot holds a compact set of expert rows — only the
ones the router is likely to ask for, shared across several layers.

When the router asks for a row:

- **If the row is already in a slot** → the GPU computes it immediately.
- **If it is not** → the engine starts copying it from computer memory into a slot, **waits for
  that copy to finish**, and then the GPU computes it.

**The problem this task is about.** Those two costs were measured separately:

| doing the row on the CPU | 59.4 µs |
|---|---|
| copying the row in, then computing it on the GPU | ~171 µs |

So for a row that is *not* resident, **standing still and waiting for the copy costs roughly
three times more than simply doing the arithmetic on the CPU**, where the bytes already are. On
top of that, the CPU result cannot be reused later, while a row that arrives stays cached and
the next token often hits it free.

## 3. The idea being tested

**Let the CPU compute a row while its copy to the GPU is still running.** The copy is not
cancelled — it still finishes and still lands in the cache, exactly as before. We just stop
*waiting* for it in that one case.

This is narrower than "run on the CPU". Three rules constrain it:

1. The CPU must read the weights **where the model loader already put them**. Nothing may be
   relocated to make this work — the design document forbids rewiring a weight source, and the
   code now checks this at run time rather than trusting a comment.
2. The router's choices must be **exactly preserved**. The CPU rows must carry the same routing
   weights the GPU would have applied, so the two halves add up to precisely what one
   un-split computation would have produced.
3. Every step must be **measured**, including the step that could quietly eat the win: the CPU
   reading the same memory the copy is streaming in.

One consequence is worth stating plainly, because it is what makes the idea possible at all:
**the CPU already has a copy of every weight**, in ordinary computer memory, because the
model file is mapped there at startup. So a row the CPU computes needs nothing copied to
reach it. Only the GPU needs the copy, and only the GPU is made to wait.

## 4. Where the work actually stands

**Done and compiled:** the CPU half of the computation. It reads the loader's weights, not the
GPU slot's copies; it gets its routing weights from a single shared derivation so the two halves
add up correctly; it is switched off by a flag (`--experts-host N`, default 0 = off) so every
existing command line is untouched; and a guard stops the run dead if it ever finds a weight in
the wrong place.

**Not done, and this is the important part:** the engine-side split has **never been run**.
Compiling is not evidence that the machine can execute it. Two of the three original unknowns
remain open:

- Will the engine's scheduler actually assign this work to the CPU, given it holds rows the GPU
  normally expects?
- What does moving the activations across to the CPU and back cost per layer?

The third original unknown -- does the CPU and the data copy fight each other for memory
bandwidth -- was **measured on 2026-10-07** (N5 result below): when both are forced to run
together the CPU side loses +12-25%, but at the engine's recorded operating point the copy and
the host compute never run at the same time, so the three-times advantage is not being eaten
today.

The package below exists to answer the remaining questions, and to keep the experiment honest
while doing it.

---

## 5. The work packages

### N2 — Keep a copy of the current program

**What it is.** Copying the built program (the `.exe` and every `.dll` beside it) into a dated
folder so the current state survives future rebuilds.

**Why.** The program is rebuilt many times an hour. The moment it is rebuilt, the previous
version is gone — including the version this experiment started from. Rebuilding first and
copying afterwards is the one mistake that cannot be undone, and it has already happened once in
this project.

**How long.** A minute. **Do it first**, because everything else destroys it.

**Done when.** A dated folder holds the executable *and* all its DLLs, verified by file size.

---

### N1 — Run it once and see what actually happens

**What it is.** A small script that runs the model in the row-set mode with the CPU leg switched
on for two of the eight routed rows, plus a checker that reads the resulting log and answers
three questions: did every layer report its rows as split the way we said they would, how much
of the work went to the CPU, and did anything fail.

**Why.** Every remaining unknown is answered by *one run*. Not a benchmark — a first look. If
the scheduler refuses the work, or the split never happens, or the arithmetic comes out wrong,
we learn it now rather than after building three more steps on top of it.

**How it works.** "Split the way we said" means a simple arithmetic identity: for every layer,
rows-on-CPU plus rows-on-GPU must equal rows-asked-for. If any record disagrees, something
computed rows that were never requested, or dropped rows that were.

**Done when.** The checker prints the CPU share of routed rows and confirms the identity holds on
every record — or names the run as a dead end, which is equally worth knowing.

---

### N3 — Move one error check earlier

**What it is.** Today, asking for the CPU leg without also asking for the mode that can run it is
caught only after the model has loaded. Moving that check to the point where the command line is
read makes it fire in milliseconds and makes it testable without touching the GPU.

**Why.** A check that only runs on a real GPU run is a check nobody will ever run.

**Done when.** The bad combination is rejected immediately with a clear message, and that
rejection is covered by an ordinary test.

---

### N4 — Decide the shape the next step must have

**What it is.** A design decision, written down, about how rows get assigned to the CPU when the
decision is made *per row* rather than by a fixed rule.

**Why.** Today the split is fixed: the last two of eight rows always go to the CPU. The real goal
is smarter — send to the CPU only the rows that are *not* already on the GPU. But then the two
halves would be different sizes on every token, and a graph that changes shape forces the engine
to rebuild it, which is far more expensive than the copy we are trying to avoid.

**The way out.** Keep both halves the same width always, and instead of *moving* rows, **switch
rows off**: a row that should have stayed on the GPU is given a weight of zero, so it contributes
nothing to the sum. The width never changes; only the numbers inside it.

**The catch this uncovered.** A switched-off row is still read by the hardware. If the maths
lands on `0 × ∞` — which IEEE arithmetic turns into `not-a-number` — the whole layer's result is
poisoned. So a switched-off row must point at a row known to be well-behaved (expert 0's), never
at an arbitrary id.

**Done when.** The decision is written down and a test proves a switched-off row contributes
exactly zero.

---

### N5 — Do the CPU and the copy fight each other?

**What it is.** A small standalone measurement: time the CPU doing the same expert arithmetic
twice — once alone, once while the GPU is simultaneously copying those same rows in.

**Why.** The entire idea assumes the CPU's work and the incoming copy are close to independent.
If they contend for the same memory bandwidth, the CPU's side gets slower and the three-times
advantage shrinks or disappears. This is the assumption most likely to be quietly wrong, and it
can be tested *before* building anything else on top of it.

**Done when.** A number for "CPU alone" and "CPU while copying", and an honest statement of
whether the advantage survived.

#### N5 result (measured 2026-10-07; host-only, no GPU, no mutex)

**Two answers, both halves stated honestly:**

1. **The mechanism is real (measured):** when the CPU does the expert arithmetic while three
   readers stream the same weight region out of memory, the CPU side gets slower by a **median
   of +11.9% to +14.5%** across the 72-cell grid, **+14% to +25% at the 4-12 thread counts of
   the engine-relevant shape**, worst cell +51.6%. A single compute-bound thread is unaffected
   (~+1%).
2. **The impact at the engine's recorded operating point is about zero (measured):** the copy
   and the host compute never run at the same time there, so nothing is actually eating the
   advantage today (cross-check below).

**How it was measured.** The standalone bench (`tools/expert-cpu-bench.cpp`, built by
`tools/build-expert-cpu-bench.bat`) ran the same 72-cell grid as the 2026-10-02 revalidation
(threads 1,2,4,6,8,12,16,24,32 x experts 1,2,4,8 x tokens 1,8; pool 256, id-stride 8, 200
iters, 20 warmup):

- **4 control runs** with nothing else competing (`build/expert-cpu-bench/n5-control{,2,3,4}-20261007.json`);
- **2 contention arms** (`n5-contend2-` / `n5-contend3-20261007.json`) where three `dd` readers
  looped over the exact layer-0 expert window (byte offsets 949-1385 MiB covering all three
  `blk.0.ffn_*_exps` banks, 437 MiB per pass). The readers moved **149-156 GB per arm**,
  ~3.2 GB/s each by `dd`'s own accounting (`n5-contend{2,3}-dd{1,2,3}.log`); arm shape recorded
  in `n5-contend-arm.sh`.

The readers are a **surrogate for the GPU copy**: they reproduce the host-memory read traffic
an H2D transfer would generate over the same pages, but they are not the DMA engine. A first
contention arm was run as well, but its JSON was overwritten by the replicate (filename reused
before renaming); its numbers survive only in the session (median +11.8%, p90 +25.4%) and are
excluded from every claim below.

**Measured -- contention delta vs the worst of the same-day controls, per cell:**

| | arm 2 | arm 3 |
|---|---|---|
| median cell delta | **+11.9%** | **+14.5%** |
| p90 | +24.6% | +27.4% |
| worst cell | +46.7% | +51.6% |
| cells slower than every control | 69/72 | 72/72 |

The two arms agree cell-by-cell (r = 0.68) and **50/72 cells are >+5% in both**. By shape:

- **TG shape (8 experts, 1 token):** 4t +14.1/+17.2%, 6t +25.4/+22.3%, 8t +18.4/+21.4%,
  12t +16.3/+15.9%, 16t +15.4/+13.4% (arm 2 / arm 3).
- **1 thread:** -0.0/+1.1% -- no bandwidth headroom left to steal.
- **PP shape (8 experts x 8 tokens):** mostly +3-15%, one 12-thread outlier at +45.4/+50.8%.

**Run-to-run honesty.** The four controls reproduce each other (control 4 vs mean of the first
three: +0.4% median across cells), but **today's absolute numbers are worse than the
2026-10-02 revalidation on the byte-identical DLL**: best 8-expert cell 35.07 -> 41.70
us/expert (+19%), 1-thread cell 58.13 -> 144.70 (+149%). A frequency probe held a constant
2006 MHz during a run and the machine had ~4.75 GB free RAM, so this is **machine state, not
code drift** (`ggml-cpu.dll` MD5 `80be38203b0081e3d2390aa0af7a35a2` matches the revalidation
record). All contention deltas use same-day controls, so they stand; **absolute us/expert
figures do not transfer across days**.

**Engine cross-check -- why impact is ~0 today.** Re-verified from
`build/llama-cuda/logs/tg72-threads/tg72-thr16.jsonl`:

- PP staging: 40 copy submits, **18,327 MB total**, t_us 5.51-20.62 s.
- TG copies: 5,120 submits, **bytes = 0** (residency hits), t_us 22.78-27.04 s -- staging ends
  **2.2 s before** the first TG copy event; the phases never overlap.
- Host expert compute (the **681.0 us** layer leg, 4,864 `layer_compute` events with `dev=0`)
  exists only in the TG phase, where byte movement is zero. The h2 timing arm
  (`build/llama-cuda/gplr82/tim-host2-a/trace.jsonl`) agrees: 0 byte-carrying copies during TG.

So the conclusion is deliberately two-part: contention is **present at surrogate strength when
forced** (measured) and **has ~nil impact when the engine actually runs** (measured). Any
future configuration that stages bytes *during* host compute (partial handoff, async-feed
overlap) should budget **+12-25%** on the host expert leg (**expected** from the surrogate,
**unproven** for real DMA). The authoritative test for that case is the GPU contention arm
drafted in the (worktree-only) `docs/plans/qwen36-staging-contention-arm.md`; it needs a CUDA
probe (`expert-cpu-contend-probe.cu`) that does not exist yet, and its pre-registered rule
places a 5-25% delta in the duty-aware-gate band rather than abandonment.

**The "~20x gap" that prompted this task is a units mistake (measured arithmetic).**
682 us / 35.07 us = 19.5 compares a **whole layer** (8 experts + attention + recurrent-state
step + plumbing) against **one expert**. The like-for-like frame already recorded in
`docs/currentstate.md` (2026-10-02 entries):

- device base per layer = **203.7 us**; host layer = 203.7 + 8 x host-expert.
- in-situ host expert = **58.0 us** (+463.8 us/layer penalty / 8) -> predicted host layer
  203.7 + 8 x 58.0 = **667.7 us** vs measured **674.6-681.0 us** (~1-2% residual).
- bench optimum 35.07 us/expert (12 threads, 2026-10-02) vs 58.0 in situ = **1.65x** (the file
  records 1.59x against the then-current 36.5 figure), worth 6,527 us/token = 19.3% of the
  wall -- the size of the prize, not a promised gain.
- 58.0 us is within 0.9% of the bench's **1-thread** cell (58.13): the engine's host leg
  behaves single-threaded regardless of engine thread settings (thread lever refuted, 1.8%
  spread across 1/4/12/16 threads), so the residual ~1.65x is **parallelism the host dispatch
  path does not take, plus ggml scheduler overhead** (suspect, **unproven** -- no arm yet
  isolates it).

**Honest per-expert overhead: ~1.6-1.7x, not 20x.** The 19.5x figure divides incompatible
units and must never be quoted.

**2026-10-09 revalidation (harness liveness; not a new claim).** The unchanged instrument
(`expert-cpu-bench.exe`; local `ggml-cpu.dll` MD5 `80be38203b0081e3d2390aa0af7a35a2`, matching
the revalidation record) re-ran the full 72-cell grid host-only — nvidia-smi clean before/after,
**exit 0**, fresh files `build/expert-cpu-bench/n5-recheck-20261009.{json,log,err}`. Best
8-expert/1-token cell **40.07 us/expert @12t**, 1-thread cell **142.62** — within ~2-3% of the
2026-10-07 controls and again far from the 2026-10-02 absolutes (35.07 / 58.13): a third-day
confirmation of the "absolute us/expert figures do not transfer across days" rule above. No
number or claim in this section is changed by it; the 19.5x unit-mismatch verdict stands.

---

### N6 — Send the right number of rows

**What it is.** Turning the measured numbers into an automatic rule: given how long a CPU row
takes and how long a copy takes, decide per row whether the CPU is worth it.

**Why.** After N1 and N5 the constants stop being guesses. Until then this rule would be built on
assumptions, and a rule built on assumptions cannot be trusted.

**Depends on.** N1 and N5. **Not worth building before them.**

---

## 6. Order

**N2, then N1.** N2 costs a minute and protects something unrecoverable. N1 answers most of the
open questions in one run and produces the numbers N5 and N6 need.

Then N5 (the bandwidth assumption), then N4 (the shape decision) before any further code is
written for the per-row version, then N6.

## 7. What "good" means here

This is an experiment, not a product. The bar is deliberately modest:

- It runs, or we learn precisely why it cannot.
- It says the same thing as before when switched off.
- It is faster, or it is not — measured either way.
- The existing quality benchmark still passes at the end.

We are not chasing byte-for-byte agreement with the unmodified engine, and we are not rebuilding
the quality evidence. The quality benchmark already in place is the check; if answers change at
the end, that is when we look.
