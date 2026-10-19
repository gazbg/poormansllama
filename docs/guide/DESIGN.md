# Design

This document explains why the system is built the way it is, and states the
current implementation and its limits. Usage is in [USAGE.md](USAGE.md). Every
flag is in [PARAMETERS.md](PARAMETERS.md). Experiments that were tried, with
their outcomes, are in [../reference/LEDGER.md](../reference/LEDGER.md).

## 1. The problem

Qwen3.6-35B-A3B is a Mixture-of-Experts model. The file is 19.9 GiB at Q4_K_S.
Most of it is the routed expert weights:

- 256 experts per layer, 8 used per token.
- Three expert banks per layer (gate, up, down), 144 MiB each, so 432 MiB per
  layer and about 17 GiB across 40 layers.

The target card has 24 GiB of VRAM. The owner's rule keeps 16,000 MiB free for
other work. That leaves about 8.3 GiB for the model, its KV cache and the
activations. The banks cannot fit, so they must live on the host, and the GPU
must be fed from them.

Transfers are the constraint. The record measured:

| Path | Rate | Source |
|---|---|---|
| Pinned host-to-device DMA, Gen4 x8 laptop link | about 14.4 GB/s (about 90% of the link) | T-GPLR-29 |
| Pageable host-to-device copy | about 3.0 GB/s | T-Q38TG-00 |
| Break-even for dense hybrid token generation | about 29 GB/s | T-Q38TG-00 |
| PCIe link after about 10 s of low GPU use | downshifts Gen4 to Gen2 to Gen1, needs about 0.25 s of traffic to recover | AGENTS.md |

So the design has two rules. Move each expert bank as few times as possible, and
keep the experts that are used most on the GPU.

## 2. The mechanisms

### 2.1 Hybrid MoE decode (`--hybrid-moe`)

Expert banks stay on the host. A VRAM cache holds the experts used most. A token
runs its routed experts on the GPU if they are resident and on the CPU if not.
The cache is sized to the `-fitt` margin: at startup the server logged
"expert cache 14 -> 54 slots per layer", so 54 of 256 experts per layer are
resident.

`--hybrid-moe-seed` seeds the popularity counts from the routing of the last
prompt tokens, so the cache starts with the experts the prompt used. The record
measured the cache hit rate directly. A hybrid run's speed depends on the
generated text through that hit rate, so the same flags can give different
speeds on different prompts.

Decoding was measured at 3.0x to 3.7x the stock server with the same flags
(G31). Dense models did not get this. The dense hybrid probe measured a
host-to-device rate far below the break-even, and the dense path was closed
(T-Q38TG-00, NO-GO).

### 2.2 Layer-major prefill (`--lm-prefill`)

The stock prefill loops over the prompt in micro-batches. Every micro-batch
copies every expert bank again, so a 50,000-token prompt copies the banks dozens
of times.

Layer-major prefill inverts the loop. For a batch of at least 1,024 tokens, each
layer is processed once for all of its chunks:

- The layer's bank is copied once into a VRAM slot.
- Two slots run as a window: the next bank is copied while the current layer
  computes.
- The activations go through a slab. The slab is split into segments
  (`--lm-prefill-segment`), so a long batch repeats the layer sweep per segment.

The record measured a 2.3x prompt-processing gain over batch-major at 4k
(2026-09-15). The first prediction of 3.5x to 4x was falsified as stated. The
G31 estimate of about 1.35x was also wrong, and the owner was right that more
was available (G32).

Below about 1,000 tokens the bank stream (about 1.3 s) is more than the stock
path costs, so the stock path wins there. That is why the threshold exists.

### 2.3 Prefix cache, checkpoints and canonical mode

Agent workloads send long prompts that repeat most of the previous turn. The
server keeps the state of the prompt as checkpoints (`--ctx-checkpoints`, 32 by
default) so a follow-up turn resumes from the last common prefix.

Two rules make this exact:

- A cache hit must equal a fresh run bit for bit. `--lm-prefill-canonical` puts
  checkpoints only at multiples of the chunk size and resumes a continuation from
  the newest checkpoint before the common prefix. The price is up to one chunk of
  extra prefill per follow-up, about 1 s.
- The default mode fails the hit-versus-fresh checks on the swapped fixtures
  (G32, round 4). The canonical mode passes them. Do not use the default mode when
  the answer must be reproducible.

### 2.4 Fused prefill and idle preparation (`--lm-prefill-fuse`, `--lm-prefill-prearm`)

With `-np 1`, the rest of a prompt can go in one pass rather than being cut at
each checkpoint (`--lm-prefill-fuse`). The pass keeps the recurrent state at the
checkpoint token counts, so the checkpoints are still taken.

When the slots go idle, `--lm-prefill-prearm auto` prepares the next pass: it
makes room, allocates the slot window and the slab, and copies the first two
banks. `auto` prepares only when at least three of the last four requests used the
pass. An unused preparation costs a cache regrow of about 0.4 s at the next decode.

Measured on the final binary (G42), against the run's own baseline (BASE, the
configuration before G42), mean request wall time: fixture A 4.84 s to 3.66 s (0.755x), with idle preparation
3.46 s (0.714x). Small-turn fixture S 2.80 s to 2.53 s (0.902x).

## 3. Invariants

These come from AGENTS.md and must hold in every change.

- Logical layer identity and device-slot identity are separate.
- A slot contains one complete immutable package for one generation. Members
  become ready and retire atomically.
- A layer graph may use a slot only after the matching copy has completed.
- A slot is reused only after every CUDA consumer of its current generation has
  completed.
- KV cache, recurrent state and other session state belong to the logical layer,
  never to a weight slot.
- Layer-major prefill processes chunks for each layer in ascending token order.
- An MoE package holds every routed expert of its layer. Cache residency never
  changes the router's output.
- A missing package, stale slot, failed CUDA event or missing VRAM must fail loudly
  with structured telemetry. It must not silently run another layer's data.
- PP to TG keeps session state and weight-slot assignments unless a task studies
  reset behaviour.

## 4. What is on by default

Nothing new is on by default. Every mechanism above is a flag, and without the
flags the server is meant to behave as the stock llama.cpp server. The record
checks this with flags-off smoke runs (G42: "flags off = s47, smoke A B B A
identical"). This is a hard rule: "Default llama.cpp behavior, including its CPU
path, must remain unchanged when the experimental option is disabled" (AGENTS.md,
scope).

Two defaults are set by owner decisions and are worth knowing:

- `--layer-slots` defaults to 2 (2026-09-15, owner decision).
- The per-layer row-cache retention depth is K=8 (2026-09-18, owner-directed,
  superseding K=4).

## 5. Current implementation state

The current state is the G42 configuration, on fork tree `34c4eee8`. The fork's
26 layer-streaming commits sit under `third_party/llama.cpp/`. Their history and
the ids in the original nested repository are in
[third_party/README.md](../../third_party/README.md).

| Feature | Flag | Status |
|---|---|---|
| Hybrid MoE decode | `--hybrid-moe` | Measured, opt-in |
| Layer-major prefill | `--lm-prefill` | Measured, opt-in |
| Canonical checkpoints | `--lm-prefill-canonical` | Measured, opt-in; passes the hit-versus-fresh checks |
| Fused prefill | `--lm-prefill-fuse` | Measured on `-np 1`, opt-in |
| Idle preparation | `--lm-prefill-prearm auto` | Measured on `-np 1`, opt-in |
| Deterministic decode | `--hybrid-moe-frozen` | Measured; costs decode speed |
| MTP self-speculation | `--mtp` and the research runner | Accepted as opt-in, default off (G30) |
| Expert frequency table at start | `--hybrid-moe-freq FILE` | Optional; default is a uniform prior |

Validated coverage:

- Model: Qwen3.6-35B-A3B MTP (MoE, `qwen35moe` architecture).
- KV: q8_0 only. f16 KV was measured in earlier work and is not the default.
- Slots: `-np 1` for the fused and prearm features. `-np 2` for the layer-major
  prefill, with declines under concurrency.
- Context: 262,144 configured. Stability soaks grew the context to 138k. Needle
  tests reached 100k. Sanitizer runs covered prompts of 3k to 9.5k tokens.
- Hardware: RTX 5090 Laptop GPU, 24 GiB, with the `-fitt 16000` margin.

Not part of the current state: multimodal input, speculative decoding combined
with `--lm-prefill` (untested), `-np` above 2, models other than Qwen3.5/3.6 MoE
on this path, and non-CUDA backends.

## 6. Limits

Measured limits (G32 and G42 entries, unless marked):

1. Batches below about 1,000 tokens are not taken. A pass has a floor of about
   1.3 s, from streaming the banks.
2. Each pass pays about 0.64 s of expert-cache resize (shrink 252 ms, regrow
   391 ms). The idle preparation hides only the shrink. An unused preparation
   costs about 0.4 s.
3. Past about 50k tokens the pass is limited by attention, not by the banks.
4. With `-np 2`, big batches decline under concurrent clients. At 70k-token contexts
   about a fifth of the big batches do not fit under the 16,000 MiB margin.
5. One request still pays about 2 s of fixed costs (cache shrink, regrow and the
   512-token checkpoint chunk). Small agentic turns see almost nothing.
6. The hybrid engine has run-to-run noise. Deterministic mode (`--hybrid-moe-frozen`)
   removes it and costs decode speed.
7. The default mode's hit-versus-fresh expectations fail on the swapped fixtures
   (G32, round 4). No default configuration passed every check in every session,
   including the stock server. The canonical mode passes.
8. `--lm-prefill-bf16` is not validated against f32 activations.
9. Fragmented sequences are not relocated. The ordinary path serves them.
10. The checkpoint chunk is not fused into the pass. About 1.3 s per turn is left on
    the table.
11. The 24 GiB card emulates the 16,000 MiB keep-free rule. It is not a physical
    8 GiB card. WDDM over-commits VRAM, so a ballast process cannot emulate a
    smaller card (G8).
12. Timings depend on power state. On battery the same machine gives different
    timings (2026-10-11).

Older limits, recorded in the 2026-10-12 known-gaps list and not re-measured since
G42. Check them before relying on them:

- The layer-major prefill's VRAM "unaccounted" term grows about 9.8 KiB per
  context token, and its cause is unproven.
- `alloc_spill` crashed at ubatch 2048 at high context (T-GPLR-66). G42 uses
  ubatch 2048 and did not report the crash.
- Partial-handoff TG token fidelity was not proven at 128 tokens (2026-10-12).

## 7. Where to look next

- Experiments and their outcomes: [../reference/LEDGER.md](../reference/LEDGER.md).
- Current record: [../currentstate.md](../currentstate.md) (newest first).
- The layer-residency design document: [../layer-residency-sdd.md](../layer-residency-sdd.md).
- The Qwen3.8 dense streaming design: [../qwen38-layer-streaming-sdd.md](../qwen38-layer-streaming-sdd.md).
- Measured facts that caught earlier mistakes: AGENTS.md, section "Vendored-engine gotchas".
