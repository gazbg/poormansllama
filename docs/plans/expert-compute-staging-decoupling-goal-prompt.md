# Goal prompt — T-GPLR-82: decoupled expert compute and staging

Paste all of this into a fresh agent; it needs no prior context.
Contract: `docs/plans/expert-compute-staging-decoupling.md`. Where they differ, the plan wins.

## Do

A routed expert that misses the device row cache should **compute on the CPU while its copy
to the GPU keeps running**, so the cache stays warm for the next token.

## Where

`src/llama-layer-streaming.cpp` → `load_row_set(int il, const std::vector<int32_t> & ids)` (~9977).
Two returns:

- **~10516** cache-served (`fetch` empty) — stages nothing. **Leave alone.**
- **~10656** staging — copies the complement. **The copy is already not cancelled. Leave alone.**

That is the entire change: in the staging branch, build a CPU sub-op for the missed rows bound
to the **in-flight source rows on the host**, and don't make the consumer wait for them.

## Order

1. **Instrument first** — otherwise a working mechanism is indistinguishable from an inert one.
   Add `experts_host`/`experts_device` per (layer, token) with
   `experts_device + experts_host == experts_total`; `ready_at_consumption` on **TG**
   `rowset_fetch` (the field exists only on PP today); per-token `t_us`; **actual** `host_us`.
   Keep emitting `ffn_dev` — at `:4942` it is computed *after* the build, so it is the only
   independent evidence a host sub-op really ran.
2. **Host sub-op** on the existing add-chain. Flag `--experts-host N`, default `0` = off.
3. **Decouple** — host-bound sub-op for missed rows, no wait; copy keeps running, retires normally.
4. **`--experts-host-budget B`**, µs of host work per token. `0` = today, unbounded = host-first.
5. **Then test:** semantic QA at 50 and 512; host suite; 512-anchor arm forced on; absent-flag arm.
6. **8 GB last**, when an arm is authorised: 2 slots / K=2 (7,961.1 MiB, fits, 230.9 spare) or
   1 slot / **K=3** — *not* K=4, which totals 8,555.7 and refuses by 363.7.

No preliminary probes: the owner directed implement-then-test.

## Acceptance — semantic, not byte-identical

- Semantic QA aggregate **PASS** at 50 and 512; negative control FAILs as designed.
- Top-1 **≥ 99%** vs control over ≥ 512 tokens; every divergence a near-tie (top-2 gap below
  the measured drift). Unexplained divergence fails.
- Lifecycle invariants hold; no hang, no TDR.
- Flag absent → `experts_host == 0`, reported as an observation, never as a pass.

## Traps

- `llama_ls_spec_victim_better` (`:324-333`) makes any `spec`-labelled row the **first eviction
  victim**. Rows you stage are demanded — stamp them **demanded**, or the benefit dies on the
  next LRU sweep. Guard it, don't comment it.
- `bytes_staged` counts slabs across **all** package members (gate/up/down). Bind all three on
  the host side or the byte accounting desynchronises from the copy loop.
- **Never rewire a weight source** — SDD §4 forbids it; the guard asserts it.
- Never cancel or consume a copy; it retires on its own lifecycle.
- `load_row_set` also serves **prefill**. That path must not change — check it explicitly.
- Don't edit `docs/plans/generic-prefill-layer-residency-plan.md`: 211 B from a 480,000 B failure.

## Rules here

- GPU work takes the mutex `Global\AI-GPU-EXCLUSIVE-v1` via `tools/with-gpu-lock.ps1`; host-only
  builds and inspections do not. Never `setsid` a detached arm.
- Verify doc edits by **heading count**, not exit code. Capture exit status **directly**, never
  through a pipe.
- **Do not commit, push or open a PR** unless explicitly asked.
- Gates are **soft**: report limits honestly, and never say "ready", "approved" or "done" on the
  strength of a passing gate.

## On failure

Record the falsification and stop. Do **not** tune placement, budget or `K` to rescue an arm
before recording what happened. `experts_host == 0` across a forced-on arm is a valid outcome —
the mechanism did not fire — and must be reported, not tuned away.
