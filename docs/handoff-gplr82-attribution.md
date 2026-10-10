# Handoff: T-GPLR-82 expert-split attribution (2026-10-10)

You are picking up an attribution investigation, not a feature. The mechanism (an
opt-in CUDA layer-residency split of the MoE expert FFN: a 6-wide head leg plus a
2-wide tail leg instead of one 8-wide op) is **built and measurable, and its
acceptance criterion AC1 is FAILING**. Everything below is evidence about *why*,
so the next change is aimed rather than guessed.

Read this whole file before touching anything. It ends with the exact commands to
run and the traps that cost this session time.

## 0. Read me first

- Governing docs: [AGENTS.md](../AGENTS.md) (repo), `third_party/llama.cpp/AGENTS.md`
  (vendored; ASCII-only comments, no PR/push/comment actions ever).
- The task contract is `docs/plans/expert-compute-staging-decoupling.md`
  (T-GPLR-82, `status: ready`, AC1..AC7).
- Undefined acronyms used here: **PP** = prompt processing (prefill), **TG** = token
  generation (decode), **slot** = one bounded VRAM window holding one layer's staged
  weight package, **row/position** = one of the router's `top_k` experts for one
  token, **head leg** = the leg that takes the first `top_k - host_rows` positions,
  **tail leg** = the leg that takes the remaining `host_rows` (the "split"), **host
  share** = `experts_host / experts_total` as the trace reports it, **decoupling (S3)**
  = admitting a row for compute without waiting for its copy to land, **AC1** = the
  task's correctness gate (semantic-QA PASS plus top-1 agreement with the recorded
  control).

## 1. The question chain and what is established

The owner asked three questions in sequence. All three now have measured answers; two
of the three answers are *negative*, and that is the point of this handoff.

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Why does the split move step-0 logits, and is it a real defect or window noise? | Real: control 16.194481 -> A2 16.167877 (0.16%). Not noise: the split's own effect. | `build/llama-cuda/gplr82-a2dev3/`, recorded control in the runner |
| 2 | Is the CPU tail leg the cause? (Run the same 6/2 partition entirely on the device.) | **No.** With zero CPU rows 81% of the shift survives (device tail = 16.172943). Separately the CPU tail leg *is* a second, finite defect (see #3). | `gplr82-a2dev3` session log + verdict |
| 3 | Why do 506/512 logits go non-finite with the CPU tail but stay finite with a device tail, and is that one root cause? | **Two root causes.** The non-finite cascade belongs to the S3 copy-wait decoupling (budget-dependent admission), not to the CPU leg. The CPU leg's own effect is finite but answer-destroying. | `build/llama-cuda/gplr82-s3off/` verdict (the 2x2) |
| 4 | (this session) Do the same expert rows compute differently in a 2-wide vs 8-wide `MUL_MAT_ID` batch, which would explain the 1.3e-3 shift that survives with no CPU compute? | **No. Refuted by measurement: bit-identical on CPU and on CUDA0, 42/42 comparisons.** | `build/llama-cuda/op-rows-{cuda-t1,cuda-t8,cuda-q6,cpu}.json`, checker `tools/check-op-rows-probe.py` |

Numbers, one place, for reference (512-token anchor, `--ubatch 512
--residency-tg-retain 2`, 512 TG tokens, exe `a9ddf1699801b235`):

| arm | configuration | step-0 logit | non-finite | notes |
|---|---|---|---|---|
| C0 | `--experts-host 0` | **16.194481** | 0/512 | the recorded control, reproduced exactly |
| A2 | `--experts-host 2 --experts-host-budget inf` (shipped split, CPU tail) | 16.167877 | 506/512 | 564 skipped waits |
| A2dev | same + `--experts-host-leg device` | **16.172943** | 0/512 | **zero CPU rows**; 81% of the shift survives |
| B1 | `--experts-host 2 --experts-host-budget 0.001` (decoupling OFF, `s3_cap=0`) | 16.214243 | **0/512** | CPU tail leg runs fillers on all 20,480 records; QA FAIL (0/5 needles) |
| B2 | `--experts-host 2 --experts-host-budget 100` (`s3_cap=2`, decoupling ON) | 16.167877 | **506/512** | reproduces A2 exactly |

## 2. The latest idea implemented this session: the per-op row-count probe

**Hypothesis.** A quantized GEMM can pick a different kernel/tiling for a different
batch width. If an expert's rows are computed in a 6-wide head and a 2-wide tail
instead of one 8-wide op, the kernel could return slightly different values for the
*same* rows, which would explain the split's 0.16% step-0 shift without any CPU leg
involved. It was the cheapest explanation left and it was testable per-op, with no
model load.

**What was built** (new, uncommitted):

- `third_party/llama.cpp/tools/layer-streaming/op-rows-probe.cpp` - a standalone
  ggml probe. No llama context and no model: it synthesizes the model's real bank
  geometry (k=2048, m=512, `n_expert`=256, `top_k`=8, Q4_K / Q6_K) and runs
  `MUL_MAT_ID` over that one bank with
  - the 8-wide whole op, twice (noise floor),
  - the same rows in a 2-wide batch, a 1-wide batch and a 6-wide batch,
  - the weighted 6+2 split against the weighted 8-wide whole,
  - both legs inside ONE graph against the same legs run alone.
  Every comparison reports `max_abs`, `max_rel`, `med_rel`, `bit_equal` and `n_diff`
  per column, and the whole run writes JSON (`--json`).
- Target `op-rows-probe` + CTest `layer-streaming-op-rows-cpu` in
  `third_party/llama.cpp/tools/layer-streaming/CMakeLists.txt`.
- `tools/check-op-rows-probe.py` - classifies the JSON against pre-registered bands
  (`exact`/`negligible` -> REFUTED, `small` -> weak, `material`/`large` -> the cause).
  Has `--selftest` (5 bands, a failed shape, an unreadable file) and is registered in
  the host suite in `tools/run-tests.ps1`.

**The result - the hypothesis is dead.** Worst `max_rel` over all 42 comparisons in
four runs: **0.0, with `bit_equal=yes` and `n_diff=0` in every one**:

- CUDA0, tokens=1, Q4_K, both shapes (`op-rows-cuda-t1.json`)
- CUDA0, tokens=8, Q4_K, both shapes (`op-rows-cuda-t8.json`)
- CUDA0, tokens=1, Q6_K, down shape (`op-rows-cuda-q6.json`)
- CPU, tokens=1, Q4_K, both shapes (`op-rows-cpu.json`)

So on this geometry a row's value is **bit-identical** whether it is computed alone, in
a 2-wide batch, in a 6-wide batch, in the 8-wide whole, in the same graph as its sibling
leg, or on the other backend (CPU is bit-identical to itself; the CUDA numbers are
internally consistent, the two backends are not compared elementwise). The split's
remaining shift is therefore **not** a batched-GEMM row-count artifact. Do not re-run
this to "confirm"; the JSON is on disk and the checker re-classifies it.

**What this refutes and what it does not.** It refutes the row-count kernel hypothesis
for this shape and quantization. It does NOT cover: the model's own *aggregation order*
(the probe sums its two weighted partials in the same order in both cases, so it is
insensitive to reassociation), realistic activation data (the bank is synthetic, drawn
uniform in [-0.05, 0.05] then quantized; activations uniform in [-1, 1]), the real router
weight derivation (`w_full` normalization lives in the graph, not the probe), and the
GLU/down chain as a whole.

**Where that leaves the search.** The GEMM and the weighted-sum structure are exact, so
the 0.16% must come from *what data each leg is given* or from *where a leg runs* -
candidates, in the order I would test them:

1. **Weight/row pairing across the two legs.** The split views `w_full` (the router
   weights, `[1, n_expert_used, n_tokens]`) at an offset while the tail leg's ids come
   from a separate INPUT (`hsel_in`) or from a view of `sel_in` at a *different* offset
   (see `build_moe_ffn` in `third_party/llama.cpp/src/models/qwen35moe.cpp`, the
   `tail_device` branch). A one-position mispairing between an expert and its router
   weight is exactly the size of error observed and would also explain why the effect
   survives with no CPU leg at all. Cheapest test: extend the probe (or add a
   host-side check over a recorded trace) to assert, per layer and token, that the union
   of the two legs' id values equals the router's ids AND that each id's weight is the
   one the graph multiplies it by. The trace already carries `route` telemetry
   (`--route-telemetry 1`) and the plan/`hsel` membership.
2. **The tail leg's activation buffer.** The host leg's activations come from its own
   companion input, filled by the executor; the device leg reads the graph's FFN input.
   A dtype or offset difference there (f32 vs bf16 slice, `ubatch` stride) would move
   values by ~1e-3 with no kernel involved.
3. **The GLU + down chain, split per leg.** Elementwise, so it should be exact - but it
   has never been measured per-op the way the first matmul now has.

## 3. What went wrong along the way, and what was fixed

Every one of these cost a build or a run; the fix is in the tree unless stated.

1. **A suffix view of an I32 `[top_k, T]` tensor is illegal in ggml.** A column-suffix
   view carrying the parent's stride bounds-checks as
   `data_size + view_offs <= ggml_nbytes(view_src)` and overruns
   (`GGML_ASSERT` at `ggml.c:1787`, exit `0xC0000409`) while building the persistent
   graphs. Fix: the device-tail diagnostic keeps the TAIL at offset 0 and puts the HEAD
   at `host_rows * len`, so the wider view ends exactly at the parent's last byte
   (`qwen35moe.cpp`, `id_off`/`w_off`).
2. **The S3 decoupling predicate ignored which graph was built.** With budget `inf` it
   skipped waits that no host leg existed to absorb, and a device leg then read unstaged
   rows: 512/512 non-finite with `ffn_dev = 1` and a zero host ledger. Fix: the predicate
   now requires a real host leg (`!experts_host_leg_device`). Session `a2dev2` (dll
   `eb878312`) is the pre-fix run and is kept as evidence; `a2dev3` (dll `c11ac969`) is
   the fixed one. Note `a2dev` (dll `f8e54ffd`) crashed in the graph build and is NOT a
   data point.
3. **My own first two attempts at the diagnostic were the bugs** (the illegal view above,
   and the predicate). Both were found by reading the failure, not by guessing.
4. **The runner's QA subprocess crashes printing non-ASCII text** (`UnicodeEncodeError`,
   cp1252), so `positive_pass=False` and `negative_fails_as_designed=False` recorded for
   B1/B2 are crash artifacts, not grades. Re-run the checker with
   `PYTHONIOENCODING=utf-8` for the real verdicts (C0 PASS; B1/B2 FAIL on all five
   needles). The crash is still in `run-gplr82-s3off.py` - fixing it (force UTF-8 in
   `subprocess.run`) is a small, welcome cleanup.
5. **The loud-failure guard cannot see a NaN that reaches the CPU.** The host leg's bank
   is Q4_K; the CPU quantized matmul quantizes activations to q8_K first, and
   `quantize_row_q8_K_ref` (`third_party/llama.cpp/ggml/src/ggml-quants.c:2768-2785`)
   tests `ax = fabsf(x[j]); if (ax > amax)`, which is FALSE for NaN, so `amax` stays 0 and
   the block is written `d = 0` with zeroed quants. A NaN activation block becomes a ZERO
   contribution and the per-row output stays finite, so `scan_host_expert_outputs`
   (`src/llama-layer-streaming.cpp:4570`) reports "all finite" even on the run whose
   logits are NaN at 506/512 steps. **This guard gap is still open and is the single most
   valuable follow-up**: check the inputs, not just the leg's own output rows.
6. **The probe initialized CUDA even for a CPU-only run.** `ggml_backend_load_all()`
   enumerates devices and the CUDA backend is linked into `ggml.dll`, so the CUDA driver
   initializes (device enumeration only, no VRAM, no compute - measured). Partial fix
   applied: the probe looks the device up first and only calls `load_all()` if the name
   is unknown, which keeps the binary from calling it explicitly. The enumeration still
   happens because of static registration, so **treat even the "CPU" CTest as touching
   CUDA** and do not run it while another project holds the GPU mutex.
7. **The GPU wrapper refuses this tree's arms as UNCERTIFIED.** The nested worktree is
   dirty (uncommitted experimental edits), so the provenance gate reports "the binary
   PREDATES its sources" and rewrites the verdict to UNCERTIFIED (exit 10). `gpu_pre`/
   `gpu_post` were clean; the evidence is on disk; the escape is `-AllowUnprovenanced`
   for non-arm probes like this one. Committing the tree (only with owner authorization)
   is what actually clears it.
8. **The file tools do not write where you think.** `write_file`/`str_replace`/
   `read_files` in this session resolved to a scratch worktree
   (`.freebuff/worktrees/b947db99-.../`), NOT the live tree - `read_files` of
   `docs/currentstate.md` returned a stale revision. Verify every read/write with a
   shell command (`ls`, `grep`, `diff`) and copy new files into the live tree
   explicitly, or just edit with Python/sed through the shell. This trap silently
   "loses" edits.
9. **Do not `setsid` a detached arm; dot-source `arm-verdict.ps1` and `Clear-ArmVerdict`
   before `Start-Process`.** Both are recorded in the repo guide because they produced
   false verdicts earlier.

## 4. Known-failing checks (pre-existing, not caused by this work)

- `tools/run-tests.ps1 -Suite host` exits 1 on the **doc-size guard**:
  `docs/plans/generic-prefill-layer-residency-plan.md` is 485,036 B (101% of the FAIL
  band). It aborts the suite before its later checks. This has been failing since before
  this campaign; the plan needs splitting, or the guard needs an explicit owner decision.
- The CTest `layer-streaming` family fails the same set of `-run`/preflight tests under
  **both** the current binary and the preserved control - environmental (the GPU mutex
  family and the model env var), not a regression from this work.
- The provenance gate above (item 7).

## 5. Exact commands

Host-only (no GPU, no model, safe to run any time):

```bash
cd /c/Development/AI/qwen38GZI
python tools/check-op-rows-probe.py --selftest                     # classifier
python tools/check-op-rows-probe.py build/llama-cuda/op-rows-cuda-t1.json
python tools/resplit-currentstate.py --verify                      # currentstate integrity
```

Build and the CPU probe CTest (touches the CUDA driver by enumeration; see item 6):

```bash
cd build/llama-cuda && cmake . && cmake --build . --target op-rows-probe --config Release
ctest -C Release -R "op-rows" --output-on-failure
```

The CUDA measurement, under the mutex (never `setsid`; check the GPU is free first):

```bash
cd /c/Development/AI/qwen38GZI
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
powershell -NoProfile -ExecutionPolicy Bypass -File tools/with-gpu-lock.ps1 \
  -VerdictTag gplr82-oprows-cuda -AllowUnprovenanced \
  -Command "C:\Windows\System32\cmd.exe /c build\llama-cuda\oprows-cuda.cmd"
```

`build/llama-cuda/oprows-cuda.cmd` is the recorded script for the three CUDA probe runs
(tokens=1, tokens=8, Q6_K down). To re-derive the *split* arms (C0/B1/B2) instead:
`powershell -File tools/with-gpu-lock.ps1 -VerdictTag <tag> -Command "...python
tools/run-gplr82-s3off.py"` - one mutex hold, ~165 s.

## 6. Tree state when this handoff was written

- **Nothing is committed.** Dirty: `third_party/llama.cpp` (the `--experts-host-leg`
  diagnostic from the previous session, plus the new probe and its CMake registration),
  `docs/currentstate.md`, `tools/run-tests.ps1`.
- Untracked: `tools/run-gplr82-s3off.py`, `tools/run-gplr82-a2dev.py`,
  `tools/check-op-rows-probe.py`, `third_party/llama.cpp/tools/layer-streaming/op-rows-probe.cpp`.
- Committed this campaign (by the owner's earlier authorization): the T-GPLR-53/54/56
  tooling series; the T-GPLR-82 falsifier records landed after that.
- A concurrent thread has its own uncommitted work in the same tree. Stage by file, never
  `git add -A`.

## 7. What to do next, in order

1. **Close the guard gap (item 5).** Extend the loud-failure contract to check expert
   activations for non-finite values before the matmul, and to check the device leg's rows
   too. Without it, a corrupt run reports `host_finite = 0` and passes AC2-style checks.
   This is the highest-value change in the whole campaign and it is independent of the
   remaining arithmetic question.
2. **Test the weight/row pairing hypothesis** (section 2, candidate 1) - it is the only
   candidate left that can produce a 1e-3-scale shift with the arithmetic proven exact,
   and it is checkable from a recorded trace plus a host-side join.
3. Only then decide AC1: if the pairing is right and the shift persists, the split's
   semantics need re-deriving, not tuning.
4. Fix the QA encoding crash in `run-gplr82-s3off.py` (item 4) so its recorded booleans
   are grades rather than crashes.
5. When the task is closed, write the owner summary in the AGENTS.md five-part form
   (goal, plain-language mechanism, glossary, pros/cons with measured/expected/unproven
   labels, recommendation). Gate PASSes are soft evidence and must never be presented as
   approval.

## 8. Evidence index

- 2x2 decoupling attribution: `build/llama-cuda/gplr82-s3off/gplr82-s3off-verdict.json`,
  arm traces/logs/tg-texts in the same directory, session log
  `build/llama-cuda/s3off-session.log`.
- Device-tail falsifier (session 3): `build/llama-cuda/gplr82-a2dev3-verdict.json`,
  per-arm files `build/llama-cuda/gplr82-a2dev3-arm-*.{jsonl,log,txt}`; pre-fix session
  `gplr82-a2dev2-verdict.json`; control binary side-copied BEFORE the rebuild at
  `build/llama-cuda/bin/a2dev-control-4412cc4a/`.
- Per-op row-count measurement: `build/llama-cuda/op-rows-{cuda-t1,cuda-t8,cuda-q6,cpu}.json`
  + human output in the wrapper verdict `build/llama-cuda/verdicts/gplr82-oprows-cuda.done`.
- The narrative record with the same numbers: `docs/currentstate.md`, entries dated
  2026-10-10 (the falsifier, the attribution, and the row-count refutation).
