# Two GTX 1070: parallel requests and 2x context (one llama-server process)

Status: measured, ready for the owner's pick (autonomous mode, owner grant 2026-11-15 for this goal; the agent's own acceptance below, not owner approval). Kept short on the owner's instruction: ideas are
measured first and documented only when they survive.

## Goal (owner, 2026-11-15)

Use both GTX 1070 (8 GiB each) from ONE `llama-server` process so that more requests can run in parallel and the server
holds at least 2x the context of today (two requests of 262k each, and / or one pool of >= 524k cells divided among slots; no
single request above the trained 262,144). The usual load is one request; `-np 2` and more must work. No preset speed bar: every
candidate is measured against the controls and the OWNER picks the winner (a later hybrid "second card joins when free" mode is
out of scope now). Model: Qwen3.6-35B-A3B-UD-Q5_K_XL + vision on demand, q8_0 KV. Machine: i7-4770K (4c/8t), 30 GB RAM.

## Acceptance contract (written before the first run)

Outcome: a table, per candidate, from `tools/pascal/par_bench.py` (same prompts, same binary family, GPU idle check before each arm):
solo decode tok/s (2 chats x 192 tokens), solo cold prefill tok/s at 4,096 and 16,384 tokens, and for n = 2 and 4 simultaneous
requests the per-request and summed decode tok/s, the prefill of n simultaneous cold 8,192-token prompts, peak memory per GPU;
plus a capacity run (`--fill`) that holds >= 2 x 262k tokens (or as near as the candidate reaches) and answers from each.

Criteria for calling the goal done (the agent's own acceptance, not owner approval):
- A1 controls measured: one-card baseline (owner's R7 command) and the existing two-GPU stack, at np 1 / 2 / 4.
- A2 every candidate in the list below is either measured with the same harness or closed with a measured reason.
- A3 at least one configuration demonstrably serves 2 simultaneous requests whose contexts sum to >= 524k tokens, or the
  measured limit and its cause are stated.
- A4 default behaviour with the new flags off is unchanged (controls rerun on the final binary agree within session noise, ~5%).
- A5 a capacity / correctness check per surviving candidate: each simultaneous request answers its own needle question
  (programmatic check), no error lines in the server log.
Claims allowed: speed ratios on this machine, model and harness only. No quality claim beyond the needle check.

## Candidates

| id | idea | state | result |
|---|---|---|---|
| C0 | control: one card, owner's R7 command | MEASURED (`build/pascal/par/c0-np1`, `c0-np4`) | solo decode 33.0-33.6 tok/s, prefill 736 / 865 tok/s at 4k / 16k; 2 chats at once 18-21 each (37-41 summed); 4 at once 9.6 each (38); 2 cold 8k prompts at once 198 tok/s summed (they fall off the layer-major pass), 4 at once 169; GPU 0 peak 7.8 GiB |
| C1 | control: second-GPU expert tier (`--hybrid-moe-gpu1 auto`) | MEASURED (`c1-tier-np4`) | solo 37; 2 at once 25-28 each (50-55); 4 at once 14.4-15.0 each (58-60); simultaneous prefill unchanged (198 / 155) |
| C1b | tier + two-GPU split prefill (`--lm-prefill-peer-gpu 1`) | FIXED, MEASURED (agent A, `p2-*`, `p4-peer-check`) | abort fixed (per-device input-batch event, 09edfb7). Needs `LLAMA_MOE_HYBRID_G1_RESERVE_MIB=5700` (4900 starves the follower pass: 4k prompt at 123 tok/s). Solo prefill 16k: 1,360 tok/s vs 850. GPU peak 7.8 / 7.5 GiB |
| C2 | pool `-c 524288 --kv-unified-per-slot 262144 -np 2` on the tier stack (one body) | CLOSED by measurement (`c2-pool524`) | starts and serves chats, but two 60k prompts fail at the 3rd pass: `lazy_commit_rows: could not commit KV rows [0, 155648) (40.0 MiB more): out of device memory` (GPU 0 peak 7,993 MiB, pass footprint 1,320 MiB). Cause: ALL KV lives on GPU 0 (10.6 KiB/token, so 524k cells = 5.4 GiB) next to the non-expert weights, the pass window and buffers; GPU 1 holds only expert-tier rows. A pool of 2 x 262k needs KV on both cards (C4, or KV placed on CUDA1, which the hybrid engine forbids: one CUDA backend per scheduler). Side result: the two prompts ran strictly one after the other (729 then 425 tok/s) because of the batch-sharing bug agent A fixed |
| C3 | BOUND: one full body per card, two processes (`tools/pascal/twoproc_bound.py`) | MEASURED (bound, not a deliverable) | solo 33-34 on either card; 2 chats at once 23-26 each (47-52); 2 cold 8k prompts at once 833 + 842 tok/s (1,675 summed); own 262k window per card (524k total); GPU 5.6 / 6.6 GiB; host RssAnon 0.5 GB per process (weights shared via mmap) |
| C4 | **`--gpu-instances 2`: one inference instance per card inside ONE process, ONE shared mapping of the model file** (opt-in flag; default untouched) | MEASURED (`g3-inst2-np2`, `g-inst2-np1`, `g-inst2-fill`, `g3-inst2-fill260`, `g-inst2-frozen`, `g3-c0-np1`) | one process, 2 x 262,144-token windows (524k cells), every `-np` per instance. np 1 per instance: solo decode 32-33, prefill 723 / 863 tok/s (4k / 16k); 2 chats at once 23.4 + 25.0 = 48 (twoproc bound 47-52); 2 cold 8k prompts at once 845 + 857 = 1,689 summed (bound 1,675). np 2 per instance (4 slots): solo 31, 2 chats 48, 4 chats 12.3 each = 49-52 summed, 2 cold 8k 1,588, 4 cold 8k 1,660 summed. Capacity: two ~200k prompts at once, both needles right (ZEBRA-4821, ORCHID-4958), 162 + 152 tok/s prefill, decode 15.9 + 17.7 at that depth, GPU peak 7.7 / 7.7 GiB; FILL260_RESULT. Host RAM: the weights are mapped ONCE (file pages 24.4 GiB one card, 24.8 GiB two instances; system use +2.1 -> +3.5 GiB, the difference is the vision projector, which each instance parks as its own 1.1 GiB page-locked copy). Vision answered through both instances, SIGINT exits 0. Default path unchanged: one-card control on the final binary decodes 33.4-33.7, prefill 734 / 848 (C0 was 33.0-33.6, 736 / 865) |
| C5 | one body, simultaneous big prompts keep the layer-major pass at np > 1 (agent A, ec57ac6; on whenever `--lm-prefill -np>1`, no flag) | MEASURED | cause: the isolation pick read the slot's cached length before the start logic settled it, so two big prompts shared a batch and fell to the ordinary path. Two 8k prompts at once: 908 tok/s summed, 24 s wall (control 198, 89 s); four: 891 summed, 48 s (control 169, 205 s). With the split prefill (C1b): 1,273 summed / 17.8 s at n=2, 1,330 / 35 s at n=4. Mixed (chat decoding + 8k prompt arriving): prompt 846 (1,222 with split), the chat decodes at 15.6 (18.1) tok/s meanwhile. 0 declines, 0 errors; first-token logprob deltas 0.001-0.036 (noise floor 0.0001-0.027); texts diverge at near ties as solo-vs-solo does |

Branch `dual-gpu-parallel` has agents A and B merged (C4 fixes 13ab053, 3b6cbd5). Resume: agents' worktrees are under `.claude/worktrees/` (check `git worktree list`); the control binaries are in `build/pascal/base-bin`.

## Decision log

- D1 Baseline model is the Q5_K_XL file without MTP (owner's answer "Q5_K_XL + vision, README numbers"); the MTP recipe is not part
  of this goal's control. Reversible.
- D2 Documentation is this one file; per-idea contracts are one table row (owner: do not document ideas that may not work).
- D3 GPU runs are serialised with `flock /tmp/gpu-2x1070.lock` (Linux; the Windows mutex wrapper does not apply here); builds in
  worktrees run in parallel.
- D4 One process on two cards is done as one instance per card (`--gpu-instances N`), not one body with KV on both cards: C2 showed all KV lives on GPU 0 and the hybrid engine allows one CUDA backend per scheduler. Reversible (opt-in flag; default path measured unchanged, A4).
- D5 (owner, 2026-11-21: "use the same weights in system ram for all requests and gpus, not mapped 2x") All instances of the process share ONE mapping of the model file (`llama_model_share_mmaps`, on only with `--gpu-instances`). Evidence: pinning the same file through a second mapping costs a second copy (2 x 6 GiB registered left 12 GiB less memory; the 23 GB of expert banks cannot be pinned twice in 30 GB, instance 1 died with "out of memory" at its first registration). A shared mapping is never partly unmapped, and uploads from it go through ordinary memory (a copy whose source starts or ends inside a page-lock registration fails with "invalid argument"). Cost: ~0.6 s more load time. Reversible.
- D6 `LLAMA_MOE_HYBRID` (the slot count an engine is created with) is a process-wide env value that every cache resize wrote back; with two instances it made instance 1 start with instance 0's full cache (no room for its layer-major slot buffers at -np 2) and made a decode graph rebuild abort ("engine ... another slot count"). An instance thread neither writes nor compares it. Reversible.
- D7 The frozen-cache greedy equality check (`--check-solo`) cannot decide here: solo vs solo again already differs (2-26 of 96 tokens in common), because two instances start with different cache contents. Recorded as UNPROVEN, not as a pass; A5 (needle + no errors) is the contract's correctness check and it passes.
- D8 Capacity fill is 2 x 200k (binary `final-bin2`) and 2 x 260k (final binary); a prompt must leave room for its answer inside 262,144, so 260,000 + 96 is the largest round fill. The agent's own acceptance below covers exactly what is listed there.

## Ranked table (one winner is the owner's call)

All rows: Qwen3.6-35B-A3B-UD-Q5_K_XL + vision on demand, q8 KV, the owner's R7 flags, `tools/pascal/par_bench.py`. Decode = tok/s per request; "summed" = all requests together. 8k prompts are cold and simultaneous.

| rank by goal fit | id | one process | solo decode / prefill 16k | 2 at once (decode summed) | 4 at once (decode summed) | 2 x cold 8k prefill (summed) | context in flight | GPU peak (MiB) |
|---|---|---|---|---|---|---|---|---|
| 1 | C4 instances, shared mapping | yes | 31-33 / 826-863 | 47-48 | 49-52 (np 2 per instance) | 1,588-1,689 | 2 x 262k windows; 2 x 200k prompts verified (needles right) | 7.8 / 7.8 GiB |
| 2 | C3 two processes (bound, not a deliverable) | no | 33-34 / ~850 | 47-52 | not run | 1,675 | 2 x 262k | 5.6 / 6.6 GiB |
| 3 | C1b tier + split prefill | yes | 37 / 1,360 | not run | not run | 1,273 | 262k (one KV on GPU 0) | 7.8 / 7.5 GiB |
| 4 | C1 expert tier on GPU 1 | yes | 37 / ~850 | 50-55 | 58-60 | 198 (155 at four) before C5 | 262k | see `c1-tier-np4` |
| 5 | C5 big-prompt isolation fix on C0 | yes | 33 / 865 | 37-41 | 38 | 908 (891 at four) | 262k | 7.8 / 0 GiB |
| 6 | C0 control, one card | yes | 33 / 865 | 37-41 | 38 | 198 (169 at four) | 262k | 7.8 / 0 GiB |
| - | C2 pool of 524k cells in one body | yes | - | - | - | - | CLOSED: KV commit fails near 155k rows (all KV on GPU 0) | 8.0 / - |

Reading it: only C3 (two processes) and C4 hold two full 262k windows; C4 is C3 inside one process with the same speed, one shared copy of the weights in RAM, vision and clean shutdown working. C1 / C1b are faster for ONE request but keep one 262k window. Nothing here makes a single request faster than one card except C1 / C1b (decode +12%, prefill +60% at 16k).

## Owner summary (agent's own autonomous acceptance, not owner approval)

**1. Goal.** One `llama-server` process on both GTX 1070 (8 GiB) cards: more parallel requests (`-np 2`, `-np 4`) and at least 2x today's 262k context (two requests of 262k, or a pool of 524k cells), keeping today's speed where possible, vision on demand, no preset speed bar.

**2. How it works.** `--gpu-instances 2` starts two complete copies of the inference engine in the same process, one per card, each with its own request slots, its own 262k window and its own thread. The model file is mapped into RAM once and both copies read the same pages; the expert weights stay in that shared RAM and are streamed to each card as needed. A small dispatcher sends a request to the copy that already holds its prompt prefix (otherwise to the idler one), so a conversation keeps its cache. A prompt longer than the window of one copy is not split between cards.

**3. Glossary.** PP = prompt processing (prefill), TG = token generation (decode), tok/s = tokens per second; `-np N` = N request slots; KV cache = the stored attention state of a conversation (about 10.6 KiB per token here); window = the 262,144-token context one instance can hold; expert banks = the large mixture-of-experts weights kept in host RAM; mmap = mapping a file into memory so pages are shared and loaded on demand; page-locked (pinned) = RAM the GPU can copy from directly; layer-major prefill = loading each layer's weights once and running the whole prompt through it; needle check = hiding a code word in a long prompt and checking the answer contains it; GiB = 1,024^3 bytes; VRAM = GPU memory.

**4. Pros and cons.**
- Pro (measured, `g3-inst2-np2`, `g-inst2-np1`): 2 requests at once decode 47-48 tok/s summed against 37-41 on one card; 4 at once 49-52 against 38; 2 cold 8k prompts at once 1,588-1,689 tok/s summed against 198 on one card before the C5 fix (908 after it).
- Pro (measured, `g-inst2-fill`, `g3-inst2-fill260`): two ~200k prompts at once in one process, both needle answers right; FILL260_PRO.
- Pro (measured): weights once in RAM (file pages 24.4 -> 24.8 GiB with two instances); vision requests answered through both instances; SIGINT exits 0; the default path (flag off) reproduces the control within 2-3% (`g3-c0-np1`).
- Con (measured): a single request is no faster than on one card (solo 31-33, prefill 726-863), only throughput under load improves; the vision projector costs 1.1 GiB of page-locked RAM per instance; each card holds nearly all 8 GiB (7.8 GiB), nothing is left for other GPU work (the later "second card joins when free" hybrid is a different design); at ~200k tokens the prefill is attention-bound (~150 tok/s per request, 22 minutes per prompt) and decode is 16-18 tok/s.
- Expected, not measured: balance when one instance is busy with a long prefill and the other idle (the dispatcher prefers prefix match, then load); behaviour over hours; `-np 2` at 70k-token contexts (memory-guard declines were an open item on one card too).
- Unproven: bit-identical greedy output against one card (D7); a one-card 200k / 260k control for the prefill speed (not run: a single card cannot hold two such requests, and one request alone was not timed).

**5. Recommendation.** Adopt C4 as the way to the goal: keep `--gpu-instances 2` opt-in, use it when two or more requests are expected; keep C0 (or C1 for +12% single-request decode) as the default one-request setup. Owner to decide: the winner among the table rows; whether the vision projector should also be shared across instances (saves 1.1 GiB of RAM, small work); whether a dispatcher policy for uneven load needs tuning after real use.

**6. Decisions** D1-D8 above. **Acceptance of the contract (the agent's own):** A1 controls measured (C0 / C1 / C1b plus the final-binary control `g3-c0-np1`, within 2-3% of C0: pass). A2 every candidate measured or closed with a measured reason (C0, C1, C1b, C3, C4, C5 measured; C2 closed by the commit failure): pass. A3 at least one configuration serves two simultaneous requests with contexts summing to the goal: C4, FILL260_A3. A4 default behaviour unchanged: pass (`g3-c0-np1`). A5 needle check per request and no error lines: pass for the two fills; the np 1 / np 2 arms have no error lines in their server logs (`build/pascal/par/g3-inst2-np2/server.log`). Scope: this machine, this model, this harness; no quality claim beyond the needle check; not certified for production.
