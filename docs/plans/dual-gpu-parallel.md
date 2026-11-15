# Two GTX 1070: parallel requests and 2x context (one llama-server process)

Status: in progress (autonomous mode, owner grant 2026-11-15 for this goal). Kept short on the owner's instruction: ideas are
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
| C1b | tier + two-GPU prefill (`--lm-prefill-peer-gpu 1`) at np 4 | CRASH at start (`c1-peer-np4`) | `g_cuda_input_event` (created on device 0) used on device 1 in `ggml_cuda_input_batch_flush`; fix assigned to worktree agent A |
| C2 | pool `-c 524288 --kv-unified-per-slot 262144 -np 2` on the tier stack | FAILED at 2 x 60k (`c2-pool524`) | HTTP 500: `lazy_commit_rows: could not commit KV rows [0, 155648): out of device memory` on GPU 0 (peak 7,993 MiB). Cause not yet isolated (row range committed vs. cache not shrinking); open |
| C3 | BOUND: one full body per card, two processes (`tools/pascal/twoproc_bound.py`) | MEASURED (bound, not a deliverable) | solo 33-34 on either card; 2 chats at once 23-26 each (47-52); 2 cold 8k prompts at once 833 + 842 tok/s (1,675 summed); own 262k window per card (524k total); GPU 5.6 / 6.6 GiB; host RssAnon 0.5 GB per process (weights shared via mmap) |
| C4 | C3 inside ONE process (an instance per card, dispatch layer) | IN PROGRESS, worktree agent B (interrupted by the usage limit) | |
| C5 | serialised layer-major passes for simultaneous big prompts at np > 1 (one body) + C1b fix | IN PROGRESS, worktree agent A (interrupted by the usage limit) | |

Resume: agents' worktrees are under `.claude/worktrees/` (check `git worktree list`); the control binaries are in `build/pascal/base-bin`.

## Decision log

- D1 Baseline model is the Q5_K_XL file without MTP (owner's answer "Q5_K_XL + vision, README numbers"); the MTP recipe is not part
  of this goal's control. Reversible.
- D2 Documentation is this one file; per-idea contracts are one table row (owner: do not document ideas that may not work).
- D3 GPU runs are serialised with `flock /tmp/gpu-2x1070.lock` (Linux; the Windows mutex wrapper does not apply here); builds in
  worktrees run in parallel.
