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
| C0 | control: one card, owner's R7 command, `-np 1/2/4` (unified KV) | | |
| C1 | control: existing two-GPU stack (`--hybrid-moe-gpu1 auto`, `--lm-prefill-peer-gpu 1`), `-np 1/2/4` | | |
| C2 | pool: `-c 524288 -np 2 --kv-unified-per-slot 262144` on the existing stack (lazy KV on GPU 0, experts pushed to GPU 1) | | |

(further candidates are added after the controls show where parallel decode loses its time)

## Decision log

- D1 Baseline model is the Q5_K_XL file without MTP (owner's answer "Q5_K_XL + vision, README numbers"); the MTP recipe is not part
  of this goal's control. Reversible.
- D2 Documentation is this one file; per-idea contracts are one table row (owner: do not document ideas that may not work).
- D3 GPU runs are serialised with `flock /tmp/gpu-2x1070.lock` (Linux; the Windows mutex wrapper does not apply here); builds in
  worktrees run in parallel.
