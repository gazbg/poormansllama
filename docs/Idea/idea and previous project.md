

# Qwen3 (Qwen3-8B) Integration Reference

- Status: reference for a future plan — not itself the plan
- Prepared: 2026-08-18
- Target paper: [Qwen3 Technical Report, arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- Target checkpoint: `Qwen/Qwen3-8B` (Hugging Face, `model_type=qwen3`)
- Related repo evidence: DeepSeek V4 (`c/deepseek_v4.c`, `c/v4/`, `docs/plans/deepseek-v4-gpu-performance-plan.md`, `currentstate.md`), OLMoE (`c/olmoe.c`), Kimi K3 (`c/kimi_k3.c`), Inkling (`c/inkling.c`), GLM-5.2 (`c/colibri.c`)

## Purpose

Comment on this line — sent with your next message
Cancel
Save comment
This document is the complete reference a Planner needs to create an
executable plan that replicates, for **Qwen3-8B**, the work this repository
already did for DeepSeek V4 and the other model families: a dependency-free C
engine, a `c/coli`-roster integration, oracle/tiny-fixture validation, the
format/quantization machinery, and (as a second phase) the GPU pipeline
pattern.

Every architecture, tokenizer, and checkpoint fact below was verified against
a primary source on 2026-08-18 (paper text, `config.json`,
`model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`,
`generation_config.json`, `transformers` v4.51.0 source, and byte arithmetic
on the checkpoint size). Where a number was edited post-release, the
discrepancy is stated explicitly; the engine must be config-driven and never
hardcode these values.

"Qwen 3.8" is interpreted as the **Qwen3 family, 8B dense model** — the arXiv
link is the Qwen3 Technical Report, and the 8B is the flagship edge-size
dense checkpoint of that release. The whole family is in scope as a
specification (the engine should be family-capable), but the acceptance
target is Qwen3-8B.

---

## 1. Scope and target

### 1.1 What "replicating the work we do here" means

The repo integrates each model family as **one self-contained C engine**
(a `c/<family>.c` file) that loads `config.json` + sharded safetensors via the
shared `c/st.h`, tokenizes via `c/tok.h`, quantizes via `c/quant.h`, samples
via `c/sample.h`, and emits routing/telemetry via `c/route_trace.h` /
`c/telemetry.h`. A Python dispatcher (`c/coli`) detects the architecture from
`config.json`, picks the engine binary, and fronts `run`/`chat`/`serve`/`web`.
Validation uses two generated fixtures per family: a **tiny deterministic
checkpoint + oracle JSON** (reference token ids from the official
`transformers` implementation) and a **real-checkpoint oracle**; both run
through a ladder of focused fixtures → teacher-forcing → greedy decode →
smoke scripts → long-context runs. DeepSeek V4 additionally established the
GPU delivery pattern: an SDD, a plan with vertical task contracts
(`docs/plans/<area>-plan.md`), a living evidence file (`currentstate.md`),
telemetry-gated smoke scripts, and a determinism/fallback contract.

Replicating that work for Qwen3-8B means delivering, in order:

1. Tokenizer support (cl100k-family regex + Qwen3 vocab/merges + special
   tokens + chat/thinking template) with a real-tokenizer fixture.
2. Config-driven weight loading (BF16 through `st.h`) for the exact tensor
   inventory in Section 4.
3. A CPU core forward (embed → 36 dense layers: QK-Norm + RoPE + GQA
   attention + SwiGLU MLP → final RMSNorm → untied `lm_head`) that is
   **token-identical to the transformers oracle** under teacher forcing and
   greedy decode.
4. KV-cache decode loop, sampling (temperature/top-p/top-k/min-p), chat
   mode with thinking/non-thinking blocks and stop-token handling.
5. Quantization + RAM residency policy (a repack tool + format decision),
   then the `c/coli` roster/Makefile/docs integration.
6. Long-context (32K native; 128K via YaRN) endgame validation.
7. (Second phase, separate plan) the CUDA pipeline replicating the V4 GPU
   pattern for a **dense** model.

### 1.2 In-scope vs out-of-scope (recommended for the first plan)

| In scope | Out of scope (later plans) |
|---|---|
| Qwen3-8B (dense, 36 layers) CPU-first engine | Qwen3 MoE (30B-A3B, 235B-A22B) routing/experts |
| Full attention as shipped (all 36 layers) | Sliding-window attention execution (only if a checkpoint enables it) |
| Thinking + non-thinking chat template, `/think` `/no_think` | Tools/function-calling template branch, grammar JSON mode |
| Greedy + top-k/top-p/temp/min-p sampling | Qwen3-2507 Instruct/Thinking variants (256K/1M context, different template) |
| 32K native context; 128K YaRN behind `rope_scaling` | Vision/multimodal, embedding/reranker heads |
| CPU + (later) CUDA GPU pipeline for dense inference | DSpark/speculative decoding, MTP |

---

## 2. Qwen3-8B architecture (verified)

### 2.1 Verified config (`Qwen/Qwen3-8B/config.json`, fetched 2026-08-18)

| Key | Value | Notes |
|---|---|---|
| `model_type` | `qwen3` | `qwen3_moe` for the MoE variants |
| `architectures` | `["Qwen3ForCausalLM"]` | |
| `vocab_size` | 151936 | **See §3.1**: tokenizer has 151,669 tokens; checkpoint rows are 151,936 (verified by byte math) |
| `hidden_size` | 4096 | |
| `intermediate_size` | 12288 | **Verified against checkpoint bytes** (§4.2); do not trust blog numbers (14436) |
| `num_hidden_layers` | 36 | |
| `num_attention_heads` | 32 | GQA |
| `num_key_value_heads` | 8 | |
| `head_dim` | 128 | explicit in config |
| `rms_norm_eps` | 1e-6 | |
| `rope_theta` | 1000000.0 | |
| `max_position_embeddings` | 40960 | model card: 32768 out + 8192 prompt reserve by design |
| `attention_bias` | false | no q/k/v/o biases anywhere |
| `use_sliding_window` | false | ⇒ all 36 layers full attention (see §2.3) |
| `sliding_window` | null | |
| `max_window_layers` | 36 | |
| `tie_word_embeddings` | false | `lm_head.weight` present in checkpoint, untied (paper Table 1 agrees) |
| `torch_dtype` | `bfloat16` | every weight is BF16 |
| `bos/eos/pad` | 151643 / 151645 / 151643 | `<\|endoftext\|>` / `<\|im_end\|>` |

### 2.2 Forward pass semantics (from `transformers` v4.51.0 `modeling_qwen3.py`)

The port must reproduce this exact ordering:

1. `inputs_embeds = embed_tokens(input_ids)` — no bias.
2. Per layer `i` in 0..35:
   - `hidden = input_layernorm(hidden)` (RMSNorm, float32 accumulation,
     `rsqrt(mean(x²)+eps)`, weight multiply, cast back to working dtype).
   - **Attention** (`Qwen3Attention`, layer `i`):
     - `q = q_proj(hidden)` → view `[*, 32, 128]`; `k = k_proj(hidden)` →
       view `[*, 8, 128]`; `v = v_proj(hidden)` → view `[*, 8, 128]`.
     - **QK-Norm**: `q = RMSNorm_128(q)` and `k = RMSNorm_128(k)` —
       RMSNorm over the head dim only, applied **before RoPE**. This is the
       novel Qwen3 step (absent from Qwen2.5 and from DeepSeek V4).
     - **RoPE** on the full head_dim (128): `freqs = pos * inv_freq` where
       `inv_freq = 1/theta^(2i/128)`, `emb = cat(freqs, freqs)`,
       `cos/sin = emb.cos()/emb.sin()`, applied as
       `q_embed = q*cos + rotate_half(q)*sin` (standard GPT-J interleave).
       `rope_scaling` is `null` in the checkpoint ⇒ `rope_type="default"`,
       attention scaling 1.0. YaRN is a *user-added* config for 128K only.
     - Attention: scaling `head_dim^-0.5`, causal mask, GQA repeat of the 8
       KV heads to 32, softmax in float32, weighted sum.
     - `attn_out = o_proj(concat heads)`.
     - SWA semantics (implement for family completeness): a layer uses a
       sliding window only if
       `use_sliding_window && sliding_window && layer_idx >= max_window_layers`;
       otherwise full attention. With the released Qwen3-8B config every
       layer is full attention.
   - Residual: `hidden += attn_out`.
   - `hidden = post_attention_layernorm(hidden)`.
   - **MLP** (SwiGLU): `hidden += down_proj(silu(gate_proj(x)) * up_proj(x))`.
   - All `nn.Linear` have `bias=False`.
3. `hidden = model.norm(hidden)` (final RMSNorm).
4. `logits = lm_head(hidden)` — `Linear(4096, 151936, bias=False)`,
   **untied** for 8B.

Weight-name inventory (verified against `model.safetensors.index.json`):

```
model.embed_tokens.weight                       [151936, 4096]
model.norm.weight                               [4096]
lm_head.weight                                  [151936, 4096]
model.layers.{i}.input_layernorm.weight         [4096]
model.layers.{i}.post_attention_layernorm.weight [4096]
model.layers.{i}.self_attn.q_proj.weight        [4096, 4096]   (32*128 x 4096)
model.layers.{i}.self_attn.k_proj.weight        [1024, 4096]   (8*128 x 4096)
model.layers.{i}.self_attn.v_proj.weight        [1024, 4096]
model.layers.{i}.self_attn.o_proj.weight        [4096, 4096]
model.layers.{i}.self_attn.q_norm.weight        [128]
model.layers.{i}.self_attn.k_norm.weight        [128]
model.layers.{i}.mlp.gate_proj.weight           [12288, 4096]
model.layers.{i}.mlp.up_proj.weight             [12288, 4096]
model.layers.{i}.mlp.down_proj.weight           [4096, 12288]
```

No biases, no MoE tensors, no extra heads. 10 tensors per layer × 36 + 3
globals = 363 tensors across 5 shards.

### 2.3 The hybrid-attention question (important for the plan)

The paper (§2) describes hybrid attention — full attention in early layers,
sliding window afterwards — as the family design. **The released dense
checkpoints ship `use_sliding_window: false`** (verified for Qwen3-8B and
Qwen3-30B-A3B), so Qwen3-8B as distributed runs **full attention in all 36
layers**. The engine must implement the config-driven `layer_types` semantics
(§2.2) so any Qwen3 checkpoint works, but the 8B acceptance runs full
attention. This removes the SWA ring complexity from the V4-style sparse
attention work entirely.

### 2.4 Family table (paper §2, Tables 1–2) — for family coverage

| Model | Layers | Q/KV heads | head_dim | Tie embed | Native ctx | Notes |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 16/8 | 128 | yes | 32K | |
| Qwen3-1.7B | 28 | 16/8 | 128 | yes | 32K | |
| Qwen3-4B | 36 | 32/8 | 128 | yes | 128K (YaRN) | |
| **Qwen3-8B** | **36** | **32/8** | **128** | **no** | **32K → 128K YaRN** | acceptance target |
| Qwen3-14B | 40 | 40/8 | 128 | no | 128K (YaRN) | |
| Qwen3-32B | 64 | 64/8 | 128 | no | 128K (YaRN) | |
| Qwen3-30B-A3B (MoE) | 48 | 32/4 | 128 | no | 128K (YaRN) | 128 experts, top-8, no shared experts |
| Qwen3-235B-A22B (MoE) | 94* | 64/4 | 128 | no | 128K (YaRN) | 128 experts, top-8 |

\* Paper Table 2 says 94; `config.json` is authoritative for a real
checkpoint. Per-size `intermediate_size` also varies (e.g. 30B-A3B: dense MLP
6144, `moe_intermediate_size` 768) — read from the checkpoint, never
hardcode.

### 2.5 MoE variant facts (only if a later plan adds Qwen3-30B-A3B)

Verified from `Qwen/Qwen3-30B-A3B/config.json`: `model_type=qwen3_moe`,
hidden 2048, 48 layers, 32/4 heads, head_dim 128, `num_experts=128`,
`num_experts_per_tok=8`, `moe_intermediate_size=768`, `norm_topk_prob=true`,
**no shared experts** (unlike Qwen2.5-MoE and unlike DeepSeek V4's
shared_experts), `router_aux_loss_coef` 0.001 (training only), attention_bias
false. Routing in transformers: top-8 by gate logits, softmax over selected,
`norm_topk_prob` divides by the sum of top-k probabilities. The MoE engine
work would be *simpler* than V4 (no shared experts, no MLA/indexer/
compressor), but it is deliberately out of the first plan.

---

## 3. Tokenizer and chat (verified)

### 3.1 Tokenizer facts (inspected directly from `Qwen/Qwen3-8B/tokenizer.json`, 2026-08-18)

| Property | Value |
|---|---|
| BPE model | `type=BPE`, **merges-based** (151,387 merges), `byte_fallback=false`, `ignore_merges=false` |
| Base vocab | **151,643** tokens (ids 0..151642) |
| Added tokens | **26** (ids 151643..151668); 13 marked `special: true` |
| Total tokens | 151,669 (= paper's "vocabulary size of 151,669") |
| Pre-tokenizer | `Split` regex **`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`** then `ByteLevel(add_prefix_space=false, use_regex=false)` |
| Normalizer | **NFC** |
| Post-processor | `ByteLevel` |
| `tokenizer_class` | `Qwen2Tokenizer` (a `PreTrainedTokenizerFast` wrapper) |
| Chat template | Jinja in `tokenizer_config.json` (simple + tools branches); `model_max_length=131072` |

**Key compatibility finding for `c/tok.h`:** the Qwen3 pre-tokenizer regex is
the **cl100k-style pattern** — it contains `\p{L}`/`\p{N}` but **no
`\p{Lu}` and no `\p{Han}`**. `tok.h` family detection (lines ~217–225)
routes `\p{Lu}` → o200k, `\p{Han}` → kimi, and everything else → the cl100k
path (`pretok_chunk`, which implements exactly this regex). **Qwen3 therefore
needs no new pre-tokenizer family**: it should take the existing cl100k path.
What must still be verified with a real fixture:

1. `tok_load` accepts the merges-based BPE + 26 added tokens (existing
   support; Qwen3 has 151,387 merges — the largest corpus yet).
2. **NFC gap**: `tok.h` applies no Unicode normalization; Qwen3 declares NFC.
   ASCII is unaffected (so oracle fixtures on ASCII prompts will not catch
   it), but CJK and accented text can diverge from HF. Decide: implement NFC
   in `tok.h` (or a Qwen3-only flag), or document the divergence. Recommend
   implementing it — Qwen3 is a 119-language model and this is a correctness
   item, not a performance item.
3. `add_prefix_space=false` matches the Qwen3 ByteLevel config; confirm the
   cl100k path honors it.

### 3.2 Special tokens (ids 151643–151668, from `tokenizer_config.json` / `tokenizer.json`)

```
151643 <|endoftext|>   (special; also pad)     151657 <tool_call>
151644 <|im_start|>    (special)                151658 </tool_call>
151645 <|im_end|>      (special; eos)           151659 <|fim_prefix|>
151646 <|object_ref_start|> (special)           151660 <|fim_middle|>
151647 <|object_ref_end|>   (special)           151661 <|fim_suffix|>
151648 <|box_start|>   (special)                151662 <|fim_pad|>
151649 <|box_end|>     (special)                151663 <|repo_name|>
151650 <|quad_start|>  (special)                151664 <|file_sep|>
151651 <|quad_end|>    (special)                151665 <tool_response>
151652 <|vision_start|>(special)                151666 </tool_response>
151653 <|vision_end|>  (special)                151667 <think>
151654 <|vision_pad|>  (special)                151668 </think>
151655 <|image_pad|>   (special)
151656 <|video_pad|>   (special)
```

`eos_token_id = [151645, 151643]` (generation_config). `<think>`/`</think>`
are **not** special (they are regular vocab tokens) — generation must not
stop on them; stop only on `<|im_end|>` / `<|endoftext|>` (exactly what
`sample.h`'s `stops_arm_tok` does from the tokenizer's special flags).

### 3.3 Chat template (simple, non-tools form — what the engine must build)

```
<|im_start|>system\n{system}<|im_end|>\n
<|im_start|>user\n{query}<|im_end|>\n
<|im_start|>assistant\n            <- generation prompt
```

- Thinking mode (default, `enable_thinking=True`): model is expected to
  output `<think>\n{reasoning}\n</think>\n\n{answer}`. The template adds
  nothing extra for thinking; the `<think>` opening is generated by the model
  (the card's parsing code finds the last `</think>` = id 151668).
- Non-thinking mode (`enable_thinking=False`): the template appends the
  **empty think block** `<think>\n\n</think>\n\n` after the assistant marker.
- Soft switches `/think` and `/no_think` in user/system content toggle mode
  per turn; the engine can support them at prompt-build time.
- `add_generation_prompt=False` variants and the tools branch are out of
  scope for the first plan (mirror V4's `coli_v4_prompt_build` CHAT/THINKING/
  RAW modes; Qwen3's THINKING is the default).

### 3.4 Generation defaults (verified)

`generation_config.json`: `temperature=0.6`, `top_p=0.95`, `top_k=20`,
`do_sample=true`, eos `[151645,151643]`, pad 151643. Model card "Best
Practices":

| Mode | temp | top_p | top_k | min_p |
|---|---|---|---|---|
| thinking (default) | 0.6 | 0.95 | 20 | 0 |
| non-thinking | 0.7 | 0.8 | 20 | 0 |

Card explicitly warns **against greedy decoding for real use** (repetition).
Implication for validation: oracle equivalence must use **greedy argmax** for
token-exact comparisons (as V4 does), while the shipped `chat`/`serve`
defaults must be sampling with the table above. **`sample.h` has no min-p
sampler today** (only temperature + nucleus) — add one (a small, focused
change; min_p=0 makes it a no-op by default).

---

## 4. Checkpoint and memory (verified)

### 4.1 Inventory

- 5 shards `model-00001-of-00005.safetensors` … `model-00005-of-00005.safetensors`
  (weights verified per-layer in §2.2), all `BF16`.
- `total_size` per the safetensors index: **16,381,470,720 bytes** ≈ 15.3 GiB.

### 4.2 Byte-arithmetic verification (how we know the config is right)

16,381,470,720 / 2 bytes = **8,190,735,360 params**. Decomposed:

```
embed + lm_head = 2 × 151,936 × 4,096        = 1,244,659,712
per layer: q 4096² + k 4096·1024 + v 4096·1024 + o 4096²
           + q_norm 128 + k_norm 128
           + in/out layernorm 2×4096
           + gate/up/down 3 × 12,288·4,096  = 192,946,432
36 layers                                    = 6,946,071,552
model.norm                                    =         4,096
total                                        = 8,190,735,360  ✓ exact
```

This pins `intermediate_size=12288`, `vocab=151936` rows for both embed and
`lm_head`, and the untied head — even though the tokenizer only defines
151,669 ids (the config-vs-tokenizer `vocab_size` mismatch is a known
post-release edit, HF discussion #25; ids 151,669..151,935 are never
produced by the tokenizer). The loader must take shapes from the shards
(`st.h` does) and never assert `tokenizer.vocab == config.vocab_size`.

### 4.3 Memory budget

| Representation | Resident size |
|---|---|
| BF16 on disk / mmap | 15.3 GiB |
| f32 in RAM (no quant) | ~32.8 GiB — over the classic 15 GB envelope |
| int8-row (fmt=1) | ~8.2 GiB (8,190,735,360 B + per-row f32 scales) |
| int4-row (fmt=2) | ~4.3 GiB |

Qwen3-8B is the first "fits comfortably at int4/int8" flagship-class model
in the roster. A repack tool (mirroring `c/tools/convert_olmoe.py` /
`repack_fp8_passthrough.py`) quantizing BF16 → existing stable formats
(fmt=1 `int8-row`, fmt=2 `int4-row`, or fmt=4 `int4-grouped`) needs **no new
format-approval step** (FORMATS.md registry: 0–8 are assigned; new formats
need the PRIVATE ORDINAL BLOCK ≥100 + owner approval procedure). The V4
FP8-E4M3 128×128 (fmt=8) path does not apply to a BF16 checkpoint unless we
emit it ourselves. Recommend: CPU correctness first at BF16→f32, then a
repack to int8-row as the default (dequant-on-use, `quant.h` `matmul_q`
pattern from `olmoe.c`), int4 later.

---

## 5. How colibri integrates a model today (the pattern being replicated)

| Concern | Where it lives today | What Qwen3 needs |
|---|---|---|
| Engine | one C file per family: `c/deepseek_v4.c`, `c/olmoe.c`, `c/kimi_k3.c`, `c/inkling.c`, `c/colibri.c` (GLM) | new `c/qwen3.c` (+ `c/qwen3.h` if a public API is wanted, like `deepseek_v4.h`) |
| Safetensors | shared `c/st.h` (`st_init`, `st_read_f32` converts BF16/F16/F32; per-shard index; `__metadata__` stamp ingest) | reuse unchanged |
| Tokenizer | shared `c/tok.h` (cl100k/o200k/kimi families; `tok_load` from `tokenizer.json`) | cl100k path + NFC decision + fixture |
| Quantized matmul | shared `c/quant.h` (fmts 0–8, `matmul_q`, `matmul_i4`, `matmul_fp8`, …) | reuse (fmt=1 first) |
| Sampling/stops | shared `c/sample.h` (`pick_tok`, `stops_arm_tok`) | reuse + add min-p |
| Dispatch | `c/coli` python: `model_arch()` (reads `config.json` `model_type`), `_BANNER_MODELS`, `engine_for()`, `run/chat/serve/web` | add `qwen3` arch + banner row + engine lookup |
| Build | `c/Makefile` per-family targets (`olmoe`, `kimi_k3`, `inkling`, `deepseek-v4`, …) | `make qwen3` target |
| Tiny fixture | `c/tools/make_deepseek_v4_tiny.py` + `make deepseek-v4-tiny-check` (pinned torch+transformers, deterministic seed, never-committed safetensors, reference tokens from the official model) | `make_qwen3_tiny.py` + `make qwen3-tiny-check` |
| Real oracle | `c/tools/make_deepseek_v4_oracle.py` + `make deepseek-v4-oracle MODEL=…` (transformers-preferred, coli-self fallback, top-1 token match) | `make_qwen3_oracle.py` (transformers ≥4.51 has official `Qwen3ForCausalLM`) |
| Model docs | `docs/deepseek-v4.md`, `docs/kimi_k3.md`, `docs/inkling.md` | `docs/qwen3.md` |
| Format registry | `docs/FORMATS.md` | only if a new format is proposed |
| Runtime env vars | `docs/ENVIRONMENT.md` (generated; see `docs/MAINTAINING-DOCS.md`) | regenerate after new vars |
| Roster/README | `README.md` "Five families run today" + banner in `c/coli` | update both |
| Agent process | `AGENTS.md` (Planner/Implementer), `docs/plans/<area>-plan.md`, `currentstate.md`, `docs/AGENT_ENVIRONMENT.md` | new plan + state updates |

### 5.1 The DeepSeek V4 GPU pattern (second phase)

The V4 GPU delivery (`docs/deepseek-v4-gpu-performance-sdd.md`,
`docs/plans/deepseek-v4-gpu-performance-plan.md`, `currentstate.md`, `c/v4/`
modules, `c/backend_cuda.cu`, `c/build-v4.ps1`, `c/run-v4-smoke.ps1`) is the
template for the Qwen3 GPU phase. Its vertical task format (Goal / Scope /
Non-goals / Design and failure contract / Implementation steps / Acceptance
criteria / Required validation / State update) must be reused verbatim in the
Qwen3 plan.

**Delta: dense vs V4's hybrid sparse MoE.** The V4 machinery splits into
"transfers to a dense model" and "MoE-only":

| V4 subsystem | Applies to Qwen3-8B? |
|---|---|
| PP chunking + device KV ring per layer (`T-v4gpu-22`) | yes (full-attention ring, 36 layers, 8 KV heads) |
| Decode/TG device-resident hidden+KV, device final head (`T-v4gpu-09`) | yes (`lm_head` 151,936×4096 BF16 ≈ 1.16 GiB resident) |
| Eager dense warm + VRAM accounting + telemetry (`T-v4gpu-20/21/23`) | yes |
| Determinism contract (same-GPU reproducibility, fallback, no CPU MoE) | yes (as "no silent precision/router changes" + `fallback=cpu` counters) |
| Sparse-attention ring, indexer, compressor, DSA | **no** (not in Qwen3) |
| Router top-k, expert store, RAM ring, full-layer residency, TG prefetch (`T-v4gpu-10..19`) | **no** for 8B; later if 30B-A3B is added |
| MXFP4 rows16 / FP8 E4M3 kernels | only if a repacked quantized checkpoint is used |

KV cache math for the plan (full attention ⇒ K+V per position = 36 × 8 × 128
× 2 × 2 B = 147,456 B/position ≈ 144 KB/position): ≈ 0.6 GB @4K, 2.4 GB
@16K, 4.8 GB @32K, 19.3 GB @128K YaRN. Native 32K fits the V4-style VRAM
accounting easily; the 128K YaRN path must budget ~19 GB for KV alone (RAM
resident at 32.8 GB f32 already; context-shifting/eviction is out of scope).
There is no sparse-window ring to build for the shipped checkpoints.

---

## 6. Proposed task backlog (vertical slices, for the Planner to contract)

Follow the AGENTS.md task format exactly. This section proposes the slices,
ordering, and falsifiable acceptance per slice; the Planner owns the final
contracts, task ids, and dependency edges.

### T-Q3-01: Environment and checkpoint acquisition
- Scope: `docs/AGENT_ENVIRONMENT.md` additions; download `Qwen/Qwen3-8B`
  (HF or ModelScope); `transformers>=4.51` + `torch` + `tokenizers` in the
  `convert`/`oracle` extras (already declared — no new dependency approval
  expected; confirm).
- Acceptance: checkpoint dir with 5 shards + `config.json` +
  `tokenizer.json` + `tokenizer_config.json` + `generation_config.json`;
  `total_size` 16,381,470,720 verified; oracle deps import cleanly.
- Validation: shard-size compare vs HF (the V4 truncated-download pitfall),
  `python -c "import transformers; assert transformers.__version__ >= '4.51'"`.

### T-Q3-02: Tokenizer + chat template
- Scope: `c/tok.h` (family routing, NFC decision), new
  `c/tests/test_tok_qwen3.c` + a **real-tokenizer fixture** generated from the
  downloaded `tokenizer.json` (fixture may be committed if kept small — or a
  generator script `c/tools/make_qwen3_tok_fixture.py` that must not need the
  full 11 MB file at `make check` time), prompt builder replicating §3.3.
- Acceptance:
  - encode/decode equality vs HF `AutoTokenizer` on a fixed corpus covering
    ASCII, CJK, accented/decomposed text, contractions, numbers, newlines —
    id-for-id.
  - special-token ids 151643–151668 decode to the exact strings; `<think>`/
    `</think>` are not stop tokens; stops arm on 151645/151643.
  - chat template output bytes equal HF `apply_chat_template` for
    thinking/non-thinking and `/no_think` cases.
  - if NFC is implemented: NFC/NFD input pairs produce identical ids.
- Validation: `make check` + the fixture test; the template cases run
  through the engine binary with `CHAT=1`.

### T-Q3-03: Config parse + weight load (BF16 via st.h)
- Scope: `c/qwen3.c` config struct mirroring `ColiDeepSeekV4Config` parsing;
  `st.h` reuse; exact-name/shape validation for all 363 tensors.
- Acceptance: engine loads the real checkpoint within the declared RAM
  budget; every tensor name + shape matches §2.2; vocab-mismatch (151,936 vs
  151,669) handled without error; missing/extra tensor → loud refusal (not a
  crash).
- Validation: loader fixture on the real checkpoint; negative tests with a
  corrupted index.

### T-Q3-04: CPU core forward + oracle equivalence (the correctness gate)
- Scope: embed → 36 layers (QK-Norm, RoPE full-dim, GQA, causal) → final
  norm → `lm_head`; `make_qwen3_tiny.py` + `make_qwen3_oracle.py`.
- Acceptance:
  - tiny checkpoint (e.g. VOCAB=128, HIDDEN=128, 3 layers, seeded) teacher
    forcing: **top-1 token match 100%** vs transformers oracle (V4's
    contract; logits need not be bit-exact).
  - real checkpoint teacher forcing on ≥1 fixed prompt: 100% top-1.
  - CPU-only build; no new external dependency at runtime.
- Validation: `make qwen3-tiny-check`; `make qwen3-oracle MODEL=…`.

### T-Q3-05: KV cache + decode loop (greedy, long sessions)
- Scope: per-layer KV cache (GQA 8 heads), causal attention with cache,
  prefix reuse, session reset; greedy argmax.
- Acceptance: greedy continuation on real checkpoint is byte-identical to the
  transformers greedy oracle on ≥2 fixed prompts (≥32 tokens); ≥3 sequential
  requests in one process keep the process warm and outputs stable
  (reproducibility contract: same input → same output across runs);
  determinism under `make check` fixtures.
- Validation: oracle script + smoke script `c/run-qwen3-smoke.sh/.ps1`
  (mirror `run-v4-smoke.ps1`: exit-code propagation, explicit config echo).

### T-Q3-06: Sampling + thinking mode + chat integration
- Scope: add min-p to `c/sample.h` (no-op at 0); TEMP/TOP_P/TOP_K/MIN_P env
  knobs (mirror olmoe's `g_temp`/`g_nuc`); thinking/non-thinking template
  handling; `/think` `/no_think`; stop-set wiring.
- Acceptance: with seed-fixed sampling the engine reproduces HF sampling
  distributions within tolerance on a fixture (or, at minimum, the samplers'
  unit tests: top-k/top-p/min-p cut sets are exactly the reference cut sets
  for known logit vectors); thinking mode emits `<think>…</think>` blocks
  and stops at im_end; non-thinking mode never emits think blocks; greedy
  still available for oracle validation.
- Validation: new `sample.h` unit cases + chat smoke.

### T-Q3-07: Quantization + RAM residency + repack tool
- Scope: decision + tool `c/tools/repack_qwen3_int4.py` (mirror
  `convert_olmoe.py`), reading BF16 and writing fmt=1/fmt=2 containers;
  `--ram` budget policy in the engine (mirror V4/GLM planner).
- Acceptance: repacked int8-row checkpoint loads and produces **byte-identical
  tokens** to the BF16 path on the oracle prompts within the established
  numeric envelope (int4: document the divergence envelope, keep exactness
  gate at int8); the registry row/FORMATS.md updated only if a new format is
  proposed (else no registry change).
- Validation: paired int8-vs-BF16 oracle run; RAM ceiling measured.

### T-Q3-08: Integration (roster, build, docs, serve/web)
- Scope: `c/coli` (`model_arch` → `qwen3`, banner row, engine lookup),
  `c/Makefile` (`make qwen3`), README roster line, `docs/qwen3.md`,
  `docs/ENVIRONMENT.md` regeneration, SUMMARY/docs index, `CHANGELOG.md`.
- Acceptance: `python ./coli chat --model <Qwen3-8B dir>` runs with the
  banner naming Qwen3-8B; `serve` answers one greedy/sampled request with
  warm-cache reuse; README + docs render correctly.
- Validation: full `make check` baseline + manual chat/serve smoke.

### T-Q3-09: Long-context endgame (32K native; 128K YaRN)
- Scope: context length handling at 32K; YaRN `rope_scaling` support
  (`factor=4.0`, `original_max_position_embeddings=32768`) behind config;
  chunked PP if needed; repetition/telemetry.
- Acceptance: 32K prompt completes; with YaRN config the 128K run completes
  and (on a needle probe) retrieves; no OOM within the declared RAM budget;
  determinism holds.
- Validation: 4K then 32K smoke; one 128K run recorded (mirror V4's
  4K/100K ladder discipline in `AGENT_ENVIRONMENT.md`).

### T-Q3-GPU-* (second plan, after CPU plan closes)
Replicate the V4 GPU pattern for a dense model: device PP attention + per-layer
KV ring (`T-v4gpu-22`/`T-v4gpu-15x` shape), decode/TG device residency + device
`lm_head` argmax (`T-v4gpu-09`), dense warm + VRAM accounting + telemetry
(`T-v4gpu-20/21/23`), CUDA-Graph capture on stable shapes, and the
determinism/fallback contract (same-GPU reproducible, `fallback=cpu`
counters, never silent precision changes). No MoE subsystems. Deliver its own
SDD + plan + currentstate, on `main` only by explicit owner decision (the V4
sequence's precedent).

### Dependency graph

```
T-Q3-01
  └─ T-Q3-02 → T-Q3-04 ─┐
  └─ T-Q3-03 ───────────┘
        T-Q3-04 → T-Q3-05 → T-Q3-06 → T-Q3-07 → T-Q3-08 → T-Q3-09
        T-Q3-04 ────────────────────────────────────────┘
(separate) T-Q3-GPU-* after T-Q3-09
```

---

## 7. Validation and acceptance ladder (to replicate)

1. Focused fixtures first (tokenizer, samplers, loader negatives) — model-free.
2. Tiny-check: deterministic tiny checkpoint + transformers oracle, top-1
   token equality (teacher forcing + greedy windows).
3. Real-model oracle: teacher forcing 100% top-1; greedy byte-identical on
   fixed prompts (md5-stamped baselines like V4's
   `068b2f5e84c113dbdbf134b34892d549`).
4. Paired runs for any quantized path (int8 vs BF16 token equality).
5. Smoke scripts with exit-code propagation and explicit config echo
   (`run-qwen3-smoke.sh`/`.ps1`).
6. Long-context: 4K → 32K → one 128K YaRN run; record command, flags,
   hardware, exit status, telemetry.
7. Repository baseline `make check`; CPU-only syntax sweep of touched units;
   amalgamation rules do not apply (Qwen3 is not part of the V4 amalgamation,
   but the same hygiene applies if `c/v4/` is ever touched).
8. The GPU phase adds the V4 ladder: `build-v4.ps1`-style helper, CUDA tests,
   sparse/dense smokes with telemetry, determinism evidence.

Acceptance criteria must be falsifiable and phrased exactly like the V4 plan
(exact inputs, exact outputs/md5, exact counters, no "works"/"optimized").

---

## 8. Process artifacts to replicate

- `docs/plans/qwen3-plan.md` (or `qwen3-cpu-plan.md` + later
  `qwen3-gpu-performance-plan.md`) — AGENTS.md task format.
- `currentstate.md` — facts only, updated per completed task with commit
  hashes; remove superseded claims.
- SDD for the GPU phase; design decisions recorded with evidence.
- `docs/AGENT_ENVIRONMENT.md` — checkpoint path (e.g.
  `C:/Models/ALL/other/qwen3-8b`), oracle deps, disk/RAM notes, smoke
  commands.
- `docs/qwen3.md` model doc; README roster; banner; CHANGELOG.
- One atomic commit per task; never commit unrelated dirty-tree changes (the
  tree currently holds uncommitted V4 CUDA work — see `git status`).

---

## 9. Risks and open questions

Technical (resolve by investigation, record evidence):
- **config vs tokenizer vocab mismatch** (151,936 vs 151,669): resolved by
  shape-driven loading; do not assert equality.
- **Released dense checkpoints disable SWA**: engine must still implement
  config-driven `layer_types` for family correctness, but acceptance runs
  full attention.
- **NFC normalizer gap in tok.h**: correctness item for a multilingual
  model; decide implement-vs-document.
- **min-p sampler missing from sample.h**: small addition; needed only for
  Qwen's recommended thinking settings.
- **`generation_config` vs model-card defaults**: engine defaults should
  match the model card table (§3.4); greedy remains available for validation
  only.
- **Qwen3-2507 variants** (Instruct/Thinking, 256K/1M context, changed
  template): different release; out of scope, but the family-capable engine
  should not hardcode template strings that would break them.
- **Checkpoint edits post-release** (HF discussion #25, config changes):
  trust shapes, not blog numbers (byte-arithmetic §4.2 is the referee).
- **Paper vs config discrepancies** (94 vs 96 layers for 235B, vocab sizes):
  config.json + shards are authoritative.

Non-technical — the Planner must ask the owner before contracting these:
1. **Acceptance model**: Qwen3-8B only, or family-wide (0.6B–32B)?
2. **CUDA phase**: in the same program of work or a separate plan, and on
   which branch (`main` precedent vs `dev` rule)?
3. **Quantization target**: BF16-f32 only, int8, or int4 for the first
   delivery? (Memory envelope differs 32.8 / 8.2 / 4.3 GiB.)
4. **Long-context target**: 32K native only, or 128K YaRN acceptance?
5. **Quality gate**: oracle top-1 token equality on which prompts; is
   byte-identical greedy output an acceptance gate (V4's contract) or only
   sampling-distribution checks?
6. **Dependency posture**: confirm `transformers>=4.51`/`torch`/`tokenizers`
   as oracle-only extras needs no new approval; any new format proposal must
   go through the FORMATS.md private-ordinal + owner-approval path.

---

## 10. Sources (all fetched/verified 2026-08-18)

- Qwen3 Technical Report, arXiv:2505.09388 (abstract + §2 Architecture,
  Tables 1–2).
- `Qwen/Qwen3-8B`: `config.json`, `generation_config.json`,
  `tokenizer_config.json`, `tokenizer.json` (inspected: 151,643 vocab,
  151,387 merges, cl100k-style Split regex, NFC, 26 added tokens),
  `model.safetensors.index.json` (weight map + `total_size`), README model
  card (best practices, thinking/non-thinking, YaRN).
- `Qwen/Qwen3-30B-A3B/config.json` (MoE reference).
- `transformers` v4.51.0: `configuration_qwen3.py`, `modeling_qwen3.py`
  (attention order, QK-Norm, SWA gating, RoPE, MLP, head).
- Repository: `AGENTS.md`, `docs/AGENT_ENVIRONMENT.md`, `docs/FORMATS.md`,
  `docs/deepseek-v4.md`, `docs/plans/deepseek-v4-gpu-performance-plan.md`,
  `currentstate.md`, `c/tok.h`, `c/sample.h`, `c/st.h`, `c/Makefile`,
  `c/coli`, `c/tools/make_deepseek_v4_{tiny,oracle}.py`, `c/olmoe.c`.