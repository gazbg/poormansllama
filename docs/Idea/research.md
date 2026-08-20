Here’s a compact, agent‑friendly spec you can hand to your Planner/Implementer. It assumes the goal is:

- A **C/C++ inference engine for Qwen3.8‑27B**,  
- Built on **ggml / llama.cpp backends**,  
- With **no custom kernels**, so you can focus on quantization, residency, and thinking‑mode behavior.

You can paste this as‑is into your repo’s docs (e.g. `docs/qwen38-27b-ggml.md`) or into an agent prompt.

***

## 1. Goal and scope

**Goal**

Implement a **text‑first C/C++ inference engine** for **Qwen3.8‑27B** that:

- Uses **ggml / llama.cpp** for all tensor ops and backends (CPU, CUDA, Vulkan).
- Loads Qwen3.8‑27B via **GGUF** (no safetensors loading logic in this engine).
- Supports:
  - Native **thinking mode** (`<think>…</think>\n\n`) and **non‑thinking mode**.
  - **Reasoning depth control** via `reasoning_effort` (`xhigh`, `medium`, `low`).
  - **Preserved thinking** (`preserve_thinking` true/false).
  - Up to **262K context**, with optional **YaRN to ~1M**.
- Exposes clean hooks for:
  - Quantization choice (Q2/Q3/Q4/Q5/Q6, IQ quants).
  - KV‑cache and long‑context policies.
  - Sampling parameters (temperature, top_p, top_k, min_p, presence_penalty).

**First implementation is text‑only** (no images/videos), but the design must not block later integration of the vision projector or MTP (multi‑token prediction) head. [lovableapp](https://lovableapp.org/blog/qwen3-8-27b)

***

## 2. Model facts for Qwen3.8‑27B (language side)

From the official README and independent spec summaries: [kingy](https://kingy.ai/blog/qwen3-8-27b-specs-benchmarks-local-hardware/)

- **Type:** Causal LM with vision encoder (VL model), but this engine targets the **language decoder** first. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md)
- **Parameters:** ~27B dense BF16 weights (~54–56 GB at full precision on disk). [orcarouter](https://www.orcarouter.ai/blog/qwen-3-8-27b-uncensored-gguf)
- **Hidden size:** 5120. [boxwrench](https://www.boxwrench.dev/posts/qwen38-27b/)
- **Layers:** 64 decoder layers. [kingy](https://kingy.ai/blog/qwen3-8-27b-specs-benchmarks-local-hardware/)
- **Hidden layout:**  
  - 16 repeats of: `3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)`.  
  - Total: **48 Gated DeltaNet linear‑attention layers + 16 full‑attention layers**. [lovableapp](https://lovableapp.org/blog/qwen3-8-27b)
- **Linear attention (Gated DeltaNet):**  
  - 48 value heads, 16 QK heads, head dim 128, convolution kernel size ≈ 4 (linear recurrent state, not KV cache). [boxwrench](https://www.boxwrench.dev/posts/qwen38-27b/)
- **Full attention (Gated Attention):**  
  - 24 query heads, 4 KV heads (GQA), head dim 256. [kingy](https://kingy.ai/blog/qwen3-8-27b-specs-benchmarks-local-hardware/)
  - RoPE dimension 64, `rope_theta ≈ 1e7`, `partial_rotary_factor ≈ 0.25`. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md)
- **FFN:** SwiGLU, intermediate dim ≈ 17 408. [lovableapp](https://lovableapp.org/blog/qwen3-8-27b)
- **Vocab / embeddings:**  
  - Token embedding and LM output dimension ≈ 248 320 (padded). [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md)
- **Context:**  
  - Native **262 144 tokens**. [boxwrench](https://www.boxwrench.dev/posts/qwen38-27b/)
  - Extensible toward **1 000 000 tokens** via **YaRN** (`rope_type="yarn"`, `factor=4.0`, `original_max_position_embeddings=262144`). [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Multi‑token prediction (MTP):** trained with multi‑step MTP for speculative decoding; MTP head is present in GGUF builds. [orcarouter](https://www.orcarouter.ai/blog/qwen-3-8-27b-uncensored-gguf)

**Critical KV‑cache detail:** Only the **16 full‑attention layers maintain a conventional KV cache**; Gated DeltaNet linear‑attention layers maintain internal state but not a standard per‑position K/V tensor. Your KV‑cache budgeting must reflect this (much less KV per token than a 64‑layer all‑attention transformer). [dev](https://dev.to/purpledoubled/run-qwen-38-27b-locally-real-gguf-sizes-the-kv-cache-trick-and-the-template-trap-114j)

***

## 3. Thinking mode, reasoning control, and chat semantics

From the official README and GGUF docs: [huggingface](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)

- **Thinking mode is on by default.**  
  - The model emits a `<think>\n…\n</think>\n\n` block (reasoning content) **before** the final answer. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md)
- **Non‑thinking (instruct) mode** disables this and returns a direct answer. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md)
- **Control flags:**
  - `enable_thinking` (bool):
    - `True`: thinking block + answer (default). [boxwrench](https://www.boxwrench.dev/posts/qwen38-27b/)
    - `False`: answer only (instruct/non‑thinking mode). [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md)
  - `reasoning_effort`: `"xhigh"` (default) / `"medium"` / `"low"` — controls reasoning depth and cost. [boxwrench](https://www.boxwrench.dev/posts/qwen38-27b/)
  - `preserve_thinking`:
    - `True` (default): keeps thinking blocks in history, so downstream turns see previous reasoning, improves agent continuity and KV cache reuse. [boxwrench](https://www.boxwrench.dev/posts/qwen38-27b/)
    - `False`: only retain thinking from the latest user message. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md)
- **Recommended sampling parameters:**
  - **Thinking mode:**  
    - `temperature = 1.0`, `top_p = 0.95`, `top_k = 20`, `min_p = 0.0`,  
    - `presence_penalty = 0.0`, `repetition_penalty = 1.0`. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B)
  - **Instruct / non‑thinking mode:**  
    - `temperature = 0.7`, `top_p = 0.80`, `top_k = 20`, `min_p = 0.0`,  
    - `presence_penalty = 1.5`, `repetition_penalty = 1.0`. [huggingface](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)

**Agent requirement:** The engine must:

- Build prompts/templates that correctly set `enable_thinking`, `reasoning_effort`, and `preserve_thinking`, and map them to GGUF/llama.cpp or your own chat‑template layer.
- Parse outputs into:
  - `reasoning_content` (everything inside `<think>…</think>`),
  - `answer_content` (the final assistant text after the think block),
  - And append them back to the conversation history according to `preserve_thinking`.

***

## 4. GGUF, llama.cpp, and backends

Use GGUF + llama.cpp/ggml to avoid kernel implementation.

### 4.1 GGUF sources and quantizations

Working GGUF packs exist and are used in the wild: [ofox](https://ofox.ai/blog/qwen-3-8-27b-run-locally-vram-gguf-2026/)

Typical sizes (text+vision+MTP):

- **BF16/F16 (reference):** ≈ 53–56 GB (plus ≈0.93 GB mmproj for vision). [ofox](https://ofox.ai/blog/qwen-3-8-27b-run-locally-vram-gguf-2026/)
- **Q8_0:** ≈ 29–31 GB. [ofox](https://ofox.ai/blog/qwen-3-8-27b-run-locally-vram-gguf-2026/)
- **Q6_K:** ≈ 22–25 GB. [orcarouter](https://www.orcarouter.ai/blog/qwen-3-8-27b-uncensored-gguf)
- **Q5_K (M/S):** ≈ 19–20 GB. [ofox](https://ofox.ai/blog/qwen-3-8-27b-run-locally-vram-gguf-2026/)
- **Q4_K (M/S/XL):** ≈ 16–18 GB (recommended default for 24 GB GPUs). [orcarouter](https://www.orcarouter.ai/blog/qwen-3-8-27b-gguf)
- **Q3_K / IQ3_M / IQ3_S:** ≈ 12–14.6 GB (for 16 GB GPUs). [kingy](https://kingy.ai/blog/qwen3-8-27b-best-quantization-gguf/)
- **Q2_K / IQ2_XXS:** ≈ 9–11 GB (for 12 GB GPUs, visible quality loss). [ollama](https://ollama.com/smtek/Qwen3.8-27B:Q4_K_XL)

**Engineering recommendation for first implementation:**

- If target GPU has **24 GB VRAM**: start with **Q4_K_M** or **Q4_K_XL** (~17–18 GB) as the baseline. [orcarouter](https://www.orcarouter.ai/blog/qwen-3-8-27b-gguf)
- If target GPU has **16 GB VRAM**: start with **Q3_K_M** or **IQ4_XS** (~13.5–15.3 GB), accept reduced context headroom. [kingy](https://kingy.ai/blog/qwen3-8-27b-best-quantization-gguf/)
- If target GPU has **12 GB VRAM**: only **IQ2_XXS/Q2_K** are feasible; treat this as an explicitly degraded mode. [ollama](https://ollama.com/smtek/Qwen3.8-27B:Q4_K_XL)

Your agent should **not** implement BF16/F16 loading or quantization itself; it should consume existing GGUFs and treat quantization choice as a configuration.

### 4.2 llama.cpp / ggml stack

The engine must:

- Use **llama.cpp** (or ggml) as the model runtime and kernel provider:
  - GGUF loader (text + optional `--mmproj` for vision). [dev](https://dev.to/purpledoubled/run-qwen-38-27b-locally-real-gguf-sizes-the-kv-cache-trick-and-the-template-trap-114j)
  - Backends: CPU, CUDA, Vulkan (Vulkan recommended for cross‑vendor and older cards). [deepwiki](https://deepwiki.com/ggml-org/llama.cpp/4.4-vulkan-backend)
  - KV‑cache implementation for the 16 full‑attention layers. [dev](https://dev.to/purpledoubled/run-qwen-38-27b-locally-real-gguf-sizes-the-kv-cache-trick-and-the-template-trap-114j)
  - MTP head handling when speculative decoding is enabled (e.g. `--spec-type draft-mtp`). [orcarouter](https://www.orcarouter.ai/blog/qwen-3-8-27b-uncensored-gguf)
- **Not** implement matmul, attention, RoPE, or norm kernels; those are supplied by ggml.

Typical llama.cpp server invocation for Qwen3.8‑27B GGUF: [orcarouter](https://www.orcarouter.ai/blog/qwen-3-8-27b-gguf)

```bash
llama-server \
  -m Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.8-27B-f16.gguf \   # optional, for vision
  --jinja \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --cache-type-k q8_0
```

Your agent should:

- **Embed llama.cpp** as a library (preferred) or call it as a sidecar process.
- Expose a clean interface from your engine to llama.cpp (e.g. `run_batch`, `run_chat`, `tokenize`, `decode_one_step`).

***

## 5. Engine responsibilities vs backend responsibilities

### 5.1 Backend (ggml / llama.cpp) responsibilities

The agent must treat these as **already solved**:

- GGUF loading (text weights, MTP head, mmproj for vision). [huggingface](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF)
- Execution of:
  - Gated DeltaNet linear‑attention blocks (recurrent state per position). [kingy](https://kingy.ai/blog/qwen3-8-27b-specs-benchmarks-local-hardware/)
  - Gated Attention full‑attention blocks (Q/K/V projections, RoPE, softmax, KV cache). [kingy](https://kingy.ai/blog/qwen3-8-27b-specs-benchmarks-local-hardware/)
  - SwiGLU FFN layers. [lovableapp](https://lovableapp.org/blog/qwen3-8-27b)
- Pointer‑level KV‑cache implementation for the 16 full‑attention layers. [dev](https://dev.to/purpledoubled/run-qwen-38-27b-locally-real-gguf-sizes-the-kv-cache-trick-and-the-template-trap-114j)
- MTP speculative decoding when requested (draft‑head next‑token prediction). [orcarouter](https://www.orcarouter.ai/blog/qwen-3-8-27b-uncensored-gguf)
- Core sampling primitives (temperature, top_p, top_k).

### 5.2 Your engine’s responsibilities

The agent must implement:

1. **Configuration and environment layer**
   - Read config:
     - Which GGUF file(s) to use.
     - Quantization tier (Q2/Q3/Q4/Q5/Q6, IQ variants).
     - Target context window (≤262K, optional YaRN extension).
   - Hardware constraints (GPU VRAM, CPU RAM, KV cache limits).

2. **Tokenizer & chat template**
   - Either reuse llama.cpp’s tokenizer or HF tokenizer; no need to implement BPE from scratch.
   - Build chat templates that:
     - Set `enable_thinking`, `reasoning_effort`, `preserve_thinking` according to user/config.
     - Default thinking mode and recommended sampling parameters. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B)
     - Provide non‑thinking mode when requested, with instruct sampling params. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B)
   - Correctly parse `<think>…</think>\n\n` from outputs into reasoning vs answer parts. [boxwrench](https://www.boxwrench.dev/posts/qwen38-27b/)

3. **Sampling policy**
   - Implement a sampler that:
     - Supports temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty.
     - Uses Qwen’s recommended defaults per mode. [huggingface](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
   - Either wrap llama.cpp’s samplers or run your sampler over logits and then feed chosen tokens back to llama.cpp.

4. **Session, KV‑cache, and long‑context management**
   - Manage sessions:
     - Conversation history (including or excluding preserved thinking).
     - Token counts per session vs context budget.
   - KV‑cache policy:
     - Recognize that only 16 layers maintain KV; linear layers keep cheaper recurrent state. [dev](https://dev.to/purpledoubled/run-qwen-38-27b-locally-real-gguf-sizes-the-kv-cache-trick-and-the-template-trap-114j)
     - Decide context length per deployment (e.g. 32K, 96K, 128K) based on VRAM and quantization tier, using guidance tables. [ollama](https://ollama.com/smtek/Qwen3.8-27B:Q3_K_XL)
   - Optional YaRN:
     - For contexts >262K, modify `text_config.rope_parameters` as documented (mrope, rope_type `"yarn"`, `factor=4.0`, `original_max_position_embeddings=262144`). [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B)

5. **Vision and MTP (later phases)**
   - Vision:
     - Load `mmproj-…gguf` when image/video input is requested. [ollama](https://ollama.com/smtek/Qwen3.8-27B:Q3_K_XL)
     - Respect video preprocessor config and recommended `longest_edge` for long videos (e.g. `469,762,048` for 224K video tokens). [huggingface](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
   - MTP:
     - Enable MTP draft head when speculative decoding is configured (e.g. `--spec-type draft-mtp`), and wire it into your sampling loop. [orcarouter](https://www.orcarouter.ai/blog/qwen-3-8-27b-uncensored-gguf)

***

## 6. Recommended implementation phases for the agent

Use this as the task outline:

### Phase 1 — Text‑only, CPU/GPU reference

- Integrate llama.cpp as a library or sidecar.
- Implement:
  - Config loading (GGUF path, quant tier, context window).
  - Tokenization via llama.cpp or HF tokenizer.
  - Thinking/non‑thinking chat templates and flags.
  - Sampling with Qwen’s recommended parameters.
- Target:
  - **Q4_K_M** on a **24 GB GPU** as baseline. [kingy](https://kingy.ai/blog/qwen3-8-27b-best-quantization-gguf/)
  - **32K–96K context** initially, measured KV‑cache usage using approximate KV tables (e.g. 0.5 GiB at 8K, 2 GiB at 32K, 4 GiB at 65K, 8 GiB at 131K, 16 GiB at 262K for BF16 full‑attention KV). [kingy](https://kingy.ai/blog/qwen3-8-27b-best-quantization-gguf/)

### Phase 2 — Long‑context and reasoning control

- Implement explicit session and KV‑cache policies:
  - Reasoning content max length vs final answer max length, e.g. 262K vs 131K. [huggingface](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
  - Token budgeting per request and per session.
- Expose `reasoning_effort` and `preserve_thinking`:
  - `xhigh`: deep reasoning.
  - `medium` / `low`: shallower reasoning, but note Qwen’s warning that lower effort may increase overall task latency via retries. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md)

### Phase 3 — Quantization permutations

- Provide configuration to select GGUF quant tier:
  - Q2/Q3/Q4/Q5/Q6, IQ variants (e.g. IQ3_M, IQ4_XS). [orcarouter](https://www.orcarouter.ai/blog/qwen-3-8-27b-gguf)
- Tie quant tier to:
  - Max context (VRAM headroom).
  - Target hardware (12/16/24 GB).
- Add simple benchmarks:
  - Tokens/s for prefill and decode.
  - KV‑cache memory vs context.

### Phase 4 — Vision and MTP

- Add optional `mmproj` loading and image/video input paths.
- Respect long video guidance (`longest_edge` in preprocessor config). [huggingface](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- Enable MTP as an option for faster decoding, using the built‑in speculative head. [orcarouter](https://www.orcarouter.ai/blog/qwen-3-8-27b-uncensored-gguf)

***

## 7. Correctness and validation expectations

For the agent, define correctness like this:

- Use Hugging Face `Qwen/Qwen3.8-27B` as reference for:
  - Teacher‑forcing logits on a small set of prompts.
  - Token‑level greedy decoding on short prompts (for sanity). [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B)
- For GGUF runs:
  - Accept small differences vs HF due to quantization, but no structural errors (e.g. mis‑parsed `<think>` blocks, wrong chat template).
- Validate:
  - Thinking/non‑thinking switching.
  - `reasoning_effort` effect on reasoning length (even if not strictly deterministic).
  - `preserve_thinking` effect on context reuse.

***

If you hand this document to your agent, the key instructions are:

- **Do not implement kernels.** Use ggml/llama.cpp backends and GGUF packs for Qwen3.8‑27B.
- **Focus on engine‑level concerns:** configuration, chat templates, thinking control, quantization selection, KV‑cache and context management.
- **Start with text‑only Q4_K_M on 24 GB GPUs, 32K–96K context.** Add longer context and multimodal/MTP in later phases.

This gives them enough architecture detail, integration boundaries, and task structure to implement Qwen3.8‑27B smoothly without getting lost in low‑level CUDA/Vulkan work.