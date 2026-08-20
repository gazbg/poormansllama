I found the official Qwen3.8-27B documentation and architecture details. The main reference is the [official Qwen3.8-27B model card on Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B). It contains the model overview, architecture, benchmarks, Transformers files, inference examples, multimodal usage, and long-context configuration. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B)

## Architecture

| Component | Specification |
|---|---|
| Model type | Dense causal language model with a vision encoder |
| Language-model parameters | 27B |
| Hidden dimension | 5,120 |
| Layers | 64 |
| Vocabulary / embeddings | 248,320 padded tokens |
| Layer organization | 16 × `[3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)]` |
| Linear-attention layers | 48 Gated DeltaNet layers |
| Full-attention layers | 16 Gated Attention layers |
| FFN intermediate size | 17,408 |
| RoPE dimension | 64 |
| MTP | Multi-Token Prediction trained over multiple steps |
| Native context | 262,144 tokens |
| Extended context | Up to 1,000,000 tokens with YaRN |

The architecture is therefore a **hybrid linear-attention/full-attention model** rather than a conventional Transformer using full attention in every layer. Three Gated DeltaNet blocks are followed by one Gated Attention block in each repeated group. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B)

## Attention details

### Gated DeltaNet

- 48 value heads.
- 16 query/key heads.
- Head dimension: 128.
- Provides the majority of the model’s linear-attention layers.
- Reduces the cost of processing long sequences compared with full quadratic attention.

### Gated Attention

- 24 query heads.
- 4 key/value heads.
- Head dimension: 256.
- This is grouped-query attention with a \(24:4\) query-to-KV head ratio.
- Rotary position encoding is applied to a 64-dimensional slice.

The SGLang documentation gives the same breakdown and describes Qwen3.8-27B as a dense hybrid Gated DeltaNet vision-language model. [docs.sglang](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)

## Multimodal capabilities

Qwen3.8-27B is not text-only. It includes a native vision encoder and supports:

- Text input.
- Image understanding.
- Video understanding.
- Documents and diagrams.
- Long video analysis.
- Multimodal agent and computer-use tasks.

The model card describes support for images and videos, including hour-scale video understanding. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B)

## Thinking and inference controls

Thinking is enabled by default. The official model card documents:

- `reasoning_effort="xhigh"` — maximum reasoning depth and the default.
- `reasoning_effort="medium"` — balance between speed and reasoning.
- `reasoning_effort="low"` — lower-cost reasoning.
- `enable_thinking=False` — disables thinking for direct responses.
- `preserve_thinking=True` — retains historical thinking blocks across turns.

Suggested sampling settings are:

```text
Thinking:
temperature=1.0
top_p=0.95
top_k=20
presence_penalty=0.0
repetition_penalty=1.0

Non-thinking:
temperature=0.7
top_p=0.80
top_k=20
presence_penalty=1.5
repetition_penalty=1.0
```

The model card provides OpenAI-compatible examples for vLLM, SGLang, and other serving backends. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B)

## Long-context configuration

The native context length is 262,144 tokens. For contexts beyond that, the model card documents static YaRN scaling up to approximately 1M tokens.

The documented YaRN settings include:

```json
{
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "rope_type": "yarn",
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "factor": 4.0,
  "original_max_position_embeddings": 262144
}
```

For a 1M-token context, the official examples use:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve Qwen/Qwen3.8-27B \
  --max-model-len 1000000 \
  --hf-overrides '{
    "text_config": {
      "rope_parameters": {
        "mrope_interleaved": true,
        "mrope_section": [11, 11, 10],
        "rope_type": "yarn",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "factor": 4.0,
        "original_max_position_embeddings": 262144
      }
    }
  }'
```

The documentation warns that static YaRN scaling can reduce performance on shorter contexts, so it should generally only be enabled when long-context processing is actually required. [huggingface](https://huggingface.co/Qwen/Qwen3.8-27B)

## Serving documentation

For deployment-specific instructions, see:

- [Official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [SGLang Qwen3.8-27B documentation](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- [FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [Qwen Hugging Face collection](https://huggingface.co/Qwen/collections)

The SGLang page focuses on deployment and lists BF16, FP8, and NVFP4 variants, along with supported parallelism and hardware configurations. [docs.sglang](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)

### Important distinction

The official checkpoint is named **Qwen3.8-27B**, not “Qwen 3.8 27B” as a separate architecture family. It is a dense 27B model with a hybrid Gated DeltaNet/Gated Attention backbone; it is not an MoE model.