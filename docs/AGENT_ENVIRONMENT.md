# Agent Environment and Dependency Guide

Status: verified 2026-08-18 by the environment-inventory task T-Q38LS-01.

This file is the canonical source of environment facts for this workspace. Do
not invent tool or model facts in source code or plans; record them here first.
Update this file when the machine, toolchain, or model artifact changes.

## Semantic-QA smoke tier (T-GPLR-07)

The runner-side semantic-QA tooling lives in `tools/semantic-qa/`:
`gen-needle-prompt.py` (deterministic needle prompts, exact token counts) and
`check-answer.py` (programmatic grader, self-test included). The 512-token
anchor needs `QWEN36_MODEL` set to the current Qwen3.6-35B-A3B GGUF path and
every GPU rung wrapped in `tools/with-gpu-lock.ps1`. Runner invocation details
and known caveats (TG budget >= ~65 tokens for the 5-item answer, single-ubatch
PP workaround for the tail-ubatch gap, checker parses mixed runner logs) are
recorded in `docs/currentstate.md` under T-GPLR-07 and in the T-GPLR-07 task
contract in `docs/plans/generic-prefill-layer-residency-plan.md`.

## Host

| Item | Value (observed) |
| --- | --- |
| OS | Microsoft Windows 11 Pro, build 10.0.26200.9168 |
| CPU | AMD Ryzen 9 9955HX 16-Core Processor |
| RAM | 66,234,945,536 bytes (~61.7 GiB) |
| Shell for tooling | Git Bash (POSIX syntax; never cmd.exe syntax) |
| Disk | C: on an MSI M461 2TB NVMe SSD (BusType NVMe, ~1863 GB); 432 GiB free at inventory time. Drive model observed 2026-09-07 via Get-PhysicalDisk. |
| Disk sequential read (measured 2026-09-07, 21.39 GB Qwen3.6 GGUF, 64 MiB reads) | 7,745 MB/s buffered (file-cache warm); 2,728 MB/s unbuffered qd1 (FILE_FLAG_NO_BUFFERING, cache bypassed). The ~52 MB/s effective page-in seen in 262K PP runs is mmap fault-path overhead, not a device limit. |
| Windows token privileges (probed 2026-09-09 via C:\Windows\System32\whoami.exe /priv, <host>\<user>, session 1, non-elevated) | NO SeLockMemoryPrivilege (VirtualLock will fail ERROR_ACCESS_DENIED); SeIncreaseWorkingSetPrivilege present but Disabled; no SeIncreaseQuotaPrivilege. Consequence: T-GPLR-09's working-set-raise + VirtualLock residency path cannot run under the current token without a Local Security Policy change ("Lock pages in memory"). |

## GPU and CUDA

| Item | Value (observed) |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 (WDDM), Bus 00000000:01:00.0 |
| VRAM | 24,463 MiB total |
| Driver / KMD | 610.74 |
| CUDA UMD | 13.3 |
| Toolkit (nvcc) | CUDA 13.2, V13.2.51, release 13.2 |
| Compute mode | Default (shared); a foreign process may occupy VRAM/cycles |

GPU sharing: at inventory time a foreign process `deepseek_v4.exe` (not owned
by this workspace) held ~12,359 MiB VRAM and ~55% GPU util. The layer-streaming
runner must treat the GPU as shared and report `run_start` facts per run; it
must never claim exclusive GPU ownership. VRAM measurements in this guide
therefore use the physical total, and per-run telemetry records observed free
VRAM at `run_start`.

## Toolchain

| Item | Value (observed) |
| --- | --- |
| CMake | 4.2.3 |
| C/C++ compiler | Microsoft Visual Studio Build Tools (vcvars64), VS 2022 and VS 18 both installed; invoked through CMake's VS generator |
| Build generator | CMake default MSVC generator (VS). Use forward slashes in `-D` tool-path values. |
| ninja | not installed; do not rely on it |
| Python | 3.13.14 (no numpy; GGUF metadata was read with a dependency-free parser) |
| gcc / cl on PATH | not present in Git Bash; do not call them directly |

## Model Artifact

| Item | Value (observed) |
| --- | --- |
| Path | `C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf` |
| Byte size | 25,924,152,384 (~25.92 GB) |
| GGUF version | 3 |
| Quantization version | 2 |
| `general.architecture` | `qwen35` |
| `general.name` | Qwen3.8-27B |
| `general.file_type` | 14. In this llama.cpp revision `llama_ftype_name(14)` prints "Q4_K - Small" (`LLAMA_FTYPE_MOSTLY_Q4_K_S`); this is a legacy label and does NOT describe the actual quant mix. |
| Tensor count | 866 |

Actual per-tensor quants (ggml enum of this tree: F32=0, Q8_0=8, Q5_K=13,
Q6_K=14): `output.weight` Q8_0, `token_embd.weight` Q6_K, `blk.*.attn_qkv`
and `blk.*.ffn_down` Q8_0, `blk.*.ffn_gate`/`ffn_up` Q6_K,
`blk.*.ssm_alpha`/`ssm_beta` Q5_K, 1-D tensors (`*_norm`, `ssm_a`, `ssm_dt`)
F32, `ssm_conv1d` F32. This is the "UD Q6_K_XL" (Unsloth Dynamic) scheme:
a per-tensor dynamic mix, not a single uniform Q6_K.

### Qwen3.5 hybrid hyperparameters (from GGUF header)

| Key | Value |
| --- | --- |
| `qwen35.block_count` | 65 (64 trunk + 1 MTP/nextn layer) |
| `qwen35.embedding_length` | 5120 |
| `qwen35.context_length` | 262144 |
| `qwen35.attention.head_count` | 24 |
| `qwen35.attention.head_count_kv` | 4 |
| `qwen35.attention.key_length` / `value_length` | 256 / 256 |
| `qwen35.attention.layer_norm_rms_epsilon` | 1e-6 |
| `qwen35.ssm.conv_kernel` | 4 |
| `qwen35.ssm.inner_size` | 6144 |
| `qwen35.ssm.state_size` | 128 |
| `qwen35.ssm.time_step_rank` | 48 |
| `qwen35.ssm.group_count` | 16 |
| `qwen35.full_attention_interval` | 4 |
| `qwen35.rope.dimension_count` | 64 |
| `tokenizer.ggml.model` | gpt2 |
| recurrent layers | not stored; loader default: layer `il` recurrent when `(il+1) % 4 != 0` (layers 0,1,2 recurrent; 3 full-attn; repeats every 4) |

### Per-layer package byte sizes (from GGUF tensor offsets, dependency-free parse; byte sizes are ground truth regardless of quant labels)

| Item | Bytes |
| --- | --- |
| Static weights (`token_embd`, `output_norm`, `output`) | 2,393,825,280 (~2.394 GB) |
| Largest recurrent layer package (e.g. layer 0) | 364,022,656 (~364.0 MB) |
| Largest full-attention layer package (e.g. layer 3) | 352,380,928 (~352.4 MB) |
| Four-slot window (3 recurrent + 1 full-attn) | ~1.444 GB |
| MTP layer (`blk.64`) | excluded from the streaming window (out of scope) |

Recurrent layer package = 14 tensors (`attn_gate`, `attn_norm`, `attn_qkv`,
`ffn_down`, `ffn_gate`, `ffn_up`, `post_attention_norm`, `ssm_a`, `ssm_alpha`,
`ssm_beta`, `ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out`). Full-attention
layer package = 11 tensors (`attn_k`, `attn_k_norm`, `attn_norm`,
`attn_output`, `attn_q`, `attn_q_norm`, `attn_v`, `ffn_down`, `ffn_gate`,
`ffn_up`, `post_attention_norm`).

## Capacity Preflight Inputs (recorded, not a VRAM-fit claim)

Physical VRAM 24,463 MiB. A minimal session budget for the four-slot
experiment is approximately: static 2.394 GB + slots 1.444 GB + KV cache
(16 full-attn layers x 4 KV heads x 256 dim x 2 bytes x 2 tensors x n_ctx)
+ Gated DeltaNet recurrent state (~0.16 GB) + graph/compute work buffers.
Whether the exact runner allocation fits is measured at `run_start` by the
runner's preflight; this table only fixes the input sizes. If a foreign
process occupies VRAM, preflight reports required vs available and aborts
loudly (no CPU fallback).

## Verified Qwen3.6 MoE Acceptance Artifact (T-GPLR-00, 2026-09-04)

Inventory produced by `llama-layer-streaming.exe --inspect-model` (host-only:
returns before `llama_backend_init()`, no CUDA device enumeration, GGUF parsed
with `no_alloc`; no tensor data read). Full JSON:
`build/llama-cuda/q36-moe-inventory.json`.

| Item | Value (observed) |
| --- | --- |
| Path | `C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf` |
| Byte size | 21,388,319,008 (~21.39 GB) |
| GGUF version / alignment / data offset | 3 / 32 / 10,991,392 |
| `general.architecture` | `qwen35moe` (vendored support present: `LLM_ARCH_QWEN35MOE` in `src/llama-arch.cpp`, `src/models/qwen35moe.cpp`, 40-layer -> `LLM_TYPE_35B_A3B`) |
| `general.name` | Qwen3.6-35B-A3B |
| Tensor count | 753 |
| `qwen35moe.block_count` | 41 (40 trunk + 1 MTP/nextn block) |
| `qwen35moe.nextn_predict_layers` | 1 |
| `qwen35moe.expert_count` | 256 |
| `qwen35moe.expert_used_count` | 8 |
| `qwen35moe.embedding_length` | 2048 |
| `qwen35moe.attention.head_count_kv` | 2 |
| `qwen35moe.attention.key_length` / `value_length` | 256 / 256 |
| `qwen35moe.full_attention_interval` | 4 |
| `qwen35moe.ssm.*` | conv 4, state 128, inner 4096, group 16, time_step_rank 32 |
| `qwen35moe.context_length` | 262144 |
| Tensor types (ggml `ggml_type_name`) | f32 368, q8_0 259, q4_K 120, q6_K 4, bf16 2 |
| Total tensor-data bytes | 21,377,327,616 |

Byte classes (every tensor assigned exactly once; completeness enforced):

| Class | Bytes |
| --- | --- |
| static (`token_embd` 540,344,320 + `output` 417,177,600 + `output_norm` 8,192) | 957,530,112 |
| per-trunk-layer non-expert total | 1,597,483,520 |
| - attention (attn_q/k/v/output + q/k norms) | 289,689,600 |
| - recurrent (attn_qkv, attn_gate, ssm_*) | 1,089,231,360 |
| - router (`ffn_gate_inp`) | 83,886,080 |
| - shared expert (`ffn_*_shexp`, `ffn_gate_inp_shexp`) | 134,021,120 |
| - dense ffn | 0 |
| - layer norms (`attn_norm`, `post_attention_norm`) | 655,360 |
| routed-expert banks (40 trunk layers x 3 `*_exps` tensors) | 18,327,011,328 |
| MTP block non-nextn tensors (blk.40 weights incl. its expert bank) | 486,365,184 |
| MTP nextn tensors (blk.40.nextn.*) | 8,937,472 |

Per-layer expert-bank packages: 37 layers at 452,984,832 B; layers 34, 38, 39
at 522,190,848 B (largest = 522,190,848 B, layer 34; their `ffn_down_exps`
carry the q6_K tensors 220,200,960 B each vs the q4_K 150,994,944 B used by
the other layers - Unsloth dynamic per-tensor mix). Full dense-style layer
package max = 563,962,624 B.

Runtime byte classes at context capacity 262,144 (derived from metadata; graph/
input, activation, workspace and CUDA-context bytes remain unknown until
runtime preflight):

| Class | Bytes |
| --- | --- |
| Q8_0 K/V (10 attention layers x 2 x row(512, q8_0)=544 x 262,144) | 2,852,126,720 |
| Gated DeltaNet recurrent state (30 layers x (r 24,576 + s 524,288) x F32) | 65,863,680 |
| full model + KV + state | 24,295,318,016 |
| bounded 1-slot (static + per-layer + largest expert bank + KV + state) | 5,995,194,880 |
| bounded 2-slot | 6,517,385,728 |

Capacity finding (measured 2026-09-04): physical VRAM 25,650,855,936 B
(`nvidia-smi memory.total`, RTX 5090). `full_model_plus_runtime` (24,295,318,016 B)
is BELOW the physical total by ~1.36 GB - raw weight+KV+state bytes alone would
fit. Observed free VRAM at `run_start` was 22,922,919,936 B (2026-08-21), and
CUDA context, activations, graphs, workspaces, and the always-resident static +
per-layer non-expert classes are not yet in the above sum, so real full
residency is not expected to fit; the inventory nevertheless reports
`require_sub_full_budget: true`. T-GPLR-04 must therefore enforce and report an
explicit residency budget below the full-model size rather than claim an
accidental out-of-core result.


## Vendored Dependency

| Item | Value |
| --- | --- |
| Tree | `third_party/llama.cpp` (private fork; no upstream-facing work) |
| HEAD at inventory | `04b569142da23d91beca090a99098d592d3f3c80` |
| Qwen3.5 support | present: `LLM_ARCH_QWEN35` in `src/llama-arch.cpp`, `src/models/qwen35.cpp`, delta-net base in `src/models/delta-net-base.cpp`, `src/llama-memory-recurrent.h` |

## Canonical Build Commands

All commands run from the workspace root in Git Bash.

```bash
cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON
cmake --build build/llama-cuda --config Release --target llama-cli
```

Build output lives under `build/` and `out/`, both ignored by `.gitignore`.
Do not use Make or DeepSeek-project build commands for this experiment.

## Dependency Policy

- No new external runtime dependency has been added or requested.
- Python `numpy` is not installed. GGUF metadata inspection used a
  dependency-free parser; do not install numpy unless a task contract
  explicitly requires it and the owner approves.
- Any new library, install, vendoring, include-path, or package-metadata
  change requires owner approval before it happens.
