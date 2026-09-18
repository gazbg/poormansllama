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

## lean-ctx MCP shell environment (observed 2026-09-18; root cause and fix
closed in addenda C-E)

The lean-ctx MCP server's `ctx_shell` tool runs PowerShell 7 with a **stripped**
child environment: `PATHEXT` is absent (PowerShell reports only `.CPL`, the
suffix it appends itself), `COMSPEC` is empty, and user-level registry variables
(`OLLAMA_HOST`, `OneDrive`, `LEAN_CTX_PROFILE`, `HF_TOKEN`, ...) are not visible
to it. `PATH` and `TEMP` do survive. Both `PATHEXT` and `COMSPEC` are set
correctly at the user and machine level (addendum A), so they are dropped on the
way into the MCP child rather than missing from Windows. Registry edits cannot
reach a running MCP child, and a relaunch does not restore them (addendum B).

Consequence in that pre-fix state: extension-less command names did not resolve
inside `ctx_shell`. `git`, `python`, `cmake` and `node` all failed with "is not
recognized as a name of a cmdlet, function, script file, or executable program".
PowerShell cmdlets, `ctx_read` / `ctx_search` / `ctx_tree`, and the lean-ctx CLI
path (`lean-ctx -c "git status"`) were unaffected — the CLI inherits the normal
shell environment.

Workaround used until 2026-09-18 09:29 (**retired as of addendum E**): pass the
two missing variables on every `ctx_shell` call. The payload did not persist
between calls, and there is no `.lean-ctx.toml` knob for it (the config schema
has no shell-environment key).

```json
"env": {
  "PATHEXT": ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.PY;.PYW;.CPL",
  "COMSPEC": "C:\\Windows\\system32\\cmd.exe"
}
```

Payload mechanics (probed 2026-09-18): the payload is merged into the shell
process environment for that call only. Arbitrary new names are forwarded
(`FOO=bar`, `OLLAMA_HOST=http://probe.invalid`, `XYZ=1` all arrived), so it is
not a name allowlist, but the `LEAN_CTX_*` namespace is stripped (`LEAN_CTX_TEST`
never arrived), which means the payload cannot set lean-ctx's own configuration
variables. Any `PATHEXT` value arrives verbatim with `;.CPL` appended by
PowerShell (`.ZZZ` -> `.ZZZ;.CPL`), so a bare `PATHEXT=[.CPL]` reading means the
inherited value was empty, not that something set it to `.CPL`. Setting the
variable in-shell does not work either: a `$env:PATHEXT = ...` assignment is
refused as a mis-split command fragment (lean-ctx splits the line at `;` and
gates the fragments), and `Set-Item` is refused as a destructive verb.

Related lean-ctx shell-policy facts observed in the same session: commands whose
first token is `$env:VAR` and commands starting with a quoted absolute
executable path are rejected by the shell allowlist ("not in the shell
allowlist ... the entire command was rejected"), `git.exe` is not allowlisted
even though `git` is, and inline interpreter evaluation (`python -c ...`) is
blocked by policy. Relaunching the client does not restore `PATHEXT`: the
falsifier was run on 2026-09-18 and failed to falsify (addendum B below
supersedes the bash-inheritance guess this passage originally made). The
per-call `env` payload is the permanent route, not a temporary one.

Addendum A (2026-09-18, registry change): the owner set both values at the
**user** level (`PATHEXT(User)` = `.COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.PY;.PYW;.CPL`,
`COMSPEC(User)` = `C:\Windows\system32\cmd.exe`); machine-level `PATHEXT` and
`COMSPEC` were already correct. Re-checked in place: `ctx_shell` still reports
`PATHEXT=[.CPL]`, `COMSPEC=[]` and `git` still does not resolve, because an MCP
child's environment is fixed when the process is spawned -- a registry write
cannot reach a running session.

**Addendum B -- decisive test run and closed (2026-09-18 09:14).** The relaunch
happened and was Windows-native: `explorer.exe` (10816) -> `Freebuff.exe`
(11588, started 09:09:48) -> `bun.exe` (13460) -> `lean-ctx.exe` (13984, the MCP
server, 09:12:29) -> `pwsh`. The registry fix was recorded at 09:09:16 (commit
`e2139b3`), so the running app post-dates it. Probe:

```powershell
Write-Output ('PATHEXT=[' + $env:PATHEXT + ']'); git --version; python --version; cmake --version | Select-Object -First 1
```

Result: `PATHEXT=[.CPL]`, `COMSPEC=[]`, and `git`, `python`, `cmake` all
unresolved -- the falsifier failed to falsify, so the MCP-server launch path
drops these variables. Localization probes in the same session:

| Probe | Reading |
| --- | --- |
| MCP child env (pwsh) | `BASH_ENV`, `MSYSTEM`, `SHELL`, `TERM` all empty; `PATH` registry-shaped (Python313, CUDA 13.2, Vulkan SDK, Azure CLI; `C:\Program Files\Git\cmd` listed twice; 2 empty entries); `PATHEXT` 4 bytes = `.CPL`; `COMSPEC` empty |
| App's own shell tool (`run_terminal_command`, Git Bash) | `PATHEXT` full, `COMSPEC` set, `OLLAMA_HOST=127.0.0.1:11434` -- the app process environment is healthy |
| lean-ctx CLI run from that shell (`lean-ctx -c ...`) | `PATHEXT` full, `COMSPEC` set, `OLLAMA_HOST=127.0.0.1:11434`, `git` resolves |

Verdict: the strip is specific to the app's MCP-server spawn -- the app's own
shell tool and a CLI lean-ctx started from it both inherit the registry-merged
environment -- so it is not bash heritage, not a missing registry value, and not
fixable by a relaunch. Superseded reading: the original "bash-flavored `PATH`
with duplicated msys2 dirs and `BASH_ENV`" note came from the pre-relaunch
session; the post-relaunch MCP child carries no bash markers at all.

Root cause (2026-09-18, read from the desktop app's bundled code, not guessed):
the app spawns MCP servers through the MCP TypeScript SDK's
`StdioClientTransport`, whose Windows `DEFAULT_INHERITED_ENV_VARS` allowlist is
exactly `APPDATA`, `HOMEDRIVE`, `HOMEPATH`, `LOCALAPPDATA`, `PATH`,
`PROCESSOR_ARCHITECTURE`, `SYSTEMDRIVE`, `SYSTEMROOT`, `TEMP`, `USERNAME`,
`USERPROFILE`, `PROGRAMFILES`. `PATHEXT` and `COMSPEC` are not on it, which is
why precisely that set survives and every user-level variable does not. The spawn
is `{ ...getDefaultEnvironment(), ...serverParams.env }`, so a per-server `env`
map in the registration is layered **on top of** the allowlist.

Registration and fix route: the app's global MCP registry is
`~/.agents/mcp.json` (read by `McpConfigStore`; the app's own "Connectors" flow
writes the same file). Its stdio entry schema is
`{ type: "stdio", command, args, cwd, env: Record<string,string> }`, and the
`lean-ctx` entry already carries `"env": {}`, so adding `PATHEXT` and `COMSPEC`
there is *expected* to fix `ctx_shell` without a per-call payload -- expected, not
verified: it needs a server restart plus re-approval before it can be observed.
Values must be literal; `$NAME` references resolve only `HOME`, `PATH`, `SHELL`,
`USER`, `LANG`, `TMPDIR` (or a stored credential binding), so `"$PATHEXT"` is
rejected as a missing binding. Caveat: `launchHash` covers the sorted `env` *key
names*, so adding keys invalidates the approval recorded in
`~/.freebuff/mcp.json` and the server must be re-approved in the app (including
tool re-approval) before it runs; later changes to a key's *value* do not.
The per-project `<project>/.freebuff/settings.json` is **not** an MCP registry --
its only recognized key is `startupScript`.

Applied (2026-09-18, owner-directed): the `lean-ctx` entry in
`~/.agents/mcp.json` now carries

```json
"env": {
  "PATHEXT": ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.PY;.PYW;.CPL",
  "COMSPEC": "C:\\Windows\\system32\\cmd.exe"
}
```

Backup: `~/.agents/mcp.json.bak-20260918`; only that one block differs, the
other four servers are untouched, and the file still parses as JSON with LF
endings. The edit invalidates the stored approval: the old launch hash
`sha256:134d74c75cbb10e6f3cf95355390d5d89a22fde61b628cd76ca88f08e1e1b804`
reproduces the **pre-edit** fingerprint exactly (empty `envNames`, `cwd: null`)
when recomputed independently from the bundled code, and the new hash is
`sha256:08805d91db7a492ac5b69fd60ef5ae2406c183e830433c41d3036288846f6dbe`.
So `lean-ctx` must be re-approved in the app (tool list included) and the MCP
server restarted before `ctx_shell` can be re-tested. Verification: run the
addendum B probe **without** an `env` payload -- full `PATHEXT`, non-empty
`COMSPEC` and a resolving `git` mean the per-call payload can be retired. Until
that reading exists, the payload stays the supported route.

**Addendum C -- re-approval did not take effect; the no-payload probe is still
unrun (2026-09-18 09:24).** The restarted Freebuff session (`bun.exe` PID 22144,
parent `Freebuff.exe` 20516, started 09:21:06 -- 75 s *after* the 09:19:51 config
edit) still exposes no `ctx_shell`. The blocker is upstream of the environment
question: the agent's MCP client does not know the server at all (`No MCP server
named "lean-ctx" is configured`; only `context7` and `wigolo` tools are
reachable) because the edited entry sits in the app's `awaiting_launch_approval`
state, so it is never spawned.

| Observation (2026-09-18) | Reading |
| --- | --- |
| `~/.freebuff/mcp.json` (approval sidecar) `lean-ctx` record | `enabled: true`, `launchHash: sha256:134d74c7...`, `manifestHash: sha256:90e2b18...`, `generation: 2` |
| That file's mtime | 2026-09-10 18:22:36 -- never written on 2026-09-18, so no approval was recorded today (`context7` and `wigolo` stand at `generation: 3`) |
| `~/.agents/mcp.json` mtime | 2026-09-18 09:19:51 (the env-block edit; `mcp.json.bak-20260918` is identical apart from that block) |
| lean-ctx processes | one only: `lean-ctx.exe` PID 17548, `proxy start --port=4444`, started 08:59:40, parent 19260 -- not an app-spawned MCP server; no MCP child exists |

All four launch hashes reproduce in the app's own runtime
(`<app>/resources/bun/bun.exe`, bun 1.4.2, using the shipped `canonical4` and
`launchHash` from `<app>/resources/orchestrator/orchestrator.js`):

```text
context7           (envNames [])                              -> sha256:7298f7c2... (equals stored)
wigolo             (envNames [WIGOLO_LLM_MODEL, WIGOLO_LLM_PROVIDER]) -> sha256:fe238bef... (equals stored)
lean-ctx pre-edit  (envNames [])                              -> sha256:134d74c7... (equals stored approval)
lean-ctx as edited (envNames [COMSPEC, PATHEXT])              -> sha256:08805d91... (live config)
```

The stored approval is therefore exactly the **pre-edit** fingerprint while the
live config hashes to `08805d91`, and the shipped gate turns that mismatch into
the observed behavior:

```js
statusOf(r, state) { if (r.approval.launchHash !== r.launchHash) return "awaiting_launch_approval"; ... }
```

Servers in that status are excluded from the available set (same file:
`... entry.status !== "awaiting_launch_approval" ...`), and
`approveLaunchAndDiscover(serverId)` is the flow that records the new hash. The
entry is registered correctly but unapproved, so none of this says anything yet
about the environment payload.

The step the fix depends on is verified in the same bundle -- the stdio spawn
merges the per-server env map onto the inherited allowlist,
`env: { ...getDefaultEnvironment2(), ...this._serverParams.env }`, with
`DEFAULT_INHERITED_ENV_VARS2` (win32) exactly `APPDATA, HOMEDRIVE, HOMEPATH,
LOCALAPPDATA, PATH, PROCESSOR_ARCHITECTURE, SYSTEMDRIVE, SYSTEMROOT, TEMP,
USERNAME, USERPROFILE, PROGRAMFILES`.

State of the PATHEXT question: **open, now localized to approval.** The per-call
`env` payload stays the supported route. After the owner approves `lean-ctx` in
the app (expected to write `launchHash: sha256:08805d91...` and bump that
record's `generation`), re-run the addendum B probe with **no** `env` payload:
a full `PATHEXT`, non-empty `COMSPEC` and a resolving `git` retire the payload; a
`.CPL`/empty reading keeps it.

**Addendum D -- the approval landed and the MCP server is running, but
`ctx_shell` was left off the tool allowlist (2026-09-18 09:27).** The owner
completed both approval steps and the record changed exactly as predicted:

| Item | Before | After (sidecar written 09:26:17) |
| --- | --- | --- |
| `lean-ctx` `launchHash` | `sha256:134d74c7...` | `sha256:08805d91...` (equals the live config hash) |
| `lean-ctx` `manifestHash` | `sha256:90e2b18...` | `sha256:81b9864b...` |
| `lean-ctx` `generation` | 2 | 4 (one bump per step: launch, then tools) |
| `~/.freebuff/mcp.json` mtime | 2026-09-10 18:22:36 | 2026-09-18 09:26:17 |

An app-spawned MCP child appeared at 09:26:02: `lean-ctx.exe` PID 5412, parent
`bun.exe` 22144 (the orchestrator) -- distinct from the long-lived
`lean-ctx.exe` PID 17548 (`proxy start --port=4444`, parent 19260). The observed
sequence matches the shipped code: `approveLaunch` writes `{enabled: false,
launchHash, manifestHash: null, generation: prev + 1}`, discovery spawns the
server through `pool.listTools`, and `approveTools` then writes the discovered
manifest with `enabled: true` and one more generation bump.

Two obstacles remain, and neither is an environment question. First, the
approval stored a **selective tool allowlist** -- 10 of the 18 manifest tools
(`ctx_callgraph`, `ctx_delta`, `ctx_expand`, `ctx_explore`, `ctx_glob`,
`ctx_overview`, `ctx_read`, `ctx_search`, `ctx_tree`, `ctx_url_read`). Excluded
are `ctx_call`, `ctx_compose`, `ctx_execute`, `ctx_graph`, `ctx_knowledge`,
`ctx_patch`, `ctx_session` and **`ctx_shell`** -- that is, all three tools whose
manifest entries carry `destructiveHint: true` (`ctx_execute`, `ctx_patch`,
`ctx_shell`), so the approval default omits destructive tools. `ctx_shell` must
be switched on in the lean-ctx tool approval before any agent can run the probe.
Second, a running agent session's MCP server set and tool list are fixed when the
session starts, so a session begun before the approval still answers `No MCP
server named "lean-ctx" is configured`.

The same bundle enforces both limits at call time. `isToolCallable(approval,
status, toolName, snapshotGeneration)` refuses unless `launchHash` and
`manifestHash` are set, `approval.generation === snapshotGeneration` (the session
holds a generation snapshot, so a session begun before an approval or tool-list
change cannot call that server's tools), the status is not `disabled` / `error` /
`auth_required` / `awaiting_launch_approval` / `awaiting_tools_approval`, and
`allowedTools.includes(toolName)`. The tool-search path skips a tool outright
when `allowed !== null && !allowed.includes(tool.name)`, so a tool left off the
allowlist is unreachable from both directions.

The no-payload probe therefore needs **both** an approved `ctx_shell` and a
**new session** (fresh chat) -- not another launch approval, and not necessarily
an app restart. Expected reading, from the code-verified merge above: full
`PATHEXT`, non-empty `COMSPEC`, and `git`, `python`, `cmake` all resolving; a
`.CPL`/empty reading would instead falsify the
`{ ...getDefaultEnvironment2(), ...serverParams.env }` merge.

Probe-tooling note (observed, mechanism unexplained): through the app's own
`run_terminal_command` tool a PowerShell one-liner containing `$_` came back
mangled -- `PATH` text spliced into the command line,
`x86 : The term 'x86' is not recognized ...` after
`powershell -NoProfile -Command "... Where-Object { $_.ProcessName -match ... }"`.
The same facts were obtained without `$_`, e.g.
`powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like '%lean-ctx%'\" | Format-List ProcessId,ParentProcessId,CreationDate,CommandLine"`;
plain `Write-Output` one-liners are unaffected.

**Addendum E -- the no-payload probe passes; the per-call payload is retired,
and a new cwd finding replaces it (2026-09-18 09:29-09:31).** With the remaining
lean-ctx tools approved (`allowedTools` 18/18, `generation` 5, sidecar written
09:28:19) and **no `env` payload**, the addendum B probe reads:

```text
PATHEXT=[.COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.PY;.PYW;.CPL]
COMSPEC=[C:\Windows\system32\cmd.exe]
OLLAMA_HOST=[]
git version 2.53.0.windows.1
Python 3.13.14
cmake version 4.2.3
```

That is the addendum C/D prediction confirmed: `{ ...getDefaultEnvironment2(),
...serverParams.env }` delivers the configured values. The control reading
settles provenance -- `OLLAMA_HOST` (a user-level variable absent from the
inherited allowlist) is still empty, so `PATHEXT` and `COMSPEC` arrived from the
per-server `env` map rather than from broad environment inheritance. A second
call (fresh call, still no payload) reproduces it as the steady state:
`PATHEXT_len=67`, `COMSPEC=[C:\Windows\system32\cmd.exe]`, `OLLAMA_HOST=[]`. 67
is exactly the length of the configured string, so the value arrives verbatim;
the earlier "`;.CPL` is always appended" note is superseded -- it describes an
*empty* inherited value, for which PowerShell supplies its own `.CPL`, whereas a
value already ending in `;.CPL` is not extended.

The per-call `env` payload is therefore **retired**. Keep the command above as the
regression check if the environment changes again.

Allowlist note from the same calls: `where.exe` is not allowlisted (`git.exe`
already was not), and a blocked segment rejects the whole pipeline before
execution -- `where.exe git` returned `[BLOCKED — DO NOT RETRY] ... [pipeline:
segment 4/5 blocked ... no part of the pipeline ran]`. Use `Get-Command git` or
simply run `git`.

New finding, which replaces the old obstacle: lean-ctx's *detected project root*
is `C:\Users\<user>` (the MCP server's own working directory), so every call that
has to *infer* the root resolves to the home directory. `ctx_shell` therefore
starts in `C:\Users\<user>`, an explicit `cwd` aimed at this repository is
refused, and `ctx_tree` with no `path` refuses `.`:

```text
[cwd: requested path rejected by project-root jail (path escapes project root:
C:/Development/AI/qwen38GZI (root: C:/Users/<user>). lean-ctx could not detect
your project automatically. ... lean-ctx doctor --fix) — ran in . instead]
```

```text
ERROR: refusing to scan '.' — it resolves to a broad or privacy-protected
directory (C:\Users\<user>). Pass a specific project directory as `path`.
```

The cause is visible in the shipped spawn spec: a stdio entry without `cwd` gets
`cwd: value2.cwd ?? env2.HOME ?? homedir()`, and the `lean-ctx` entry declares no
`cwd`, so the server runs in `$HOME` and lean-ctx infers `$HOME` as the project.

Scope of the limitation (all measured in the same session): absolute paths work
-- `ctx_read path=C:/Development/AI/qwen38GZI/AGENTS.md` read that file, and
`ctx_tree path=C:/Development/AI/qwen38GZI depth=1` listed the repository -- and
`ctx_shell` reaches the repository through command-level flags rather than cwd:
`git -C C:/Development/AI/qwen38GZI log --oneline -2` and
`git -C C:/Development/AI/qwen38GZI status --short` both ran and returned this
worktree's state. The jail therefore bites only when a call must infer the root
(`ctx_shell` cwd; any `ctx_*` call with `path` omitted). Optional fix if rootless
calls matter: declare `"cwd"` on the `lean-ctx` entry in `~/.agents/mcp.json`
(changes the launch hash, so it costs another launch + tool approval) or repair
detection lean-ctx-side (`lean-ctx doctor --fix`, which writes outside this
repository). Whether this repository's own `.lean-ctx.toml` profile is honored
while the detected root is `$HOME` is untested.





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
