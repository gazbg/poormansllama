Yes, clone `llama.cpp` inside the empty project, but **do not separately clone `ggml` beside it for the build**. The normal `llama.cpp` repository already contains a synchronized `ggml/` subdirectory and builds it as part of the project. [github](https://github.com/ggml-org/llama.cpp)

## Recommended layout

```text
qwen38-engine/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── engine.cpp
│   ├── session.cpp
│   └── policy.cpp
├── include/
├── tests/
├── docs/
├── models/                 # ignored by git
│   └── Qwen3.8-27B/
└── third_party/
    └── llama.cpp/          # official clone; contains ggml/
        ├── ggml/
        ├── src/
        ├── common/
        ├── tools/
        └── examples/
```

Clone it like this:

```bash
mkdir qwen38-engine
cd qwen38-engine

git init

git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
```

Then:

```text
third_party/llama.cpp/ggml/
```

is the ggml source you should use.

The architecture is intentionally layered: `llama.cpp` contains the high-level model runtime, while its `ggml/` directory contains the tensor library and hardware backends. [deepwiki](https://deepwiki.com/qualcomm/llama.cpp/1.2-repository-structure)

## Why not clone both independently?

Avoid this layout initially:

```text
qwen38-engine/
├── llama.cpp/
└── ggml/
```

It creates unnecessary problems:

- You may compile two different ggml revisions.
- `llama.cpp` already expects its own synchronized ggml tree.
- CMake target names can conflict.
- Vulkan backend files may be duplicated.
- Your agent may accidentally link headers from one copy and libraries from another.
- Bugs become harder to reproduce because the llama.cpp/ggml commit pairing is lost.

The official llama.cpp tree maintains its internal ggml copy through synchronization machinery, so using the bundled copy is the safest starting point. [deepwiki](https://deepwiki.com/qualcomm/llama.cpp/1.2-repository-structure)

## What should you use as the reference?

Use both, but in different ways:

```text
llama.cpp:
    reference runtime
    Qwen3.8 model integration
    GGUF loading
    tokenizer/chat templates
    KV-cache behavior
    sampling
    multimodal and MTP support

ggml inside llama.cpp:
    tensor operations
    graph execution
    Vulkan backend
    CPU backend
    CUDA backend
    quantized kernels
```

Your own engine should initially link against **libllama**, not directly against ggml. This saves substantially more implementation work because Qwen3.8 is not a conventional transformer: it contains Gated DeltaNet layers, full-attention layers, vision support, and MTP.

## Best first architecture

```text
Your qwen38-engine
        |
        | calls
        v
libllama
        |
        | uses
        v
ggml + Vulkan backend
        |
        v
GTX 1070
```

Your code should own:

- Request/session management.
- Prompt and conversation policy.
- `reasoning_effort`.
- `enable_thinking`.
- `preserve_thinking`.
- Output parsing.
- Quantization selection.
- Context-budget policy.
- Telemetry.
- RAM/VRAM policy where exposed by the API.

Let llama.cpp own:

- GGUF loading.
- Qwen3.8 graph construction.
- Gated DeltaNet execution.
- Full attention.
- KV/state management.
- Quantized matrix operations.
- Vulkan kernels.
- Vision/MTP support when you add those phases.

## CMake integration

Start with a minimal top-level `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(qwen38_engine LANGUAGES C CXX)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp)

add_executable(qwen38-engine
    src/main.cpp
    src/engine.cpp
    src/session.cpp
    src/policy.cpp
)

target_compile_features(qwen38-engine PRIVATE cxx_std_17)

target_link_libraries(qwen38-engine
    PRIVATE
        llama
)
```

The exact target names can change with upstream revisions. If `llama` is not found, inspect the generated CMake targets or the upstream `CMakeLists.txt`; the official project builds with CMake and supports enabling Vulkan using `GGML_VULKAN=ON`. [github](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

Build:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON

cmake --build build --config Release -j
```

If you are using Windows with Visual Studio:

```powershell
cmake -S . -B build `
  -D CMAKE_BUILD_TYPE=Release `
  -D GGML_VULKAN=ON

cmake --build build --config Release --parallel
```

## Pin the dependency

Do not leave `llama.cpp` floating on an unrecorded moving commit. After cloning:

```bash
cd third_party/llama.cpp
git rev-parse HEAD
```

Record the commit in your top-level `DEPENDENCIES.md`:

```text
llama.cpp:
  repository: https://github.com/ggml-org/llama.cpp
  commit: <insert-commit-sha>
  usage: libllama, GGUF, Qwen3.8 integration, Vulkan backend

ggml:
  source: third_party/llama.cpp/ggml
  version: synchronized with the llama.cpp commit above
```

For development, a Git submodule is better:

```bash
rm -rf third_party/llama.cpp
git submodule add https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git submodule update --init --recursive
```

Then commit:

```bash
git add .gitmodules third_party/llama.cpp
git commit -m "Add pinned llama.cpp dependency"
```

This makes the exact llama.cpp/ggml pairing reproducible.

## Do not modify the dependency first

Your agent should initially avoid changing:

```text
third_party/llama.cpp/
```

Instead, create adapters in your own project:

```text
src/
├── llama_adapter.cpp
├── llama_adapter.hpp
├── qwen38_policy.cpp
├── qwen38_policy.hpp
├── session_manager.cpp
└── telemetry.cpp
```

Use an adapter boundary:

```cpp
struct GenerateRequest {
    std::string prompt;
    int32_t max_tokens;
    bool enable_thinking;
    bool preserve_thinking;
    std::string reasoning_effort;
};

struct GenerateResult {
    std::string reasoning;
    std::string answer;
    uint64_t prompt_tokens;
    uint64_t generated_tokens;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual GenerateResult generate(
        const GenerateRequest& request) = 0;
};
```

Then implement:

```text
Qwen38LlamaBackend
    -> llama.cpp C API
```

This keeps your optimization experiments independent from upstream internals.

## When to use standalone ggml

Use standalone ggml only if you specifically need to build your own computation graph or replace llama.cpp’s model execution.

That would look like:

```text
Your engine
    |
ggml API
    |
ggml Vulkan backend
```

However, for Qwen3.8‑27B this is a much larger undertaking because you would need to reproduce:

- The 64-layer hybrid Gated DeltaNet/full-attention architecture.
- Linear-attention recurrent state.
- Full-attention KV state.
- Qwen3.8 tensor mapping.
- Vision encoder/projector integration.
- MTP heads.
- Chat-template behavior.

Therefore, the correct progression is:

```text
Stage 1: Link libllama and use its Qwen3.8 support.
Stage 2: Add your scheduler, session, telemetry, and policy layers.
Stage 3: Profile.
Stage 4: Modify ggml/llama.cpp only for a measured bottleneck.
Stage 5: Use standalone ggml only if the required optimization cannot be
         expressed through libllama.
```

## Recommended `.gitignore`

```gitignore
/build/
/out/
/models/
/third_party/llama.cpp/build/
/*.gguf
*.gguf
*.safetensors
*.bin
*.log
```

Do not put the 17–19 GB Q4 model into Git. Keep it under `models/` and document how to download it.

## Important correction for the GTX 1070

For Qwen3.8‑27B, the GTX 1070 should be treated as a **development and fallback device**, not a realistic full-VRAM target.

A Q4 model is roughly 17–19 GB, already larger than the GTX 1070’s 8 GB VRAM, before runtime buffers and context state. You would need CPU/RAM offloading and a short context. For initial validation, use: [docs.nvidia](https://docs.nvidia.com/cutlass/4.6.0/media/docs/cpp/cutlass_3x.html)

```text
Qwen3.8-27B Q2 or IQ2 GGUF
Vulkan backend
CPU/RAM + partial GPU offload
context: 4K–8K initially
vision disabled
MTP disabled initially
```

For serious Q4 performance, use a GPU with at least 24 GB VRAM. The Qwen3.8 architecture has only 16 conventional full-attention KV-cache layers, but its model weights remain the dominant memory cost. [docs.nvidia](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/005_grouped_gemm_contiguous_offset.html)

## Final recommendation

Use this exact structure:

```text
qwen38-engine/
├── src/                         # your engine
├── include/                     # your public interfaces
├── tests/                       # your tests and oracle harness
├── docs/
├── models/                      # ignored model files
├── CMakeLists.txt
├── DEPENDENCIES.md
└── third_party/
    └── llama.cpp/               # pinned clone
        └── ggml/                # use this copy; do not clone separately
```

Your agent instruction should be:

> Clone `https://github.com/ggml-org/llama.cpp` as a pinned submodule under `third_party/llama.cpp`. Use the ggml source bundled inside that checkout. Do not maintain a second standalone ggml checkout. Initially link against libllama and use the existing Qwen3.8/GGUF/Vulkan implementation. Keep all project-specific session management, quantization policy, RAM/VRAM policy, reasoning controls, telemetry, and experiments outside the dependency tree. Modify upstream only after profiling identifies a required missing feature or bottleneck.