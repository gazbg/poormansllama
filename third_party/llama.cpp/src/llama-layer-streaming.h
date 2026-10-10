#pragma once

// Experimental CUDA-only decoder-layer weight streaming for dense Qwen3.5
// (private research path, gated by LLAMA_LAYER_STREAMING). This is NOT an
// upstream-facing feature. The default llama.cpp execution paths never touch
// this module when the CMake option is off.
//
// The module owns:
//  - N device weight slots (one CUDA buffer + tensor bundles per slot)
//  - a persistent CUDA activation pair (hidden states between layer graphs)
//  - a persistent host buffer for graph input tensors
//  - a dedicated CUDA copy backend (separate stream from compute)
//  - per-slot CUDA events for copy/compute ordering and timing
//  - a JSONL telemetry sink
//
// Logical layer identity and device-slot identity are separate. Each slot owns
// one tensor bundle per logical layer assigned to it (il % n_slots == slot).
// All per-layer bundles alias the same device buffer — only one layer is
// resident at a time. This lets each layer carry its own quant types
// (dynamic/imatrix models) without inflating VRAM.

#include "llama-model.h"
#include "llama-layer-streaming-pool.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct llama_context;
struct llm_graph_result;
struct ggml_context;
struct ggml_tensor;
// the predictor's learning state (defined with the pure policy API below): the
// engine holds it behind a pointer so the module's own declaration order -- class
// first, pure API after -- does not have to change for mode 2.
struct llama_ls_pred_state;
struct llama_ls_d1_state;

// one immutable PP chunk descriptor for the layer-major schedule
// (T-Q38CTX-01): the chunk's first token position and its real token count
// (1..ubatch; all chunks but the last are full-size, the last is 1..ubatch)
struct llama_ls_chunk {
    int64_t token_start = 0;
    int64_t n_tokens    = 0;
};

// PP schedule for the streaming executor. BATCH_MAJOR is the normative SDD
// schedule (ubatch-major: every layer runs per ubatch). LAYER_MAJOR is the
// opt-in experiment: each executed logical layer processes all PP chunks in
// ascending token-position order before the next layer starts (SDD section 13).
enum llama_pp_schedule {
    LLAMA_PP_SCHEDULE_BATCH_MAJOR = 0,
    LLAMA_PP_SCHEDULE_LAYER_MAJOR = 1,
};

// ---------------------------------------------------------------------------
// T-GPLR-01: architecture-neutral package descriptor and adapter interface.
//
// An adapter enumerates the immutable tensor members of one pageable package
// (a decoder layer, an expert bank, ...) as source tensors plus destination
// slot members. The common slot/lifecycle code never inspects architecture
// tensor names or member lists; it only consumes validated descriptors.
// Pointer-to-member into the common llama_layer/llama_layer_nextn structs is
// the destination identity; it is not architecture-specific.
// ---------------------------------------------------------------------------

struct llama_ls_member {
    const ggml_tensor * src = nullptr;                  // host/mmap source tensor
    ggml_tensor * llama_layer::* dst_base = nullptr;    // destination member, or
    ggml_tensor * llama_layer_nextn::* dst_nextn = nullptr; // exactly one is set
    bool mutable_state = false; // state tensors must never be package members
};

struct llama_ls_package {
    std::vector<llama_ls_member> members; // deterministic order (copy = order)
    const char * cls = "";                // "recurrent" | "attention" | "mtp" | ...
    const char * policy = "";             // "dense-layer" | "moe-expert-bank" | ...
    bool excludes_static = false;         // embeddings/output head are not members
    bool excludes_state = false;          // KV/recurrent state are not members
};

// Pure host-side descriptor validation (no CUDA, no model): non-empty, exactly
// one destination slot per member, no duplicate destinations, no mutable
// members, no zero-byte sources, explicit exclusion flags. require_src also
// demands every member carry a non-null source tensor (host-side checks).
LLAMA_API bool llama_ls_validate_package(const llama_ls_package & pkg, std::string & err, bool require_src);

// T-GPLR-56 A1: process-life markers for the acceptance quantity Q, which runs from
// process start to the end of the first prefill pass. The telemetry sink does not exist
// until the streaming object's init() opens it (mode "w"), and by then the model load is
// already over - so a marker is BUFFERED here and flushed, in call order, as the sink's
// first events. Telemetry only: marking cannot fail, cannot change any caller's behaviour,
// and with no sink the markers are dropped. Main-thread use only.
//
// A marker taken AFTER a flush is a LATE marker: it did not come out of the pre-sink buffer,
// so it is emitted with `late:1` and WITHOUT `pre_run` (which asserts exactly that origin),
// and still without a run_id. Before 2026-09-22 a late marker was buffered into a vector
// that nothing ever flushed again: silently lost, with no error and no flag. There is no
// separate file handle to write to - one buffer, one emission path - so a caller that
// closes a sink can never leave a marker writing to a dead stream.
LLAMA_API void llama_ls_tel_mark(const char * event, const char * detail);

// Writes the buffered markers to `f` (if non-null) and clears the buffer. NULL means "no
// sink": the buffer is cleared, and markers taken afterwards are still buffered (as late)
// so a later flush with a real sink emits them. init() calls it with the sink the moment it
// opens; this file's teardown, run-end and refusal paths call it with the sink again right
// BEFORE their fclose, which is what makes a post-sink marker a written line instead of a
// lost one. Exported so a host-side test can prove the emitted line shape without a model,
// a device or a telemetry path.
LLAMA_API void llama_ls_tel_flush_early(FILE * f);

// T-GPLR-05: deterministic host-only capability result for any model artifact.
// `supported` is true only for adapter-declared policies validated in this
// workspace; `reason` is a stable lowercase code-like sentence for telemetry
// and refusals. `auto` mode may select only `policy` when supported, and must
// fall back to `off` with `reason` otherwise.
struct llama_ls_capability {
    bool        supported = false;
    std::string arch;      // general.architecture, when readable
    std::string adapter;   // adapter name, when supported
    std::string policy;    // adapter-declared primary policy, when supported
    std::string reason;    // stable reason string in every case
};

// Host-only capability probe: reads only the GGUF header KV (no CUDA, no
// tensor data, no normal model loading) and never throws for unreadable
// files. Returns the recommended policy string ("off" when unsupported) and
// fills `cap` with the deterministic result.
LLAMA_API std::string llama_ls_probe_capability(const char * model_path, llama_ls_capability & cap);

struct llama_ls_adapter {
    const char * name = "";   // e.g. "qwen35"
    const char * policy = ""; // primary pageable policy, e.g. "dense-layer"

    // When true, the slot bundle for a layer is the resident model layer with
    // ONLY the package members replaced by device slot tensors; every other
    // member keeps pointing at its existing (CUDA-resident) buffer. Used by
    // the MoE expert-bank policy, where the package is one weight class and
    // attention/router/shared-expert/norm weights stay resident. Dense
    // adapters set this false (the package is the whole decoder layer).
    bool merge_resident = false;

    // When true, the adapter's layer-major graphs may use ggml causal op
    // params instead of dense KQ masks (replaces the former hard-coded
    // arch == LLM_ARCH_QWEN35 gate in llm_graph_context).
    bool causal_implicit = false;

    // enumerate the rolling package for trunk logical layer il; returns false
    // with a stable reason when the layer cannot be packaged. model may be
    // null for host-side fixture tests (the dense adapter needs no hparams).
    bool (*package)(const llama_model * model, int il, const llama_layer & src,
                    llama_ls_package & out, std::string & err) = nullptr;

    // optional: the NextN/MTP package (T-Q38HG-06 migration).
    bool (*mtp_package)(const llama_model * model, const llama_layer & src,
                        llama_ls_package & out, std::string & err) = nullptr;
};

// Host-testable capability probe: returns the adapter for an architecture or
// nullptr with a stable reason. No CUDA, no model allocation.
LLAMA_API const llama_ls_adapter * llama_ls_select_adapter(llm_arch arch, std::string & reason);

// The Qwen3.5 dense adapter (defined in src/models/llama-layer-streaming-qwen35.cpp).
LLAMA_API const llama_ls_adapter * llama_ls_qwen35_adapter();

// The Qwen3.5/3.6 MoE adapter (same file; T-GPLR-02). Rolling package =
// complete routed-expert bank for one MoE layer.
LLAMA_API const llama_ls_adapter * llama_ls_qwen35moe_adapter();

// one decoder-layer weight package bound to one device slot
struct llama_layer_streaming_slot {
    // meta context owning the bundle tensor objects (freed at destruction)
    ggml_context * ctx_meta = nullptr;

    // per-logical-layer tensor bundles. Every bundle assigned to this slot
    // (il % n_slots == slot index) has its own llama_layer of device tensor
    // objects with that layer's exact shapes and quant types. All bundles
    // alias the same device buffer; only one layer is resident at a time.
    std::unordered_map<int, llama_layer> layers;

    // per-logical-layer package byte size (copy size for the slot)
    std::unordered_map<int, size_t> layer_bytes;

    // device buffer backing all per-layer bundles
    ggml_backend_buffer_ptr buf;

    // T-GPLR-62 S2: the COMPACT row-set window's own tensor metadata and device
    // buffer. Built by build_row_set_window() at begin_row_set, after the
    // full-bank buf has been released, so the two never coexist and the ~1.99 GiB
    // full-bank reservation is what gets handed back. The narrow bundles are
    // written into slot.layers[il] in place of the full-bank ones, so every
    // consumer sees one shape; layer_bytes_rowset records what they cost.
    ggml_context * ctx_rowset = nullptr;
    std::unordered_map<int, size_t> layer_bytes_rowset;
    ggml_backend_buffer_ptr buf_rowset;

    // optional pinned host buffer for one complete package copy
    ggml_backend_buffer_ptr buf_staging;

    // T-GPLR-37: the page-locked span of the mapped SOURCE currently backing
    // this slot (direct mmap-sourced staging). Unregistered before the next
    // staging of the same slot — safe, because a slot is only re-staged after
    // its previous generation's consumer completed, which implies that
    // generation's DMA finished — and at slot teardown.
    const void * reg_base = nullptr;
    size_t       reg_bytes = 0;

    // ordering/timing events (ggml_backend_event_t on the CUDA device)
    void * ev_copy_ready    = nullptr; // recorded on the copy stream after last copy into the slot
    void * ev_consumer_done = nullptr; // recorded on the compute stream after the last consuming graph
    void * ev_copy_start    = nullptr; // timing: copy submit
    void * ev_copy_end      = nullptr; // timing: copy completion
    // T-GPLR-32: stream-ordering event for the one-DMA-in-flight gate. When
    // pp_af_serialize is set, every worker copy waits on this event (recorded
    // by the PREVIOUS worker copy) before its own ev_copy_start, so at most
    // one worker H2D is in flight on the copy stream at any time. Stream wait
    // only: the CPU hop stays fully overlapped (no host blocking).
    void * ev_af_gate       = nullptr;
    // T-GPLR-33 V1: per-layer single-copy overlay. When the bundle's members
    // tile the buffer region [base, base+total) exactly (verified at slot
    // build), an I8 1-D tensor spanning the whole bank aliases that region so
    // staging can issue ONE H2D for the package. Absent entry (or nullptr)
    // = that layer stays on the legacy per-member path.
    std::unordered_map<int, ggml_tensor *> bank_span;

    // monotonically increasing generation; the logical layer last copied here
    uint32_t generation    = 0;
    int32_t  logical_layer = -1;

    // host timestamps for the current copy lifecycle
    int64_t t_copy_submit_us       = 0;
    int64_t t_copy_enqueue_done_us = 0;

    // whether a consumer is still outstanding (copy must wait for ev_consumer_done)
    bool consumer_pending = false;

    // T-GPLR-25 residue fix: set when the async-feed WORKER submitted this
    // slot's current copy. wait_slot_ready consumes it (clearing the submit
    // timestamps so the next wait is a worker-fed hit) and gates
    // tel_async_feed_ready, so consumption events never report readiness for
    // a copy the worker did not submit (the main-thread lookahead path sets
    // this never; its copies are reported by copy_submit/copy_complete only).
    bool af_submitted = false;
};

// LLAMA_API: exported so the experimental runner can link the module from
// the shared llama library (BUILD_SHARED_LIBS)
struct LLAMA_API llama_layer_streaming {
    // number of device weight slots
    int n_slots = 0;

    // CUDA device the module runs on
    int device = 0;

    // dedicated copy backend (separate stream from the compute backend)
    ggml_backend_t backend_copy = nullptr;
    // T-GPLR-19: the scheduler's CUDA compute backend, resolved once when the
    // row cache is created. The cache->slot refresh needs it as the copy
    // DESTINATION so the cross-backend copy queues on the copy stream and the
    // compute stream waits on it (ordering without a device synchronize).
    ggml_backend_t backend_compute = nullptr;

    // copy sources are either direct mmap pages or per-slot pinned staging
    bool use_pinned_staging = false;

    // T-GPLR-37: direct mmap-sourced staging (--copy-source mmap-registered).
    // When on, staging page-locks the package's mapped SOURCE span
    // (cudaHostRegister) instead of copying every member into a pinned staging
    // buffer, and DMAs straight from the model's mapping: no CPU hop, pinned
    // transfer rate, and RAM-neutral (no staging buffers are allocated at all
    // in this mode). Mutually exclusive with use_pinned_staging by construction
    // (the runner maps one --copy-source value to one of the two).
    bool copy_src_registered = false;
    // T-GPLR-37 churn-free variant (--copy-source mmap-registered-churn): the
    // first implementation registered and unregistered each 436 MiB file region
    // per bank, which measured 80 registrations / 34.46 GiB of churn per run and
    // inflated the per-launch host windows (setup 11 -> 136-228 ms, compute +0.5
    // to +0.9 s) for a net warm regression of 15-24% at 4k. That path is kept
    // reproducible behind this flag; the default mmap-registered mode registers
    // every executed layer's span ONCE before the timed PP instead.
    bool copy_src_reg_churn = false;
    // per-layer registered source spans from register_all_pkg_sources (index =
    // logical layer; nullptr = not registered). Read-only after that call.
    std::vector<const void *> all_reg_base;
    std::vector<size_t>       all_reg_bytes;
    std::vector<bool>         all_reg_covered;   // T-GPLR-39: member of a coalesced registration group (span owned by the group's first layer)

    // T-GPLR-56 A3 - PACED REGISTRATION (owner-ratified 2026-09-22; DEFAULT OFF).
    //
    // The problem the mode attacks is a LATENCY one: `register_all_pkg_sources`
    // issues every cudaHostRegister up front, and the DEFERRED work of those calls
    // commits on the first submission, so the first staging copy of the sweep
    // stalls for it (measured 1814.3 ms at 4k; the calls themselves are ~1.6-2.3 ms).
    // Paced mode keeps the same total commit but issues it in GROUP order a bounded
    // distance ahead of the copy that needs it: the first group before the timed
    // pass, every later group from stage_members() when the sweep reaches the layer
    // `ahead` groups before it. Pending (unlocked) bytes therefore stay near one
    // group instead of 17.23 GiB, and the commit is spread across the pass.
    //
    // It is additive BY CONSTRUCTION: `pp_span_pace == 0` never enters this state
    // and register_all_pkg_sources() is not modified by the mode at all, so a
    // default run's code path, events and bytes are unchanged. Nothing about the
    // measured quantity changes either: Q is still proc_start -> pass1_end, and
    // each arm declares it. Telemetry for the mode is emitted under NEW event names
    // (`span_pace`, `span_pace_stage`, `span_pace_end`) so no existing parser sees
    // a new shape.
    struct ls_reg_group {
        uintptr_t lo       = 0;   // page-rounded group start
        uintptr_t hi       = 0;   // page-rounded group end (exclusive)
        size_t    bytes    = 0;   // hi - lo
        int       first_il = 0;   // the group's FIRST logical layer (it owns the registration)
        size_t    merged   = 0;   // spans coalesced into this group
        size_t    members  = 0;   // package members inside it
        bool      done     = false;
        // The logical layers whose spans lie inside this group. They are NOT
        // necessarily contiguous: exec_layers may be a boundary subset, and a
        // group is coalesced by ADDRESS, so coverage has to be marked per layer.
        std::vector<int> layers;
    };
    std::vector<ls_reg_group> paced_groups;
    size_t                    paced_next_group = 0;
    std::vector<uint8_t>      paced_touch_seen;   // per layer: its first staging copy was observed
    int64_t                   paced_bytes_total  = 0;
    int64_t                   paced_bytes_issued = 0;
    int                       pp_span_pace       = 0;   // 0 = off (existing path), 1 = paced
    int                       pp_span_pace_ahead = 1;   // groups issued ahead of the staging copy

    // T-GPLR-30: persistent pinned pool. Empty when off (no --pp-pool).
    // One pinned host buffer per entry, mirroring one bank's package bytes.
    // fixed: entry i mirrors bank i, populated once at configure, never
    // rewritten. lru: entries refill on miss (the caller's stage loop writes
    // the members through the acquired base, so a fill costs exactly the
    // per-slot hop it replaces).
    struct pp_pool_entry {
        ggml_backend_buffer_ptr buf;   // pinned host buffer, one bank
        size_t bytes = 0;              // exact package byte size
        bool   populated = false;      // contents valid (lru fills lazily)
        int    bank = -1;              // logical layer currently mirrored
        bool   active = false;         // acquired by a stager right now
        bool   inflight = false;       // last fill's DMAs may still be pending
    };
    std::vector<pp_pool_entry> pp_pool;
    int  pp_pool_n     = 0;          // pool capacity (banks)
    int  pp_pool_policy= 0;          // 0 = fixed, 1 = lru
    std::mutex pp_pool_mtx;          // guards the pool protocol; both stager
                                     // paths (af worker + main lookahead) run
    uint64_t pool_n_hit = 0;
    uint64_t pool_n_miss = 0;
    uint64_t pool_bytes_saved = 0;   // memcpy bytes the hop would have cost
    // T-GPLR-30 pool protocol (thread-safe):
    //   acquire(il, bytes): fixed — returns the pre-populated mirror base for
    //     banks 0..N-1 (nullptr otherwise). lru — hit returns the entry base
    //     and refreshes recency; miss evicts the least-recent INACTIVE entry
    //     (draining the copy stream first when that entry's previous fill may
    //     still have DMAs pending) and returns its base for the caller to
    //     fill; nullptr when nothing is evictable. One pool_event per call.
    //     The base is exclusively owned until the matching release.
    //   release(il, base, staged_bytes): clears the entry's active flag and,
    //     under lru, marks its contents possibly-DMA-inflight so the next
    //     eviction of that entry drains the copy stream before rewriting —
    //     the ordering that makes pool reuse safe against the async DMAs that
    //     read the pool (no torn weights, ever).
    char * pp_pool_acquire(int il, size_t & bytes, bool & contents_valid);
    void   pp_pool_release(int il, char * base, size_t staged_bytes, bool ok);
    // T-GPLR-30 observability: `why` names the OUTCOME so a zero hit rate is
    // self-describing. A bare hit/miss cannot distinguish a structural cliff
    // (capacity < working set) from a broken policy, which is how the lru4
    // 0% was misread as a policy result. Reasons:
    //   resident      hit: the wanted bank was populated and idle
    //   evicted       miss: an inactive entry was evicted to place the bank
    //   out_of_range  miss: fixed policy, bank index beyond the pool (CAPACITY)
    //   not_populated miss: fixed policy, entry exists but was never filled
    //   entry_active  miss: fixed policy, entry is held by a live acquirer
    //   no_free_entry miss: lru, every entry active or oversized (OCCUPANCY)
    //   no_pool       NOT emitted: a disabled pool is not a capacity failure
    void   tel_pool_event(int il, bool hit, size_t bytes, const char * why);

    // Qwen layer-major PP uses causal metadata instead of dense KQ masks.
    bool use_implicit_causal_mask = false;

    // T-Q38PW-00: per-width graph-sets probe — when true, the prebuild creates
    // only one per-width set (full-512), and the layer-major controller patches
    // tensor data pointers per chunk rather than selecting per-chunk sets.
    bool pw_probe = false;

    // T-Q38PW-02: element type of the two activation slabs (F32 or BF16).
    // Changing from F32 to BF16 halves the VRAM slab footprint without altering
    // graph topology (ggml handles type promotion/demotion transparently).
    // Default unchanged (F32); the parity gate (100K f32 vs bf16) is required
    // before any bf16 claim.
    ggml_type act_slab_dtype = GGML_TYPE_F32;

    // largest TG verify ubatch width with a bound act pair (1 + n_rs_seq under
    // MTP batch verify, 1 otherwise); drives the activation-slab TG budget
    int tg_max_width = 1;

    // cumulative patch time across all chunks (µs)
    double total_patch_us = 0.0;

    // per-slot buffers/bundles/events
    std::vector<llama_layer_streaming_slot> slots;

    // embedding width used for the activation pair (n_embd_inp of the model)
    int64_t n_embd_io = 0;

    // persistent CUDA buffer for the activation pair
    ggml_backend_buffer_ptr buf_act;

    // layer-major activation slabs (SDD section 13): two stable full-prompt
    // F32 slabs A and B. One 1-D slab tensor owns both halves; every PP chunk
    // gets a non-aliasing ggml_view_2d slice of each slab, bound at graph
    // build time (per-chunk graph sets). A persistent graph is never
    // retargeted by reassigning a tensor data pointer.
    ggml_backend_buffer_ptr buf_slab;
    size_t slab_off_A = 0;               // cursor inside slab half A (bytes)
    size_t slab_off_B = 0;               // cursor inside slab half B (bytes)
    // T-GPLR-49: the [start,end) byte intervals this module has already BOUND
    // inside each slab half (index 0 = half A, 1 = half B), in claim order. One
    // claim per bound slice. Two live sets claiming overlapping intervals would
    // silently share activation storage, so every claim is checked and a
    // collision refuses loudly instead of running.
    std::vector<std::pair<size_t, size_t>> slab_claims[2];
    int64_t pp_tokens = 0;               // full-prompt token count (slab width)
    int64_t n_chunks = 0;                // chunks.size() (layer-major)

    // immutable per-run chunk descriptors (T-Q38CTX-01): chunk c occupies
    // token range [token_start, token_start+n_tokens). All chunks but the
    // last are full ubatch width; the last is 1..ubatch. Populated by init
    // for the layer-major schedule; the prebuild, the layer-major controller,
    // and the causal metadata all derive positions from this list.
    std::vector<llama_ls_chunk> chunks;

    // persistent host buffer for graph input tensors
    ggml_backend_buffer_ptr buf_inputs;

    // spill buffers for input tensors that no longer fit the budgeted
    // buf_inputs (long-context masks exceed the init() sizing model: more
    // sets bind full-context masks than the n_masks formula counts). Host
    // side only -- no VRAM, no placement change; never created in a shape
    // that fits, so allocation there is unchanged.
    std::vector<ggml_backend_buffer_ptr> buf_inputs_spill;
    size_t buf_inputs_spill_off = 0;

    // T-GPLR-24: allocate a host spill buffer holding `tensors` whole, after
    // the budgeted buf_inputs rejected them. Returns false on failure.
    bool alloc_spill(const std::vector<ggml_tensor *> & tensors);

    // T-GPLR-49: record a slice claim inside one slab half, refusing (with
    // error_msg naming both intervals) when it would overlap an existing one.
    bool claim_slab_region(int which, size_t start, size_t bytes);

    // allocation cursors inside the persistent buffers (advanced per graph set,
    // so coexisting variants never alias each other's regions)
    size_t buf_act_off = 0;
    size_t buf_inputs_off = 0;

    // persistent graph-entry keys per the SDD: (execution_mode, logical_layer,
    // device_slot). mode is the phase at build time (0 = PP pp_u64, 1 = TG
    // tg_u1); embed/output graphs carry layer -1/-2 and slot -1. The executor
    // verifies each graph runs only against its keyed slot assignment.
    struct graph_entry {
        int mode = 0;
        int logical_layer = -1;
        int slot = -1;
    };

    // per-variant persistent graph set. Key: (mode, n_tokens, rs_z, n_outputs) where
    // rs_z is the recurrent state-zero variant baked into the graph at build
    // time (0 = fresh state, the graph zeros one state cell in-graph; -1 =
    // continuing state, the zeroing view is zero-sized). Each set owns its io
    // context (graphs + input tensor objects), its activation pair (its region
    // of buf_act), its input tensors (its region of buf_inputs), and its result
    // object (input holders + output references). Sets never alias.
    struct graph_set {
        ggml_context *     ctx  = nullptr;
        llm_graph_result * res  = nullptr;
        ggml_tensor *      act[2] = { nullptr, nullptr };
        int64_t            n_tokens = -1;
        int32_t            rs_z = -2;
        int32_t            n_outputs = -1; // out_ids width (per-ubatch logits rows)
        int                mode = -1;
        int32_t            chunk_id = -1; // >= 0: layer-major PP chunk identity
        // T-GPLR-71: the attention view width (llama_kv_cache::get_n_kv) this
        // set's graphs were actually built against. -1 = not recorded (a set
        // that has not finished a build). Recorded in finish_build() from
        // cur_build_n_kv, which the two build sites set from the live memory
        // context. A set whose recorded view can no longer COVER the live view
        // is rebuilt, never reused: an under-reading view is a silently wrong
        // answer, not a slow one. Deliberately NOT part of find_set()'s variant
        // identity -- a narrower view must REPLACE its set in place (the
        // T-GPLR-47 fresh-context branch), not allocate a second set beside it.
        int32_t            n_kv = -1;
        bool               built = false;
        // T-GPLR-62 S2: set when the set's baked weight pointers were invalidated
        // from under it -- build_row_set_window() replaced the slot bundle after
        // this set was built. The set is still `built` (its ctx, res and act
        // regions are all still valid and are REPLAYED, not re-derived), but its
        // graphs must be re-derived against the new bundle before next use. The
        // executor treats this exactly like a build-on-miss so the T-GPLR-47
        // fresh-context replacement runs and re-binds every pointer.
        bool               needs_rebuild = false;

        // graphs[0] = embed, graphs[1+il] = layer il, graphs[1+n_layer] = output.
        // graph_layer[i] is -1 (embed), il, or -2 (output).
        std::vector<ggml_cgraph *> graphs;
        std::vector<int> graph_layer;
        std::vector<graph_entry> graph_entries;

        // T-Q38FA-04: cached node/tensor lists for the per-chunk patch walk.
        // Populated once on first prepare_chunk; the persistent graphs are
        // never rebuilt, so these pointers are stable across all chunks.
        std::vector<ggml_tensor *> fa_nodes;       // FLASH_ATTN_EXT nodes (layer graphs only)
        std::vector<ggml_tensor *> input_tensors;  // INPUT-flagged tensors, capture order; zero-consumer filtered (T-GPLR-45)
        bool patch_cache_built = false;    // T-GPLR-02: per-layer MoE route nodes captured at build time

        // T-GPLR-44: per-graph slices of the two caches above. Built ONCE by
        // build_patch_split, which is also the ONE owner of the member list:
        // since T-GPLR-45 the member list contains only zero-consumer-filtered
        // members (the launch restore, the chunk snapshot capture and this
        // split all iterate the same list, so they cannot drift).
        std::vector<std::vector<size_t>>        graph_inputs;
        std::vector<std::vector<ggml_tensor *>> graph_fa_nodes;
        std::vector<size_t>                     input_offsets;
        // Total bytes the split describes (equal to the chunk snapshot's size
        // when both walk the same ctx; a mismatch is itself evidence the ctx
        // accumulated).
        size_t input_snap_bytes = 0;
        bool   patch_split_ok   = false;

        // T-GPLR-45: INPUT candidates that NO graph of the set consumes
        // (transitively through views) and that the filter therefore dropped
        // from the member list and from every snapshot. Reported by
        // tel_patch_cache as excluded_* so the analyzer can assert
        // kept_bytes + excluded_bytes == walked_bytes mechanically.
        std::vector<ggml_tensor *> excluded_inputs;
        size_t excluded_bytes = 0;   // sum of ggml_nbytes over excluded_inputs
    // (selected_experts, "ffn_moe_topk"). Observational only; read after
    // each chunk launch to emit route telemetry.
        std::unordered_map<int, ggml_tensor *> routes;

        // T-GPLR-13: row-set mode builds each layer graph in two parts
        // (attn+router, then FFN on the routed expert rows). ff_part[il] is
        // the FFN graph of layer il, ff_layer its layer tag; the router part
        // keeps its own entries (graphs/graph_layer/graph_entries).
        std::vector<ggml_cgraph *> ff_graphs;
        std::vector<int>           ff_layer;
        // per-FFN-graph host INPUT tensor for the router's selected experts
        // ([n_expert_used, n_tokens] I32, written by the executor between the
        // two parts; allocated from buf_inputs by finish_build)
        std::vector<ggml_tensor *> ff_input;
        // T-GPLR-62 S2: per-FFN-graph host INPUT tensor for the same experts
        // expressed as WINDOW POSITIONS rather than absolute expert ids
        // ([n_expert_used, n_tokens] I32, same layout and same lifetime as
        // ff_input). Only meaningful once the row-set slot is narrower than the
        // bank; while the slot is full width it is populated and audited but the
        // graph keeps consuming absolute ids, so nothing depends on it yet.
        std::vector<ggml_tensor *> ff_input_wpos;
        // T-GPLR-82 S2: per-FFN-graph INPUT for the HOST expert leg's routed
        // ids ([experts_host_n, n_tokens] I32). Deliberately NOT a view of
        // ff_input: the host leg runs on the CPU backend and must not inherit a
        // view of a device tensor, and the host membership rule is executor
        // state that S3 will choose per token. Null (and a null back()) when the
        // host leg is off, which is the default.
        std::vector<ggml_tensor *> ff_input_hsel;
        // T-GPLR-61 step C: per-FFN-graph OUTPUT carrying the host leg's
        // PRE-AGGREGATION per-row expert outputs ([n_embd, experts_host_n,
        // n_tokens] F32), stashed by build_moe_host_leg so the executor can
        // scan for non-finite values after part B. Null (and a null back())
        // when the host leg is off, which is the default.
        std::vector<ggml_tensor *> ff_host_experts;
        // the A/B residual crossing uses the set's FREE act slot (act[(il+1)&1]
        // at part-A time; the slot pair alternates per layer exactly as the
        // single-part graphs do): part A copies the pre-FFN residual there, and
        // part B reads it back as its input. No extra persistent buffer.
    };
    std::vector<graph_set> sets;

    // the set being built (valid between begin_build and finish_build)
    graph_set * cur = nullptr;

    // chunk id used by begin_build when the in-flight model.build_graph calls
    // build_streaming_graphs -> begin_build. Set by the prebuild/executor
    // before model.build_graph so the per-chunk layer-major set is selected
    // (default -1 = batch-major / TG set).
    int32_t cur_chunk_id = -1;

    // number of decoder layers (from the model)
    int n_layer = 0;

    // T-GPLR-01: architecture adapter (package enumeration) + model reference
    // selected in init(); null before init.
    const llama_ls_adapter * adapter = nullptr;
    const llama_model *      model_ref = nullptr;

    // validated per-logical-layer package descriptors (index = il, il <
    // n_layer). Built once in init(). The MTP layer is not part of this cache;
    // its descriptor lives in handoff_pkg_mtp when begin_handoff(mtp) runs.
    std::vector<llama_ls_package> pkg_cache;

    // T-GPLR-02: package member count captured when each slot bundle was
    // created. load_layer refuses to copy when the descriptor later diverges
    // from the bundle (stale/partial package must fail before graph launch).
    std::vector<size_t> bundle_members;

    // T-GPLR-02: observational router-route capture. The MoE graph reports its
    // selected_experts tensor through the cb hook during a streaming graph
    // build; the executor reads the device values after each layer compute and
    // emits telemetry. Routing telemetry never feeds back into routing or
    // graph construction.
    void capture_route(int il, ggml_tensor * t) {
        // during a streaming graph build the tensor belongs to the set being
        // built; the executor reads set->routes after each chunk launch
        if (cur != nullptr) {
            cur->routes[il] = t;
        }
    }
    // read the layer graph's computed ARGSORT (selected_experts) buffer and
    // emit one route event (exact sampled IDs, top-k cardinality, route union,
    // FNV-1a hash over the full ID sequence); observational, never feeds routing
    void tel_route(int il, int64_t chunk, ggml_cgraph * g);

    // PP schedule (default batch-major; set by the runner before prebuild)
    llama_pp_schedule schedule = LLAMA_PP_SCHEDULE_BATCH_MAJOR;

    // Layer-major PP emits chunk 0, every stride boundary, and the final chunk.
    int telemetry_chunk_stride = 10;

    // host-side source tensors per logical layer (from the loaded model)
    std::vector<const llama_layer *> host_layers;

    // recurrent mask: recr[il] == true -> Gated DeltaNet layer
    std::vector<uint8_t> recr;

    // T-Q38HG-06 (MTP): the NextN/MTP layer resides in model.layers[n_layer]
    // (a full-attention block). This flag makes begin_handoff() include its
    // package in the attention-only GPU-resident set so the draft context's
    // LLM_GRAPH_TYPE_DECODER_MTP graph reads device-backed blk.64 tensors.
    bool handoff_mtp = false;

    // optional subset of layers to execute (empty = all layers); used by the
    // one-layer boundary proof. KV/state of skipped layers is untouched.
    std::vector<int> exec_layers;

    // execution phase tag for telemetry (0 = PP, 1 = TG); set by the runner
    // (read from the executor thread; the prefetch worker snapshots it at
    // enqueue time, a benign race for a telemetry tag)
    int phase = 0;

    // number of ubatches executed so far (incremented by the executor)
    uint64_t n_ubatches = 0;
    uint64_t build_generation = 0; // incremented on each finish_build (debug)

    // T-GPLR-71: the attention view width the in-flight build is being made
    // against, handed to finish_build() the same way cur_chunk_id hands the
    // chunk identity to begin_build(). Set by both build sites (the prebuild's
    // build_variant and layer_streaming_compute's build-on-miss) immediately
    // before model.build_graph(); -1 = unknown (consumed and cleared by
    // finish_build()).
    int32_t cur_build_n_kv = -1;

    // T-GPLR-71 step 5 (observability only; no behaviour reads these).
    // last_live_n_kv: the live llama_kv_cache::get_n_kv() observed at the most
    // recent execution, i.e. exactly the width the coverage rule judged
    // against. -1 until an execution has run (the layer-major PP controller
    // never calls layer_streaming_compute, so PP reports -1 under layer-major,
    // which is itself informative: no phase-1 set exists during PP).
    // view_rebuild_ctr: cumulative arm_tg_view_rebuild() firings since run
    // start. Deliberately CUMULATIVE, not per-phase like graph_builds: every PP
    // pass shares ONE graph_builds baseline, so a per-phase base here would let
    // pass 2 hide pass 1's rebuilds. Cumulative cannot hide anything, and since
    // PP's summaries are emitted before TG's, "0 on the PP lines, 1 on the TG
    // line" attributes the firing exactly.
    int32_t  last_live_n_kv  = -1;
    uint64_t view_rebuild_ctr = 0;

        // telemetry sink (JSON Lines) and run identity
    FILE * tel_out = nullptr;
    std::string tel_path;
    uint64_t run_id = 0;
    int64_t t0_us = 0;
    bool telemetry_failed = false;

    // per-layer timing accumulation (filled by the executor)
    struct layer_stats {
        uint64_t n = 0;
        double transfer_us = 0.0;
        double wait_us = 0.0;
        double compute_us = 0.0;
        double setup_us = 0.0;
        double wall_us = 0.0;
        double transfer_min = 1e30, transfer_max = 0.0;
        double wait_min = 1e30, wait_max = 0.0;
        double compute_min = 1e30, compute_max = 0.0;
        uint64_t n_ready_at_consumption = 0; // copy completed before consumption
        uint64_t n_wait_at_consumption  = 0; // consumption blocked on the copy
    };
    std::vector<layer_stats> stats; // indexed by logical layer

    // totals for the phase summary (cumulative; reset at the PP->TG boundary)
    uint64_t total_bytes_transferred = 0;
    uint64_t total_evictions = 0;
    uint64_t stats_ubatch_base = 0; // n_ubatches at the start of the current phase

    // failure capture: the first error aborts the run with a structured event
    std::string error_msg;
    bool failed = false;

    // --- setup / build ---

    // create slots, buffers, events, copy backend, telemetry.
    // max_ubatch is the largest PP ubatch and max_n_kv is its padded mask width.
    // schedule selects the PP execution order; pp_tokens is the full prompt
    // length (drives the layer-major activation slabs when layer-major).
    bool init(const llama_model & model, int n_slots_, int device_, const char * tel_path_, int max_ubatch, int max_n_kv,
              llama_pp_schedule schedule_ = LLAMA_PP_SCHEDULE_BATCH_MAJOR, int64_t pp_tokens_ = 0,
               bool use_pinned_staging_ = false, int telemetry_chunk_stride_ = 10,
               bool use_implicit_causal_mask_ = false, bool pw_probe_ = false,
               ggml_type act_slab_dtype_ = GGML_TYPE_F32, int tg_max_width_ = 1,
               bool copy_src_registered_ = false, bool copy_src_reg_churn_ = false);

    // T-GPLR-37 churn-free variant: register every executed layer's mapped
    // source span ONCE, before the timed PP, so the pass never touches page
    // tables. Totals are returned for telemetry; failure is loud and rolls back
    // what it registered, leaving the box unpinned. Public because the runner
    // must call it after the prebuild (the packages exist only then) and before
    // the timed region starts.
    bool register_all_pkg_sources(int64_t & bytes_out, double & us_out);
    void unregister_all_pkg_sources();
    // T-GPLR-56 A3: the paced variant of the call above. `set_pp_span_pace` arms it
    // (mode 2 is refused, not clamped, so a typo cannot silently run the default);
    // `register_paced_begin` replaces the one-shot call at the runner's registration
    // site and page-locks the FIRST group only; `register_paced_ahead(il)` is called
    // from stage_members() and issues every group whose first layer is within
    // `ahead` of il. With pp_span_pace == 0 these are never reached.
    bool set_pp_span_pace(int mode, int ahead);
    bool register_paced_begin(int64_t & bytes_out, double & us_out);
    bool register_paced_ahead(int il);
    // T-GPLR-56 A3: read-only accounting of the paced mode, so the runner can report
    // the TOTAL span a paced run is committing. It needs this because the paced call
    // returns only what it has pinned SO FAR (the first group), while the one-shot
    // call returned the whole span - the same number would otherwise be read as if the
    // registration had shrunk. All three are 0 while the mode is off.
    size_t  paced_reg_group_count()  const { return paced_groups.size(); }
    int64_t paced_reg_bytes_total()  const { return paced_bytes_total; }
    int64_t paced_reg_bytes_issued() const { return paced_bytes_issued; }

    // find or create the graph set for (mode, n_tokens, rs_z, n_outputs) and make it the
    // current build target. The set's result object and graph list are reset
    // because a rebuild is about to replace them; the in-flight build MUST use
    // gparams.res == set->res (the executor and prebuild guarantee this).
    // chunk_id >= 0 marks a layer-major per-chunk PP set (distinct non-aliasing
    // act-slab slices); -1 marks batch-major / TG sets.
    graph_set * begin_set(int mode, int64_t n_tokens, int32_t rs_z, int32_t n_outputs, int32_t chunk_id = -1);

    // T-GPLR-71: true when a set built against `built` may still be reused for
    // an execution that needs `required` columns.
    //
    // T-GPLR-80: this is EXACT EQUALITY, not `built >= required`. The original
    // coverage test answered a different question from the one the executor
    // asks, and the gap is the whole --kv-graph-pad failure:
    //
    //   * A graph's kq_mask is BAKED at its build width. Reuse is admitted only
    //     by can_reuse_kq_mask(), which is `kq_mask->ne[0] == mctx->get_n_kv()`
    //     -- exact. A set built at 262,144 columns is therefore not merely
    //     "wide enough" when the live view is 50,176; it is UNUSABLE, and the
    //     executor died on "graph set ... does not match current ubatch shapes".
    //     Measured 2026-10-02 (tg73-d1-pad, --kv-graph-pad 256): the phase-1 set
    //     is prebuilt BEFORE the pad is armed at the prefill->generation
    //     boundary (main.cpp:5797), so it carries built = n_ctx = 262,144 while
    //     the first TG token needs GGML_PAD(50001,256) = 50,176. `built >=
    //     required` passed, nothing rebuilt, and the shape guard refused.
    //   * The wider-built direction is also the dangerous one. If the exactness
    //     were ever relaxed to a range, a 262,144-column view reused against a
    //     50,176-column live read would silently drop the tail of the context --
    //     exactly the failure T-GPLR-71 exists to prevent. Equality cannot: equal
    //     widths cannot under-read.
    //
    // UNCHANGED from the coverage version, deliberately:
    //   required <= 0 : nothing to check (no attention cache in play), reuse.
    //   built    <= 0 : width UNKNOWN, so the match is UNPROVEN -> rebuild.
    //     (Not every build site records a width: graph_params() attaches
    //     `layer_streaming` unconditionally, so the reserve, speculative-
    //     optimisation and non-default-graph-type builds reach
    //     begin_build()/finish_build() too. An unrecorded width must never read
    //     as reusable.) An unknown width costs at most one extra rebuild,
    //     because the rebuild goes through layer_streaming_compute, which
    //     records the real width.
    //
    // LEGACY EQUIVALENCE: with the view pad unarmed (--kv-graph-pad 0) both
    // sides are the same constant (cparams.n_pad = n_ctx), so `==` and `>=`
    // agree on every input -- covered, uncovered, unknown and the no-cache
    // cases alike. The default run is therefore unchanged bit-for-bit; only the
    // armed case, which previously could not run at all, gains behaviour.
    //
    // COST OF THE ARMED CASE, stated: the rebuild happens once per pad block,
    // i.e. once per `pad` generated tokens (128 tokens at pad 256 is at most
    // one). Each rebuild re-derives the whole phase-1 set inside the timed TG
    // window. That is the price of not pinning every view to the whole cache,
    // and it is the trade T-GPLR-71 chose; within a block every token reuses one
    // shape, which is what makes shape-matched comparison possible at all.
    //
    // T-GPLR-71: the executor's guard condition as ONE named predicate, so a
    // host test exercises the shipped decision. A test that re-spelled this
    // `if` would prove nothing about production. Phase-gated on purpose: the
    // phase-0 sets are driven by the layer-major controller, whose per-chunk
    // causal narrowing is a separate mechanism that must not gain a rebuild
    // trigger here.
    static bool should_arm_view_rebuild(int phase, const graph_set * set, int32_t required) {
        return phase == 1 && set != nullptr && set->built &&
               !view_matches_live(set->n_kv, required);
    }

    static bool view_matches_live(int32_t built, int32_t required) {
        if (required <= 0) {
            return true;
        }
        if (built <= 0) {
            return false;
        }
        return built == required;
    }

    // T-GPLR-71: flag one phase-1 (TG) set whose built view no longer covers
    // the live one, and arm rebuild_replacement so the executor's next
    // build-on-miss for it takes begin_set's T-GPLR-47 fresh-context branch --
    // the SAME mechanism (and the same flag pair) T-GPLR-62 S2 uses after a
    // row-set narrowing, so there is no new lifecycle path and no re-run of
    // layer_streaming_prebuild (which would advance recurrent session state).
    // Emits one tg_view_rebuild event naming both widths, so a trace can count
    // rebuilds without any pw-probe gating. Cannot fail: the set is finished by
    // the caller's precondition.
    void arm_tg_view_rebuild(llama_layer_streaming::graph_set & set, int32_t built, int32_t required);

    // T-GPLR-47: construct a fresh graph_set OUTSIDE the live one (new ggml ctx
    // with the same no-alloc params, new llm_graph_result, act tensors REPLAYED
    // onto the old set's exact device regions -- never re-derived from a cursor,
    // never re-sliced) and move-assign it into the old set's slot, then release
    // the old state (ggml_free + delete). Vector element addresses are unchanged
    // by assignment, so every stored graph_set* survives. Fails loudly
    // (error_msg, null) on any invariant violation; the old set stays live.
    bool replace_set_with_fresh_rebuild(graph_set & old, int64_t max_nodes);

    // T-GPLR-62 S2: flag every phase-1 (TG) set as needing a fresh-context
    // rebuild, and arm rebuild_replacement so the executor's next build-on-miss
    // for one of them performs the replacement. Called by begin_row_set() right
    // after build_row_set_window() commits. Returns false (loudly) if any phase-1
    // set is not in a finished-build state, because the replacement refuses to
    // touch one.
    bool mark_row_set_sets_stale();

    // begin a streaming graph build for the given ubatch size and recurrent
    // state-zero variant; returns the set's io context (ctx0 for the builder).
    // mode is taken from `phase` (set by the runner/prebuild) and the chunk id
    // from `cur_chunk_id` (set by the caller before model.build_graph).
    ggml_context * begin_build(int64_t n_tokens, int32_t rs_z, int32_t n_outputs);

    // the set currently being built (the builder binds companion-graph
    // inputs through it, so it must be mutable)
    graph_set * cur_set() const { return cur; }

    // find an existing graph set (nullptr if absent); used by the executor to
    // select the prebuilt variant for the current ubatch
    graph_set * find_set(int mode, int64_t n_tokens, int32_t rs_z, int32_t n_outputs, int32_t chunk_id = -1);

    // create the next graph of the current build (in the set's ctx) and register
    // it. layer: -1 embed, il for a decoder layer, -2 output graph.
    ggml_cgraph * new_graph(int layer);

    // after the graph build completes: allocate all INPUT-flagged tensors from
    // the persistent host buffer (at the set's cursor) and run sanity checks.
    bool finish_build();

    // T-GPLR-71: stamp the width these graphs were built against onto the set
    // and CLEAR the pending value. Split out of finish_build() because the
    // reset is otherwise untestable: only two of the five build sites set
    // cur_build_n_kv, so the clear is the only thing stopping an
    // un-instrumented build from inheriting the PREVIOUS build's plausible-
    // looking width -- the defect the clear exists to prevent, and invisible
    // from finish_build()'s sanity returns.
    void record_built_view(graph_set & set);

    // device tensor bundle for a logical layer (for graph building).
    // During handoff, returns the handoff-layers bundle (full resident copy of
    // all 64 packages) instead of the slot bundle.
    const llama_layer & slot_layer(int il) const {
        // T-GPLR-16: partial handoff (NCMOE-style mixed placement). Decided
        // FIRST: layers below the migration split execute from their host-
        // mmap CPU tensors (attention/router/shared/norms of those layers
        // stay GPU-resident in the same llama_layer struct), layers at/above
        // the split fall through to the handoff-bundle branch. No phase gate:
        // begin_handoff_partial releases the slot window, so slot-bundle
        // routing is unreachable for a live partial handoff in any phase —
        // host routing is the data-correct choice everywhere.
        if (handoff_active && handoff_partial) {
            if (il < n_layer && il < handoff_partial_from) {
                return *host_layers[il];
            }
            if (handoff_layers.count(il)) {
                return handoff_layers.at(il);
            }
            // else: il == n_layer (MTP) — refused with partial handoff at the
            // runner gate; route to host like the no-mtp case below.
        }
        if (handoff_active && (!handoff_attention_only || handoff_layers.count(il))) {
            return handoff_layers.at(il);
        }
        // T-Q38HG-04: attention-only handoff — recurrent layers route to
        // CPU-resident host mmap tensors so the ggml scheduler places them
        // on CPU instead of CUDA. The MTP layer (il == n_layer) is a
        // full-attention block; it routes here like the same attention
        // layers unless --mtp migrates it via handoff_mtp.
        if (handoff_active && handoff_attention_only && il < (int) recr.size() && recr[il]) {
            return *host_layers[il];
        }
        // T-Q38HG-06: the MTP layer (il == n_layer) is never in a slot
        // bundle. When handoff is not active, or attention-only without
        // --mtp, route to the host model tensors (the scheduler copies them
        // on demand).
        if (il >= n_layer) {
            return *host_layers[il];
        }
        return slots[il % n_slots].layers.at(il);
    }

    // package byte size for a logical layer (copy size into its slot).
    // During handoff, returns the size from the handoff map (same value,
    // since handoff_layers mirrors the slot bundles' byte sizes).
    // In attention-only mode, handoff_bytes only has attention layers.
    size_t package_bytes(int il) const {
        if (handoff_active && handoff_bytes.count(il)) {
            return handoff_bytes.at(il);
        }
        return slots[il % n_slots].layer_bytes.at(il);
    }

    // --- execution (called by llama_context::layer_streaming_compute) ---

    // schedule the async copy of logical layer il into its slot (does NOT
    // block; completion is awaited by wait_slot_ready). copied == false means
    // the slot already held this layer. returns false and sets error_msg on
    // failure.
    bool load_layer(int il, bool & copied);

    // host-blocking wait until the slot's copy for il is complete; measures
    // the availability wait (wait_us) and the submit-to-complete transfer
    // time (transfer_us). returns false and sets error_msg on failure.
    bool wait_slot_ready(int il, double & wait_us, double & host_submit_us, double & ready_latency_us);

    // record the compute-side consumer-done event for layer il's slot
    void record_consumer_done(int il, ggml_backend_t backend_compute);

    // --- telemetry ---

    // pp_tokens / pp_n_chunks / pp_chunks are the layer-major manifest (0 and
    // empty for batch-major); the host checker cross-checks act_bind chunk ids
    // and widths against it (T-Q38CTX-01).
    //
    // tg_tokens_ (T-GPLR-63 B) is the RESOLVED token-generation budget, and 0 is
    // a REAL value here, not a "unset" sentinel: it means the run has no TG
    // phase at all, which is the prefill-only case every CTest arm exercises.
    // It is recorded because run_end.n_tokens_generated cannot answer "did this
    // run have a TG phase" -- that counter is 0 for a run that FAILED before
    // generating anything, so using it as a schedule witness makes a crashed
    // prefill run look prefill-only for the wrong reason. -1 (a legacy caller
    // that never resolves the flag) leaves the key absent.
    void tel_run_start(const llama_model & model, size_t static_vram, size_t slot_vram, size_t free_vram, size_t total_vram,
                       const char * schedule, const char * copy_source, const char * kv_cache, size_t act_slab_bytes,
                       int64_t pp_tokens_, const std::vector<llama_ls_chunk> & chunks_,
                       ggml_type act_slab_dtype_ = GGML_TYPE_F32, int64_t ctx_size_ = -1,
                       const char * handoff_mode_ = "off", bool rowset_on_ = false,
                       int pp_prebuild_per_pass_ = -1, const char * pp_prebuild_per_pass_source_ = "",
                       const char * copy_source_source_ = "",
                       int route_telemetry_ = -1, const char * route_telemetry_source_ = "",
                       int tg_logit_margin_ = -1, int tg_tokens_ = -1);
    void tel_run_end(int exit_status, size_t peak_vram, int64_t n_tokens_generated, const char * err);
    // T-GPLR-22: a run-level refusal is a structured, TERMINAL record -- the
    // trace's last event, which closes the sink -- so a refused run (whether it
    // was refused before any work or aborted before the first timed phase) is
    // recoverable from the JSONL alone instead of reading as a crashed partial
    // trace. Not to be confused with tel_error(), which both checkers reject.
    void tel_run_refused(const char * reason, const char * detail, int exit_status,
                         uint64_t bytes_needed, uint64_t bytes_allowed, const char * cap_source);
    // T-GPLR-22 (PRE-SINK): the capacity/handoff preflights run before init()
    // opens the sink, so they have no instance to emit from. Writes the SAME
    // record through the same writer; false only when the file cannot be opened.
    static bool tel_run_refused_oneshot(const char * tel_path, const char * reason, const char * detail,
                                        int exit_status, uint64_t bytes_needed, uint64_t bytes_allowed,
                                        const char * cap_source);

    // T-GPLR-28: per-pass marker for --pp-repeat (in-process PP repeat);
    // emitted at the start of passes 2..N after the KV clear + stats reset.
    void tel_pp_pass(int pass_index, int pp_tokens);

    // T-GPLR-41 (diagnostics only, gated by pp_mctx_recon): pass-boundary
    // memory-lifecycle recon. Emits one mctx event per chunk after its apply
    // (live n_kv, recurrent rs_z, set selection) and one mctx_kvview event per
    // exec layer per chunk (the prebuilt graph's baked KV-view shapes and the
    // KV storage device addresses). Emitted in the mctx loop BEFORE the sweep,
    // so a crashing run still leaves a complete pass-2 picture.
    void tel_mctx_lifecycle(int pass_index, int64_t chunk, int64_t n_chunks,
                            int64_t n_tokens, uint32_t live_n_kv, int32_t rs_z,
                            int32_t set_chunk_id, int32_t set_rs_z, int32_t set_n_out);
    void tel_mctx_kvview(int pass_index, int64_t chunk, int il,
                         long long k_ne2, long long v_ne2,
                         const void * k_storage, const void * v_storage,
                         long long k_view_src, long long v_view_src);

    // T-GPLR-05: minimal telemetry for a forced `off` run (normal llama.cpp
    // placement, no streaming init). One-shot after the run: emits run_start
    // with policy "off" plus the stable capability reason, then run_end, and
    // closes the file. Returns false only when the file cannot be opened.
    bool tel_off_run(const char * tel_path_, const char * arch, const char * reason,
                     int exit_status_, int64_t n_generated, int64_t t0_us_,
                     int64_t pp_tokens_, int64_t tg_tokens_, int64_t ctx_size_);
    // T-GPLR-05: run_end notes for a successful run emit "error":"none"
    // (not an empty string) so consumers cannot misread absent diagnostics as
    // a failure indicator.
    void tel_slot_request(int il, int phase, size_t bytes);
    void tel_copy_submit(int il, size_t bytes, uint32_t generation, int af_inflight = 0, int submit_batch = 0);
    void tel_copy_complete(int il, double transfer_us, double host_submit_us, double ready_latency_us, uint32_t generation);
    void tel_layer_ready_wait(int il, double wait_us, bool ready_at_consumption, uint32_t generation);
    // Optional F3/H8 part split: pass parts >= 0 (row-set TG only) to append the
    // part A attn+router / part B FFN timings; the default omits them, leaving
    // the record exactly as the other executors write it.
    // T-GPLR-72 host/device split INSIDE a layer's span. cpu_us is per-thread
    // CPU time consumed across the span (-1 = this executor does not
    // instrument it). dev is 1 when the layer's weights live in a handoff
    // bundle (device resident), else 0. Additive and defaulted to the
    // unmeasured sentinel so an uninstrumented executor is visible.
    // T-GPLR-75: attn_dev / ffn_dev carry the backend each PART of a layer ran
    // on, classified per graph (0 = all host, 1 = all device, 2 = MIXED,
    // -1 = not reported). They exist because the row-set executor dispatches
    // BOTH parts through ggml_backend_sched, so one per-layer `dev` cannot
    // describe it -- and before this the row-set call sites passed no `dev`
    // at all, so every row-set TG event carried dev = -1 ("unreported") and
    // the K=2 arms could not say whether they used CPU inference.
    // T-GPLR-61 structure pass: the layer_compute record's optional tail as a
    // record struct; defaults are the old parameter defaults (-1 / -1.0 = "not
    // reported"). Identity fields stay positional; emitted records are
    // byte-identical.
    struct tel_layer_compute_params {
        double attn_compute_us = -1.0;
        double attn_setup_us   = -1.0;
        double ffn_compute_us  = -1.0;
        double ffn_setup_us    = -1.0;
        double cpu_us          = -1.0;
        int    dev             = -1;
        int    attn_dev        = -1;
        int    ffn_dev         = -1;
        int    experts_total   = -1;
        int    experts_host    = -1;
        int    experts_device  = -1;
        double host_us         = -1.0;
        // T-GPLR-61 S4/S1: the gate's own record, emitted even
        // when the split never fires (SDD moe-row-split telemetry).
        int    rows_gate_cpu   = -1;
        int    rows_gate_gpu   = -1;
        double gate_host_us    = -1.0;
        double gate_wait_us    = -1.0;
        double staged_bytes_avoided = -1.0;
        // T-GPLR-61 step C: the non-finite scan result. -1 = not scanned
        // (no host leg), 0 = scanned clean, 1 = a non-finite value was
        // caught (recorded, then the run aborts with a structured error
        // naming layer, token and expert ID).
        int    host_finite     = -1;
    };
    void tel_layer_compute(int il, double compute_us, double setup_us, uint32_t generation, int64_t chunk,
                           const tel_layer_compute_params & params = {});
    // T-GPLR-76: WHICH nodes made a graph "mixed". attn_dev/ffn_dev say a
    // graph straddled host and device; these say which OPS sat on the host
    // side, because that is the actionable half. Accumulated across every
    // layer of the first row-set round and emitted once, so a trace answers
    // "what runs on the CPU in here" without a per-layer flood.
    void tel_graph_backend_ops();
    void accum_graph_ops(const struct ggml_cgraph * g, bool part_a);
    // T-GPLR-53 A2 (diagnostics only, self-gated on --pp-recon): step 1 of
    // layer-major PP stages no weights, so it produces no layer_compute record.
    // One embed_chunk per chunk carries that chunk's own span and its split.
    void tel_embed_chunk(int64_t chunk, int64_t n_tokens, double setup_us, double compute_us, double span_us);
    void tel_slot_evict(int prior_layer, int slot, uint32_t generation);
    // T-Q38CAP-01: persistent graph-set footprint after the prebuild (before
    // the PP timer): evidence of the reserved/used io-context memory, host
    // input usage, slab and slot VRAM
    void tel_graph_footprint(int64_t n_graph_sets, int64_t total_graphs, size_t io_reserved_bytes,
                             size_t io_used_bytes, size_t host_input_bytes, size_t host_input_used_bytes,
                             size_t act_slab_bytes, size_t slot_vram_bytes, int64_t graph_budget_mib);
    // T-GPLR-01: emit the adapter-selected package catalog (per-layer class,
    // member count, package bytes) for host-side parity validation.
    void tel_pkg_catalog();
    // layer-major: record that a chunk's act pair was bound to a fixed slab
    // offset (slab 0 = A, 1 = B); the host validator uses these to prove the
    // per-chunk slices are non-aliasing
    void tel_act_bind(int32_t chunk_id, int slab, size_t off_bytes, int64_t n_tokens);
    void tel_error(const char * msg);
    // T-GPLR-04: explicit out-of-core capacity proof. One structured event
    // after the runtime preflight with the full byte-class accounting; the
    // host checker validates the additive sums (static + per_layer + slots +
    // kv + state == bounded) and the full/bounded/physical relationship.
    // T-GPLR-19 adds the two fields that make the TOTAL residency cap provable
    // from this record alone: `row_cache_bytes` (the cache's provisioned
    // private VRAM) and `residency_cap_bytes` (the operator's explicit
    // --residency-budget-mib in bytes; 0 = no explicit cap, the cache is
    // free-derived). The checker holds bounded + row_cache under the cap and
    // ties row_cache to the declared retention policy.
    void tel_byte_account(uint64_t full_model_plus_runtime, uint64_t bounded_window,
                          uint64_t static_bytes, uint64_t per_layer_bytes, uint64_t slot_window_bytes,
                          uint64_t kv_bytes, uint64_t state_bytes, uint64_t act_slab_bytes,
                          uint64_t reserve_bytes, uint64_t free_vram_bytes, uint64_t total_vram_bytes,
                          int64_t residency_budget_mib,
                          uint64_t row_cache_bytes, uint64_t residency_cap_bytes,
                          uint64_t prediction_pool_bytes = 0);
    // T-GPLR-36 follow-up: host-memory samples, one pair per PP pass, emitted
    // under --pp-recon exactly like the T-GPLR-36 ledger (an unknown event type
    // is ignored by the checkers, so a recon-only event cannot disturb an
    // existing gate). `host_bytes_accounted` is the engine's OWN host-side byte
    // holdings at the same instant; the emitter compares it against the process
    // delta, which is what makes a pass-boundary RSS rise attributable rather
    // than merely visible.
    void     tel_host_mem_begin();       // one pass boundary; bumps the pass number
    void     tel_host_mem_end();
    uint64_t host_bytes_accounted() const;
    void tel_tg_tokens(const std::vector<int64_t> & ids);
    // T-GPLR-54 A2: one read-only diagnostics line per greedy step, for the
    // first tg_logit_margin steps. The values are the SAME penalized top-1 and
    // top-2 the pick consumed (read before the sampler state advances), so the
    // event annotates the pick; it never perturbs it.
    void tel_tg_logits(int step, int64_t top1_id, float top1_logit, int64_t top2_id,
                       float top2_logit);
    // T-Q38HG-06: MTP draft/accept counters (drafted, accepted, acceptance rate)
    void tel_mtp(int64_t n_tokens, int64_t drafted, int64_t accepted);

    // Reset per-layer stats and cumulative phase counters so the next
    // phase_summary reports only that phase's data (call at the PP->TG boundary)
    void reset_stats();

    // phase summary (computed by the runner from stats); builds_at_phase_start
    // is the build_generation captured before the phase timer, graph_prep_us
    // is the pre-timer graph preparation time for this phase
    void tel_phase_summary(const char * phase, int64_t tokens, int64_t wall_us, uint64_t builds_at_phase_start, int64_t graph_prep_us);

    // accumulate one layer's timings
    void add_layer_stats(int il, double transfer_us, double wait_us, double compute_us, double setup_us, double wall_us, bool ready_at_consumption);

    // whether the recurrent mask says a layer is recurrent
    bool is_recr(int il) const { return recr[il] != 0; }

    // --- handoff (T-Q38HG-02 / T-Q38HG-04 / T-Q38HG-06) ---

    // Allocate a single CUDA buffer, create handoff layer bundles aliasing it,
    // and copy packages from the host mmap to the device.
    // Default (attention_only=false): copy all 64 layers (T-Q38HG-02).
    // attention_only=true: copy only the 16 full-attention layers; recurrent
    // layers stay in slot bundles (T-Q38HG-04, constrained handoff).
    // mtp=true additionally includes the MTP layer package (blk.64, ~235 MB)
    // in the attention-only set so the MTP draft head runs device-resident
    // (T-Q38HG-06). Returns false on allocation/copy failure (error_msg set).
    bool begin_handoff(bool attention_only = false, bool mtp = false);

    // T-GPLR-16: partial handoff (NCMOE-style mixed placement) for a
    // merge-resident (moe-expert-bank) policy. Migrates ONLY the routed-
    // expert bank packages with il >= partial_from into the handoff buffer;
    // banks of lower layers keep their host-mmap CPU tensors (the loader
    // override stages every bank on CPU). During TG (phase 1) those lower
    // layers execute their FFN on CPU while attention/router/shared-expert
    // stay GPU-resident; during PP the slot window banks all layers
    // unchanged. One call allocates and copies exactly n_layer - partial_from
    // banks. attention_only must be false (refused); mtp is refused (stock
    // decode has no draft loop under moe-expert-bank). Returns false with
    // error_msg set on any failure.
    bool begin_handoff_partial(int partial_from_);

    // T-GPLR-16: clear all partial-handoff state after a failed
    // begin_handoff_partial. The slot buffers/slab released by the peak-order
    // rule are not restorable; the runner must treat the failure as terminal.
    void abort_handoff_partial();

    // Resolve the device layer bundle a graph should build against for a
    // logical layer index, including the MTP layer (il == n_layer).
    // Routing mirrors slot_layer(): during a handoff with the package present
    // in the handoff buffer, the handoff bundle is returned; otherwise the
    // host mmap tensor (the plain graph path reads model.layers[il] directly
    // when no streaming module is attached).
    const llama_layer & accessor_layer(int il) const { return slot_layer(il); }

    // Free the handoff buffer and bundles, deactivate the handoff path.
    void end_handoff();

    // True after begin_handoff() succeeds and before end_handoff().
    bool handoff_active = false;

    // True when the handoff covers only attention layers (T-Q38HG-04).
    bool handoff_attention_only = false;

    // Device buffer holding all 64 packages during handoff.
    ggml_backend_buffer_ptr buf_handoff;

    // Meta context owning the handoff tensor objects (handoff_layers bundles).
    ggml_context * ctx_handoff = nullptr;

    // Per-layer byte size in the handoff buffer (mirrors package_bytes).
    std::unordered_map<int, size_t> handoff_bytes;

    // Per-layer tensor bundles aliasing buf_handoff (like slot.layers, but
    // for all 64 layers simultaneously). The MTP layer, when migrated, is
    // present under index n_layer.
    std::unordered_map<int, llama_layer> handoff_layers;

    // T-Q38HG-06: per-layer byte size of the MTP package (present when the
    // MTP layer was migrated into the handoff buffer).
    size_t mtp_bytes = 0;

    // T-GPLR-01: validated MTP package descriptor used by begin_handoff(mtp).
    llama_ls_package handoff_pkg_mtp;

    // T-GPLR-16: partial handoff (NCMOE-style mixed placement). When true,
    // begin_handoff migrated only the routed-expert banks with il >=
    // partial_from; lower layers route to their host-mmap tensors in
    // slot_layer() so the ggml scheduler places them on CPU.
    bool handoff_partial = false;
    int  handoff_partial_from = 0;

    // T-GPLR-13: reactive expert row-set mode. begin_row_set() allocates one
    // full-size expert-bank buffer per slot (bundle shapes/types are the
    // complete bank, so graph topology is unchanged) and after PP the TG path
    // copies only the deduplicated routed expert ROWS per layer/window into
    // them (rowset_fetch events). The layer graph is built in two parts:
    // attn+router (which yields the router's exact top-k IDs) and FFN bound
    // to a host-supplied selected_experts input. Routing stays bit-exact:
    // before any FFN runs, every routed row of THIS graph's own router output
    // is present or copied (missing rows are copied; present rows are never
    // re-copied within one generation), so no wrong row can ever execute.
    // Set by the runner BEFORE init/prebuild when the moe-row-set TG policy
    // was requested. TG graph sets are then built in two parts; activation
    // still happens later via begin_row_set() (after PP). Default-off runs
    // never set this and never see two-part graphs.
    bool     row_set_requested = false;
    bool     row_set_active    = false; // TG row-set staging active
    // per-SLOT row coverage (NOT per layer): a bounded slot window serves
    // several logical layers per token (il % n_slots), so the resident-row
    // bookkeeping belongs to the slot with an owner-layer guard. When the
    // owner changes (another layer took the slot), the coverage map is reset
    // and every needed row re-stages — foreign rows can never be claimed.
    std::vector<std::vector<uint8_t>> row_set_have;  // per slot, per expert row resident
    std::vector<int>                  row_set_owner; // per slot, owning layer (-1 = none yet)
    std::vector<uint32_t> row_set_gen;  // per-slot generation the coverage belongs to
    std::vector<std::vector<int32_t>> row_set_ids; // per layer, last staged union (informational)

    // T-GPLR-62 S2: the union POSITION lookup table for the most recent
    // successful load_row_set, and whether it has ever been written. Sized
    // n_expert; every entry is -1 except the |uniq| union members, which carry
    // their index in the union. This is the ONLY sanctioned way to turn a
    // router-selected expert id into a slot storage position, and it is
    // re-initialised in full on every window precisely so that a miss reads -1
    // (a refusal) rather than a position left over from a previous window.
    std::vector<int32_t> row_set_union_pos_;
    bool                 row_set_union_valid_ = false;

    // T-GPLR-19 (F2-M1): per-layer row residency, SDD section 4 execution
    // shape (i) — "cache to slot-bundle device-side refresh before the
    // row-consuming phase". The slot buffers above are keyed by SLOT and every
    // layer assigned to a slot aliases one device buffer at offset 0, so an
    // intervening owner overwrites a logical layer's rows: cross-token reuse
    // is structurally zero (rows_reused == 0). With retention on, each logical
    // layer ALSO owns private CUDA storage holding up to K*W*top_k expert
    // slabs per member, so a window stages only the complement of the retained
    // union, H2D into the cache, and refreshes the layer's slot bundle
    // device-side from that cache before the FFN phase. The cache is a COPY
    // SOURCE, never a graph tensor: the FFN still consumes the slot bundle
    // with the router's own IDs, so routing, quantization bytes and every
    // arithmetic result are untouched. Default-off allocates nothing.
    bool row_set_retain   = false;  // shape (i) retention requested (default off)
    int  row_set_retain_k = 4;      // K windows retained; 4 = the default depth

    // T-GPLR-82 S2: routed expert rows per (layer, token) that the HOST leg
    // computes instead of the device leg. 0 = off, which is the default and
    // reproduces the pre-S2 graph exactly (no host leg is built at all). Must
    // be strictly less than n_expert_used or no device row is left to compute;
    // the graph builder refuses that. S2 is COUPLED: the device leg still waits
    // for the same row-set readiness it always did, so this buys no speedup
    // yet -- it exists so the host arithmetic is real, guarded and measured
    // before S3 removes the wait.
    int  experts_host_n   = 0;
    // T-GPLR-82 S2: guard counters. host_leg_builds counts host sub-ops that
    // were actually constructed; host_leg_weights_bound counts host-leg weights
    // the rewire guard CHECKED and found still host-placed. A bound count of 0
    // with a non-zero --experts-host means the guard never ran, which must not
    // be mistaken for a pass.
    uint32_t host_leg_builds        = 0;
    uint32_t host_leg_weights_bound = 0;
    // T-GPLR-61 S4: --experts-host-budget B, microseconds of host expert work
    // per token that the S3 copy-wait decoupling may spend (one admitted row =
    // the measured 36.5 us/expert). 0 (default) admits nothing and keeps every
    // blocking wait byte-for-byte as it was; HUGE_VAL admits every missed row
    // the fixed host leg can hold. Refused without experts_host_n > 0.
    double experts_host_budget_us = 0.0;
    // T-GPLR-82 FALSIFIER (2026-10-10, diagnostic, default false): build the
    // split's TAIL leg on the backend that owns the slot's staged weights (the
    // device) instead of on the CPU. It exists to break a confound, not to place
    // work: the shipped host leg is CPU-placed because it binds the MODEL
    // LOADER's host-placed expert banks (qwen35moe.cpp, build_moe_host_leg), so
    // the split's measured numeric signature (step-0 logits 16.167877 against the
    // control's 16.194481) cannot be attributed to the CPU without a partition
    // that keeps every other property -- the fixed widths, the ONE w_full
    // derivation, the positional suffix, the add-chain reduction -- and moves
    // only the backend. This flag is that partition. It is NEVER a placement
    // policy and must not be read as one: computing the tail on the device is the
    // opposite of the mechanism's intent. It requires experts_host_n > 0 (refused
    // at bring-up otherwise), it builds NO host sub-op and NO hsel input, and it
    // leaves row_set_split_built at 0 for every layer so the executor's host-side
    // paths (plan reorder, id fill, per-row scan) stay off and the ledger
    // honestly reports 0 host rows and host_finite -1.
    bool experts_host_leg_device = false;
    // after PP, run TG decode through the stock graph path (the AER hybrid engine hooks it) instead of this executor
    bool tg_stock = false;
    // frees the PP slot window and activation slabs, then routes decode through the stock graph path; call once after PP
    bool begin_tg_stock();
    // T-GPLR-61 S3: per slot, per expert: copies ISSUED but never observed
    // complete on the host (a skipped wait leaves them outstanding). An
    // outstanding row classifies PENDING -- it may compute on the host leg,
    // never on the device leg, until a synchronization of that slot's
    // ev_copy_ready (which covers every earlier record on the copy stream)
    // clears the whole slot. Cleared wherever row_set_have is cleared.
    std::vector<std::vector<uint8_t>> row_set_outstanding;
    // T-GPLR-61 S3: per layer, per expert classification stashed by
    // load_row_set AFTER staging for the executor's ls_plan_split and fill:
    // the values of llama_ls_policy::ls_row_state (0=READY, 1=PENDING,
    // 2=ABSENT). ABSENT cannot occur post-staging; the default of 2 makes a
    // read before any load fail loudly instead of pretending READY.
    std::vector<std::vector<uint8_t>> row_set_state;
    // per layer, per expert: 1 when the slot slab holds this expert for the consume
    // that follows the layer's last load. Independent of row_set_state on purpose:
    // the executor checks it for every row the device leg reads
    std::vector<std::vector<uint8_t>> row_set_fresh;
    // T-GPLR-61 S4: the ls_evaluate_gate verdict for the layer's last load,
    // stashed so the executor's ls_plan_split and the layer_compute record
    // see the SAME evaluation the wait decision used.
    std::vector<uint8_t> row_set_gate_open;
    std::vector<double> row_set_gate_host_us;
    std::vector<double> row_set_gate_wait_us;
    // T-GPLR-82 S3/S4 (2026-10-10): whether the split FFN graph was actually
    // BUILT for this layer. The builder sets it when lp_split_in_play says the
    // mechanism can admit a row; the executor reads it instead of experts_host_n
    // so the two halves cannot disagree. Measured, budget at its default: with
    // experts_host_n > 0 alone the executor demanded a host-leg id input
    // (ff_input_hsel) that an unsplit graph never creates -- a loud abort
    // (rc=1, "host-leg id input missing or misshaped for layer 0 (expected [2, 1])").
    std::vector<uint8_t> row_set_split_built;
    // T-GPLR-62 S2: compact row-set WINDOW WIDTH, in experts. 0 (the default)
    // is the pre-S2 full-bank slot: ne[2] stays n_expert, the staging write
    // addresses the absolute expert id, and the graph consumes absolute ids.
    // Setting W>0 makes begin_row_set rebuild every slot bundle at ne[2] = W
    // into a fresh compact buffer, free the full-bank buffer, and from then on
    // the staging write addresses the union INDEX while the graph consumes
    // window positions gathered from row_set_union_pos(). The three changes are
    // made together in build_row_set_window(); setting this field alone is
    // inert, because the switch is only taken once the narrow bundles exist.
    int  row_set_window_width = 0;

    // T-GPLR-62 S2 (lifetime): a compact window replaces every slot bundle
    // tensor inside build_row_set_window(), which runs in begin_row_set() AFTER
    // the bring-up prebuild. The phase-1 (TG) sets were built before that and
    // had the FULL-BANK pointers baked into their MoE ops, so the swap left them
    // reading a released buffer -- a dangling reference, observed as an illegal
    // memory access in part B of layer 0.
    //
    // The fix is the T-GPLR-47 fresh-context replacement, NOT a deferred
    // prebuild. Deferring was tried and is wrong: llama_context::
    // layer_streaming_prebuild advances recurrent session state (each variant
    // calls vmctx->apply() and the pass ends in memory->clear(false)), so
    // running it after prefill made the timed TG decode observe rs_z=0 (fresh)
    // where it must observe -1 (continuing), and it generated from the wrong
    // recurrent state. A graph-shape probe must not mutate session state.
    //
    // So the TG sets are still prebuilt at bring-up, and immediately after the
    // narrowing commits, mark_row_set_sets_stale() flags every phase-1 set.
    // The executor then re-derives each flagged set through
    // replace_set_with_fresh_rebuild(): new ctx and result object, act regions
    // REPLAYED onto the same device addresses, graphs rebuilt against the
    // compact bundle, and the whole set move-assigned back into the same vector
    // slot so no graph_set pointer ever moves. That re-binds every pointer with
    // no state advance and no buffer reuse.
    //
    // No-op at W=0: build_row_set_window() is never called, nothing is flagged,
    // and every pre-S2 command line is unaffected.

    // T-GPLR-62 S2 (A9, the LOAD-BEARING gate): byte-level recon of the staged
    // window. OFF by default; when on, every staged slot slab is read back and
    // byte-compared against the checkpoint slab for the expert that position is
    // SUPPOSED to hold. A mismatch refuses the run with the stable reason
    // "recon_mismatch" rather than merely reporting, because a wrong expert in
    // the window produces a plausible answer and the run's output would be
    // worthless. This is the gate that route parity cannot replace: parity only
    // detects nondeterminism, and a deterministic mapping error passes it.
    bool row_set_recon = false;
    // A9 counters, reported at the end of the run.
    uint64_t rowset_recon_windows  = 0;
    uint64_t rowset_recon_members  = 0;
    uint64_t rowset_recon_compared = 0;
    uint64_t rowset_recon_bad      = 0;
                                    // (32 rows/layer, ~2.19 GiB over 40 layers,
                                    // the shallowest K whose offline (b) clears
                                    // the 0.6 gate). A caller may name any 1..64.
    int  row_set_cap      = 0;      // rows per layer = min(n_expert, K*W*top_k)

    // T-GPLR-76: per-op host/device node census for the row-set TG graphs.
    // attn_dev/ffn_dev say a graph STRADDLED host and device; these say WHICH
    // OPS sat on the host side, which is the actionable half. Accumulated
    // across every layer of the first row-set round and emitted once.
    std::vector<std::pair<std::string, int> > rowset_host_ops_a;
    std::vector<std::pair<std::string, int> > rowset_dev_ops_a;
    std::vector<std::pair<std::string, int> > rowset_host_ops_b;
    std::vector<std::pair<std::string, int> > rowset_dev_ops_b;
    // T-GPLR-79: op NAMES of the host-side nodes. The op census answers "which
    // op"; this answers "which tensor", which is what identifies a state
    // tensor or a staging buffer. Host-only, so it stays small even if a graph
    // is mostly device.
    std::vector<std::pair<std::string, int> > rowset_host_names_a;
    std::vector<std::pair<std::string, int> > rowset_host_names_b;
    bool rowset_ops_emitted = false;

    // per layer: private bank storage, one cache tensor per package member
    // (shaped [ne0, ne1, cap] so a position is addressable at pos*slab), and
    // the provenance that makes a resident claim sound: position -> expert and
    // the window index the slab serves (-1 = free). The cache is a SLIDING
    // union of the last K windows: a row that is routed again has its
    // provenance PROMOTED to the current window, and a row that can no longer
    // be claimed is retired before placement, so a stale or foreign slab can
    // never satisfy a fetch and the retained count can never exceed what the
    // declared K permits.
    std::vector<ggml_backend_buffer_ptr>    cache_bufs;   // per layer private storage
    std::vector<std::vector<ggml_tensor *>> cache_tens;   // per layer x member cache tensors
    std::vector<std::vector<int32_t>>       cache_expert; // per layer, pos -> expert (-1 free)
    // per layer, pos -> RECENCY stamp (larger = more recently routed, -1 free).
    // T-GPLR-19 policy resolution (2026-09-13): the cache is a row-capacity LRU
    // over the layer's routed rows, NOT a window-count membership set. Its size
    // is the same cap = min(n_expert, K*top_k) rows either way, so recency eviction
    // costs nothing and weakly dominates: every row in the last K windows is
    // among the cap most recently routed (at most cap-1 rows were routed after
    // it), so union-of-K-windows is a SUBSET of the LRU set at equal capacity.
    // The recency value is a monotonic per-run counter, never a window index.
    std::vector<std::vector<int64_t>>       cache_recency;
    int64_t                                 cache_tick = 0; // monotonic recency source
    bool cache_active = false;
    // owns the cache tensor objects (persistent) and the per-window slab views
    // used by the device-side refresh (re-created when exhausted)
    ggml_context * cache_meta      = nullptr;
    ggml_context * cache_view_ctx  = nullptr;
    size_t         cache_view_used = 0;
    size_t         cache_view_cap  = 0;
    // T-GPLR-19: the private VRAM begin_row_cache() ACTUALLY allocated, summed
    // over every layer (0 when retention is off / no cache is live). Recorded
    // from the allocation itself, never re-derived, so the byte_account figure
    // says what exists; the runner cross-checks it against
    // row_cache_bytes_for()'s prediction for the resolved depth, and the host
    // checker holds bounded residency + this figure under the operator's total
    // residency cap. It stays 0 until an allocation succeeds: a cache that
    // never allocated must not report bytes.
    size_t         row_cache_alloc_bytes = 0;
    // -----------------------------------------------------------------------
    // T-GPLR-58 A0 (2026-09-23): speculative placement on the retention path.
    //
    // Default-off. Mode 0 leaves every array below EMPTY and every guard inert
    // by construction (no position is ever PENDING and no event is ever
    // created), which is what P6's byte-identity claim rests on.
    //   mode 1 = scripted proposals (A0's test vehicle: the sequence comes from
    //            a file, so the mechanism is exercised with no predictor)
    //   mode 2 = RESERVED for A1's real proposer, and REFUSED today, loudly, so
    //            a scripted run can never be mistaken for a predictor run.
    // Guesses are placed ONLY in the per-layer row cache (the retention path);
    // the slot-keyed reactive path has no per-layer residency to speculate into,
    // so modes 1/2 precondition on retention and are refused otherwise.
    //
    // ONE OWNER PER FACT: `cache_expert` / `cache_recency` remain the sole owners
    // of which expert sits at a position and how recently it was routed. The
    // arrays below add the speculative STATE only, indexed by the same position,
    // and are empty whenever the mode is off.
    //
    // P1's guard is `spec_in_demand_window`: a demand's own row wait must never
    // cover a speculative copy's completion, so a reap taken while that window is
    // open is COUNTED in `spec_waited_by_demand` instead of being assumed away.
    // It reads 0 in a correct run and non-zero the moment a reap moves inside the
    // window -- the falsifier the invariant asks for.
    // -----------------------------------------------------------------------
    int          row_set_spec_mode    = 0;   // 0 off, 1 scripted, 2 reserved (refused)
    int          row_set_spec_budget  = 0;   // max PENDING fills per layer (0 = auto = top_k)
    int          row_set_spec_reserve = 0;   // positions ALWAYS kept free for demand (top_k)
    std::string  row_set_spec_script;        // path, recorded for the run_start echo
    bool         row_set_spec_on       = false;
    bool         spec_in_demand_window = false;  // P1 guard, see above
    std::vector<std::vector<uint8_t>>  spec_state;  // per layer, pos -> llama_ls_spec_state
    std::vector<std::vector<uint8_t>>  spec_flag;   // per layer, pos -> placed by speculation
    std::vector<std::vector<uint32_t>> spec_gen;    // per layer, pos -> slot generation at issue
    std::vector<std::vector<int64_t>>  spec_tick;   // per layer, pos -> recency to stamp on completion
    std::vector<std::vector<void *>>   spec_ev;     // per layer, event slot -> fill's OWN completion event
    std::vector<std::vector<int32_t>>  spec_ev_pos; // per layer, event slot -> position it serves (-1 free)
    std::vector<int>                   spec_calls;  // demand stages per layer (script ordinals)
    // T-GPLR-58 A2: per layer, pos -> the demand-stage ORDINAL that issued the
    // fill now sitting there (-1 = no fill, or already consumed). A1/A2 compute
    // M7's lead from this: lead = (stage that consumed) - (stage that issued), so a
    // fill issued at the end of stage t and consumed by stage t+2 has lead 2.
    std::vector<std::vector<int32_t>>  spec_issue_ord;
    // scripted proposals: key = (layer << 32) | ordinal -> the ids to speculate
    // at the END of that layer's (ordinal-1)-th demand stage
    std::unordered_map<uint64_t, std::vector<int32_t>> spec_script;
    // counters, emitted on every demand stage by tel_spec(). The stage_scoped ones
    // are per-stage; the rest are run totals, so a reader never has to guess the grain.
    int64_t spec_submitted = 0;          // fills admitted and issued, RUN TOTAL
    // T-GPLR-58 A2: the `submitted` field of a `spec` event is PER-STAGE -- the fills
    // THIS stage issued, and by the admission rule never more than the budget. The run
    // total (spec_submitted) is emitted beside it as `submitted_total`. Measured on the
    // 2026-09-23 K-probe: a run-cumulative value on a per-stage event read as a +306
    // increment against a budget of 8, because between two events of one layer every
    // OTHER layer's stage in that token also issued its fills into the same counter.
    int64_t spec_stage_submitted = 0;    // fills issued during THIS demand stage
    // The ids THIS stage actually issued, in issue order, as `fill_ids`. The runner's
    // retention cache model (ls_retention_errors) replays them, so a spec event without
    // them cannot reproduce the layer's occupancy: that is what made every fetch after a
    // fill-issuing stage look like a retention-accounting violation (216 on the recorded
    // null@K8 leg, 2026-09-23) while the runtime's own numbers were self-consistent.
    std::vector<int32_t> spec_fills_last;
    // Per-stage planner telemetry. These explain an empty issue list without requiring
    // a second admission pass. They are populated only for mode 2.
    std::vector<int32_t> spec_candidates_last;
    int spec_skipped_present_last      = 0;
    int spec_skipped_dup_last          = 0;
    int spec_plan_refused_capacity_last = 0;
    int spec_plan_stop_last             = 0;
    int64_t spec_ready     = 0;          // fills whose copy completed and were claimed resident
    int64_t spec_cancelled = 0;          // fills abandoned before being needed (P4)
    // P4 wants the FIRING POINT named, not just the total (colibri's UPDATE101
    // recorded predictions_cancelled=0 in every leg, which is an untested hook,
    // not a passing one): one counter per cancellation point.
    int64_t spec_cancelled_capacity   = 0; // reclaimed for a demand's capacity
    int64_t spec_cancelled_demand     = 0; // the demand asked for the very row in flight
    int64_t spec_cancelled_generation = 0; // issued under a stale slot generation
    int64_t spec_refused_capacity = 0;   // admissions refused by the P3 reserve/budget
    int64_t spec_dup_refused = 0;        // admissions refused because the row was present/pending
    int64_t spec_matched_by_demand = 0;  // fills a demand asked for while still PENDING (P2 fallback)
    int64_t spec_waited_by_demand = 0;   // P1 falsifier: reaps inside a demand window (stays 0)
    int64_t spec_reap_wait_us = 0;       // host time waiting at reaps (speculation path only)
    int64_t spec_issued_bytes = 0;       // bytes H2D'd by speculative fills
    // T-GPLR-58 A2: M7's datum. `lead` counts DEMAND STAGES OF THAT LAYER
    // between the stage that issued a fill and the stage whose demand was served
    // from it. Only a CONSUMED fill is recorded: a fill a demand meets while it is
    // still PENDING is the fallback (spec_matched_by_demand, effective lead 0) and
    // is deliberately NOT folded into this distribution, so the histogram is the
    // whole evidence M7 asks for. Bucket 7 = lead >= 8.
    int64_t spec_consumed  = 0;          // fills a demand was actually served from
    int64_t spec_lead_sum  = 0;          // sum of lead over consumed fills
    int32_t spec_lead_min  = 0;          // 0 = none consumed yet
    int32_t spec_lead_max  = 0;
    std::vector<int64_t> spec_lead_hist; // 8 buckets: lead 1..7, then >= 8

    // T-GPLR-58 MODE 2: the ENGINE-CONSUMED proposer. `row_set_spec_arm` < 0 means
    // no proposer (mode 1's script is the source); otherwise it is a
    // llama_ls_pred_arm and the engine asks it at the end of every demand stage,
    // through the same planner and the same admission rules the scripted source
    // uses. The learning state is ONE per run, indexed by layer, and is allocated
    // only for the families the arm actually reads -- the null needs ALL of them,
    // because its count IS the arm's count and the arm must be computed to have
    // one. `spec_null_rng` is the null's fixed-seed xorshift state, seeded once
    // from llama_ls_pred_null_seed(), so a run is reproducible and a null leg is
    // identifiable from its own trace.
    int          row_set_spec_arm  = -1;     // llama_ls_pred_arm, or -1 = no proposer
    bool         row_set_spec_null = false;  // proposal source = the count-matched null
    std::unique_ptr<llama_ls_pred_state> spec_pred;  // past-only mode 2: allocated by set_row_set_spec_arm()
    // T-GPLR-59 D1: proposed-future source. It shares mode 2's placement state and
    // planner, but owns a separate sparse table and publishes at the MTP
    // pre-verify boundary rather than after a demand stage.
    std::unique_ptr<llama_ls_d1_state> d1_state;
    int          row_set_window_tokens = 1;  // target rows in one row-set demand (MTP: 1 + draft depth)
    std::vector<std::vector<std::vector<int32_t>>> d1_last_routes; // [target row][layer]
    std::vector<std::vector<std::vector<int32_t>>> d1_last_candidates; // [draft index][layer]
    bool         d1_shape_emitted = false;
    uint32_t     d1_round = 0;
    uint32_t     spec_null_rng     = 0;      // mode 2 + null: the xorshift state
    // Reported per stage by tel_spec(): what the SOURCE published, and -- for a null
    // leg -- the ARM's count it was matched to. Emitted so count-matching is
    // checkable from the trace alone, never inferred from a filename.
    int          spec_prop_last      = 0;
    int          spec_arm_count_last = 0;

    // Configure speculative placement. Validated here AND by the runner before
    // the model load; returns false with a structured error rather than silently
    // degrading to mode 0.
    bool set_row_set_spec(int mode, int budget, const std::string & script_path);

    // Declare the number of target rows staged by one row-set demand. The
    // ordinary TG path is one; MTP batch verify is 1 + the configured draft
    // depth. Must be set before set_row_set_spec() so the demand reserve covers
    // the full possible union.
    bool set_row_set_window_tokens(int tokens);

    // MODE 2's own configuration, called after set_row_set_spec(2, ...) and
    // before begin_row_cache() allocates the cache: WHICH arm proposes and
    // whether the proposals come from the arm or from the count-matched null.
    // Allocates the arm's learning state; refuses an arm the port does not
    // define, and refuses the null without an arm to match (the null takes its
    // count FROM the arm, so there is nothing to match otherwise).
    bool set_row_set_spec_arm(int arm, bool null_source);

    // Issue the fills that follow layer il's demand stage: mode 1 takes them from
    // the script's entry for (il, spec_calls[il] + 1), mode 2 asks the arm --
    // observing the row the demand just served and proposing for this layer's
    // NEXT demand row (llama_ls_spec_propose). Reaps first (on this, the
    // speculation path), then admits and issues. No-op when the mode is off.
    // `demand_ids` / `n_demand` are the ids that demand just placed, in the
    // trace's order, and are only ever passed AFTER that demand's own wait closed.
    void spec_issue_next(int il, const int32_t * demand_ids, int n_demand);

    // T-GPLR-59 D1 engine wiring. The MTP runner calls d1_issue_preverify()
    // after drafting and before target verification, then d1_observe_round()
    // after the accept prefix is known. Publication uses the same real-cache
    // planner/count-matched-null path as mode 2; observation learns accepted
    // pairs only. Both methods fail loudly rather than falling back to a
    // past-only proposal.
    bool d1_issue_preverify(const std::vector<int32_t> & draft_tokens);
    bool d1_observe_round(const std::vector<int32_t> & draft_tokens,
                          const std::vector<uint8_t> & accepted);
    void d1_emit_summary();

    // Issue ONE speculative fill, or refuse it with a reason (P3/P5).
    bool spec_issue_one(int il, int32_t expert, uint32_t generation, int64_t tick);
    bool spec_ensure_events();

    // T-GPLR-58 A2 (M7): record the lead of one fill a DEMAND was served from,
    // and clear its issue ordinal. Returns false when the position holds no fill
    // (an ordinary resident placed by a demand), in which case nothing is counted.
    bool spec_note_lead(int il, int pos);


    // Complete the fills whose copies finished (claiming residency) and free the
    // cancelled ones. The ONLY place a PENDING position changes state.
    void spec_reap(int il);

    // Cancel the layer's OLDEST pending fills (0 = none by capacity pressure; a
    // generation boundary passes the whole pending count). Marks only: it never
    // waits, so it is safe inside a demand stage.
    void spec_cancel(int il, int need_free, const char * reason);

    // Pending / ready-speculative row counts for one layer (telemetry + planning).
    int spec_pending_of(int il) const;
    int spec_ready_of(int il) const;

    // Free the speculative state, events included; idempotent.
    void spec_reset();

    // T-GPLR-59: prediction-only row pool. This is deliberately separate from
    // spec_state/cache_expert: a pool row is never a real-cache residency claim.
    int  row_set_spec_pool = 0;             // positions per logical layer, 0 = off
    std::vector<ggml_backend_buffer_ptr>    pool_bufs;
    std::vector<std::vector<ggml_tensor *>> pool_tens;
    ggml_context * pool_meta = nullptr;
    std::vector<std::vector<int32_t>>       pool_expert;
    std::vector<std::vector<uint8_t>>       pool_state;
    std::vector<std::vector<int64_t>>       pool_recency;
    std::vector<std::vector<uint32_t>>      pool_gen;
    std::vector<std::vector<int32_t>>       pool_issue_ord;
    std::vector<std::vector<void *>>        pool_ev;
    std::vector<std::vector<int32_t>>       pool_ev_pos;
    bool pool_events_ready = false;
    std::vector<int>                        pool_calls;
    size_t row_set_spec_pool_alloc_bytes = 0;
    int64_t pool_submitted_last = 0;
    int64_t pool_submitted = 0;
    int64_t pool_ready = 0;
    int64_t pool_cancelled = 0;
    int64_t pool_cancelled_demand = 0;
    int64_t pool_cancelled_generation = 0;
    int64_t pool_cancelled_capacity = 0;
    int64_t pool_stale_retired = 0;
    int64_t pool_evicted_last = 0;
    int64_t pool_evicted = 0;
    int64_t pool_consumed_last = 0;
    int64_t pool_consumed = 0;
    int64_t pool_pending_fallbacks_last = 0;
    int64_t pool_pending_fallbacks = 0;
    int64_t pool_d2d_bytes_last = 0;
    int64_t pool_d2d_bytes_total = 0;
    int64_t pool_h2d_bytes_last = 0;
    int64_t pool_h2d_bytes_total = 0;
    int64_t pool_reap_wait_us_last = 0;
    int64_t pool_reap_wait_us = 0;
    std::vector<int32_t> pool_candidates_last;
    std::vector<int32_t> pool_fills_last;
    std::vector<int32_t> pool_hit_ids_last;
    int pool_skipped_real_last = 0;
    int pool_skipped_pool_last = 0;
    int pool_skipped_dup_last = 0;
    int pool_skipped_invalid_last = 0;
    int pool_refused_budget_last = 0;
    int pool_refused_capacity_last = 0;
    int pool_stop_last = 0;

    int pool_pending_of(int il) const;
    int pool_ready_of(int il) const;
    bool pool_cancel_demand(int il, int32_t expert, uint32_t generation);
    bool pool_issue_one(int il, int32_t expert, int pos, uint32_t generation, int64_t tick);
    void pool_reap(int il);
    void pool_begin_generation(int il, uint32_t generation);
    void pool_cancel_generation();
    void pool_reset();
    void tel_spec_pool(int il, uint32_t generation, int pending_now, int ready_now);
    size_t row_set_spec_pool_bytes_for(int cap) const;
    bool begin_row_pool();
    void end_row_pool();

    // one `spec` event per demand/pre-verify stage while the mode is on
    void tel_spec(int il, int pending_now, int ready_spec_now);
    void tel_d1_shape();
    void tel_d1_pair(int round, int draft_index, int32_t draft_token,
                     int layer, bool accepted, const int32_t * ids, int n_ids,
                     const int32_t * predicted, int n_predicted);
    // allocate the per-layer cache storage; called only when row_set_retain is
    // set, before any TG graph is built (the cache is not a graph tensor, but
    // a failed allocation must be loud at bring-up, never mid-generation).
    bool begin_row_cache();
    void end_row_cache();

    // T-GPLR-19 (F2-M1): predicted private VRAM for a retention depth, derived
    // from the members' real CUDA alloc sizes WITHOUT allocating any of it, so
    // a caller can pick a depth that fits before committing to one. Returns 0
    // when the policy cannot host the cache or the footprint cannot be
    // derived; a caller must treat 0 as "cannot size" and let begin_row_cache()
    // report the real reason. Because this runs the same per-member arithmetic
    // the allocation runs, the predicted and allocated totals cannot drift.
    size_t row_cache_bytes_for(int k) const;

    // Allocate the per-slot row-set buffers (full-size expert banks). Fails
    // loudly for non-merge-resident policies or on allocation failure.
    bool begin_row_set();

    // create the FFN companion graph of the layer being built (two-part
    // row-set TG sets); the builder fills the companion's selected_experts
    // INPUT through the returned set's ff_input.back()
    ggml_cgraph * new_ff_graph(int layer);

    // Deactivate row-set staging and free the row-set buffers.
    void end_row_set();

    // T-GPLR-62 S2: rebuild every slot's bundles at ne[2] = width, allocate the
    // compact device buffer, and only then release the full-bank buffer. Called
    // from begin_row_set() when row_set_window_width > 0. Returns false with a
    // structured error and WITHOUT having released the full-bank buffer if any
    // step fails, so a failed narrowing leaves the pre-S2 shape intact rather
    // than a half-narrowed slot.
    bool build_row_set_window(int width);

    // T-GPLR-62 S2 / gate A9: byte-compare every staged slot slab against the
    // checkpoint slab for the expert that position is specified to hold. Called
    // from load_row_set AFTER the copy-ready event is synchronized, so it reads
    // exactly the bytes the FFN is about to consume. Returns false on any
    // mismatch, having emitted a "recon_mismatch" refusal. Diagnostics only:
    // does nothing unless row_set_recon is set.
    bool recon_row_set_window(int il, const llama_ls_package & pkg, llama_layer & dst,
                              const std::vector<int32_t> & uniq, uint32_t gen);

    // Stage the union of expert rows in ids[il] into the layer's row-set
    // slot (copying only rows not resident for the slot's current
    // generation). Emits one rowset_fetch event; returns false with a
    // structured error on any inconsistency (never silently runs foreign
    // rows).
    bool load_row_set(int il, const std::vector<int32_t> & ids,
                      const std::vector<std::vector<int32_t>> * per_token = nullptr);

    // T-GPLR-62 S2: union position lookup table written by the LAST successful
    // load_row_set. Sized n_expert, ALL entries set to -1, then
    // row_set_union_pos[uniq[i]] = i for every union member. An expert absent
    // from the window reads -1, which the executor must treat as a loud
    // refusal and never as a position. Re-initialising the whole table every
    // window is the contract's hard requirement: a partial write leaves a
    // stale position from a previous window pointing at a different expert,
    // which computes silently and wrongly.
    const std::vector<int32_t> & row_set_union_pos() const { return row_set_union_pos_; }
    // True once load_row_set has populated the table at least once.
    bool                     row_set_union_valid() const { return row_set_union_valid_; }
    // Configured compact window width is the public field row_set_window_width.

    // Host-side exact top-k derivation from the router's live selection
    // probs (the same stable descending sort the CUDA router uses; shared
    // with tel_route). Reads the device tensor into a scratch buffer.
    bool read_router_ids(int il, ggml_cgraph * g, std::vector<std::vector<int32_t>> & ids_per_token, std::vector<float> & probs_out);

    // --- T-GPLR-13 telemetry (moe-row-set policy; rowset_fetch is the
    // single per-token route record in row-set TG; PP route events are
    // unchanged) ---
    void tel_rowset_init(int n_slots_, size_t slot_bytes, int window_tokens, int n_expert, int top_k, int retain_k);
    void tel_rowset_fetch(int il, int slot, uint32_t generation, int64_t chunk, int top_k,
                          const std::vector<int32_t> & union_ids, size_t n_rows_fetched,
                          size_t bytes_staged, size_t bytes_full_bank,
                          double host_submit_us, double transfer_us, double ready_latency_us,
                          size_t n_rows_resident, size_t n_rows_retained, size_t n_rows_retired,
                          const std::vector<int32_t> * pool_hit_ids = nullptr);
    void tel_rowset_refused(int il, int slot, uint32_t generation, const char * reason, const char * detail);
    void tel_rowset_round(int64_t round, size_t bytes_staged, size_t bytes_full_bank,
                          uint64_t n_fetched, uint64_t n_reused);

    // per-TG-round row-set byte account accumulators; the executor emits and
    // resets them once per TG ubatch via rowset_round_end()
    uint64_t rowset_round_bytes_staged = 0;
    uint64_t rowset_round_bytes_bank   = 0;
    uint64_t rowset_round_rows_fetched = 0;
    uint64_t rowset_round_rows_reused  = 0;
    // last completed fetch's timing split (set by load_row_set, consumed by
    // the row-set executor so the TG phase_summary carries the row-set terms;
    // single-threaded: the executor synchronizes with the copy before reading)
    double rowset_last_submit_us = 0.0;
    double rowset_last_transfer_us = 0.0;
    double rowset_last_ready_us = 0.0;
    // T-GPLR-19: cross-token cache-served rows and window-age retirements;
    // the checker validates coverage conservation against them (gate 10).
    uint64_t rowset_round_rows_retained = 0;
    uint64_t rowset_round_rows_retired  = 0;
    // T-GPLR-82 S1: round START stamp. rowset_round's own t_us is the END
    // stamp; with no start stamp the gap between consecutive stamps cannot
    // be attributed, which is why the 2,662.864 ms tg82 outlier could not be
    // placed (round stamps spanned 7.249 s of a 10.680 s wall). Telemetry
    // only: nothing in the execution path reads this member.
    int64_t  rowset_round_t0_us = 0;
    void rowset_round_end();

    // --- T-GPLR-08: one-layer-ahead PrefetchVirtualMemory probe (host feed) ---
    // Configure the rolling prefetch: build per-layer validated member ranges
    // (VirtualQuery MEM_MAPPED guard; unvalidated members are skipped with a
    // recorded count) and start the 1-deep worker thread. n == 0 keeps today's
    // behavior byte-identically (no worker, no events). Hint semantics only:
    // a failed or skipped prefetch degrades to the fault path and can never
    // produce wrong data; every skip/failure is recorded in telemetry.
    bool set_pp_prefetch_ahead(int n);

    // Stop and join the prefetch worker (idempotent; destructor calls it
    // before closing the telemetry sink so no event is written post-close).
    void stop_prefetch_worker();

    // --- T-GPLR-25: PP async-feed prototype (device-side copy overlap) ---
    // Configure a worker thread that submits the NEXT executed layer's H2D
    // package copy into a free slot WHILE the compute thread runs the current
    // layer. This moves the ~47-180 ms/layer pageable host feed (the driver's
    // internal staging inside cudaMemcpyAsync) off the main thread so it
    // overlaps graph compute instead of serializing after it — the measured
    // PP structure (0% submit-before-compute overlap at 1 slot; feed 185 s vs
    // compute 22 s at 50k) is exactly this serialization. n == 0 keeps
    // today's behavior byte-identically (no worker, no events). Requires
    // n_slots >= 2: with one slot the reuse invariant forbids overwriting
    // the weights being computed from, and the configure refuses loudly.
    bool set_pp_async_feed(int n);

    // T-GPLR-32: serialize the worker's H2D DMAs to ONE in flight (default
    // off). The gate is a stream dependency (ev_af_gate), so the CPU memcpy
    // hop stays overlapped; only the DMA submission is ordered. Must be set
    // before the async-feed worker starts. Off by default; the T-GPLR-29/32
    // traces show concurrent worker DMAs stretch each other ~2.2x, and the
    // A/B measures whether serializing recovers it.
    bool set_pp_af_serialize(bool on);

    // T-GPLR-33 V1: member-batched staging — when the per-slot contiguity
    // check passed for a layer, stage its bank with ONE H2D (a single span
    // set over the tiled member region) instead of one submission per
    // member. Off by default; the A/B measures the predicted window shrink
    // (91 -> ~44-58 ms) against the falsifier. Must be set before the run
    // starts (staging reads it per submission).
    bool set_pp_submit_batch(bool on);
    bool pp_submit_batch = false;
    // T-GPLR-35: staging DMAs on an auxiliary stream inside the COMPUTE
    // backend's CUDA context, so the staging/compute ordering waits are
    // intra-context. This records the intent only; the compute backend is
    // resolved later (scheduler), where the aux stream is validated and a run
    // without it is refused loudly rather than falling back silently.
    bool set_pp_copy_overlap(bool on);
    bool pp_copy_overlap = false;

    // T-GPLR-35 recon (diagnostics only, off by default): records stream/context
    // identity once per run plus, per staged bank, the copy's own GPU window and
    // its start offset against the compute stream's backlog at enqueue time.
    // Decides which coupling actually chains transfer to compute before any
    // ordering change is made. Ordering- and arithmetic-neutral.
    bool set_pp_recon(bool on);
    bool pp_recon = false;

    // T-GPLR-53 A4 (modality discriminator): pre-timed pre-warm modes, a bitfield.
    // 0 = off (DEFAULT: no touch, no allocation, no submission -- every existing
    // shape stays byte-identical), 1 = CPU read-touch of the package source
    // spans (H1's page-side candidate), 2 = one dummy GPU launch + event sync
    // (H3's submission-side candidate), 4 = duration-matched no-touch delay (the
    // FALSIFIER for 1: equal wall time, zero package pages touched), 8 =
    // duration-matched wrong-region page touch (equal activity and duration as 1,
    // on a private 64 MiB scratch buffer). The runner calls pp_prewarm_run()
    // BEFORE it opens the timed PP region: A4 measures what moves OUT of the
    // pass-1 wall, not what the pre-warm itself costs.
    //
    // delay_ms governs the mode-4 no-touch delay ONLY (runner flag
    // --pp-prewarm-delay-ms, default 1700 = the compiled control duration, so an
    // un-flagged run is unchanged). Mode 8 keeps the compiled duration on purpose:
    // it was matched to the WALK, and the dose sweep varies only the delay.
    bool set_pp_prewarm(int mode);
    bool pp_prewarm_run(int delay_ms = 1700);
    int  pp_prewarm = 0;
    // T-GPLR-54 A2: per-step top-1/top-2 logit readout for the first N greedy
    // TG steps. Default 0 (off): a default run's trace stays byte-identical.
    int tg_logit_margin = 0;

    // T-GPLR-47: swap-in-slot rebuild. Set by the runner around the pass-2+
    // per-pass prebuild call (and cleared after): a prebuild that re-enters an
    // existing BUILT set then REPLACES it with a fresh-context rebuild -- new
    // ggml ctx and llm_graph_result, act tensors replayed onto the old set's
    // exact slab/buffer regions -- move-assigned into the SAME vector slot, so
    // no tensors accumulate across passes and every graph_set pointer stays
    // valid by construction (the T-GPLR-44 address-stability accident becomes
    // structural). A first-build prebuild (no sets yet) is unaffected; explicit
    // flag, not build_generation counting (INV1 of the task contract).
    void set_rebuild_replacement(bool on) { rebuild_replacement = on; }
    bool rebuild_replacement = false;

    // T-GPLR-41: pass-boundary memory-lifecycle recon (diagnostics only; off
    // by default). Adds per-chunk mctx + per-layer KV-view telemetry events.
    bool set_pp_mctx_recon(bool on);
    bool pp_mctx_recon = false;
    // T-GPLR-39 (owner decision 2026-09-20): the route DIAGNOSTIC defaults OFF
    // at the runner. tel_route reads the router probs back from the device and
    // hashes/sorts them on the host 2x n_layer times per pass; at u2048 that
    // readback+sort was 1161.5 of 1152.8 ms of the `other` bucket (100.8%,
    // closure exact) - the diagnostic measuring itself. This gates ONLY the
    // observational event: the FUNCTIONAL read_router_ids row-set routing path
    // is untouched, and routing/arithmetic are byte-identical either way.
    // Engine-level default stays TRUE (the pre-T-GPLR-39 behavior) so a direct
    // API consumer keeps today's semantics; the runner resolves its own
    // --route-telemetry default to 0 and passes it down.
    bool set_route_telemetry(bool on);
    bool route_telemetry = true;
    // T-GPLR-41: 1-based layer_major_pp invocation counter within the current
    // stats epoch (zeroed by reset_stats, which the runner calls per pass).
    int pp_mctx_pass_ctr = 0;

    // T-GPLR-36 follow-up: host-memory instrument state. Deliberately NOT reset
    // by reset_stats(): the runner calls reset_stats() at every pass boundary
    // (including before pass 2), and growth ACROSS passes is the whole point of
    // this instrument -- the same reason pp_mctx_pass_ctr is left alone there.
    // mem_pass_ctr is monotonic process-wide; the baseline is the first sample
    // taken, so `g_*` fields are growth since the run's first PP pass began.
    int      mem_pass_ctr     = 0;      // 1-based pass number (never reset)
    bool     mem_first_seen   = false;
    uint64_t mem_first_ws     = 0;
    uint64_t mem_first_commit = 0;
    uint64_t mem_first_eng    = 0;
    uint64_t mem_prev_ws      = 0;
    uint64_t mem_prev_commit  = 0;
    uint64_t mem_prev_eng     = 0;
    uint64_t mem_peak_ws      = 0;      // max of OUR samples (the OS keeps its own lifetime peak)
    void tel_host_mem_emit(const char * label);   // shared shape; self-gated on --pp-recon

    // T-GPLR-44: per-graph slices of the patch caches, from one transitive src
    // walk. RECON ONLY after the scoping A/B failed identity (see graph_set):
    // consumed by the dump, never by the launches.
    // T-GPLR-62 S2: returns false when the INPUT classification would leave a
    // row-set companion input in a state the snapshot contract cannot describe
    // (see the guard in the body). The caller MUST honour the return: a guard
    // that cannot stop the run is decoration, not a check.
    bool build_patch_split(graph_set & gs);

    // T-GPLR-44: patch-cache composition dump. Emitted once per graph
    // set at patch-cache build time, host-only and self-gated on --pp-recon.
    // Reports the split built by build_patch_split, so the dump and the patch
    // cannot disagree. The definition carries the field contract.
    void tel_patch_cache(const graph_set & gs, size_t set_idx);

    // Stop and join the async-feed worker (idempotent; destructor calls it
    // before closing the telemetry sink so no event is written post-close).
    void stop_pp_async_feed();

    // T-GPLR-30: persistent pinned pool — N bank-sized pinned host buffers
    // populated once at configure time; both stage sites (af worker and
    // load_layer) then serve a POOLED bank's members directly from the pool,
    // skipping the mmap->pinned memcpy hop. The DMA into the slot is
    // unchanged, so VRAM behavior, quantization bytes, and arithmetic are
    // byte-exact (weights are immutable; a mirror cannot go stale).
    // policy "fixed": banks 0..N-1 are pooled permanently (savings-curve dial).
    // policy "lru": N-entry LRU refilled on miss (structural control: under
    // layer-major, LRU < working set provably hits zero — kept for the cliff
    // measurement only).
    // Requires pinned staging (--copy-source pinned): the pool IS a pinned
    // buffer, and with direct staging there is no hop to remove. Must be
    // called after init (needs per-layer byte sizes) and BEFORE any stage
    // work; returns false with error_msg set on any failure (allocation,
    // configuration, or sizing mismatch).
    bool set_pp_pool(int n_banks, const char * policy);

    // Aggregate pool counters (zeros when the pool is off).
    void pool_stats(uint64_t & n_hit, uint64_t & n_miss, uint64_t & bytes_saved);

    // Aggregate async-feed counters (guarded; zeros when off).
    void af_stats(uint64_t & n_submit, uint64_t & n_ok, uint64_t & n_err,
                  uint64_t & bytes, uint64_t & us);

    // Aggregate prefetch counters (guarded; zeros when the probe is off).
    // Skipped members are reported once in the prefetch_init event.
    void pf_stats(uint64_t & n_submit, uint64_t & n_complete, uint64_t & n_err,
                  uint64_t & bytes, uint64_t & us);

private:
    // Defined in llama-layer-streaming.cpp; owns the thread, queue, and stats.
    // Out-of-line deleter so the header stays free of the worker definition.
    struct pf_worker;
    struct pf_worker_deleter { void operator()(pf_worker * w) const; };
    std::unique_ptr<pf_worker, pf_worker_deleter> pf;
    int  pp_prefetch_ahead = 0;

    // T-GPLR-25: async-feed worker (device copy overlap), same shape as the
    // T-GPLR-08 prefetch worker. Owns the thread, queue, and counters.
    struct af_worker;
    struct af_worker_deleter { void operator()(af_worker * w) const; };
    std::unique_ptr<af_worker, af_worker_deleter> af;
    int  pp_async_feed = 0;
    // Layers whose copy is in flight on the worker (slot -> logical layer);
    // guarded by af->mtx. The main thread drains these in wait_slot_ready.
    // depth bound: how many layers may sit queued/in-flight (K config).
    int af_depth = 0;
    // Per-layer validated member ranges (read-only after set_pp_prefetch_ahead
    // returns; the worker thread may read them concurrently).
    std::vector<std::vector<void *>> pf_ranges_ptr;
    std::vector<std::vector<size_t>> pf_ranges_len;
    void pf_enqueue_from(int il);
    void tel_prefetch_init(int ahead, int layers, size_t ranges, size_t skipped, bool api_ok);
    void tel_prefetch_submit(int il, int phase, int ranges, size_t bytes);
    void tel_prefetch_complete(int il, int phase, int ranges, size_t bytes, double dur_us, uint32_t err);

    // T-GPLR-25: submit layer il's copy WITHOUT waiting (the load_layer body
    // minus the caller-side wait) on the worker thread; records the same
    // copy_submit/copy_complete events. Used only by the async-feed worker.
    bool af_submit_copy(int il);
    // T-GPLR-33 V1: shared staging loop for both copy sites — member hops
    // into the pinned/pool source, then ONE span H2D when the slot-build
    // contiguity check produced a bank_span and the sources are contiguous,
    // else the legacy per-member submissions (byte-identical either way).
    // T-GPLR-35 staging-stream plumbing: every staging submission and every
    // staging-side ordering call goes through these, so the mode switch is
    // atomic — an event must never be recorded on a stream that carries no
    // copy work, and a wait must always target the stream that does.
    bool ls_stage_aux() const;
    void ls_stage_record(void * ev);
    void ls_stage_wait(void * ev);
    void ls_stage_drain();

    // T-GPLR-35 recon (private): ring of timing samples, flushed a few banks
    // later once both events have completed (reading an unfinished timing event
    // returns an error, so samples are emitted late rather than provisionally).
    struct ls_recon_ev {
        void *   copy_start  = nullptr;
        void *   copy_end    = nullptr;
        void *   compute_ref = nullptr;
        ggml_backend_t copy_be = nullptr;
        int      copy_stream_no = -1;
        int      il   = -1;
        uint32_t gen  = 0;
        // T-GPLR-53 addendum: the ubatch the bank was staged under, captured at
        // ENQUEUE time. The ring flushes lazily at the next recon_begin_bank, so
        // emitting the CURRENT n_ubatches there mislabeled a pass's final sampled
        // banks with the NEXT pass's ubatch (measured: the -14/-15 label class on
        // gplr50/gplr52; generation stamps were always correct).
        unsigned long long ubatch = 0;
        bool     emitted = true;
    };
    std::vector<ls_recon_ev> recon_ring;
    int  recon_next = 0;
    bool recon_armed = false;
    bool recon_identity_done = false;
    void recon_begin_bank(int il, uint32_t gen);
    void recon_end_bank();
    void recon_flush_all();
    // T-GPLR-37: page-lock the span covering every member source of this
    // package so staging can DMA straight from the model's mapping. Cached on
    // the slot (re-staging the same package reuses it; a different package
    // unregisters the old span first). Loud refusal on a driver error: the
    // run aborts with the cudaGetErrorString text, never a silent hop.
    bool ls_register_pkg_source(int il, llama_layer_streaming_slot & slot, const llama_ls_package & pkg);
    void ls_unregister_slot_source(llama_layer_streaming_slot & slot);

    // `gen` is the generation the copies issued below will be tagged with
    // (slot.generation is still the pre-increment value when this runs).
    bool stage_members(ggml_backend_t backend, const llama_ls_package & pkg,
                       llama_layer & bundle, int il, uint32_t gen, llama_layer_streaming_slot & slot,
                       char * pool_base, size_t pool_bytes, bool pool_valid,
                       char * staging, size_t * staged_out,
                       int * submit_batch_out = nullptr);
    // pick the next executed logical layer after il (empty exec_layers = il+1)
    int af_next_layer(int il) const;
    // T-GPLR-32 state (private): gate enabled + in-flight worker DMA counter.
    bool pp_af_serialize = false;
    // count of worker DMAs submitted but not yet consumed (copy_complete
    // observed by wait_slot_ready). Atomic: the worker increments at submit,
    // the main thread decrements at consume; af->mtx is NOT held for these.
    std::atomic<int> af_dma_inflight{0};
    void tel_async_feed_init(int depth, int n_slots);
    void tel_async_feed_submit(int il, int slot, uint32_t generation);
    void tel_async_feed_ready(int il, int slot, uint32_t generation, double overlap_us);

    // T-GPLR-19 (F2-M1): per-layer cache cost for `cap` rows/layer, measured
    // from the members' real CUDA alloc sizes. Tensors are created in `meta`
    // (metadata only -- this call never allocates device memory). When
    // `members` is non-null the created cache tensors are appended in member
    // order with nullptr holes preserved and named after their source; when it
    // is null they are discarded with the context. Returns false only when a
    // cache tensor could not be created.
    bool row_cache_layer_cost(int il, int cap, ggml_context * meta,
                              std::vector<ggml_tensor *> * members, size_t * out_cost) const;

public:
    ~llama_layer_streaming();

    friend struct llama_context;
};

#ifdef LLAMA_LAYER_STREAMING

// Pure host-side chunk layout (T-Q38CTX-01): split pp_tokens into full
// ubatch-sized chunks plus one optional 1..ubatch-1 token tail. No CUDA, no
// model: testable on the CPU. Returns an empty vector on invalid input.
LLAMA_API std::vector<llama_ls_chunk> llama_ls_chunk_layout(int64_t pp_tokens, int ubatch);

// The layer-major activation-slab arithmetic, as ONE pure owner (T-GPLR-49).
// `offset` is chunk `chunk_id`'s byte offset inside a slab half, derived from
// the immutable chunk descriptor -- never from a cursor, so a reordered or
// repeated pass cannot move a chunk's slice. `bytes` is that chunk's slice and
// `half` is one half's total. Every size uses ggml_type_size(act_dtype): the
// pre-T-GPLR-49 formula used sizeof(float), which is 2x too large for bf16 (the
// default since T-GPLR-12). Returns false with *out untouched when there is no
// such region -- the caller must refuse loudly, never clamp.
struct llama_ls_slab_region {
    size_t offset = 0;
    size_t bytes  = 0;
    size_t half   = 0;
};

LLAMA_API bool llama_ls_slab_region_of(const llama_ls_chunk * chunks, size_t n_chunks, size_t chunk_id,
                                       int64_t n_embd_io, ggml_type act_dtype, llama_ls_slab_region * out);

// Half-open interval overlap, so the non-aliasing claim rule is testable on the
// host without an instance (T-GPLR-49). Touching intervals ([0,1024) and
// [1024,2048)) do NOT overlap; any shared byte does. A degenerate (empty)
// interval overlaps nothing.
LLAMA_API bool llama_ls_slab_overlaps(size_t a_start, size_t a_end, size_t b_start, size_t b_end);

// Exact top-k expert IDs of ONE router probability row (T-GPLR-39), in the
// order a stable descending sort produces: probability descending, ties by
// ascending expert index (the CUDA router's stable radix-sort tie-break). It
// is the fully sorted row truncated to k, computed as an O(n_expert * k)
// bounded insertion so the per-token route telemetry does not sort all
// n_expert experts per token. probs is a row of n_expert F32 selection
// probabilities (softmax output: finite and non-negative -- NaN is outside
// the equivalence domain, and the sorted form's behavior there was
// unspecified). Writes min(k, n_expert) ids into ids_out. Pure host
// arithmetic, no CUDA: unit-tested against the stable-sort form in
// tests/test-layer-streaming-route-topk.cpp.
LLAMA_API void llama_ls_route_topk_row(const float * probs, int64_t n_expert, int64_t k, int32_t * ids_out);

// ---------------------------------------------------------------------------
// T-GPLR-58 A0 (2026-09-23): speculative TG expert-row PLACEMENT, as one pure,
// host-testable owner.
//
// A speculative fill is a GUESSED row for a demand that has not happened yet
// (the proposer is A1's deliverable; in A0 the guesses come from a scripted
// sequence, so the mechanism is exercised without a predictor). Three rules
// decide every outcome, and all three are arithmetic rather than effects -- so
// they live here, are called by the engine, and are driven directly by
// tests/test-layer-streaming-tg-spec.cpp with no CUDA and no model. The arrays
// are the caller's own storage, indexed by cache position:
//
//   experts[p]  the expert id at position p (-1 = none). Owned by the engine's
//               cache (cache_expert); this API never writes it.
//   states[p]   llama_ls_spec_state: FREE / READY / PENDING / CANCELLED.
//   recency[p]  the LRU stamp (larger = more recently routed).
//   spec[p]     1 = placed by speculation and no demand has consumed it yet.
//
//   P2  a position is a HIT only in READY state (its own copy completed); a
//       PENDING position is reserved but never served, and its bytes are never
//       overwritten while the copy is in flight.
//   P3  speculation is a HARD BARRIER below demand capacity: an admission is
//       refused unless at least `reserve` positions stay FREE for the demand
//       side, speculative fills are capped at `max_pending` per layer, and a
//       ready speculative row is the FIRST victim -- never a residency claim
//       that crowds out demand.
//   P4  cancellation is a PLAN, not an assumption: the caller names how many
//       positions it must reclaim (`need_free`) and gets back the OLDEST
//       pending fills to cancel. A cancelled fill frees its position only after
//       its own copy completed (the caller reaps it), so a write in flight is
//       never reused, and the mark is non-blocking -- which is what lets a
//       demand stage cancel without waiting (P1).
// ---------------------------------------------------------------------------
struct llama_ls_spec_admission {
    bool         ok     = false;
    int          pos    = -1;
    const char * reason = "";    // why it was refused (structured telemetry)
};

// Admission decision for one speculative fill (P3/P5). Refused when the expert
// is already present (READY or PENDING -- no duplicate admission), when no FREE
// position exists, when `n_pending` is already at `max_pending`, or when taking
// a position would leave fewer than `reserve` FREE positions for demand. It
// never evicts and never displaces a READY row, so a speculative source cannot
// reduce the incumbent's coverage (P5).
LLAMA_API llama_ls_spec_admission llama_ls_spec_admit(const int32_t * experts, const uint8_t * states,
                                                      int cap, int reserve, int max_pending,
                                                      int n_pending, int32_t expert);

// The first FREE position, or -1. The demand path uses it too, so "where does a
// fetched row go" has exactly one owner.
LLAMA_API int llama_ls_spec_free_pos(const int32_t * experts, const uint8_t * states, int cap);

// P2's hit test, in one place: ONLY READY is a hit. FREE, PENDING and CANCELLED
// must all be fetched by the demand itself.
LLAMA_API bool llama_ls_spec_is_hit(uint8_t state);

// P2/P3 eviction rules: a PENDING or CANCELLED position is NEVER a victim (its
// bytes are in flight); a READY speculative row is preferred over any demanded
// row; among comparable candidates the smaller recency loses.
LLAMA_API bool llama_ls_spec_may_evict(uint8_t state);
LLAMA_API bool llama_ls_spec_victim_better(int64_t a_recency, uint8_t a_spec, bool a_has,
                                           int64_t b_recency, uint8_t b_spec, bool b_has);

// Cancellation plan (P4): which PENDING positions must be cancelled to reclaim
// `need_free` positions? OLDEST (smallest recency) first. Writes at most
// `max_out` positions and returns how many. A generation boundary cancels ALL
// pending fills, which the caller asks for by passing the layer's whole pending
// count as `need_free`.
LLAMA_API int llama_ls_spec_cancel_plan(const int64_t * recencies, const uint8_t * states,
                                        int cap, int need_free, int * out_positions, int max_out);

// T-GPLR-58 A2 (M7): the LEAD of one consumed fill, in demand stages of its own
// layer -- the head start a fill actually got before a demand was served from it.
// The issuing stage counts as 0, so a fill issued at the end of stage t and consumed
// by stage t+2 has lead 2. Returns 0 when there is no fill to credit (`issue_ord`
// < 0) or when the fill was consumed in the stage that issued it: 0 therefore always
// means "not a consumed speculative row", never "a consumed fill with no head start".
LLAMA_API int llama_ls_spec_lead(int32_t consume_ord, int32_t issue_ord);

// M7's histogram bucket for a lead: 1..7 -> 0..6, >= 8 -> 7 (the overflow bucket),
// and -1 for a lead that is not a consumption (see llama_ls_spec_lead). The engine
// keeps exactly 8 counters, so the emitted distribution always has this shape.
LLAMA_API int llama_ls_spec_lead_bucket(int lead);

// ---------------------------------------------------------------------------
// T-GPLR-58 A1 - the PROPOSER and its PLANNER: POLICY, not mechanism.
//
// A0 owns the staging mechanism (whether a fill is admitted, what a completion
// means, what a cancellation does). A1 owns WHICH rows are guessed, IN WHAT
// ORDER, and WHEN a cancel is needed ahead of a demand stage, and this section
// is that policy as pure functions over caller-owned state: no globals, no
// clock, no I/O, no engine members. The engine and
// tests/test-layer-streaming-tg-predict.cpp call exactly these functions.
//
// Why the learning state is a PARAMETER and not a member: the policy has to be
// provable host-side, and the offline screen it is ported from
// (tools/predictor-screen-gplr.py, pinned by file hash in the A1 result) is a
// library of stateful candidates driven predict()-then-observe() over one
// demand row at a time. Making the state explicit is what lets a test hold two
// copies of the same stream and prove that the families are deterministic and
// that the null control is demand-blind.
//
// Faithfulness rules, inherited from the screen verbatim:
//   * predict() BEFORE observe() for every (token, layer) row;
//   * a row may only see rows of tokens < t (all layers) and of token t at
//     layers < L -- which is what the engine's demand order gives it;
//   * ranking ties break on the SMALLER expert id (the screen's (-score, id));
//   * integer family counts are compared as DOUBLE, so their sums are exact,
//     and the two float-scored families add one term per source expert in the
//     observed order -- the port therefore reproduces the screen's ids exactly,
//     not approximately (the comparator proves it per row).
//
// The count-matched structure-free NULL is here too, and it is deliberately
// NOT a family: it reads only the ARM's published COUNT for the row (a scalar)
// and publishes that many ids from a fixed-seed xorshift over [0, n_expert),
// duplicates rejected. It consults no ids, no demand and no router output,
// which is why its sequence is invariant under a reordered demand stream - and
// why it cannot be confused with a predictor that merely proposes fewer rows.
// ---------------------------------------------------------------------------

enum llama_ls_pred_family : uint8_t {
    LLAMA_LS_PRED_HISTORY  = 0, // the K=1 retention baseline: previous token's set, same layer
    LLAMA_LS_PRED_MARKOV1  = 1, // per-layer id-bigram transitions
    LLAMA_LS_PRED_COACT    = 2, // normalized per-layer co-activation
    LLAMA_LS_PRED_COUPLING = 3, // cross-layer, lift-weighted (same token, previous layer)
};

#define LLAMA_LS_PRED_BIT_HISTORY  (1u << LLAMA_LS_PRED_HISTORY)
#define LLAMA_LS_PRED_BIT_MARKOV1  (1u << LLAMA_LS_PRED_MARKOV1)
#define LLAMA_LS_PRED_BIT_COACT    (1u << LLAMA_LS_PRED_COACT)
#define LLAMA_LS_PRED_BIT_COUPLING (1u << LLAMA_LS_PRED_COUPLING)
#define LLAMA_LS_PRED_BIT_ALL      (LLAMA_LS_PRED_BIT_HISTORY | LLAMA_LS_PRED_BIT_MARKOV1 | \
                                    LLAMA_LS_PRED_BIT_COACT   | LLAMA_LS_PRED_BIT_COUPLING)

// The three arms T-GPLR-58 pre-registered, as the port expresses them. `bar09`
// wraps MARKOV-1, not coact: the block quotes the markov-1@bar0.9 row
// (25.1/29.0 honest, +0.5 recall and +4.3 ppb over history) as the safety
// point, while coact@bar0.9 is the stronger but equally barred row
// (25.8/29.8). Switching the arm to coact is a Planner call, not a port
// detail, so the port implements the row the contract quotes.
enum llama_ls_pred_arm : uint8_t {
    LLAMA_LS_ARM_COACT  = 0, // unbarred co-activation, top-B
    LLAMA_LS_ARM_BAR09  = 1, // markov-1 ranked, published while 0.9^rank >= 0.5
    LLAMA_LS_ARM_FUSION = 2, // vote fusion over {history, markov-1, coact, coupling-copula}
    LLAMA_LS_ARM_D1     = 3, // T-GPLR-59: MTP proposed-token table (not a past-only arm)
};

#define LLAMA_LS_PRED_MAX_OUT 64   // arms are never allowed to publish more than this

// The proposer's learning state. Every table is allocated only for the families
// the caller asks for, so an arm pays for what it uses (markov-1, coact and
// coupling are each n_layer * n_expert^2 ints at the real 40 x 256 shape).
struct llama_ls_pred_state {
    int      n_layer  = 0;
    int      n_expert = 0;
    uint32_t families = 0;                    // bitmask of LLAMA_LS_PRED_BIT_*
    std::vector<std::vector<int32_t>> prev;   // per layer: last observed ids, in demand order
    std::vector<int32_t> trans;               // markov-1: [layer][e][nxt]
    std::vector<int32_t> pair;                // coact:    [layer][e][nxt]
    std::vector<int32_t> unit;                // coact:    [layer][e]
    std::vector<int32_t> joint;               // coupling: [layer-1][e][nxt]
    std::vector<int32_t> marg;                // coupling: [layer][nxt]
};

// Allocate and zero the state for the requested families. Call once per stream.
LLAMA_API void llama_ls_pred_init(llama_ls_pred_state * st, int n_layer, int n_expert, uint32_t families);

// One family's proposal for one demand row, BEFORE that row is observed: at most
// `b` ids in rank order, written to `out`, returning how many. Pure over (st,
// family, layer, b) -- calling it twice on the same state returns the same ids.
LLAMA_API int llama_ls_pred_predict(const llama_ls_pred_state * st, int family, int layer, int b,
                                    int32_t * out, int max_out);

// The ARM-level proposal: the bar and the merge operator that the pre-registered
// arms declare, over the same pure state. VOTE fusion (param-free, equal weights,
// in-source dedup, rank-decay tiebreak) is what `arm-fusion` means here.
LLAMA_API int llama_ls_pred_arm_predict(const llama_ls_pred_state * st, int arm, int layer, int b,
                                        int32_t * out, int max_out);

// Fold one observed demand row into the state. Call AFTER predict() for it.
LLAMA_API void llama_ls_pred_observe(llama_ls_pred_state * st, int layer, const int32_t * ids, int n_ids);

// T-GPLR-59 D1: an online table keyed by the token proposed by the MTP drafter.
// The table is deliberately separate from the past-token families: D1 is allowed
// to use one proposed future token, while the existing state remains past-only.
struct llama_ls_d1_entry {
    int32_t expert = -1;
    int32_t count  = 0;
};

struct llama_ls_d1_state {
    int n_layer = 0;
    int n_expert = 0;
    uint64_t rows_seen = 0;
    uint64_t accepted_rows = 0;
    std::unordered_map<uint64_t, std::vector<llama_ls_d1_entry>> table;
};

LLAMA_API void llama_ls_d1_init(llama_ls_d1_state * st, int n_layer, int n_expert);
LLAMA_API int llama_ls_d1_predict(const llama_ls_d1_state * st, int draft_token, int layer, int b,
                                  int32_t * out, int max_out);
LLAMA_API void llama_ls_d1_observe(llama_ls_d1_state * st, int draft_token, int layer, bool accepted,
                                   const int32_t * ids, int n_ids);
LLAMA_API bool llama_ls_d1_pairing_valid(const llama_ls_d1_state * st);

// The structure-free null's fixed seed and its fill primitive: publish `count`
// distinct ids from a xorshift32 over [0, n_expert). Returns how many were
// written (fewer only when max_out or n_expert cannot supply `count`). It takes
// no ids and no demand, so its output depends on the RNG and `count` alone.
LLAMA_API uint32_t llama_ls_pred_null_seed(void);
LLAMA_API int llama_ls_pred_null_fill(uint32_t * rng, int n_expert, int count,
                                      int32_t * out, int max_out);

// Why the planner stopped issuing fills for one demand stage.
enum llama_ls_plan_stop : uint8_t {
    LLAMA_LS_STOP_PROPOSALS = 0, // the proposal list was exhausted
    LLAMA_LS_STOP_BUDGET    = 1, // the per-row budget is spent
    LLAMA_LS_STOP_PENDING   = 2, // max_pending fills already outstanding for this layer
    LLAMA_LS_STOP_RESERVE   = 3, // admitting would leave fewer than `reserve` positions FREE
    LLAMA_LS_STOP_NO_FREE   = 4, // no FREE position at all
};

// The planner's plan for ONE demand stage: the fills to issue, in issue order,
// plus what it declined and why. It is a pure counting plan equivalent to asking
// llama_ls_spec_admit for each proposal in rank order (each admission consumes
// exactly one FREE position and one pending slot), which is what makes the
// engine's use of it and the host test's assertion of it the same thing.
struct llama_ls_plan {
    int32_t issue[LLAMA_LS_PRED_MAX_OUT];  // experts to stage, issue order
    int     n_issue          = 0;
    int     stop             = LLAMA_LS_STOP_PROPOSALS;
    int     skipped_present  = 0;  // already READY or PENDING: no duplicate fill
    int     skipped_dup      = 0;  // repeated inside the proposal list
    int     refused_capacity = 0;  // admissible in principle, refused by the barrier
};

LLAMA_API llama_ls_plan llama_ls_spec_plan(const int32_t * experts, const uint8_t * states, int cap,
                                           int reserve, int max_pending, int n_pending, int budget,
                                           const int32_t * proposals, int n_proposals);

// The cancellation POINT, as policy: a demand that wants `n_want` rows must be
// able to place them without waiting. The reserve is what normally makes that
// true, so a cancel is planned only when the demand cannot be placed from the
// FREE positions at all; `used_reserve` reports the legal-but-notable case where
// the demand is spending the positions the barrier had held for it.
struct llama_ls_cancel_policy {
    bool         cancel       = false;
    int          need         = 0;      // pending fills to cancel, oldest-first
    bool         used_reserve = false;
    const char * reason       = "";
};

LLAMA_API llama_ls_cancel_policy llama_ls_spec_cancel_policy(int n_free, int reserve, int n_want,
                                                            int n_pending);

// ---------------------------------------------------------------------------
// T-GPLR-58 MODE 2 - the ENGINE-CONSUMED proposer: ONE pure call per demand stage.
//
// Modes 1 and 2 differ ONLY in where the proposal comes from. Mode 1 reads a
// script (A0's test vehicle: a proposal source with no opinion, so the mechanism
// can be measured on its own); mode 2 asks the arm itself, with the same planner,
// the same admission rules and the same issue path. Folding that sequence into one
// function is what makes the WIRING host-provable instead of a code claim: the
// engine calls this and executes the plan it returns, and
// tests/test-layer-streaming-tg-spec.cpp drives THIS function over a scripted
// demand stream -- so the engine cannot drift from the test.
//
// The order is A1's faithfulness rule (predict() BEFORE observe() for every
// (token, layer) row), expressed once:
//
//   1. OBSERVE the row the demand just placed, deduplicated in the caller's order
//      (the screen observes the row as a SET). After observing row k the state
//      holds rows <= k, which is exactly the state the offline screen predicts row
//      k+1 FROM -- so this proposal is the screen's own one-step prediction, and
//      the offline recall figures apply to it as proposal precision.
//   2. PROPOSE the arm's ranked list, or -- with `null_source` -- the count-matched
//      null: the ARM's count for that row, published from a fixed-seed xorshift
//      over [0, n_expert) that consults no ids, no demand and no router output.
//   3. PLAN those proposals against the live placement state, so P3's reserve, the
//      per-row budget and the pending cap decide how many are admitted. Not one of
//      them may displace a resident row (P5).
//
// THE HORIZON IS NOT DECIDED HERE, and that is deliberate rather than unfinished:
// the fill this plans is issued at the end of demand stage k of its layer, and
// A0/A2's MEASURED reap ordering (the reap runs at the END of a stage, after that
// stage's demand was served) makes the fill first consumable at stage k+2, while
// the proposal targets the row of stage k+1. So an id that the stage k+1 demand
// wants is the MATCHED fallback -- cancelled, its copy paid for twice -- and only
// ids that survive that demand can be consumed at k+2. M7b's
// matched/(consumed+matched) band is exactly the score of that asymmetry; moving
// the horizon or the reap point is a MECHANISM change (A0/A2's settled territory),
// not a wiring change.
//
// A mode-2 run is therefore a bet on id PERSISTENCE across one token, not on the
// proposal being right at the right time, and its telemetry is what tells the two
// apart: `consumed` counts the fills a demand was served from, `matched_by_demand`
// counts the ones that arrived late.
struct llama_ls_propose_result {
    llama_ls_plan plan;             // the fills to issue, in issue order, plus the stop reason
    int           n_proposals = 0;  // ids the SOURCE published (for the null: its own count)
    int           n_arm_count = 0;  // the ARM's published count (== n_proposals unless the null is on)
    // The published ids themselves, kept so a caller (or a test) can compare a null
    // leg against its arm directly instead of inferring a difference from what the
    // cache happened to admit.
    int32_t       published[LLAMA_LS_PRED_MAX_OUT] = {};
};

LLAMA_API llama_ls_propose_result llama_ls_spec_propose(llama_ls_pred_state * st, int arm, bool null_source,
                                                        int layer, int b,
                                                        const int32_t * demand_ids, int n_demand,
                                                        const int32_t * experts, const uint8_t * states, int cap,
                                                        int reserve, int max_pending, int n_pending,
                                                        uint32_t * null_rng, int n_expert);

// T-GPLR-59 D1: publish from the proposed-token table, then use the same planner
// and count-matched-null path as the past-only proposer. The caller supplies the
// current draft token and observes accepted target routes separately.
LLAMA_API llama_ls_propose_result llama_ls_spec_propose_d1(const llama_ls_d1_state * d1,
                                                         int draft_token, int layer, int b,
                                                         bool null_source,
                                                         const int32_t * experts, const uint8_t * states, int cap,
                                                         int reserve, int max_pending, int n_pending,
                                                         uint32_t * null_rng, int n_expert);

// T-GPLR-36 follow-up (2026-09-18): HOST-memory instrument. The 200k ledger run
// attributed the time residual but left one measured quantity unattributed: the
// process's host resident set, observed climbing ~19 GB -> ~30 GB across a run,
// while every byte figure in the trace is VRAM-side. These are the process and
// system counters that make that growth measurable, sampled by tel_host_mem().
//
// The split that answers "whose memory is it": ws_bytes is the whole working
// set; ws_private_bytes is this process's own ANONYMOUS resident pages, so
// ws_bytes - ws_private_bytes is the file-backed / shared share (the touched
// model mapping, mapped GGUF regions, DLLs); commit_bytes is private commit,
// allocated-or-reserved private bytes regardless of residency. Growth in the
// private share is the runner's own allocations; growth in the file-backed
// share is the model mapping being paged in or pinned.
//
// Windows-only in fact, not by design: the residency path is CUDA/Windows-only,
// and a platform without these counters reports ok == false rather than a
// fabricated zero (the test skips rather than pretending to measure).
struct llama_ls_host_mem {
    bool     ok                        = false;  // process counters were read
    uint64_t ws_bytes                  = 0;      // working set (resident pages)
    uint64_t ws_peak_bytes             = 0;      // OS lifetime peak working set
    uint64_t ws_private_bytes          = 0;      // private (anonymous) resident pages
    // Whether ws_private_bytes is a real reading. PROCESS_MEMORY_COUNTERS_EX2
    // provides the private resident split; an older SDK/OS falls back to EX and
    // this goes false with ws_private_bytes == 0. That distinction is load-
    // bearing: a zero read as "no private pages" would make the derived
    // file-backed share look like the ENTIRE working set, so consumers are told
    // which case they are in rather than left to infer it.
    bool     ws_private_known         = false;
    uint64_t commit_bytes              = 0;      // private commit (pagefile usage)
    uint64_t sys_total_bytes           = 0;      // installed physical RAM
    uint64_t sys_free_bytes            = 0;      // available physical RAM
    uint64_t sys_commit_total_bytes    = 0;      // system commit limit
    uint64_t sys_commit_free_bytes     = 0;      // system commit available
};

LLAMA_API llama_ls_host_mem llama_ls_host_mem_sample();

// T-GPLR-00: host-only GGUF inventory for the generic prefill layer-residency
// plan. Parses GGUF metadata with no_alloc (no tensor data, no CUDA backend
// init, no device enumeration) and returns a deterministic JSON report:
// architecture facts, per-class byte totals (static / router / shared-expert /
// attention / recurrent / expert-bank / mtp-nextn), per-layer expert-bank
// package sizes, KV bytes for the requested cache type at a context capacity,
// recurrent-state bytes, and the capacity comparison against a host-recorded
// physical VRAM total. Missing required metadata or unclassified tensors
// produce an explicit error result, never a guessed zero.
struct llama_ls_inventory {
    bool ok = false;
    std::string error; // set when ok == false
    std::string json;  // deterministic JSON report when ok == true
};

// T-GPLR-02: host-only architecture key reader (no CUDA, no tensor data) used
// by the runner to resolve the residency policy and the pageable-tensor
// override before model load. Returns general.architecture or empty on error.
LLAMA_API std::string llama_ls_model_arch(const char * path);

LLAMA_API llama_ls_inventory llama_ls_model_inventory(
        const char * path,
        int64_t      ctx_capacity,
        ggml_type    kv_type,
        uint64_t     physical_vram_bytes,
        const char * physical_vram_source);

// attach the streaming context to a llama context (before first decode)
void llama_context_set_layer_streaming(llama_context * ctx, llama_layer_streaming * ls);

// pre-build the persistent streaming graphs without executing them or leaving
// session state behind: both PP recurrent variants (fresh state rs_z = 0 and
// continuing state rs_z = -1) of `batch`, plus, when tg_tokens > 0, the TG
// u1 continuing-state variant (exported so the experimental runner can call it
// across the shared-library boundary; wraps llama_context::layer_streaming_prebuild)
LLAMA_API bool llama_layer_streaming_prebuild(llama_context * ctx, const llama_batch & batch, int64_t tg_tokens, ggml_status & ret);

// Layer-major PP controller (SDD section 13, T-Q38LM-02): run the full prompt
// in layer-major order (each executed logical layer processes every PP chunk in
// ascending token-position order before the next layer). Replaces the batch-major
// llama_decode for the timed PP region in layer-major mode. Returns 0 on success.
LLAMA_API int llama_layer_major_pp(llama_context * ctx, const llama_batch & batch, ggml_status & ret);

// T-GPLR-03: argmax of the last output row of the PP logits (the first token a
// normal-path greedy sampler would generate). layer_major_pp bypasses decode(),
// so the host logits buffer is never populated; this reads the exposed result
// tensor directly. Returns -1 when unavailable.
LLAMA_API int64_t llama_layer_major_pp_first_token(llama_context * ctx);

#endif // LLAMA_LAYER_STREAMING
