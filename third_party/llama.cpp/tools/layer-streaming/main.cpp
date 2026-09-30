// Experimental CUDA-only decoder-layer streaming runner (private research,
// gated by LLAMA_LAYER_STREAMING). Not an upstream-facing tool.
//
// Modes:
//   default        : PP (512 tokens / u-batch 64) and optional TG (greedy)
//   --boundary-layers 0,3 : execute only the selected layers through the
//                    one-layer execution boundary (T-Q38LS-02 proof)
//
// The runner loads the model with n_gpu_layers = -1 (static tensors, KV cache
// and recurrent state resident on CUDA) and a `blk\..*` tensor-buft override
// to the CPU buffer so decoder-layer packages stay in the host mmap and become
// the H2D copy sources for the weight slots.
//
// Modes:
//   default            : PP (+ optional TG)
//   --boundary-layers  : one-layer execution boundary proof
//   --check-telemetry  : host-side JSONL scheduler-invariant validator

#include "llama.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-layer-streaming.h"
#include "telemetry-json.h"
#include "telemetry-pool.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace json = ls_json;

static void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s (--model <path> | --model-env <name>) [options]\n"
        "options:\n"
        "  --layer-slots N        number of device weight slots (default 2; auto-upgraded to 3 when the\n"
        "                         budget allows; 1 = minimal-residency shape, no copy/compute overlap)\n"
        "  --boundary-layers L,M  execute only these logical layers (default: all)\n"
        "  --pp-tokens N          prompt tokens to process (default 512)\n"
        "  --pp-repeat N          run the same prefill N times in-process (T-GPLR-28 first-touch\n"
        "                         experiment; passes 2..N clear KV and reset per-pass stats; TG runs once)\n"
        "  --ubatch N             physical ubatch size (default 512; 64 is a manual small-ubatch override)\n"
        "  --tg-tokens N          greedy tokens to generate (default 0)\n"
        "  --no-mtp               skip MTP layers (default: skipped)\n"
        "  --no-mmap              load the model into ordinary host memory\n"
        "  --model-env NAME       read the model path from environment variable NAME\n"
        "  --kv-cache S           f16 (default), q8, or q4 for both K and V\n"
        "  --telemetry PATH       JSONL telemetry output (default: none)\n"
        "  --telemetry-chunk-stride N  emit layer-major PP chunk timings every N chunks (default: 10; always includes first and final)\n"
        "  --graph-budget-mib N   fail after the prebuild, before the timed PP, if the summed reserved\n"
        "                         graph-set io-context memory exceeds N MiB (default: no budget)\n"
        "  --pw-probe            T-Q38PW-00: per-width graph sets probe (prebuild one set, patch per chunk)\n"
        "  --handoff             T-Q38HG-02: PP->TG handoff (migrate all weights to device for resident TG)\n"
        "  --handoff-attention-only  T-Q38HG-04: constrained handoff (16 attention layers only)\n"
        "  --handoff-partial N   T-GPLR-16: NCMOE-style partial handoff (migrate banks il >= N; first N\n"
        "                         layers' expert banks stay host-resident for mixed-placement TG)\n"
        "  --residency-tg       T-GPLR-13: moe-row-set TG policy -- stage only the deduplicated routed\n"
        "                         expert ROWS each target row's router selects (reactive; requires\n"
        "                         --residency moe-expert-bank and --tg-tokens > 0; MTP verify\n"
        "                         widths are supported; emits rowset_* telemetry)\n"
        "  --residency-tg-spec <0|1|2>\n"
        "                       T-GPLR-58: speculative placement of GUESSED expert rows on the\n"
        "                         retention path (requires --residency-tg-retain). 0 = off (default),\n"
        "                         1 = scripted proposals via --residency-tg-spec-script (the A0 test\n"
        "                         vehicle), 2 = the A1 PROPOSER: the engine asks the arm itself.\n"
        "                         A guessed row is claimed resident only after its own copy\n"
        "                         completes, demands never wait on it, and a demand's capacity is\n"
        "                         never taken (emits spec telemetry)\n"
        "  --residency-tg-spec-arm <coact|bar09|fusion|d1>\n"
        "                       T-GPLR-58 A1: which past-only arm proposes, or T-GPLR-59 D1's\n"
        "                         MTP proposed-token table (mode 2 only; d1 requires --mtp)\n"
        "  --residency-tg-spec-null <0|1>\n"
        "                       T-GPLR-58 A1: replace the arm's PROPOSALS with the count-matched\n"
        "                         structure-free null (mode 2 only). The arm is still computed: it\n"
        "                         supplies the count the null is matched to (default 0)\n"
        "  --residency-tg-spec-pool N\n"
        "                       T-GPLR-59: separate prediction-only row pool with N positions per\n"
        "                         logical layer (first arm: 12). Requires mode 2; 0 = off.\n"
        "                         READY predictions may serve exact demand; PENDING never waits.\n"
        "  --residency-tg-spec-budget N\n"
        "                       T-GPLR-58 A0: max pending speculative fills per layer (default: top_k;\n"
        "                         the first T-GPLR-59 pool arm uses 12)\n"
        "  --residency-tg-spec-script <path>\n"
        "                       T-GPLR-58 A0: the scripted sequence, one line per entry:\n"
        "                         <layer> <ordinal> <expert> [<expert> ...]\n"
        "                         meaning speculate these ids for this layer's ordinal-th demand\n"
        "  --residency-tg-retain [N]\n"
        "                       T-GPLR-19: per-layer row residency (F2-M1) -- keep the last N\n"
        "                         windows' routed expert rows in PRIVATE per-layer storage so a\n"
        "                         token stages only the complement (requires --residency-tg;\n"
        "                         emits cache telemetry). N defaults to 8 (64 rows/layer,\n"
        "                         ~4.27 GiB over 40 MoE layers): the best measured TG gain\n"
        "                         and the highest measured honest (b), 0.7283, on the\n"
        "                         non-echoing stream; N=1 is the T-GPLR-19\n"
        "                         four-arm gate's measured shape. With NO value the depth is\n"
        "                         fitted to the device: the largest depth <= 8 that fits free\n"
        "                         VRAM less --reserve-mib and the slot window this run still\n"
        "                         has to allocate. A named N is never reduced -- it allocates\n"
        "                         in full or fails loudly\n"
        "  --mtp                 T-Q38HG-06: MTP speculative draft head in handoff TG (loads blk.64, migrates it, greedy-exact verify)\n"
        "  --mtp-draft N         draft tokens per round (default 2)\n"
        "  --prompt TEXT         real prompt text (overrides synthetic ids; pads to ubatch multiple for --mtp)\n"
        "  --rep-penalty F       greedy repeat-penalty factor (default 1 = off) applied identically\n"
        "                        at every greedy pick site (target, draft head, serial TG)\n"
        "  --rep-last-n N        repeat-penalty window (default 64)\n"
        "  --prompt-env NAME     read the prompt from environment variable NAME\n"
        "  --prompt-file PATH   T-GPLR-07: read the prompt text from PATH (conflicts loudly\n"
        "                         with --prompt/--prompt-env; required for >32 KB prompts)\n"
        "  --tg-text PATH       T-GPLR-07: write the detokenized generated text to PATH\n"
        "                         (default off; streaming path only; conflicts with --residency off)\n"
        "  --act-slab-dtype S    activation slab element type: f32 (default) or bf16\n"
        "  --pp-prefetch-ahead N T-GPLR-08: one-layer-ahead PrefetchVirtualMemory probe (default 0 = off;\n"
        "                         hint semantics; failed/skipped hints degrade to the page-fault path and\n"
        "                         are recorded; the probe runs N=1)\n"
        "  --pp-pool N[,policy]  T-GPLR-30: persistent pinned pool of N banks (default 0 = off); policy fixed\n"
        "                        (banks 0..N-1, savings-curve dial) or lru (N-entry LRU, cliff control);\n"
        "                        requires --copy-source pinned\n"
        "  --pp-async-feed N     T-GPLR-25: device copy-overlap prototype (default 1 since profile v3; N=1 stages the\n"
        "                         next layer's H2D copy on a worker thread while the current layer computes;\n"
        "                         requires --layer-slots >= 2)\n"
        "  --pp-af-serialize N   T-GPLR-32: keep ONE worker H2D DMA in flight (default 0 = off; N=1 orders each\n"
        "                         worker copy after the previous one via a stream event - the CPU staging hop\n"
        "                         stays overlapped; requires --pp-async-feed 1, ignored otherwise)\n"
        "  --pp-submit-batch N   T-GPLR-33: one H2D per bank when the members tile the slot region exactly\n"
        "                         (default 0 = off; N=1 issues a single span copy instead of one submission\n"
        "                         per member; layers that fail the contiguity check degrade to the legacy\n"
        "                         path and report submit_batch=0 on their copy_submit)\n"
        "  --pp-copy-overlap N   T-GPLR-35: staging DMAs on an auxiliary stream inside the compute\n"
        "                         backend's CUDA context, so bank transfers overlap layer compute\n"
        "                         (default 0 = off; N=1 requires the CUDA compute backend and refuses\n"
        "                         loudly instead of falling back to the serialized copy-stream path)\n"
        "  --pp-recon N         T-GPLR-35 diagnostics: stream/context identity once per run plus one\n"
        "                         per-bank sample of the staging copy's GPU window and its start offset\n"
        "                         against the compute stream's backlog (default 0 = off; adds no ordering)\n"
        "  --pp-mctx-recon N    T-GPLR-41 diagnostics: one mctx event per chunk + one mctx_kvview event per\n"
        "                       exec layer per chunk in the layer-major pass (pass-boundary memory\n"
        "                       lifecycle recon; default 0 = off)\n"
        "  --pp-prebuild-per-pass N\n"
        "                       T-GPLR-43: rebuild the persistent graph sets between --pp-repeat passes\n"
        "                       (fix for the quantized-KV pass-2 crash).\n"
        "                       DEFAULT (T-GPLR-50): on whenever --pp-repeat > 1 (f16 included:\n"
        "                       the T-GPLR-47 fresh-context rebuild made it compute-neutral;\n"
        "                       the 200k pass-2 excess returns without it). Previously (T-GPLR-48)\n"
        "                       on only for quantized KV (q8/q4).\n"
        "                       Explicit 0/1 always overrides the default.\n"
        "  --pp-no-logits       suppress PP logits (default: every PP token)\n"
        "  --tg-logit-margin N   T-GPLR-54 A2: record one `tg_logits` telemetry event per greedy\n"
        "                        TG step for the first N steps (<= 0 = off, the default). Read-only:\n"
        "                        annotates the existing pick with its top-1/top-2 values; never\n"
        "                        perturbs sampling. Consumed by tools/check-gplr54-logits.py.\n"
        "  --pp-schedule S      PP schedule: batch-major (default) or layer-major (opt-in experiment)\n"
        "  --residency-tg-window W  T-GPLR-62 S2: narrow the row-set slot window to W experts\n"
        "                       (0 = full bank, default). W must cover the worst-case\n"
        "                       union; a wider union refuses with 'window_overflow'.\n"
        "  --residency-tg-recon   T-GPLR-62 S2 A9: byte-compare every staged slot slab against\n"
        "                       the checkpoint bytes for that position. Slow; a mismatch\n"
        "                       refuses the run with 'recon_mismatch'.\n"
        "  --pp-prewarm N       T-GPLR-53 A4 pre-timed modality pre-warm, executed BEFORE the timed\n"
        "                       PP region: 0=off (default; no path runs), 1=CPU read-touch of the\n"
        "                       package source spans, 2=dummy GPU launch + sync, 3=both,\n"
        "                       4=duration-matched no-touch delay (the falsifier for 1),\n"
        "                       8=duration-matched wrong-region page touch (a 64 MiB scratch).\n"
        "                       Requires --pp-schedule layer-major; refused on the default path.\n"
        "  --pp-prewarm-delay-ms N\n"
        "                       Duration of the mode-4 no-touch delay in ms (0..60000, default\n"
        "                       1700 = the compiled control duration, so an un-flagged run is\n"
        "                       unchanged). Requires a pre-warm mode with bit 4 set.\n"
        "  --pp-span-pace N     T-GPLR-56 A3: paced registration of the mmap-registered\n"
        "                       source spans. 0=off (default; the one-shot path, byte-identical),\n"
        "                       1=paced: the first group is page-locked before the timed PP and\n"
        "                       every later group is issued from stage_members() a bounded\n"
        "                       distance ahead of the layer that needs it, so pending unlocked\n"
        "                       bytes stay near one group instead of 17.23 GiB. Requires\n"
        "                       --copy-source mmap-registered; refused, never clamped.\n"
        "  --pp-span-pace-ahead N\n"
        "                       Groups issued ahead of the layer being staged (0..64, default 1).\n"
        "                       Requires --pp-span-pace 1.\n"
        "  --pp-register-site S T-GPLR-56 A3 arm control: post-prebuild (default, today's\n"
        "                       site) or early (the earliest legal point, right after the\n"
        "                       boundary-layer selection; A2 measured that head start at 66.4 ms).\n"
        "                       Mutually exclusive with --pp-span-pace 1. Registration during\n"
        "                       the model load is forbidden (owner decision 2026-09-22).\n"
        "  --residency S        weight-residency policy: auto (default), off, dense-layer, moe-expert-bank\n"
        "  --fault-drop-member  T-GPLR-02 negative fixture: drop the last package member from the\n"
        "                       descriptor after init so load_layer must reject the partial package\n"
        "  --copy-source S      mmap-registered (DEFAULT since T-GPLR-37: page-lock the model's own\n"
        "                       mapped source ranges ONCE before the timed PP and DMA from them -\n"
        "                       no CPU hop, ~17 GiB of RAM held as pinned file pages; mutually\n"
        "                       exclusive with --pp-pool; measured -23.3%%/-15.4%% warm 4k PP at\n"
        "                       2/3 slots, wall-neutral at 200k), pinned (the pre-T-GPLR-37\n"
        "                       staging-buffer path), or direct host staging\n"
        "  --route-telemetry N  T-GPLR-39: emit the observational route event (the DIAGNOSTIC that\n"
        "                       reads router probs back from the device and sorts/hashes them on\n"
        "                       the host; 1161.5 ms/pass at u2048). Default 0 = off; the row-set\n"
        "                       ROUTING path is functional telemetry-independent and unaffected.\n"
        "                       Pass 1 to record expert routing (required by tools/check-route-parity.py\n"
        "                       and every recorded route analysis)\n"
        "  --copy-source mmap-registered-churn\n"
        "                       same, but re-registering per bank (the T-GPLR-37 variant that\n"
        "                       measured a 15-24%% warm regression; kept reproducible)\n"
        "  --check-telemetry PATH validate a JSONL file's scheduler invariants, print the per-phase telemetry summary (tok/s, transfer/wait/compute, peak VRAM), and exit\n"
        "  --validate-telemetry PATH strict validation + write a profile report (path.summary.json) and exit\n"
        "  --device N             CUDA device (default 0)\n"
        "  --inspect-model PATH   T-GPLR-00: host-only GGUF inventory (no CUDA, no tensor data), print JSON and exit\n"
        "  --ctx-size N           context/KV capacity for --inspect-model (default 262144)\n"
        "  --inspect-json PATH    write the inventory JSON to PATH (default: stdout)\n"
        "  --physical-vram-bytes N  host-recorded physical VRAM total for the capacity comparison (0 = unavailable)\n"
        "  --physical-vram-source S label for the VRAM source (e.g. nvidia-smi)\n"
        "  --residency-budget-mib N T-GPLR-04: explicit experimental TOTAL residency budget in MiB (0 = derive\n"
        "                         from the bounded one/two-slot estimate). The preflight proves the bounded\n"
        "                         configuration fits under it, and the T-GPLR-19 row cache must fit in what it\n"
        "                         leaves (bounded residency + cache <= budget); both abort loudly, before the\n"
        "                         timed PP or before the cache allocates, when they do not.\n"
        "  --reserve-mib N        T-GPLR-04: headroom reserve the bounded configuration must leave below physical\n"
        "                         VRAM (default 512 MiB).\n"
        "  --allow-full-residency T-GPLR-04: permit the run even if the full model would fit (the task contract\n"
        "                         requires an out-of-core proof; default refuses with a structured error).\n",
        prog);
}

// T-GPLR-19 (F2-M1): default retention depth for the per-layer row cache when
// --residency-tg-retain is given without a value. K=8 (64 rows/layer = 8 x
// top_k=8, ~4.27 GiB over the 40 MoE layers) is the depth with the best measured
// TG gain (+11.414 tok/s on the non-echoing stream) and the highest honest
// statistic (b) measured, 0.7283.
//
// Owner-directed 2026-09-18, superseding the 2026-09-14 K=4 default. K=4 was
// chosen as "the shallowest depth whose (b) clears the 0.6 gate", justified by
// 0.6284 on the preserved probe stream -- a stream that echoes its own anchor.
// The non-echoing re-probe that comment named as its outstanding validation was
// then executed and MISSES the threshold: (b) falls to 0.5355 at K=4, 0.6487 at
// K=6 and 0.7283 at K=8 (K=1 0.2468, K=2 0.3778; the echoed readings are
// 0.3088/0.4343/0.6284/0.7169/0.7775). So the one measured depth that failed
// the gate it was chosen for was the default itself, while already costing
// ~2.19 GiB -- the shallowest depth that clears 0.6 on the deciding stream is
// K=6, and K=8 is the owner's operating depth. The bare flag therefore resolves
// to the operating point instead of to a superseded one.
//
// This is still a CEILING, not a fixed allocation: the depth is fitted DOWNWARD
// to the cache allowance (free VRAM less --reserve-mib and the slot window, or
// --residency-budget-mib less the bounded residency), so a device that cannot
// afford K=8 provisions the largest depth that fits and one that cannot take
// K=1 is still refused loudly before timed work. It is never fitted UP, and a
// NAMED depth is never reduced in either direction.
//
// Cost, stated because it is real: where the device affords it the bare flag now
// allocates up to ~4.27 GiB instead of ~2.19 GiB, every byte of it out of the
// same allowance the slot window shares. The gain is measured and the lever is
// TG-only, so no depth changes prefill.
static const int kRowSetRetainDefault = 8;

// True when argv[i+1] is a plain decimal token, so an optional numeric value is
// consumed and a following flag never is. A non-decimal token after
// --residency-tg-retain is left to the argument loop, which rejects it loudly.
static bool is_decimal_token(int argc, char ** argv, int i) {
    if (i + 1 >= argc) {
        return false;
    }
    const char * s = argv[i + 1];
    if (s[0] == '\0') {
        return false;
    }
    for (const char * p = s; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }
    }
    return true;
}

// T-GPLR-19 (F2-M1): VRAM-adaptive retention depth. Returns the largest depth
// in [1, requested] whose predicted private cache fits `budget` bytes; 0 when
// not even K=1 fits (the caller must then fail loudly); -1 when the footprint
// cannot be derived at all. The prediction is the library's own per-member
// arithmetic, so a depth accepted here is a depth that allocates.
static int row_cache_fit_depth(const llama_layer_streaming & ls, int requested, size_t budget,
                               size_t * out_want, size_t * out_fit) {
    *out_want = 0;
    *out_fit  = 0;
    for (int k = requested; k >= 1; --k) {
        const size_t bytes = ls.row_cache_bytes_for(k);
        if (bytes == 0) {
            return -1; // not sizeable: begin_row_cache() reports the real reason
        }
        if (k == requested) {
            *out_want = bytes;
        }
        if (bytes <= budget) {
            *out_fit = bytes;
            return k;
        }
    }
    return 0;
}

// T-GPLR-22: the run_refused reason enum -- ONE owner for the emit sites and
// both checkers, so a reason cannot be emitted without being checkable. The
// first six are post-ls.init refusals; the last five are the pre-sink preflight
// gates and the pre-timed aborts. `detail` carries the exact gate and
// `cap_source` the limiting term, so the deciding condition is readable from the
// JSONL alone.
static const char * const kRunRefusalReasons[] = {
    "row_cache_derive",     // per-layer cache footprint could not be derived
    "row_cache_fit",        // nothing fits at any depth (not even K=1)
    "row_cache_depth",      // an explicit depth does not fit the allowance
    "row_cache_alloc",      // the cache allocation itself failed
    "row_cache_divergence", // fit prediction != actual allocation
    "prefetch_configure",   // the T-GPLR-08 prefetch probe could not be configured
    "capacity_preflight",   // T-GPLR-04: the capacity preflight refused
    "handoff_preflight",    // T-GPLR-10 / T-GPLR-16: the handoff fit gate refused
    "init_failed",          // ls.init returned false (streaming bring-up failed)
    "prebuild_failed",      // the PP persistent-graph prebuild failed
    "graph_budget",         // reserved io bytes exceeded --graph-budget-mib
};

static bool ls_run_refusal_reason_ok(const std::string & reason) {
    for (const char * r : kRunRefusalReasons) {
        if (reason == r) {
            return true;
        }
    }
    return false;
}

// T-GPLR-22 (PRE-SINK): the capacity/handoff preflights below run before
// `llama_layer_streaming ls` is constructed, so they refuse through the
// standalone writer. No-op when no telemetry path was requested.
static void refuse_before_sink(const std::string & tel_path, const char * reason,
                               const std::string & detail, uint64_t bytes_needed,
                               uint64_t bytes_allowed, const char * cap_source) {
    if (tel_path.empty()) {
        return;
    }
    if (!llama_layer_streaming::tel_run_refused_oneshot(tel_path.c_str(), reason, detail.c_str(), 1,
                                                       bytes_needed, bytes_allowed, cap_source)) {
        fprintf(stderr, "error: failed to write refusal telemetry %s\n", tel_path.c_str());
    }
}

static std::string arg_value(int argc, char ** argv, int & i, const char * name) {
    if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", name);
        exit(2);
    }
    return argv[++i];
}

// T-GPLR-06 follow-up: minimal repeat-penalty for the greedy pick sites,
// replicating llama_sampler_penalties_apply's penalty_repeat semantics
// exactly (positive logits divided by the factor, non-positive multiplied;
// ring of the last rep_last_n accepted tokens with a count map). One helper
// so the target verify rows, the draft-head chain, and the serial TG path
// apply the identical transform — the draft must predict what the penalized
// target picks, or acceptance becomes meaningless. Freq/present penalties
// are not implemented (the loop we are breaking is a pure repeat loop).
struct rep_penalty_ring {
    int last_n = 0;
    std::vector<llama_token> prev;
    std::unordered_map<llama_token, int> count;

    explicit rep_penalty_ring(int n) : last_n(n) { prev.reserve(n > 0 ? n : 1); }

    void accept(llama_token t) {
        if (last_n == 0) return;
        count[t]++;
        if ((int) prev.size() >= last_n) {
            const llama_token old = prev.front();
            if (--count[old] == 0) count.erase(old);
            prev.erase(prev.begin());
        }
        prev.push_back(t);
    }

    void apply(float * logits, int n_vocab, float penalty) const {
        if (last_n == 0 || penalty == 1.0f) return;
        for (int v = 0; v < n_vocab; ++v) {
            const auto it = count.find(v);
            if (it == count.end()) continue;
            logits[v] = (logits[v] <= 0.0f) ? logits[v] * penalty : logits[v] / penalty;
        }
    }
};

// T-GPLR-04: shared capacity arithmetic for the runner's two preflight/emission
// sites so they cannot drift apart. What "per-layer always-resident" means is a
// policy fact, not a model fact: moe-expert-bank pages only the routed-expert
// banks (every other blk.* tensor stays on CUDA0, so the per-layer non-expert
// classes are resident); dense-layer pages ALL blk.* tensors, so nothing
// per-layer is resident outside the slot window. Returns false when a required
// inventory field is missing so the caller can fail loudly.
struct llama_ls_bounded_window {
    uint64_t full_model_plus_runtime = 0;
    uint64_t bounded                 = 0;
    uint64_t static_bytes            = 0;
    uint64_t per_layer_resident      = 0;
    uint64_t slot_window             = 0;
    uint64_t kv_bytes                = 0;
    uint64_t state_bytes             = 0;
};

// T-GPLR-05: normal llama.cpp execution path used by `--residency off`.
// One owner for the off branch: plain batch-major PP + greedy TG exactly as
// stock llama.cpp would run, plus the off audit telemetry record. Returns the
// process exit status (0 = success).
static int run_residency_off(llama_context * ctx, llama_model * model,
                             const std::vector<llama_token> & prompt_tokens,
                             int64_t pp_tokens, int tg_tokens, bool pp_no_logits,
                             int64_t n_ctx_total, const char * model_arch,
                             const char * cap_reason, const char * tel_path) {
    fprintf(stderr, "residency off: normal llama.cpp placement (no streaming module)\n");
    const int64_t t0 = ggml_time_us();
    int exit_status = 0;
    std::vector<llama_token> generated;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init(pp_tokens, 0, 1);
    for (int i = 0; i < pp_tokens; ++i) {
        batch.token[i] = prompt_tokens.empty() ? (i % n_vocab) : prompt_tokens[i];
        batch.pos[i]   = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = !pp_no_logits;
    }
    batch.n_tokens = (int) std::min<int64_t>(pp_tokens, (int64_t) INT_MAX);
    const int ret = llama_decode(ctx, batch);
    if ( ret != 0) {
        fprintf(stderr, "error: off-mode PP decode failed with %d\n", ret);
        exit_status = 1;
    }
    llama_batch_free(batch);

    int64_t pp_first = -1;
    if (exit_status == 0 && !pp_no_logits) {
        // -1 = the LAST output row (output_resolve_row(-1) = n_outputs-1),
        // matching the last-prompt-position quantity the streaming path
        // exposes; index 0 would be the first batch token's logits here.
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits != nullptr) {
            llama_token best = 0;
            float best_val = logits[0];
            for (int v = 1; v < n_vocab; ++v) {
                if (logits[v] > best_val) { best_val = logits[v]; best = v; }
            }
            pp_first = best;
            fprintf(stderr, "PP-logits first greedy ID: %d\n", best);
        }
    }

    for (int t = 0; exit_status == 0 && t < tg_tokens; ++t) {
        llama_batch tb = llama_batch_init(1, 0, 1);
        const llama_token cur = generated.empty()
            ? (prompt_tokens.empty() ? (llama_token) (pp_tokens - 1) : prompt_tokens[pp_tokens - 1])
            : generated.back();
        tb.token[0] = cur;
        tb.pos[0]   = (llama_pos) (pp_tokens + t);
        tb.n_seq_id[0] = 1;
        tb.seq_id[0][0] = 0;
        tb.logits[0]  = true;
        tb.n_tokens   = 1;
        const int r = llama_decode(ctx, tb);
        if (r != 0) {
            fprintf(stderr, "error: off-mode TG decode failed with %d\n", r);
            exit_status = 1;
        } else {
            const float * logits = llama_get_logits_ith(ctx, 0);
            if (logits == nullptr) {
                fprintf(stderr, "error: no logits at off-mode TG step %d\n", t);
                exit_status = 1;
            } else {
                llama_token best = 0;
                float best_val = logits[0];
                for (int v = 1; v < n_vocab; ++v) {
                    if (logits[v] > best_val) { best_val = logits[v]; best = v; }
                }
                generated.push_back(best);
            }
        }
        llama_batch_free(tb);
    }

    if (!generated.empty()) {
        fprintf(stderr, "first 16 generated ids:");
        for (size_t i = 0; i < generated.size() && i < 16; ++i) {
            fprintf(stderr, " %d", (int) generated[i]);
        }
        fprintf(stderr, "\n");
    }

    if (tel_path != nullptr && tel_path[0] != '\0') {
        llama_layer_streaming ls_audit;
        if (!ls_audit.tel_off_run(tel_path, model_arch, cap_reason, exit_status,
                                  (int64_t) generated.size(), t0,
                                  pp_tokens, (int64_t) tg_tokens, n_ctx_total)) {
            fprintf(stderr, "error: failed to write off-mode telemetry %s\n", tel_path);
            return 1;
        }
    }
    (void) pp_first;
    return exit_status;
}

static bool ls_bounded_window_for_policy(const llama_ls_inventory & inv,
                                         const char * model_arch, int n_slots,
                                         llama_ls_bounded_window & out,
                                         int window_width = 0) {
    auto field = [&](const char * key, uint64_t & dst) {
        const std::string pat = std::string("\"") + key + "\":";
        const size_t p = inv.json.find(pat);
        if (p == std::string::npos) {
            return false;
        }
        dst = strtoull(inv.json.c_str() + p + pat.size(), nullptr, 10);
        return true;
    };
    uint64_t total_data = 0, kv_b = 0, st_b = 0, static_b = 0, per_layer = 0,
             moe_largest = 0, dense_largest = 0;
    if (!field("total_data_bytes", total_data) || !field("kv_bytes", kv_b) ||
        !field("recurrent_state_bytes", st_b) || !field("static", static_b) ||
        !field("per_layer", per_layer)) {
        return false;
    }
    out.full_model_plus_runtime = total_data + kv_b + st_b;
    out.kv_bytes                = kv_b;
    out.state_bytes             = st_b;
    out.static_bytes            = static_b;
    const bool moe_policy = (std::string(model_arch) == "qwen35moe");
    if (moe_policy) {
        if (!field("moe_largest_bytes", moe_largest) || moe_largest == 0) {
            return false;
        }
        out.per_layer_resident = per_layer;
        // T-GPLR-62 S2: the routed-expert bank is n_expert experts wide, and
        // `moe_largest_bytes` is the WHOLE bank for the largest layer. A compact
        // row-set window of W experts needs W/n_expert of it, so the preflight
        // has to reserve the narrow figure or the run refuses for memory it is
        // not going to use. Rounded UP so the estimate never under-reserves.
        //
        // The dense policy below is NOT narrowed: a dense layer's slot window
        // pages the whole layer, so its extent is not an expert axis.
        if (window_width > 0) {
            uint64_t n_exp = 0;
            if (!field("expert_count", n_exp) || n_exp == 0) {
                // Cannot scale without the bank width: refuse to guess, and keep
                // the full-bank figure so the run is over-reserved rather than
                // under-reserved.
                fprintf(stderr,
                        "warning: --residency-tg-window %d requested but the inventory has no "
                        "expert_count; preflight keeps the FULL-BANK slot window\n", window_width);
                out.slot_window = (uint64_t) n_slots * moe_largest;
            } else if ((uint64_t) window_width > n_exp) {
                fprintf(stderr,
                        "warning: --residency-tg-window %d exceeds expert_count %llu; "
                        "preflight keeps the FULL-BANK slot window\n",
                        window_width, (unsigned long long) n_exp);
                out.slot_window = (uint64_t) n_slots * moe_largest;
            } else {
                const uint64_t numer = moe_largest * (uint64_t) window_width;
                out.slot_window = (uint64_t) n_slots * ((numer + n_exp - 1) / n_exp);
            }
        } else {
            out.slot_window        = (uint64_t) n_slots * moe_largest;
        }
    } else {
        // dense-layer policy: all blk.* tensors page through the slot window;
        // nothing per-layer stays resident outside it.
        if (!field("dense_largest_bytes", dense_largest) || dense_largest == 0) {
            return false;
        }
        out.per_layer_resident = 0;
        out.slot_window        = (uint64_t) n_slots * dense_largest;
    }
    out.bounded = static_b + out.per_layer_resident + out.slot_window + kv_b + st_b;
    return true;
}

// T-GPLR-04 / T-GPLR-19: ONE VRAM budget model for the whole bring-up path.
//
// The capacity preflight (--residency-budget-mib / --reserve-mib) and the
// per-layer row-cache fit used to derive their own reserve, their own
// held-back set and their own device reading, so two sites could report
// different amounts of already-allocated or still-to-allocate memory for one
// device. They now share this object: the inputs come from the budget flags
// and from ls_bounded_window_for_policy(), the device is read only through the
// functions below, and every derived byte quantity is computed once, here.
//
// Two readings are kept deliberately, because they measure different moments
// and both are load-bearing:
//   * free_start -- taken BEFORE ls.init's static allocations. The T-GPLR-04
//     preflight proves its out-of-core and capacity conditions against this
//     one, and run_start telemetry reports it.
//   * free_now   -- refreshed AFTER ls.init, where the static/KV/state classes
//     are already outside free VRAM. The T-GPLR-19 row-cache fit budgets from
//     this one, so it needs no second notion of what is already allocated.
//
// --residency-budget-mib is a TOTAL residency cap, not a window-only cap: the
// bounded residency (static + per-layer resident + slot window + KV + state)
// PLUS the row cache must fit under it, so no retain flag can place a cache on
// top of a budget the operator set. Activation slabs stay outside it: they are
// a measured runtime class gated against observed free VRAM (condition 3), not
// a residency this mechanism manages. With no explicit budget the bounded
// window is its own cap and the cache stays free-derived, exactly as before.
struct llama_ls_vram_budget {
    // model inputs: the budget flags and the bounded-window estimate
    int64_t  reserve_mib          = 0; // --reserve-mib, the preflight headroom
    int64_t  residency_budget_mib = 0; // --residency-budget-mib, 0 = derive
    uint64_t bounded              = 0; // static + per-layer resident + slot window + KV + state
    uint64_t slot_window          = 0; // begin_row_set() allocates this AFTER the row cache

    // device readings
    size_t   free_start = 0;
    size_t   free_now   = 0;
    size_t   total      = 0; // a device property, not a per-reading quantity
    // T-GPLR-24 (H-B): the lowest free-VRAM reading ever observed this run.
    // The row cache is released before run_end samples the device, so a
    // two-sample peak understates true usage by exactly the cache size.
    // Updated at every refresh point; tel_run_end reports total - free_min.
    size_t   free_min   = 0;

    // derived
    uint64_t reserve_bytes          = 0; // the reserve in bytes
    uint64_t residency_budget_bytes = 0; // the operator's TOTAL residency cap in bytes
    uint64_t cache_cap_free         = 0; // what free VRAM leaves the row cache
    uint64_t cache_cap_budget       = 0; // what the operator's total cap leaves it
    size_t   cache_budget           = 0; // the cache's allowance: the binding minimum
};

// The one derivation. Every function below calls it and nothing else does this
// arithmetic, so the preflight's numbers and the cache fit's numbers are the
// same numbers.
static void ls_vram_budget_derive(llama_ls_vram_budget & b) {
    b.reserve_bytes = (uint64_t) (b.reserve_mib > 0 ? b.reserve_mib : 0) * 1024u * 1024u;
    // No explicit residency budget: the bounded window itself is the cap.
    b.residency_budget_bytes = b.residency_budget_mib > 0
        ? (uint64_t) b.residency_budget_mib * 1024u * 1024u
        : b.bounded;
    // The row cache may own whatever the CURRENT reading leaves after the
    // operator's reserve and the slot window this run has not allocated yet:
    // the cache must never be the allocation that starves the streaming window
    // it exists to serve.
    const uint64_t held = b.reserve_bytes + b.slot_window;
    b.cache_cap_free = (uint64_t) b.free_now > held ? (uint64_t) b.free_now - held : 0;
    // An explicit --residency-budget-mib caps the cache from the other side:
    // `bounded` is exactly the non-cache residency that is live next to the
    // cache (it already counts the slot window begin_row_set() allocates
    // later), so the cap leaves the cache the budget's remainder. With no
    // explicit budget this term does not bind and the cache stays free-derived.
    b.cache_cap_budget = b.residency_budget_mib > 0
        ? (b.residency_budget_bytes > b.bounded ? b.residency_budget_bytes - b.bounded : 0)
        : b.cache_cap_free;
    b.cache_budget = (size_t) (b.cache_cap_free < b.cache_cap_budget ? b.cache_cap_free : b.cache_cap_budget);
}

// True when the operator's explicit --residency-budget-mib, not free VRAM, is
// what limited the row cache. A refusal that blames free VRAM when the budget
// bound sends the operator to the wrong flag.
static bool ls_vram_budget_cache_capped_by_budget(const llama_ls_vram_budget & b) {
    return b.residency_budget_mib > 0 && b.cache_cap_budget <= b.cache_cap_free;
}

// Adopt the budget flags and take the pre-ls.init reading.
static void ls_vram_budget_measure_start(llama_ls_vram_budget & b, int device,
                                         int64_t reserve_mib, int64_t residency_budget_mib) {
    b.reserve_mib          = reserve_mib;
    b.residency_budget_mib = residency_budget_mib;
    ggml_backend_cuda_get_device_memory(device, &b.free_start, &b.total);
    b.free_now = b.free_start;
    b.free_min = b.free_start;
    ls_vram_budget_derive(b);
}

// Adopt the bounded-window estimate; the derived quantities follow from it.
static void ls_vram_budget_adopt_window(llama_ls_vram_budget & b, const llama_ls_bounded_window & win) {
    b.bounded     = win.bounded;
    b.slot_window = win.slot_window;
    ls_vram_budget_derive(b);
}

// Re-read the device against the same inputs. Used after ls.init, where the
// static classes the preflight's own reading still counted are allocated.
static void ls_vram_budget_refresh(llama_ls_vram_budget & b, int device) {
    ggml_backend_cuda_get_device_memory(device, &b.free_now, &b.total);
    if (b.free_min == 0 || b.free_now < b.free_min) {
        b.free_min = b.free_now;
    }
    ls_vram_budget_derive(b);
}

// ---------------------------------------------------------------------------
// --check-telemetry: host-side JSONL scheduler-invariant validator (no GPU)
// ---------------------------------------------------------------------------

// T-GPLR-10: runner-captured expected handoff migrate event values for the
// in-process trace self-check (captured right after begin_handoff succeeds;
// 0 = no expectation, e.g. standalone --check-telemetry PATH runs).
static long long g_check_handoff_layers = 0;
static long long g_check_handoff_bytes = 0;

// T-GPLR-05: single owner of the off-run audit-trace rules shared by the
// check-telemetry and validate-telemetry validators. Only rules that are
// byte-identical in both validators live here; class-specific sequencing
// (duplicate run_start, run_end required, phase aggregates) stays with each
// validator.
static bool ls_line_is_off_audit(const std::string & line) {
    return json::value(line, "policy") == "off" || json::value(line, "residency") == "off";
}

// run_start metadata rule: pp/tg/ctx must be present and sane in off audit
// records. Reports the violation and counts it (mirrors the per-validator err
// lambda).
static void ls_check_off_run_start(const std::string & line, int line_no, const char * tag, int & errors) {
    bool found = false;
    const long long pp_m = json::integer(line, "pp_tokens", &found);
    const long long tg_m = json::integer(line, "tg_tokens", &found);
    const long long ctx_m = json::integer(line, "ctx_size", &found);
    if (!found || pp_m <= 0 || tg_m < 0 || ctx_m <= 0) {
        fprintf(stderr, "%s: line %d: off run_start missing valid pp_tokens/tg_tokens/ctx_size\n", tag, line_no);
        errors++;
    }
}

// Streaming scheduler events must never appear in an off-mode audit trace.
// Reports the violation and counts it (mirrors the per-validator err lambda).
static void ls_err_off_streaming_event(int line_no, const char * tag, int & errors) {
    fprintf(stderr, "%s: line %d: streaming scheduler event in an off-mode audit trace\n", tag, line_no);
    errors++;
}

// T-GPLR-19 (F2-M1): per-layer retention conservation, shared by BOTH
// validators so they stay identical on rowset events. Under retention
// residency is the LAYER's private cache, never slot coverage, so the
// slot-keyed lifecycle rule must not run; this rule instead RECOMPUTES the
// retained set from the cache's own provenance at the window the trace
// reports. That is what makes the byte verdict safe: an implementation that
// retained rows older than the declared K would report a saving larger than
// the K it declared, and fails here instead of inflating the gate.
//
// The cache is a SLIDING union of the last K windows, so this must mirror the
// runtime exactly: a row older than K windows is unusable and is retired (it
// can never be claimed as retained), a row in this window's union is kept and
// its provenance PROMOTED to this window, and a row that is live now but
// cannot survive into the next window retires with it. Bounded by
// min(n_expert, K*W*top_k) at W=1 as a result, not as an assumption.
// `cache` is keyed by ((layer << 32) | expert) and holds each row's serving
// window, so it is the trace's own claim about what is resident.
// T-GPLR-19 retention rule, checker side. Policy resolved 2026-09-13: the
// per-layer row cache is a ROW-CAPACITY LRU over the layer's routed rows, the
// policy SDD gate 9's statistic (b) measures -- NOT a window-count membership
// set. Footprint is cap = min(n_expert, K*top_k) rows per layer under either
// rule, and recency eviction weakly dominates: a row routed in the last K
// windows has at most cap-1 rows routed after it, so it is always among the cap
// most recently routed. Both rules are identical at K=1; they diverge only at
// K>=2, where the LRU can only retain more.
// `tick` is a monotonic recency source shared across the whole trace; only the
// RELATIVE order matters, so it need not match the runtime's counter.
//
// T-GPLR-58: the model also replays the SPECULATIVE fills, which is what makes it a
// model of the cache the runtime actually held and not of the demand stream alone.
// It calls the runtime's OWN pure rules (llama_ls_spec_is_hit / _may_evict /
// _victim_better), so the checker cannot drift from the engine, and it keeps the
// per-position speculative state the runtime keeps:
//   spec_state  FREE / READY / PENDING / CANCELLED -- an in-flight position holds its
//               bytes: never a hit, never a victim, and a demand for that very row
//               fetches its own copy (P1/P2/P5);
//   spec        1 = placed by a fill and not yet consumed, so it is the FIRST victim
//               (P3). A consumed fill is an ordinary resident again.
// The recency order is the runtime's: the union is stamped in its ascending-id order
// and each fill at ISSUE, so only the absolute tick value differs (nothing reads it.
struct ls_retention_row {
    long long recency    = 0;   // the runtime's cache_tick stamp for this row
    bool      spec       = false; // placed by a fill, not yet consumed: first victim
    uint8_t   spec_state = LLAMA_LS_SPEC_FREE; // the runtime's per-position spec state
};

// T-GPLR-58: a `spec` event is written BEFORE the rowset_fetch of the stage it belongs
// to (spec_issue_next runs inside load_row_set, ahead of tel_rowset_fetch), so its
// fills are STAGED here and applied after that fetch -- which is also the order the
// runtime applies them in, and the only order in which a fill is PENDING while the next
// window classifies it and READY when the window after that may evict it.
struct ls_retention_model {
    std::unordered_map<long long, ls_retention_row> rows; // (layer<<32)|expert -> the row
    long long tick = 0;                       // monotonic recency source (the runtime's)
    std::unordered_map<long long, std::vector<long long>> staged; // layer -> this stage's fills
    std::unordered_set<long long>             staged_seen;         // layers with a staged spec event
    std::unordered_map<long long, long long>  last_gen;            // layer -> last generation
    std::unordered_map<long long, long long>  last_phase;          // layer -> last phase
    // A trace recorded before `fill_ids` existed reports the fills' COUNT but not their
    // ids, so the model cannot replay them. That is UNVERIFIABLE, not wrong: the per-fetch
    // rules are suppressed and the leg fails ONCE with the cause named, rather than
    // blaming every later fetch (216 violations on the recorded null@K8 leg, 2026-09-23).
    bool      unreplayable         = false;
    long long unreplayable_stages  = 0;
};

// One `spec` event: record the fills this stage issued. Sets model.unreplayable when the
// event reported fills without their ids.
static std::vector<std::string> ls_spec_errors(ls_retention_model & model, const std::string & line) {
    std::vector<std::string> out;
    bool have_layer = false;
    const long long layer = json::integer(line, "layer", &have_layer);
    if (!have_layer || layer < 0) {
        out.push_back("spec missing layer");
        return out;
    }
    bool have_fills = false;
    const std::vector<long long> fills = json::integer_list(line, "fill_ids", &have_fills);

    // Mode 2 admission telemetry is additive: older traces have no candidate_ids and
    // remain replayable only through their legacy path. A trace with any new field must
    // carry the complete per-stage schema, or the checker refuses the ambiguous record.
    const bool has_admission_field =
        line.find("\"candidate_ids\"") != std::string::npos ||
        line.find("\"skipped_present\"") != std::string::npos ||
        line.find("\"skipped_dup\"") != std::string::npos ||
        line.find("\"plan_refused_capacity\"") != std::string::npos ||
        line.find("\"stop_reason\"") != std::string::npos;
    if (has_admission_field) {
        const char * required[] = {"candidate_ids", "skipped_present", "skipped_dup",
                                   "plan_refused_capacity", "stop", "stop_reason", "fill_ids"};
        for (const char * key : required) {
            if (line.find(std::string("\"") + key + "\"") == std::string::npos) {
                out.push_back(std::string("spec admission telemetry missing ") + key);
            }
        }
        bool have_candidates = false;
        const std::vector<long long> candidates = json::integer_list(line, "candidate_ids", &have_candidates);
        bool have_stop = false;
        const long long stop = json::integer(line, "stop", &have_stop);
        const std::string stop_reason = json::value(line, "stop_reason");
        if (have_candidates && have_stop) {
            if (stop < 0 || stop > 4) {
                out.push_back("spec admission stop is outside the planner enum");
            }
            const char * expected = stop == 0 ? "proposals" :
                                    stop == 1 ? "budget" :
                                    stop == 2 ? "pending" :
                                    stop == 3 ? "reserve" :
                                    stop == 4 ? "no_free" : "";
            if (expected[0] == '\0' || stop_reason != expected) {
                out.push_back("spec admission stop_reason does not match stop");
            }
            bool have_prop = false;
            const long long prop = json::integer(line, "prop", &have_prop);
            if (!have_prop || prop != (long long) candidates.size()) {
                out.push_back("spec candidate_ids length does not match prop");
            }
            std::unordered_set<long long> unique;
            for (long long id : candidates) {
                if (id < 0 || !unique.insert(id).second) {
                    out.push_back("spec candidate_ids carries a negative or duplicate id");
                    break;
                }
            }
        }
        if (have_fills) {
            const long long submitted = json::integer(line, "submitted", nullptr);
            if (submitted != (long long) fills.size()) {
                out.push_back("spec fill_ids length does not match submitted");
            }
            std::unordered_set<long long> unique;
            for (long long id : fills) {
                if (id < 0 || !unique.insert(id).second) {
                    out.push_back("spec fill_ids carries a negative or duplicate id");
                    break;
                }
            }
        }
    }

    if (!have_fills) {
        // `submitted` is PER-STAGE since T-GPLR-58 A2, so only a positive `submitted_total`
        // (or a pre-A2 cumulative `submitted`) proves fills were issued at all. The reap still
        // happened either way, so the layer stays staged.
        const long long submitted = json::integer(line, "submitted", nullptr);
        const long long total     = json::integer(line, "submitted_total", nullptr);
        if (submitted > 0 || total > 0) {
            model.unreplayable = true;
            model.unreplayable_stages++;
        }
        model.staged_seen.insert(layer);
        return out;
    }
    for (long long x : fills) {
        if (x < 0) {
            out.push_back("spec fill_ids carries a negative expert id");
        }
    }
    model.staged[layer] = fills;
    model.staged_seen.insert(layer);
    return out;
}

// The deferred, leg-level form of the rule above: one error, naming the cause.
static std::vector<std::string> ls_spec_replay_errors(const ls_retention_model & model) {
    std::vector<std::string> out;
    if (model.unreplayable) {
        out.push_back("spec: " + std::to_string(model.unreplayable_stages) +
                      " stage(s) reported speculative fills with no `fill_ids`, so the retention "
                      "cache model cannot replay them and the per-fetch retention rules are "
                      "UNVERIFIED for this trace (recorded before T-GPLR-58 A2)");
    }
    return out;
}

static std::vector<std::string> ls_retention_errors(
        ls_retention_model & model,
        long long layer, long long generation, long long phase,
        long long route_union, long long n_rows_resident,
        const std::vector<long long> & pool_hit_ids,
        const std::string & ids, long long n_retained, long long n_retired,
        long long retain_k, long long n_expert, long long top_k) {
    std::vector<std::string> out;
    if (model.unreplayable) {
        return out; // every per-fetch rule would be downstream of the missing fills
    }
    const long long layer_hi = layer << 32;
    const long long cap = std::min(n_expert, retain_k * top_k);

    // this window's union, needed for set membership below
    std::vector<long long> uniq;
    {
        size_t pos = 0;
        while (pos < ids.size()) {
            const size_t comma = ids.find(',', pos);
            const std::string tok = ids.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            uniq.push_back(atoll(tok.c_str()));
            if (comma == std::string::npos) {
                break;
            }
            pos = comma + 1;
        }
    }

    // 0. a generation boundary: the runtime cancels every in-flight fill issued under a
    //    stale generation before it classifies the window, and a PHASE boundary
    //    (begin_row_set) cancels AND reaps them before the phase's first window, so those
    //    positions are already free when the new phase starts.
    {
        const auto g = model.last_gen.find(layer);
        if (g != model.last_gen.end() && g->second != generation) {
            const auto ph = model.last_phase.find(layer);
            const bool new_phase = ph != model.last_phase.end() && ph->second != phase;
            for (auto it = model.rows.begin(); it != model.rows.end(); ) {
                if ((it->first & ~0xffffffffLL) != layer_hi) {
                    ++it;
                    continue;
                }
                const bool in_flight = it->second.spec_state == LLAMA_LS_SPEC_PENDING ||
                                       it->second.spec_state == LLAMA_LS_SPEC_CANCELLED;
                if (in_flight && new_phase) {
                    it = model.rows.erase(it); // drained at the phase boundary
                    continue;
                }
                if (it->second.spec_state == LLAMA_LS_SPEC_PENDING) {
                    it->second.spec_state = LLAMA_LS_SPEC_CANCELLED; // stale generation
                }
                ++it;
            }
        }
        model.last_gen[layer]   = generation;
        model.last_phase[layer] = phase;
    }

    // 1. classify: which union members the cache already holds as a HIT. Only a READY row
    //    is a hit (P2): a PENDING row this window wants is a duplicate (cancelled here -- its
    //    bytes are superseded by the demand's own copy) and a CANCELLED row is already
    //    superseded. Eviction never displaces a row this window needs, so this is also the
    //    reported resident count both before and after the eviction step below.
    long long resident = 0;
    std::unordered_set<long long> resident_ids;
    for (long long x : uniq) {
        auto it = model.rows.find(layer_hi | (long long) (unsigned) x);
        if (it == model.rows.end()) {
            continue;
        }
        if (llama_ls_spec_is_hit(it->second.spec_state)) {
            ++resident;
            resident_ids.insert(x);
        } else if (it->second.spec_state == LLAMA_LS_SPEC_PENDING) {
            it->second.spec_state = LLAMA_LS_SPEC_CANCELLED; // demand duplicate
        }
    }
    for (long long id : pool_hit_ids) {
        if (resident_ids.count(id)) {
            out.push_back("rowset_fetch reports a pool hit for an id already resident in the real cache");
        }
    }
    long long occupied = 0;
    for (const auto & e : model.rows) {
        if ((e.first & ~0xffffffffLL) == layer_hi) {
            ++occupied; // physical occupancy: an in-flight fill holds its position
        }
    }
    const long long need = (long long) uniq.size() - resident - (long long) pool_hit_ids.size();

    // 2. recency eviction, BEFORE placement -- the same rule the runtime applies, so the
    //    trace cannot describe a cache the runtime could not have held: a speculative row is
    //    the FIRST victim (P3), then the least recently routed position that this window does
    //    not need, until the fetched complement fits. An in-flight position is never a victim.
    long long expect_retired = 0;
    while (occupied + need > cap) {
        auto victim = model.rows.end();
        for (auto it = model.rows.begin(); it != model.rows.end(); ++it) {
            if ((it->first & ~0xffffffffLL) != layer_hi) {
                continue;
            }
            if (!llama_ls_spec_may_evict(it->second.spec_state)) {
                continue; // bytes in flight: the reap releases it, an eviction never does
            }
            const long long x = (long long) (unsigned) (it->first & 0xffffffffLL);
            if (std::find(uniq.begin(), uniq.end(), x) != uniq.end()) {
                continue; // needed by this window: never evicted
            }
            if (victim == model.rows.end() ||
                llama_ls_spec_victim_better(it->second.recency, it->second.spec, true,
                                            victim->second.recency, victim->second.spec, true)) {
                victim = it;
            }
        }
        if (victim == model.rows.end()) {
            out.push_back("rowset_fetch retention cannot fit the union in the declared capacity");
            return out;
        }
        model.rows.erase(victim);
        --occupied;
        ++expect_retired;
    }
    if (expect_retired != n_retired) {
        out.push_back("rowset_fetch n_rows_retired != rows evicted by the row-capacity policy");
    }

    // 3. stamp recency across the WHOLE union in ascending expert-id order, so a hit and a
    //    miss refresh recency alike and the in-window tie-break is deterministic -- the
    //    runtime's own order
    const long long expect_retained = resident;
    std::vector<long long> asc = uniq;
    std::sort(asc.begin(), asc.end());
    for (long long x : asc) {
        if (std::find(pool_hit_ids.begin(), pool_hit_ids.end(), x) != pool_hit_ids.end()) {
            continue; // a pool hit is temporary slot input, never real-cache retention
        }
        const long long k = layer_hi | (long long) (unsigned) x;
        auto it = model.rows.find(k);
        if (it != model.rows.end() && it->second.spec_state == LLAMA_LS_SPEC_CANCELLED) {
            // The demand fetched its own copy because the fill was still pending.
            // The cancelled position is released by the reap after this window, so
            // the post-window cache is represented by the demand row, not the
            // superseded fill that happened to have the same expert id.
            it->second.recency    = ++model.tick;
            it->second.spec_state = LLAMA_LS_SPEC_READY;
            it->second.spec       = false;
        } else if (it != model.rows.end()) {
            it->second.recency    = ++model.tick;
            it->second.spec_state = LLAMA_LS_SPEC_READY;
            it->second.spec       = false; // a consumed fill is an ordinary resident
        } else {
            ls_retention_row r;
            r.recency    = ++model.tick;
            r.spec       = false; // a row the demand placed is never speculative
            r.spec_state = LLAMA_LS_SPEC_READY;
            model.rows.emplace(k, r);
        }
    }
    if (expect_retained != n_retained || expect_retained != n_rows_resident) {
        out.push_back("rowset_fetch retention accounting mismatch (cache-derived != reported)");
    }

    // 4. the cache can never exceed its declared capacity
    long long layer_rows = 0;
    for (const auto & e : model.rows) {
        if ((e.first & ~0xffffffffLL) == layer_hi) {
            ++layer_rows;
        }
    }
    if (layer_rows > cap) {
        out.push_back("rowset_fetch layer cache exceeds min(n_expert, K*top_k)");
    }
    if (route_union > n_expert) {
        out.push_back("rowset_fetch route_union exceeds n_expert");
    }

    // 5. this stage's speculation, applied AFTER the window it belongs to: the runtime reaps
    //    first (PENDING -> READY and speculative, CANCELLED released), then issues this
    //    stage's fills, each into a FREE position with a fresh recency stamp, in the order
    //    `fill_ids` lists them.
    if (model.staged_seen.count(layer)) {
        for (auto it = model.rows.begin(); it != model.rows.end(); ) {
            if ((it->first & ~0xffffffffLL) != layer_hi) {
                ++it;
                continue;
            }
            if (it->second.spec_state == LLAMA_LS_SPEC_CANCELLED) {
                it = model.rows.erase(it); // released without a residency claim
                continue;
            }
            if (it->second.spec_state == LLAMA_LS_SPEC_PENDING) {
                it->second.spec_state = LLAMA_LS_SPEC_READY;
                it->second.spec       = true; // a ready fill is the first victim
            }
            ++it;
        }
        const auto st = model.staged.find(layer);
        if (st != model.staged.end()) {
            for (long long x : st->second) {
                const long long k = layer_hi | (long long) (unsigned) x;
                if (model.rows.find(k) != model.rows.end()) {
                    out.push_back("spec fill_ids repeats a row the cache already holds");
                    continue;
                }
                ls_retention_row r;
                r.recency    = ++model.tick; // stamped at ISSUE, as the runtime does
                r.spec       = false;        // the flag is set at the reap
                r.spec_state = LLAMA_LS_SPEC_PENDING;
                model.rows.emplace(k, r);
            }
        }
        model.staged.erase(layer);
        model.staged_seen.erase(layer);
    }
    return out;
}

// T-GPLR-19: single owner of the byte_account TOTAL-CAP rule shared by the
// check-telemetry and validate-telemetry validators. The producer emits
// `row_cache` (the private VRAM the per-layer cache actually allocated) and
// `residency_cap` (the operator's explicit --residency-budget-mib in bytes, 0
// = no explicit cap) next to the bounded-window account, so the cap the runner
// enforces at bring-up is checkable from the JSONL alone: the bounded
// residency PLUS the provisioned cache must fit under it. Traces recorded
// before the fields existed carry neither and are accepted -- the rule can only
// be enforced where the producer reports it -- but a record that reports one
// field without the other is rejected, since that is a half-migrated producer.
static std::vector<std::string> ls_byte_account_errors(
        long long bounded, long long residency_budget_mib,
        bool have_row_cache, long long row_cache,
        bool have_prediction_pool, long long prediction_pool,
        bool have_cap, long long cap) {
    std::vector<std::string> out;
    if (!have_row_cache && !have_prediction_pool) {
        return out; // pre-field trace
    }
    if (have_row_cache && row_cache < 0) {
        out.push_back("byte_account: row_cache must not be negative");
    }
    if (have_prediction_pool && prediction_pool <= 0) {
        out.push_back("byte_account: prediction_pool must be positive when present");
    }
    if (!have_cap) {
        out.push_back("byte_account: cache bytes present without residency_cap");
        return out;
    }
    // The cap field must be the flag it claims to be, in bytes; 0 means the run
    // set no budget (a nonzero cap there would claim a bound nobody asked for).
    const long long expect_cap = residency_budget_mib > 0 ? residency_budget_mib * 1024 * 1024 : 0;
    if (cap != expect_cap) {
        out.push_back("byte_account: residency_cap != --residency-budget-mib in bytes");
    }
    const long long cache_bytes = (have_row_cache ? row_cache : 0) +
                                  (have_prediction_pool ? prediction_pool : 0);
    if (cap > 0 && bounded + cache_bytes > cap) {
        out.push_back("byte_account: bounded residency + real cache + prediction pool exceeds the total residency cap");
    }
    return out;
}

// T-GPLR-19: the deferred half of the same rule. byte_account precedes
// rowset_init, so the cache figure can only be tied to the declared retention
// policy once the whole trace is read. Without that tie a zero (or a stray
// figure) could pass as a real account; with it, the record must agree with the
// mechanism it describes. retain_k < 0 means this trace declared no rowset
// policy at all, which nothing here can contradict.
static std::vector<std::string> ls_byte_account_policy_errors(
        bool have_row_cache, long long row_cache, long long retain_k) {
    std::vector<std::string> out;
    if (!have_row_cache || retain_k < 0) {
        return out;
    }
    if (retain_k > 0 && row_cache <= 0) {
        out.push_back("byte_account: row_cache is zero in a run that declares per-layer retention");
    } else if (retain_k == 0 && row_cache > 0) {
        out.push_back("byte_account: row_cache is nonzero with per-layer retention off");
    }
    return out;
}

// T-GPLR-22: the refusal-trace rule. ONE owner, called by both `--check-telemetry`
// and `--validate-telemetry`. Rejected on a refusal trace: a phase_summary (a
// completed phase means this was a failed run, not a refusal) and a run_end
// (run_refused is terminal). A run_start is ADMISSIBLE -- the pre-timed abort
// shape -- so both messages name the shape they reject. With no refusal present
// every input is false and this is a no-op, so pre-event traces keep the rules
// they had and the pre-event refusal shape (a bare pkg_member fragment) is still
// rejected rather than retroactively accepted.
static std::vector<std::string> ls_refusal_trace_errors(bool saw_refusal, bool reason_ok,
                                                        bool saw_run_start, bool saw_phase_summary,
                                                        bool saw_run_end) {
    std::vector<std::string> out;
    if (!saw_refusal) {
        return out;
    }
    if (!reason_ok) {
        out.push_back("run_refused: reason is missing or not one of the recognized refusal reasons");
    }
    // A run_start is admissible (the ABORTED-RUN shape); naming the shape keeps
    // the two clauses below honest about what they reject.
    const std::string shape = saw_run_start ? "an aborted-run refusal" : "a pre-work refusal";
    if (saw_phase_summary) {
        out.push_back("run_refused: " + shape +
                      " must have no phase_summary (a completed phase means this was a failed run, not a refusal)");
    }
    if (saw_run_end) {
        out.push_back("run_refused: " + shape +
                      " must have no run_end (run_refused is terminal and closes the sink)");
    }
    return out;
}

// The facts a run_refused record contributes, parsed identically by both
// checkers. `saw` is the only input to the shared rule that lives outside this
// struct (the other two are the run_start / phase_summary / run_end flags each
// checker already tracks).
struct ls_refusal_facts {
    bool             saw = false;
    bool             reason_ok = false;
    std::string      reason, detail, cap;
    long long        exit_status = -1, bytes_needed = -1, bytes_allowed = -1;
};

// Consume a run_refused line, if this is one. Returns true when it consumed the
// line, so the callers' event chains stay flat -- and false for every other
// event, which is why the event name must be checked here rather than by the
// caller's branch order.
static bool ls_parse_run_refused(const std::string & event, const std::string & line, ls_refusal_facts & f) {
    if (event != "run_refused") {
        return false;
    }
    f.saw = true;
    f.reason = json::value(line, "reason");
    f.detail = json::value(line, "detail");
    f.cap    = json::value(line, "cap_source");
    bool found = false;
    f.exit_status   = json::integer(line, "exit_status", &found);
    f.bytes_needed  = json::integer(line, "bytes_needed", &found);
    f.bytes_allowed = json::integer(line, "bytes_allowed", &found);
    f.reason_ok = ls_run_refusal_reason_ok(f.reason);
    return true;
}

// The refusal OK report, shared by both checkers (only the prefix and whether
// the event count is shown differ; pass n_events < 0 to omit it).
static void ls_print_refusal_ok(const char * tool, const ls_refusal_facts & f, bool saw_run_start,
                                long long n_events) {
    if (n_events >= 0) {
        fprintf(stderr, "%s: OK (REFUSAL TRACE: %lld events, reason=%s, exit_status=%lld, %lld/%lld B, cap_source=%s)\n",
            tool, n_events, f.reason.c_str(), f.exit_status, f.bytes_needed, f.bytes_allowed, f.cap.c_str());
    } else {
        fprintf(stderr, "%s: OK (REFUSAL TRACE: reason=%s, exit_status=%lld, %lld/%lld B, cap_source=%s)\n",
            tool, f.reason.c_str(), f.exit_status, f.bytes_needed, f.bytes_allowed, f.cap.c_str());
    }
    fprintf(stderr, "  refusal detail: %s\n", f.detail.c_str());
    fprintf(stderr, "  shape: %s\n", saw_run_start
        ? "ABORTED RUN - run_start was written, no timed phase completed, no run_end"
        : "PRE-WORK - no run_start, no timed phase ran, no run_end");
}

static int check_telemetry(const std::string & path) {
    FILE * f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        fprintf(stderr, "check-telemetry: cannot open %s\n", path.c_str());
        return 2;
    }

    char buf[65536];
    int line_no = 0;
    int n_slots = -1;
    int n_layer = -1;
    bool have_run_start = false;
    bool have_phase_summary = false;
    bool handoff_seen = false;         // T-Q38HG-02: track handoff events (reset per phase)
    bool handoff_ever = false;         // T-GPLR-12: handoff events ever seen (never reset)
    bool ho_mode_valid = false;        // T-GPLR-12: run_start carried a valid handoff_mode
    bool handoff_active_expected = false; // T-GPLR-12: mode on/auto-on
    bool have_run_end = false;
    bool off_audit_trace = false;      // T-GPLR-05: policy-off audit record schema
    // T-GPLR-13: moe-row-set trace facts (run_start marker) + per-slot row
    // coverage for the generation-lifecycle rule (schema section 4.5: reuse
    // requires a generation bump; coverage resets when the owner changes).
    bool rowset_mode = false;
    bool rowset_ever = false;          // any rowset_* event seen (default-off immunity)
    long long rowset_n_expert = -1, rowset_top_k = -1, rowset_window_tokens = 1;
    struct rowset_slot_state {
        long long gen = -1;          // generation the coverage was counted for
        long long owner = -1;        // last layer staged on this slot
        long long resident = 0;
    };
    std::unordered_map<int, rowset_slot_state> rowset_slots;
    // T-GPLR-19: >0 when rowset_init declares per-layer retention
    long long rowset_retain_k = -1;
    // T-GPLR-58: the per-layer retention cache model (rows, recency, and the staged
    // speculative fills of the `spec` events that precede each rowset_fetch).
    ls_retention_model rowset_retention;
    ls_pool_validator pool_validator;
    long long rowset_rounds = 0;
    // T-GPLR-62 S2 / A9: the byte-level staged-window recon. Declared at
    // rowset_recon_begin, evidenced per round by rowset_recon. Two failure
    // modes are enforced, not merely reported: any mismatch, and "declared but
    // no evidence at all" (which is what a run that silently skipped the check
    // would look like).
    bool     a9_declared = false;
    long long a9_rounds_seen = 0;
    long long a9_windows = 0, a9_members = 0, a9_compared = 0, a9_mismatches = 0;
    // T-GPLR-19: byte_account's row_cache figure, held until rowset_init has
    // been read so the record can be tied to the policy it claims.
    bool ba_row_cache_seen = false;
    long long ba_row_cache = 0;
    // T-GPLR-22: when a run_refused is present the trace is a REFUSAL, not a
    // run; ls_refusal_trace_errors() owns every consequence.
    ls_refusal_facts refusal;
    int errors = 0;

    struct slot_state {
        long long gen = 0;           // last copy_submit generation
        long long pending_gen = 0;   // copy_submit seen, copy_complete not yet
        long long pending_layer = -1;
        long long last_copy_layer = -1;
    };
    std::unordered_map<int, slot_state> slots;

    // layer-major (SDD section 13): per-layer ascending chunk order and
    // non-aliasing act-slab binds
    std::string schedule;
    std::unordered_map<int, long long> lm_next_chunk;      // layer -> expected next chunk
    std::unordered_map<int, long long> lm_last_chunk;       // layer -> last sampled chunk
    std::unordered_map<int, std::unordered_set<long long>> lm_seen_chunks; // layer -> chunk ids seen (stride-1 completeness, T-Q38CAP-01)
    int lm_chunk_stride = 1;                                // absent in legacy traces
    std::map<long long, long long> lm_act_off_A, lm_act_off_B; // slab -> off -> n_bytes (aliasing proof)
    // T-Q38CTX-01: immutable per-run chunk manifest (absent in legacy traces)
    int    lm_pp_n_chunks = -1;                  // -1 = absent
    std::vector<std::pair<long long, long long>> lm_manifest; // (token_start, n_tokens)
    std::unordered_map<long long, int> lm_act_binds;          // chunk -> binds seen (must end at 2)

    // per-phase summary data for the telemetry printout (tok/s, transfer,
    // wait, compute, evictions, graph builds) after the invariant check
    struct phase_data {
        std::string name;
        long long tokens = 0, ubatches = 0, wall_us = 0, layers_run = 0, evictions = 0, graph_builds = 0, bytes = 0;
        double transfer_us = 0.0, wait_us = 0.0, compute_us = 0.0;
    };
    std::vector<phase_data> phases;
    long long peak_vram = -1;

    auto err = [&](const char * msg) {
        fprintf(stderr, "check-telemetry: line %d: %s\n", line_no, msg);
        errors++;
    };

    while (fgets(buf, sizeof(buf), f) != nullptr) {
        line_no++;
        const std::string line(buf);
        if (line.empty() || line[0] != '{') {
            continue;
        }

        const std::string event = json::value(line, "event");
        if (event == "run_start") {
            if (have_run_start) {
                err("duplicate run_start");
            }
            // T-GPLR-05: `--residency off` emits an audit record (policy
            // "off") with run metadata only; the streaming-scheduler fields
            // (n_slots/n_layer/schedule) are absent by design. Fork the
            // validation per trace class instead of erroring on the schema.
            const bool off_audit = ls_line_is_off_audit(line);
            bool found = false;
            n_slots = (int) json::integer(line, "n_slots", &found);
            if (!off_audit && (!found || n_slots <= 0)) {
                err("run_start missing valid n_slots");
            }
            n_layer = (int) json::integer(line, "n_layer", &found);
            if (!off_audit && (!found || n_layer <= 0)) {
                err("run_start missing valid n_layer");
            }
            for (const auto & pe : pool_validator.run_start_errors(line)) {
                err(pe.c_str());
            }
            schedule = json::value(line, "schedule");
            if (off_audit) {
                off_audit_trace = true;
                ls_check_off_run_start(line, line_no, "check-telemetry", errors);
            }
            const long long chunk_stride = json::integer(line, "telemetry_chunk_stride", &found);
            if (found && chunk_stride <= 0) {
                err("run_start has invalid telemetry_chunk_stride");
            } else if (found) {
                lm_chunk_stride = (int) chunk_stride;
            }
            // T-GPLR-48: the RESOLVED per-pass-rebuild decision and whether it
            // came from the flag or from the KV-conditional default. Both keys
            // are validated when present, so a run can be classified as
            // explicit-vs-defaulted from the trace alone (AC5); traces from
            // before this key existed stay admissible.
            const long long pbp = json::integer(line, "pp_prebuild_per_pass", &found);
            if (found && (pbp < 0 || pbp > 1)) {
                err("run_start has invalid pp_prebuild_per_pass");
            }
            const std::string pbp_src = json::value(line, "pp_prebuild_per_pass_source");
            if (!pbp_src.empty() && pbp_src != "explicit" && pbp_src != "default") {
                err("run_start has invalid pp_prebuild_per_pass_source");
            }
            // T-GPLR-52 A3/AC4: the engine-side staging source and its
            // cross-check vs the declared copy_source. Optional for legacy
            // traces; when present, a 0 match is FATAL - a run must never
            // carry a staging_source that disagrees with its label (the F5
            // concern, now enforceable from the trace alone).
            const std::string stage_src = json::value(line, "staging_source");
            if (!stage_src.empty()) {
                if (stage_src != "direct" && stage_src != "pinned" &&
                    stage_src != "mmap-registered" && stage_src != "mmap-registered-churn") {
                    err("run_start has invalid staging_source");
                }
                const long long stage_match = json::integer(line, "staging_source_match", &found);
                if (!found) {
                    err("run_start has staging_source without staging_source_match");
                } else if (stage_match != 1) {
                    err("run_start staging_source does not match the declared copy_source");
                }
            }
            // T-GPLR-37: provenance of the resolved copy source; same contract
            // as the rebuild key above (optional for legacy traces, validated
            // when present).
            const std::string cs_src = json::value(line, "copy_source_source");
            if (!cs_src.empty() && cs_src != "explicit" && cs_src != "default") {
                err("run_start has invalid copy_source_source");
            }
            // T-GPLR-39: the resolved route-DIAGNOSTIC switch. Optional for
            // legacy traces; when present it must be 0/1 with known provenance,
            // and its VALUE decides whether route events are legal in this
            // trace (enforced at run_end below).
            bool rt_valid = false;
            bool rt_must_be_absent = false;
            const long long rt = json::integer(line, "route_telemetry", &found);
            if (found) {
                if (rt < 0 || rt > 1) {
                    err("run_start has invalid route_telemetry");
                } else {
                    rt_valid = true;
                    rt_must_be_absent = (rt == 0);
                }
            }
            const std::string rt_src = json::value(line, "route_telemetry_source");
            if (!rt_src.empty() && rt_src != "explicit" && rt_src != "default") {
                err("run_start has invalid route_telemetry_source");
            }
            // T-GPLR-12: the auditable handoff decision. Optional for legacy
            // traces; when present it must carry a known mode and is
            // cross-checked against the handoff events (an off/auto-off run
            // must not emit handoff events; on/auto-on must emit them).
            ho_mode_valid = false;
            handoff_active_expected = false;
            if (line.find("\"handoff_mode\":") != std::string::npos) {
                if (off_audit_trace) {
                    err("off-audit run_start carries handoff_mode (off runs have no handoff decision)");
                } else {
                    const std::string hm = json::value(line, "handoff_mode");
                    if (hm != "on" && hm != "off" && hm != "auto-on" && hm != "auto-off") {
                        err("run_start has invalid handoff_mode");
                    } else {
                        ho_mode_valid = true;
                        handoff_active_expected = (hm == "on" || hm == "auto-on");
                    }
                }
            }
            // T-GPLR-13: the moe-row-set opt-in marker. Its ABSENCE is the
            // default-off immunity assertion (schema section 4.9) enforced at
            // run_end: no rowset_* events may appear without it.
            if (line.find("\"rowset\":\"on\"") != std::string::npos) {
                rowset_mode = true;
            }
            // T-Q38CTX-01: parse the chunk manifest ["start:width", ...]
            if (!off_audit && schedule == "layer-major") {
                bool f_ppn = false;
                const long long pp_tokens_m = json::integer(line, "pp_tokens", &found);
                lm_pp_n_chunks = (int) json::integer(line, "pp_n_chunks", &f_ppn);
                if (f_ppn) {
                    if (!found || lm_pp_n_chunks <= 0 || pp_tokens_m <= 0) {
                        err("run_start has an invalid chunk manifest");
                    } else {
                        const std::string key = "\"pp_chunks\":";
                        size_t b = line.find(key);
                        size_t e = std::string::npos;
                        if (b != std::string::npos) {
                            b += key.size();
                            while (b < line.size() && (line[b] == ' ' || line[b] == '\t')) {
                                b++;
                            }
                            if (b >= line.size() || line[b] != '[') {
                                b = std::string::npos;
                            } else {
                                e = line.find(']', b);
                            }
                        }
                        if (b == std::string::npos || e == std::string::npos) {
                            err("run_start has pp_n_chunks but no pp_chunks list");
                        } else {
                            std::string entry;
                            for (size_t i = b + 1; ; ++i) {  // b points at '['
                                if (i == e || line[i] == ',') {
                                    if (!entry.empty()) {
                                        // entries are JSON strings "start:width"; strip quotes
                                        auto strip = [](std::string s) {
                                            while (!s.empty() && (s.front() == '"' || s.front() == ' ')) s.erase(s.begin());
                                            while (!s.empty() && (s.back() == '"' || s.back() == ' ')) s.pop_back();
                                            return s;
                                        };
                                        entry = strip(entry);
                                        const size_t cpos = entry.find(':');
                                        if (cpos == std::string::npos || cpos == 0) {
                                            err("malformed pp_chunks entry");
                                        } else {
                                            lm_manifest.push_back({ atoll(entry.substr(0, cpos).c_str()),
                                                                   atoll(entry.substr(cpos + 1).c_str()) });
                                        }
                                        entry.clear();
                                    }
                                } else {
                                    entry += line[i];
                                }
                                if (i == e) {
                                    break;
                                }
                            }
                            if ((int) lm_manifest.size() != lm_pp_n_chunks) {
                                err("pp_chunks list length != pp_n_chunks");
                            } else {
                                long long cursor = 0;
                                bool contig = true;
                                for (const auto & mk : lm_manifest) {
                                    if (mk.first != cursor || mk.second < 1) {
                                        contig = false;
                                        break;
                                    }
                                    cursor += mk.second;
                                }
                                if (!contig || cursor != pp_tokens_m) {
                                    err("pp_chunks manifest is not a contiguous coverage of pp_tokens");
                                }
                            }
                        }
                    }
                }
            }
            have_run_start = true;
        } else if (event == "pp_pass") {
            // T-GPLR-28: --pp-repeat emits this marker at the start of pass
            // 2..N (after the KV clear + stats reset). The checker treats it
            // as a new prefill: per-layer ascending-chunk expectations reset.
            lm_next_chunk.clear();
            lm_last_chunk.clear();
        } else if (event == "error") {
            err("run produced an error event");
        } else if (event == "handoff") {
            // T-Q38HG-02: handoff invalidates TG graphs, so graph_builds > 0
            // is expected on the next TG phase.
            handoff_seen = true;
            handoff_ever = true;
            // T-GPLR-12: a run whose run_start declared off/auto-off must
            // never emit handoff events.
            if (ho_mode_valid && !handoff_active_expected) {
                err("handoff event in a run whose run_start declared handoff_mode off/auto-off");
            }
            // T-GPLR-10: validate the migrate event's migration list. The
            // release (peak-order slot/slab release before buf_handoff) and
            // teardown actions carry no layer fields.
            const std::string action = json::value(line, "action");
            if (action != "migrate" && action != "release" && action != "teardown") {
                err("handoff event with unknown action");
            }
            if (action == "migrate") {
                bool f = false;
                const long long layers = json::integer(line, "layers", &f);
                const long long total_bytes = json::integer(line, "total_bytes", &f);
                const bool attn_only = json::value(line, "attention_only") == "true";
                const bool mtp_flag = json::value(line, "mtp") == "true";
                if (!f) {
                    err("handoff migrate event missing layers/total_bytes");
                } else {
                    if (total_bytes <= 0) {
                        err("handoff migrate total_bytes <= 0");
                    }
                    const long long expect_full = n_layer + (mtp_flag ? 1 : 0);
                    // T-GPLR-16: partial migration — the event must carry
                    // partial=true with a positive split below n_layer, and
                    // the layer count must equal n_layer - partial_from.
                    // Checked BEFORE the full-handoff count rule (which does
                    // not apply to a partial migration list) and via raw-
                    // JSON search (not json_str) to avoid the key-prefix
                    // collision: "partial" is a prefix of "partial_from".
                    const bool is_partial = line.find("\"partial\":true") != std::string::npos;
                    if (is_partial) {
                        bool fp = false;
                        const long long pf_ = json::integer(line, "partial_from", &fp);
                        if (!fp || pf_ <= 0 || pf_ >= n_layer) {
                            err("partial handoff migrate has invalid partial_from");
                        } else if (layers != n_layer - pf_) {
                            err("partial handoff layers != n_layer - partial_from");
                        }
                    }
                    if (!attn_only && !is_partial && layers != expect_full) {
                        err("full handoff layers != n_layer (+1 when MTP)");
                    }
                    if (attn_only && (layers <= 0 || layers >= expect_full)) {
                        err("attention-only handoff layer count out of range");
                    }
                    // T-GPLR-10: when present, the free-VRAM reading taken
                    // right after the handoff allocation (true post-handoff
                    // peak = total - free_vram) must be sane; absent on
                    // legacy traces is accepted (field added this task)
                    bool ff = false;
                    const long long free_vram_ho = json::integer(line, "free_vram", &ff);
                    if (ff && free_vram_ho <= 0) {
                        err("handoff migrate event has invalid free_vram");
                    }
                    // in-process self-check: exact match against the
                    // migration list captured at begin_handoff time
                    if (g_check_handoff_layers > 0 && layers != g_check_handoff_layers) {
                        err("handoff migrate layers != runner-captured migration list");
                    }
                    if (g_check_handoff_bytes > 0 && total_bytes != g_check_handoff_bytes) {
                        err("handoff migrate total_bytes != runner-captured migration list");
                    }
                }
            }
        } else if (event == "byte_account") {
            // T-GPLR-04: additive byte-class sums must be internally consistent
            // and the bounded window must sit below observed free VRAM.
            bool f = false;
            const long long full = json::integer(line, "full_model_plus_runtime", &f);
            const long long bounded = json::integer(line, "bounded_window", &f);
            const long long st_ = json::integer(line, "static", &f);
            const long long pl = json::integer(line, "per_layer", &f);
            const long long sw = json::integer(line, "slot_window", &f);
            const long long kvb = json::integer(line, "kv", &f);
            const long long stt = json::integer(line, "state", &f);
            const long long free_b = json::integer(line, "free_vram", &f);
            if (!f) {
                err("byte_account missing fields");
            } else if (st_ + pl + sw + kvb + stt != bounded) {
                err("byte_account: static + per_layer + slot_window + kv + state != bounded_window");
            }
            if (full <= bounded) {
                err("byte_account: full_model_plus_runtime <= bounded_window (not an out-of-core proof)");
            }
            if (free_b > 0 && bounded >= free_b) {
                err("byte_account: bounded_window >= observed free VRAM");
            }
            // T-GPLR-19: the row cache and the TOTAL residency cap. Shared
            // rule (one owner, ls_byte_account_errors); the policy tie is
            // deferred to the end of the trace because this event precedes
            // rowset_init. Fields are absent on pre-field traces, accepted.
            {
                bool fb = false, fc = false, fp = false, fcap = false;
                const long long budget_mib = json::integer(line, "residency_budget_mib", &fb);
                const long long row_cache_b = json::integer(line, "row_cache", &fc);
                const long long pool_b = json::integer(line, "prediction_pool", &fp);
                const long long cap_b = json::integer(line, "residency_cap", &fcap);
                ba_row_cache_seen = fc;
                ba_row_cache = row_cache_b;
                for (const auto & be : ls_byte_account_errors(bounded, budget_mib, fc, row_cache_b,
                                                               fp, pool_b, fcap, cap_b)) {
                    err(be.c_str());
                }
                for (const auto & be : pool_validator.byte_account_errors(line)) {
                    err(be.c_str());
                }
            }
        } else if (event == "phase_summary") {
            if (off_audit_trace) {
                err("phase_summary in an off-mode audit trace (off runs emit run_start + run_end only)");
                continue;
            }
            bool found = false;
            const long long graph_builds = json::integer(line, "graph_builds", &found);
            // T-Q38HG-02: graph_builds > 0 is acceptable after a handoff event.
            if (found && graph_builds != 0 && !handoff_seen) {
                err("phase_summary.graph_builds != 0 (graph construction in timed region)");
            }
            const bool after_handoff = handoff_seen; // T-GPLR-10: hybrid-TG bytes rule below
            handoff_seen = false; // reset per phase
            have_phase_summary = true;

            phase_data pd;
            pd.name = json::value(line, "phase");
            bool f = false;
            pd.tokens       = json::integer(line, "tokens", &f);
            pd.ubatches     = json::integer(line, "ubatches", &f);
            pd.wall_us      = json::integer(line, "wall_us", &f);
            pd.transfer_us  = (double) json::integer(line, "total_transfer_us", &f);
            pd.wait_us      = (double) json::integer(line, "total_wait_us", &f);
            pd.compute_us   = (double) json::integer(line, "total_compute_us", &f);
            pd.layers_run   = json::integer(line, "layers_run", &f);
            pd.evictions    = json::integer(line, "evictions", &f);
            pd.graph_builds = json::integer(line, "graph_builds", &f);
            pd.bytes        = json::integer(line, "bytes_transferred", &f);
            // T-GPLR-10: the phase right after a handoff migrate is the hybrid
            // TG -- stock llama_decode with the banks already resident, so no
            // bank copies exist by construction: bytes_transferred must be ~0.
            // A nonzero value would mean a paged-TG fallback or a copy leak.
            if (after_handoff && pd.name == "tg" && pd.bytes != 0) {
                err("post-handoff TG phase transferred bytes (hybrid TG runs stock llama_decode; expected 0 bank bytes)");
            }
            phases.push_back(pd);
        } else if (event == "spec_pool_init") {
            for (const auto & pe : pool_validator.init_errors(line)) {
                err(pe.c_str());
            }
        } else if (event == "spec_pool") {
            for (const auto & pe : pool_validator.event_errors(line)) {
                err(pe.c_str());
            }
        } else if (event == "rowset_init") {
            // T-GPLR-13: policy header. n_expert/top_k scope the ID range and
            // cardinality checks of every following rowset_fetch.
            rowset_ever = true;
            bool f = false;
            rowset_n_expert = json::integer(line, "n_expert", &f);
            pool_validator.set_n_expert(rowset_n_expert);
            if (!f || rowset_n_expert <= 0) {
                err("rowset_init missing valid n_expert");
            }
            rowset_top_k = json::integer(line, "top_k", &f);
            if (!f || rowset_top_k <= 0) {
                err("rowset_init missing valid top_k");
            }
            rowset_window_tokens = json::integer(line, "window_tokens", &f);
            if (!f || rowset_window_tokens < 1) {
                err("rowset_init missing valid window_tokens");
            }
            // T-GPLR-19: 0/absent = reactive slot-keyed policy (unchanged);
            // >0 declares per-layer retention at that K
            rowset_retain_k = json::integer(line, "retain_k", &f);
            if (f && rowset_retain_k < 0) {
                err("rowset_init retain_k must not be negative");
            }
        } else if (event == "rowset_fetch") {
            rowset_ever = true;
            if (!rowset_mode) {
                err("rowset_fetch in a run whose run_start carries no rowset marker");
            }
            bool f = false, rf = false, bf = false;
            const long long layer = json::integer(line, "layer", &f);
            const long long slot  = json::integer(line, "slot", &f);
            const long long gen   = json::integer(line, "generation", &f);
            const long long ru    = json::integer(line, "route_union", &f);
            const long long nf    = json::integer(line, "n_rows_fetched", &f);
            const long long nr    = json::integer(line, "n_rows_resident", &f);
            const long long bs    = json::integer(line, "bytes_staged", &f);
            const long long ub    = json::integer(line, "ubatch", &rf);
            const long long nret  = json::integer(line, "n_rows_retained", &rf);
            const long long nrtd  = json::integer(line, "n_rows_retired", &rf);
            const long long bank  = json::integer(line, "bytes_full_bank", &bf);
            const long long phase = json::integer(line, "phase", nullptr);
            if (!f || !rf || !bf || layer < 0 || slot < 0 || ru < 0 || nf < 0 || nr < 0 ||
                ub < 1 || nret < 0 || nrtd < 0 || bank < 0) {
                err("rowset_fetch missing required fields");
                continue;
            }
            const std::vector<long long> pool_ids = json::integer_list(line, "pool_hit_ids", nullptr);
            const bool pool_enabled = pool_validator.enabled();
            const long long pool_rows = pool_enabled ? (long long) pool_ids.size() : 0;
            for (const auto & pe : pool_validator.fetch_errors(line, layer, gen, ub, ru, nf, nr,
                                                  nret, bs, bank)) {
                err(pe.c_str());
            }
            // The union is top-k for W=1, and at most W*top-k for an MTP verify
            // window (duplicates across target rows can make it smaller).
            if (rowset_top_k >= 0 && rowset_window_tokens >= 1) {
                const long long max_union = std::min(rowset_n_expert,
                                                      rowset_window_tokens * rowset_top_k);
                const bool valid_union = rowset_window_tokens == 1
                    ? ru == rowset_top_k
                    : ru >= 1 && ru <= max_union;
                if (!valid_union) {
                    err("rowset_fetch route_union outside declared window*top_k bound");
                }
            }
            // accounting identity (schema 4.3): n_rows_resident is real-cache only.
            if (ru != nf + nr + pool_rows) {
                err("rowset_fetch route_union != real fetched + real resident + pool hits");
            }
            if ((bs > 0) != (nf > 0)) {
                err("rowset_fetch bytes_staged/n_rows_fetched mismatch");
            }
            // hash integrity (schema 4.6): FNV-1a-64 over union_ids bytes
            {
                const std::string ids = json::value(line, "union_ids");
                const std::string hash_s = json::value(line, "hash");
                if (ids.empty() || hash_s.empty()) {
                    err("rowset_fetch missing union_ids/hash");
                } else {
                    uint64_t h = 1469598103934665603ull;
                    for (unsigned char c : ids) {
                        h ^= (uint64_t) c;
                        h *= 1099511628211ull;
                    }
                    // strtoull: the U64 hash exceeds LLONG_MAX (atoll saturates)
                    if (h != strtoull(hash_s.c_str(), nullptr, 10)) {
                        err("rowset_fetch hash mismatch over union_ids");
                    }
                }
            }
            // T-GPLR-19: retention redefines what "resident" means, so the
            // slot-keyed lifecycle rule is switched off when rowset_init
            // declared retention and the per-layer conservation rule runs
            // instead. Default-off is byte-for-byte the old behaviour.
            if (rowset_retain_k > 0) {
                for (const auto & e : ls_retention_errors(rowset_retention,
                         layer, gen, phase, ru, nr, pool_ids,
                         json::value(line, "union_ids"), nret, nrtd, rowset_retain_k,
                         rowset_n_expert, rowset_top_k)) {
                    err(e.c_str());
                }
            } else {
            // generation lifecycle (schema 4.5): coverage lives per slot;
            // reuse requires the same generation, an owner change resets it.
            {
                auto & rs = rowset_slots[(int) slot];
                if (rs.gen != gen || rs.owner != layer) {
                    if (nr != 0) {
                        err("rowset_fetch first window per (slot, generation) claims resident rows");
                    }
                    if (nf != ru) {
                        err("rowset_fetch first window must fetch the full union");
                    }
                    rs.gen = gen;
                    rs.owner = layer;
                    rs.resident = ru;
                } else {
                    if (nr != rs.resident) {
                        err("rowset_fetch n_rows_resident != tracked slot coverage");
                    }
                    rs.resident += nf;
                }
                if (ru > rowset_n_expert) {
                    err("rowset_fetch route_union exceeds n_expert");
                }
            }
            }
        } else if (event == "rowset_round") {
            // byte-account aggregate (schema 4.8): the acceptance gate reads
            // this ratio, never a claimed speedup.
            rowset_ever = true;
            bool f = false;
            const long long bs = json::integer(line, "bytes_staged", &f);
            const long long bb = json::integer(line, "bytes_full_bank", &f);
            if (!f || bs < 0 || bb <= 0) {
                err("rowset_round missing required fields");
            } else {
                fprintf(stderr, "rowset_round %lld: staged %lld / full-bank %lld bytes (%.2f%%)\n",
                        rowset_rounds, bs, bb, 100.0 * (double) bs / (double) bb);
            }
            rowset_rounds++;
        } else if (event == "rowset_refused") {
            // structured loud refusal (schema 4: reason enum) -- always
            // followed by the loud failed run_end path.
            rowset_ever = true;
            const std::string reason = json::value(line, "reason");
            err((std::string("rowset_refused: ") +
                 (reason.empty() ? "<no reason>" : reason) + " (" +
                 json::value(line, "detail") + ")").c_str());
        } else if (event == "spec" && rowset_retain_k > 0) {
            // T-GPLR-58: the speculative fills this stage issued. They are STAGED
            // here and replayed by ls_retention_errors after the rowset_fetch of the
            // same stage, which is the order the runtime applies them in.
            for (const auto & e : ls_spec_errors(rowset_retention, line)) {
                err(e.c_str());
            }
        } else if (ls_parse_run_refused(event, line, refusal)) {
        } else if (event == "run_end") {
            bool found = false;
            const long long exit_status = json::integer(line, "exit_status", &found);
            if (found && exit_status != 0) {
                err("run_end.exit_status != 0");
            }
            have_run_end = true;
            // T-GPLR-13 default-off immunity: rowset_* events require the
            // run_start "rowset":"on" marker.
            if (!rowset_mode && rowset_ever) {
                err("rowset events in a run whose run_start carries no rowset marker");
            }
            peak_vram = json::integer(line, "peak_vram_bytes", &found);
            // T-GPLR-12: a run whose run_start declared on/auto-on must have
            // emitted handoff events by run_end. Use handoff_ever, not
            // handoff_seen (the latter is reset by each phase_summary).
            if (ho_mode_valid && handoff_active_expected && !handoff_ever) {
                err("run_start declared handoff_mode on/auto-on but no handoff events followed");
            }
        } else if (event == "slot_request") {
            bool f_layer = false, f_slot = false;
            const long long layer = json::integer(line, "layer", &f_layer);
            const long long slot  = json::integer(line, "slot",  &f_slot);

            if (off_audit_trace) {
                ls_err_off_streaming_event(line_no, "check-telemetry", errors);
                continue;
            }
            if (n_slots <= 0) {
                err("layer event before run_start");
                continue;
            }
            if (!f_layer || !f_slot) {
                err("slot_request missing layer/slot");
                continue;
            }
            if (slot != layer % n_slots) {
                err("slot_request slot != layer % n_slots");
            }
            if (slot < 0 || slot >= n_slots) {
                err("slot_request slot out of range");
            }
        } else if (event == "rowset_recon_begin") {
            // T-GPLR-62 S2 / A9: the mode was declared. The evidence is checked
            // at the end of the trace, because a refusal (the correct outcome of
            // a mismatch) legitimately produces no per-round evidence.
            a9_declared = true;
        } else if (event == "rowset_recon") {
            bool f_w = false, f_m = false, f_cb = false, f_mm = false;
            const long long w  = json::integer(line, "windows", &f_w);
            const long long m  = json::integer(line, "members", &f_m);
            const long long cb = json::integer(line, "compared_bytes", &f_cb);
            const long long mm = json::integer(line, "mismatches", &f_mm);
            if (!f_w || !f_m || !f_cb || !f_mm) {
                err("rowset_recon missing windows/members/compared_bytes/mismatches");
            } else {
                a9_rounds_seen += 1;
                a9_windows     += w;
                a9_members     += m;
                a9_compared    += cb;
                a9_mismatches  += mm;
                if (mm > 0) {
                    err(("A9: staged slot bytes do NOT match the checkpoint for " +
                         std::to_string(mm) + " member(s) -- the window holds the WRONG expert").c_str());
                }
                if (w > 0 && cb <= 0) {
                    err("A9: rowset_recon claims windows but compared zero bytes");
                }
            }
        } else if (event == "copy_submit" || event == "copy_complete" ||
                   event == "layer_ready_wait" || event == "layer_compute") {
            bool f_layer = false, f_slot = false, f_gen = false;
            const long long layer = json::integer(line, "layer", &f_layer);
            const long long slot  = json::integer(line, "slot",  &f_slot);
            const long long gen   = json::integer(line, "generation", &f_gen);

            if (off_audit_trace) {
                ls_err_off_streaming_event(line_no, "check-telemetry", errors);
                continue;
            }
            if (n_slots <= 0) {
                err("layer event before run_start");
                continue;
            }
            if (!f_layer || !f_slot || !f_gen) {
                err("layer event missing layer/slot/generation");
                continue;
            }
            if (slot != layer % n_slots) {
                err("slot != layer % n_slots");
            }
            if (slot < 0 || slot >= n_slots) {
                err("slot out of range");
            }

            auto & st = slots[(int) slot];

            if (event == "copy_submit") {
                // generation must strictly increase per slot
                if (gen <= st.gen) {
                    err("copy_submit generation did not strictly increase");
                }
                // deterministic policy: slot s serves layers s + k*n_slots in
                // order; slots may own different counts when n_slots does not
                // divide n_layer evenly.
                const long long per_slot = (n_layer - 1 - slot) / n_slots + 1;
                const long long expect_layer = slot + ((gen - 1) % per_slot) * n_slots;
                if (layer != expect_layer) {
                    err("copy_submit layer violates deterministic slot policy");
                }
                st.gen = gen;
                st.pending_gen = gen;
                st.pending_layer = layer;
            } else if (event == "copy_complete") {
                if (st.pending_gen != gen || st.pending_layer != layer) {
                    err("copy_complete has no matching copy_submit");
                }
                const std::string host_submit = json::value(line, "host_submit_us");
                const std::string ready_latency = json::value(line, "ready_latency_us");
                if (!host_submit.empty() || !ready_latency.empty()) {
                    if (host_submit.empty() || ready_latency.empty()) {
                        err("copy_complete has partial timing split");
                    } else {
                        const double submit_us = atof(host_submit.c_str());
                        const double ready_us = atof(ready_latency.c_str());
                        const double total_us = atof(json::value(line, "transfer_us").c_str());
                        if (submit_us < 0.0 || ready_us < 0.0 || total_us < 0.0 ||
                            std::abs((submit_us + ready_us) - total_us) > 0.2) {
                            err("copy_complete timing split is invalid");
                        }
                    }
                }
                st.pending_gen = 0;
                st.pending_layer = -1;
                st.last_copy_layer = layer;
            } else if (event == "layer_ready_wait") {
                if (st.last_copy_layer != layer) {
                    err("layer_ready_wait before matching copy_complete");
                }
            } else if (event == "layer_compute") {
                // T-GPLR-13: the row-set executor stages rows directly into
                // CUDA-resident slot buffers under the copy-event lifecycle
                // (one rowset_fetch per layer window) and emits NO
                // copy_submit/copy_complete pair, so the copy-completed rule
                // does not apply in row-set mode.
                if (!rowset_mode && st.last_copy_layer != layer) {
                    err("layer_compute before matching copy_complete");
                }
                // T-GPLR-13: TG row-set layer_compute carries the TG window
                // ordinal (ubatch), not a PP chunk id; the phase == 0 gate
                // below keeps the PP chunk rules off the TG events.
                bool f_phase = false;
                const long long phase = json::integer(line, "phase", &f_phase);
                if (schedule == "layer-major" && f_phase && phase == 0) {
                    bool f_chunk = false;
                    const long long chunk = json::integer(line, "chunk", &f_chunk);
                    if (!f_chunk) {
                        err("layer-major layer_compute missing chunk");
                    } else {
                        if (lm_pp_n_chunks > 0 && (chunk < 0 || chunk >= lm_pp_n_chunks)) {
                            err("layer_compute chunk is outside the pp_chunks manifest");
                        }
                        const bool first_chunk = !lm_next_chunk.count(layer);
                        const long long expect = first_chunk ? 0 : lm_next_chunk[layer];
                        const long long previous = first_chunk ? -1 : lm_last_chunk[layer];
                        if ((first_chunk && chunk != 0) ||
                            (!first_chunk && chunk != expect && (chunk <= previous || chunk % lm_chunk_stride == 0))) {
                            err("layer-major chunk does not match telemetry stride");
                        }
                        lm_next_chunk[(int) layer] = chunk + lm_chunk_stride;
                        lm_last_chunk[(int) layer] = chunk;
                        lm_seen_chunks[(int) layer].insert(chunk);
                    }
                }
            }
        } else if (event == "act_bind") {
            // layer-major: act-slab binds must be non-aliasing per slab
            if (schedule == "layer-major") {
                bool f_off = false, f_bytes = false, f_slab = false;
                const long long off   = json::integer(line, "off_bytes", &f_off);
                const long long bytes = json::integer(line, "n_bytes",   &f_bytes);
                const long long slab  = json::integer(line, "slab",      &f_slab);
                if (!f_off || !f_bytes || !f_slab) {
                    err("act_bind missing off_bytes/n_bytes/slab");
                    continue;
                }
                if (slab != 0 && slab != 1) {
                    err("act_bind slab out of range");
                    continue;
                }
                auto & offmap = (slab == 0) ? lm_act_off_A : lm_act_off_B;
                for (const auto & kv : offmap) {
                    // [off, off+bytes) must not overlap an existing bind
                    if (off < kv.first + kv.second && kv.first < off + bytes) {
                        err("act_bind aliases an existing slab slice");
                    }
                }
                offmap[off] = bytes;
                // T-Q38CTX-01: with a chunk manifest, the bind's chunk id is
                // in range, its byte width is proportional to the chunk's real
                // width (512-wide chunks get 512x the bytes of a 1-token tail),
                // and each chunk gets exactly one bind per slab.
                if (lm_pp_n_chunks > 0) {
                    bool f_chunk = false;
                    const long long bchunk = json::integer(line, "chunk", &f_chunk);
                    if (!f_chunk || bchunk < 0 || bchunk >= lm_pp_n_chunks) {
                        err("act_bind chunk is outside the pp_chunks manifest");
                    } else {
                        const int n_binds = ++lm_act_binds[bchunk];
                        if (n_binds > 2) {
                            err("act_bind repeated for a chunk/slab pair");
                        }
                        // manifest entries are (start,width); the chunk id is the list index
                        const long long width = (bchunk < (long long) lm_manifest.size()) ? lm_manifest[bchunk].second : -1;
                        if (width > 0) {
                            // n_bytes / width must be constant across chunks of
                            // this slab (n_embd * 4); check against the first
                            // recorded bind of the run.
                            static long long per_token = -1;
                            if (per_token < 0) {
                                per_token = bytes / width;
                            }
                            if (bytes != per_token * width) {
                                err("act_bind width does not match the pp_chunks manifest");
                            }
                        }
                    }
                }
            }
        } else if (event == "slot_evict") {
            bool f_layer = false, f_slot = false, f_gen = false;
            const long long layer = json::integer(line, "layer", &f_layer);
            const long long slot  = json::integer(line, "slot",  &f_slot);
            const long long gen   = json::integer(line, "generation", &f_gen);
            if (!f_layer || !f_slot || !f_gen) {
                err("slot_evict missing layer/slot/generation");
                continue;
            }
            // evicted layer must be the prior resident: slot + ((gen-1) mod
            // per_slot) * n_slots (generation wraps each pass). per_slot uses
            // the SAME uneven-aware formula as the copy_submit rule above --
            // (n_layer - 1 - slot)/n_slots + 1 -- because slot ownership need
            // not divide evenly (the recorded 3-slot rung exposed this: the
            // old n_layer/n_slots form drifted after the first full wrap and
            // rejected 1,253 correct slot_evict records).
            const long long per_slot = (n_layer - 1 - slot) / n_slots + 1;
            const long long expect_layer = slot + ((gen - 1) % per_slot) * n_slots;
            if (layer != expect_layer) {
                err("slot_evict layer is not the prior resident layer");
            }
        }
    }
    fclose(f);

    if (schedule == "layer-major") {
        const auto pp = std::find_if(phases.begin(), phases.end(), [](const phase_data & phase) {
            return phase.name == "pp";
        });
        // T-GPLR-28: --pp-repeat emits one prefill per pass, so per-layer
        // chunk state is judged against the per-pass chunk manifest rather
        // than the aggregated ubatch count; the expected final chunk is the
        // last manifest width - 1 (legacy no-manifest traces keep the old
        // ubatches-1 rule).
        if (pp != phases.end() && pp->layers_run > 0 && pp->tokens > 0) {
            // chunk ids are 0..n_chunks-1 (the manifest entries are
            // {token_start, width} pairs, not chunk ids)
            const long long expect_last = lm_manifest.empty()
                ? pp->ubatches - 1
                : (long long) lm_pp_n_chunks - 1;
            for (const auto & entry : lm_last_chunk) {
                if (entry.second != expect_last) {
                    err("layer-major layer_compute missing final chunk");
                }
            }
        }
        // T-Q38CAP-01: at stride 1 every layer must have exactly the full
        // 0..n_chunks-1 chunk sequence (no holes); the ascending rule alone
        // would accept a skipped interior chunk
        if (lm_chunk_stride == 1 && lm_pp_n_chunks > 0) {
            // T-GPLR-28: --pp-repeat emits one prefill per pass, so the
            // per-layer seen-chunk multiset is judged per segment; the final
            // (sampled) chunk already passed the expected-last rule above.
            for (const auto & kv : lm_seen_chunks) {
                if ((int) kv.second.size() % lm_pp_n_chunks != 0) {
                    err(("layer " + std::to_string(kv.first) +
                         " has " + std::to_string(kv.second.size()) +
                         " layer_compute chunks at stride 1, not a multiple of the " +
                        std::to_string(lm_pp_n_chunks) + " per-pass count").c_str());
                }
            }
        }
        // T-Q38CTX-01: with a chunk manifest, every chunk must have been bound
        // to both slabs (the prebuild created one per-chunk set)
        // T-GPLR-28: with --pp-repeat each pass binds both slabs again, so
        // the per-chunk bind count is 2 x passes; require an even count of
        // at least one full pass.
        if (lm_pp_n_chunks > 0 && pp != phases.end() && pp->layers_run > 0) {
            for (int c = 0; c < lm_pp_n_chunks; ++c) {
                const auto it = lm_act_binds.find((long long) c);
                if (it == lm_act_binds.end() || it->second < 2 || it->second % 2 != 0) {
                    err(("pp_chunks manifest chunk " + std::to_string(c) + " was not fully bound (both slabs)").c_str());
                }
            }
        }
    }

    // T-GPLR-19: the deferred half of the byte_account rule -- the cache figure
    // must agree with the retention policy this trace declared.
    for (const auto & be : ls_byte_account_policy_errors(ba_row_cache_seen, ba_row_cache, rowset_retain_k)) {
        err(be.c_str());
    }

    // T-GPLR-62 S2 / A9: the deferred half. A trace that DECLARED the recon must
    // carry evidence, and must have compared a non-trivial number of bytes. A
    // refusal trace is exempt: a refusal is the correct outcome, and a mismatch
    // refuses BEFORE the round ever ends, so the absence of evidence there is the
    // gate working rather than the gate being skipped.
    if (a9_declared && !refusal.saw) {
        if (a9_rounds_seen == 0) {
            err("A9: rowset_recon_begin present but no rowset_recon evidence -- "
                "the byte-level check never ran");
        } else if (a9_mismatches == 0 && a9_compared <= 0) {
            err("A9: recon evidence present but zero bytes were compared");
        }
    }

    // T-GPLR-58: the deferred form of the `spec` rule -- a trace whose fills cannot be
    // replayed fails ONCE, with the cause named, instead of once per later fetch.
    for (const auto & se : ls_spec_replay_errors(rowset_retention)) {
        err(se.c_str());
    }
    for (const auto & pe : pool_validator.final_errors(refusal.saw)) {
        err(pe.c_str());
    }

    // T-GPLR-22: the three rules below describe a run that started; a refusal
    // trace may not have started one, so they are suppressed for it and replaced
    // by the shared refusal rule (one owner, ls_refusal_trace_errors).
    for (const auto & re : ls_refusal_trace_errors(refusal.saw, refusal.reason_ok,
                                                  have_run_start, have_phase_summary, have_run_end)) {
        err(re.c_str());
    }
    if (!have_run_start && !refusal.saw) {
        fprintf(stderr, "check-telemetry: missing run_start\n");
        errors++;
    }
    if (!have_phase_summary && !off_audit_trace && !refusal.saw) {
        fprintf(stderr, "check-telemetry: missing phase_summary\n");
        errors++;
    }
    if (!have_run_end && !refusal.saw) {
        fprintf(stderr, "check-telemetry: missing run_end\n");
        errors++;
    }

    if (errors == 0) {
        if (refusal.saw) {
            ls_print_refusal_ok("check-telemetry", refusal, have_run_start, line_no);
            return 0;
        }
        fprintf(stderr, "check-telemetry: OK (%d events, %d slots)\n", line_no, n_slots);

        fprintf(stderr, "  telemetry summary:\n");
        for (const auto & pd : phases) {
            const double tok_s = pd.wall_us > 0 ? (double) pd.tokens * 1e6 / (double) pd.wall_us : 0.0;
            fprintf(stderr,
                "    phase %-3s: %lld tokens, %.3f tok/s (wall %.1f ms) | transfer %.1f ms | wait %.1f ms | compute %.1f ms | layers %lld | evictions %lld | graph_builds %lld\n",
                pd.name.c_str(), pd.tokens, tok_s, pd.wall_us / 1000.0,
                pd.transfer_us / 1000.0, pd.wait_us / 1000.0, pd.compute_us / 1000.0,
                pd.layers_run, pd.evictions, pd.graph_builds);
        }
        if (peak_vram >= 0) {
            fprintf(stderr, "    peak VRAM: %.2f MiB\n", (double) peak_vram / (1024.0 * 1024.0));
        }
        return 0;
    }

    fprintf(stderr, "check-telemetry: %d invariant violation(s)\n", errors);
    return 1;
}

// ---------------------------------------------------------------------------
// --validate-telemetry: strict validation + profile report (T-Q38LS-06)
//
// Validates the full event sequence and required fields, then writes
// <path>.summary.json: a reproducible, evidence-only profile record. The
// report distinguishes transfer, availability wait, compute, graph-prep, and
// wall time; reports raw PP/TG tokens/s and peak VRAM; and states that the
// cold source state was best-effort (uncontrolled Windows file cache) and that
// no speedup baseline was run. Malformed or incomplete telemetry returns 1
// and writes no report.
// ---------------------------------------------------------------------------

struct layer_agg {
    uint64_t n = 0;
    double transfer_us = 0.0, transfer_min = 1e30, transfer_max = 0.0;
    double wait_us     = 0.0, wait_min     = 1e30, wait_max     = 0.0;
    double compute_us  = 0.0, compute_min  = 1e30, compute_max  = 0.0;
    uint64_t ready_at_consumption = 0;
    uint64_t n_ready_waits = 0;
    bool seen_copy = false;
    bool seen_wait = false;
    bool seen_compute = false;

    void reset() {
        n = 0;
        transfer_us = wait_us = compute_us = 0.0;
        transfer_min = wait_min = compute_min = 1e30;
        transfer_max = wait_max = compute_max = 0.0;
        ready_at_consumption = 0;
        n_ready_waits = 0;
        seen_copy = seen_wait = seen_compute = false;
    }
};

struct phase_agg {
    bool have_summary = false;
    std::string phase;
    long long tokens = 0, wall_us = 0, graph_builds = 0, graph_prep_us = 0;
    double tok_s = 0.0, total_transfer_us = 0.0, total_wait_us = 0.0, total_compute_us = 0.0;
    long long bytes_transferred = 0, evictions = 0, layers_run = 0;
    std::map<int, layer_agg> layers;
};

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\""; break;
            case '\\': out += "\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default:   out += c; break;
        }
    }
    return out;
}

static int validate_telemetry(const std::string & path) {
    FILE * f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        fprintf(stderr, "validate-telemetry: cannot open %s\n", path.c_str());
        return 2;
    }

    char buf[65536];
    int line_no = 0;
    int n_slots = -1, n_layer = -1, n_events = 0;
    bool have_run_start = false;
    bool have_run_end = false;
    long long exit_status = -1;
    long long n_tokens_generated = -1;
    long long peak_vram = -1;
    std::string model_name, arch, error_str;
    long long total_vram = -1, free_vram_at_start = -1, slot_vram = -1, device = -1;
    long long tg_ids_count = -1;
    int errors = 0;

    auto err = [&](const char * msg) {
        fprintf(stderr, "validate-telemetry: line %d: %s\n", line_no, msg);
        errors++;
    };

    // per-phase aggregation; phase 0 = PP, 1 = TG
    std::map<int, phase_agg> phases;
    int current_phase = -1;
    bool handoff_seen = false;         // T-Q38HG-02 (reset per phase)
    bool handoff_ever = false;         // T-GPLR-12: handoff events ever seen (never reset)
    bool ho_mode_valid = false;        // T-GPLR-12: run_start carried a valid handoff_mode
    bool handoff_active_expected = false; // T-GPLR-12: mode on/auto-on
    bool off_audit_trace = false;      // T-GPLR-05: policy-off audit record schema
    // T-GPLR-13: moe-row-set trace facts (run_start marker) + per-slot row
    // coverage for the generation-lifecycle rule.
    bool rowset_mode = false;
    bool rowset_ever = false;          // any rowset_* event seen (default-off immunity)
    long long rowset_n_expert = -1, rowset_top_k = -1, rowset_window_tokens = 1;
    struct rowset_slot_state {
        long long gen = -1;
        long long owner = -1;
        long long resident = 0;
    };
    std::unordered_map<int, rowset_slot_state> rowset_slots;
    // T-GPLR-19: >0 when rowset_init declares per-layer retention
    long long rowset_retain_k = -1;
    // T-GPLR-58: the per-layer retention cache model (rows, recency, and the staged
    // speculative fills of the `spec` events that precede each rowset_fetch).
    ls_retention_model rowset_retention;
    ls_pool_validator pool_validator;
    // T-GPLR-19: byte_account's row_cache figure, held until rowset_init has
    // been read so the record can be tied to the policy it claims.
    bool ba_row_cache_seen = false;
    long long ba_row_cache = 0;
    // T-GPLR-22 (see the light checker).
    ls_refusal_facts refusal;

    struct slot_state {
        long long gen = 0;
        long long pending_gen = 0;
        long long pending_layer = -1;
        long long last_copy_layer = -1;
    };
    std::unordered_map<int, slot_state> slots;

    while (fgets(buf, sizeof(buf), f) != nullptr) {
        line_no++;
        std::string line(buf);
        if (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
        }
        if (line.empty() || line[0] != '{' || line.back() != '}') {
            // an event line must be a single-line JSON object; anything else is malformed
            err("malformed line (expected a single-line JSON object)");
            continue;
        }
        n_events++;

        const std::string event = json::value(line, "event");

        if (event == "run_start") {
            if (have_run_start) {
                err("duplicate run_start");
            }
            have_run_start = true;
            // T-GPLR-05: off-mode audit traces carry run metadata only.
            const bool off_audit = ls_line_is_off_audit(line);
            bool found = false;
            n_slots = (int) json::integer(line, "n_slots", &found);
            if (!off_audit && (!found || n_slots <= 0)) err("run_start missing valid n_slots");
            n_layer = (int) json::integer(line, "n_layer", &found);
            if (!off_audit && (!found || n_layer <= 0)) err("run_start missing valid n_layer");
            for (const auto & pe : pool_validator.run_start_errors(line)) {
                err(pe.c_str());
            }
            if (off_audit) {
                off_audit_trace = true;
                ls_check_off_run_start(line, line_no, "validate-telemetry", errors);
                continue;
            }
            model_name = json::value(line, "model");
            arch       = json::value(line, "arch");
            total_vram = json::integer(line, "total_vram_bytes", &found);
            if (!found) err("run_start missing total_vram_bytes");
            free_vram_at_start = json::integer(line, "free_vram_bytes", &found);
            if (!found) err("run_start missing free_vram_bytes");
            slot_vram = json::integer(line, "slot_vram_bytes", &found);
            if (!found) err("run_start missing slot_vram_bytes");
            device = json::integer(line, "device", &found);
            if (!found) err("run_start missing device");
            // T-GPLR-12: the auditable handoff decision (optional for legacy
            // traces; cross-checked against handoff events like the strict
            // checker).
            ho_mode_valid = false;
            handoff_active_expected = false;
            if (line.find("\"handoff_mode\":") != std::string::npos) {
                const std::string hm = json::value(line, "handoff_mode");
                if (hm != "on" && hm != "off" && hm != "auto-on" && hm != "auto-off") {
                    err("run_start has invalid handoff_mode");
                } else {
                    ho_mode_valid = true;
                    handoff_active_expected = (hm == "on" || hm == "auto-on");
                }
            }
            // T-GPLR-13: the moe-row-set opt-in marker (absent when off).
            if (line.find("\"rowset\":\"on\"") != std::string::npos) {
                rowset_mode = true;
            }
            // T-GPLR-52 A3/AC4: the engine-side staging source + match
            // cross-check. Same rule as the strict checker: optional for
            // legacy traces, fatal when the pair disagrees or the value is
            // unknown - a run must never carry a staging_source that
            // contradicts its declared copy_source.
            {
                const std::string stage_src = json::value(line, "staging_source");
                if (!stage_src.empty()) {
                    if (stage_src != "direct" && stage_src != "pinned" &&
                        stage_src != "mmap-registered" && stage_src != "mmap-registered-churn") {
                        err("run_start has invalid staging_source");
                    }
                    bool stage_found = false;
                    const long long stage_match = json::integer(line, "staging_source_match", &stage_found);
                    if (!stage_found) {
                        err("run_start has staging_source without staging_source_match");
                    } else if (stage_match != 1) {
                        err("run_start staging_source does not match the declared copy_source");
                    }
                }
            }
        } else if (event == "handoff") {
            // T-Q38HG-02: handoff invalidates TG graphs, so graph_builds > 0
            // is expected on the next TG phase.
            handoff_seen = true;
            handoff_ever = true;
            // T-GPLR-12: a run whose run_start declared off/auto-off must
            // never emit handoff events.
            if (ho_mode_valid && !handoff_active_expected) {
                err("handoff event in a run whose run_start declared handoff_mode off/auto-off");
            }
            // T-GPLR-10: validate the migrate event's migration list (the
            // release/teardown actions carry no layer fields).
            const std::string action = json::value(line, "action");
            if (action != "migrate" && action != "release" && action != "teardown") {
                err("handoff event with unknown action");
            }
            if (action == "migrate") {
                bool f = false;
                const long long layers = json::integer(line, "layers", &f);
                const long long total_bytes = json::integer(line, "total_bytes", &f);
                const bool attn_only = json::value(line, "attention_only") == "true";
                const bool mtp_flag = json::value(line, "mtp") == "true";
                if (!f) {
                    err("handoff migrate event missing layers/total_bytes");
                } else {
                    if (total_bytes <= 0) {
                        err("handoff migrate total_bytes <= 0");
                    }
                    const long long expect_full = n_layer + (mtp_flag ? 1 : 0);
                    // T-GPLR-16: partial migration — the event must carry
                    // partial=true with a positive split below n_layer, and
                    // the layer count must equal n_layer - partial_from.
                    // Checked BEFORE the full-handoff count rule (which does
                    // not apply to a partial migration list) and via raw-
                    // JSON search (not json_str) to avoid the key-prefix
                    // collision: "partial" is a prefix of "partial_from".
                    const bool is_partial = line.find("\"partial\":true") != std::string::npos;
                    if (is_partial) {
                        bool fp = false;
                        const long long pf_ = json::integer(line, "partial_from", &fp);
                        if (!fp || pf_ <= 0 || pf_ >= n_layer) {
                            err("partial handoff migrate has invalid partial_from");
                        } else if (layers != n_layer - pf_) {
                            err("partial handoff layers != n_layer - partial_from");
                        }
                    }
                    if (!attn_only && !is_partial && layers != expect_full) {
                        err("full handoff layers != n_layer (+1 when MTP)");
                    }
                    if (attn_only && (layers <= 0 || layers >= expect_full)) {
                        err("attention-only handoff layer count out of range");
                    }
                    // T-GPLR-10: when present, the free-VRAM reading taken
                    // right after the handoff allocation must be sane; absent
                    // on legacy traces is accepted (field added this task)
                    bool ff = false;
                    const long long free_vram_ho = json::integer(line, "free_vram", &ff);
                    if (ff && free_vram_ho <= 0) {
                        err("handoff migrate event has invalid free_vram");
                    }
                }
            }
        } else if (event == "byte_account") {
            // T-GPLR-04: additive byte-class sums must be internally consistent
            // and the bounded window must sit below observed free VRAM.
            bool f = false;
            const long long full = json::integer(line, "full_model_plus_runtime", &f);
            const long long bounded = json::integer(line, "bounded_window", &f);
            const long long st_ = json::integer(line, "static", &f);
            const long long pl = json::integer(line, "per_layer", &f);
            const long long sw = json::integer(line, "slot_window", &f);
            const long long kvb = json::integer(line, "kv", &f);
            const long long stt = json::integer(line, "state", &f);
            const long long free_b = json::integer(line, "free_vram", &f);
            if (!f) {
                err("byte_account missing fields");
            } else if (st_ + pl + sw + kvb + stt != bounded) {
                err("byte_account: static + per_layer + slot_window + kv + state != bounded_window");
            }
            if (full <= bounded) {
                err("byte_account: full_model_plus_runtime <= bounded_window (not an out-of-core proof)");
            }
            if (free_b > 0 && bounded >= free_b) {
                err("byte_account: bounded_window >= observed free VRAM");
            }
            // T-GPLR-19: the row cache and the TOTAL residency cap (shared
            // rule with check-telemetry, one owner); the policy tie is
            // deferred to the end of the trace because this event precedes
            // rowset_init.
            {
                bool fb = false, fc = false, fp = false, fcap = false;
                const long long budget_mib = json::integer(line, "residency_budget_mib", &fb);
                const long long row_cache_b = json::integer(line, "row_cache", &fc);
                const long long pool_b = json::integer(line, "prediction_pool", &fp);
                const long long cap_b = json::integer(line, "residency_cap", &fcap);
                ba_row_cache_seen = fc;
                ba_row_cache = row_cache_b;
                for (const auto & be : ls_byte_account_errors(bounded, budget_mib, fc, row_cache_b,
                                                               fp, pool_b, fcap, cap_b)) {
                    err(be.c_str());
                }
                for (const auto & be : pool_validator.byte_account_errors(line)) {
                    err(be.c_str());
                }
            }
        } else if (event == "error") {
            error_str = json::value(line, "message");
            err((std::string("run produced an error event: ") + error_str).c_str());
        } else if (event == "phase_summary") {
            // phase is a string: "pp" or "tg"; map to 0/1
            const std::string ph = json::value(line, "phase");
            if (ph == "pp") {
                current_phase = 0;
            } else if (ph == "tg") {
                current_phase = 1;
            } else {
                err("phase_summary with unknown phase");
                continue;
            }
            auto & pa = phases[current_phase];
            if (pa.have_summary) {
                err("duplicate phase_summary for phase");
                continue;
            }
            pa.have_summary = true;
            bool found = false;
            pa.phase = ph;
            pa.tokens = json::integer(line, "tokens", &found);
            if (!found) err("phase_summary missing tokens");
            pa.wall_us = json::integer(line, "wall_us", &found);
            if (!found) err("phase_summary missing wall_us");
            pa.tok_s = atof(json::value(line, "tokens_per_s").c_str());
            pa.graph_builds = json::integer(line, "graph_builds", &found);
            if (!found) err("phase_summary missing graph_builds");
            // T-Q38HG-02: graph_builds > 0 acceptable after handoff
            if (pa.graph_builds != 0 && !handoff_seen) err("phase_summary.graph_builds != 0");
            const bool after_handoff = handoff_seen; // T-GPLR-10: hybrid-TG bytes rule below
            handoff_seen = false; // reset per phase
            pa.graph_prep_us = json::integer(line, "graph_prep_us", &found);
            if (!found) err("phase_summary missing graph_prep_us");
            pa.total_transfer_us = atof(json::value(line, "total_transfer_us").c_str());
            pa.total_wait_us = atof(json::value(line, "total_wait_us").c_str());
            pa.total_compute_us = atof(json::value(line, "total_compute_us").c_str());
            pa.bytes_transferred = json::integer(line, "bytes_transferred", &found);
            if (!found) err("phase_summary missing bytes_transferred");
            pa.evictions = json::integer(line, "evictions", &found);
            if (!found) err("phase_summary missing evictions");
            pa.layers_run = json::integer(line, "layers_run", &found);
            if (!found) err("phase_summary missing layers_run");
            // T-GPLR-10: the phase right after a handoff migrate is the hybrid
            // TG -- stock llama_decode with the banks already resident, so no
            // bank copies exist by construction: bytes_transferred must be ~0.
            if (after_handoff && ph == "tg" && pa.bytes_transferred != 0 && !rowset_mode) {
                err("post-handoff TG phase transferred bytes (hybrid TG runs stock llama_decode; expected 0 bank bytes)");
            }
        } else if (event == "tg_tokens") {
            const std::string v = json::value(line, "n_tokens_generated");
            tg_ids_count = v.empty() ? -1 : atoll(v.c_str());
        } else if (ls_parse_run_refused(event, line, refusal)) {
        } else if (event == "run_end") {
            if (have_run_end) {
                err("duplicate run_end");
            }
            have_run_end = true;
            // T-GPLR-13 default-off immunity: rowset_* events require the
            // run_start "rowset":"on" marker.
            if (!rowset_mode && rowset_ever) {
                err("rowset events in a run whose run_start carries no rowset marker");
            }
            bool found = false;
            exit_status = json::integer(line, "exit_status", &found);
            if (!found) err("run_end missing exit_status");
            else if (exit_status != 0) err("run_end.exit_status != 0");
            // T-GPLR-12: on/auto-on must have emitted handoff events. Use
            // handoff_ever (handoff_seen is reset by each phase_summary).
            if (ho_mode_valid && handoff_active_expected && !handoff_ever) {
                err("run_start declared handoff_mode on/auto-on but no handoff events followed");
            }
            peak_vram = json::integer(line, "peak_vram_bytes", &found);
            if (!found) err("run_end missing peak_vram_bytes");
            n_tokens_generated = json::integer(line, "n_tokens_generated", &found);
            if (!found) err("run_end missing n_tokens_generated");
            error_str = json::value(line, "error");
        } else if (event == "spec_pool_init") {
            for (const auto & pe : pool_validator.init_errors(line)) {
                err(pe.c_str());
            }
        } else if (event == "spec_pool") {
            for (const auto & pe : pool_validator.event_errors(line)) {
                err(pe.c_str());
            }
        } else if (event == "rowset_init" || event == "rowset_fetch" ||
                   event == "rowset_round" || event == "rowset_refused") {
            // T-GPLR-13: row-set events require the run_start marker.
            rowset_ever = true;
            if (!rowset_mode) {
                err("rowset event in a run whose run_start carries no rowset marker");
            }
            if (event == "rowset_init") {
                bool f = false;
                rowset_n_expert = json::integer(line, "n_expert", &f);
                pool_validator.set_n_expert(rowset_n_expert);
                if (!f || rowset_n_expert <= 0) err("rowset_init missing valid n_expert");
                rowset_top_k = json::integer(line, "top_k", &f);
                if (!f || rowset_top_k <= 0) err("rowset_init missing valid top_k");
                // same rule as check_telemetry: the two validators must stay
                // identical on rowset events (T-GPLR-13 validator parity)
                rowset_window_tokens = json::integer(line, "window_tokens", &f);
                if (!f || rowset_window_tokens < 1) err("rowset_init missing valid window_tokens");
                // T-GPLR-19: 0/absent = reactive slot-keyed policy (unchanged)
                rowset_retain_k = json::integer(line, "retain_k", &f);
                if (f && rowset_retain_k < 0) err("rowset_init retain_k must not be negative");
            } else if (event == "rowset_fetch") {
                bool f = false, rf = false, bf = false;
                const long long layer = json::integer(line, "layer", &f);
                const long long slot  = json::integer(line, "slot", &f);
                const long long gen   = json::integer(line, "generation", &f);
                const long long ru    = json::integer(line, "route_union", &f);
                const long long nf    = json::integer(line, "n_rows_fetched", &f);
                const long long nr    = json::integer(line, "n_rows_resident", &f);
                const long long bs    = json::integer(line, "bytes_staged", &f);
                const long long ub    = json::integer(line, "ubatch", &rf);
                const long long nret  = json::integer(line, "n_rows_retained", &rf);
                const long long nrtd  = json::integer(line, "n_rows_retired", &rf);
                const long long bank  = json::integer(line, "bytes_full_bank", &bf);
                const long long phase = json::integer(line, "phase", nullptr);
                if (!f || !rf || !bf || layer < 0 || slot < 0 || ru < 0 || nf < 0 || nr < 0 ||
                    ub < 1 || nret < 0 || nrtd < 0 || bank < 0) {
                    err("rowset_fetch missing required fields");
                    continue;
                }
                const std::vector<long long> pool_ids = json::integer_list(line, "pool_hit_ids", nullptr);
                const bool pool_enabled = pool_validator.enabled();
                const long long pool_rows = pool_enabled ? (long long) pool_ids.size() : 0;
                for (const auto & pe :pool_validator.fetch_errors(line, layer, gen, ub, ru, nf, nr,
                                                      nret, bs, bank)) {
                    err(pe.c_str());
                }
                if (rowset_top_k >= 0 && rowset_window_tokens >= 1) {
                    const long long max_union = std::min(rowset_n_expert,
                                                          rowset_window_tokens * rowset_top_k);
                    const bool valid_union = rowset_window_tokens == 1
                        ? ru == rowset_top_k
                        : ru >= 1 && ru <= max_union;
                    if (!valid_union) {
                        err("rowset_fetch route_union outside declared window*top_k bound");
                    }
                }
                if (ru != nf + nr + pool_rows) {
                    err("rowset_fetch route_union != real fetched + real resident + pool hits");
                }
                if ((bs > 0) != (nf > 0)) {
                    err("rowset_fetch bytes_staged/n_rows_fetched mismatch");
                }
                {
                    const std::string ids = json::value(line, "union_ids");
                    const std::string hash_s = json::value(line, "hash");
                    if (ids.empty() || hash_s.empty()) {
                        err("rowset_fetch missing union_ids/hash");
                    } else {
                        uint64_t h = 1469598103934665603ull;
                        for (unsigned char c : ids) {
                            h ^= (uint64_t) c;
                            h *= 1099511628211ull;
                        }
                        // strtoull: the U64 hash exceeds LLONG_MAX (atoll saturates)
                        if (h != strtoull(hash_s.c_str(), nullptr, 10)) {
                            err("rowset_fetch hash mismatch over union_ids");
                        }
                    }
                }
                // T-GPLR-19: same branch as check_telemetry (validator parity)
                if (rowset_retain_k > 0) {
                    for (const auto & e : ls_retention_errors(rowset_retention,
                             layer, gen, phase, ru, nr, pool_ids,
                             json::value(line, "union_ids"), nret, nrtd, rowset_retain_k,
                             rowset_n_expert, rowset_top_k)) {
                        err(e.c_str());
                    }
                } else {
                {
                    auto & rs = rowset_slots[(int) slot];
                    if (rs.gen != gen || rs.owner != layer) {
                        if (nr != 0) {
                            err("rowset_fetch first window per (slot, generation) claims resident rows");
                        }
                        if (nf != ru) {
                            err("rowset_fetch first window must fetch the full union");
                        }
                        rs.gen = gen;
                        rs.owner = layer;
                        rs.resident = ru;
                    } else {
                        if (nr != rs.resident) {
                            err("rowset_fetch n_rows_resident != tracked slot coverage");
                        }
                        rs.resident += nf;
                    }
                    if (ru > rowset_n_expert) {
                        err("rowset_fetch route_union exceeds n_expert");
                    }
                }
                }
            } else if (event == "rowset_round") {
                bool f = false;
                const long long bs = json::integer(line, "bytes_staged", &f);
                const long long bb = json::integer(line, "bytes_full_bank", &f);
                if (!f || bs < 0 || bb <= 0) {
                    err("rowset_round missing required fields");
                }
            } else if (event == "spec" && rowset_retain_k > 0) {
                // T-GPLR-58 (validator parity with check_telemetry): the speculative
                // fills this stage issued, staged for the rowset_fetch of the same
                // stage to replay.
                for (const auto & e : ls_spec_errors(rowset_retention, line)) {
                    err(e.c_str());
                }
            } else {
                err((std::string("rowset_refused: ") +
                     (json::value(line, "reason").empty() ? "<no reason>" : json::value(line, "reason"))).c_str());
            }
        } else if (event == "slot_request" || event == "copy_submit" || event == "copy_complete" ||
                   event == "layer_ready_wait" || event == "layer_compute" || event == "slot_evict") {
            if (off_audit_trace) {
                ls_err_off_streaming_event(line_no, "validate-telemetry", errors);
                continue;
            }
            if (n_slots <= 0) {
                err("layer event before run_start");
                continue;
            }
            bool f_layer = false, f_slot = false, f_gen = false, f_phase = false;
            long long layer = json::integer(line, "layer", &f_layer);
            long long slot  = json::integer(line, "slot",  &f_slot);
            long long gen   = json::integer(line, "generation", &f_gen);
            const long long phase = json::integer(line, "phase", &f_phase);
            if (f_phase && phase >= 0) {
                current_phase = (int) phase;
            }
            if (!f_layer || !f_slot) {
                err("layer event missing layer/slot");
                continue;
            }
            if (slot != layer % n_slots) {
                err("slot != layer % n_slots");
            }
            if (slot < 0 || slot >= n_slots) {
                err("slot out of range");
                continue;
            }
            if (event == "slot_request") {
                // slot_request carries no generation; only layer/slot are checked above
                continue;
            }
            if (event == "slot_evict") {
                if (!f_gen) {
                    err("slot_evict missing generation");
                    continue;
                }
                const long long per_slot = (n_layer - 1 - slot) / n_slots + 1;
                const long long expect_layer = slot + ((gen - 1) % per_slot) * n_slots;
                if (layer != expect_layer) {
                    err("slot_evict layer is not the prior resident layer");
                }
                continue;
            }
            if (!f_gen) {
                err("layer event missing generation");
                continue;
            }

            auto & st = slots[(int) slot];
            auto & la = phases[current_phase < 0 ? 0 : current_phase].layers[(int) layer];

            if (event == "copy_submit") {
                if (gen <= st.gen) {
                    err("copy_submit generation did not strictly increase");
                }
                const long long per_slot = n_layer / n_slots;
                const long long expect_layer = slot + ((gen - 1) % per_slot) * n_slots;
                if (layer != expect_layer) {
                    err("copy_submit layer violates deterministic slot policy");
                }
                st.gen = gen;
                st.pending_gen = gen;
                st.pending_layer = layer;
            } else if (event == "copy_complete") {
                if (st.pending_gen != gen || st.pending_layer != layer) {
                    err("copy_complete has no matching copy_submit");
                    continue;
                }
                const std::string host_submit = json::value(line, "host_submit_us");
                const std::string ready_latency = json::value(line, "ready_latency_us");
                if (!host_submit.empty() || !ready_latency.empty()) {
                    if (host_submit.empty() || ready_latency.empty()) {
                        err("copy_complete has partial timing split");
                    } else {
                        const double submit_us = atof(host_submit.c_str());
                        const double ready_us = atof(ready_latency.c_str());
                        const double total_us = atof(json::value(line, "transfer_us").c_str());
                        if (submit_us < 0.0 || ready_us < 0.0 || total_us < 0.0 ||
                            std::abs((submit_us + ready_us) - total_us) > 0.2) {
                            err("copy_complete timing split is invalid");
                        }
                    }
                }
                st.pending_gen = 0;
                st.pending_layer = -1;
                st.last_copy_layer = layer;
                const double tr = atof(json::value(line, "transfer_us").c_str());
                la.n++;
                la.transfer_us += tr;
                la.transfer_min = std::min(la.transfer_min, tr);
                la.transfer_max = std::max(la.transfer_max, tr);
                la.seen_copy = true;
            } else if (event == "layer_ready_wait") {
                if (st.last_copy_layer != layer) {
                    err("layer_ready_wait before matching copy_complete");
                    continue;
                }
                const double wt = atof(json::value(line, "wait_us").c_str());
                la.wait_us += wt;
                la.wait_min = std::min(la.wait_min, wt);
                la.wait_max = std::max(la.wait_max, wt);
                la.n_ready_waits++;
                if (json::value(line, "ready_at_consumption") == "true") {
                    la.ready_at_consumption++;
                }
                la.seen_wait = true;
            } else if (event == "layer_compute") {
                // T-GPLR-13: the row-set executor emits no copy pair (see the
                // light checker); the copy-completed rule is row-set exempt.
                if (!rowset_mode && st.last_copy_layer != layer) {
                    err("layer_compute before matching copy_complete");
                    continue;
                }
                const double cm = atof(json::value(line, "compute_us").c_str());
                la.compute_us += cm;
                la.compute_min = std::min(la.compute_min, cm);
                la.compute_max = std::max(la.compute_max, cm);
                la.seen_compute = true;
            }
        } else if (event.empty()) {
            err("unrecognized or missing event field");
        }
        // unknown event types are ignored (forward compatibility)
    }
    fclose(f);

    // sequence completeness. T-GPLR-22: for a refusal trace the three "missing"
    // rules are the opposite of what it must satisfy; the shared refusal rule
    // replaces them.
    for (const auto & re : ls_refusal_trace_errors(refusal.saw, refusal.reason_ok,
                                                  have_run_start, !phases.empty(), have_run_end)) {
        fprintf(stderr, "validate-telemetry: %s\n", re.c_str());
        errors++;
    }
    if (!have_run_start && !refusal.saw) {
        fprintf(stderr, "validate-telemetry: missing run_start\n");
        errors++;
    }
    // T-GPLR-19: the deferred half of the byte_account rule -- the cache figure
    // must agree with the retention policy this trace declared.
    for (const auto & be : ls_byte_account_policy_errors(ba_row_cache_seen, ba_row_cache, rowset_retain_k)) {
        fprintf(stderr, "validate-telemetry: %s\n", be.c_str());
        errors++;
    }
    // T-GPLR-58: the deferred form of the `spec` rule (validator parity).
    for (const auto & se : ls_spec_replay_errors(rowset_retention)) {
        fprintf(stderr, "validate-telemetry: %s\n", se.c_str());
        errors++;
    }
    for (const auto & pe : pool_validator.final_errors(refusal.saw)) {
        fprintf(stderr, "validate-telemetry: %s\n", pe.c_str());
        errors++;
    }
    if (phases.empty() && !off_audit_trace && !refusal.saw) {
        fprintf(stderr, "validate-telemetry: missing phase_summary\n");
        errors++;
    }
    if (!have_run_end && !refusal.saw) {
        fprintf(stderr, "validate-telemetry: missing run_end\n");
        errors++;
    }
    if (have_run_end && n_tokens_generated >= 0 && tg_ids_count >= 0 && tg_ids_count != n_tokens_generated) {
        fprintf(stderr, "validate-telemetry: tg_tokens count (%lld) != run_end.n_tokens_generated (%lld)\n",
                (long long) tg_ids_count, (long long) n_tokens_generated);
        errors++;
    }
    if (n_tokens_generated > 0 && !phases.count(1) && !off_audit_trace) {
        fprintf(stderr, "validate-telemetry: run_end reports generated tokens but no tg phase_summary\n");
        errors++;
    }

    if (errors == 0 && refusal.saw) {
        ls_print_refusal_ok("validate-telemetry", refusal, have_run_start, -1);
        return 0; // a refusal trace has no phases, so there is no report to build
    }

    if (errors > 0) {
        fprintf(stderr, "validate-telemetry: %d invariant violation(s); no report written\n", errors);
        return 1;
    }

    // ------------------------------------------------------------------
    // build the report
    // ------------------------------------------------------------------
    std::string summary_path = path + ".summary.json";
    std::string r;
    r += "{\n";
    r += "  \"report\": \"q38ls-layer-streaming-profile\",\n";
    r += "  \"validated\": true,\n";
    r += "  \"events\": " + std::to_string(n_events) + ",\n";
    r += "  \"environment\": {\n";
    r += "    \"model\": \"" + json_escape(model_name) + "\",\n";
    r += "    \"arch\": \"" + json_escape(arch) + "\",\n";
    r += "    \"n_layer\": " + std::to_string(n_layer) + ",\n";
    r += "    \"n_slots\": " + std::to_string(n_slots) + ",\n";
    r += "    \"device\": " + std::to_string(device) + ",\n";
    r += "    \"total_vram_bytes\": " + std::to_string(total_vram) + ",\n";
    r += "    \"free_vram_at_start_bytes\": " + std::to_string(free_vram_at_start) + ",\n";
    r += "    \"slot_vram_bytes\": " + std::to_string(slot_vram) + ",\n";
    r += "    \"peak_vram_bytes\": " + std::to_string(peak_vram) + ",\n";
    r += "    \"notes\": \"cold source state is best-effort: layer packages live in the host mmap and the Windows file cache is uncontrolled, so the H2D source may be warm; this is recorded as uncontrolled, not a guaranteed cold condition\"\n";
    r += "  },\n";
    r += "  \"caveats\": {\n";
    r += "    \"source_cache\": \"best-effort cold (uncontrolled Windows file cache)\",\n";
    r += "    \"baseline\": \"no speedup baseline was run; these are raw absolute measurements only\",\n";
    r += "    \"claims\": \"no quality, equivalence, or speedup claim is made\"\n";
    r += "  },\n";
    r += "  \"phases\": {\n";

    auto emit_layer = [&r](const char * key, const layer_agg & la, bool trailing_comma) {
        char b[512];
        snprintf(b, sizeof(b),
            "        \"%s\": {\"n\": %llu,"
            "\"transfer_us_mean\": %.1f, \"transfer_us_min\": %.1f, \"transfer_us_max\": %.1f,"
            "\"wait_us_mean\": %.1f, \"wait_us_min\": %.1f, \"wait_us_max\": %.1f,"
            "\"compute_us_mean\": %.1f, \"compute_us_min\": %.1f, \"compute_us_max\": %.1f,"
            "\"ready_at_consumption\": %llu, \"n_waits\": %llu}%s\n",
            key,
            (unsigned long long) la.n,
            la.n ? la.transfer_us / (double) la.n : 0.0, la.transfer_min == 1e30 ? 0.0 : la.transfer_min, la.transfer_max,
            la.n_ready_waits ? la.wait_us / (double) la.n_ready_waits : 0.0,
            la.wait_min == 1e30 ? 0.0 : la.wait_min, la.wait_max,
            la.n ? la.compute_us / (double) la.n : 0.0, la.compute_min == 1e30 ? 0.0 : la.compute_min, la.compute_max,
            (unsigned long long) la.ready_at_consumption, (unsigned long long) la.n_ready_waits,
            trailing_comma ? "," : "");
        r += b;
    };

    bool first_phase = true;
    for (const auto & kv : phases) {
        const phase_agg & pa = kv.second;
        // Each phase object closes itself at the end of its own iteration with
        // "    }" (no trailing comma here); a comma is appended on the next
        // iteration's prefix to separate sibling phases.
        if (first_phase) {
            r += "    \"" + pa.phase + "\": {\n";
            first_phase = false;
        } else {
            r += ",\n    \"" + pa.phase + "\": {\n";
        }
        r += "      \"tokens\": " + std::to_string(pa.tokens) + ",\n";
        r += "      \"wall_us\": " + std::to_string(pa.wall_us) + ",\n";
        r += "      \"tokens_per_s\": " + std::to_string(pa.tok_s) + ",\n";
        r += "      \"total_transfer_us\": " + std::to_string((long long) pa.total_transfer_us) + ",\n";
        r += "      \"total_wait_us\": " + std::to_string((long long) pa.total_wait_us) + ",\n";
        r += "      \"total_compute_us\": " + std::to_string((long long) pa.total_compute_us) + ",\n";
        r += "      \"bytes_transferred\": " + std::to_string(pa.bytes_transferred) + ",\n";
        r += "      \"evictions\": " + std::to_string(pa.evictions) + ",\n";
        r += "      \"layers_run\": " + std::to_string(pa.layers_run) + ",\n";
        r += "      \"graph_prep_us\": " + std::to_string(pa.graph_prep_us) + ",\n";
        r += "      \"graph_builds\": " + std::to_string(pa.graph_builds) + ",\n";
        r += "      \"per_layer\": {\n";
        {
            uint32_t i = 0;
            const uint32_t total = (uint32_t) pa.layers.size();
            for (const auto & lk : pa.layers) {
                char key[16];
                snprintf(key, sizeof(key), "%d", lk.first);
                emit_layer(key, lk.second, (i + 1) < total);
                i++;
            }
        }
        r += "      }\n    }\n";
    }
    r += "  },\n";
    r += "  \"generated_tokens\": {\n";
    r += "    \"n\": " + std::to_string(n_tokens_generated) + ",\n";
    r += "    \"ids_reported\": " + std::to_string(tg_ids_count) + "\n";
    r += "  }\n";
    r += "}\n";

    FILE * out = fopen(summary_path.c_str(), "w");
    if (out == nullptr) {
        fprintf(stderr, "validate-telemetry: cannot write %s\n", summary_path.c_str());
        return 2;
    }
    fputs(r.c_str(), out);
    fclose(out);

    fprintf(stderr, "validate-telemetry: OK (%d events, %d slots, %d layers) -> %s\n",
            n_events, n_slots, n_layer, summary_path.c_str());
    for (const auto & kv : phases) {
        fprintf(stderr, "  phase %s: %lld tokens, %.2f tok/s, transfer %.1f ms, wait %.1f ms, compute %.1f ms, peak VRAM %.2f MiB\n",
                kv.second.phase.c_str(), (long long) kv.second.tokens, kv.second.tok_s,
                kv.second.total_transfer_us / 1000.0, kv.second.total_wait_us / 1000.0, kv.second.total_compute_us / 1000.0,
                peak_vram / 1048576.0);
    }
    return 0;
}

int main(int argc, char ** argv) {
    // T-GPLR-56 A1: the first mark of the process. Buffered until the telemetry sink
    // opens (ls.init), then written as the trace's first line, so the acceptance quantity
    // Q = pass1_end_ts - proc_start_ts is defined and the load span becomes divisible.
    // ggml_time_init() must come first: ggml_time_us() divides by a QPC frequency that
    // this initializes, and until llama_backend_init() (line ~3604) nothing else does it
    // (measured crash 0xC0000094 before this line existed). It is InitOnce-guarded, so
    // llama_backend_init()'s own call later stays a no-op - but every t_us in a P1 trace
    // is now measured from process entry instead of from backend init.
    ggml_time_init();
    llama_ls_tel_mark("proc_start", nullptr);

    std::string model_path;
    int n_slots = 2;                     // owner default 2026-09-15 (was 1): overlap-capable minimum
    bool layer_slots_explicit = false;   // T-GPLR-05: --layer-slots actually passed (a contract: used verbatim)
    int pp_tokens = 512;
    int ubatch = 512;
    int tg_tokens = 0;
    int pp_repeat = 1;               // T-GPLR-28: in-process PP repeat (first-touch experiment)
    int pp_prewarm = 0;              // T-GPLR-53 A4: pre-timed modality pre-warm (0=off default; bitfield 1/2/4/8)
    int pp_prewarm_delay_ms = 1700;  // T-GPLR-53 A4: mode-4 no-touch delay duration (default = compiled control)
    int pp_span_pace = 0;            // T-GPLR-56 A3: paced registration of the mapped spans (0=off default, 1=paced)
    int pp_span_pace_ahead = 1;      // T-GPLR-56 A3: groups issued ahead of the layer being staged
    int pp_register_site = 0;        // T-GPLR-56 A3 arm control: 0 = post-prebuild (today), 1 = early (before the prebuild)
    int device = 0;
    std::string tel_path;
    std::string check_path;
    bool validate_mode = false;
    bool pp_no_logits = false;
    bool no_mmap = false;
    std::string pp_schedule = "layer-major"; // T-GPLR-12: layer-major + per-width graphs default
    bool layer_major_explicit = false; // T-GPLR-12: --pp-schedule actually passed
    bool pw_probe = true;            // T-GPLR-12: default within layer-major
    // T-GPLR-29/30 default profile v3 (owner decision 2026-09-16): the
    // measured optimum for PP wall on this program — pinned staging + the
    // async-feed worker at 2 slots. Measured 2.80-2.87 s warm 4k PP vs
    // 3.33-3.43 s (af=0) and 13.3-30.4 s (batch-major); 1,363 tok/s at 200k
    // F16 with only 996 MiB of slot windows. Requires --layer-slots >= 2
    // (enforced below, matching --pp-async-feed's existing gate).
    // T-GPLR-37 owner decision (2026-09-20): the default copy source becomes
    // the churn-free mmap-registered mode (register every package source span
    // ONCE at init, DMA straight from the model's own pinned file pages). The
    // pre-change default was "pinned". Measured basis (T-GPLR-37 addenda 2-3):
    // -23.3%/-15.4% warm 4k PP wall at 2/3 slots (the fastest recorded), hop
    // deleted (45.24 -> 0.06 ms/bank), zero churn (40/40 registrations before
    // the first layer_compute, setup back in band); the crowded 200k shape is
    // wall-neutral (-0.22%/-0.38%, within noise) so the flip is a pure win
    // there too, at a measured 17.23 GiB of pinned file pages for the run.
    // Explicit --copy-source still overrides everything, so every recorded arm
    // command stays reproducible. Source provenance is recorded per run in
    // run_start (copy_source + copy_source_source), and a defaulted run
    // announces itself on stderr (below, after validation).
    std::string copy_source = "mmap-registered";
    bool copy_source_explicit = false;   // T-GPLR-37: --copy-source actually passed
    std::string kv_cache = "q8";     // T-GPLR-12: q8 default (f16/q4 stay explicit)
    std::string residency = "auto";        // T-GPLR-05: auto | off | dense-layer | moe-expert-bank
    bool        probe_only = false;        // T-GPLR-05: --probe-capability (host-only)
    bool        forced_off = false;        // T-GPLR-05: --residency off branch
    bool fault_drop_member = false;        // T-GPLR-02 negative fixture
    int telemetry_chunk_stride = 10;
    int64_t graph_budget_mib = -1;   // -1 = no budget (T-Q38CAP-01)
    bool pw_probe_explicit = false;  // T-GPLR-12: --pw-probe / --no-pw-probe actually passed
    bool mtp_explicit = false;       // T-GPLR-12: --mtp actually passed
    bool handoff = false;            // T-Q38HG-02: PP->TG handoff (resident TG), set by the tri-state
    std::string handoff_mode_str = "auto"; // T-GPLR-12: auto | on | off (default auto)
    bool handoff_auto = false;       // T-GPLR-12: auto mode (fit decided at the capacity preflight)
    bool handoff_attention_only = false;  // T-Q38HG-04: only 16 attention layers
    int handoff_partial_n = -1;      // T-GPLR-16: NCMOE-style partial handoff (first N banks stay host-resident; -1 = off)
    bool residency_tg = false;       // T-GPLR-13: --residency-tg moe-row-set TG policy (reactive expert row windows)
    int  residency_tg_retain = 0;    // T-GPLR-19: --residency-tg-retain [N] per-layer row cache (0 = off, default; bare flag = kRowSetRetainDefault)
    int  residency_tg_window = 0;    // T-GPLR-62 S2: --residency-tg-window W compact row-set slot width in experts (0 = full bank, default)
    bool residency_tg_recon  = false; // T-GPLR-62 S2 A9: --residency-tg-recon byte-level staged-window recon (off by default)
    int         residency_tg_spec = 0;        // T-GPLR-58 A0: --residency-tg-spec <0|1|2> speculative placement (0 off, 1 scripted, 2 reserved for the A1 proposer)
    int         residency_tg_spec_budget = 0; // T-GPLR-58 A0: --residency-tg-spec-budget <N> pending fills per layer (0 = auto = top_k)
    int         residency_tg_spec_pool = 0;   // T-GPLR-59: separate prediction-only positions per logical layer
    std::string residency_tg_spec_script;     // T-GPLR-58 A0: --residency-tg-spec-script <path> the scripted proposal sequence
    int         residency_tg_spec_arm = -1;   // T-GPLR-58 A1: --residency-tg-spec-arm, -1 = unset (0 coact, 1 bar09, 2 fusion)
    int         residency_tg_spec_null = 0;   // T-GPLR-58 A1: --residency-tg-spec-null <0|1> the count-matched null source
    bool mtp = false;                // T-Q38HG-06: MTP speculative decode in handoff TG
    int mtp_draft = 2;               // T-Q38HG-06: draft tokens per verify round
    std::string act_slab_dtype_str = "bf16"; // T-GPLR-12: bf16 default (f32 stays explicit)
    int pp_prefetch_ahead = 0;       // T-GPLR-08: one-layer-ahead PrefetchVirtualMemory (0 = off, default)
    int pp_async_feed = 1;           // T-GPLR-25: async-feed worker (default 1 since the
                                     // 2026-09-16 default profile v3; --pp-async-feed 0 reverts)
    bool pp_af_serialize = false;    // T-GPLR-32: one worker H2D in flight (default 0 = off)
    bool pp_submit_batch = false;    // T-GPLR-33: one H2D per bank (V1 span copy; default 0 = off)
    bool pp_copy_overlap = false;    // T-GPLR-35: staging DMAs on the compute context's aux stream (default 0 = off)
    bool pp_recon = false;           // T-GPLR-35 recon: stream identity + per-bank copy/compute timing (default 0 = off)
    bool pp_mctx_recon = false;      // T-GPLR-41 recon: pass-boundary memory-lifecycle events (default off)
    // T-GPLR-39 (owner decision 2026-09-20): the route DIAGNOSTIC defaults OFF.
    // It reads the router probs back from the device and sorts/hashes them on
    // the host 2x n_layer times per pass; at u2048 that was 1161.5 of 1152.8 ms
    // of the `other` ledger bucket (100.8%, closure exact) - the diagnostic
    // measuring itself. The FUNCTIONAL row-set routing path (read_router_ids)
    // is untouched, so routing and output are identical either way. Explicit
    // --route-telemetry 1 restores the observational event (required by the
    // route-parity gate and every recorded route analysis).
    int route_telemetry = -1;        // -1 = not explicitly set (resolved below)
    // T-GPLR-43: re-run the graph prebuild between --pp-repeat passes. DEFAULT
    // ON for repeat > 1 (T-GPLR-50: every KV, f16 included - the T-GPLR-47
    // fresh-context rebuild removed the compute cost; executing pass 2 against
    // graphs derived at prebuild time against the pre-clear cache remains the
    // proven quantized-KV crash mechanism). Explicit --pp-prebuild-per-pass 0
    // restores the single-prebuild behavior for evidence reproducibility.
    int pp_prebuild_per_pass = -1;
    int pp_pool_n = 0;               // T-GPLR-30: persistent pinned pool (0 = off, default)
    std::string pp_pool_policy = "fixed";
    double rep_penalty = 1.0;        // T-GPLR-06: greedy repeat penalty (1 = off)
    int tg_logit_margin = 0;         // T-GPLR-54 A2: per-step logit readout (0 = off)
    int    rep_last_n = 64;          // T-GPLR-06: repeat-penalty window
    std::string prompt_text;         // real prompt text (overrides synthetic ids)
    std::string prompt_file;         // T-GPLR-07: --prompt-file (file prompt source)
    std::string tg_text_path;        // T-GPLR-07: --tg-text (generated-text file output)
    std::vector<int> boundary_layers;
    // T-GPLR-00: host-only inventory mode (no CUDA, no tensor data)
    std::string inspect_path;
    std::string inspect_json_path;
    int64_t     ctx_size = 262144;
    uint64_t    physical_vram_bytes = 0;
    std::string physical_vram_source = "unspecified";
    // T-GPLR-04: explicit out-of-core capacity proof. run_ctx_size > 0 forces
    // the run-mode context capacity (independent of the actual PP token count);
    // residency_budget_mib (optional) forces an explicit residency budget in
    // MiB that the preflight must honor; reserve_mib is the headroom the
    // preflight must leave between the bounded configuration and physical VRAM.
    int64_t     run_ctx_size = 0;
    int64_t     residency_budget_mib = 0;   // 0 = derive from the bounded estimate
    int64_t     reserve_mib = 512;          // preflight headroom reserve
    bool        allow_full_residency = false; // T-GPLR-04 contract: off by default
    // T-GPLR-19 (F2-M1): true when --residency-tg-retain named a depth. A named
    // depth is a contract (a gate run must get the depth it declares), so only
    // the bare-flag DEFAULT is fitted to the device.
    bool        residency_tg_retain_explicit = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model") {
            model_path = arg_value(argc, argv, i, "--model");
        } else if (arg == "--model-env") {
            const std::string name = arg_value(argc, argv, i, "--model-env");
            const char * value = getenv(name.c_str());
            if (value == nullptr || value[0] == '\0') {
                fprintf(stderr, "missing or empty model environment variable: %s\n", name.c_str());
                return 2;
            }
            model_path = value;
        } else if (arg == "--check-telemetry") {
            check_path = arg_value(argc, argv, i, "--check-telemetry");
        } else if (arg == "--inspect-model") {
            inspect_path = arg_value(argc, argv, i, "--inspect-model");
        } else if (arg == "--inspect-json") {
            inspect_json_path = arg_value(argc, argv, i, "--inspect-json");
        } else if (arg == "--ctx-size") {
            ctx_size = atoll(arg_value(argc, argv, i, "--ctx-size").c_str());
            // T-GPLR-04: --ctx-size is honored in run mode as the explicit
            // context/KV capacity, independent of the PP token count.
            run_ctx_size = ctx_size;
        } else if (arg == "--residency-budget-mib") {
            residency_budget_mib = atoll(arg_value(argc, argv, i, "--residency-budget-mib").c_str());
        } else if (arg == "--reserve-mib") {
            reserve_mib = atoll(arg_value(argc, argv, i, "--reserve-mib").c_str());
        } else if (arg == "--allow-full-residency") {
            allow_full_residency = true;
        } else if (arg == "--physical-vram-bytes") {
            physical_vram_bytes = strtoull(arg_value(argc, argv, i, "--physical-vram-bytes").c_str(), nullptr, 10);
        } else if (arg == "--physical-vram-source") {
            physical_vram_source = arg_value(argc, argv, i, "--physical-vram-source");
        } else if (arg == "--validate-telemetry") {
            // strict validation + profile report (T-Q38LS-06)
            check_path = arg_value(argc, argv, i, "--validate-telemetry");
            validate_mode = true;
        } else if (arg == "--layer-slots") {
            n_slots = atoi(arg_value(argc, argv, i, "--layer-slots").c_str());
            layer_slots_explicit = true;
        } else if (arg == "--boundary-layers") {
            const std::string v = arg_value(argc, argv, i, "--boundary-layers");
            size_t pos = 0;
            while (pos <= v.size()) {
                const size_t comma = v.find(',', pos);
                const std::string tok = v.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!tok.empty()) {
                    boundary_layers.push_back(atoi(tok.c_str()));
                }
                if (comma == std::string::npos) {
                    break;
                }
                pos = comma + 1;
            }
        } else if (arg == "--pp-tokens") {
            pp_tokens = atoi(arg_value(argc, argv, i, "--pp-tokens").c_str());
        } else if (arg == "--ubatch") {
            ubatch = atoi(arg_value(argc, argv, i, "--ubatch").c_str());
        } else if (arg == "--tg-tokens") {
            tg_tokens = atoi(arg_value(argc, argv, i, "--tg-tokens").c_str());
        } else if (arg == "--pp-repeat") {
            pp_repeat = atoi(arg_value(argc, argv, i, "--pp-repeat").c_str());
        } else if (arg == "--pp-prewarm") {
            // T-GPLR-53 A4: modality discriminator for the pass-1 embed excess.
            // Default-off: 0 runs nothing at all, so every existing shape stays
            // byte-identical. 1/2/3 are the (i)/(ii)/(iii) pre-warm variants.
            pp_prewarm = atoi(arg_value(argc, argv, i, "--pp-prewarm").c_str());
            if (pp_prewarm < 0 || pp_prewarm > 15) {
                fprintf(stderr, "error: --pp-prewarm expects 0..15 (0=off default, 1=CPU read-touch, 2=dummy GPU launch+sync, 3=both, 4=no-touch delay, 8=wrong-region touch); got %d\n", pp_prewarm);
                return 2;
            }
        } else if (arg == "--pp-prewarm-delay-ms") {
            // T-GPLR-53 A4 dose sweep: the mode-4 no-touch delay duration. Default
            // stays the compiled control duration (1700 ms), so this flag is opt-in.
            pp_prewarm_delay_ms = atoi(arg_value(argc, argv, i, "--pp-prewarm-delay-ms").c_str());
            if (pp_prewarm_delay_ms < 0 || pp_prewarm_delay_ms > 60000) {
                fprintf(stderr, "error: --pp-prewarm-delay-ms expects 0..60000; got %d\n", pp_prewarm_delay_ms);
                return 2;
            }
        } else if (arg == "--pp-span-pace") {
            // T-GPLR-56 A3: paced registration (see the block). Default 0 = off,
            // which leaves the code path, the events and the bytes unchanged.
            pp_span_pace = atoi(arg_value(argc, argv, i, "--pp-span-pace").c_str());
            if (pp_span_pace < 0 || pp_span_pace > 1) {
                fprintf(stderr, "error: --pp-span-pace expects 0 (off, default) or 1 (paced); got %d\n", pp_span_pace);
                return 2;
            }
        } else if (arg == "--pp-span-pace-ahead") {
            // How many GROUPS ahead of the layer being staged a registration is
            // issued. 1 mirrors the block's "a bounded distance ahead".
            pp_span_pace_ahead = atoi(arg_value(argc, argv, i, "--pp-span-pace-ahead").c_str());
            if (pp_span_pace_ahead < 0 || pp_span_pace_ahead > 64) {
                fprintf(stderr, "error: --pp-span-pace-ahead expects 0..64 groups; got %d\n", pp_span_pace_ahead);
                return 2;
            }
        } else if (arg == "--pp-register-site") {
            // T-GPLR-56 A3 arm control: WHERE the span registration is issued.
            // post-prebuild = today's site; early = the earliest legal point, right
            // after exec_layers is assigned (A2 measured that head start at 66.4 ms).
            // Registration during the model LOAD stays FORBIDDEN (owner decision).
            const std::string site = arg_value(argc, argv, i, "--pp-register-site");
            if (site == "post-prebuild") {
                pp_register_site = 0;
            } else if (site == "early") {
                pp_register_site = 1;
            } else {
                fprintf(stderr, "error: --pp-register-site expects post-prebuild (default) or early; got %s\n", site.c_str());
                return 2;
            }
        } else if (arg == "--greedy") {
            // TG is greedy by default in this runner; the flag is accepted for
            // contract-command compatibility
        } else if (arg == "--no-mtp") {
            // always skipped in this experimental path
        } else if (arg == "--no-mmap") {
            no_mmap = true;
        } else if (arg == "--tg-logit-margin") {
            // T-GPLR-54 A2: per-step top-1/top-2 readout depth (<= 0 = off).
            tg_logit_margin = atoi(arg_value(argc, argv, i, "--tg-logit-margin").c_str());
        } else if (arg == "--pp-no-logits") {
            pp_no_logits = true;
        } else if (arg == "--pp-schedule") {
            pp_schedule = arg_value(argc, argv, i, "--pp-schedule");
            layer_major_explicit = (pp_schedule == "layer-major");
        } else if (arg == "--residency") {
            residency = arg_value(argc, argv, i, "--residency");
        } else if (arg == "--probe-capability") {
            // T-GPLR-05: host-only capability probe; ignores GPU/model-load
            // work entirely after parsing the model path.
            probe_only = true;
        } else if (arg == "--fault-drop-member") {
            fault_drop_member = true;
        } else if (arg == "--copy-source") {
            copy_source = arg_value(argc, argv, i, "--copy-source");
            copy_source_explicit = true;
        } else if (arg == "--kv-cache") {
            kv_cache = arg_value(argc, argv, i, "--kv-cache");
        } else if (arg == "--telemetry") {
            tel_path = arg_value(argc, argv, i, "--telemetry");
        } else if (arg == "--telemetry-chunk-stride") {
            telemetry_chunk_stride = atoi(arg_value(argc, argv, i, "--telemetry-chunk-stride").c_str());
        } else if (arg == "--graph-budget-mib") {
            graph_budget_mib = atoll(arg_value(argc, argv, i, "--graph-budget-mib").c_str());
        } else if (arg == "--pw-probe") {
            pw_probe = true;
            pw_probe_explicit = true;
        } else if (arg == "--no-pw-probe") {
            // T-GPLR-12: opt-out of the per-width default
            pw_probe = false;
            pw_probe_explicit = true;
        } else if (arg == "--handoff") {
            handoff_mode_str = "on";
        } else if (arg == "--no-handoff") {
            // T-GPLR-12: explicit opt-out of the auto default
            handoff_mode_str = "off";
        } else if (arg == "--handoff-mode") {
            handoff_mode_str = arg_value(argc, argv, i, "--handoff-mode");
        } else if (arg == "--handoff-attention-only") {
            handoff_mode_str = "on";
            handoff_attention_only = true;
        } else if (arg == "--handoff-partial") {
            // T-GPLR-16: NCMOE-style partial handoff — migrate only the
            // routed-expert banks with il >= N; banks of the first N layers
            // stay host-resident (CPU) for mixed-placement TG. Implies an
            // explicit handoff-on request.            handoff_mode_str = "on";
            const std::string partial_arg = arg_value(argc, argv, i, "--handoff-partial");
            handoff_partial_n = atoi(partial_arg.c_str());
            if (handoff_partial_n < 0) {
                fprintf(stderr, "error: --handoff-partial requires a non-negative N (got '%s')\n",
                        partial_arg.c_str());
                return 2;
            }
        } else if (arg == "--residency-tg") {
            // T-GPLR-13: opt-in moe-row-set TG policy (reactive expert row
            // windows). Default-off; every use requires residency moe-expert-bank.
            residency_tg = true;
        } else if (arg == "--residency-tg-retain") {
            // T-GPLR-19 (F2-M1): opt-in per-layer row residency over the
            // row-set TG path (SDD section 4 execution shape (i)). Default-off.
            // The depth N is optional: a bare flag selects the default depth,
            // so an opt-in run gets the resolved operating point unless it
            // names another one.
            residency_tg_retain = kRowSetRetainDefault;
            if (is_decimal_token(argc, argv, i)) {
                residency_tg_retain = atoi(arg_value(argc, argv, i, "--residency-tg-retain").c_str());
                residency_tg_retain_explicit = true;
            }
            if (residency_tg_retain < 1 || residency_tg_retain > 64) {
                fprintf(stderr, "error: --residency-tg-retain expects 1..64 windows (got %d)\n", residency_tg_retain);
                return 2;
            }
        } else if (arg == "--residency-tg-window") {
            // T-GPLR-62 S2: narrow the row-set slot window from the full expert
            // bank to W experts. Default 0 = the pre-S2 full-bank slot, so every
            // existing command line is unchanged. W must be at least the
            // worst-case union the router can demand; a union wider than W is
            // refused at run time with the stable reason "window_overflow"
            // rather than truncated, because truncating would silently drop
            // experts the router executed.
            residency_tg_window = atoi(arg_value(argc, argv, i, "--residency-tg-window").c_str());
            if (residency_tg_window < 0 || residency_tg_window > 4096) {
                fprintf(stderr, "error: --residency-tg-window expects 0..4096 experts (got %d)\n",
                        residency_tg_window);
                return 2;
            }
        } else if (arg == "--residency-tg-recon") {
            // T-GPLR-62 S2 / gate A9: byte-compare every staged slot slab
            // against the checkpoint slab for the expert that position is
            // specified to hold. Diagnostics only, off by default, and SLOW: it
            // reads back each staged slab. A mismatch refuses the run.
            residency_tg_recon = true;
        } else if (arg == "--residency-tg-spec") {
            // T-GPLR-58: opt-in SPECULATIVE placement of guessed expert rows on the
            // retention path. Default-off. Mode 1 = scripted proposals (the A0 test
            // vehicle: the guesses come from a file, so the mechanism is exercised
            // without a predictor); mode 2 = the A1 proposer, where the engine asks
            // the arm itself (--residency-tg-spec-arm).
            residency_tg_spec = atoi(arg_value(argc, argv, i, "--residency-tg-spec").c_str());
            if (residency_tg_spec < 0 || residency_tg_spec > 2) {
                fprintf(stderr, "error: --residency-tg-spec expects 0, 1 or 2 (got %d)\n", residency_tg_spec);
                return 2;
            }
        } else if (arg == "--residency-tg-spec-pool") {
            residency_tg_spec_pool = atoi(arg_value(argc, argv, i, "--residency-tg-spec-pool").c_str());
            if (residency_tg_spec_pool < 0 || residency_tg_spec_pool > 256) {
                fprintf(stderr, "error: --residency-tg-spec-pool expects 0..256 (got %d)\n", residency_tg_spec_pool);
                return 2;
            }
        } else if (arg == "--residency-tg-spec-budget") {
            residency_tg_spec_budget = atoi(arg_value(argc, argv, i, "--residency-tg-spec-budget").c_str());
            if (residency_tg_spec_budget < 0) {
                fprintf(stderr, "error: --residency-tg-spec-budget expects a non-negative count (got %d)\n",
                        residency_tg_spec_budget);
                return 2;
            }
        } else if (arg == "--residency-tg-spec-script") {
            residency_tg_spec_script = arg_value(argc, argv, i, "--residency-tg-spec-script");
        } else if (arg == "--residency-tg-spec-arm") {
            // T-GPLR-58 A1 (mode 2): WHICH arm proposes. Only the three names the
            // block pre-registered are accepted; an unknown name is refused here
            // rather than defaulted, because a leg that silently ran a different
            // arm than it named is worse than no leg.
            const std::string arm_name = arg_value(argc, argv, i, "--residency-tg-spec-arm");
            if      (arm_name == "coact")  { residency_tg_spec_arm = 0; }
            else if (arm_name == "bar09")  { residency_tg_spec_arm = 1; }
            else if (arm_name == "fusion") { residency_tg_spec_arm = 2; }
            else if (arm_name == "d1")     { residency_tg_spec_arm = 3; }
            else {
                fprintf(stderr, "error: --residency-tg-spec-arm expects coact, bar09, fusion or d1 (got \"%s\")\n",
                        arm_name.c_str());
                return 2;
            }
        } else if (arg == "--residency-tg-spec-null") {
            residency_tg_spec_null = atoi(arg_value(argc, argv, i, "--residency-tg-spec-null").c_str());
            if (residency_tg_spec_null < 0 || residency_tg_spec_null > 1) {
                fprintf(stderr, "error: --residency-tg-spec-null expects 0 or 1 (got %d)\n",
                        residency_tg_spec_null);
                return 2;
            }
        } else if (arg == "--mtp") {            // T-Q38HG-06: enable MTP speculative decoding in the handoff TG
            // path. Requires --handoff (-attention-only) and --tg-tokens.
            mtp = true;
            mtp_explicit = true;
        } else if (arg == "--mtp-draft") {
            mtp_draft = atoi(arg_value(argc, argv, i, "--mtp-draft").c_str());
        } else if (arg == "--rep-penalty") {
            rep_penalty = atof(arg_value(argc, argv, i, "--rep-penalty").c_str());
        } else if (arg == "--rep-last-n") {
            rep_last_n = atoi(arg_value(argc, argv, i, "--rep-last-n").c_str());
        } else if (arg == "--prompt") {
            prompt_text = arg_value(argc, argv, i, "--prompt");
        } else if (arg == "--prompt-env") {
            const std::string name = arg_value(argc, argv, i, "--prompt-env");
            const char * val = getenv(name.c_str());
            prompt_text = val ? val : "";
            if (prompt_text.empty()) {
                fprintf(stderr, "error: environment variable '%s' is empty or unset\n", name.c_str());
                return 2;
            }
        } else if (arg == "--prompt-file") {
            // T-GPLR-07: file prompt source (validated + read below, before
            // any CUDA init; see the run-mode validation section).
            prompt_file = arg_value(argc, argv, i, "--prompt-file");
        } else if (arg == "--tg-text") {
            // T-GPLR-07: generated-text file output (streaming path only;
            // rejected with --residency off via the off-conflict table).
            tg_text_path = arg_value(argc, argv, i, "--tg-text");
        } else if (arg == "--act-slab-dtype") {
            act_slab_dtype_str = arg_value(argc, argv, i, "--act-slab-dtype");
        } else if (arg == "--pp-prefetch-ahead") {
            // T-GPLR-08: one-layer-ahead host-feed prefetch probe (streaming
            // path only; rejected with --residency off via the off-conflict
            // table; range validation happens in the module at configure time).
            pp_prefetch_ahead = atoi(arg_value(argc, argv, i, "--pp-prefetch-ahead").c_str());
        } else if (arg == "--pp-pool") {
            // T-GPLR-30: N[,policy] — N bank-sized persistent pinned buffers;
            // policy 'fixed' (banks 0..N-1) or 'lru' (N-entry LRU, cliff control)
            const std::string v = arg_value(argc, argv, i, "--pp-pool");
            const size_t comma = v.find(',');
            pp_pool_n = atoi((comma == std::string::npos ? v : v.substr(0, comma)).c_str());
            if (comma != std::string::npos) {
                pp_pool_policy = v.substr(comma + 1);
            }
        } else if (arg == "--pp-async-feed") {
            // T-GPLR-25: device-side copy-overlap prototype (0 = off = today's
            // byte-identical behavior; 1 = one-layer-ahead worker feed).
            pp_async_feed = atoi(arg_value(argc, argv, i, "--pp-async-feed").c_str());
        } else if (arg == "--pp-af-serialize") {
            // T-GPLR-32: one worker H2D in flight (stream-event gate; the CPU
            // hop stays overlapped). Only 0/1 accepted; requires af=1.
            const int v = atoi(arg_value(argc, argv, i, "--pp-af-serialize").c_str());
            if (v != 0 && v != 1) {
                fprintf(stderr, "error: --pp-af-serialize must be 0 (off) or 1 (one worker H2D in flight)\n");
                return 2;
            }
            pp_af_serialize = (v == 1);
        } else if (arg == "--pp-submit-batch") {
            // T-GPLR-33 V1: one H2D per bank when the slot-build contiguity
            // check tiled the members exactly; per-member otherwise (loud
            // degradation via the submit_batch telemetry field). 0/1 only.
            const int v = atoi(arg_value(argc, argv, i, "--pp-submit-batch").c_str());
            if (v != 0 && v != 1) {
                fprintf(stderr, "error: --pp-submit-batch must be 0 (off) or 1 (one H2D per bank)\n");
                return 2;
            }
            pp_submit_batch = (v == 1);
        } else if (arg == "--pp-copy-overlap") {
            // T-GPLR-35: staging DMAs on the aux stream (No. 1) of the compute
            // backend's CUDA context, so the staging/compute ordering waits
            // stay intra-context. 0/1 only.
            const int v = atoi(arg_value(argc, argv, i, "--pp-copy-overlap").c_str());
            if (v != 0 && v != 1) {
                fprintf(stderr, "error: --pp-copy-overlap must be 0 (off) or 1 (staging on the compute context's aux stream)\n");
                return 2;
            }
            pp_copy_overlap = (v == 1);
        } else if (arg == "--pp-mctx-recon") {
            pp_mctx_recon = atoi(arg_value(argc, argv, i, "--pp-mctx-recon").c_str()) != 0;
        } else if (arg == "--route-telemetry") {
            route_telemetry = atoi(arg_value(argc, argv, i, "--route-telemetry").c_str());
        } else if (arg == "--pp-prebuild-per-pass") {
            pp_prebuild_per_pass = atoi(arg_value(argc, argv, i, "--pp-prebuild-per-pass").c_str());
        } else if (arg == "--pp-recon") {
            // T-GPLR-35 recon: diagnostics only (stream identity + per-bank
            // copy/compute timestamps). Ordering- and arithmetic-neutral.
            const int v = atoi(arg_value(argc, argv, i, "--pp-recon").c_str());
            if (v != 0 && v != 1) {
                fprintf(stderr, "error: --pp-recon must be 0 (off) or 1 (stream identity + per-bank copy timing)\n");
                return 2;
            }
            pp_recon = (v == 1);
        } else if (arg == "--device") {
            device = atoi(arg_value(argc, argv, i, "--device").c_str());
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (!check_path.empty()) {
        // host-side validation mode: no model, no CUDA
        if (validate_mode) {
            return validate_telemetry(check_path);
        }
        return check_telemetry(check_path);
    }

    if (!inspect_path.empty()) {
        // T-GPLR-00: host-only inventory. Runs before llama_backend_init() and
        // any CUDA device enumeration; parses GGUF metadata with no_alloc.
        if (ctx_size <= 0) {
            fprintf(stderr, "error: --ctx-size must be > 0 for --inspect-model\n");
            return 2;
        }
        if (kv_cache != "f16" && kv_cache != "q8" && kv_cache != "q4") {
            fprintf(stderr, "error: --kv-cache must be 'f16', 'q8', or 'q4' (got '%s')\n", kv_cache.c_str());
            return 2;
        }
        ggml_type kv_t = GGML_TYPE_F16;
        if (kv_cache == "q8") {
            kv_t = GGML_TYPE_Q8_0;
        } else if (kv_cache == "q4") {
            kv_t = GGML_TYPE_Q4_0;
        }
        const llama_ls_inventory inv = llama_ls_model_inventory(
            inspect_path.c_str(), ctx_size, kv_t, physical_vram_bytes, physical_vram_source.c_str());
        if (!inv.ok) {
            fprintf(stderr, "inspect-model: error: %s\n", inv.error.c_str());
            return 1;
        }
        if (inspect_json_path.empty()) {
            fwrite(inv.json.data(), 1, inv.json.size(), stdout);
        } else {
            FILE * out = fopen(inspect_json_path.c_str(), "wb");
            if (out == nullptr) {
                fprintf(stderr, "inspect-model: cannot write %s\n", inspect_json_path.c_str());
                return 2;
            }
            fwrite(inv.json.data(), 1, inv.json.size(), out);
            fclose(out);
            fprintf(stderr, "inspect-model: wrote %s\n", inspect_json_path.c_str());
        }
        return 0;
    }

    // T-GPLR-07: --prompt-file reads the prompt text from a file (Windows
    // argv/env limits are ~32 KB; a 200k-token prompt is ~800 KB of text, so
    // file input is required). The bytes feed the exact same prompt_text
    // pipeline as --prompt/--prompt-env (tokenize, pp_tokens override, MTP
    // padding); combining sources is a loud error, and the absent-flag path
    // (--prompt/--prompt-env last-wins) is unchanged. All failures are
    // host-only and fire before llama_backend_init.
    if (!prompt_file.empty()) {
        if (!prompt_text.empty()) {
            fprintf(stderr, "error: --prompt-file cannot be combined with --prompt or --prompt-env\n");
            return 2;
        }
        FILE * pf = fopen(prompt_file.c_str(), "rb");
        if (pf == nullptr) {
            fprintf(stderr, "error: cannot open prompt file %s\n", prompt_file.c_str());
            return 2;
        }
        fseek(pf, 0, SEEK_END);
        const long sz = ftell(pf);
        fseek(pf, 0, SEEK_SET);
        if (sz < 0) {
            fclose(pf);
            fprintf(stderr, "error: cannot determine the size of prompt file %s\n", prompt_file.c_str());
            return 2;
        }
        std::string buf;
        buf.resize((size_t) sz);
        const size_t got = fread(&buf[0], 1, (size_t) sz, pf);
        const bool read_ok = (got == (size_t) sz);
        fclose(pf);
        if (!read_ok) {
            fprintf(stderr, "error: failed reading prompt file %s\n", prompt_file.c_str());
            return 2;
        }
        if (buf.empty()) {
            fprintf(stderr, "error: prompt file %s is empty\n", prompt_file.c_str());
            return 2;
        }
        prompt_text = std::move(buf);
        fprintf(stderr, "prompt-file: read %zu bytes from %s\n", prompt_text.size(), prompt_file.c_str());
    }

    if (model_path.empty()) {
        usage(argv[0]);
        return 2;
    }
    if (n_slots < 1 || n_slots > 16) {
        fprintf(stderr, "error: --layer-slots must be in 1..16\n");
        return 2;
    }
    if (pp_tokens <= 0 || ubatch <= 0) {
        fprintf(stderr, "error: --pp-tokens and --ubatch must be > 0\n");
        return 2;
    }
    if (pp_repeat < 1 || pp_repeat > 16) {
        fprintf(stderr, "error: --pp-repeat must be in 1..16\n");
        return 2;
    }
    if (pp_repeat > 1 && mtp) {
        fprintf(stderr, "error: --pp-repeat > 1 is incompatible with --mtp (the draft-head mirror state spans exactly one prefill)\n");
        return 1;
    }
    // T-GPLR-48 step 1 (ordering): the KV-conditional default below READS
    // kv_cache, so its validation is hoisted to here, above the resolution. In
    // the pre-change tree the main-path check sat at the copy_source block
    // (former line 3095, AFTER the resolution), so "already validated" was only
    // true for the --inspect-model path; the inspect-path check stays where it
    // is (2888). The accepted value set is unchanged (f16 | q8 | q4); the only
    // observable difference is diagnostic order for a command that is invalid
    // in two ways at once (an invalid --kv-cache now reports before a later
    // invalid flag instead of after).
    if (kv_cache != "f16" && kv_cache != "q8" && kv_cache != "q4") {
        fprintf(stderr, "error: --kv-cache must be 'f16', 'q8', or 'q4' (got '%s')\n", kv_cache.c_str());
        return 2;
    }
    // T-GPLR-43: the per-pass rebuild rides the --pp-repeat layer-major pass
    // loop; MTP uses a different (per-ubatch) prefill path that never reaches it.
    // -1 = not explicitly set.
    // T-GPLR-48 (owner decision from the T-GPLR-46 verdict): the default was
    // made KV-CONDITIONAL. The rebuild's original justification was the
    // quantized-KV pass-2 crash it fixes; at f16 it measured as a pure cost -
    // a quadratic pass-2 compute excess and a TG answer->echo at 4k layer-major
    // (T-GPLR-46), and the echo needs the pw schedule AND the rebuild together
    // (T-GPLR-49b), so defaulting it off for f16 removed that defect from the
    // default path while leaving the crash fix exactly where it is load-bearing.
    // Explicit 0/1 still overrides everything, so every recorded arm command is
    // reproducible. kv_cache is the runner's --kv-cache string (declared at
    // 2569, assigned in the option loop, validated immediately above) - the
    // single source of truth, so no second KV-type inference path is introduced.
    // T-GPLR-50 (owner decision 2026-09-21): the f16 branch flips back to ON.
    // The T-48 rationale did not survive its own evidence: the "pure cost" was
    // the PRE-T-47 in-place rebuild, which also caused the pass-1v2 route-hash
    // anomaly and the pw+rebuild echo; T-47's fresh-context swap-in-slot
    // rebuild is compute-neutral at the charter shape (200k excess +0.10 s ON
    // vs +67.4 s OFF, recorded ledger pair in the T-GPLR-50 draft), and the
    // echo trigger is re-validated on the current tree before this default
    // ships (T-GPLR-50 AC2). Explicit 0/1 still overrides everything.
    // T-GPLR-39: the route diagnostic defaults OFF (owner decision). Resolution
    // sits AFTER the flag-parse and needs no other input, so it is placed beside
    // the other default resolutions. Explicit 0/1 always overrides.
    if (route_telemetry != -1 && route_telemetry != 0 && route_telemetry != 1) {
        fprintf(stderr, "error: --route-telemetry must be 0 or 1 (got '%d')\n", route_telemetry);
        return 2;
    }
    const bool route_telemetry_explicit = (route_telemetry != -1);
    if (!route_telemetry_explicit) {
        route_telemetry = 0;
        fprintf(stderr, "preflight: route-telemetry = 0 (default since T-GPLR-39: the route event is a diagnostic whose device readback + host sort cost 1161.5 ms/pass at u2048; pass --route-telemetry 1 to record expert routing)\n");
    }
    const bool pp_prebuild_per_pass_explicit = (pp_prebuild_per_pass != -1);
    if (!pp_prebuild_per_pass_explicit) {
        pp_prebuild_per_pass = (pp_repeat > 1) ? 1 : 0;
        // Self-describing runs: the announcement is emitted ONLY when the value
        // came from the default, so an explicit arm is distinguishable from a
        // defaulted one by the log alone (T-GPLR-48 AC5).
        if (pp_prebuild_per_pass) {
            fprintf(stderr, "layer-major preflight: pp-prebuild-per-pass = 1 (default since T-GPLR-50: %s KV, pass-2 rebuild active - fresh-context swap-in-slot, compute-neutral per T-GPLR-47)\n",
                    kv_cache.c_str());
        } else if (pp_repeat > 1) {
            // Unreachable via the default after T-GPLR-50 (default gives 1 for
            // every repeat > 1); kept as the invariant net.
            fprintf(stderr, "layer-major preflight: pp-prebuild-per-pass = 0 (invariant net: repeat > 1 with rebuild off)\n");
        } else {
            fprintf(stderr, "layer-major preflight: pp-prebuild-per-pass = 0 (default: --pp-repeat 1, there is no pass boundary)\n");
        }
    }
    if (pp_prebuild_per_pass && mtp) {
        fprintf(stderr, "error: --pp-prebuild-per-pass is incompatible with --mtp\n");
        return 1;
    }
    if (pp_prebuild_per_pass && pp_repeat <= 1) {
        fprintf(stderr, "error: --pp-prebuild-per-pass requires --pp-repeat > 1 (there is no pass boundary to rebuild at)\n");
        return 2;
    }
    if (pp_prebuild_per_pass < 0 || pp_prebuild_per_pass > 1) {
        fprintf(stderr, "error: --pp-prebuild-per-pass must be 0 or 1\n");
        return 2;
    }
    // T-GPLR-56 A3: the paced mode's preconditions are refused HERE, before the
    // model load, so a misconfigured arm fails in seconds and is observable with
    // no GPU and no trace (the T-GPLR-53 prewarm-flag lesson: a refusal that can
    // only be reached after the load is indistinguishable from a run that
    // silently took the default path). These are FLAG-level facts, so nothing is
    // measured by deferring them; the engine keeps its own checks as the second
    // line of defence.
    if (pp_span_pace > 0 && copy_source != "mmap-registered") {
        fprintf(stderr, "error: --pp-span-pace %d requires --copy-source mmap-registered; got %s\n",
                pp_span_pace, copy_source.c_str());
        return 2;
    }
    if (pp_span_pace > 0 && pp_register_site == 1) {
        fprintf(stderr, "error: --pp-span-pace 1 cannot be combined with --pp-register-site early "
                        "(the paced groups emit events that must follow run_start, and the early site precedes it); run one arm or the other\n");
        return 2;
    }
    if (pp_span_pace == 0 && pp_span_pace_ahead != 1) {
        fprintf(stderr, "error: --pp-span-pace-ahead %d requires --pp-span-pace 1 (nothing to pace)\n", pp_span_pace_ahead);
        return 2;
    }
    if (telemetry_chunk_stride <= 0 || telemetry_chunk_stride > 1000000) {
        fprintf(stderr, "error: --telemetry-chunk-stride must be in 1..1000000\n");
        return 2;
    }
    if (pp_prefetch_ahead < 0 || pp_prefetch_ahead > 4) {
        fprintf(stderr, "error: --pp-prefetch-ahead must be in 0..4 (0 = off; the T-GPLR-08 probe runs N=1)\n");
        return 2;
    }
    if (pp_pool_n < 0) {
        fprintf(stderr, "error: --pp-pool must be a non-negative bank count (got %d)\n", pp_pool_n);
        return 1;
    }
    if (pp_pool_n > 0 && pp_pool_policy != "fixed" && pp_pool_policy != "lru") {
        fprintf(stderr, "error: --pp-pool policy must be 'fixed' or 'lru' (got '%s')\n", pp_pool_policy.c_str());
        return 1;
    }
    if (pp_async_feed < 0 || pp_async_feed > 1) {
        fprintf(stderr, "error: --pp-async-feed must be 0 (off) or 1 (one-layer-ahead worker feed; requires --layer-slots >= 2)\n");
        return 2;
    }
    if (pp_async_feed > 0 && n_slots < 2) {
        fprintf(stderr, "error: --pp-async-feed requires --layer-slots >= 2 (with one slot the reuse invariant forbids staging the next layer while the current one computes)\n");
        return 2;
    }
    if (graph_budget_mib == 0 || graph_budget_mib < -1) {
        fprintf(stderr, "error: --graph-budget-mib must be a positive integer (MiB)\n");
        return 2;
    }
    // PP schedule selector (SDD section 13, T-GPLR-12): the default is
    // layer-major + per-width graphs; batch-major is the opt-in alternative
    // (--pp-schedule batch-major, or a legacy pinned harness script). Note
    // `--residency off` (the normal llama.cpp path) rejects only an EXPLICIT
    // layer-major request; the default never blocks the off path because a
    // layer-major PP cannot run without the streaming module. Validate before
    // any CUDA work so an unknown value fails before model execution.
    llama_pp_schedule schedule = LLAMA_PP_SCHEDULE_BATCH_MAJOR;
    if (pp_schedule == "batch-major") {
        schedule = LLAMA_PP_SCHEDULE_BATCH_MAJOR;
    } else if (pp_schedule == "layer-major") {
        schedule = LLAMA_PP_SCHEDULE_LAYER_MAJOR;
    } else {
        fprintf(stderr, "error: --pp-schedule must be 'batch-major' or 'layer-major' (got '%s')\n", pp_schedule.c_str());
        return 2;
    }
    if (schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR) {
        // T-Q38CTX-01: any positive --pp-tokens is legal; the layout is full
        // ubatch-sized chunks plus one real 1..ubatch-1 token tail (no fake
        // padding). The per-run chunk manifest is recorded in run_start and
        // cross-checked by --check-telemetry.
        // The layer-major PP is driven by the dedicated controller (T-Q38LM-02);
        // a following TG uses the standard one-token TG-u1 prebuilt set (the same
        // llama_decode path as batch-major), so --tg-tokens is allowed. The
        // PP->TG handoff state invariants are validated by the host checker.
        // T-GPLR-10: the cap lifts when --handoff is active -- post-handoff
        // TG is stock llama_decode regardless of the PP schedule, so longer TG
        // budgets are legal there; the cap remains for layer-major without
        // handoff.
            // T-GPLR-12: the cap binds only an explicitly handoff-free request.
        // auto passes here (its fit decision happens at the capacity
        // preflight: auto-on lifts the cap, auto-off re-enforces it loudly
        // before the timed PP), and explicit --handoff lifts it as in
        // T-GPLR-10.
        if (tg_tokens > 3 && handoff_mode_str == "off") {
            fprintf(stderr, "error: --pp-schedule layer-major supports at most 3 --tg-tokens in this experiment\n");
            return 2;
        }
    }
    if (copy_source != "direct" && copy_source != "pinned" && copy_source != "mmap-registered" &&
        copy_source != "mmap-registered-churn") {
        fprintf(stderr, "error: --copy-source must be 'direct', 'pinned', 'mmap-registered' or "
                        "'mmap-registered-churn' (got '%s')\n", copy_source.c_str());
        return 2;
    }
    // T-GPLR-37: --pp-pool IS a pinned staging structure. Under the
    // mmap-registered default a bare --pp-pool would only fail at the engine's
    // pool configure (after model load); refuse here, before any GPU work,
    // with the one-flag fix. An explicit --copy-source pinned keeps the old
    // combined command working verbatim (the only recorded consumer already
    // pins it, so this only bites genuinely new commands).
    if (pp_pool_n > 0 && copy_source != "pinned") {
        fprintf(stderr, "error: --pp-pool requires --copy-source pinned (the pool IS a pinned staging "
                        "structure; T-GPLR-37 made mmap-registered the default - pass --copy-source pinned explicitly)\n");
        return 2;
    }
    // T-GPLR-37: self-describing runs - announce ONLY when the source came from
    // the default, so an explicit arm is distinguishable from a defaulted one
    // by the log alone (same rule as the T-GPLR-48 rebuild announcement).
    if (!copy_source_explicit) {
        fprintf(stderr, "preflight: copy-source = %s (default since T-GPLR-37: register-once mmap staging, "
                        "~17 GiB of pinned file pages for the run; --copy-source pinned restores the staging-buffer path)\n",
                copy_source.c_str());
    }

    // T-GPLR-12: pw-probe is a layer-major default; the cross-check fires
    // only when the user explicitly set the schedule to batch-major (a
    // default-driven conflict would be a self-inflicted refusal).
    if (pw_probe && !pw_probe_explicit && schedule != LLAMA_PP_SCHEDULE_LAYER_MAJOR && !layer_major_explicit) {
        pw_probe = false; // user pinned batch-major; drop the layer-major-only default quietly
    }
    if (pw_probe && schedule != LLAMA_PP_SCHEDULE_LAYER_MAJOR) {
        fprintf(stderr, "error: --pw-probe requires --pp-schedule layer-major\n");
        return 2;
    }

    // T-Q38HG-06: MTP needs the draft head's weights resident. The exact
    // policy check runs below, after auto resolution knows the final policy
    // (T-GPLR-06: moe-expert-bank keeps the MTP package resident without
    // any handoff migration).
    if (mtp && mtp_draft < 1) {
        fprintf(stderr, "error: --mtp-draft must be >= 1\n");
        return 2;
    }
    if (mtp && tg_tokens <= 0) {
        fprintf(stderr, "error: --mtp requires --tg-tokens > 0\n");
        return 2;
    }
    // T-GPLR-06 step 2: batch verify decodes 1 + mtp_draft rows (frontier +
    // drafts) in one ubatch, so the draft chain must fit the context's
    // ubatch. Capping is the deliberate, minimal contract (a config that
    // cannot run batched fails loudly instead of silently starving the
    // amortization).
    if (mtp && ubatch < 1 + mtp_draft) {
        fprintf(stderr, "error: --mtp-draft %d requires --ubatch >= %d (batch verify decodes the frontier plus all drafts in one ubatch)\n",
                mtp_draft, 1 + mtp_draft);
        return 2;
    }
    // T-Q38HG-06: the MTP prompt catch-up mirrors each PP ubatch into the
    // draft context, so the runner drives PP itself (batch-major ubatches).
    // The layer-major controller has no per-ubatch hook; reject the combo.
    if (mtp && schedule != LLAMA_PP_SCHEDULE_BATCH_MAJOR) {
        fprintf(stderr, "error: --mtp requires --pp-schedule batch-major\n");
        return 2;
    }
    if (mtp && prompt_text.empty() && pp_tokens % ubatch != 0) {
        fprintf(stderr, "error: --mtp requires pp_tokens %% ubatch == 0 (prompt ubatch mirrors must be exact)");
        return 2;
    }

    // T-GPLR-05: the capability probe is strictly host-only (GGUF header KV
    // only), so it must run before llama_backend_init to guarantee zero CUDA
    // enumeration and zero allocation.
    if (probe_only) {
        llama_ls_capability cap;
        const std::string rec = llama_ls_probe_capability(model_path.c_str(), cap);
        printf("capability: {\"arch\":\"%s\",\"supported\":%s,\"adapter\":\"%s\",\"policy\":\"%s\",\"reason\":\"%s\",\"recommended\":\"%s\"}\n",
               cap.arch.c_str(), cap.supported ? "true" : "false", cap.adapter.c_str(),
               cap.policy.c_str(), cap.reason.c_str(), rec.c_str());
        return 0;
    }

    // T-GPLR-05 (post-audit): the off-compatibility contract lives in ONE
    // table. Each entry pairs an argv-decided condition with its stable
    // rejection reason; every violation exits 2. Adding a streaming-only flag
    // means adding one entry here — a hand-written comparison elsewhere can
    // no longer bypass the gate. Forced `off` is decidable from argv alone,
    // so its gate fires below, before the host-only model-arch read, so a
    // gate rejection with exit 2 wins over any later failure class (the
    // arch-read path previously leaked exit 1 for the same violation).
    struct ls_off_conflict { bool hit; const char * reason; };
    const ls_off_conflict off_conflicts[] = {
        // T-GPLR-12: only an EXPLICIT layer-major request refuses `off`; the
        // default schedule must keep the normal llama.cpp path reachable.
        { layer_major_explicit,
          "--residency off is incompatible with --pp-schedule layer-major (off is the normal batch-major llama.cpp path)" },
        { mtp,
          "--residency off is incompatible with --mtp (MTP speculative decode requires the streaming module)" },
        // T-GPLR-12: only an explicit handoff request (--handoff/--handoff-mode
        // on) refuses `off`; the auto default must degrade to auto-off with a
        // notice instead (condition (a): auto enables handoff only for a
        // streamable policy).
        { handoff_mode_str == "on",
          "--residency off is incompatible with --handoff (handoff TG is a streaming-module mode)" },
        { handoff_partial_n >= 0,
          "--residency off is incompatible with --handoff-partial (partial handoff TG is a streaming-module mode)" },
        // T-GPLR-13: row-set TG is a streaming-module mode like the others.
        { residency_tg,
          "--residency off is incompatible with --residency-tg (moe-row-set TG is a streaming-module mode)" },
        // T-GPLR-58 A0: speculative placement is likewise a streaming-module mode.
        { residency_tg_spec != 0,
          "--residency off is incompatible with --residency-tg-spec (speculative placement is a streaming-module mode)" },
        { residency_tg_spec_pool != 0,
          "--residency off is incompatible with --residency-tg-spec-pool (the prediction pool is a streaming-module mode)" },
        { residency_tg_retain > 0,
          "--residency off is incompatible with --residency-tg-retain (per-layer row residency is a streaming-module mode)" },
        // T-GPLR-12: pw-probe is a layer-major default now; only an explicit
        // --pw-probe signals conflict intent on the off path (the off path
        // never uses the probe, so a default-driven entry would refuse the
        // plain `--residency off` shape that used to pass).
        { pw_probe && pw_probe_explicit,
          "--residency off is incompatible with --pw-probe (per-width graph probes require the streaming module)" },
        { pp_prefetch_ahead > 0,
          "--residency off is incompatible with --pp-prefetch-ahead (the prefetch probe is a streaming-module host-feed experiment)" },
        { pp_prewarm > 0,
          "--residency off is incompatible with --pp-prewarm (the pre-warm modes belong to the layer-major staging experiment)" },
        { pp_async_feed > 0,
          "--residency off is incompatible with --pp-async-feed (the async-feed probe is a streaming-module copy-overlap experiment)" },
        { layer_slots_explicit,
          "--residency off is incompatible with --layer-slots (no slot window on the normal path)" },
        { !tg_text_path.empty(),
          "--residency off is incompatible with --tg-text (generated-text file output requires the streaming module)" },
        { residency_budget_mib > 0 || reserve_mib != 512 || allow_full_residency || graph_budget_mib >= 0,
          "--residency off is incompatible with --residency-budget-mib/--reserve-mib/--allow-full-residency/--graph-budget-mib (no streaming allocations to budget)" },
    };
    const char * off_conflict = nullptr;
    for (const ls_off_conflict & c : off_conflicts) {
        if (c.hit) { off_conflict = c.reason; break; }
    }
    if (residency == "off" && off_conflict) {
        fprintf(stderr, "error: %s\n", off_conflict);
        return 2;
    }

    // T-GPLR-53 A4 dose sweep: --pp-prewarm-delay-ms is a no-op unless the mode-4
    // no-touch delay is engaged, so it is refused here (pre-model-load, host-only,
    // like the policy refusals above) rather than silently ignored. The default
    // 1700 ms stays the compiled control duration, so an un-flagged run is
    // unchanged.
    if (pp_prewarm_delay_ms != 1700 && !(pp_prewarm & 4)) {
        fprintf(stderr, "error: --pp-prewarm-delay-ms requires --pp-prewarm with bit 4 set (the no-touch delay modality); got --pp-prewarm %d\n", pp_prewarm);
        return 2;
    }

    // T-GPLR-05: llama_backend_init (CUDA enumeration/allocation) is deferred
    // until AFTER the capability probe and policy refusals, so unsupported
    // models and forced-policy refusals stay strictly host-only.

    // T-GPLR-02: resolve the residency policy from the model's architecture
    // key (host-only metadata read) BEFORE model load, because the policy
    // determines which tensor class becomes the pageable CPU override.
    const std::string model_arch = llama_ls_model_arch(model_path.c_str());
    if (model_arch.empty()) {
        fprintf(stderr, "error: failed to read general.architecture from %s\n", model_path.c_str());
        return 1;
    }
    if (residency != "auto" && residency != "off" && residency != "dense-layer" && residency != "moe-expert-bank") {
        fprintf(stderr, "error: --residency must be auto|off|dense-layer|moe-expert-bank (got '%s')\n", residency.c_str());
        return 2;
    }

    // T-GPLR-05: resolve the residency policy through the capability probe so
    // `auto` can only ever select an adapter-declared policy (never a guessed
    // dense-layer for an unknown arch), and so the refusal reason is stable.
    llama_ls_capability cap;
    const std::string recommended = llama_ls_probe_capability(model_path.c_str(), cap);
    if (model_arch.empty()) {
        fprintf(stderr, "error: %s\n", cap.reason.c_str());
        return 1;
    }
    if (residency == "auto") {
        if (cap.supported) {
            residency = recommended;
        } else {
            // SDD section 8: insufficient evidence selects `off` with the
            // capability reason; never silently pick a guessed policy.
            fprintf(stderr, "residency: auto -> off (%s); running normal llama.cpp placement\n", cap.reason.c_str());
            forced_off = true;
        }
    } else if (residency == "off") {
        // T-GPLR-05: forced off runs the NORMAL llama.cpp path inside the
        // runner (no streaming init, no CUDA placement overrides). The model
        // params for this branch are set below so the normal path is
        // preserved exactly; telemetry records the capability result.
        forced_off = true;
    }
    // T-GPLR-05 follow-up: `off` executes the stock execution path, which is
    // batch-major single-sequence only. Every streaming-only mode is invalid
    // with it and must be rejected loudly instead of being ignored.
    if (forced_off && off_conflict) {
        // auto -> off resolved above (residency == "off" already returned at
        // the early gate): the same single-owner table rejects with exit 2 so
        // both off entry points enforce one contract.
        fprintf(stderr, "error: %s\n", off_conflict);
        return 2;
    }
    // T-Q38HG-06 / T-GPLR-12: --mtp + handoff under moe-expert-bank is
    // refused loudly: handoff TG is stock llama.cpp decode with no draft
    // loop, and the MoE MTP verify path requires the resident streaming
    // executor. Fires only on an explicit handoff-on request (auto can
    // still legally end at streaming MTP when the fit fails, which is the
    // T-GPLR-06 MoE streaming-MTP shape). Dense --mtp + --handoff stays
    // legal (T-Q38HG-06). Placed before the arch-match gates so the refusal
    // is decidable from the resolved policy alone (host-only, exit 2 before
    // llama_backend_init).
    if (!forced_off && mtp && handoff_mode_str == "on" && residency == "moe-expert-bank") {
        fprintf(stderr, "error: --mtp + --handoff is not supported under residency moe-expert-bank (handoff TG is stock llama.cpp decode without a draft loop; MoE MTP verify requires the streaming executor)\n");
        return 2;
    }
    // T-GPLR-16: partial handoff is an moe-expert-bank-only mode (it splits
    // the routed-expert bank class); dense-layer has no bank packages.
    if (!forced_off && handoff_partial_n >= 0 && residency != "moe-expert-bank") {
        fprintf(stderr, "error: --handoff-partial requires --residency moe-expert-bank (the partial split applies to routed-expert banks; got '%s')\n", residency.c_str());
        return 2;
    }
    // T-GPLR-13: --residency-tg is an moe-expert-bank-only TG mode (it pages
    // ROWS of the routed-expert bank class; dense-layer has no bank packages).
    if (!forced_off && residency_tg && residency != "moe-expert-bank") {
        fprintf(stderr, "error: --residency-tg requires --residency moe-expert-bank (row-set pages routed-expert rows; got '%s')\n", residency.c_str());
        return 2;
    }
    if (!forced_off && residency_tg && tg_tokens <= 0) {
        fprintf(stderr, "error: --residency-tg requires --tg-tokens > 0 (row-set staging exists only in the TG phase)\n");
        return 2;
    }
    if (!forced_off && residency_tg && (handoff_mode_str == "on" || handoff_partial_n >= 0)) {
        fprintf(stderr, "error: --residency-tg is incompatible with --handoff/--handoff-partial (row-set TG stages rows reactively instead of migrating whole packages)\n");
        return 2;
    }
    // T-GPLR-53 A4: the pre-warm targets the layer-major package source spans,
    // so a non-zero mode outside that path is refused loudly. Off (0) is the
    // default and is admitted everywhere.
    if (!forced_off && pp_prewarm > 0 && pp_schedule != "layer-major") {
        fprintf(stderr, "error: --pp-prewarm requires --pp-schedule layer-major (the pre-warm touches the layer-major package spans)\n");
        return 2;
    }
    if (!forced_off && residency_tg_retain > 0 && !residency_tg) {
        fprintf(stderr, "error: --residency-tg-retain requires --residency-tg (the per-layer cache pages the same routed rows the reactive path stages)\n");
        return 2;
    }
    if (!forced_off && residency_tg_spec != 0 && residency_tg_retain <= 0) {
        fprintf(stderr, "error: --residency-tg-spec requires --residency-tg-retain "
                        "(speculation places guessed rows in the PER-LAYER cache; the reactive slot path has no per-layer residency to speculate into)\n");
        return 2;
    }
    if (!forced_off && residency_tg_spec == 2 && residency_tg_spec_arm < 0) {
        fprintf(stderr, "error: --residency-tg-spec 2 needs --residency-tg-spec-arm <coact|bar09|fusion|d1> "
                        "(a mode-2 run with no proposer is refused rather than run: an arm that proposes "
                        "nothing on purpose and an arm that was never configured must not look alike)\n");
        return 2;
    }
    if (!forced_off && residency_tg_spec == 2 && !residency_tg_spec_script.empty()) {
        fprintf(stderr, "error: --residency-tg-spec 2 takes its proposals from the arm; drop "
                        "--residency-tg-spec-script <path> (that is mode 1's source)\n");
        return 2;
    }
    if (!forced_off && residency_tg_spec_pool != 0 && residency_tg_spec != 2) {
        fprintf(stderr, "error: --residency-tg-spec-pool requires --residency-tg-spec 2 (the pool uses the engine-consumed arm)\n");
        return 2;
    }
    if (!forced_off && residency_tg_spec_pool != 0 && residency_tg_spec_pool != 12) {
        fprintf(stderr, "error: --residency-tg-spec-pool first arm requires capacity 12 (got %d)\n", residency_tg_spec_pool);
        return 2;
    }
    if (!forced_off && residency_tg_spec_pool != 0 && residency_tg_spec_budget != 0 && residency_tg_spec_budget != 12) {
        fprintf(stderr, "error: --residency-tg-spec-pool first arm requires --residency-tg-spec-budget 12 (got %d)\n", residency_tg_spec_budget);
        return 2;
    }
    if (!forced_off && residency_tg_spec_pool != 0 && residency_tg_spec_budget == 0) {
        residency_tg_spec_budget = 12;
    }
    if (!forced_off && residency_tg_spec_arm == 3 && !mtp) {
        fprintf(stderr, "error: --residency-tg-spec-arm d1 requires --mtp (D1 is keyed by the token proposed by the MTP draft head)\n");
        return 2;
    }
    if (!forced_off && residency_tg_spec_arm == 3 && residency_tg_spec_pool != 0) {
        fprintf(stderr, "error: --residency-tg-spec-arm d1 currently uses the real per-layer cache; drop --residency-tg-spec-pool\n");
        return 2;
    }
    if (!forced_off && mtp && residency_tg && residency_tg_spec_arm == 3 && residency_tg_retain < 2 + mtp_draft) {
        fprintf(stderr, "error: MTP D1 --residency-tg requires --residency-tg-retain >= %d (the %d-row target union plus one free speculative position per layer)\n",
                2 + mtp_draft, 1 + mtp_draft);
        return 2;
    }
    if (!forced_off && residency_tg_spec != 2 && residency_tg_spec_arm >= 0) {
        fprintf(stderr, "error: --residency-tg-spec-arm is only meaningful with --residency-tg-spec 2 "
                        "(mode 1 proposes from its script)\n");
        return 2;
    }
    if (!forced_off && residency_tg_spec != 2 && residency_tg_spec_null != 0) {
        fprintf(stderr, "error: --residency-tg-spec-null is only meaningful with --residency-tg-spec 2 "
                        "(the null is a proposal SOURCE, matched to the arm's count)\n");
        return 2;
    }
    if (!forced_off && residency_tg_spec == 1 && residency_tg_spec_script.empty()) {
        fprintf(stderr, "error: --residency-tg-spec 1 needs --residency-tg-spec-script <path> "
                        "(the A0 driver supplies the guesses; a real proposer is A1's deliverable)\n");
        return 2;
    }
    if (!forced_off && residency_tg_spec == 0 && residency_tg_spec_budget != 0) {
        fprintf(stderr, "error: --residency-tg-spec-budget is only meaningful with --residency-tg-spec "
                        "(got budget %d with the mode off)\n", residency_tg_spec_budget);
        return 2;
    }
    // T-GPLR-16: the split point must name real trunk layers.
    if (!forced_off && handoff_partial_n >= 0) {
        // n_layer is unknown before the inventory; the arch key read above
        // gives us the model family only. Defer the range check to the
        // inventory-backed preflight (below), where per-bank bytes exist.
    }
    // T-Q38HG-06 / T-GPLR-12: --mtp without --handoff is legal only when the
    // residency policy pages just its own package class, leaving the MTP
    // package (blk.40) GPU-resident. moe-expert-bank is that policy (only
    // ffn_*_exps is pageable); dense-layer still needs the explicit handoff
    // migration. The auto handoff mode must not satisfy this gate on its
    // own: auto's fit decision happens later (post-backend), so an explicit
    // --mtp with the default auto refuses here exactly like handoff-off did.
    // Placed before the arch-match gates so the refusal is decidable from
    // the resolved policy alone (same ordering rationale as the gate above).
    if (!forced_off && mtp && handoff_mode_str == "off" && residency != "moe-expert-bank") {
        fprintf(stderr, "error: --mtp requires --handoff (or residency moe-expert-bank, where the MTP package is already GPU-resident)\n");
        return 2;
    }
    if (!forced_off && mtp_explicit && handoff_mode_str == "auto" && residency != "moe-expert-bank") {
        // explicit --mtp with the default handoff auto: refuse with the same
        // shape so an MoE-less mtp invocation cannot silently become a fit-
        // dependent behavior (dense --mtp must pass --handoff explicitly).
        fprintf(stderr, "error: --mtp requires --handoff (or residency moe-expert-bank, where the MTP package is already GPU-resident)\n");
        return 2;
    }
    if (!forced_off && residency == "moe-expert-bank" && model_arch != "qwen35moe") {
        fprintf(stderr, "error: --residency moe-expert-bank requires architecture qwen35moe (got '%s')\n", model_arch.c_str());
        return 2;
    }
    if (!forced_off && residency == "dense-layer" && model_arch != "qwen35") {
        fprintf(stderr, "error: --residency dense-layer requires architecture qwen35 (got '%s')\n", model_arch.c_str());
        return 2;
    }
    // T-GPLR-05: forced policies (dense-layer / moe-expert-bank) run only when
    // the capability probe declares them supported; off runs unconditionally.
    if (!forced_off && !cap.supported) {
        fprintf(stderr, "error: --residency %s: %s\n", residency.c_str(), cap.reason.c_str());
        return 2;
    }
    fprintf(stderr, "residency: %s (architecture %s)%s%s\n", residency.c_str(), model_arch.c_str(),
            forced_off ? " reason: " : "",
            forced_off ? cap.reason.c_str() : "");

    // T-GPLR-12: resolve the handoff tri-state. Explicit on/off are decided
    // here (all gates below consume `handoff`). Auto stays tentative until
    // the capacity preflight measures the real fit against observed free
    // VRAM, so the host gates treat auto as handoff-requested (they fire
    // only on shapes that are wrong regardless of the fit outcome).
    if (handoff_mode_str == "auto") {
        handoff_auto = true;
        handoff = false; // resolved at the preflight
    } else if (handoff_mode_str == "on") {
        handoff = true;
    } else if (handoff_mode_str == "off") {
        handoff = false;
    } else {
        fprintf(stderr, "error: --handoff-mode must be auto|on|off (got '%s')\n", handoff_mode_str.c_str());
        return 2;
    }
    // T-GPLR-13: an explicit --residency-tg request IS the TG policy choice;
    // the T-GPLR-12 auto handoff fit heuristic must not override it (on a
    // fitting shape auto would migrate the banks and begin_row_set would
    // refuse loudly). Row-set targets the not-fitting regime by construction;
    // on a fitting shape it simply stages rows and records the bytes.
    if (!forced_off && residency_tg && handoff_mode_str == "auto") {
        handoff_mode_str = "off";
        handoff_auto = false;
        fprintf(stderr, "residency-tg: handoff auto -> off (row-set TG is the requested TG policy)\n");
    }

    // T-GPLR-16 option (a), REVERTED 2026-09-22: this runner used to force
    // GGML_OP_OFFLOAD_MIN_BATCH=1 under --handoff-partial so CUDA took the
    // host-resident expert banks through the scheduler's expert row-copy. That
    // route deadlocked the device stream on the third TG token (docs/currentstate,
    // 2026-09-12), so --handoff-partial now runs the pre-fix configuration: with
    // host-resident banks the expert FFN falls back to the CPU backend, whose
    // arithmetic differs from CUDA's MMQ and flips the step-1 gold token. That
    // mixed-arithmetic PP->TG result is the accepted, documented cost of option
    // (a). Do not reinstate this override without a new owner call.

    llama_backend_init();

    // Model params: all layers resident on CUDA except the pageable package
    // class, which is overridden to the CPU buffer (host model-copy sources).
    // dense-layer: every blk.* tensor; moe-expert-bank: only the routed-expert
    // bank tensors of blk layers (attention/router/shared/norm stay resident).
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = -1;
    mparams.main_gpu = device;
    // T-Q38HG-06: --mtp loads the NextN/MTP layer (blk.64) so the draft
    // context can build its MTP graph against it.
    mparams.load_mtp = mtp;
    mparams.load_mode = no_mmap ? LLAMA_LOAD_MODE_NONE : LLAMA_LOAD_MODE_AUTO;

    const char * blk_pattern = "blk\\.[0-9]+\\.*";
    const char * exps_pattern = "blk\\.[0-9]+\\.ffn_(gate|up|down|gate_up)_exps\\.weight";
    llama_model_tensor_buft_override overrides[] = {
        { residency == "moe-expert-bank" ? exps_pattern : blk_pattern, ggml_backend_cpu_buffer_type() },
        { nullptr,     nullptr },
    };
    // T-GPLR-05: forced `off` = normal llama.cpp placement: no tensor-buffer
    // overrides, nothing pageable. Every weight stays where the stock loader
    // puts it with n_gpu_layers = -1.
    mparams.tensor_buft_overrides = forced_off ? nullptr : overrides;

        fprintf(stderr, "loading model %s (static+resident on CUDA, pageable %s in host %s)...\n",
            model_path.c_str(),
            residency == "moe-expert-bank" ? "expert banks" : "blk.*",
            no_mmap ? "memory" : "mmap");
    // T-GPLR-56 A1: the load is the only span long enough to hide the registered-span
    // commit, so its begin/end are marked; without them the trace says nothing about how
    // much of the pre-run_start clock span (2.97 s measured) the load actually occupies.
    llama_ls_tel_mark("load_begin", model_path.c_str());
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    llama_ls_tel_mark("load_end", model == nullptr ? "failed" : "ok");
    if (model == nullptr) {
        fprintf(stderr, "error: failed to load model\n");
        return 1;
    }

    const int n_layer = llama_model_n_layer(model);

    if (!boundary_layers.empty()) {
        for (int il : boundary_layers) {
            if (il < 0 || il >= n_layer) {
                fprintf(stderr, "error: --boundary-layers layer %d out of range [0, %d)\n", il, n_layer);
                return 2;
            }
        }
        fprintf(stderr, "boundary mode: executing only layers:");
        for (int il : boundary_layers) {
            fprintf(stderr, " %d", il);
        }
        fprintf(stderr, "\n");
    }

    // Tokenize a real prompt if provided (overrides synthetic ids)
    std::vector<llama_token> prompt_tokens;
    if (!prompt_text.empty()) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_max = (int) prompt_text.size() + 256;
        prompt_tokens.resize(n_max);
        const int n_tok = llama_tokenize(vocab, prompt_text.c_str(), (int) prompt_text.size(),
                                         prompt_tokens.data(), n_max, true, true);
        if (n_tok < 0) {
            prompt_tokens.resize(-n_tok);
            llama_tokenize(vocab, prompt_text.c_str(), (int) prompt_text.size(),
                           prompt_tokens.data(), -n_tok, true, true);
        } else {
            prompt_tokens.resize(n_tok);
        }
        // For MTP, pad to next ubatch multiple (mirror requires exact ubatches)
        if (mtp) {
            const llama_token pad = llama_vocab_pad(vocab);
            while ((int) prompt_tokens.size() % ubatch != 0) {
                prompt_tokens.push_back(pad);
            }
        }
        pp_tokens = (int64_t) prompt_tokens.size();
        fprintf(stderr, "prompt: %d tokens\n", (int) pp_tokens);
    }

    // T-GPLR-06: shared repeat-penalty ring, seeded with the prompt tokens so
    // the very first TG pick is already penalized (matches stock sampler
    // semantics where the prompt is consumed before generation).
    rep_penalty_ring rep_ring(rep_last_n);
    if (rep_penalty != 1.0) {
        for (llama_token t : prompt_tokens) {
            rep_ring.accept(t);
        }
        fprintf(stderr, "rep-penalty %.3f over last %d tokens (prompt-seeded)\n",
                rep_penalty, rep_last_n);
    }

    // context params
    // n_ctx must cover PP + TG positions and be a 256-multiple: the KV cache
    // stores kv_size = n_ctx_seq = GGML_PAD(n_ctx, 256) cells and asserts
    // kv_size % n_pad == 0. We pin n_pad to the same rounded value so the
    // padded n_kv (and therefore every graph shape) is constant for the whole
    // session, letting the prebuild's persistent graphs be reused by every
    // ubatch with zero timed graph construction. (PP-only 512/64 and the 512+128
    // transition all satisfy this; the 512+128 case rounds 640 -> 768.)
    // T-GPLR-04: an explicit --ctx-size forces the session capacity
    // independently of the PP token count (the 262K proof runs a 512-token
    // PP against a 262,144-token KV allocation).
    const int n_ctx_total = run_ctx_size > 0
        ? (int) GGML_PAD(run_ctx_size, 256)
        : ((pp_tokens + tg_tokens + 255) / 256) * 256;
    if (n_ctx_total < (int) pp_tokens + tg_tokens) {
        fprintf(stderr, "error: --ctx-size %lld is smaller than pp+tg (%lld tokens)\n",
                (long long) run_ctx_size, (long long) (pp_tokens + tg_tokens));
        return 2;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = n_ctx_total;
    // T-Q38HG-06: the target emits dense per-position hidden rows (unmasked
    // nextn); with --mtp the row buffer must cover PP + TG positions, so the
    // batch reserve grows to the padded session length.
    cparams.n_batch      = mtp ? n_ctx_total : pp_tokens;
    cparams.n_ubatch     = ubatch;
    cparams.n_seq_max    = 1;
    // n_outputs_max caps the widest decode that requests logits rows: PP
    // (every token when PP logits are on) or TG. With --pp-no-logits the
    // PP decodes request no logits, but TG still does: 1 row per frontier
    // decode and --mtp's batch verify requests logits on the frontier plus
    // every draft (1 + mtp_draft rows) in one ubatch, so the clamp must
    // cover that MTP draft width or output_reserve aborts at the first
    // verify round. Must remain nonzero in all cases for graph
    // reservation.
    cparams.n_outputs_max = pp_no_logits
        ? (mtp ? (uint32_t) (1 + mtp_draft) : 1)
        : (uint32_t) pp_tokens;
    cparams.n_threads       = 16;
    cparams.n_threads_batch = 16;
    // T-Q38HG-06: the target may roll back a partial draft acceptance, which
    // requires recurrent-state snapshots for the Gated DeltaNet layers (the
    // hybrid memory clamps to 0 when the arch lacks rollback support).
    cparams.n_rs_seq = mtp ? (uint32_t) mtp_draft : 0;
    if (kv_cache == "q8") {
        cparams.type_k = GGML_TYPE_Q8_0;
        cparams.type_v = GGML_TYPE_Q8_0;
    } else if (kv_cache == "q4") {
        cparams.type_k = GGML_TYPE_Q4_0;
        cparams.type_v = GGML_TYPE_Q4_0;
    }
    // KV cache graph padding = n_ctx (the rounded total): keeps padded n_kv
    // (and thus every graph shape) constant for the whole session so the
    // persistent graphs built by the prebuild are reused by every ubatch with
    // no timed graph construction. Must divide kv_size (= n_ctx_seq).
    cparams.n_pad = (uint32_t) n_ctx_total;

    // T-GPLR-04 / T-GPLR-19: the ONE VRAM budget model, seeded here with the
    // budget flags and the pre-ls.init device reading. The capacity preflight
    // below adopts the bounded-window estimate into it and the row-cache fit
    // refreshes the reading; both then read its fields, so the two sites cannot
    // report different amounts of already-allocated memory for one device.
    llama_ls_vram_budget vram;
    ls_vram_budget_measure_start(vram, device, reserve_mib, residency_budget_mib);

    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to create context (is VRAM sufficient?)\n");
        return 1;
    }

    // ------------------------------------------------------------------
    // T-GPLR-05: forced `off` — the normal llama.cpp path inside the
    // runner. No streaming init, no capacity preflight, no activation
    // slabs, no graph prebuild, no handoff/MTP. Batch-major llama_decode
    // with the model exactly as the stock loader placed it. The run still
    // emits an audit telemetry record (policy "off" + capability reason)
    // and reports the first greedy ID for parity comparisons.
    // ------------------------------------------------------------------
    if (forced_off) {
        const int rc = run_residency_off(ctx, model, prompt_tokens, pp_tokens, tg_tokens,
                                         pp_no_logits, n_ctx_total, model_arch.c_str(),
                                         cap.reason.c_str(), tel_path.c_str());
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return rc;
    }

    // layer-streaming module
    const char * schedule_name = (schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR) ? "layer-major" : "batch-major";
    const bool use_bf16_slab = (act_slab_dtype_str == "bf16");
    const ggml_type act_slab_dtype = use_bf16_slab ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    const size_t slab_elt = use_bf16_slab ? 2 : 4;
    const size_t act_slab_bytes = (schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR)
        ? (size_t) llama_model_n_embd_inp(model) * (size_t) pp_tokens * 2 * slab_elt
        : 0;

    // layer-major preflight (SDD section 13): report the two-slab activation
    // requirement and the available VRAM before any inference. The slab is
    // allocated by ls.init; a failed allocation aborts loudly here.
    if (schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR) {
        fprintf(stderr, "layer-major preflight: act slabs require %zu bytes (2 x %zu, %s) ; free VRAM %zu, total %zu\n",
                act_slab_bytes, act_slab_bytes / 2, use_bf16_slab ? "bf16" : "f32",
                vram.free_start, vram.total);
    }

    // T-GPLR-04: explicit capacity preflight. After CUDA-context setup (so the
    // free-VRAM reading reflects the real session) but BEFORE the streaming
    // init's large allocations and the timed PP, derive the observed byte
    // classes from the host-only inventory arithmetic, prove the out-of-core
    // condition, check the bounded configuration against the budget, and emit
    // a structured byte_account event. Every failure here is loud: no silent
    // context shrink, no CPU fallback. All byte quantities come from the shared
    // `vram` budget model (see llama_ls_vram_budget), never from a local
    // re-derivation of the same quantity.
    {
        ggml_type pf_kv = GGML_TYPE_F16;
        if (kv_cache == "q8") pf_kv = GGML_TYPE_Q8_0;
        else if (kv_cache == "q4") pf_kv = GGML_TYPE_Q4_0;
        const llama_ls_inventory pf = llama_ls_model_inventory(
            model_path.c_str(), n_ctx_total, pf_kv, vram.total, "cuda-device-total");
        if (!pf.ok) {
            fprintf(stderr, "error: capacity preflight failed: %s\n", pf.error.c_str());
            refuse_before_sink(tel_path, "capacity_preflight",
                               "capacity preflight failed: " + pf.error, 0, 0, "none");
            return 1;
        }
        // T-GPLR-25 (owner decision 2026-09-15): the slot count is a VRAM-fit
        // decision with a transparent ladder. An EXPLICIT --layer-slots is a
        // contract and is used verbatim (a gate run gets the shape it names or
        // fails loudly). The DEFAULT ladder is 3 when the budget admits it,
        // else 2 (the overlap-capable minimum), else a loud refusal -- never a
        // silent drop to 1, because 1 slot forbids copy/compute overlap by the
        // reuse invariant and produces the serialized PP shape the H-D
        // evidence identified. The arithmetic is the preflight's own bounded
        // window: the same static + per-layer + KV + state + slots + reserve
        // terms the byte_account event reports, so the decision and its
        // reason are recoverable from the run record.
        if (!layer_slots_explicit) {
            // need(S) = the preflight's own terms at S slots, re-derived per
            // rung through the SAME helper the adopted window uses (no second
            // arithmetic path). act slabs and reserve are slot-independent.
            auto need = [&](int slots) -> uint64_t {
                llama_ls_bounded_window w;
                if (!ls_bounded_window_for_policy(pf, model_arch.c_str(), slots, w, residency_tg_window)) {
                    return UINT64_MAX;
                }
                return w.bounded + (uint64_t) act_slab_bytes + vram.reserve_bytes;
            };
            const uint64_t n3 = need(3), n2 = need(2);
            if (n3 <= vram.free_start) {
                if (n_slots != 3) {
                    fprintf(stderr, "layer-slots: default 2 -> 3 (T-GPLR-25 owner ladder: budget admits it; "
                                    "bounded 3-slot need %llu B <= free %zu B; +1 slot = %llu B for copy/compute overlap)\n",
                            (unsigned long long) n3, vram.free_start, (unsigned long long) (n3 - n2));
                }
                n_slots = 3;
            } else if (n2 <= vram.free_start) {
                fprintf(stderr, "layer-slots: default 3 refused, running 2 (T-GPLR-25 owner ladder: the 3-slot window does not fit; "
                                "3-slot need %llu B > free %zu B, shortfall %llu B; the 2-slot minimum fits with %zu B spare)\n",
                        (unsigned long long) n3, vram.free_start, (unsigned long long) (n3 - vram.free_start),
                        vram.free_start - n2);
                n_slots = 2;
            } else {
                const uint64_t n1 = need(1);
                fprintf(stderr, "error: the default 2-slot residency window does not fit observed free VRAM "
                                "(need %llu B = bounded 2-slot window %llu B + act slabs %zu B + reserve %llu B; "
                                "free %zu B; shortfall %llu B). 1 slot would%s fit but is NOT taken silently: "
                                "one slot forbids copy/compute overlap (slot-reuse invariant) and reproduces the "
                                "serialized PP staging shape. Pass --layer-slots 1 explicitly to run the "
                                "minimal-residency shape deliberately, or free VRAM / lower --ctx-size / --kv-cache.\n",
                        (unsigned long long) n2, (unsigned long long) (n2 - (uint64_t) act_slab_bytes - vram.reserve_bytes),
                        act_slab_bytes, (unsigned long long) vram.reserve_bytes, vram.free_start,
                        (unsigned long long) (n2 - vram.free_start), n1 <= vram.free_start ? "" : " NOT");
                refuse_before_sink(tel_path, "capacity_preflight",
                                   "default 2-slot residency window exceeds observed free VRAM: need " +
                                   std::to_string(n2) + " B, free " + std::to_string(vram.free_start) +
                                   " B (pass --layer-slots 1 explicitly for the minimal no-overlap shape)",
                                   n2, vram.free_start, "free");
                return 1;
            }
        }

        // Reuse the inventory's own byte classes through the shared helper so
        // this preflight cannot drift from the telemetry emission below.
        llama_ls_bounded_window pf_win;
        if (!ls_bounded_window_for_policy(pf, model_arch.c_str(), n_slots, pf_win, residency_tg_window)) {
            fprintf(stderr, "error: capacity preflight could not parse the inventory byte classes\n");
            refuse_before_sink(tel_path, "capacity_preflight",
                               "capacity preflight could not parse the inventory byte classes", 0, 0, "none");
            return 1;
        }
        // The one budget model owns the bounded window: adopting it here makes
        // the conditions below, the row-cache fit later and the byte_account
        // telemetry all read the same bounded/slot/reserve/budget numbers.
        ls_vram_budget_adopt_window(vram, pf_win);
        const uint64_t full_model_plus_runtime = pf_win.full_model_plus_runtime;

        // The unmeasured runtime classes (activation slabs, graph io, compute
        // workspaces) are reported as evidence, not silently assumed to fit:
        // the act-slab and graph-footprint figures below are real measured
        // allocations from ls.init / the prebuild.
        fprintf(stderr, "capacity preflight (T-GPLR-04):\n");
        fprintf(stderr, "  full model + KV + state : %llu B\n", (unsigned long long) full_model_plus_runtime);
        fprintf(stderr, "  bounded %d-slot window   : %llu B (total residency budget %llu B, reserve %llu B)\n",
                n_slots, (unsigned long long) vram.bounded,
                (unsigned long long) vram.residency_budget_bytes, (unsigned long long) vram.reserve_bytes);
        fprintf(stderr, "  KV (%s @ %d)         : %llu B\n", kv_cache.c_str(), n_ctx_total, (unsigned long long) pf_win.kv_bytes);
        fprintf(stderr, "  recurrent state         : %llu B\n", (unsigned long long) pf_win.state_bytes);
        fprintf(stderr, "  observed free VRAM      : %zu B of %zu B\n", vram.free_start, vram.total);

        // T-GPLR-12: the auto handoff decision (condition (b)), decided
        // FIRST so the out-of-core proof below can honor it. Evaluated with
        // the SAME arithmetic as the explicit T-GPLR-10 fit gate below --
        // no second arithmetic path, always BEFORE the timed PP. Auto
        // enables handoff only for the policy the T-GPLR-10 branch covers
        // (moe-expert-bank) and only when the full post-handoff peak fits
        // observed free VRAM. No fit -> the run continues as streaming TG
        // with a loud notice (never a mid-run fallback); the decision is
        // recorded in run_start as handoff_mode "auto-on" / "auto-off" and
        // audited by the checker.
        if (handoff_auto) {
            const uint64_t handoff_need = full_model_plus_runtime + act_slab_bytes + vram.reserve_bytes;
            if (residency == "moe-expert-bank" && handoff_need <= vram.free_start) {
                handoff = true;
                fprintf(stderr, "handoff: auto -> ON (T-GPLR-10 fit gate passed: %llu B <= observed free %zu B)\n",
                        (unsigned long long) handoff_need, vram.free_start);
            } else {
                handoff = false;
                fprintf(stderr, "handoff: auto -> OFF (%s); running streaming TG (paged expert banks)\n",
                        residency != "moe-expert-bank"
                            ? "policy has no T-GPLR-10 fit branch; explicit --handoff remains available"
                            : "post-handoff peak exceeds observed free VRAM");
            }
        }
        // 1. Out-of-core proof: the contract requires full residency to NOT fit
        //    (below observed free VRAM), unless explicitly overridden.
        //    Note: these pre-init failures report through stderr + exit status;
        //    the telemetry sink does not exist until ls.init below.
        //    T-GPLR-12: an auto run whose fit decision turned handoff ON is
        //    exempt -- the handoff fit gate below is strictly stronger (the
        //    same terms plus the reserve), so the weaker proof is redundant
        //    there; explicit --handoff keeps today's hard-refuse semantics.
        if (full_model_plus_runtime + act_slab_bytes <= vram.free_start && !(handoff_auto && handoff)) {
            const char * msg = "capacity preflight: the full model plus runtime WOULD FIT in free VRAM; "
                               "this is not an out-of-core configuration"
                               " (pass --allow-full-residency to override)";
            fprintf(stderr, "error: %s\n", msg);
            if (!allow_full_residency) {
                refuse_before_sink(tel_path, "capacity_preflight", msg,
                                   full_model_plus_runtime + act_slab_bytes, vram.free_start, "free");
                return 1;
            }
        }
        // 2. The bounded configuration must fit under the explicit budget. The
        //    row cache is capped by the same budget (see ls_vram_budget_derive),
        //    so bounded <= budget here is what keeps cache_cap_budget sane.
        if (vram.bounded > vram.residency_budget_bytes) {
            fprintf(stderr, "error: capacity preflight: bounded %d-slot window %llu B exceeds the residency budget %llu B (aborted before the timed PP)\n",
                    n_slots, (unsigned long long) vram.bounded, (unsigned long long) vram.residency_budget_bytes);
            refuse_before_sink(tel_path, "capacity_preflight",
                               "capacity preflight: bounded " + std::to_string(n_slots) + "-slot window " +
                               std::to_string(vram.bounded) + " B exceeds the residency budget " +
                               std::to_string(vram.residency_budget_bytes) + " B",
                               vram.bounded, vram.residency_budget_bytes, "budget");
            return 1;
        }
        // 3. The bounded configuration plus the measured activation slabs must
        //    leave the required reserve below OBSERVED free VRAM (foreign
        //    processes and CUDA context already subtracted from the reading).
        if (vram.bounded + act_slab_bytes + vram.reserve_bytes > vram.free_start) {
            fprintf(stderr, "error: capacity preflight: bounded configuration %llu B + act slabs %zu B + reserve %llu B exceeds observed free VRAM %zu B (aborted before the timed PP)\n",
                    (unsigned long long) vram.bounded, act_slab_bytes,
                    (unsigned long long) vram.reserve_bytes, vram.free_start);
            refuse_before_sink(tel_path, "capacity_preflight",
                               "capacity preflight: bounded configuration " + std::to_string(vram.bounded) +
                               " B + act slabs " + std::to_string(act_slab_bytes) + " B + reserve " +
                               std::to_string(vram.reserve_bytes) + " B exceeds observed free VRAM " +
                               std::to_string(vram.free_start) + " B",
                               vram.bounded + act_slab_bytes + vram.reserve_bytes, vram.free_start, "free");
            return 1;
        }
        // T-GPLR-10: --handoff under moe-expert-bank migrates the 40 expert
        // banks (18,327,011,328 B) into one device buffer after PP. The
        // peak-order rule releases the slot buffers and activation slabs
        // before that allocation, so the post-PP peak is resident classes +
        // KV/state + handoff buffer, and the fit gate is full_model_plus_runtime
        // + act slabs + reserve <= observed free VRAM. No fit claim is made:
        // the free VRAM measured at run start decides, and a refusal here is
        // loud and structured (the byte table below), before the timed PP --
        // never a silent fallback to paged TG.
        // T-GPLR-16: the full-handoff gate is skipped for a partial request
        // (handoff_partial_n >= 0): the partial gate below replaces it with
        // the split arithmetic (retained banks stay host-resident).
        if (handoff && residency == "moe-expert-bank" && handoff_partial_n < 0) {
            const uint64_t handoff_need = full_model_plus_runtime + act_slab_bytes + vram.reserve_bytes;
            if (handoff_need > vram.free_start) {
                fprintf(stderr,
                    "error: handoff capacity preflight (T-GPLR-10): weights+KV+state %llu B (KV %llu B, state %llu B) + act slabs %zu B + reserve %llu B = %llu B exceeds observed free VRAM %zu B (aborted before the timed PP; no silent fallback)\n",
                    (unsigned long long) full_model_plus_runtime,
                    (unsigned long long) pf_win.kv_bytes,
                    (unsigned long long) pf_win.state_bytes,
                    act_slab_bytes,
                    (unsigned long long) vram.reserve_bytes,
                    (unsigned long long) handoff_need,
                    vram.free_start);
                refuse_before_sink(tel_path, "handoff_preflight",
                                   "handoff capacity preflight (T-GPLR-10): weights+KV+state " +
                                   std::to_string(full_model_plus_runtime) + " B (KV " +
                                   std::to_string(pf_win.kv_bytes) + " B, state " +
                                   std::to_string(pf_win.state_bytes) + " B) + act slabs " +
                                   std::to_string(act_slab_bytes) + " B + reserve " +
                                   std::to_string(vram.reserve_bytes) + " B = " +
                                   std::to_string(handoff_need) + " B exceeds observed free VRAM " +
                                   std::to_string(vram.free_start) + " B",
                                   handoff_need, vram.free_start, "free");
                return 1;
            }
        }
        // T-GPLR-16: partial-handoff fit gate. The post-handoff peak keeps
        // the migrating banks' bytes (il >= N) in VRAM but NOT the banks that
        // stay host-resident, so the need is full_model_plus_runtime minus
        // the retained-bank bytes, plus slabs + reserve. Computed from the
        // inventory's per-layer bank package list (host-only arithmetic,
        // pre-timed-PP; a refusal is loud and structured, never a fallback).
        if (handoff && handoff_partial_n >= 0 && residency == "moe-expert-bank") {
            // Parse the per-layer bank bytes out of the inventory JSON.
            uint64_t retained_bank_bytes = 0;
            int bank_count = 0;
            {
                const std::string tag = "\"moe_expert_bank\":[";
                const size_t p = pf.json.find(tag);
                const size_t pe = p == std::string::npos ? std::string::npos : pf.json.find(']', p);
                if (p == std::string::npos || pe == std::string::npos) {
                    fprintf(stderr, "error: partial handoff preflight: inventory JSON lacks the moe_expert_bank package list\n");
                    refuse_before_sink(tel_path, "handoff_preflight",
                                       "partial handoff preflight: inventory JSON lacks the moe_expert_bank package list",
                                       0, 0, "none");
                    return 1;
                }
                size_t pos = p + tag.size();
                while (pos < (size_t) pe) {
                    const size_t lo = pf.json.find("\"layer\":", pos);
                    if (lo == std::string::npos || lo >= pe) break;
                    const int il = atoi(pf.json.c_str() + lo + strlen("\"layer\":"));
                    const size_t bo = pf.json.find("\"bytes\":", lo);
                    if (bo == std::string::npos || bo >= pe) break;
                    const uint64_t b = strtoull(pf.json.c_str() + bo + strlen("\"bytes\":"), nullptr, 10);
                    if (il >= 0 && il < handoff_partial_n) {
                        retained_bank_bytes += b;
                    }
                    bank_count++;
                    pos = bo;
                }
            }
            if (handoff_partial_n > bank_count) {
                fprintf(stderr, "error: --handoff-partial N=%d exceeds the model's %d trunk layers\n",
                        handoff_partial_n, bank_count);
                return 2;
            }
            if (bank_count > 0 && handoff_partial_n == 0) {
                fprintf(stderr, "error: --handoff-partial 0 leaves no bank resident (use full --handoff instead)\n");
                return 2;
            }
            const uint64_t partial_need = full_model_plus_runtime - retained_bank_bytes + act_slab_bytes + vram.reserve_bytes;
            if (partial_need > vram.free_start) {
                fprintf(stderr,
                    "error: partial handoff capacity preflight (T-GPLR-16): full %llu B - retained banks %llu B (N=%d) + act slabs %zu B + reserve %llu B = %llu B exceeds observed free VRAM %zu B (aborted before the timed PP; no silent fallback)\n",
                    (unsigned long long) full_model_plus_runtime,
                    (unsigned long long) retained_bank_bytes,
                    handoff_partial_n,
                    act_slab_bytes,
                    (unsigned long long) vram.reserve_bytes,
                    (unsigned long long) partial_need,
                    vram.free_start);
                refuse_before_sink(tel_path, "handoff_preflight",
                                   "partial handoff capacity preflight (T-GPLR-16): full " +
                                   std::to_string(full_model_plus_runtime) + " B - retained banks " +
                                   std::to_string(retained_bank_bytes) + " B (N=" +
                                   std::to_string(handoff_partial_n) + ") + act slabs " +
                                   std::to_string(act_slab_bytes) + " B + reserve " +
                                   std::to_string(vram.reserve_bytes) + " B = " +
                                   std::to_string(partial_need) + " B exceeds observed free VRAM " +
                                   std::to_string(vram.free_start) + " B",
                                   partial_need, vram.free_start, "free");
                return 1;
            }
            fprintf(stderr, "partial handoff preflight (T-GPLR-16): first %d banks stay host-resident (%llu B), %d banks migrate; need %llu B <= free %zu B\n",
                    handoff_partial_n,
                    (unsigned long long) retained_bank_bytes,
                    bank_count - handoff_partial_n,
                    (unsigned long long) partial_need,
                    vram.free_start);
        }
        // T-GPLR-12: the layer-major TG cap holds whenever handoff did not
        // actually turn on (auto-off included): streaming layer-major TG
        // beyond 3 tokens is exactly the transfer-bound shape the cap exists
        // for. Refusing here is loud and pre-work (before the timed PP).
        if (schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR && !handoff && tg_tokens > 3) {
            fprintf(stderr, "error: --pp-schedule layer-major supports at most 3 --tg-tokens without an active handoff (handoff auto did not activate; free VRAM or pass --handoff explicitly on a fitting shape)\n");
            return 2;
        }
    }


    llama_layer_streaming ls;
    const bool use_implicit_causal_mask = schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR;
    fprintf(stderr, "layer-major preflight: initializing streaming module\n");
    if (!ls.init(*model, n_slots, device, tel_path.empty() ? nullptr : tel_path.c_str(), ubatch, n_ctx_total,
                 schedule, (int64_t) pp_tokens, copy_source == "pinned", telemetry_chunk_stride, use_implicit_causal_mask, pw_probe,
                 act_slab_dtype, mtp ? 1 + mtp_draft : 1,
                 copy_source == "mmap-registered" || copy_source == "mmap-registered-churn",
                 copy_source == "mmap-registered-churn")) {
        fprintf(stderr, "error: layer streaming init failed (preflight): %s\n", ls.error_msg.c_str());
        // T-GPLR-22: ls.init can fail before OR after it opens the sink, so this
        // is the pre-work shape either way -- the standalone emitter truncates
        // whatever half-written catalog exists and leaves one refusal record.
        refuse_before_sink(tel_path, "init_failed",
                           "layer streaming init failed (preflight): " + ls.error_msg, 0, 0, "none");
        return 1;
    }
    fprintf(stderr, "layer-major preflight: streaming module initialized\n");
    // T-GPLR-13: set BEFORE init/prebuild so the MoE builder creates two-part
    // TG graph sets (attn+router / FFN companion) for the row-set executor.
    ls.row_set_requested = residency_tg;
    // T-GPLR-19 (F2-M1): per-layer row residency is requested here and its
    // private storage allocated below, both BEFORE the prebuild, so the whole
    // run works against a settled configuration and a failed allocation is
    // loud at bring-up rather than mid-generation.
    ls.row_set_retain   = residency_tg_retain > 0;
    ls.row_set_retain_k = residency_tg_retain > 0 ? residency_tg_retain : kRowSetRetainDefault;
    ls.row_set_spec_pool = residency_tg_spec_pool;
    // T-GPLR-62 S2: 0 leaves every existing command line on the full-bank slot.
    ls.row_set_window_width = residency_tg_window;
    ls.row_set_recon = residency_tg_recon;

    // T-GPLR-58 A0: configure speculative placement BEFORE begin_row_cache()
    // allocates its state and long before any TG graph exists, so a bad
    // configuration -- an unreadable script, an expert id outside the model, a
    // budget above the cache's capacity -- is loud at bring-up rather than
    // mid-generation. The refusal goes out as a structured run refusal, like the
    // row-cache refusals below.
    if (residency_tg_spec != 0) {
        if (!ls.set_row_set_window_tokens(mtp ? 1 + mtp_draft : 1)) {
            fprintf(stderr, "error: row-set window configuration refused: %s\n", ls.error_msg.c_str());
            ls.tel_run_refused("rowset_window_config", ls.error_msg.c_str(), 1, 0, 0, "none");
            return 1;
        }
        if (!ls.set_row_set_spec(residency_tg_spec, residency_tg_spec_budget, residency_tg_spec_script)) {
            fprintf(stderr, "error: speculative TG placement refused: %s\n", ls.error_msg.c_str());
            ls.tel_run_refused("tg_spec_config", ls.error_msg.c_str(), 1, 0, 0, "none");
            return 1;
        }
        // MODE 2 only: name the arm and (optionally) switch the proposal SOURCE to
        // the count-matched null. This allocates the arm's learning state, so it
        // must happen before begin_row_cache() -- the same bring-up rule the mode-2
        // branch of set_row_set_spec follows: every refusal is loud, before any
        // graph exists.
        if (residency_tg_spec == 2 &&
            !ls.set_row_set_spec_arm(residency_tg_spec_arm, residency_tg_spec_null != 0)) {
            fprintf(stderr, "error: speculative TG placement refused: %s\n", ls.error_msg.c_str());
            ls.tel_run_refused("tg_spec_config", ls.error_msg.c_str(), 1, 0, 0, "none");
            return 1;
        }
    }
    // T-GPLR-19 (F2-M1): the DEFAULT depth is fitted to the device so a tight
    // device provisions what fits instead of failing at the full ~2.19 GiB. The
    // allowance is the shared VRAM model's cache_budget: whichever binds of the
    // reading refreshed after ls.init (where the static/KV/state classes the
    // preflight's own reading still counted are already allocated), less the
    // operator's headroom and less the slot window this run has not allocated
    // yet -- the cache must never be the allocation that starves the streaming
    // window it exists to serve -- and the remainder an explicit
    // --residency-budget-mib leaves once the bounded residency is counted.
    // An EXPLICIT depth is never reduced: a gate run gets the depth it named or
    // fails loudly, never a silently shallower (b) than it declares -- but it
    // must fit the same allowance, so an explicit depth cannot allocate past a
    // budget the operator set either.
    // T-GPLR-19: the depth the fit accepts is the depth that must allocate.
    // fit_bytes is the prediction for the RESOLVED depth (0 while retention is
    // off), kept here so the allocation can be checked against it below.
    size_t fit_bytes = 0;
    if (ls.row_set_retain) {
        ls_vram_budget_refresh(vram, device);

        size_t want_bytes = 0;
        const int fit = row_cache_fit_depth(ls, ls.row_set_retain_k, vram.cache_budget, &want_bytes, &fit_bytes);
        if (fit < 0) {
            fprintf(stderr, "error: per-layer row-cache footprint could not be derived for --residency-tg-retain K=%d\n", ls.row_set_retain_k);
            ls.tel_run_refused("row_cache_derive",
                               "per-layer row-cache footprint could not be derived",
                               1, 0, vram.cache_budget, "none");
            return 1;
        }
        // Say WHICH cap bound the allowance and which flag lifts it, so a
        // refusal sends the operator to the right knob.
        const bool capped_by_budget = ls_vram_budget_cache_capped_by_budget(vram);
        char cap_note[512];
        if (capped_by_budget) {
            snprintf(cap_note, sizeof(cap_note),
                     "the %lld MiB --residency-budget-mib total leaves %zu B of it for the cache after the %llu B bounded residency",
                     (long long) vram.residency_budget_mib, (size_t) vram.cache_cap_budget, (unsigned long long) vram.bounded);
        } else {
            snprintf(cap_note, sizeof(cap_note),
                     "%zu B of %zu B free, less %llu B held back for --reserve-mib %lld and the %llu B slot window",
                     vram.free_now, vram.total, (unsigned long long) (vram.reserve_bytes + vram.slot_window),
                     (long long) vram.reserve_mib, (unsigned long long) vram.slot_window);
        }
        const char * fix_hint = capped_by_budget ? "--residency-budget-mib (or --reserve-mib)" : "--reserve-mib or free VRAM";
        if (fit == 0) {
            // Nothing fits at any depth: refuse for the default AND for an
            // explicit depth -- there is no smaller depth to fall back to.
            fprintf(stderr, "error: --residency-tg-retain %s depth K=%d needs %zu B but the row cache is allowed only %zu B (%s); not even K=1 fits. Raise %s.\n",
                    residency_tg_retain_explicit ? "explicit" : "default",
                    ls.row_set_retain_k, want_bytes, vram.cache_budget, cap_note, fix_hint);
            ls.tel_run_refused("row_cache_fit", cap_note, 1,
                               (uint64_t) want_bytes, (uint64_t) vram.cache_budget,
                               capped_by_budget ? "budget" : "free");
            return 1;
        }
        if (fit != ls.row_set_retain_k) {
            if (residency_tg_retain_explicit) {
                fprintf(stderr, "error: --residency-tg-retain %d needs %zu B but the row cache is allowed only %zu B (%s); the largest depth that fits is K=%d. An explicit depth is never reduced: raise %s or name a depth that fits.\n",
                        ls.row_set_retain_k, want_bytes, vram.cache_budget, cap_note, fit, fix_hint);
                char depth_detail[768];
                snprintf(depth_detail, sizeof(depth_detail),
                         "explicit depth K=%d needs %zu B but only %zu B is allowed (%s); the largest depth that fits is K=%d",
                         ls.row_set_retain_k, want_bytes, vram.cache_budget, cap_note, fit);
                ls.tel_run_refused("row_cache_depth", depth_detail, 1,
                                   (uint64_t) want_bytes, (uint64_t) vram.cache_budget,
                                   capped_by_budget ? "budget" : "free");
                return 1;
            }
            fprintf(stderr, "row-cache (T-GPLR-19): default depth K=%d needs %zu B but only %zu B is allowed to the cache (%s); provisioning the largest fitting depth K=%d (%zu B). Pass an explicit --residency-tg-retain N to require a depth instead of fitting one.\n",
                    ls.row_set_retain_k, want_bytes, vram.cache_budget, cap_note, fit, fit_bytes);
            ls.row_set_retain_k = fit;
        } else if (!residency_tg_retain_explicit) {
            fprintf(stderr, "row-cache (T-GPLR-19): default depth K=%d fitted to the device (%zu B of a %zu B cache allowance; %s)\n",
                    fit, fit_bytes, vram.cache_budget, cap_note);
        }
    }
    size_t pool_fit_bytes = 0;
    if (ls.row_set_spec_pool > 0) {
        pool_fit_bytes = ls.row_set_spec_pool_bytes_for(ls.row_set_spec_pool);
        if (pool_fit_bytes == 0 || fit_bytes + pool_fit_bytes > vram.cache_budget) {
            fprintf(stderr, "error: prediction pool capacity %d needs %zu B and the total cache allowance is %zu B (real cache %zu B + pool %zu B)\n",
                    ls.row_set_spec_pool, pool_fit_bytes, vram.cache_budget, fit_bytes, pool_fit_bytes);
            ls.tel_run_refused("spec_pool_fit", "real row cache plus prediction pool exceeds the cache allowance",
                               1, (uint64_t) (fit_bytes + pool_fit_bytes), (uint64_t) vram.cache_budget,
                               ls_vram_budget_cache_capped_by_budget(vram) ? "budget" : "free");
            return 1;
        }
    }
    if (ls.row_set_retain && !ls.begin_row_cache()) {
        fprintf(stderr, "error: per-layer row-cache allocation failed: %s\n", ls.error_msg.c_str());
        ls.tel_run_refused("row_cache_alloc", ls.error_msg.c_str(), 1, fit_bytes, vram.cache_budget,
                           ls_vram_budget_cache_capped_by_budget(vram) ? "budget" : "fail:free");
        return 1;
    }
    // T-GPLR-24 (H-B): the row cache is the run's largest allocation that is
    // released before run_end re-samples the device, so it must fold into
    // the free-min tracker HERE, right after it lands.
    if (ls.row_set_retain) {
        ls_vram_budget_refresh(vram, device);
    }
    // T-GPLR-19: prediction and allocation are one arithmetic (the library's
    // per-member cost helper). If they ever disagree, the emitted row_cache
    // figure -- and the total-residency-cap check the checker runs against it
    // -- would describe a configuration the run does not have. Loud, before
    // any timed work, never a silently different cache than the fit promised.
    if (ls.row_set_retain && ls.row_cache_alloc_bytes != fit_bytes) {
        fprintf(stderr, "error: per-layer row-cache allocated %zu B but the fit predicted %zu B for K=%d (prediction and allocation diverged)\n",
                ls.row_cache_alloc_bytes, fit_bytes, ls.row_set_retain_k);
        ls.tel_run_refused("row_cache_divergence",
                           "the allocated cache does not match the fit prediction",
                           1, (uint64_t) fit_bytes, (uint64_t) ls.row_cache_alloc_bytes, "none");
        return 1;
    }
    if (ls.row_set_spec_pool > 0 && ls.row_set_spec_pool_alloc_bytes != pool_fit_bytes) {
        fprintf(stderr, "error: prediction pool allocated %zu B but the fit predicted %zu B (prediction and allocation diverged)\n",
                ls.row_set_spec_pool_alloc_bytes, pool_fit_bytes);
        ls.tel_run_refused("spec_pool_divergence", "the allocated prediction pool does not match its fit prediction",
                           1, (uint64_t) pool_fit_bytes, (uint64_t) ls.row_set_spec_pool_alloc_bytes, "none");
        return 1;
    }
    // T-GPLR-08: configure the one-layer-ahead prefetch probe. Range
    // validation is host-only (VirtualQuery over package members); a failed
    // configure aborts loudly here, before any timed work.
    if (pp_prefetch_ahead > 0 && !ls.set_pp_prefetch_ahead(pp_prefetch_ahead)) {
        fprintf(stderr, "error: prefetch probe configure failed: %s\n", ls.error_msg.c_str());
        // T-GPLR-22: was a bare run_end, which is the confusing shape this
        // event replaces -- run_end with no run_start before it reads as a
        // crashed trace. A refusal is pre-work, so it emits run_refused only.
        ls.tel_run_refused("prefetch_configure", ls.error_msg.c_str(), 1, 0, 0, "none");
        return 1;
    }
    // T-GPLR-25: configure the async-feed worker (device copy overlap). The
    // flag was validated (0/1) and gated to --layer-slots >= 2 at parse time;
    // a module-side configure failure here aborts loudly before any timed work.
    // T-GPLR-32: the serialize gate must be configured BEFORE the worker starts.
    if (pp_af_serialize && pp_async_feed > 0 && !ls.set_pp_af_serialize(true)) {
        fprintf(stderr, "error: af-serialize configure failed: %s\n", ls.error_msg.c_str());
        ls.tel_run_refused("af_serialize_configure", ls.error_msg.c_str(), 1, 0, 0, "none");
        return 1;
    }
    // T-GPLR-33: member-batched staging reads the flag per submission; no
    // ordering constraint beyond "before the run starts".
    if (pp_submit_batch && !ls.set_pp_submit_batch(true)) {
        fprintf(stderr, "error: submit-batch configure failed: %s\n", ls.error_msg.c_str());
        ls.tel_run_refused("submit_batch_configure", ls.error_msg.c_str(), 1, 0, 0, "none");
        return 1;
    }
    // T-GPLR-35: records the intent only — the compute backend is resolved with
    // the scheduler, where the aux stream is validated and a run without it is
    // refused loudly (never a silent fallback to the serialized path).
    if (pp_copy_overlap) {
        ls.set_pp_copy_overlap(true);
    }
    if (pp_recon) {
        ls.set_pp_recon(true);
    }
    // T-GPLR-53 A4: pre-timed modality pre-warm. Off (0) is the default and runs
    // nothing; a non-zero mode requires --pp-schedule layer-major (validated above).
    if (pp_prewarm > 0 && !ls.set_pp_prewarm(pp_prewarm)) {
        fprintf(stderr, "error: %s\n", ls.error_msg.c_str());
        return 1;
    }
    // T-GPLR-41: pass-boundary memory-lifecycle recon (diagnostics only).
    // Requires the layer-major controller; refused under --mtp, which uses a
    // different (per-ubatch) path and never reaches layer_major_pp.
    if (pp_mctx_recon) {
        if (mtp) {
            fprintf(stderr, "error: --pp-mctx-recon is incompatible with --mtp\n");
            ls.tel_run_refused("mctx_recon_configure", "incompatible with --mtp", 1, 0, 0, "none");
            return 1;
        }
        ls.set_pp_mctx_recon(true);
    }
    // T-GPLR-39: the route-DIAGNOSTIC switch. Pass the runner's RESOLVED value
    // down (the engine-level default stays on for direct API consumers).
    ls.set_route_telemetry(route_telemetry != 0);
    if (pp_async_feed > 0 && !ls.set_pp_async_feed(pp_async_feed)) {
        fprintf(stderr, "error: async-feed configure failed: %s\n", ls.error_msg.c_str());
        ls.tel_run_refused("async_feed_configure", ls.error_msg.c_str(), 1, 0, 0, "none");
        return 1;
    }
    // T-GPLR-30: configure the persistent pinned pool (after init — it needs
    // the per-layer package sizes — and after async-feed, since a failed pool
    // allocation must not leave a running worker behind). A failure aborts
    // loudly: the locked-page quota refusal is itself the datapoint.
    if (pp_pool_n > 0 && !ls.set_pp_pool(pp_pool_n, pp_pool_policy.c_str())) {
        fprintf(stderr, "error: pp-pool configure failed: %s\n", ls.error_msg.c_str());
        ls.tel_run_refused("pp_pool_configure", ls.error_msg.c_str(), 1, 0, 0, "none");
        return 1;
    }
    // T-GPLR-56 A3: arm the paced registration. The mode's flag-level preconditions
    // were already refused before the load; what remains here is the engine's own
    // check against the copy source it actually resolved, and it refuses (never
    // clamps), so a typo cannot silently run the default path and be reported as pace.
    if (pp_span_pace > 0 && !ls.set_pp_span_pace(pp_span_pace, pp_span_pace_ahead)) {
        fprintf(stderr, "error: pp-span-pace configure failed: %s\n", ls.error_msg.c_str());
        ls.tel_run_refused("span_pace_configure", ls.error_msg.c_str(), 1, 0, 0, "none");
        return 1;
    }
    // T-GPLR-56 A3: set by whichever registration site runs, so the other site skips.
    bool spans_registered = false;
    // T-GPLR-04: emit the structured byte_account evidence now that the
    // telemetry sink exists (ls.init opened it). The preflight arithmetic
    // above already passed; this re-derivation is cheap (host-only metadata)
    // and uses the same shared helper so the two sites cannot drift.
    {
        ggml_type pf_kv = GGML_TYPE_F16;
        if (kv_cache == "q8") pf_kv = GGML_TYPE_Q8_0;
        else if (kv_cache == "q4") pf_kv = GGML_TYPE_Q4_0;
        const llama_ls_inventory pf = llama_ls_model_inventory(
            model_path.c_str(), n_ctx_total, pf_kv, vram.total, "cuda-device-total");
        llama_ls_bounded_window pf_win;
        if (pf.ok && ls_bounded_window_for_policy(pf, model_arch.c_str(), n_slots, pf_win, residency_tg_window)) {
            // The window's class bytes come from the re-parsed inventory (the
            // deliberate second derivation that keeps this emission from
            // drifting from the preflight), but the bounded total, the slot
            // window and the reserve are read from the shared budget model so
            // the emitted account is provably the one the checks used.
            // T-GPLR-19: the row cache and the operator's total residency cap
            // go in the same record as the bounded-window account, so the cap
            // this run enforced at bring-up is verifiable from JSONL alone.
                        // T-GPLR-59: the pool is a separate, named VRAM class. It is
                // omitted entirely when zero so legacy no-pool traces retain
                // their byte-account schema.
            // residency_cap is 0 when no explicit --residency-budget-mib was
            // given: then the bounded window is its own cap and the cache is
            // free-derived, exactly as before this field existed.
            ls.tel_byte_account(pf_win.full_model_plus_runtime, vram.bounded,
                                pf_win.static_bytes, pf_win.per_layer_resident, vram.slot_window,
                                pf_win.kv_bytes, pf_win.state_bytes, act_slab_bytes,
                                vram.reserve_bytes,
                                vram.free_start, vram.total, vram.residency_budget_mib,
                                (uint64_t) ls.row_cache_alloc_bytes,
                                vram.residency_budget_mib > 0 ? vram.residency_budget_bytes : 0,
                                (uint64_t) ls.row_set_spec_pool_alloc_bytes);
        }
    }
    ls.exec_layers = boundary_layers;

    // T-GPLR-56 A3 arm `p1-early`: the earliest LEGAL registration site - after the
    // load, after ls.init and after exec_layers is assigned, before the prebuild.
    // A2 measured the head start this buys at 66.4 ms, so this arm is the negative
    // control for "earlier is enough". Trace consequence, accepted and recorded:
    // this site's registration events precede run_start. No checker rule objects
    // (only NAMED layer events carry a before-run_start rule, and unknown event
    // names are ignored by design), and registration during the model LOAD remains
    // forbidden, so this is the earliest a mode may go.
    if (pp_register_site == 1 && copy_source == "mmap-registered") {
        llama_ls_tel_mark("register_issue", "early");   // pre-sink-safe: WHERE it was issued
        fprintf(stderr, "mmap-registered staging: registering mapped source spans (site=early, before the prebuild)...\n");
        int64_t reg_bytes = 0;
        double  reg_us    = 0.0;
        if (!ls.register_paced_begin(reg_bytes, reg_us)) {
            ls.failed = true;
            fprintf(stderr, "error: %s (aborted before the timed PP)\n", ls.error_msg.c_str());
            ls.tel_run_refused("mmap_register", ls.error_msg.c_str(), 1, 0, 0, "none");
            return 1;
        }
        fprintf(stderr, "mmap-registered staging: %lld B (%.2f GiB) pinned in %.1f ms (site=early, pre-timed setup)\n",
                (long long) reg_bytes, reg_bytes / 1073741824.0, reg_us / 1000.0);
        spans_registered = true;
    }

    // T-GPLR-02 negative fixture: drop the last member from the first executed
    // layer's descriptor AFTER its slot bundle was built, so the descriptor
    // diverges from the bundle and load_layer must reject the partial package
    // before any copy submission.
    if (fault_drop_member) {
        const int fl = boundary_layers.empty() ? 0 : boundary_layers[0];
        if (fl >= 0 && fl < (int) ls.pkg_cache.size() && !ls.pkg_cache[fl].members.empty()) {
            ls.pkg_cache[fl].members.pop_back();
            fprintf(stderr, "fault: dropped last package member from layer %d descriptor\n", fl);
        } else {
            fprintf(stderr, "error: --fault-drop-member needs a valid packaged layer\n");
            return 2;
        }
    }

    size_t slot_vram = 0;
    for (const auto & slot : ls.slots) {
        // T-GPLR-62 S2: once a compact window is built, the full-bank buffer is
        // RELEASED and the narrow one lives in buf_rowset. Summing only buf
        // would report 0 B of slot VRAM for exactly the run whose A1 number
        // matters most, so the compact buffer is counted when present.
        if (slot.buf_rowset) {
            slot_vram += ggml_backend_buffer_get_size(slot.buf_rowset.get());
        } else if (slot.buf) {
            slot_vram += ggml_backend_buffer_get_size(slot.buf.get());
        }
    }
    // T-GPLR-12: the auditable handoff decision (post-preflight resolution).
    std::string handoff_mode_tel = handoff_auto ? (handoff ? "auto-on" : "auto-off")
                                                : handoff_mode_str; // "on" | "off"
    // T-GPLR-48: how the per-pass rebuild was resolved, and by whom.
    const char * pp_prebuild_per_pass_src = pp_prebuild_per_pass_explicit ? "explicit" : "default";
    // T-GPLR-37: how the copy source was resolved, and by whom.
    const char * copy_source_src = copy_source_explicit ? "explicit" : "default";
    // T-GPLR-39: how the route-diagnostic switch was resolved, and by whom.
    const char * route_telemetry_src = route_telemetry_explicit ? "explicit" : "default";
    ls.tel_run_start(*model, 0, slot_vram, vram.free_start, vram.total,
                     schedule_name, copy_source.c_str(), kv_cache.c_str(), act_slab_bytes,
                     (int64_t) pp_tokens, ls.chunks, act_slab_dtype, (int64_t) n_ctx_total,
                     handoff_mode_tel.c_str(), residency_tg,
                     pp_prebuild_per_pass, pp_prebuild_per_pass_src, copy_source_src,
                     route_telemetry, route_telemetry_src,
                     tg_logit_margin);

    ctx->set_layer_streaming(&ls);

    // T-Q38HG-06: the MTP draft context. Created before the PP so the prompt
    // ubatches can be mirrored into the draft head (KV alignment). It shares
    // the model (blk.64 loaded), decodes with hidden-state rows from the
    // target, and carries the streaming attachment so its MTP graph binds
    // the handoff-resident blk.64 bundle (non-default graph types bypass the
    // streaming executor via the gtype guard).
    const int n_embd_out = (int) llama_model_n_embd_out(model);
    const size_t row_bytes = (size_t) n_embd_out * sizeof(float);
    llama_context * ctx_dft = nullptr;
    if (mtp) {
        llama_context_params dparams = llama_context_default_params();
        dparams.n_ctx        = n_ctx_total;
        dparams.n_batch      = n_ctx_total;
        // T-GPLR-06 step 2: the per-round mirror fill decodes one row per
        // committed target cell (up to 1 + mtp_draft), so the draft ubatch
        // must hold a full round's fill.
        dparams.n_ubatch     = 1 + mtp_draft;
        dparams.n_seq_max    = 1;
        dparams.n_outputs_max = 1;
        dparams.n_threads       = 16;
        dparams.n_threads_batch = 16;
        dparams.ctx_type     = LLAMA_CONTEXT_TYPE_MTP;
        dparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        dparams.ctx_other    = ctx;

        ctx_dft = llama_init_from_model(model, dparams);
        if (ctx_dft == nullptr) {
            fprintf(stderr, "error: failed to create MTP draft context\n");
            return 1;
        }
        ctx_dft->set_layer_streaming(&ls);
        llama_set_embeddings_nextn(ctx, true, false);   // target: per-position hidden rows
        llama_set_embeddings_nextn(ctx_dft, true, true); // draft: chained head-internal rows
        fprintf(stderr, "MTP: draft head (blk.%d) enabled, n_embd_out %d, n_rs_seq %u\n",
                n_layer, n_embd_out, (unsigned) cparams.n_rs_seq);
    }

    // T-Q38HG-06: shared MTP catch-up batch (ubatch capacity: prompt mirrors,
    // draft steps and per-round catch-up decodes all reuse it) and the h_nextn
    // row carried across decodes (the "pending" row of the stock speculative
    // path). Zero-initial: the first prompt token has no previous hidden row.
    std::vector<float> pending_h((size_t) n_embd_out, 0.0f);
    // T-Q38HG-06: the DEFERRED mirror row's hidden across PP ubatches and TG
    // frontier decodes (see mtp_mirror); mtail_off is the position of the
    // last completed-but-unpairable row (-1 = none pending).
    std::vector<float> mtail_h((size_t) n_embd_out, 0.0f);
    int mtail_off = -1;
    std::vector<float> fill_h((size_t) n_embd_out, 0.0f);  // latch: h of the most recent target decode
    llama_batch dbatch = llama_batch_init(mtp ? ubatch : 0, mtp ? n_embd_out : 0, 1);
    if (mtp) {
        dbatch.token = (llama_token *) malloc(sizeof(llama_token) * (size_t) ubatch);
        if (dbatch.token == nullptr) {
            fprintf(stderr, "error: failed to allocate MTP batch\n");
            return 1;
        }
    }

    int exit_status = 0;
    // T-GPLR-22: which pre-timed check aborted the run, and the byte arithmetic
    // that decided (0/"none" where the check is not about bytes). Each check
    // sets these immediately before it sets exit_status, and the single terminal
    // write consumes them, so every pre-timed abort lands in the same
    // `run_refused` record the pre-work refusals use. The initial value is the
    // first check in the sequence.
    const char * abort_reason  = "prebuild_failed";
    uint64_t     abort_needed  = 0;
    uint64_t     abort_allowed = 0;
    const char * abort_cap     = "none";
    std::vector<llama_token> generated;

    // graph-prep accounting shared by the PP and TG phase summaries (both
    // phases consume graph sets built in the same pre-timer prebuild)
    int64_t    graph_prep_us = 0;
    uint64_t   builds0 = 0;

    // ------------------------------------------------------------------
    // PP: pp_tokens deterministic ids 0..pp_tokens-1, ubatch-sized splits
    // ------------------------------------------------------------------
    {
        const int n_vocab_pp = llama_vocab_n_tokens(llama_model_get_vocab(model));
        llama_batch batch = llama_batch_init(pp_tokens, 0, 1);
        for (int i = 0; i < pp_tokens; ++i) {
            batch.token[i] = prompt_tokens.empty() ? (i % n_vocab_pp) : prompt_tokens[i];
            batch.pos[i]   = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = !pp_no_logits;
        }
        batch.n_tokens = pp_tokens;

        ls.phase = 0;

        // pre-build the persistent graph sets BEFORE the timer starts (both PP
        // recurrent variants, plus the TG u1 variant when --tg-tokens > 0; no
        // execution, cells released again). The timed decode then selects the
        // matching prebuilt set per ubatch, so the timed region performs zero
        // graph construction (T-Q38LS-04B).
        ggml_status st = GGML_STATUS_SUCCESS;
        const int64_t t_prep0 = ggml_time_us();
        fprintf(stderr, "layer-major preflight: building persistent graphs\n");
        const bool prebuilt = llama_layer_streaming_prebuild(ctx, batch, tg_tokens, st);
        graph_prep_us = ggml_time_us() - t_prep0;
        builds0 = ls.build_generation;

        if (!prebuilt) {
            fprintf(stderr, "error: PP graph prebuild failed: %s\n", ls.error_msg.c_str());
            ls.failed = true;
            abort_reason = "prebuild_failed";
            exit_status = 1;
        } else {
            fprintf(stderr, "PP graph preparation: %.1f ms (build_generation %llu)\n",
                graph_prep_us / 1000.0, (unsigned long long) builds0);

            // T-Q38CAP-01: graph-footprint preflight (evidence, before the PP
            // timer): reserved/used io-context memory of every persistent
            // graph set, host input usage, slab and slot VRAM. A caller-
            // specified --graph-budget-mib bounds the summed RESERVED io bytes
            // and aborts before the timed PP when exceeded.
            {
                size_t n_graphs = 0, io_reserved = 0, io_used = 0;
                for (const auto & s : ls.sets) {
                    n_graphs += s.graphs.size();
                    if (s.ctx != nullptr) {
                        io_reserved += ggml_get_mem_size(s.ctx);
                        for (ggml_tensor * t = ggml_get_first_tensor(s.ctx); t != nullptr; t = ggml_get_next_tensor(s.ctx, t)) {
                            io_used += (size_t) ggml_nbytes(t);
                        }
                    }
                }
                size_t host_input_bytes = ggml_backend_buffer_get_size(ls.buf_inputs.get());
                ls.tel_graph_footprint((int64_t) ls.sets.size(), (int64_t) n_graphs, io_reserved, io_used,
                                       host_input_bytes, ls.buf_inputs_off, act_slab_bytes, slot_vram, graph_budget_mib);
                fprintf(stderr,
                    "graph footprint: sets=%zu total_graphs=%zu io_reserved=%zu B io_used=%zu B "
                    "host_input=%zu B (used %zu B) act_slab=%zu B slot_vram=%zu B budget=%s\n",
                    ls.sets.size(), n_graphs, io_reserved, io_used,
                    host_input_bytes, ls.buf_inputs_off, act_slab_bytes, slot_vram,
                    graph_budget_mib > 0 ? (std::to_string(graph_budget_mib) + " MiB").c_str() : "none");
                if (graph_budget_mib > 0 && io_reserved > (size_t) graph_budget_mib * 1024u * 1024u) {
                    const size_t budget_bytes = (size_t) graph_budget_mib * 1024u * 1024u;
                    ls.error_msg = "graph footprint exceeds the budget: io_ctx_reserved " +
                                   std::to_string(io_reserved) + " B > " +
                                   std::to_string(budget_bytes) +
                                   " B (--graph-budget-mib)";
                    ls.failed = true;
                    // T-GPLR-22: this abort is a REFUSAL, so it must NOT also
                    // write a bare `error` event -- both checkers reject any
                    // `error` event as a failed run, which is what made the old
                    // shape unreadable. The message still reaches the trace, as
                    // the refusal's `detail`.
                    abort_reason  = "graph_budget";
                    abort_needed  = (uint64_t) io_reserved;
                    abort_allowed = (uint64_t) budget_bytes;
                    abort_cap     = "budget";
                    fprintf(stderr, "error: %s (aborted before the timed PP)\n", ls.error_msg.c_str());
                    exit_status = 1;
                }
            }
            // T-GPLR-37 churn-free variant: page-lock every executed layer's
            // mapped package source ONCE, here (after the prebuild, before the
            // timed PP), so staging never touches page tables during the pass.
            // Pre-timed, so the cost is setup and it cannot pollute the pass
            // wall; a failure is a REFUSAL (same shape as the graph-budget
            // abort) rather than a silent fall back to the copy hop.
            // T-GPLR-56 A3: `spans_registered` is set by the early site above, so the
            // registration happens exactly once whichever site ran. With the mode off
            // this block is byte-for-byte today's: register_paced_begin() delegates to
            // register_all_pkg_sources() when pp_span_pace == 0.
            if (pp_register_site == 0 && !spans_registered && copy_source == "mmap-registered") {
                int64_t reg_bytes = 0;
                double  reg_us    = 0.0;
                fprintf(stderr, "mmap-registered staging: registering mapped source spans (once)...\n");
                if (!ls.register_paced_begin(reg_bytes, reg_us)) {
                    ls.failed = true;
                    abort_reason  = "mmap_register";
                    abort_needed  = 0;
                    abort_allowed = 0;
                    abort_cap     = "none";
                    fprintf(stderr, "error: %s (aborted before the timed PP)\n", ls.error_msg.c_str());
                    exit_status = 1;
                } else if (pp_span_pace > 0) {
                    // T-GPLR-56 A3: the paced call pins the FIRST group only and returns
                    // what it pinned so far, so the one-shot wording would understate the
                    // span this run is committing. Print the mode's own accounting instead
                    // (total groups / total bytes / issued so far), and keep the off path's
                    // line byte-identical above.
                    fprintf(stderr, "mmap-registered staging: PACED registration armed - %lld of %lld B in %zu groups; first group pinned in %.1f ms (the rest issues ahead of each layer's first copy, %d group(s) ahead)\n",
                            (long long) ls.paced_reg_bytes_issued(), (long long) ls.paced_reg_bytes_total(),
                            ls.paced_reg_group_count(), reg_us / 1000.0, pp_span_pace_ahead);
                } else {
                    fprintf(stderr, "mmap-registered staging: %lld B (%.2f GiB) pinned in %.1f ms (pre-timed setup)\n",
                            (long long) reg_bytes, reg_bytes / 1073741824.0, reg_us / 1000.0);
                }
            }
            if (exit_status != 0) {
                // T-GPLR-22: a PRE-TIMED abort is a refusal, not a failed run.
                // run_refused is terminal and closes the sink, so the trace
                // reads as the ABORTED-RUN refusal shape (run_start present, no
                // completed phase, no run_end) instead of as a crashed run that
                // `--check-telemetry` had to reject.
                ls.tel_run_refused(abort_reason, ls.error_msg.c_str(), exit_status,
                                   abort_needed, abort_allowed, abort_cap);
                return exit_status;
            }

            // T-Q38HG-06: MTP prompt catch-up, mirroring the stock
            // speculative process() hook: every row p stores (x_{p+1}, h_p)
            // — the NEXT prompt token paired with the TARGET's nextn hidden
            // of p — at position p. The LAST row of each ubatch is deferred
            // (its partner x_{p+1} is the first token of the next ubatch,
            // unknown at mirror time) and completed by the NEXT ubatch's
            // bridge row at pos p = (off-1), using the deferred hidden
            // stashed from the previous call (the target's per-call nextn
            // slots are overwritten by the next decode). The first prompt
            // row (x_1, h_0) has no h_0 at all and pairs with zeros. After
            // the final ubatch, the deferred row (x_P, h_{P-1}) is completed
            // by the first draft round's seed at pos P.
            auto mtp_mirror = [&](const llama_batch & ub, int off, int n_tok) -> int {
                int n_rows = 0;
                if (mtail_off >= 0) {
                    // bridge: the previous ubatch's deferred row
                    // (x_off, h_{off-1}) at pos off-1
                    dbatch.token[n_rows]   = ub.token[0];
                    dbatch.pos[n_rows]     = (llama_pos) (off - 1);
                    dbatch.n_seq_id[n_rows] = 1;
                    dbatch.seq_id[n_rows][0] = 0;
                    dbatch.logits[n_rows]   = false;
                    memcpy(dbatch.embd, mtail_h.data(), row_bytes);
                    n_rows++;
                }
                // shifted rows: (x[i+1], h[off+i]) at pos off+i, reading the
                // current call's slots 0..n_tok-2 (h of off..off+n_tok-2)
                for (int i = 0; i < n_tok - 1; ++i) {
                    dbatch.token[n_rows]   = ub.token[i + 1];
                    dbatch.pos[n_rows]     = (llama_pos) (off + i);
                    dbatch.n_seq_id[n_rows] = 1;
                    dbatch.seq_id[n_rows][0] = 0;
                    dbatch.logits[n_rows]   = false;
                    const float * h = llama_get_embeddings_nextn_ith(ctx, i);
                    if (h == nullptr) {
                        fprintf(stderr, "error: no target nextn row at PP mirror slot %d\n", i);
                        return -1;
                    }
                    memcpy(dbatch.embd + (size_t) n_rows * n_embd_out, h, row_bytes);
                    n_rows++;
                }
                if (n_rows == 0) {
                    // stash the deferred h (n_tok == 1 ubatch: only the tail)
                    const float * h_last = llama_get_embeddings_nextn_ith(ctx, n_tok - 1);
                    if (h_last != nullptr) {
                        memcpy(mtail_h.data(), h_last, row_bytes);
                    }
                    mtail_off = off;
                    return 0;
                }
                dbatch.n_tokens = n_rows;
                const int rc = llama_decode(ctx_dft, dbatch);
                if (rc != 0) {
                    fprintf(stderr, "error: MTP prompt mirror decode failed (%d)\n", rc);
                    return rc;
                }
                // stash the deferred h (the LAST row of this call = slot
                // n_tok-1; the next ubatch's bridge pairs with it)
                const float * h_last = llama_get_embeddings_nextn_ith(ctx, n_tok - 1);
                if (h_last == nullptr) {
                    fprintf(stderr, "error: no target nextn row at PP mirror slot %d\n", n_tok - 1);
                    return -1;
                }
                memcpy(mtail_h.data(), h_last, row_bytes);
                mtail_off = off;
                return 0;
            };

            // T-GPLR-28: in-process PP repeat. Passes 1..N-1 clear KV (so
            // positions restart at 0) and reset the per-layer stats; the slot
            // window is deliberately NOT reset — banks left resident from the
            // previous pass are the warm-restart condition being measured. TG
            // runs once after the final pass. Per-pass phase_summaries share
            // the prebuild's builds0 baseline, so graph_builds stays 0 unless
            // the timed region really constructs graphs (checker-enforced).
            for (int pp_pass_i = 0; pp_pass_i < pp_repeat && exit_status == 0; ++pp_pass_i) {
            if (pp_pass_i > 0) {
                llama_memory_clear(llama_get_memory(ctx), true);
                ls.reset_stats();
                ls.tel_pp_pass(pp_pass_i, pp_tokens);
                fprintf(stderr, "PP pass %d/%d: re-running prefill on cleared KV\n", pp_pass_i + 1, pp_repeat);
                // T-GPLR-43 probe (--pp-prebuild-per-pass): rebuild ALL
                // persistent graph sets against the post-clear memory state
                // so the new pass executes fresh cgraphs (new KV views, new
                // copy nodes). begin_set's rebuild path reuses the same set
                // keys/slab offsets; builds0 advances so per-pass phase
                // summaries keep graph_builds = 0 (checker-clean). Refused
                // under --mtp below.
                if (pp_prebuild_per_pass == 1) {
                    // T-GPLR-47: pass-2+ prebuilds REPLACE each built set with a
                    // fresh-context rebuild (swap-in-slot) instead of resetting in
                    // place -- no ctx accumulation, pointers valid by construction.
                    // Explicit flag around the call, per the task contract (INV1).
                    ls.set_rebuild_replacement(true);
                    const int64_t t_rb0 = ggml_time_us();
                    ggml_status st_rb = GGML_STATUS_SUCCESS;
                    const bool rb_ok = llama_layer_streaming_prebuild(ctx, batch, tg_tokens, st_rb);
                    ls.set_rebuild_replacement(false);
                    if (!rb_ok) {
                        fprintf(stderr, "error: per-pass prebuild failed: %s\n", ls.error_msg.c_str());
                        ls.failed = true;
                        abort_reason = "prebuild_failed";
                        exit_status = 1;
                        break;
                    }
                    builds0 = ls.build_generation;
                    fprintf(stderr, "PP pass %d/%d: graphs rebuilt in %.1f ms (build_generation %llu)\n",
                        pp_pass_i + 1, pp_repeat, (ggml_time_us() - t_rb0) / 1000.0,
                        (unsigned long long) builds0);
                }
            }
            // T-GPLR-53 A4: the pre-warm runs BEFORE the timed region, and pass 1
            // only (later passes are already warm). Its own cost is reported on
            // stderr and is deliberately outside the PP wall brackets, because
            // A4 measures what moved OUT of the pass-1 wall.
            if (pp_pass_i == 0 && pp_prewarm > 0) {
                if (!ls.pp_prewarm_run(pp_prewarm_delay_ms)) {
                    fprintf(stderr, "error: pp pre-warm failed: %s\n", ls.error_msg.c_str());
                    ls.failed = true;
                    exit_status = 1;
                    break;
                }
            }
            const int64_t t0 = ggml_time_us();
            int ret = 0;
            if (mtp) {
                // per-ubatch prompt decodes so the draft head can mirror each
                // row (T-Q38HG-06).
                for (int off = 0; off < pp_tokens && exit_status == 0; off += ubatch) {
                    const int n_tok = std::min(ubatch, pp_tokens - off);
                    llama_batch ub = llama_batch_init(n_tok, 0, 1);
                    for (int i = 0; i < n_tok; ++i) {
                        ub.token[i]   = batch.token[off + i];
                        ub.pos[i]     = off + i;
                        ub.n_seq_id[i] = 1;
                        ub.seq_id[i][0] = 0;
                        ub.logits[i]   = !pp_no_logits;
                    }
                    ub.n_tokens = n_tok;
                    ret = llama_decode(ctx, ub);
                    if (ret == 0 && mtp && ctx_dft) {
                        // mirror the target's rows into the draft head
                        ret = mtp_mirror(ub, off, n_tok);
                    }
                    llama_batch_free(ub);
                }
            } else if (schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR) {
                // layer-major PP (SDD section 13, T-Q38LM-02): drive the full
                // prompt through the dedicated layer-major controller rather
                // than the batch-major ubatch decode loop.
                ggml_status st = GGML_STATUS_SUCCESS;
                ret = llama_layer_major_pp(ctx, batch, st);
                if (st != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "error: layer-major PP failed: %s\n", ls.error_msg.c_str());
                    ls.failed = true;
                }
            } else {
                ret = llama_decode(ctx, batch);
            }

            const int64_t t1 = ggml_time_us();

            if (ret != 0) {
                fprintf(stderr, "error: PP decode failed with %d: %s\n", ret, ls.error_msg.c_str());
                ls.failed = true;
                exit_status = 1;
            } else {
                ls.tel_phase_summary("pp", pp_tokens, t1 - t0, builds0, graph_prep_us);
                fprintf(stderr, "PP: %d tokens in %.1f ms (%.2f tok/s)  [transfer+wait+compute per layer in telemetry]",
                    pp_tokens, (t1 - t0) / 1000.0, pp_tokens * 1e6 / (double) (t1 - t0));
                if (pw_probe) {
                    fprintf(stderr, "  total_patch_us=%.1f", ls.total_patch_us);
                }
                fprintf(stderr, "\n");
                // T-GPLR-03: the agreed first-greedy-ID parity quantity: the
                // argmax of the PP logits at the last prompt position (what a
                // normal-path greedy sampler generates first). The TG probe
                // below re-decodes the last prompt token and computes a
                // different conditional, so it cannot be compared against the
                // normal path directly.
                const int64_t pp_first = llama_layer_major_pp_first_token(ctx);
                if (pp_first >= 0) {
                    fprintf(stderr, "PP-logits first greedy ID: %lld\n", (long long) pp_first);
                }
            }
            } // T-GPLR-28: per-pass loop

            // T-Q38HG-06: keep the MTP catch-up batch alive for the TG round
            // (the draft head and catch-up rows); freed below after TG.
            llama_batch_free(batch);
        }
    }

    // ------------------------------------------------------------------
    // TG: tg_tokens greedy steps immediately after PP (no state reset)
    // ------------------------------------------------------------------
    if (exit_status == 0 && tg_tokens > 0) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

        // T-GPLR-06 step 2: under MTP the batch must hold one full verify
        // round (frontier + all drafts) in a single ubatch decode.
        llama_batch batch = llama_batch_init(mtp ? 1 + mtp_draft : 1, 0, 1);
        generated.reserve(tg_tokens);

        llama_token cur = prompt_tokens.empty()
            ? (llama_token) (pp_tokens - 1)
            : prompt_tokens[pp_tokens - 1]; // last PP token; first generated comes from logits

        ls.phase = 1;

        // T-Q38HG-02/04/06: handoff — copy packages to a single device
        // buffer so the standard TG path (and the MTP draft head) sees
        // resident weights. The MTP layer joins the attention-only set only
        // when --mtp is active.
        if (handoff) {
            // T-GPLR-10: policy-aware migration description -- under
            // moe-expert-bank the migration list is the 40 routed-expert
            // banks, not the 64 dense layers.
            const char * handoff_desc =
                handoff_partial_n >= 0 ? "partial (banks il >= N)"
                : residency == "moe-expert-bank"
                    ? (handoff_attention_only ? "attention-only (refused under moe-expert-bank)" : "all (40 expert banks)")
                    : (handoff_attention_only ? "attention-only (16 layers)" : "all (64 layers)");
            fprintf(stderr, "Handoff: migrating %s weights to device...\n", handoff_desc);
            bool ok = false;
            if (handoff_partial_n >= 0) {
                // T-GPLR-16: migrate only the banks at/above the split; the
                // first N layers' banks stay host-resident (CPU) for
                // mixed-placement TG.
                ok = ls.begin_handoff_partial(handoff_partial_n);
            } else {
                ok = ls.begin_handoff(handoff_attention_only, mtp);
            }
            if (!ok) {
                fprintf(stderr, "error: handoff migration failed: %s\n", ls.error_msg.c_str());
                ls.failed = true;
                exit_status = 1;
            } else {
                // T-GPLR-10: capture the migration list for the in-process
                // trace self-check: the same ascending per-layer 256-alignment
                // and final 4096-alignment as begin_handoff, so the migrate
                // event's layers/total_bytes must match exactly (40 bank
                // packages, 18,327,011,328 B + alignment on the MoE model).
                g_check_handoff_layers = 0;
                g_check_handoff_bytes = 0;
                for (int il = 0; il <= n_layer; ++il) {
                    const auto it = ls.handoff_bytes.find(il);
                    if (it == ls.handoff_bytes.end()) {
                        continue;
                    }
                    g_check_handoff_bytes = (g_check_handoff_bytes + 255) / 256 * 256 + (long long) it->second;
                    g_check_handoff_layers++;
                }
                g_check_handoff_bytes = (g_check_handoff_bytes + 4095) / 4096 * 4096;
                fprintf(stderr, "Handoff: migrated %lld packages, %lld bytes\n",
                        g_check_handoff_layers, g_check_handoff_bytes);
            }
        }

        // (ctx_dft was created before the PP so prompt ubatches could be
        // mirrored into the draft head KV; it carries the streaming
        // attachment and its MTP graphs so the handoff-resident blk.64 is
        // bound when handoff is active)

        // T-GPLR-13: activate reactive row-set staging for TG (fences the last
        // PP consumer, bumps slot generations, allocates the per-slot row
        // coverage maps, emits rowset_init). Loud failure before any TG decode.
        if (residency_tg && !ls.begin_row_set()) {
            fprintf(stderr, "error: row-set activation failed: %s\n", ls.error_msg.c_str());
            ls.failed = true;
            exit_status = 1;
        }
        // T-GPLR-24 (H-B): fold the slot window into the free-min tracker
        // whether or not activation failed — it allocates before failing.
        if (residency_tg) {
            ls_vram_budget_refresh(vram, device);
        }

        // isolate the TG phase: the per-layer stats and cumulative counters were
        // filled by PP, so reset them so the TG phase_summary reports TG-only
        // transfer/wait/compute and ubatch counts (T-Q38LS-05)
        ls.reset_stats();
        const int64_t t0 = ggml_time_us();

        // MTP acceptance counters (T-Q38HG-06) + batch-verify counters
        // (T-GPLR-06 step 2)
        int64_t mtp_drafted     = 0;
        int64_t mtp_accepted    = 0;
        int64_t mtp_pushed      = 0;   // target rows committed (accepted prefix + frontier)
        int64_t mrounds_batch   = 0;   // verify rounds
        int64_t mrounds_full    = 0;   // all drafts accepted
        int64_t mrounds_partial = 0;   // some draft rejected
        int64_t mrounds_rewind  = 0;   // rounds that rolled KV/RS back

        if (mtp && exit_status == 0) {
            // --- MTP draft/verify/accept loop (T-Q38HG-06) ---
            // The target context keeps the authoritative greedy state. The
            // draft head consumes (token, h_nextn row of the previous target
            // position): after the PP, the very first MTP row pairs the last
            // prompt token with the pending hidden row carried from PP;
            // thereafter every draft step grows the chain (token, hidden of
            // the token just drafted). T-GPLR-06 step 2: each round drafts d
            // tokens, then verifies the frontier plus ALL drafts in ONE
            // target decode (one expert-bank pass instead of d+1), commits
            // the matching gold prefix, and rolls rejected rows back with an
            // exact KV seq_rm plus the recurrent snapshot rewind.
            std::vector<llama_token> draft;
            draft.reserve((size_t) mtp_draft);

            // 0. The first frontier decode: (x[P-1] at pos P). This is the
            //    standard greedy first TG step: its argmax is the first
            //    generated token (x[P]) and the seed of the first draft
            //    round. From here on every frontier decode is (cur at the
            //    frontier), the exact decode sequence of the non-MTP greedy
            //    path, so parity holds by construction.
            fprintf(stderr, "MTP: initial frontier decode (x[%d] at pos %d)\n", (int) cur, (int) pp_tokens); fflush(stderr);
            {
                batch.token[0]   = cur;
                batch.pos[0]     = pp_tokens;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0]  = true;
                batch.n_tokens   = 1;
                const int ret_i = llama_decode(ctx, batch);
                if (ret_i != 0) {
                    fprintf(stderr, "error: MTP initial verify decode failed with %d: %s\n", ret_i, ls.error_msg.c_str());
                    ls.failed = true;
                    exit_status = 1;
                } else {
                    const float * logits = llama_get_logits_ith(ctx, 0);
                    if (logits == nullptr) {
                        fprintf(stderr, "error: no logits at MTP initial verify\n");
                        ls.failed = true;
                        exit_status = 1;
                    } else {
                        llama_token best = 0;
                        float best_val = logits[0];
                        {
                            // T-GPLR-06: penalized copy for the greedy pick
                            std::vector<float> plogits(logits, logits + n_vocab);
                            rep_ring.apply(plogits.data(), n_vocab, (float) rep_penalty);
                            best_val = plogits[0];  // penalized seed: token 0 may itself be in the ring
                            for (int v = 1; v < n_vocab; ++v) {
                                if (plogits[v] > best_val) { best_val = plogits[v]; best = v; }
                            }
                        }
                        cur = best;
                        generated.push_back(cur);
                        rep_ring.accept(cur);
                        // T-GPLR-06 step 2 diagnosis: the top-2 logit margin
                        // of every greedy pick discriminates a shape-numerics
                        // argmax flip (near-tie) from a real wiring bug
                        // (large margin) when batch and serial paths diverge
                        {
                            llama_token sbest = -1;
                            float sval = -INFINITY;
                            for (int v = 0; v < n_vocab; ++v) {
                                if (v == best) continue;
                                if (logits[v] > sval) { sval = logits[v]; sbest = v; }
                            }
                            fprintf(stderr, "MTP: initial frontier margin %.6f (gold %d, runner-up %d)\n",
                                    (double) (best_val - sval), (int) best, (int) sbest);
                            fflush(stderr);
                        }

                        // The prompt's final DEFERRED mirror row (x_P,
                        // h_{P-1}) at pos P-1 completes now that x_P exists;
                        // the draft head's KV then covers 0..P-1 contiguously
                        // (the round's seed pair (x_{P-1}, h_{P-1}) sits at P).
                        if (mtail_off >= 0) {
                            dbatch.token[0]   = cur;
                            dbatch.pos[0]     = (llama_pos) (pp_tokens - 1);
                            dbatch.n_seq_id[0] = 1;
                            dbatch.seq_id[0][0] = 0;
                            dbatch.logits[0]   = false;
                            memcpy(dbatch.embd, mtail_h.data(), row_bytes);
                            dbatch.n_tokens = 1;
                            const int mrc = llama_decode(ctx_dft, dbatch);
                            if (mrc != 0) {
                                fprintf(stderr, "error: MTP initial deferred fill failed (%d)\n", mrc);
                                ls.failed = true;
                                exit_status = 1;
                            }
                            mtail_off = -1;
                        }
                        // the frontier cell's h[P] pairs with x[P+1] once it
                        // exists: it feeds the FIRST verify iteration's
                        // deferred-row fill at pos P (fill_h latches the h of
                        // the most recent target decode)
                        const float * hp = llama_get_embeddings_nextn_ith(ctx, 0);
                        if (hp != nullptr) {
                            memcpy(fill_h.data(), hp, row_bytes);
                        }
                        // NOTE: pending_h deliberately keeps h[P-1] (the
                        // mirror's last row): the first seed pair is
                        // (x[P-1], h[P-1]). The frontier's own hidden
                        // (h[P]) goes to mtail_h for the deferred fill.
                    }
                }
            }
            fprintf(stderr, "MTP: rounds begin (generated %d)\n", (int) generated.size()); fflush(stderr);

            int mround = 0;
            while (exit_status == 0 && (int) generated.size() < tg_tokens && mround++ < 64 * tg_tokens + 64) {
                // 1. Draft up to mtp_draft tokens on the draft head. The
                //    seed pairs the frontier token (cur) with the hidden row
                //    of the frontier cell carried in pending_h; each
                //    following draft pairs its drafted token with the head's
                //    own nextn row. The head's KV was trimmed to the
                //    frontier at the end of the previous round, so the
                //    draft cells at dpos..dpos+n-1 are free.
                draft.clear();
                llama_token td = cur;
                // T-GPLR-06: the draft chain penalizes against the true ring
                // (prompt + generated) extended with its own hypothetical
                // drafts — mirror of the verify rows' view below.
                rep_penalty_ring draft_ring = rep_ring;
                for (llama_token t : draft) {
                    draft_ring.accept(t);
                }

                for (int d = 0; d < mtp_draft && (int) generated.size() + (int) draft.size() < tg_tokens; ++d) {
                    // the chain rows sit at the frontier cell and follow:
                    // the seed row is (x_{v-1}, h_{v-1}) at v, its
                    // prediction is the draft candidate for x_v
                    const llama_pos dpos = (llama_pos) (pp_tokens + (int) generated.size() - 1 + (int) draft.size());
                    llama_batch db = llama_batch_init(1, n_embd_out, 1);
                    db.token = (llama_token *) malloc(sizeof(llama_token));
                    db.token[0]  = td;
                    memcpy(db.embd, pending_h.data(), row_bytes);
                    db.pos[0]      = dpos;
                    db.n_seq_id[0] = 1;
                    db.seq_id[0][0] = 0;
                    db.logits[0]  = true;
                    db.n_tokens = 1;

                    const int drc = llama_decode(ctx_dft, db);
                    if (drc == 0) {
                        const float * dlogits = llama_get_logits_ith(ctx_dft, 0);
                        llama_token bd = 0;
                        float bdv = dlogits[0];
                        {
                            // T-GPLR-06: penalized copy; the ring must see the
                            // chain built so far (prompt + generated + drafts)
                            // so the head drafts what the penalized target will
                            // verify-accept.
                            std::vector<float> plogits(dlogits, dlogits + n_vocab);
                            draft_ring.apply(plogits.data(), n_vocab, (float) rep_penalty);
                            bdv = plogits[0];  // penalized seed
                            for (int v = 1; v < n_vocab; ++v) {
                                if (plogits[v] > bdv) { bdv = plogits[v]; bd = v; }
                            }
                        }
                        draft.push_back(bd);
                        td = bd;
                        draft_ring.accept(bd);
                        // chained h row of the drafted token (draft head's
                        // own nextn row)
                        const float * hd = llama_get_embeddings_nextn_ith(ctx_dft, 0);
                        if (hd != nullptr) {
                            memcpy(pending_h.data(), hd, row_bytes);
                        }
                    }
                    llama_batch_free(db); // frees token + embd + pos...
                    if (drc != 0) {
                        fprintf(stderr, "error: MTP draft decode failed (%d)\n", drc);
                        ls.failed = true;
                        exit_status = 1;
                        break;
                    }
                }
                if (exit_status != 0) {
                    break;
                }

                mtp_drafted += (int64_t) draft.size();

                // 2. T-GPLR-06 step 2: verify the frontier plus ALL drafts
                //    in ONE target decode — one expert-bank pass per round
                //    instead of one per draft. Rows follow the session's TG
                //    convention (the decode at pos p holds token x[p-1] and
                //    predicts x[p]): row 0 is (cur @ vpos) — the exact first
                //    decode of the serial path — and rows 1..nd are
                //    (draft[j-1] @ vpos+j), the hypothetical draft chain.
                //    Row j's prediction is the true next-token gold iff the
                //    chain through j-1 held, i.e. iff draft[j-1] was
                //    accepted (gold[j-1] == draft[j-1], the serial loop's
                //    exact accept sequence). Committed rows 0..A-1 are the
                //    true stream; rows A.. are rejected and rolled back
                //    with an exact KV seq_rm plus the recurrent snapshot
                //    rewind (n_rs_seq = mtp_draft bounds it:
                //    rmax - committed <= mtp_draft).
                const int g0   = (int) generated.size();
                const int nd   = (int) draft.size();
                const int rmax = 1 + nd;          // frontier row + draft rows
                std::vector<llama_token> gold(rmax, 0);
                std::vector<float>       margins(rmax, 0.0f);  // top-2 logit gap per row
                std::vector<llama_token> seconds(rmax, 0);     // runner-up token per row

                for (int j = 0; j < rmax; ++j) {
                    batch.token[j]    = (j == 0) ? cur : draft[j - 1];
                    batch.pos[j]      = (llama_pos) (pp_tokens + g0 + j);
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j]   = true;
                }
                batch.n_tokens = rmax;

                // T-GPLR-59 D1: the draft head has produced the proposed
                // futures, so publish their table rows BEFORE target decode.
                // The engine settles these copies before returning; no demand
                // wait can cover a speculative completion.
                if (residency_tg_spec_arm == 3) {
                    std::vector<int32_t> d1_tokens(draft.begin(), draft.end());
                    if (!ls.d1_issue_preverify(d1_tokens)) {
                        fprintf(stderr, "error: D1 pre-verify publication failed: %s\n", ls.error_msg.c_str());
                        ls.tel_rowset_refused(0, 0, 0, "d1_preverify", ls.error_msg.c_str());
                        ls.failed = true;
                        exit_status = 1;
                        break;
                    }
                }

                const int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    fprintf(stderr, "error: MTP batch verify decode failed with %d: %s\n", ret, ls.error_msg.c_str());
                    ls.failed = true;
                    exit_status = 1;
                    break;
                }

                for (int j = 0; j < rmax; ++j) {
                    const float * logits = llama_get_logits_ith(ctx, j);
                    if (logits == nullptr) {
                        fprintf(stderr, "error: no logits at MTP batch verify row %d\n", j);
                        ls.failed = true;
                        exit_status = 1;
                        break;
                    }
                    llama_token best = 0;
                    float best_val = logits[0];
                    float second_val = -INFINITY;
                    llama_token second_tok = 0;
                    {
                        // T-GPLR-06: penalized copy; row j's ring view is the
                        // true stream plus the draft chain prefix before row j
                        // (row 0 = the frontier decode sees the true ring).
                        std::vector<float> plogits(logits, logits + n_vocab);
                        rep_penalty_ring row_ring = rep_ring;
                        // row j's ring includes its own input token (the
                        // draft head accepts the input before predicting, so
                        // the verify must see the same ring or acceptance
                        // becomes meaningless under the penalty)
                        for (int k = 0; k <= j; ++k) {
                            row_ring.accept(batch.token[k]);
                        }
                        row_ring.apply(plogits.data(), n_vocab, (float) rep_penalty);
                        best_val = plogits[0];  // penalized seed
                        for (int v = 1; v < n_vocab; ++v) {
                            if (plogits[v] > best_val) { second_val = best_val; second_tok = best; best_val = plogits[v]; best = v; }
                            else if (plogits[v] > second_val) { second_val = plogits[v]; second_tok = v; }
                        }
                    }
                    gold[j]    = best;
                    margins[j] = best_val - second_val;
                    seconds[j] = second_tok;
                }
                if (exit_status != 0) {
                    break;
                }

                // A = committed rows: row 0 always commits (its input cur is
                // the true stream); row j commits iff draft[j-1] was
                // accepted. A capped round (the draft loop hit --tg-tokens)
                // must not push past tg_tokens.
                int A = 1;
                for (int j = 1; j < rmax; ++j) {
                    if (gold[j - 1] != draft[j - 1]) {
                        break;
                    }
                    A = j + 1;
                }
                const int committed = std::min(A, tg_tokens - g0);

                if (residency_tg_spec_arm == 3) {
                    std::vector<uint8_t> d1_accepted((size_t) nd, 0);
                    for (int j = 0; j < nd && j < committed - 1; ++j) {
                        d1_accepted[(size_t) j] = gold[j] == draft[j] ? 1 : 0;
                    }
                    std::vector<int32_t> d1_tokens(draft.begin(), draft.end());
                    if (!ls.d1_observe_round(d1_tokens, d1_accepted)) {
                        fprintf(stderr, "error: D1 route observation failed: %s\n", ls.error_msg.c_str());
                        ls.tel_rowset_refused(0, 0, 0, "d1_observe", ls.error_msg.c_str());
                        ls.failed = true;
                        exit_status = 1;
                        break;
                    }
                }

                for (int j = 0; j < committed; ++j) {
                    generated.push_back(gold[j]);
                    rep_ring.accept(gold[j]);  // T-GPLR-06: only committed tokens penalize
                }
                // Per-round top-2 margins: near-tie rows (< ~0.01 in fp16
                // accumulation) are expected shape-numerics flip candidates;
                // a flip at a large-margin row would instead indicate a
                // state/wiring defect.
                fprintf(stderr, "MTP: round %d rows:", mround - 1);
                for (int j = 0; j < rmax; ++j) {
                    fprintf(stderr, " %d(%.4f|%d)", (int) gold[j], (double) margins[j], (int) seconds[j]);
                }
                fprintf(stderr, "\n");
                fflush(stderr);

                cur = gold[committed - 1];
                const int n_acc = committed - 1;  // row 0 is the frontier, not a draft
                mtp_accepted += (int64_t) n_acc;
                mrounds_batch++;
                if (n_acc == nd) {
                    mrounds_full++;
                } else {
                    mrounds_partial++;
                }

                // The last committed cell's hidden becomes the next round's
                // seed pair (pending_h) and the next round's bridge fill
                // (fill_h) — the serial loop's exact handoff. Slot i holds
                // row i (every row outputs, so slots follow ubatch order).
                {
                    const float * h = llama_get_embeddings_nextn_ith(ctx, committed - 1);
                    if (h == nullptr) {
                        fprintf(stderr, "error: no target nextn row at MTP batch verify slot %d\n", committed - 1);
                        ls.failed = true;
                        exit_status = 1;
                        break;
                    }
                    memcpy(pending_h.data(), h, row_bytes);
                    memcpy(fill_h.data(), h, row_bytes);
                }

                // Rollback the rejected rows (if any): the decode wrote cells
                // vpos..vpos+rmax-1; the true stream keeps rows 0..committed-1
                // only. seq_rm removes the KV rows and rewinds the recurrent
                // snapshot ring (rewind = rmax - committed <= n_rs_seq).
                const llama_pos push_pos = (llama_pos) (pp_tokens + g0 + committed);
                if (committed < rmax) {
                    mrounds_rewind++;
                    llama_memory_t mem_t = llama_get_memory(ctx);
                    if (!llama_memory_seq_rm(mem_t, 0, push_pos, -1)) {
                        fprintf(stderr, "error: MTP batch verify rollback failed at pos %lld (committed %d of %d rows)\n",
                                (long long) push_pos, committed, rmax);
                        ls.failed = true;
                        exit_status = 1;
                        break;
                    }
                }

                // Mirror fill: the committed cells' true stream rows, written
                // as ONE multi-row draft-head decode (the serial path's
                // per-iteration trim+fill, batched). Row 0 is the bridge:
                // (x[vpos], h[vpos-1]) at pos vpos-1 completes the previous
                // round's deferred cell. Rows 1..committed-1 carry
                // (chain[j], h[j-1]) — the mirror row of committed cell j.
                // The LAST committed cell's mirror row is written by the
                // next round's seed decode (cur, pending_h at push_pos-1),
 // exactly like the serial path's deferred fill.
                {
                    llama_memory_t mem_d = llama_get_memory(ctx_dft);
                    if (!llama_memory_seq_rm(mem_d, 0, (llama_pos) (pp_tokens + g0 - 1), -1)) {
                        fprintf(stderr, "error: draft KV trim before batch mirror fill failed at pos %lld\n",
                                (long long) (pp_tokens + g0 - 1));
                        ls.failed = true;
                        exit_status = 1;
                        break;
                    }
                    int n_rows = 0;
                    dbatch.token[n_rows]    = gold[0];
                    dbatch.pos[n_rows]      = (llama_pos) (pp_tokens + g0 - 1);
                    dbatch.n_seq_id[n_rows] = 1;
                    dbatch.seq_id[n_rows][0] = 0;
                    dbatch.logits[n_rows]   = false;
                    memcpy(dbatch.embd, fill_h.data(), row_bytes);
                    n_rows++;
                    for (int j = 1; j < committed; ++j) {
                        const float * h = llama_get_embeddings_nextn_ith(ctx, j - 1);
                        if (h == nullptr) {
                            fprintf(stderr, "error: no target nextn row at batch mirror slot %d\n", j - 1);
                            ls.failed = true;
                            exit_status = 1;
                            break;
                        }
                        dbatch.token[n_rows]    = draft[j - 1];  // chain[j] = gold[j]
                        dbatch.pos[n_rows]      = (llama_pos) (pp_tokens + g0 + j - 1);
                        dbatch.n_seq_id[n_rows] = 1;
                        dbatch.seq_id[n_rows][0] = 0;
                        dbatch.logits[n_rows]   = false;
                        memcpy(dbatch.embd + (size_t) n_rows * n_embd_out, h, row_bytes);
                        n_rows++;
                    }
                if (exit_status != 0) {
                    break;
                }
                dbatch.n_tokens = n_rows;
                    const int mrc = llama_decode(ctx_dft, dbatch);
                    if (mrc != 0) {
                        fprintf(stderr, "error: MTP batch mirror fill failed (%d)\n", mrc);
                        ls.failed = true;
                        exit_status = 1;
                        break;
                    }
                }

                // 3. Re-arm the draft chain for the next round: the head's
                //    KV beyond the frontier was written with hypothetical
                //    draft rows, so trim it back; the next round's first
                //    draft then starts from the accepted cur/pending_h pair
                //    at the fresh frontier.
                if (exit_status == 0 && (int) generated.size() < tg_tokens) {
                    llama_memory_t mem = llama_get_memory(ctx_dft);
                    const llama_pos rm_from = (llama_pos) (pp_tokens + generated.size());
                    if (!llama_memory_seq_rm(mem, 0, rm_from, -1)) {
                        fprintf(stderr, "error: draft KV chain rollback failed at pos %lld\n", (long long) rm_from);
                        ls.failed = true;
                        exit_status = 1;
                    }
                }
                fprintf(stderr, "MTP: round %d: drafted %d, accepted %d, gen %d, next pos %d\n",
                        (int) (mround - 1), (int) draft.size(), n_acc, (int) generated.size(),
                        (int) (pp_tokens + generated.size())); fflush(stderr);
            }
            if (exit_status == 0 && (int) generated.size() < tg_tokens) {
                fprintf(stderr, "error: MTP round guard exhausted at gen %d (no progress?)\n", (int) generated.size());
                ls.failed = true;
                exit_status = 1;
            }
        } else {
            // --- standard greedy TG (unchanged) ---
            // exit_status guard: a failed TG setup (e.g. row-set activation)
            // must not keep decoding (T-GPLR-13 probe evidence).
            for (int t = 0; t < tg_tokens && exit_status == 0; ++t) {
                batch.token[0]   = cur;
                batch.pos[0]     = pp_tokens + t;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0]  = true;
                batch.n_tokens   = 1;

                const int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    fprintf(stderr, "error: TG decode step %d failed with %d: %s\n", t, ret, ls.error_msg.c_str());
                    ls.failed = true;
                    exit_status = 1;
                    break;
                }

                const float * logits = llama_get_logits_ith(ctx, 0);
                if (logits == nullptr) {
                    fprintf(stderr, "error: no logits at TG step %d\n", t);
                    ls.failed = true;
                    exit_status = 1;
                    break;
                }

                llama_token best = 0;
                float best_val = logits[0];
                float second_val = -INFINITY;
                llama_token second_tok = 0;
                {
                    // T-GPLR-06: penalized copy of the serial TG pick
                    std::vector<float> plogits(logits, logits + n_vocab);
                    rep_ring.apply(plogits.data(), n_vocab, (float) rep_penalty);
                    best_val = plogits[0];  // penalized seed
                    for (int v = 1; v < n_vocab; ++v) {
                        if (plogits[v] > best_val) { second_val = best_val; second_tok = best; best_val = plogits[v]; best = v; }
                        else if (plogits[v] > second_val) { second_val = plogits[v]; second_tok = v; }
                    }
                }
                // T-GPLR-06 step 2 diagnosis: top-2 margin of every greedy
                // pick — near-tie steps are where a different ubatch shape
                // (batch verify) may legitimately flip the argmax
                fprintf(stderr, "TG: step %d margin %.6f (gold %d, runner-up %d)\n",
                        t, (double) (best_val - second_val), (int) best, (int) second_tok);
                fflush(stderr);
                // T-GPLR-54 A2: the same penalized top-1/top-2 into the trace,
                // before the sampler state advances (read-only annotation; the
                // emitter self-gates on tg_logit_margin, so an off run pays one
                // integer compare here).
                ls.tel_tg_logits(t, (int64_t) best, best_val, (int64_t) second_tok, second_val);
                cur = best;
                generated.push_back(cur);
                rep_ring.accept(cur);
            }
        }

        const int64_t t1 = ggml_time_us();
        llama_batch_free(batch);
        if (dbatch.token) {
            free(dbatch.token);
            dbatch.token = nullptr;
        }
        llama_batch_free(dbatch);
        if (ctx_dft) {
            llama_free(ctx_dft);
            ctx_dft = nullptr;
        }

        // T-Q38HG-02: tear down the handoff buffer after TG completes
        if (handoff && exit_status == 0) {
            ls.end_handoff();
        }

        if (residency_tg_spec_arm == 3 && exit_status == 0) {
            ls.d1_emit_summary();
        }

        // T-GPLR-13: deactivate row-set staging and free the row-set buffers
        // (no-op when the flag is off or the run already failed)
        ls.end_row_set();

        if (exit_status == 0) {
            // the TG graph set was prebuilt before the PP timer, so the phase
            // baseline is the build_generation captured at prebuild completion
            ls.tel_phase_summary("tg", tg_tokens, t1 - t0, builds0, graph_prep_us);
            if (mtp) {
                ls.tel_mtp((int64_t) generated.size(), mtp_drafted, mtp_accepted);
            }

            // emit the greedy token IDs as a single telemetry event
            std::vector<int64_t> ids;
            ids.reserve(generated.size());
            for (llama_token tok : generated) {
                ids.push_back((int64_t) tok);
            }
            ls.tel_tg_tokens(ids);
            // T-GPLR-03: human-readable detokenized first token for parity
            // comparison against the normal-path control (llama-cli)
            {
                const llama_vocab * vocab = llama_model_get_vocab(model);
                std::string piece;
                for (llama_token tok : generated) {
                    char buf[64];
                    const int n = llama_token_to_piece(vocab, tok, buf, (int) sizeof(buf), 0, true);
                    if (n > 0) {
                        piece.append(buf, (size_t) n);
                    }
                }
                fprintf(stderr, "TG text: \"%s\"\n", piece.c_str());
            }
            fprintf(stderr, "TG: %d tokens in %.1f ms (%.2f tok/s)\n",
                tg_tokens, (t1 - t0) / 1000.0, tg_tokens * 1e6 / (double) (t1 - t0));
            if (mtp) {
                fprintf(stderr, "MTP: drafted %lld, accepted %lld, acceptance %.3f\n",
                    (long long) mtp_drafted, (long long) mtp_accepted,
                    mtp_drafted > 0 ? (double) mtp_accepted / (double) mtp_drafted : 0.0);
            }
            if (mtp && mrounds_batch > 0) {
                fprintf(stderr, "MTP batch verify: rounds %lld, full-accept %lld, partial-accept %lld (avg accepted/round %.2f), rollback rounds %lld\n",
                    (long long) mrounds_batch, (long long) mrounds_full, (long long) mrounds_partial,
                    (double) mtp_accepted / (double) mrounds_batch, (long long) mrounds_rewind);
            }
            fprintf(stderr, "first 16 generated ids: ");
            for (int i = 0; i < (int) generated.size() && i < 16; ++i) {
                fprintf(stderr, "%d ", generated[i]);
            }
            fprintf(stderr, "\n");

            // For real prompts, detokenize the generated tokens to text
            if (!prompt_tokens.empty() && !generated.empty()) {
                const llama_vocab * vocab = llama_model_get_vocab(model);
                fprintf(stderr, "\n--- generated text ---\n");
                for (llama_token tok : generated) {
                    char buf[256];
                    const int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
                    if (n > 0) {
                        fwrite(buf, 1, n, stderr);
                    }
                }
                fprintf(stderr, "\n--- end ---\n");
            }

            // T-GPLR-07: --tg-text writes the detokenized generated text to a
            // file so the ladder checker can grade the answer without parsing
            // logs. Same special=false detokenization as the section above
            // (byte-identical piece rendering), written whenever TG produced
            // tokens regardless of the prompt source. A failed write is loud
            // and fails the run (exit_status recorded in run_end); it must
            // never silently produce a missing answer file.
            if (!tg_text_path.empty() && !generated.empty()) {
                FILE * tf = fopen(tg_text_path.c_str(), "wb");
                if (tf == nullptr) {
                    fprintf(stderr, "error: cannot write tg-text file %s\n", tg_text_path.c_str());
                    exit_status = 1;
                } else {
                    const llama_vocab * vocab = llama_model_get_vocab(model);
                    for (llama_token tok : generated) {
                        char buf[1024];
                        int n = llama_token_to_piece(vocab, tok, buf, (int) sizeof(buf), 0, false);
                        if (n < 0) {
                            // piece longer than the buffer: retry with the
                            // required size (negative return = needed bytes)
                            std::vector<char> big((size_t) -n);
                            n = llama_token_to_piece(vocab, tok, big.data(), (int) big.size(), 0, false);
                            if (n > 0) {
                                fwrite(big.data(), 1, (size_t) n, tf);
                            }
                            continue;
                        }
                        if (n > 0) {
                            fwrite(buf, 1, (size_t) n, tf);
                        }
                    }
                    fclose(tf);
                    fprintf(stderr, "TG text written to %s\n", tg_text_path.c_str());
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // summary + telemetry close
    // ------------------------------------------------------------------
    {
        // T-GPLR-24 (H-B): the free_after sample moved into the free_min
        // tracker — every large allocation/release between init and here now
        // stays visible to the reported peak.
        // T-GPLR-08: effective feed-rate summary (bytes / prefetch duration).
        // Zeros when the probe is off (no events, no columns) — the N=0 path
        // must stay byte-identical to today.
        if (pp_prefetch_ahead > 0) {
            uint64_t pf_s = 0, pf_c = 0, pf_e = 0, pf_b = 0, pf_us = 0;
            ls.pf_stats(pf_s, pf_c, pf_e, pf_b, pf_us);
            const double pf_rate = pf_us > 0 ? (double) pf_b / ((double) pf_us / 1e6) / (1024.0 * 1024.0) : 0.0;
            fprintf(stderr, "prefetch probe (ahead=%d): submit=%llu complete=%llu err=%llu bytes=%llu us=%llu feed=%.1f MB/s\n",
                    pp_prefetch_ahead, (unsigned long long) pf_s, (unsigned long long) pf_c,
                    (unsigned long long) pf_e, (unsigned long long) pf_b, (unsigned long long) pf_us, pf_rate);
        }

        fprintf(stderr, "\nper-layer averages (over executed ubatches):\n");
        fprintf(stderr, "%6s %12s %12s %12s %12s %10s\n", "layer", "transfer_us", "wait_us", "compute_us", "setup_us", "type");
        for (int il = 0; il < n_layer; ++il) {
            const auto & st = ls.stats[il];
            if (st.n == 0) {
                continue;
            }
            fprintf(stderr, "%6d %12.1f %12.1f %12.1f %12.1f %10s\n",
                il,
                st.transfer_us / (double) st.n,
                st.wait_us / (double) st.n,
                st.compute_us / (double) st.n,
                st.setup_us / (double) st.n,
                ls.is_recr(il) ? "recurrent" : "full-attn");
        }

        // T-GPLR-24 (H-B): report the run's true low-water mark. The old
        // two-sample max (free_start vs free_after) missed every allocation
        // released before this point — notably the T-GPLR-19 row cache —
        // understating peak usage by exactly those bytes.
        ls.tel_run_end(exit_status, vram.total - vram.free_min,
                       (int64_t) generated.size(),
                       ls.error_msg.c_str());
        ls.stop_prefetch_worker(); // T-GPLR-08: join after run_end closes the sink
    }

    // inline self-check: re-validate the trace that was just written and
    // print the per-phase telemetry summary (tok/s, transfer/wait/compute,
    // evictions, graph builds, peak VRAM). Fails the run if the trace
    // violates the scheduler invariants. Boundary mode (--boundary-layers)
    // executes a reduced layer subset by design, so the full-run deterministic
    // slot-policy invariant does not apply and the check is skipped there.
    if (!tel_path.empty() && boundary_layers.empty()) {
        const int rc = check_telemetry(tel_path);
        if (exit_status == 0 && rc != 0) {
            exit_status = 1;
        }
    } else if (!tel_path.empty()) {
        fprintf(stderr, "(boundary mode: skipping trace self-check; deterministic slot policy does not apply to a reduced layer subset)\n");
    }

    ctx->set_layer_streaming(nullptr);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return exit_status;
}
