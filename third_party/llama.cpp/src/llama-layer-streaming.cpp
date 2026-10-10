// Experimental CUDA-only decoder-layer weight streaming (private research
// path, gated by LLAMA_LAYER_STREAMING). Not an upstream-facing feature.
//
// Execution model per ubatch:
//   the graph builder produces one ggml_cgraph per piece: embed graph, one
//   graph per logical decoder layer, and the output graph. Layer graphs read
//   the persistent activation pair and write the next one (and write their
//   logical KV / Gated DeltaNet state in-graph, exactly like the normal path).
//   The executor, before computing layer L, ensures L's package is resident in
//   slot L % n_slots: it issues the async H2D copies on a dedicated copy
//   backend (separate stream), records a copy-ready event, then the compute
//   backend waits on that event before running the layer graph. A slot is
//   overwritten only after the previous consumer's done event fires.
//
// Timing semantics per layer (all wall-clock, distinct in telemetry):
//   transfer_us : copy submit -> copy complete (includes queueing)
//   wait_us     : host block at consumption until the slot copy is complete
//   setup_us    : scheduler reset + graph allocation
//   compute_us  : async launch + host block until the layer graph completes

#ifdef LLAMA_LAYER_STREAMING

#include "llama-layer-streaming.h"
#include "llama-expert-split-policy.h"

#include <chrono>
#include <thread>

#include "llama-context.h"
#include "llama-graph.h"
#include "llama-memory-hybrid.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "../ggml/src/ggml-impl.h" // struct ggml_cgraph for the debug dump

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <unordered_set>

// T-GPLR-08: one-layer-ahead PrefetchVirtualMemory probe. Windows-only, hint
// semantics: a failed/skipped hint degrades to today's page-fault path and can
// never produce wrong data. The worker is a private member (pf_worker, defined
// below); all shared state is mutex-guarded; the worker never runs on the
// copy/compute thread.
#if defined(_WIN32) && _WIN32_WINNT >= 0x602
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define LS_PF_ENABLED 1
#endif
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <system_error>
#include <thread>

// ---------------------------------------------------------------------------
// T-GPLR-01: common package-descriptor plumbing.
//
// Architecture member lists live behind the llama_ls_adapter interface
// (src/models/llama-layer-streaming-<arch>.cpp). This file only consumes
// validated descriptors: it never inspects Qwen tensor names or builds a
// package classification of its own.
// ---------------------------------------------------------------------------

// resolve the destination tensor slot of a member (base or nested nextn)
static ggml_tensor *& ls_dst(llama_layer & l, const llama_ls_member & m) {
    GGML_ASSERT((m.dst_base == nullptr) != (m.dst_nextn == nullptr) && "member needs exactly one destination");
    return m.dst_base ? l.*(m.dst_base) : l.nextn.*(m.dst_nextn);
}

static const ggml_tensor * ls_dst(const llama_layer & l, const llama_ls_member & m) {
    GGML_ASSERT((m.dst_base == nullptr) != (m.dst_nextn == nullptr) && "member needs exactly one destination");
    return m.dst_base ? l.*(m.dst_base) : l.nextn.*(m.dst_nextn);
}

bool llama_ls_validate_package(const llama_ls_package & pkg, std::string & err, bool require_src) {
    if (pkg.members.empty()) {
        err = "package has no members";
        return false;
    }
    if (!pkg.excludes_static || !pkg.excludes_state) {
        err = "package must declare static-weight and mutable-state exclusions";
        return false;
    }
    if (pkg.cls[0] == '\0' || pkg.policy[0] == '\0') {
        err = "package must carry a class and policy label";
        return false;
    }
    for (const auto & m : pkg.members) {
        if (m.mutable_state) {
            err = "mutable state member in package";
            return false;
        }
        if ((m.dst_base == nullptr) == (m.dst_nextn == nullptr)) {
            err = "member needs exactly one destination slot";
            return false;
        }
        if (require_src && m.src == nullptr) {
            err = "member without a source tensor";
            return false;
        }
        if (m.src != nullptr && ggml_nbytes(m.src) == 0) {
            err = "zero-byte source tensor in package";
            return false;
        }
    }
    // duplicate destination ranges: the same member listed twice would alias
    for (size_t i = 0; i < pkg.members.size(); ++i) {
        for (size_t j = i + 1; j < pkg.members.size(); ++j) {
            const auto & a = pkg.members[i];
            const auto & b = pkg.members[j];
            if ((a.dst_base && a.dst_base == b.dst_base) ||
                (a.dst_nextn && a.dst_nextn == b.dst_nextn)) {
                err = "duplicate package destination member";
                return false;
            }
        }
    }
    return true;
}

const llama_ls_adapter * llama_ls_select_adapter(llm_arch arch, std::string & reason) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
            return llama_ls_qwen35_adapter();
        case LLM_ARCH_QWEN35MOE:
            return llama_ls_qwen35moe_adapter();
        default:
            reason = "no layer-residency adapter for this architecture";
            return nullptr;
    }
}

// T-GPLR-05: host-only capability probe (SDD section 6, adapter duty 1).
// Reads only the GGUF header KV via llama_ls_model_arch (no CUDA, no tensor
// data, no normal model loading) and returns a deterministic result for every
// input: an adapter-declared policy, or off with a stable reason. Forced
// policy validation happens in the runner; this probe never allocates.
std::string llama_ls_probe_capability(const char * model_path, llama_ls_capability & cap) {
    cap = {};
    if (model_path == nullptr || model_path[0] == '\0') {
        cap.supported = false;
        cap.reason = "missing model path";
        return "off";
    }
    const std::string arch_name = llama_ls_model_arch(model_path);
    if (arch_name.empty()) {
        cap.supported = false;
        cap.reason = "unreadable or missing model file (no general.architecture)";
        return "off";
    }
    cap.arch = arch_name;
    std::string sel_reason;
    const llama_ls_adapter * adapter = llama_ls_select_adapter(llm_arch_from_string(arch_name), sel_reason);
    if (adapter == nullptr) {
        cap.supported = false;
        cap.reason = sel_reason.empty() ? "no layer-residency adapter for this architecture" : sel_reason;
        return "off";
    }
    cap.supported = true;
    cap.adapter = adapter->name;
    cap.policy = adapter->policy;
    cap.reason = "adapter-declared policy";
    return cap.policy;
}

// T-Q38CTX-01: pure host-side chunk layout. Splits pp_tokens into full
// ubatch-sized chunks plus one optional 1..ubatch-1 token tail. Contiguous,
// starts at 0, covers exactly pp_tokens. Empty on invalid input.
std::vector<llama_ls_chunk> llama_ls_chunk_layout(int64_t pp_tokens, int ubatch) {
    std::vector<llama_ls_chunk> out;
    if (pp_tokens <= 0 || ubatch <= 0) {
        return out;
    }
    const int64_t n = (pp_tokens + ubatch - 1) / ubatch;
    out.reserve((size_t) n);
    for (int64_t c = 0; c < n; ++c) {
        llama_ls_chunk ck;
        ck.token_start = c * (int64_t) ubatch;
        ck.n_tokens    = std::min<int64_t>((int64_t) ubatch, pp_tokens - ck.token_start);
        out.push_back(ck);
    }
    return out;
}

// T-GPLR-49: the ONE owner of the layer-major slab element-size arithmetic.
// The pre-T-GPLR-49 begin_set offset formula used sizeof(float) for the bytes
// per token while slab_slice bound slices sized with
// ggml_type_size(act_slab_dtype); at bf16 -- the default since T-GPLR-12 --
// those differ by exactly 2x, so every chunk >= 1 was placed at twice its
// descriptor-derived offset. Deriving here, purely, also removes the second
// defect of that formula: chunk 0 skipped it entirely and inherited whatever
// slab_off_A/B held. Nothing here reads or writes a cursor.
bool llama_ls_slab_region_of(const llama_ls_chunk * chunks, size_t n_chunks, size_t chunk_id,
                             int64_t n_embd_io, ggml_type act_dtype, llama_ls_slab_region * out) {
    if (chunks == nullptr || out == nullptr || n_chunks == 0 || n_embd_io <= 0) {
        return false;
    }
    const size_t elt = ggml_type_size(act_dtype);
    if (elt == 0) {
        return false;
    }
    if (chunk_id >= n_chunks) {
        return false;
    }
    const size_t row = (size_t) n_embd_io * elt;   // bytes per token, in the slab dtype
    size_t offset = 0;
    size_t half   = 0;
    for (size_t c = 0; c < n_chunks; ++c) {
        if (chunks[c].n_tokens <= 0) {
            return false;
        }
        const size_t b = (size_t) chunks[c].n_tokens * row;
        if (c < chunk_id) {
            offset += b;
        }
        half += b;
    }
    const size_t bytes = (size_t) chunks[chunk_id].n_tokens * row;
    if (offset + bytes > half) {
        return false;
    }
    out->offset = offset;
    out->bytes  = bytes;
    out->half   = half;
    return true;
}

bool llama_ls_slab_overlaps(size_t a_start, size_t a_end, size_t b_start, size_t b_end) {
    if (a_start >= a_end || b_start >= b_end) {
        return false;   // a degenerate (empty) interval occupies nothing
    }
    return a_start < b_end && b_start < a_end;
}

// T-GPLR-39: exact top-k expert IDs of one router probability row. Equivalent
// to sorting the whole row stable-descending (probability descending, ties by
// ascending expert index -- the CUDA router's stable radix-sort tie-break) and
// keeping the first k, but computed as a bounded insertion pass over the row:
// O(n_expert * k) with no per-row allocation and no indirect comparator. That
// equivalence is what lets tel_route keep emitting identical ids, hash and
// union while dropping the per-token sort of all n_expert experts that the
// route telemetry was paying for (measured as the ubatch-scaled `other`
// residue, T-GPLR-38/39). Softmax probabilities are finite and non-negative;
// NaN is outside the equivalence domain (see the header comment).
void llama_ls_route_topk_row(const float * probs, int64_t n_expert, int64_t k, int32_t * ids_out) {
    if (probs == nullptr || ids_out == nullptr || n_expert <= 0 || k <= 0) {
        return;
    }
    const int64_t kk = (k < n_expert) ? k : n_expert;
    // ids_out[0..min(e+1, kk)-1] always holds the best ids of experts 0..e in
    // stable descending order.
    for (int64_t e = 0; e < n_expert; ++e) {
        const float p = probs[e];
        int64_t j;
        if (e < kk) {
            j = e;
        } else if (p > probs[ids_out[kk - 1]]) {
            j = kk - 1;
        } else {
            continue; // not strictly better than the current k-th: outside the top k
        }
        // Shift entries strictly worse than p to the right; entries with equal
        // probability stay left of e, which keeps ties in ascending expert
        // index order (stability).
        while (j > 0 && probs[ids_out[j - 1]] < p) {
            ids_out[j] = ids_out[j - 1];
            --j;
        }
        ids_out[j] = (int32_t) e;
    }
}

// ---------------------------------------------------------------------------
// T-GPLR-58 A0: the speculative-placement rules, pure and host-testable. See the
// header for the contract (P2 readiness, P3 the demand reserve, P4 cancellation,
// P5 no displacement); tests/test-layer-streaming-tg-spec.cpp drives these
// directly with scripted sequences and no CUDA.
// ---------------------------------------------------------------------------

bool llama_ls_spec_is_hit(uint8_t state) {
    // P2: ONLY a row whose own copy completed may be served. FREE (nothing),
    // PENDING (copy in flight) and CANCELLED (abandoned) are all misses, so a
    // demand for a row that a speculative fill is copying fetches its own.
    return state == LLAMA_LS_SPEC_READY;
}

bool llama_ls_spec_may_evict(uint8_t state) {
    // A position whose bytes are in flight -- or whose cancellation is still
    // awaiting its copy's completion -- must never be reused: the reap releases
    // it, an eviction never does.
    return state == LLAMA_LS_SPEC_READY || state == LLAMA_LS_SPEC_FREE;
}

int llama_ls_spec_free_pos(const int32_t * experts, const uint8_t * states, int cap) {
    // The PHYSICAL occupancy is the truth: a position with no expert holds no
    // row, so it is free. The state array is consulted only to keep the two in
    // step (a PENDING or CANCELLED position always HAS an expert, so this is not
    // an escape hatch for an in-flight copy -- the differential test pins that a
    // stale state can never make a free position unavailable).
    for (int p = 0; p < cap; ++p) {
        if (experts[p] >= 0) {
            continue; // physically occupied
        }
        if (states != nullptr && states[p] == LLAMA_LS_SPEC_PENDING) {
            continue; // an in-flight fill must have its expert; refuse to guess otherwise
        }
        return p;
    }
    return -1;
}

bool llama_ls_spec_victim_better(int64_t a_recency, uint8_t a_spec, bool a_has,
                                 int64_t b_recency, uint8_t b_spec, bool b_has) {
    if (!a_has) {
        return false; // nothing to prefer
    }
    if (!b_has) {
        return true; // first candidate always wins over "none"
    }
    if (a_spec != b_spec) {
        return a_spec > b_spec; // P3: a speculative row is the FIRST victim
    }
    // Then the incumbent's own rule, unchanged: least recently routed loses.
    // The strict `<` keeps the LOWEST position on a tie, which is what the
    // per-window in-window tie-break matches.
    return a_recency < b_recency;
}

llama_ls_spec_admission llama_ls_spec_admit(const int32_t * experts, const uint8_t * states, int cap,
                                            int reserve, int max_pending, int n_pending, int32_t expert) {
    llama_ls_spec_admission r;
    if (expert < 0) {
        r.reason = "bad_shape";
        return r;
    }
    if (n_pending >= max_pending) {
        r.reason = "budget_exhausted";
        return r;
    }
    int first_free = -1;
    int n_free     = 0;
    for (int p = 0; p < cap; ++p) {
        const uint8_t st = states != nullptr ? states[p] : LLAMA_LS_SPEC_FREE;
        if (st == LLAMA_LS_SPEC_PENDING || st == LLAMA_LS_SPEC_CANCELLED) {
            if (experts[p] == expert) {
                r.reason = "already_pending"; // never two fills for one row
                return r;
            }
            continue;
        }
        if (st == LLAMA_LS_SPEC_READY) {
            if (experts[p] == expert) {
                r.reason = "already_resident";
                return r;
            }
            continue;
        }
        if (experts[p] >= 0) {
            continue; // unreachable in a consistent cache: a free position holds no row
        }
        n_free++;
        if (first_free < 0) {
            first_free = p;
        }
    }
    if (first_free < 0) {
        r.reason = "no_free_position";
        return r;
    }
    // P3's hard barrier: after this admission at least `reserve` positions stay
    // FREE for the demand side, so a demand can always place its own rows
    // without reclaiming -- or waiting on -- a speculative fill.
    if (n_free - 1 < reserve) {
        r.reason = "demand_reserve";
        return r;
    }
    r.ok     = true;
    r.pos    = first_free;
    r.reason = "";
    return r;
}

int llama_ls_spec_cancel_plan(const int64_t * recencies, const uint8_t * states, int cap, int need_free,
                              int * out_positions, int max_out) {
    if (need_free <= 0 || out_positions == nullptr || max_out <= 0 || recencies == nullptr || states == nullptr) {
        return 0;
    }
    int n = 0;
    std::vector<char> taken((size_t) cap, 0);
    while (n < need_free && n < max_out) {
        int best = -1;
        for (int p = 0; p < cap; ++p) {
            if (states[p] != LLAMA_LS_SPEC_PENDING || taken[(size_t) p]) {
                continue;
            }
            if (best < 0 || recencies[p] < recencies[best]) {
                best = p; // OLDEST (smallest recency) first
            }
        }
        if (best < 0) {
            break;
        }
        taken[(size_t) best] = 1;
        out_positions[n++]   = best;
    }
    return n;
}
// T-GPLR-58 A2 (M7): the LEAD of one consumed fill, in demand stages of its own
// layer -- the head start a fill actually got before a demand was served from it.
// The issuing stage counts as 0, so a fill issued at the end of stage t and consumed
// by stage t+2 has lead 2. Returns 0 when there is no fill to credit (`issue_ord`
// < 0) or when the fill was consumed in (or before) the stage that issued it: 0
// therefore always means "not a consumed speculative row", never "a consumed fill
// with no head start".
int llama_ls_spec_lead(int32_t consume_ord, int32_t issue_ord) {
    if (issue_ord < 0) {
        return 0; // the position holds an ordinary resident, not a fill
    }
    if (consume_ord <= issue_ord) {
        return 0; // consumed in (or before) its own issuing stage: no head start
    }
    return (int) (consume_ord - issue_ord);
}

// M7's histogram bucket for a lead: 1..7 -> 0..6, >= 8 -> 7 (the overflow bucket),
// and -1 for a lead that is not a consumption (see llama_ls_spec_lead). The engine
// keeps exactly 8 counters, so the emitted distribution always has this shape.
int llama_ls_spec_lead_bucket(int lead) {
    if (lead < 1) {
        return -1;
    }
    return lead >= 8 ? 7 : lead - 1;
}

// ---------------------------------------------------------------------------
// T-GPLR-58 A1: the proposer families, the two merge operators, the
// structure-free null and the planner. Pure policy -- the learning tables are
// the caller's state, and nothing here touches the engine, a clock or a file.
// Every matching rule is ported verbatim from tools/predictor-screen-gplr.py
// (the reference is pinned by file hash in the A1 result); the comment on each
// step names the screen's line it reproduces, because "looks equivalent" is
// exactly the claim this port must not make.
// ---------------------------------------------------------------------------

void llama_ls_pred_init(llama_ls_pred_state * st, int n_layer, int n_expert, uint32_t families) {
    if (st == nullptr) {
        return;
    }
    st->n_layer  = n_layer > 0 ? n_layer : 0;
    st->n_expert = n_expert > 0 ? n_expert : 0;
    st->families = families;
    const size_t n_l = (size_t) st->n_layer;
    const size_t n_e = (size_t) st->n_expert;
    st->prev.assign(n_l, std::vector<int32_t>());
    st->trans.assign((families & LLAMA_LS_PRED_BIT_MARKOV1)  ? n_l * n_e * n_e : 0, 0);
    st->pair.assign( (families & LLAMA_LS_PRED_BIT_COACT)    ? n_l * n_e * n_e : 0, 0);
    st->unit.assign( (families & LLAMA_LS_PRED_BIT_COACT)    ? n_l * n_e       : 0, 0);
    st->joint.assign((families & LLAMA_LS_PRED_BIT_COUPLING) ? n_l * n_e * n_e : 0, 0);
    st->marg.assign( (families & LLAMA_LS_PRED_BIT_COUPLING) ? n_l * n_e       : 0, 0);
}

int llama_ls_pred_predict(const llama_ls_pred_state * st, int family, int layer, int b,
                          int32_t * out, int max_out) {
    if (st == nullptr || out == nullptr || b <= 0 || max_out <= 0) {
        return 0;
    }
    if (layer < 0 || layer >= st->n_layer || st->n_expert <= 0) {
        return 0;
    }
    if (family < 0 || family > (int) LLAMA_LS_PRED_COUPLING) {
        return 0;
    }
    if ((st->families & (1u << family)) == 0) {
        return 0; // that family's table was never allocated: refuse, never guess
    }
    const int n_e  = st->n_expert;
    const int want = b < max_out ? b : max_out;

    if (family == LLAMA_LS_PRED_HISTORY) {
        // screen: History.predict -> list(self.prev.get(layer, ()))[:b]
        const std::vector<int32_t> & pv = st->prev[(size_t) layer];
        const int n = (int) pv.size() < want ? (int) pv.size() : want;
        for (int i = 0; i < n; ++i) {
            out[i] = pv[(size_t) i];
        }
        return n;
    }

    // The two scored families share one accumulation shape: one term per source
    // expert, added in the OBSERVED order of that row's ids. Per candidate id
    // that is one term per source, so the accumulation order -- and therefore
    // every last bit of the score -- is identical to the screen's.
    std::vector<double> score((size_t) n_e, 0.0);
    std::vector<char>   has((size_t) n_e, 0);

    if (family == LLAMA_LS_PRED_COUPLING) {
        // screen: CouplingCopula.predict -> prev = prev.get(layer - 1); if not prev: []
        if (layer < 1) {
            return 0;
        }
        const std::vector<int32_t> & pv = st->prev[(size_t) (layer - 1)];
        if (pv.empty()) {
            return 0;
        }
        for (size_t i = 0; i < pv.size(); ++i) {
            const int32_t e = pv[i];
            if (e < 0 || e >= n_e) {
                continue;
            }
            const int32_t * row = st->joint.data() +
                                  (((size_t) (layer - 1) * (size_t) n_e) + (size_t) e) * (size_t) n_e;
            for (int nxt = 0; nxt < n_e; ++nxt) {
                if (row[nxt] == 0) {
                    continue; // screen skips absent dict entries
                }
                const int32_t mg = st->marg[(size_t) layer * (size_t) n_e + (size_t) nxt];
                if (mg == 0) {
                    continue; // screen: if marg_e:
                }
                score[(size_t) nxt] += (double) row[nxt] / (double) mg;
                has[(size_t) nxt] = 1;
            }
        }
    } else {
        const std::vector<int32_t> & pv = st->prev[(size_t) layer];
        if (pv.empty()) {
            return 0; // screen: if not prev: return []
        }
        for (size_t i = 0; i < pv.size(); ++i) {
            const int32_t e = pv[i];
            if (e < 0 || e >= n_e) {
                continue;
            }
            if (family == LLAMA_LS_PRED_MARKOV1) {
                // screen: Markov1.predict -> scores[nxt] += c over trans[(layer, e)]
                const int32_t * row = st->trans.data() +
                                      (((size_t) layer * (size_t) n_e) + (size_t) e) * (size_t) n_e;
                for (int nxt = 0; nxt < n_e; ++nxt) {
                    if (row[nxt] == 0) {
                        continue;
                    }
                    score[(size_t) nxt] += (double) row[nxt]; // integer counts: exact
                    has[(size_t) nxt] = 1;
                }
            } else {
                // screen: Coact.predict -> scores[nxt] += c / (unit[(layer, e)] or 1)
                const int32_t * row = st->pair.data() +
                                      (((size_t) layer * (size_t) n_e) + (size_t) e) * (size_t) n_e;
                const double unit = (double) st->unit[(size_t) layer * (size_t) n_e + (size_t) e];
                const double den  = unit > 0.0 ? unit : 1.0;
                for (int nxt = 0; nxt < n_e; ++nxt) {
                    if (row[nxt] == 0) {
                        continue;
                    }
                    score[(size_t) nxt] += (double) row[nxt] / den;
                    has[(size_t) nxt] = 1;
                }
            }
        }
    }

    // screen: sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])) -- ties go
    // to the SMALLER expert id, which is why the id is the second key here.
    std::vector<int32_t> idx;
    idx.reserve((size_t) n_e);
    for (int e = 0; e < n_e; ++e) {
        if (has[(size_t) e]) {
            idx.push_back((int32_t) e);
        }
    }
    std::sort(idx.begin(), idx.end(), [&score](int32_t a, int32_t c) {
        const double sa = score[(size_t) a];
        const double sc = score[(size_t) c];
        if (sa != sc) {
            return sa > sc;
        }
        return a < c;
    });
    const int n = (int) idx.size() < want ? (int) idx.size() : want;
    for (int i = 0; i < n; ++i) {
        out[i] = idx[(size_t) i];
    }
    return n;
}

// The bar-0.9 operator over markov-1: publish rank r while 0.9^r >= 0.5.
// screen: Barred.predict -> out = [raw[0]]; while r < len(raw) and bar**r >= 0.5: ...
static int ls_pred_bar09(const llama_ls_pred_state * st, int layer, int b, int32_t * out, int max_out) {
    int32_t raw[LLAMA_LS_PRED_MAX_OUT];
    const int n_raw = llama_ls_pred_predict(st, LLAMA_LS_PRED_MARKOV1, layer, b,
                                            raw, LLAMA_LS_PRED_MAX_OUT);
    if (n_raw <= 0) {
        return 0;
    }
    int n = 0;
    out[n++] = raw[0]; // the top candidate is always published
    for (int r = 1; r < n_raw && n < max_out; ++r) {
        if (std::pow(0.9, (double) r) < 0.5) {
            break;
        }
        out[n++] = raw[r];
    }
    return n;
}

// screen: VoteFusion.predict -- agreement count first (deduped per source), then
// the summed w*decay^r, then the smaller id. Param-free: every weight is 1.0 and
// decay is 0.9, which is the operator arm-fusion declares.
static int ls_pred_vote_fusion(const llama_ls_pred_state * st, int layer, int b, int32_t * out, int max_out) {
    static const int kFams[4] = { LLAMA_LS_PRED_HISTORY, LLAMA_LS_PRED_MARKOV1,
                                  LLAMA_LS_PRED_COACT,   LLAMA_LS_PRED_COUPLING };
    const int n_e  = st->n_expert;
    const int want = b < max_out ? b : max_out;
    std::vector<double>  score((size_t) n_e, 0.0);
    std::vector<int32_t> votes((size_t) n_e, 0);
    std::vector<char>    has((size_t) n_e, 0);
    int32_t ranked[LLAMA_LS_PRED_MAX_OUT];
    for (int f = 0; f < 4; ++f) {
        const int n = llama_ls_pred_predict(st, kFams[f], layer, b, ranked, LLAMA_LS_PRED_MAX_OUT);
        std::vector<char> seen((size_t) n_e, 0); // in-source dedup (dict.fromkeys)
        for (int i = 0; i < n; ++i) {
            const int32_t e = ranked[i];
            if (!seen[(size_t) e]) {
                seen[(size_t) e] = 1;
                votes[(size_t) e] += 1;
            }
        }
        for (int r = 0; r < n; ++r) {
            const int32_t e = ranked[r];
            score[(size_t) e] += 1.0 * std::pow(0.9, (double) r);
            has[(size_t) e] = 1;
        }
    }
    std::vector<int32_t> idx;
    idx.reserve((size_t) n_e);
    for (int e = 0; e < n_e; ++e) {
        if (has[(size_t) e]) {
            idx.push_back((int32_t) e);
        }
    }
    std::sort(idx.begin(), idx.end(), [&votes, &score](int32_t a, int32_t c) {
        if (votes[(size_t) a] != votes[(size_t) c]) {
            return votes[(size_t) a] > votes[(size_t) c];
        }
        const double sa = score[(size_t) a];
        const double sc = score[(size_t) c];
        if (sa != sc) {
            return sa > sc;
        }
        return a < c;
    });
    const int n = (int) idx.size() < want ? (int) idx.size() : want;
    for (int i = 0; i < n; ++i) {
        out[i] = idx[(size_t) i];
    }
    return n;
}

int llama_ls_pred_arm_predict(const llama_ls_pred_state * st, int arm, int layer, int b,
                              int32_t * out, int max_out) {
    if (st == nullptr || out == nullptr) {
        return 0;
    }
    switch (arm) {
        case LLAMA_LS_ARM_COACT:
            return llama_ls_pred_predict(st, LLAMA_LS_PRED_COACT, layer, b, out, max_out);
        case LLAMA_LS_ARM_BAR09:
            return ls_pred_bar09(st, layer, b, out, max_out);
        case LLAMA_LS_ARM_FUSION:
            return ls_pred_vote_fusion(st, layer, b, out, max_out);
        default:
            return 0;
    }
}

void llama_ls_pred_observe(llama_ls_pred_state * st, int layer, const int32_t * ids, int n_ids) {
    if (st == nullptr || layer < 0 || layer >= st->n_layer) {
        return;
    }
    const int n_e = st->n_expert;

    if (!st->trans.empty()) {
        // screen: Markov1.observe -> trans[(layer, e)][nxt] += 1 for e in prev, nxt in ids
        const std::vector<int32_t> & pv = st->prev[(size_t) layer];
        for (size_t i = 0; i < pv.size(); ++i) {
            const int32_t e = pv[i];
            if (e < 0 || e >= n_e) {
                continue;
            }
            int32_t * row = st->trans.data() +
                            (((size_t) layer * (size_t) n_e) + (size_t) e) * (size_t) n_e;
            for (int j = 0; j < n_ids; ++j) {
                if (ids[j] >= 0 && ids[j] < n_e) {
                    row[ids[j]] += 1;
                }
            }
        }
    }

    if (!st->pair.empty()) {
        // screen: Coact.observe -> pair[(layer, e)][nxt] += 1 and unit[(layer, e)] += 1
        const std::vector<int32_t> & pv = st->prev[(size_t) layer];
        for (size_t i = 0; i < pv.size(); ++i) {
            const int32_t e = pv[i];
            if (e < 0 || e >= n_e) {
                continue;
            }
            int32_t * row = st->pair.data() +
                            (((size_t) layer * (size_t) n_e) + (size_t) e) * (size_t) n_e;
            for (int j = 0; j < n_ids; ++j) {
                if (ids[j] >= 0 && ids[j] < n_e) {
                    row[ids[j]] += 1;
                }
            }
            st->unit[(size_t) layer * (size_t) n_e + (size_t) e] += 1;
        }
    }

    if (!st->joint.empty()) {
        // screen: CouplingCopula.observe -> joint[(layer-1, e)][nxt] += 1, marg[(layer, nxt)] += 1
        if (layer >= 1) {
            const std::vector<int32_t> & pv = st->prev[(size_t) (layer - 1)];
            for (size_t i = 0; i < pv.size(); ++i) {
                const int32_t e = pv[i];
                if (e < 0 || e >= n_e) {
                    continue;
                }
                int32_t * row = st->joint.data() +
                                (((size_t) (layer - 1) * (size_t) n_e) + (size_t) e) * (size_t) n_e;
                for (int j = 0; j < n_ids; ++j) {
                    if (ids[j] >= 0 && ids[j] < n_e) {
                        row[ids[j]] += 1;
                    }
                }
            }
        }
        for (int j = 0; j < n_ids; ++j) {
            if (ids[j] >= 0 && ids[j] < n_e) {
                st->marg[(size_t) layer * (size_t) n_e + (size_t) ids[j]] += 1;
            }
        }
    }

    // The observed row becomes "the previous row" for this layer, in the demand's
    // own id order (the screen keeps the list verbatim: order matters to the bar
    // and to the rank-decay merge).
    std::vector<int32_t> & dst = st->prev[(size_t) layer];
    dst.assign((size_t) (n_ids > 0 ? n_ids : 0), -1);
    for (int j = 0; j < n_ids; ++j) {
        dst[(size_t) j] = ids[j];
    }
}

static uint64_t ls_d1_key(int draft_token, int layer) {
    return (uint64_t) (uint32_t) layer << 32 | (uint32_t) draft_token;
}

void llama_ls_d1_init(llama_ls_d1_state * st, int n_layer, int n_expert) {
    if (st == nullptr) {
        return;
    }
    st->n_layer = n_layer > 0 ? n_layer : 0;
    st->n_expert = n_expert > 0 ? n_expert : 0;
    st->rows_seen = 0;
    st->accepted_rows = 0;
    st->table.clear();
}

int llama_ls_d1_predict(const llama_ls_d1_state * st, int draft_token, int layer, int b,
                       int32_t * out, int max_out) {
    if (st == nullptr || out == nullptr || b <= 0 || max_out <= 0 ||
        draft_token < 0 || layer < 0 || layer >= st->n_layer) {
        return 0;
    }
    const auto it = st->table.find(ls_d1_key(draft_token, layer));
    if (it == st->table.end()) {
        return 0;
    }
    std::vector<llama_ls_d1_entry> ranked = it->second;
    std::sort(ranked.begin(), ranked.end(), [](const llama_ls_d1_entry & a, const llama_ls_d1_entry & b) {
        if (a.count != b.count) {
            return a.count > b.count;
        }
        return a.expert < b.expert;
    });
    const int n = std::min<int>(b, std::min<int>(max_out, (int) ranked.size()));
    for (int i = 0; i < n; ++i) {
        out[i] = ranked[(size_t) i].expert;
    }
    return n;
}

void llama_ls_d1_observe(llama_ls_d1_state * st, int draft_token, int layer, bool accepted,
                         const int32_t * ids, int n_ids) {
    if (st == nullptr || draft_token < 0 || layer < 0 || layer >= st->n_layer) {
        return;
    }
    ++st->rows_seen;
    if (!accepted) {
        return;
    }
    ++st->accepted_rows;
    if (ids == nullptr || n_ids <= 0) {
        return;
    }
    auto & entries = st->table[ls_d1_key(draft_token, layer)];
    for (int i = 0; i < n_ids; ++i) {
        const int32_t expert = ids[i];
        if (expert < 0 || expert >= st->n_expert) {
            continue;
        }
        bool duplicate_input = false;
        for (int j = 0; j < i; ++j) {
            if (ids[j] == expert) {
                duplicate_input = true;
                break;
            }
        }
        if (duplicate_input) {
            continue;
        }
        size_t found = entries.size();
        for (size_t j = 0; j < entries.size(); ++j) {
            if (entries[j].expert == expert) {
                found = j;
                break;
            }
        }
        if (found == entries.size()) {
            entries.push_back({expert, 1});
        } else {
            ++entries[found].count;
        }
    }
}

bool llama_ls_d1_pairing_valid(const llama_ls_d1_state * st) {
    return st != nullptr && st->rows_seen > 0 && st->accepted_rows * 5 >= st->rows_seen * 3;
}

uint32_t llama_ls_pred_null_seed(void) {
    return 0x9E3779B9u;
}

int llama_ls_pred_null_fill(uint32_t * rng, int n_expert, int count, int32_t * out, int max_out) {
    if (rng == nullptr || out == nullptr || n_expert <= 0 || count <= 0 || max_out <= 0) {
        return 0;
    }
    int n     = 0;
    int guard = count * 64 + 64; // a bounded draw loop: the null must terminate
    while (n < count && n < max_out && guard-- > 0) {
        uint32_t x = *rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *rng = x;
        const int32_t id = (int32_t) (x % (uint32_t) n_expert);
        bool dup = false;
        for (int i = 0; i < n; ++i) {
            if (out[i] == id) {
                dup = true;
                break;
            }
        }
        if (dup) {
            continue; // duplicates rejected, as the contract says
        }
        out[n++] = id;
    }
    return n;
}

llama_ls_plan llama_ls_spec_plan(const int32_t * experts, const uint8_t * states, int cap,
                                 int reserve, int max_pending, int n_pending, int budget,
                                 const int32_t * proposals, int n_proposals) {
    llama_ls_plan pl;
    if (experts == nullptr || states == nullptr || proposals == nullptr ||
        cap <= 0 || n_proposals <= 0) {
        return pl;
    }
    if (reserve < 0) {
        reserve = 0;
    }
    if (budget < 0) {
        budget = 0;
    }

    int n_free = 0;
    for (int p = 0; p < cap; ++p) {
        const uint8_t st = states[p];
        if (st == LLAMA_LS_SPEC_PENDING || st == LLAMA_LS_SPEC_CANCELLED ||
            st == LLAMA_LS_SPEC_READY) {
            continue; // occupied, in flight, or awaiting a reap
        }
        if (experts[p] >= 0) {
            continue; // physically occupied
        }
        ++n_free;
    }

    int stop = LLAMA_LS_STOP_PROPOSALS;
    for (int i = 0; i < n_proposals; ++i) {
        const int32_t e = proposals[i];
        if (e < 0) {
            continue;
        }
        bool dup = false;
        for (int k = 0; k < pl.n_issue; ++k) {
            if (pl.issue[k] == e) {
                dup = true;
                break;
            }
        }
        if (dup) {
            ++pl.skipped_dup;
            continue;
        }
        bool present = false;
        for (int p = 0; p < cap; ++p) {
            if (states[p] != LLAMA_LS_SPEC_FREE && experts[p] == e) {
                present = true; // READY (resident) or PENDING/CANCELLED (a fill is already in flight)
                break;
            }
        }
        if (present) {
            ++pl.skipped_present; // llama_ls_spec_admit's already_resident / already_pending
            continue;
        }
        // The three limits llama_ls_spec_admit applies, in its own order: pending
        // cap, a FREE position, then the demand reserve.
        const char * why = nullptr;
        if (n_pending + pl.n_issue >= max_pending) {
            why  = "pending";
            stop = LLAMA_LS_STOP_PENDING;
        } else if (n_free - pl.n_issue <= 0) {
            why  = "no_free";
            stop = LLAMA_LS_STOP_NO_FREE;
        } else if (n_free - pl.n_issue - 1 < reserve) {
            why  = "reserve";
            stop = LLAMA_LS_STOP_RESERVE;
        } else if (pl.n_issue >= budget) {
            why  = "budget";
            stop = LLAMA_LS_STOP_BUDGET;
        }
        if (why != nullptr) {
            ++pl.refused_capacity; // admissible in principle; the barrier refused it
            continue;
        }
        pl.issue[pl.n_issue++] = e;
        if (pl.n_issue >= (int) LLAMA_LS_PRED_MAX_OUT) {
            stop = LLAMA_LS_STOP_BUDGET;
            break;
        }
    }
    pl.stop = stop;
    return pl;
}

llama_ls_cancel_policy llama_ls_spec_cancel_policy(int n_free, int reserve, int n_want, int n_pending) {
    llama_ls_cancel_policy c;
    if (n_want <= 0) {
        c.reason = "no_demand";
        return c;
    }
    if (reserve < 0) {
        reserve = 0;
    }
    if (n_free >= n_want) {
        // The barrier did its job: the demand places its rows in FREE positions
        // and no fill has to be abandoned. `used_reserve` still reports the case
        // where the demand is spending what the reserve had held for it.
        c.used_reserve = (n_free - n_want) < reserve;
        c.reason       = c.used_reserve ? "reserve_spent" : "reserve_holds";
        return c;
    }
    const int need = n_want - n_free;
    if (n_pending <= 0) {
        c.reason = "no_pending_to_cancel"; // nothing to reclaim: the caller must evict instead
        return c;
    }
    c.cancel = true;
    c.need   = need < n_pending ? need : n_pending;
    c.reason = "demand_over_free";
    return c;
}

// The arm's name as the pre-registered contract spells it. Used by the trace
// (run_start) and by bring-up, so a leg is identifiable from its own recording
// rather than from a filename or a command history.
static const char * ls_pred_arm_name(int arm) {
    switch (arm) {
        case LLAMA_LS_ARM_COACT:  return "coact";
        case LLAMA_LS_ARM_BAR09:  return "bar09";
        case LLAMA_LS_ARM_FUSION: return "fusion";
        case LLAMA_LS_ARM_D1:     return "d1";
        default:                  return "none";
    }
}

static const char * ls_spec_plan_stop_name(int stop) {
    switch (stop) {
        case LLAMA_LS_STOP_PROPOSALS: return "proposals";
        case LLAMA_LS_STOP_BUDGET:    return "budget";
        case LLAMA_LS_STOP_PENDING:   return "pending";
        case LLAMA_LS_STOP_RESERVE:   return "reserve";
        case LLAMA_LS_STOP_NO_FREE:   return "no_free";
        default:                      return "unknown";
    }
}

// ---------------------------------------------------------------------------
// T-GPLR-58 MODE 2: the ENGINE-CONSUMED proposer, as ONE pure call per demand
// stage. Everything the engine does at the end of a demand stage when the mode
// is 2 lives here -- observe the row the demand just placed, propose (the arm,
// or the count-matched null), plan against the live placement state -- so that
// tests/test-layer-streaming-tg-spec.cpp can drive the WIRING itself over a
// scripted demand stream instead of re-implementing it, and the engine cannot
// drift from the test. The engine's remaining job is to execute plan.issue[]
// through spec_issue_one(), the same call the scripted source uses.
//
// The order is A1's faithfulness rule, and it is why the proposal is made HERE
// rather than at the demand: the offline screen calls predict() before observe()
// for every (token, layer) row, so its proposal for row k+1 is made from the state
// observed through row k. Observing this row and THEN proposing is therefore the
// same proposal the screen would make for the next row of this layer -- not an
// approximation of it -- which is what lets the recorded offline recall figures be
// read as this proposer's precision.
//
// What it does NOT do: decide the horizon. The fill it plans is issued at the end
// of demand stage k of this layer, and A0/A2's measured reap ordering makes it
// first consumable at stage k+2 while the proposal targets stage k+1's row, so an
// id the k+1 demand wants is the MATCHED fallback (cancelled, its copy paid for
// twice) and only ids that survive it can be consumed at k+2. That asymmetry is
// what M7b's matched/(consumed+matched) band scores; moving the horizon or the
// reap point is a mechanism change (A0/A2's settled territory), not a wiring one.
// ---------------------------------------------------------------------------
llama_ls_propose_result llama_ls_spec_propose(llama_ls_pred_state * st, int arm, bool null_source,
                                              int layer, int b,
                                              const int32_t * demand_ids, int n_demand,
                                              const int32_t * experts, const uint8_t * states, int cap,
                                              int reserve, int max_pending, int n_pending,
                                              uint32_t * null_rng, int n_expert) {
    llama_ls_propose_result r;
    if (st == nullptr || experts == nullptr || states == nullptr) {
        return r;
    }

    // 1. OBSERVE the row that was just served, deduplicated in the CALLER's order:
    //    the screen observes the row as a SET, and the trace order is the order the
    //    engine actually has, so this is the one order both sides can agree on.
    int32_t obs[256];
    int     n_obs = 0;
    for (int i = 0; i < n_demand && n_obs < (int) (sizeof(obs) / sizeof(obs[0])); ++i) {
        if (demand_ids == nullptr) {
            break;
        }
        const int32_t e = demand_ids[i];
        if (e < 0) {
            continue;
        }
        bool dup = false;
        for (int j = 0; j < n_obs; ++j) {
            if (obs[j] == e) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            obs[n_obs++] = e;
        }
    }
    llama_ls_pred_observe(st, layer, obs, n_obs);

    // 2. PROPOSE. The null is COUNT-MATCHED to the arm, so the arm is computed in
    //    BOTH cases: that is what makes "the same number of proposals, with no
    //    signal in them" a controlled comparison instead of a smaller arm. A null
    //    over an arm whose count is 0 is a null with nothing to match, and it
    //    publishes nothing -- honestly, not as a zero that looks like a result.
    int32_t cand[LLAMA_LS_PRED_MAX_OUT];
    const int n_arm = llama_ls_pred_arm_predict(st, arm, layer, b, cand, LLAMA_LS_PRED_MAX_OUT);
    r.n_arm_count = n_arm;
    r.n_proposals = n_arm;
    if (null_source) {
        r.n_proposals = llama_ls_pred_null_fill(null_rng, n_expert, n_arm, cand, LLAMA_LS_PRED_MAX_OUT);
    }
    for (int i = 0; i < r.n_proposals && i < LLAMA_LS_PRED_MAX_OUT; ++i) {
        r.published[i] = cand[i];
    }

    // 3. PLAN against the live placement state: P3's reserve, the per-row budget and
    //    the pending cap decide how many proposals are admitted, and not one of them
    //    may displace a resident row (P5).
    r.plan = llama_ls_spec_plan(experts, states, cap, reserve, max_pending, n_pending, b, cand, r.n_proposals);
    return r;
}

llama_ls_propose_result llama_ls_spec_propose_d1(const llama_ls_d1_state * d1,
                                                  int draft_token, int layer, int b,
                                                  bool null_source,
                                                  const int32_t * experts, const uint8_t * states, int cap,
                                                  int reserve, int max_pending, int n_pending,
                                                  uint32_t * null_rng, int n_expert) {
    llama_ls_propose_result r;
    int32_t cand[LLAMA_LS_PRED_MAX_OUT];
    r.n_arm_count = llama_ls_d1_predict(d1, draft_token, layer, b, cand, LLAMA_LS_PRED_MAX_OUT);
    r.n_proposals = r.n_arm_count;
    if (null_source) {
        r.n_proposals = llama_ls_pred_null_fill(null_rng, n_expert, r.n_arm_count,
                                                 cand, LLAMA_LS_PRED_MAX_OUT);
    }
    for (int i = 0; i < r.n_proposals; ++i) {
        r.published[i] = cand[i];
    }
    r.plan = llama_ls_spec_plan(experts, states, cap, reserve, max_pending, n_pending,
                                b, cand, r.n_proposals);
    return r;
}

// create one device tensor object with the same shape/type as the source (in
// T-GPLR-62 S2: ne2_override replaces the source's expert-axis extent, which is
// how a compact row-set window is made narrow. 0 means "copy src->ne[2]", i.e.
// the pre-S2 full-bank shape, so every existing caller is unchanged.)
static ggml_tensor * make_dev_tensor(ggml_context * ctx, const ggml_tensor * src, int64_t ne2_override = 0) {
    return ggml_new_tensor_4d(ctx, src->type, src->ne[0], src->ne[1],
                              ne2_override > 0 ? ne2_override : src->ne[2], src->ne[3]);
}

// allocate tensors from a buffer with explicit address walking
// (ggml_backend_tensor_alloc requires a concrete address in this tree).
// off is a running cursor into the buffer so coexisting graph sets (and the
// slot bundles) never alias each other's regions; it advances past the last
// allocation on success.
static bool alloc_tensors(ggml_backend_buffer_t buf, ggml_backend_buffer_type_t buft, const std::vector<ggml_tensor *> & tensors, size_t & off) {
    const size_t align = ggml_backend_buffer_get_alignment(buf);
    const size_t buffer_size = ggml_backend_buffer_get_size(buf);
    char * base = (char *) ggml_backend_buffer_get_base(buf);
    for (ggml_tensor * t : tensors) {
        off = (off + align - 1) / align * align;
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
        if (off > buffer_size || alloc_size > buffer_size - off) {
            fprintf(stderr, "layer-streaming: tensor '%s' exceeds backing buffer (offset=%zu size=%zu buffer=%zu align=%zu)\n",
                    t->name, off, alloc_size, buffer_size, align);
            return false;
        }
        ggml_backend_tensor_alloc(buf, t, base + off);
        if (t->data == nullptr) {
            return false;
        }
        off += alloc_size;
    }
    return true;
}

// T-GPLR-24 (200k prefill): spill allocator for persistent graph inputs.
// The budgeted host input buffer is sized by init()'s formula (n_masks * mask
// bytes + embd + slack), which undercounts when more sets bind full-context
// masks than n_masks counts; the next mask then exceeds the buffer and
// finish_build used to abort the whole run. Host memory, no VRAM, no placement
// change; the tensor objects and their addresses are exactly what they would
// have been inside buf_inputs.
// T-GPLR-72: per-thread CPU time in microseconds, or -1 if unavailable.
// Wall time cannot answer "was the host working?", because a CUDA region's
// host block also waits on the device. GetThreadTimes reports this thread's
// kernel + user CPU time at 100 ns resolution, which is what separates host
// execution from a blocked span.
static int64_t ls_thread_cpu_us() {
#ifdef _WIN32
    FILETIME ct, et, kt, ut;
    if (GetThreadTimes(GetCurrentThread(), &ct, &et, &kt, &ut) == 0) {
        return -1;
    }
    ULARGE_INTEGER k, u;
    k.LowPart = kt.dwLowDateTime; k.HighPart = kt.dwHighDateTime;
    u.LowPart = ut.dwLowDateTime; u.HighPart = ut.dwHighDateTime;
    return (int64_t) ((k.QuadPart + u.QuadPart) / 10);   // 100 ns -> us
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) {
        return -1;
    }
    return (int64_t) ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
#endif
}

bool llama_layer_streaming::alloc_spill(const std::vector<ggml_tensor *> & tensors) {
    // the failed alloc_tensors pass may have allocated a prefix into
    // buf_inputs before hitting the tensor that did not fit; ggml_backend_
    // tensor_alloc asserts buffer == NULL, so the spill takes only the
    // tensors the failed pass left unallocated.
    std::vector<ggml_tensor *> pending;
    for (ggml_tensor * t : tensors) {
        if (t->buffer == nullptr && t->data == nullptr) {
            pending.push_back(t);
        }
    }
    if (pending.empty()) {
        error_msg = "alloc_spill: nothing left to allocate (the primary buffer exhausted but every tensor is already bound)";
        return false;
    }
    // T-GPLR-66 (measured 2026-10-01, ab15): the size must be the size the
    // ALLOCATOR WILL WALK, not a flat sum of the tensor sizes rounded once.
    // alloc_tensors() realigns every tensor to the buffer alignment before it
    // places it, so a pending set containing odd-sized tensors needs per-tensor
    // padding that a flat sum omits. The flat sum made the buffer short by
    // exactly the padding and the overflow always landed on the LAST tensor of
    // the set -- measured: "tensor 'leaf_3' exceeds backing buffer
    // (offset=802848 size=4 buffer=802848 align=32)", i.e. the spill buffer was
    // allocated at precisely the cursor the walk had already reached.
    //
    // This is why the failure was non-monotonic in --ubatch and looked like a
    // resource problem: at ubatch 4096 the pending set is dominated by 32 MiB
    // aligned inp_embd slices, where the padding is zero and the flat sum is
    // correct; at 1024/2048 the set carries small tensors ('leaf_*', 4 B) whose
    // padding is not. ubatch 512 and 4096 completed, 1024 and 2048 aborted with
    // exit 0xC0000409 on both slots=2 and (earlier) slots=3.
    const ggml_backend_buffer_type_t buft_cpu = ggml_backend_cpu_buffer_type();
    const size_t spill_align = ggml_backend_buft_get_alignment(buft_cpu);
    size_t bytes = 0;
    for (const ggml_tensor * t : pending) {
        bytes = (bytes + spill_align - 1) / spill_align * spill_align;
        bytes += ggml_backend_buft_get_alloc_size(buft_cpu, t);
    }
    ggml_backend_buffer_ptr spill(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), bytes));
    if (!spill) {
        error_msg = "alloc_spill: host allocation of " + std::to_string(bytes) + " B failed for " +
                    std::to_string(pending.size()) + " tensor(s)";
        return false;
    }
    size_t off = 0;
    if (!alloc_tensors(spill.get(), ggml_backend_cpu_buffer_type(), pending, off)) {
        error_msg = "alloc_spill: " + std::to_string(pending.size()) + " tensor(s) did not fit the " +
                    std::to_string(bytes) + " B spill buffer (walk reached " + std::to_string(off) + " B)";
        return false;
    }
    buf_inputs_spill.push_back(std::move(spill));
    return true;
}

// ---------------------------------------------------------------------------
// telemetry helpers
// ---------------------------------------------------------------------------

static void tel_write(FILE * f, const char * fmt, ...) {
    if (f == nullptr) {
        return;
    }
    // T-GPLR-08: the prefetch worker writes telemetry from its own thread, so
    // one line (format + newline + flush) must be atomic or two writers can
    // interleave mid-line. Single funnel: every tel_* site flows through here.
    static std::mutex s_tel_mtx;
    std::lock_guard<std::mutex> lk(s_tel_mtx);
    va_list args;
    va_start(args, fmt);
    vfprintf(f, fmt, args);
    va_end(args);
    fprintf(f, "\n");
    fflush(f);
}

// T-GPLR-56 A1: the pre-sink marker buffer. The trace used to begin at `run_start`,
// leaving the model load, init and the capacity preflight entirely unmeasured - so the
// acceptance quantity Q (process start -> end of pass 1) could not be formed at all. These
// markers are recorded before the sink exists and flushed, in call order, as its first
// events. Missing sink = dropped; nothing else changes.
namespace {
struct tel_early_rec {
    std::string event;
    std::string detail;
    long long   t_us = 0;
    bool        late = false;   // taken while a sink was already live (emitted late, not from the pre-sink buffer)
};

std::vector<tel_early_rec> g_tel_early;

// `g_tel_early_sink_seen` is the only state this buffer keeps: it becomes true once a flush
// has written to a REAL sink, i.e. once telemetry is live. A marker taken after that is a
// LATE marker. It exists because such a marker used to be pushed into `g_tel_early` and
// nothing ever flushed it again: silent telemetry loss, no error and no flag (found
// 2026-09-22 by inspection; every call site at the time marked before the sink, so the case
// was latent, not hit). The fix keeps ONE emission path - the buffer - so no FILE* is ever
// stashed (a stashed sink would be written to after a caller fclose'd it, which is undefined
// behaviour), and the engine's three close sites now call llama_ls_tel_flush_early(tel_out)
// BEFORE fclose, which is what turns a late marker into a written line instead of a lost one.
// A NULL flush means "no sink": it does NOT set the flag, so a marker taken before any
// session started still reads as pre_run.
bool   g_tel_early_sink_seen = false;
size_t g_tel_early_seq       = 0;

std::string tel_json_escape(const std::string & s) {
    std::string out;
    for (const char * p = s.c_str(); p != nullptr && *p != '\0'; ++p) {
        const unsigned char c = (unsigned char) *p;
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out;
}
}   // namespace

void llama_ls_tel_mark(const char * event, const char * detail) {
    if (event == nullptr || event[0] == '\0') {
        return;
    }
    // MEASURED TRAP (2026-09-22): on Windows `ggml_time_us()` is
    // ((QPC - timer_start) * 1000000) / timer_freq, and both statics are set by
    // ggml_time_init(). A marker taken before any llama init therefore divides by
    // timer_freq == 0 and dies with 0xC0000094 - measured with this very test, on a
    // marker placed at the top of main(). ggml_time_init() is InitOnce-guarded, so
    // calling it here is free and a later call by llama_backend_init() is a no-op.
    // A telemetry helper must never be able to kill the process.
    ggml_time_init();
    // One buffer, one emission path. `late` records only WHEN the marker was taken - after
    // a sink had already been flushed once - so a reader can tell a pre-sink marker from a
    // post-sink one without guessing, and so the flush can label it accordingly.
    g_tel_early.push_back({ event, detail != nullptr ? detail : "", ggml_time_us(), g_tel_early_sink_seen });
}

void llama_ls_tel_flush_early(FILE * f) {
    if (f == nullptr) {
        g_tel_early.clear();   // no sink: nothing to write to (unchanged semantics)
        return;
    }
    // From here on, a marker taken later is a LATE marker: it is still buffered, but it is
    // emitted by the next flush instead of accumulating in a buffer nothing reaches again.
    g_tel_early_sink_seen = true;
    // `pre_run` marks a line that precedes run_start and therefore carries no run_id; `seq`
    // is the call order, so a reader can prove the markers were not reordered. `late` is the
    // third fact and it is needed BECAUSE a late line is otherwise indistinguishable from a
    // second emission of the same event name (the close-site flush could look like a
    // duplicated `load_end`): `late:1` means "recorded after the sink had already been
    // flushed once, so it did not come out of the pre-sink buffer". Both flags are true of
    // every marker this engine takes, which are all taken before run_start.
    for (const tel_early_rec & r : g_tel_early) {
        tel_write(f, "{\"event\":\"%s\",\"seq\":%zu,\"detail\":\"%s\"%s,\"pre_run\":1,\"t_us\":%lld}",
                  tel_json_escape(r.event).c_str(), g_tel_early_seq++, tel_json_escape(r.detail).c_str(),
                  r.late ? ",\"late\":1" : "", r.t_us);
    }
    g_tel_early.clear();
}

// ---------------------------------------------------------------------------
// teardown
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// T-GPLR-08: one-layer-ahead PrefetchVirtualMemory probe (host feed)
// ---------------------------------------------------------------------------

struct llama_layer_streaming::pf_worker {
    std::thread             thread;
    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<std::pair<int, int>> queue; // (logical layer, telemetry phase)
    bool                    stop = false;
    llama_layer_streaming * owner = nullptr;
    // Bound once at configure time; the worker is only started when non-null.
    BOOL (WINAPI * pPrefetchVirtualMemory)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG) = nullptr;

    // stats (guarded by mtx)
    uint64_t n_submit   = 0;
    uint64_t n_complete = 0;
    uint64_t n_err      = 0;
    uint64_t bytes      = 0;
    uint64_t us         = 0;

    void run() {
        for (;;) {
            int il = -1;
            int phase = 0;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [this] { return stop || !queue.empty(); });
                if (stop && queue.empty()) {
                    return;
                }
                il    = queue.front().first;
                phase = queue.front().second;
                queue.pop_front();
            }
            // Ranges are read-only after set_pp_prefetch_ahead returns; the
            // vectors are sized once before the thread starts.
            const auto & ptrs = owner->pf_ranges_ptr.at(il);
            const auto & lens = owner->pf_ranges_len.at(il);
            if (pPrefetchVirtualMemory == nullptr) {
                // API absent: record the skipped hint, never call through null.
                std::lock_guard<std::mutex> lk(mtx);
                n_err++;
                owner->tel_prefetch_complete(il, phase, (int) ptrs.size(), 0, 0.0, 1);
                continue;
            }
            std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges(ptrs.size());
            size_t bytes_total = 0;
            for (size_t i = 0; i < ptrs.size(); ++i) {
                ranges[i].VirtualAddress = ptrs[i];
                ranges[i].NumberOfBytes  = lens[i];
                bytes_total += lens[i];
            }
            const int64_t t0 = ggml_time_us();
            const BOOL ok = pPrefetchVirtualMemory(GetCurrentProcess(), (ULONG_PTR) ranges.size(), ranges.data(), 0);
            const int64_t t1 = ggml_time_us();
            const uint32_t err = ok ? 0u : (uint32_t) GetLastError();
            {
                std::lock_guard<std::mutex> lk(mtx);
                n_submit++;
                if (ok) {
                    n_complete++;
                    bytes += bytes_total;
                    us += (uint64_t) (t1 - t0);
                } else {
                    n_err++;
                }
            }
            // Telemetry outside the stats lock; tel_write itself serializes
            // lines, and no code path takes pf->mtx while holding the telemetry
            // lock (single lock per site: no inverse ordering).
            owner->tel_prefetch_complete(il, phase, (int) ranges.size(), bytes_total, (double) (t1 - t0), err);
        }
    }
};

void llama_layer_streaming::pf_worker_deleter::operator()(pf_worker * w) const {
    delete w;
}

// ---------------------------------------------------------------------------
// T-GPLR-25: PP async-feed worker (device-side copy overlap prototype)
// ---------------------------------------------------------------------------
// The measured PP structure this prototype targets (2026-09-15 trace analysis,
// build/analyze_pp_overlap.py over the recorded arms): with `--layer-slots 1`,
// ZERO layer_compute events have the next copy already submitted at compute
// start; the ~47-180 ms/layer pageable host feed runs on the MAIN thread after
// the previous layer's compute returns (50k: feed 185 s vs compute 22 s of a
// 208 s wall). The copy path itself is fast (200k shows ~9.7 GB/s effective
// while submitting); the loss is the serialization. This worker submits the
// next executed layer's package copy (the load_layer body: slot-reservation
// check, event records, member set_async loop, telemetry) into a DIFFERENT
// slot while the main thread computes, so the feed overlaps graph execution.
// Invariants held:
// - slot reuse only after ev_consumer_done: the worker submits into the next
//   slot only after the main thread has recorded consumer-done for it (tracked
//   by af_free_mask under af->mtx), identical to load_layer's own rule.
// - a layer graph consumes a slot only after the matching copy event completes:
//   wait_slot_ready still blocks on ev_copy_ready, which the worker records —
//   the main thread's ordering guarantee is unchanged.
// - the slot->logical-layer/generation mutation for the WORKER's copy happens
//   under af->mtx; the main thread takes the same mutex around its own slot
//   bookkeeping so the two writers cannot interleave. Telemetry is thread-safe
//   (tel_write serializes lines since T-GPLR-08).

struct llama_layer_streaming::af_worker {
    std::thread             thread;
    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<int>         queue;      // logical layers to stage
    // slots the main thread has released (consumer done) and the worker may
    // fill; index = slot id, value = true when free
    std::vector<uint8_t>    free_slot;
    bool                    stop = false;
    llama_layer_streaming * owner = nullptr;

    // stats (guarded by mtx)
    uint64_t n_submit = 0;
    uint64_t n_ok     = 0;
    uint64_t n_err    = 0;
    uint64_t bytes    = 0;
    uint64_t us       = 0;

    void run() {
        for (;;) {
            int il = -1;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [this] { return stop || !queue.empty(); });
                if (stop && queue.empty()) {
                    return;
                }
                il = queue.front();
                queue.pop_front();
            }
            const int64_t t0 = ggml_time_us();
            const bool ok = owner->af_submit_copy(il);
            const int64_t t1 = ggml_time_us();
            {
                std::lock_guard<std::mutex> lk(mtx);
                n_submit++;
                if (ok) {
                    n_ok++;
                    bytes += (uint64_t) owner->package_bytes(il);
                    us += (uint64_t) (t1 - t0);
                } else {
                    n_err++;
                }
            }
        }
    }
};

void llama_layer_streaming::af_worker_deleter::operator()(af_worker * w) const {
    delete w;
}

int llama_layer_streaming::af_next_layer(int il) const {
    if (exec_layers.empty()) {
        return il + 1 < n_layer ? il + 1 : -1;
    }
    for (size_t i = 0; i < exec_layers.size(); ++i) {
        if (exec_layers[i] == il) {
            return i + 1 < exec_layers.size() ? exec_layers[i + 1] : -1;
        }
    }
    return -1;
}

// T-GPLR-37: direct mmap-sourced staging helpers. The registered span is the
// page-aligned superset of every member source of one package, so a DMA may
// read the model's own mapping at the pinned rate with no CPU hop. Cached on
// the slot so re-staging the SAME package is free; a different package first
// unregisters the previous span, which is safe because a slot is only re-staged
// after the previous generation's consumer completed (record_consumer_done),
// and therefore after that generation's DMA completed.
bool llama_layer_streaming::ls_register_pkg_source(int il, llama_layer_streaming_slot & slot, const llama_ls_package & pkg) {
    // T-GPLR-37 churn-free variant: the span was registered once, before the
    // timed PP (register_all_pkg_sources). Nothing to do here — and a missing
    // span is a LOUD error rather than a silent fall back into per-bank churn.
    if (!copy_src_reg_churn) {
        const size_t idx = (size_t) il;
        if (idx < all_reg_base.size() && all_reg_base[idx] != nullptr) {
            return true;
        }
        // T-GPLR-39: a coalesced dense group registers one span owned by the
        // group's first layer; later members are marked covered and stage
        // against that same span (their bytes live inside it).
        if (idx < all_reg_covered.size() && all_reg_covered[idx]) {
            return true;
        }
        error_msg = "mmap-registered staging (register-once): logical layer " + std::to_string(il) +
                    " has no pre-registered source span (register_all_pkg_sources must run before the pass)";
        return false;
    }
    const uintptr_t page_mask = (uintptr_t) 4095;
    uintptr_t lo = (uintptr_t) -1;
    uintptr_t hi = 0;
    size_t    members = 0;
    for (const auto & m : pkg.members) {
        if (m.src == nullptr || m.src->data == nullptr) {
            continue;
        }
        const uintptr_t a = (uintptr_t) m.src->data;
        const uintptr_t b = a + (uintptr_t) ggml_nbytes(m.src);
        lo = std::min(lo, a);
        hi = std::max(hi, b);
        members++;
    }
    if (members == 0 || hi <= lo) {
        error_msg = "mmap-registered staging: package for logical layer " + std::to_string(il) +
                    " has no member sources to register";
        return false;
    }
    const void * base  = (const void *) (lo & ~page_mask);
    const size_t bytes = (size_t) (((hi + page_mask) & ~page_mask) - ((uintptr_t) base));
    if (slot.reg_base == base && slot.reg_bytes == bytes) {
        return true;   // this slot already holds this package's registered span
    }
    if (slot.reg_base != nullptr) {
        if (!ggml_backend_cuda_unregister_host_range(slot.reg_base)) {
            // not fatal: the previous span belongs to a consumed generation and
            // its pages stay mapped; report it rather than abort a good run
            fprintf(stderr, "layer-streaming: cudaHostUnregister failed for %p (%zu bytes)\n",
                    slot.reg_base, slot.reg_bytes);
        }
        slot.reg_base = nullptr;
        slot.reg_bytes = 0;
    }
    char err[256] = {0};
    if (!ggml_backend_cuda_register_host_range(base, bytes, err, sizeof(err))) {
        error_msg = "mmap-registered staging: cudaHostRegister failed for logical layer " + std::to_string(il) +
                    " (" + std::to_string(bytes) + " bytes): " + err;
        return false;   // loud refusal: never silently fall back to the copy hop
    }
    slot.reg_base  = base;
    slot.reg_bytes = bytes;
    if (tel_out != nullptr) {
        tel_write(tel_out, "{\"event\":\"mmap_register\",\"run_id\":%llu,\"ubatch\":%llu,\"layer\":%d,"
                  "\"bytes\":%zu,\"members\":%zu,\"base\":%llu,\"t_us\":%lld}",
                  (unsigned long long) run_id, (unsigned long long) n_ubatches, il,
                  bytes, members, (unsigned long long) (uintptr_t) base, (long long) ggml_time_us());
    }
    return true;
}

// T-GPLR-37 churn-free variant: one registration per address-coherent span
// group, before the timed PP. Cost is setup (visible in the runner's pre-timer
// report), not pass time, and the pinned footprint is the sum of the registered
// spans (bounded by the model's package bytes — 17.0 GiB for the 40-bank MoE
// vehicle), which is the price this variant pays against the per-bank churn it
// removes.
//
// T-GPLR-39 note (dense refusal, measured 2026-09-20): consecutive DENSE
// layers' page-rounded spans share a boundary page (one layer's `hi` rounds up
// into the page where the next layer starts), and cudaHostRegister refuses a
// range that is already (partially) registered ("already mapped"). The MoE
// vehicle never hit this because non-package tensors (router/attention/shared)
// separate consecutive banks. Spans of executed layers are therefore coalesced
// greedily wherever their page-rounded ranges touch: MoE groups stay
// single-layer (its recorded 40 events and byte totals are unchanged), dense
// groups merge into one registration owned by the group's first layer.
bool llama_layer_streaming::register_all_pkg_sources(int64_t & bytes_out, double & us_out) {
    bytes_out = 0;
    us_out    = 0.0;
    if (!copy_src_registered) {
        error_msg = "register_all_pkg_sources: requires a mmap-registered copy source";
        return false;
    }
    all_reg_base.assign(pkg_cache.size(), nullptr);
    all_reg_bytes.assign(pkg_cache.size(), (size_t) 0);
    all_reg_covered.assign(pkg_cache.size(), false);
    const uintptr_t page_mask = (uintptr_t) 4095;
    const int64_t t0 = ggml_time_us();
    int64_t registered = 0;

    // First pass: collect the page-rounded span of every executed layer.
    struct span_rec { int il; uintptr_t lo, hi; size_t members; };
    std::vector<span_rec> spans;
    for (int il = 0; il < (int) pkg_cache.size(); ++il) {
        if (!exec_layers.empty()) {
            bool want = false;
            for (int e : exec_layers) {
                if (e == il) { want = true; break; }
            }
            if (!want) {
                continue;
            }
        }
        const llama_ls_package & pkg = pkg_cache[(size_t) il];
        uintptr_t lo = (uintptr_t) -1;
        uintptr_t hi = 0;
        size_t    members = 0;
        for (const auto & m : pkg.members) {
            if (m.src == nullptr || m.src->data == nullptr) {
                continue;
            }
            const uintptr_t a = (uintptr_t) m.src->data;
            const uintptr_t b = a + (uintptr_t) ggml_nbytes(m.src);
            lo = std::min(lo, a);
            hi = std::max(hi, b);
            members++;
        }
        if (members == 0 || hi <= lo) {
            continue;   // no package for this layer: nothing to pin
        }
        spans.push_back({ il, lo, hi, members });
    }

    // Second pass: coalesce page-touching spans (file order gives ascending
    // addresses) and register each group once.
    size_t si = 0;
    while (si < spans.size()) {
        size_t sj = si;
        uintptr_t glo = spans[si].lo & ~page_mask;
        uintptr_t ghi = (spans[si].hi + page_mask) & ~page_mask;
        size_t    gmembers = spans[si].members;
        while (sj + 1 < spans.size()) {
            const uintptr_t nlo = spans[sj + 1].lo & ~page_mask;
            const uintptr_t nhi = (spans[sj + 1].hi + page_mask) & ~page_mask;
            if (nlo > ghi) {
                break;   // at least one free page between the groups
            }
            ghi = std::max(ghi, nhi);
            gmembers += spans[sj + 1].members;
            ++sj;
        }
        const void * base  = (const void *) glo;
        const size_t bytes = (size_t) (ghi - glo);
        char err[256] = {0};
        if (!ggml_backend_cuda_register_host_range(base, bytes, err, sizeof(err))) {
            error_msg = "mmap-registered staging (register-once): cudaHostRegister failed for logical layer " +
                        std::to_string(spans[si].il) + " (" + std::to_string(bytes) + " bytes): " + err;
            unregister_all_pkg_sources();   // leave the box unpinned on refusal
            return false;
        }
        // Ownership: the group's FIRST layer carries the registration (base and
        // byte extent, so footprint and teardown stay single-counted); the rest
        // are marked covered so their staging finds the span without a
        // duplicate register.
        all_reg_base[(size_t) spans[si].il]  = base;
        all_reg_bytes[(size_t) spans[si].il] = bytes;
        for (size_t k = si + 1; k <= sj; ++k) {
            all_reg_covered[(size_t) spans[k].il] = true;
        }
        registered += (int64_t) bytes;
        if (tel_out != nullptr) {
            tel_write(tel_out, "{\"event\":\"mmap_register\",\"run_id\":%llu,\"mode\":\"init-once\","
                      "\"layer\":%d,\"bytes\":%zu,\"members\":%zu,\"merged\":%zu,\"base\":%llu,\"t_us\":%lld}",
                      (unsigned long long) run_id, spans[si].il, bytes, gmembers,
                      (size_t) (sj - si),
                      (unsigned long long) (uintptr_t) base, (long long) ggml_time_us());
        }
        si = sj + 1;
    }
    us_out = (double) (ggml_time_us() - t0);
    bytes_out = registered;
    if (tel_out != nullptr) {
        tel_write(tel_out, "{\"event\":\"mmap_register_init\",\"run_id\":%llu,\"bytes\":%lld,"
                  "\"us\":%.1f,\"layers\":%d,\"t_us\":%lld}",
                  (unsigned long long) run_id, (long long) registered, us_out,
                  (int) pkg_cache.size(), (long long) ggml_time_us());
    }
    return true;
}

// ---------------------------------------------------------------------------
// T-GPLR-56 A3: PACED REGISTRATION (contract in the header). Off by default and
// additive by construction: with pp_span_pace == 0 nothing below ever runs and
// register_all_pkg_sources() is not modified. Invariants kept:
//   * every executed layer is registered (through all_reg_base for a group's
//     first layer, or all_reg_covered for the rest) BEFORE any staging copy that
//     needs it, else the existing loud refusal fires - never a silent fall back;
//   * single-counted ownership and teardown reuse the same vectors as the
//     one-shot path, so unregister_all_pkg_sources() needs no paced variant;
//   * telemetry uses NEW event names only, so no existing checker sees a new
//     shape in a trace.
// ---------------------------------------------------------------------------
bool llama_layer_streaming::set_pp_span_pace(int mode, int ahead) {
    if (mode < 0 || mode > 1) {
        error_msg = "set_pp_span_pace: mode must be 0 (off) or 1 (paced) - refused, not clamped";
        return false;
    }
    if (mode == 1) {
        if (!copy_src_registered) {
            error_msg = "set_pp_span_pace: paced registration requires --copy-source mmap-registered";
            return false;
        }
        if (ahead < 0 || ahead > 64) {
            error_msg = "set_pp_span_pace: ahead must be 0..64 groups";
            return false;
        }
    }
    pp_span_pace       = mode;
    pp_span_pace_ahead = ahead;
    if (mode == 0) {
        paced_groups.clear();
        paced_touch_seen.clear();
        paced_next_group   = 0;
        paced_bytes_total  = 0;
        paced_bytes_issued = 0;
    }
    return true;
}

bool llama_layer_streaming::register_paced_begin(int64_t & bytes_out, double & us_out) {
    bytes_out = 0;
    us_out    = 0.0;
    if (pp_span_pace == 0) {
        return register_all_pkg_sources(bytes_out, us_out);   // off: the existing one-shot path
    }
    if (!copy_src_registered) {
        error_msg = "register_paced_begin: requires a mmap-registered copy source";
        return false;
    }
    all_reg_base.assign(pkg_cache.size(), nullptr);
    all_reg_bytes.assign(pkg_cache.size(), (size_t) 0);
    all_reg_covered.assign(pkg_cache.size(), false);
    paced_groups.clear();
    paced_touch_seen.assign(pkg_cache.size(), 0);
    paced_next_group   = 0;
    paced_bytes_total  = 0;
    paced_bytes_issued = 0;

    const uintptr_t page_mask = (uintptr_t) 4095;
    const int64_t   t0        = ggml_time_us();

    // (1) page-rounded span of every EXECUTED layer: the same rule as the one-shot
    // path (span of the members' mapped addresses, never a file-offset table), and
    // the reason this mode needs the loaded model and cannot run earlier.
    struct span_rec { int il; uintptr_t lo, hi; size_t members; };
    std::vector<span_rec> spans;
    for (int il = 0; il < (int) pkg_cache.size(); ++il) {
        if (!exec_layers.empty()) {
            bool want = false;
            for (int e : exec_layers) {
                if (e == il) { want = true; break; }
            }
            if (!want) {
                continue;
            }
        }
        const llama_ls_package & pkg = pkg_cache[(size_t) il];
        uintptr_t lo = (uintptr_t) -1;
        uintptr_t hi = 0;
        size_t    members = 0;
        for (const auto & m : pkg.members) {
            if (m.src == nullptr || m.src->data == nullptr) {
                continue;
            }
            const uintptr_t a = (uintptr_t) m.src->data;
            const uintptr_t b = a + (uintptr_t) ggml_nbytes(m.src);
            lo = std::min(lo, a);
            hi = std::max(hi, b);
            members++;
        }
        if (members == 0 || hi <= lo) {
            continue;
        }
        spans.push_back({ il, lo, hi, members });
    }

    // (2) coalesce page-touching spans in file order - the SAME grouping the
    // one-shot path registers, kept in group order so it can be issued one group
    // at a time. Each group remembers its member layers (they are not necessarily
    // contiguous: exec_layers may be a boundary subset).
    size_t si = 0;
    while (si < spans.size()) {
        size_t sj = si;
        uintptr_t glo = spans[si].lo & ~page_mask;
        uintptr_t ghi = (spans[si].hi + page_mask) & ~page_mask;
        size_t    gmembers = spans[si].members;
        while (sj + 1 < spans.size()) {
            const uintptr_t nlo = spans[sj + 1].lo & ~page_mask;
            const uintptr_t nhi = (spans[sj + 1].hi + page_mask) & ~page_mask;
            if (nlo > ghi) {
                break;   // at least one free page between the groups
            }
            ghi = std::max(ghi, nhi);
            gmembers += spans[sj + 1].members;
            ++sj;
        }
        ls_reg_group g;
        g.lo       = glo;
        g.hi       = ghi;
        g.bytes    = (size_t) (ghi - glo);
        g.first_il = spans[si].il;
        g.merged   = sj - si + 1;
        g.members  = gmembers;
        g.done     = false;
        for (size_t k = si; k <= sj; ++k) {
            g.layers.push_back(spans[k].il);
        }
        paced_groups.push_back(g);
        paced_bytes_total += (int64_t) g.bytes;
        si = sj + 1;
    }
    if (paced_groups.empty()) {
        error_msg = "register_paced_begin: no package source spans to register";
        return false;
    }

    // (3) the FIRST group, before the timed pass: the same head start the one-shot
    // path gave every group at once. The rest is issued from stage_members().
    if (!register_paced_ahead(-1)) {
        return false;
    }
    us_out    = (double) (ggml_time_us() - t0);
    bytes_out = paced_bytes_issued;
    if (tel_out != nullptr) {
        tel_write(tel_out, "{\"event\":\"span_pace_init\",\"run_id\":%llu,\"mode\":%d,\"ahead\":%d,"
                  "\"groups\":%zu,\"bytes_total\":%lld,\"bytes_issued\":%lld,\"us\":%.1f,\"t_us\":%lld}",
                  (unsigned long long) run_id, pp_span_pace, pp_span_pace_ahead,
                  paced_groups.size(), (long long) paced_bytes_total, (long long) paced_bytes_issued,
                  us_out, (long long) ggml_time_us());
    }
    return true;
}

bool llama_layer_streaming::register_paced_ahead(int il) {
    if (pp_span_pace == 0 || paced_groups.empty()) {
        return true;   // off: a no-op by construction
    }

    // (a) the layer's FIRST staging copy is being reached: record whether pacing
    // had already covered it (the p1-pace discriminator: every layer covered at
    // its first touch means the commit was issued ahead of the copy that needs it).
    if (il >= 0 && (size_t) il < paced_touch_seen.size() && paced_touch_seen[(size_t) il] == 0) {
        paced_touch_seen[(size_t) il] = 1;
        const bool covered = ((size_t) il < all_reg_base.size()    && all_reg_base[(size_t) il] != nullptr) ||
                             ((size_t) il < all_reg_covered.size() && all_reg_covered[(size_t) il]);
        if (tel_out != nullptr) {
            tel_write(tel_out, "{\"event\":\"span_pace_stage\",\"run_id\":%llu,\"layer\":%d,\"covered\":%d,"
                      "\"groups_issued\":%zu,\"groups_total\":%zu,\"bytes_unissued\":%lld,\"t_us\":%lld}",
                      (unsigned long long) run_id, il, covered ? 1 : 0,
                      paced_next_group, paced_groups.size(),
                      (long long) (paced_bytes_total - paced_bytes_issued),
                      (long long) ggml_time_us());
        }
        // NOT a check: an uncovered layer is caught by ls_register_pkg_source's own
        // loud refusal when its copy is staged, so the failure has one owner.
    }

    // (b) issue the groups that are now due. il < 0 = the pre-pass head start
    // (the first group only); otherwise every group whose FIRST layer is within
    // `ahead` groups of the layer being staged.
    size_t end = 1;
    if (il >= 0) {
        end = 0;
        while (end < paced_groups.size() && paced_groups[end].first_il <= il + pp_span_pace_ahead) {
            ++end;
        }
    }
    while (paced_next_group < end) {
        ls_reg_group & g = paced_groups[paced_next_group];
        const void * base  = (const void *) g.lo;
        const size_t bytes = g.bytes;
        char err[256] = {0};
        if (!ggml_backend_cuda_register_host_range(base, bytes, err, sizeof(err))) {
            error_msg = "paced registration: cudaHostRegister failed for the group starting at logical layer " +
                        std::to_string(g.first_il) + " (" + std::to_string(bytes) + " bytes): " + err;
            unregister_all_pkg_sources();   // leave the box unpinned on refusal
            paced_next_group = paced_groups.size();
            return false;
        }
        // ownership: the group's FIRST layer carries the registration; its other
        // members are marked covered so their staging finds the span without a
        // duplicate register (identical convention to the one-shot path).
        for (size_t k = 0; k < g.layers.size(); ++k) {
            const int l2 = g.layers[k];
            if ((size_t) l2 >= all_reg_base.size()) {
                continue;
            }
            if (k == 0) {
                all_reg_base[(size_t) l2]  = base;
                all_reg_bytes[(size_t) l2] = bytes;
            } else {
                all_reg_covered[(size_t) l2] = true;
            }
        }
        g.done = true;
        paced_bytes_issued += (int64_t) bytes;
        if (tel_out != nullptr) {
            tel_write(tel_out, "{\"event\":\"span_pace\",\"run_id\":%llu,\"gi\":%zu,\"layer\":%d,"
                      "\"bytes\":%zu,\"members\":%zu,\"merged\":%zu,\"base\":%llu,\"issued\":%zu,"
                      "\"total\":%zu,\"bytes_unissued\":%lld,\"t_us\":%lld}",
                      (unsigned long long) run_id, paced_next_group, g.first_il, bytes, g.members,
                      (size_t) (g.layers.size() - 1), (unsigned long long) (uintptr_t) base,
                      paced_next_group + 1, paced_groups.size(),
                      (long long) (paced_bytes_total - paced_bytes_issued),
                      (long long) ggml_time_us());
        }
        ++paced_next_group;
    }
    if (paced_next_group >= paced_groups.size() && paced_groups.size() > 0 && tel_out != nullptr) {
        // the last group is issued: the total commit is complete
        tel_write(tel_out, "{\"event\":\"span_pace_end\",\"run_id\":%llu,\"groups\":%zu,\"bytes\":%lld,\"t_us\":%lld}",
                  (unsigned long long) run_id, paced_groups.size(), (long long) paced_bytes_issued,
                  (long long) ggml_time_us());
    }
    return true;
}

void llama_layer_streaming::unregister_all_pkg_sources() {
    for (size_t i = 0; i < all_reg_base.size(); ++i) {
        if (all_reg_base[i] != nullptr) {
            (void) ggml_backend_cuda_unregister_host_range(all_reg_base[i]);
            all_reg_base[i]  = nullptr;
            all_reg_bytes[i] = 0;
        }
        all_reg_covered[i] = false;
    }
}

void llama_layer_streaming::ls_unregister_slot_source(llama_layer_streaming_slot & slot) {
    if (slot.reg_base == nullptr) {
        return;
    }
    (void) ggml_backend_cuda_unregister_host_range(slot.reg_base);
    slot.reg_base  = nullptr;
    slot.reg_bytes = 0;
}

// T-GPLR-33 V1: shared staging loop for both copy sites (af worker and the
// main thread's load_layer). Hops every member into the pinned staging/pool
// source (unchanged byte movement), then submits either ONE H2D for the
// whole bank (when the slot-build contiguity check produced a bank_span for
// this layer and the staging sources are contiguous) or one H2D per member
// (legacy path, byte-identical). Returns false on a member/descriptor
// mismatch (the caller owns the pool release and the error surface in that
// case); *staged_out always receives the staged byte count.
bool llama_layer_streaming::stage_members(ggml_backend_t backend,
                                          const llama_ls_package & pkg,
                                          llama_layer & bundle,
                                          int il, uint32_t gen,
                                          llama_layer_streaming_slot & slot,
                                          char * pool_base, size_t pool_bytes, bool pool_valid,
                                          char * staging, size_t * staged_out,
                                          int * submit_batch_out) {
    // T-GPLR-56 A3: paced registration is issued HERE, before a single copy for
    // this layer is enqueued, so the group covering a bounded distance ahead of
    // the sweep is locked (and its deferred commit started) while this layer's
    // copies run. No-op when the mode is off; a failure is a refusal.
    if (pp_span_pace > 0 && !register_paced_ahead(il)) {
        return false;
    }

    // T-GPLR-33: report the actually-taken submission path so traces are
    // self-verifying (the flag alone cannot distinguish a span copy from a
    // degraded fallback). 0 = per-member only, 1 = single span copy, 2 =
    // mixed (partial span flushed, then per-member). The mismatch cases
    // below reset span33 BEFORE any submission, so 0 is authoritative there.
    int submit_batch_path = 0;
    ggml_tensor * span33 = nullptr;
    if (pp_submit_batch) {
        auto it33 = slot.bank_span.find(il);
        if (it33 != slot.bank_span.end()) {
            span33 = it33->second;
        }
    }
    if (span33 != nullptr) {
        // the span is sized from the DST tiling at slot build; re-validate
        // against the live sources so a skipped member can never corrupt
        size_t sum33 = 0;
        for (const auto & m : pkg.members) {
            if (m.src == nullptr) { span33 = nullptr; break; }
            sum33 += ggml_nbytes(m.src);
        }
        if (span33 != nullptr && sum33 != ggml_nbytes(span33)) {
            span33 = nullptr;
        }
    }
    char * span_src33 = nullptr;
    size_t span_staged33 = 0;
    bool   span_dead33 = false;
    size_t staging_off = 0;

    // T-GPLR-35: single staging-submission point. pp_copy_overlap == 0 keeps the
    // dedicated copy backend's own stream (bit-identical to the pre-T-GPLR-35
    // path); == 1 issues the DMA on the auxiliary stream of the COMPUTE
    // backend's context, so every staging/compute ordering wait is
    // intra-context. A missing compute backend with the flag on is a loud
    // failure — never a silent fallback to the serialized path.
    auto submit = [&](ggml_tensor * dst_t, const void * src_p, size_t bytes) -> bool {
        if (pp_copy_overlap) {
            if (backend_compute == nullptr) {
                error_msg = "copy-overlap is on but the CUDA compute backend is unresolved; "
                            "refusing to fall back to the copy-stream staging path";
                return false;
            }
            ggml_backend_cuda_memcpy_1d_aux(backend_compute, (char *) dst_t->data, src_p, bytes);
            return true;
        }
        ggml_backend_tensor_set_async(backend, dst_t, src_p, 0, bytes);
        return true;
    };

    // T-GPLR-53: the recon sample carries the generation of the copy it brackets,
    // not the slot's pre-increment one, so recon_attr joins copy_submit / copy_complete.
    recon_begin_bank(il, gen);
    for (const auto & m : pkg.members) {
        const ggml_tensor * src_t = m.src;
        if (src_t == nullptr) {
            continue;
        }
        ggml_tensor * dst_t = ls_dst(bundle, m);
        if (dst_t == nullptr || ggml_nbytes(src_t) != ggml_nbytes(dst_t)) {
            *staged_out = staging_off;
            return false;
        }
        const size_t bytes = ggml_nbytes(src_t);
        const void * copy_src = src_t->data;
        if (copy_src_registered) {
            // T-GPLR-37: skip the hop entirely. The source IS the model's
            // mapping, already page-locked for this bank by
            // ls_register_pkg_source, so the DMA below reads it at the pinned
            // rate (measured 14.44 GB/s) with no CPU memcpy (which measured
            // 20.22 ms per 452 MiB bank, ~809 ms per 40-bank pass).
        } else if (pool_base != nullptr && staging_off + bytes <= pool_bytes) {
            if (!pool_valid) {
                memcpy(pool_base + staging_off, src_t->data, bytes); // lru fill
            }
            copy_src = pool_base + staging_off;
        } else if (staging != nullptr) {
            memcpy(staging + staging_off, src_t->data, bytes);
            copy_src = staging + staging_off;
        }
        if (span33 != nullptr && !span_dead33) {
            if (span_src33 == nullptr) {
                span_src33 = (char *) copy_src;
                span_staged33 = 0;
                submit_batch_path = 1;
            } else if ((char *) copy_src != span_src33 + span_staged33) {
                // staging sources turned non-contiguous mid-bank: flush the
                // accumulated prefix as one partial-span copy, then degrade
                // to per-member for the rest (every byte still lands once).
                if (!submit(span33, span_src33, span_staged33)) { *staged_out = staging_off; return false; }
                span_src33 = nullptr;
                span_staged33 = 0;
                span_dead33 = true;
                submit_batch_path = 2;
                if (!submit(dst_t, copy_src, bytes)) { *staged_out = staging_off; return false; }
                staging_off += bytes;
                continue;
            }
            span_staged33 += bytes;
            if (span_staged33 == ggml_nbytes(span33)) {
                if (!submit(span33, span_src33, span_staged33)) { *staged_out = staging_off; return false; }
                span_src33 = nullptr;
                span_staged33 = 0;
            }
        } else {
            if (!submit(dst_t, copy_src, bytes)) { *staged_out = staging_off; return false; }
        }
        staging_off += bytes;
    }
    if (span_src33 != nullptr) {
        if (!submit(span33, span_src33, span_staged33)) { *staged_out = staging_off; return false; }
        span_src33 = nullptr;
    }
    recon_end_bank();
    *staged_out = staging_off;
    if (submit_batch_out != nullptr) {
        *submit_batch_out = submit_batch_path;
    }
    return true;
}

// The load_layer body for one layer, executed on the worker thread. Differs
// from load_layer only in the mutual-exclusion owner (af->mtx guards the slot
// bookkeeping the main thread also writes) and in skipping the handoff and
// pinned-staging fast paths the PP feed never exercises in this prototype.
bool llama_layer_streaming::af_submit_copy(int il) {
    if (il < 0 || il >= n_layer) {
        return false;
    }
    const int s = il % n_slots;

    // free-slot + slot-bookkeeping critical section (shared with the main
    // thread's load_layer/wait_slot_ready/record_consumer_done)
    std::unique_lock<std::mutex> lk(af->mtx);
    llama_layer_streaming_slot & slot = slots[s];

    // the slot must have been released by the main thread (consumer done).
    // A slot that already holds THIS logical layer resident (preload or a
    // prior generation) needs no staging: return success without touching
    // anything, so a redundant enqueue can never re-copy over a resident
    // generation (the old OR-gate allowed exactly that — including over an
    // UNCONSUMED resident layer, e.g. the layer-major seed enqueue).
    if (!af->free_slot[s]) {
        if (slot.logical_layer == il && slot.generation > 0) {
            return true;   // already resident; nothing to stage
        }
        return false;      // claimed by another layer; not ours to stage
    }

    // T-GPLR-25 residue fix: the slot-reuse invariant requires the OVERWRITER
    // to wait for the prior resident generation's consumer to finish before
    // the copy touches the slot buffer. record_consumer_done frees the slot
    // (free_slot=1) BEFORE ev_consumer_done is guaranteed complete, so a
    // released slot can still have consumer_pending set. The wait can span a
    // full layer compute; it must not hold af->mtx (the main thread needs it
    // for its own load_layer/wait_slot_ready bookkeeping). Mirrors the
    // load_layer eviction block exactly (wait outside the lock, finish the
    // bookkeeping under the lock).
    if (slot.consumer_pending) {
        const ggml_backend_event_t ev_cd = (ggml_backend_event_t) slot.ev_consumer_done;
        lk.unlock();
        ls_stage_wait(ev_cd);
        lk.lock();
        // re-validate: while the lock was released the main thread's lookahead
        // may have staged the SAME layer into this slot (it targets the same
        // slot by construction: next = layer + n_slots). In that case the
        // copy is already issued and owned by the main thread — do not
        // double-stage and do not touch af_submitted (the main copy reports
        // through copy_submit/copy_complete only).
        if (slot.logical_layer == il && slot.generation > 0 && slot.t_copy_submit_us != 0) {
            return true;
        }
        slot.consumer_pending = false;
        total_evictions++;
        tel_slot_evict(slot.logical_layer, s, slot.generation);
    }

    const llama_ls_package & pkg = pkg_cache.at(il);
    if ((int) bundle_members.size() <= il || bundle_members[il] != pkg.members.size()) {
        return false;
    }
    const size_t nbytes = slot.layer_bytes.count(il) ? slot.layer_bytes.at(il) : 0;
    if (nbytes == 0) {
        return false;
    }

    // T-GPLR-32: one worker H2D in flight. Stream-ordering dependency only —
    // this call enqueues the wait on the copy stream and returns immediately
    // (no host block, no deadlock: the previous copy runs on this same stream
    // and completes by construction), so the memcpy hop below stays fully
    // overlapped with whatever the main thread is computing.
    if (pp_af_serialize) {
        ls_stage_wait(slot.ev_af_gate);
    }
    ls_stage_record(slot.ev_copy_start);
    slot.t_copy_submit_us = ggml_time_us();
    const uint32_t gen = slot.generation + 1;

    // T-GPLR-30: acquire the persistent pool mirror for this bank. A valid
    // mirror serves members straight from the pool (no hop). An lru miss
    // hands out an empty entry: the loop below FILLS it while staging (fill
    // cost = the per-slot hop it replaces) and the next pass hits.
    // T-GPLR-37: direct mmap-sourced staging — page-lock this package's mapped
    // source span so the DMAs below read the model's mapping directly (no hop).
    // A worker-side failure is not silent: the same registration runs on the
    // main thread's load_layer path and aborts the run with the driver error.
    if (copy_src_registered && !ls_register_pkg_source(il, slot, pkg)) {
        return false;
    }
    size_t pool_bytes = 0;
    bool pool_valid = false;
    char * pool_base = pp_pool_acquire(il, pool_bytes, pool_valid);

    char * staging = slot.buf_staging ? (char *) ggml_backend_buffer_get_base(slot.buf_staging.get()) : nullptr;
    size_t staging_off = 0;
    int sb_path_af = 0; // T-GPLR-33: the taken submission path (0/1/2)
    const bool staged_ok = stage_members(backend_copy, pkg, slot.layers.at(il), il, gen, slot,
                                         pool_base, pool_bytes, pool_valid, staging, &staging_off,
                                         &sb_path_af);
    pp_pool_release(il, pool_base, staging_off, staged_ok);
    if (!staged_ok) {
        return false;
    }

    ls_stage_record(slot.ev_copy_end);
    ls_stage_record(slot.ev_copy_ready);
    // T-GPLR-32: this worker DMA is now the one in flight; the next gated
    // worker copy waits on this event (recorded on the same stream, already
    // ordered — the record exists so the NEXT copy's wait has an event).
    if (pp_af_serialize) {
        ls_stage_record(slot.ev_af_gate);
    }
    slot.t_copy_enqueue_done_us = ggml_time_us();

    slot.generation   = gen;
    slot.logical_layer = il;
    slot.af_submitted  = true;      // wait_slot_ready consumes this flag
    af->free_slot[s]  = 0;          // now claimed by layer il
    total_bytes_transferred += nbytes;
    const double submit_us = (double) (slot.t_copy_enqueue_done_us - slot.t_copy_submit_us);
    // T-GPLR-08 parity: warm the NEXT slot's next layer's mmap pages from the
    // worker too (hint-only, pf has its own mutex; no lock-order risk since
    // af->mtx is released below and pf never takes af->mtx).
    pf_enqueue_from(il);
    lk.unlock();

    af_dma_inflight.fetch_add(1, std::memory_order_relaxed);
    tel_copy_submit(il, nbytes, gen, pp_af_serialize ? af_dma_inflight.load(std::memory_order_relaxed) : 0, sb_path_af);
    tel_async_feed_submit(il, s, gen);
    (void) submit_us;
    return true;
}

bool llama_layer_streaming::set_pp_async_feed(int n) {
    if (n <= 0) {
        pp_async_feed = 0;
        return true;
    }
    if (n_slots < 2) {
        error_msg = "set_pp_async_feed: requires --layer-slots >= 2 (with one slot the "
                    "reuse invariant forbids staging the next layer while the current "
                    "one computes)";
        return false;
    }
    if (af != nullptr) {
        error_msg = "set_pp_async_feed: already configured";
        return false;
    }
    af = std::unique_ptr<af_worker, af_worker_deleter>(new af_worker());
    af->owner = this;
    af->free_slot.assign(n_slots, 1);
    // depth 1: one layer ahead (the prototype contract). The queue is 1-deep;
    // the main thread enqueues the next layer right after its own wait.
    af_depth = 1;
    pp_async_feed = n;
    af->thread = std::thread([this] { af->run(); });
    tel_async_feed_init(af_depth, n_slots);
    return true;
}

// T-GPLR-32: serialize the worker's H2D DMAs to one in flight. Call before
// set_pp_async_feed (any time before the run starts is fine; the gate reads
// pp_af_serialize per submission). No worker is running yet, so there is no
// race window.
bool llama_layer_streaming::set_pp_af_serialize(bool on) {
    if (af != nullptr) {
        error_msg = "set_pp_af_serialize: must be set before the async-feed worker starts";
        return false;
    }
    pp_af_serialize = on;
    return true;
}

// T-GPLR-33 V1: member-batched staging. Reads pp_submit_batch per stage;
// safe to set any time before the run starts. The per-slot contiguity
// check ran at slot build (bank_span); a layer without a valid span (or a
// mid-bank source discontinuity) degrades to the legacy per-member path
// by design — every byte still lands exactly once.
// T-GPLR-35: staging-stream plumbing. See the header note; the mode switch is
// atomic across every staging site so an event can never be recorded on a
// stream that carries no copy work, and a wait always targets the stream that
// does.
bool llama_layer_streaming::ls_stage_aux() const {
    return pp_copy_overlap && backend_compute != nullptr && backend_copy != nullptr;
}

void llama_layer_streaming::ls_stage_record(void * ev) {
    if (ev == nullptr) {
        return;
    }
    if (ls_stage_aux()) {
        ggml_backend_cuda_event_record_aux(backend_compute, (ggml_backend_event_t) ev);
        return;
    }
    ggml_backend_event_record((ggml_backend_event_t) ev, backend_copy);
}

void llama_layer_streaming::ls_stage_wait(void * ev) {
    if (ev == nullptr) {
        return;
    }
    if (ls_stage_aux()) {
        ggml_backend_cuda_event_wait_aux(backend_compute, (ggml_backend_event_t) ev);
        return;
    }
    ggml_backend_event_wait(backend_copy, (ggml_backend_event_t) ev);
}

void llama_layer_streaming::ls_stage_drain() {
    // Conservative in BOTH modes: always drain the copy backend's own stream
    // (handoff migration and the TG row cache still submit there) and, when
    // staging runs on the aux stream, drain that too. Draining an idle stream
    // costs microseconds and cannot hide a pending copy.
    ggml_backend_synchronize(backend_copy);
    if (ls_stage_aux()) {
        ggml_backend_cuda_aux_synchronize(backend_compute);
    }
}

bool llama_layer_streaming::set_pp_copy_overlap(bool on) {
    // Intent only: the compute backend is resolved later, with the scheduler.
    // The aux stream is validated there (see layer_streaming_compute), where a
    // run that asked for overlap but cannot get it fails loudly.
    pp_copy_overlap = on;
    return true;
}

bool llama_layer_streaming::set_pp_recon(bool on) {
    pp_recon = on;
    return true;
}

// T-GPLR-53 A4 (modality discriminator): pre-timed pre-warm modes, as a bitfield.
// 0 = off (the default) runs nothing at all, so every existing shape stays
// byte-identical; 1 = CPU read-touch of the package source spans (the mapped bytes
// the staging path reads -- H1's page-side completion candidate); 2 = one dummy GPU
// launch + event sync (H3's submission-side candidate); 4 = a duration-matched
// no-touch delay (the FALSIFIER for (1): it burns the same wall time while touching
// nothing, so "the excess is page residency" is separable from "any ~1.7 s of
// elapsed time lets a lazy completion finish"); 8 = a duration-matched WRONG-REGION
// page touch (same page-touching activity and duration as (1), on a private scratch
// buffer instead of the package spans, so "touching pages at all" is separable from
// "touching the staged source's pages").
bool llama_layer_streaming::set_pp_prewarm(int mode) {
    if (mode < 0 || mode > 15) {
        error_msg = "set_pp_prewarm: mode must be 0..15 (bitfield: 1 source touch, 2 dummy launch, 4 no-touch delay, 8 wrong-region touch)";
        return false;
    }
    pp_prewarm = mode;
    return true;
}

// T-GPLR-53 A4: the duration the CONTROL modes (4, 8) are matched to. Measured
// 2026-09-22 on the Qwen3.6-35B-A3B 4k vehicle: the (1) walk of 17.068 GiB over
// 4,474,368 pages took 1673.5 ms (0.374 us/page). Rounded up, so a control can
// never be shorter than the walk it exists to falsify. Mode 4's duration is the
// runner's --pp-prewarm-delay-ms (default = this constant); mode 8 keeps it.
static const int    k_prewarm_control_ms    = 1700;
static const size_t k_prewarm_scratch_bytes = ((size_t) 64) * 1024u * 1024u;

bool llama_layer_streaming::pp_prewarm_run(int delay_ms) {
    if (pp_prewarm == 0) {
        return true; // off: no touch, no allocation, no submission
    }
    const int64_t t0 = ggml_time_us();

    // (1) CPU read-touch of the package source spans: one byte per 4 KiB page,
    // accumulated into a volatile sink that the fprintf below consumes, so the
    // walk cannot be elided. Read-only -- not one byte is written.
    if (pp_prewarm & 1) {
        volatile unsigned char sink = 0;
        size_t pages = 0;
        size_t bytes = 0;
        for (const llama_ls_package & pkg : pkg_cache) {
            for (const llama_ls_member & m : pkg.members) {
                const ggml_tensor * src = m.src;
                if (src == nullptr || src->data == nullptr) {
                    continue;
                }
                const unsigned char * base = (const unsigned char *) src->data;
                const size_t n = ggml_nbytes(src);
                for (size_t off = 0; off < n; off += 4096) {
                    sink = (unsigned char) (sink + base[off]);
                    pages++;
                }
                bytes += n;
            }
        }
        if (bytes == 0) {
            error_msg = "pp_prewarm: no package source spans to touch (is the residency policy active?)";
            return false;
        }
        fprintf(stderr, "pp-prewarm(1): CPU read-touch %.3f GiB / %zu pages -> sink %u in %.1f ms\n",
                (double) bytes / (1024.0 * 1024.0 * 1024.0), pages, (unsigned) sink,
                (ggml_time_us() - t0) / 1000.0);
    }

    // (2) One dummy GPU launch + sync: a 64 B H2D on the copy backend, recorded
    // and synchronised through a private event. The tensor is a throwaway, so no
    // weight byte is ever written; everything is freed before returning.
    if (pp_prewarm & 2) {
        if (backend_copy == nullptr) {
            error_msg = "pp_prewarm: copy backend not resolved";
            return false;
        }
        ggml_init_params mp = { 1024u * 1024u, nullptr, /*no_alloc=*/ true };
        ggml_context * meta = ggml_init(mp);
        if (meta == nullptr) {
            error_msg = "pp_prewarm: failed to create the scratch context";
            return false;
        }
        ggml_tensor * t = ggml_new_tensor_1d(meta, GGML_TYPE_F32, 16);
        const ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device);
        ggml_backend_buffer_t buf = (t != nullptr)
            ? ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, t))
            : nullptr;
        bool ok = (t != nullptr && buf != nullptr);
        std::vector<ggml_tensor *> one = { t };
        size_t off = 0;
        if (ok) {
            ok = alloc_tensors(buf, buft, one, off);
        }
        ggml_backend_event_t ev = nullptr;
        if (ok) {
            const float zeros[16] = { 0.0f };
            ggml_backend_tensor_set_async(backend_copy, t, zeros, 0, sizeof(zeros));
            ev = ggml_backend_event_new(ggml_backend_get_device(backend_copy));
            ok = (ev != nullptr);
            if (ok) {
                ggml_backend_event_record(ev, backend_copy);
                ggml_backend_event_synchronize(ev);
            }
        }
        if (ev != nullptr)  { ggml_backend_event_free(ev); }
        if (buf != nullptr) { ggml_backend_buffer_free(buf); }
        ggml_free(meta);
        if (!ok) {
            error_msg = "pp_prewarm: dummy GPU launch + sync failed";
            return false;
        }
        fprintf(stderr, "pp-prewarm(2): dummy GPU launch + sync ok in %.1f ms\n",
                (ggml_time_us() - t0) / 1000.0);
    }

    // (4) Duration-matched NO-TOUCH delay: the falsifier for (1). Same wall time,
    // zero package pages touched. If this clears the pass-1 excess, the excess is
    // not page residency and the (1) reading collapses.
    // The duration is the runner's --pp-prewarm-delay-ms (default 1700 = the
    // compiled control duration), so an un-flagged run is unchanged and the dose
    // sweep varies only this number.
    if (pp_prewarm & 4) {
        const int64_t t_start = ggml_time_us();
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        fprintf(stderr, "pp-prewarm(4): no-touch delay %d ms / 0 pages touched in %.1f ms\n",
                delay_ms, (ggml_time_us() - t_start) / 1000.0);
    }

    // (8) Duration-matched WRONG-REGION touch: same page-touching activity and the
    // same duration as (1), but on a private scratch buffer rather than the package
    // spans. Sized small on purpose (64 MiB, looped until the duration is met) so
    // the control stays cheap; read-only, nothing but the scratch is written.
    if (pp_prewarm & 8) {
        std::vector<unsigned char> scratch(k_prewarm_scratch_bytes, 0);
        const int64_t t_start = ggml_time_us();
        volatile unsigned char sink = 0;
        size_t pages = 0;
        int loops = 0;
        while ((ggml_time_us() - t_start) < (int64_t) k_prewarm_control_ms * 1000) {
            for (size_t off = 0; off < scratch.size(); off += 4096) {
                sink = (unsigned char) (sink + scratch[off]);
                pages++;
            }
            loops++;
        }
        fprintf(stderr, "pp-prewarm(8): wrong-region touch %.0f MiB x %d loop(s) / %zu page touches -> sink %u in %.1f ms\n",
                (double) scratch.size() / (1024.0 * 1024.0), loops, pages, (unsigned) sink,
                (ggml_time_us() - t_start) / 1000.0);
    }

    fprintf(stderr, "pp-prewarm(%d): total %.1f ms (pre-timed: outside the PP wall brackets)\n",
            pp_prewarm, (ggml_time_us() - t0) / 1000.0);
    return true;
}

// T-GPLR-41 (diagnostics only): flag setter for the pass-boundary
// memory-lifecycle recon. No behavior change beyond extra telemetry writes.
bool llama_layer_streaming::set_pp_mctx_recon(bool on) {
    pp_mctx_recon = on;
    return true;
}

bool llama_layer_streaming::set_route_telemetry(bool on) {
    route_telemetry = on;
    return true;
}

// T-GPLR-35 recon (diagnostics only). One identity line per run proves the
// stream topology the attribution numbers were measured on; one sample per
// staged bank measures the copy's own GPU window and where it started relative
// to the compute stream's backlog. Nothing here orders work.
void llama_layer_streaming::recon_begin_bank(int il, uint32_t gen) {
    recon_armed = false;
    if (!pp_recon) {
        return;
    }
    if (backend_compute == nullptr) {
        // Never a silent no-op: a recon run that cannot measure the compute
        // stream's position says so in the trace instead of emitting nothing.
        if (!recon_identity_done) {
            recon_identity_done = true;
            tel_write(tel_out, "{\"event\":\"recon_unavailable\",\"run_id\":%llu,\"reason\":\"compute_backend_unresolved\",\"t_us\":%lld}",
                (unsigned long long) run_id, (long long) ggml_time_us());
        }
        return;
    }
    if (!recon_identity_done) {
        recon_identity_done = true;
        const unsigned long long cs0 = (unsigned long long) (uintptr_t) ggml_backend_cuda_stream_handle(backend_compute, 0);
        const unsigned long long cs1 = (unsigned long long) (uintptr_t) ggml_backend_cuda_stream_handle(backend_compute, 1);
        const unsigned long long cp0 = (unsigned long long) (uintptr_t) ggml_backend_cuda_stream_handle(backend_copy, 0);
        tel_write(tel_out, "{\"event\":\"recon_identity\",\"run_id\":%llu,\"ubatch\":%llu,\"overlap\":%d,\"compute_s0\":%llu,\"compute_s1\":%llu,\"copy_s0\":%llu,\"same_ctx_stream\":%d,\"t_us\":%lld}",
            (unsigned long long) run_id, (unsigned long long) n_ubatches, pp_copy_overlap ? 1 : 0,
            cs0, cs1, cp0, (cs0 == cp0) ? 1 : 0, (long long) ggml_time_us());
    }
    if (recon_ring.empty()) {
        recon_ring.resize(8);
    }
    recon_flush_all();
    ls_recon_ev & e = recon_ring[recon_next];
    // the events must sit on the stream that actually carries this bank's DMA
    ggml_backend_t copy_be = pp_copy_overlap ? backend_compute : backend_copy;
    const int copy_stream_no = pp_copy_overlap ? 1 : 0;
    if (e.copy_start == nullptr) {
        e.copy_be        = copy_be;
        e.copy_stream_no = copy_stream_no;
        e.copy_start  = ggml_backend_cuda_timing_event_new(copy_be, copy_stream_no);
        e.copy_end    = ggml_backend_cuda_timing_event_new(copy_be, copy_stream_no);
        e.compute_ref = ggml_backend_cuda_timing_event_new(backend_compute, 0);
        if (e.copy_start == nullptr || e.copy_end == nullptr || e.compute_ref == nullptr) {
            return;
        }
    } else if (e.copy_be != copy_be || e.copy_stream_no != copy_stream_no) {
        return;   // flag changed mid-run: attribution would be ambiguous
    }
    e.il = il;
    e.gen = gen;
    // T-GPLR-53 addendum: capture the ubatch NOW, while n_ubatches still is the
    // staged bank's ubatch - the ring flushes lazily and the emitter must not
    // read the global again at flush time (that was the stale-label defect).
    e.ubatch = (unsigned long long) n_ubatches;
    e.emitted = false;
    recon_next = (recon_next + 1) % (int) recon_ring.size();
    // compute_ref marks this bank's position in the compute stream's queue; a
    // copy_start that fires BEFORE it means the DMA ran while that backlog was
    // still draining (overlap), at/after it means the copy waited for compute.
    ggml_backend_cuda_timing_event_record(e.compute_ref);
    ggml_backend_cuda_timing_event_record(e.copy_start);
    recon_armed = true;
}

void llama_layer_streaming::recon_end_bank() {
    if (!recon_armed) {
        return;
    }
    recon_armed = false;
    ls_recon_ev & e = recon_ring[(recon_next + (int) recon_ring.size() - 1) % (int) recon_ring.size()];
    ggml_backend_cuda_timing_event_record(e.copy_end);
}

void llama_layer_streaming::recon_flush_all() {
    for (auto & e : recon_ring) {
        if (e.emitted || e.il < 0) {
            continue;
        }
        const double copy_ms = ggml_backend_cuda_timing_event_elapsed_ms(e.copy_start, e.copy_end);
        const double lag_ms  = ggml_backend_cuda_timing_event_elapsed_ms(e.compute_ref, e.copy_start);
        if (copy_ms < 0.0 || lag_ms < 0.0) {
            continue;   // events not complete yet: retry on a later bank
        }
        e.emitted = true;
        tel_write(tel_out, "{\"event\":\"recon_attr\",\"run_id\":%llu,\"ubatch\":%llu,\"layer\":%d,\"generation\":%u,\"copy_gpu_ms\":%.3f,\"lag_ms\":%.3f,\"overlap\":%d,\"t_us\":%lld}",
            (unsigned long long) run_id, e.ubatch, e.il, e.gen,
            copy_ms, lag_ms, pp_copy_overlap ? 1 : 0, (long long) ggml_time_us());
    }
}

bool llama_layer_streaming::set_pp_submit_batch(bool on) {
    pp_submit_batch = on;
    return true;
}

void llama_layer_streaming::stop_pp_async_feed() {
    if (af == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(af->mtx);
        af->stop = true;
    }
    af->cv.notify_all();
    if (af->thread.joinable()) {
        af->thread.join();
    }
    af.reset();
    pp_async_feed = 0;
}

void llama_layer_streaming::af_stats(uint64_t & n_submit, uint64_t & n_ok, uint64_t & n_err,
                                     uint64_t & bytes, uint64_t & us) {
    n_submit = n_ok = n_err = bytes = us = 0;
    if (af == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(af->mtx);
    n_submit = af->n_submit;
    n_ok     = af->n_ok;
    n_err    = af->n_err;
    bytes    = af->bytes;
    us       = af->us;
}

// --- T-GPLR-30: persistent pinned pool -----------------------------------

// T-GPLR-30 pool protocol. Acquire returns an exclusively-owned base the
// caller fills (lru miss) or reads (hit); release ends the ownership. The
// eviction path drains the copy stream before handing out a buffer whose
// previous fill's DMAs may still be pending — the ordering that makes pool
// reuse safe against the async DMAs that read the pool (no torn weights).
char * llama_layer_streaming::pp_pool_acquire(int il, size_t & bytes, bool & contents_valid) {
    bytes = 0;
    contents_valid = false;
    if (pp_pool.empty()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(pp_pool_mtx);
    if (pp_pool_policy == 0) {
        // fixed: banks 0..N-1 pooled permanently, populated at configure
        if (il < 0 || il >= (int) pp_pool.size()) {
            // CAPACITY, not policy: the pool is smaller than the bank index.
            pool_n_miss++;
            tel_pool_event(il, false, 0, "out_of_range");
            return nullptr;
        }
        pp_pool_entry & e = pp_pool[il];
        if (!e.populated || e.active) {
            pool_n_miss++;
            tel_pool_event(il, false, 0, e.active ? "entry_active" : "not_populated");
            return nullptr;
        }
        e.active = true;
        pool_n_hit++;
        pool_bytes_saved += e.bytes;
        bytes = e.bytes;
        contents_valid = true;
        tel_pool_event(il, true, e.bytes, "resident");
        return (char *) ggml_backend_buffer_get_base(e.buf.get());
    }
    // lru: hit = populated entry mirroring il (reads only, so a pending DMA
    // from the fill pass is harmless — no rewrite happens on a hit)
    for (int i = (int) pp_pool.size() - 1; i >= 0; --i) {
        pp_pool_entry & e = pp_pool[i];
        if (e.populated && e.bank == il && !e.active) {
            if (i != (int) pp_pool.size() - 1) {
                pp_pool_entry moved = std::move(pp_pool[i]);
                pp_pool.erase(pp_pool.begin() + i);
                pp_pool.push_back(std::move(moved));
            }
            pp_pool_entry & back = pp_pool.back();
            back.active = true;
            pool_n_hit++;
            pool_bytes_saved += back.bytes;
            bytes = back.bytes;
            contents_valid = true;
            tel_pool_event(il, true, back.bytes, "resident");
            return (char *) ggml_backend_buffer_get_base(back.buf.get());
        }
    }
    // miss: evict the least-recent INACTIVE entry for il. A pending-DMA entry
    // (inflight) is drained first: its old contents are about to be rewritten
    // on CPU while earlier async DMAs may still read them.
    for (int i = 0; i < (int) pp_pool.size(); ++i) {
        pp_pool_entry & e = pp_pool[i];
        if (e.active) {
            continue;
        }
        llama_layer_streaming_slot & sl = slots[il % n_slots];
        const size_t nbytes = sl.layer_bytes.count(il) ? sl.layer_bytes.at(il) : 0;
        if (nbytes == 0 || nbytes > e.bytes) {
            continue; // oversized/unknown bank never fits this entry
        }
        if (e.inflight) {
            ls_stage_drain(); // drain pending staging copies
            e.inflight = false;
        }
        e.bank = il;
        e.bytes = nbytes;
        e.populated = false; // valid again once release(ok=true) runs
        e.active = true;
        // refresh recency: move to the back
        if (i != (int) pp_pool.size() - 1) {
            pp_pool_entry moved = std::move(pp_pool[i]);
            pp_pool.erase(pp_pool.begin() + i);
            pp_pool.push_back(std::move(moved));
        }
        pool_n_miss++;
        tel_pool_event(il, false, nbytes, "evicted");
        bytes = nbytes;
        return (char *) ggml_backend_buffer_get_base(pp_pool.back().buf.get());
    }
    // OCCUPANCY/CAPACITY: nothing was evictable. Counted, because leaving it
    // silent is what made the published rate's denominator incomplete.
    pool_n_miss++;
    tel_pool_event(il, false, 0, "no_free_entry");
    return nullptr; // every entry busy or oversized: serve from mmap
}

void llama_layer_streaming::pp_pool_release(int il, char * base, size_t staged_bytes, bool ok) {
    if (base == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(pp_pool_mtx);
    for (auto & e : pp_pool) {
        if ((char *) ggml_backend_buffer_get_base(e.buf.get()) != base) {
            continue;
        }
        e.active = false;
        if (pp_pool_policy == 1) {
            if (ok && staged_bytes == e.bytes) {
                e.populated = true;       // fill completed exactly
            } else {
                e.populated = false;      // partial/failed fill: never serve it
                e.bank = -1;
            }
            e.inflight = true;            // the fill's DMAs may still be reading
        }
        return;
    }
}

void llama_layer_streaming::tel_pool_event(int il, bool hit, size_t bytes, const char * why) {
    // cap/policy ride on every event so no join is needed to read the rate:
    // a 0% over cap=4 is a capacity statement, not a policy statement.
    tel_write(tel_out, "{\"event\":\"pool_event\",\"run_id\":%llu,\"layer\":%d,\"hit\":%s,\"why\":\"%s\",\"bytes\":%zu,\"cap\":%d,\"policy\":%d,\"phase\":%d,\"t_us\":%lld}",
        (unsigned long long) run_id, il, hit ? "true" : "false",
        why ? why : "unspecified", bytes,
        (int) pp_pool.size(), pp_pool_policy, phase, (long long) ggml_time_us());
}

void llama_layer_streaming::pool_stats(uint64_t & n_hit, uint64_t & n_miss, uint64_t & bytes_saved) {
    n_hit = pool_n_hit;
    n_miss = pool_n_miss;
    bytes_saved = pool_bytes_saved;
}

bool llama_layer_streaming::set_pp_pool(int n_banks, const char * policy) {
    if (n_banks <= 0) {
        return true; // pool off
    }
    if (!use_pinned_staging) {
        error_msg = "set_pp_pool: requires --copy-source pinned (the pool IS a pinned "
                    "buffer; with direct staging there is no memcpy hop to remove)";
        return false;
    }
    if (n_banks > n_layer) {
        error_msg = "set_pp_pool: pool size " + std::to_string(n_banks) + " exceeds n_layer " +
                    std::to_string(n_layer);
        return false;
    }
    std::string pol = policy ? policy : "fixed";
    if (pol != "fixed" && pol != "lru") {
        error_msg = "set_pp_pool: policy must be 'fixed' or 'lru' (got '" + pol + "')";
        return false;
    }
    if (slots.empty()) {
        error_msg = "set_pp_pool: called before init (no slots)";
        return false;
    }
    pp_pool_n = n_banks;
    pp_pool_policy = pol == "lru" ? 1 : 0;
    // per-layer package sizes live in slot.layer_bytes (slot = il % n_slots)
    auto layer_sz = [&](int il) -> size_t {
        llama_layer_streaming_slot & sl = slots[il % n_slots];
        return sl.layer_bytes.count(il) ? sl.layer_bytes.at(il) : 0;
    };
    // fixed: entry i mirrors bank i; sized exactly to that bank's package.
    // lru: all entries sized to the max bank package (any bank may land in
    // any entry); starts empty (populated=false).
    size_t max_bytes = 0;
    for (int il = 0; il < n_layer; ++il) {
        if (layer_sz(il) > max_bytes) {
            max_bytes = layer_sz(il);
        }
    }
    if (max_bytes == 0) {
        error_msg = "set_pp_pool: no package byte sizes (model has no packageable layers?)";
        return false;
    }
    const int capacity = n_banks;
    pp_pool.resize(capacity);
    const int64_t t_pool0 = ggml_time_us();
    for (int i = 0; i < capacity; ++i) {
        pp_pool_entry & e = pp_pool[i];
        const size_t sz = pp_pool_policy == 0 ? layer_sz(i) : max_bytes;
        ggml_backend_buffer_type_t buft_staging = ggml_backend_cuda_host_buffer_type();
        e.buf.reset(ggml_backend_buft_alloc_buffer(buft_staging, sz));
        if (!e.buf || ggml_backend_buffer_get_type(e.buf.get()) != buft_staging) {
            error_msg = "set_pp_pool: failed to allocate pinned pool buffer " + std::to_string(i) +
                        " (" + std::to_string(sz) + " bytes) — driver locked-page quota may cap the "
                        "persistent pool on this box (T-GPLR-29 cudaHostRegister precedent)";
            pp_pool.clear();
            pp_pool_n = 0;
            return false;
        }
        e.bytes = sz;
        if (pp_pool_policy == 0) {
            // populate once from the mmap source (the only memcpy the fixed
            // policy ever pays for this bank) — this doubles as first-touch
            // warming for the pooled banks
            const llama_ls_package & pkg = pkg_cache.at(i);
            char * base = (char *) ggml_backend_buffer_get_base(e.buf.get());
            size_t off = 0;
            for (const auto & m : pkg.members) {
                if (m.src == nullptr) {
                    continue;
                }
                memcpy(base + off, m.src->data, ggml_nbytes(m.src));
                off += ggml_nbytes(m.src);
            }
            if (off != layer_sz(i)) {
                error_msg = "set_pp_pool: package member bytes " + std::to_string(off) +
                            " != layer_bytes " + std::to_string(layer_sz(i)) + " for layer " +
                            std::to_string(i);
                pp_pool.clear();
                pp_pool_n = 0;
                return false;
            }
            e.populated = true;
            e.bank = i;
        } else {
            e.populated = false;
            e.bank = -1;
        }
    }
    const int64_t t_pool1 = ggml_time_us();
    // total pinned commitment + populated bytes for the init event
    size_t total_bytes = 0, populated_bytes = 0;
    for (const auto & e : pp_pool) {
        total_bytes += e.bytes;
        if (e.populated) {
            populated_bytes += e.bytes;
        }
    }
    // working_set is published because capacity/working_set IS the whole
    // explanation for a zero hit rate; without it a reader cannot tell a
    // structural cliff from a broken policy.
    tel_write(tel_out, "{\"event\":\"pool_init\",\"run_id\":%llu,\"n\":%d,\"working_set\":%d,\"policy\":\"%s\",\"bytes\":%zu,\"populated_bytes\":%zu,\"populate_us\":%.1f,\"t_us\":%lld}",
        (unsigned long long) run_id, pp_pool_n, n_layer, pol.c_str(), total_bytes, populated_bytes,
        (double) (t_pool1 - t_pool0), (long long) ggml_time_us());
    return true;
}

void llama_layer_streaming::tel_async_feed_init(int depth, int n_slots_) {
    tel_write(tel_out, "{\"event\":\"async_feed_init\",\"run_id\":%llu,\"depth\":%d,\"n_slots\":%d,\"t_us\":%lld}",
        (unsigned long long) run_id, depth, n_slots_, (long long) ggml_time_us());
}

void llama_layer_streaming::tel_async_feed_submit(int il, int slot, uint32_t generation) {
    tel_write(tel_out, "{\"event\":\"async_feed_submit\",\"run_id\":%llu,\"ubatch\":%llu,\"layer\":%d,\"slot\":%d,\"generation\":%u,\"phase\":%d,\"t_us\":%lld}",
        (unsigned long long) run_id, (unsigned long long) n_ubatches, il, slot, generation, phase, (long long) ggml_time_us());
}

void llama_layer_streaming::tel_async_feed_ready(int il, int slot, uint32_t generation, double overlap_us) {
    tel_write(tel_out, "{\"event\":\"async_feed_ready\",\"run_id\":%llu,\"ubatch\":%llu,\"layer\":%d,\"slot\":%d,\"generation\":%u,\"phase\":%d,\"overlap_us\":%.1f,\"t_us\":%lld}",
        (unsigned long long) run_id, (unsigned long long) n_ubatches, il, slot, generation, phase, overlap_us, (long long) ggml_time_us());
}

bool llama_layer_streaming::set_pp_prefetch_ahead(int n) {
    if (n <= 0) {
        pp_prefetch_ahead = 0;
        return true;
    }
#if defined(LS_PF_ENABLED)
    if (pf != nullptr) {
        error_msg = "set_pp_prefetch_ahead: already configured";
        return false;
    }
    if ((int) pkg_cache.size() < n_layer) {
        error_msg = "set_pp_prefetch_ahead: package cache not initialized";
        return false;
    }
    // Bind the API once (dynamic, mirroring the vendored llama-mmap.cpp
    // precedent for Win8+ APIs). A failed bind degrades to today's behavior:
    // configure succeeds, nothing is prefetched, and the init event records
    // api_ok=false so the probe result is never silently empty.
    HMODULE h_kernel32 = GetModuleHandleW(L"kernel32.dll");
    BOOL (WINAPI * pPrefetchVirtualMemoryFn)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG) = nullptr;
    if (h_kernel32 != nullptr) {
        pPrefetchVirtualMemoryFn = (decltype(pPrefetchVirtualMemoryFn))(void *) GetProcAddress(h_kernel32, "PrefetchVirtualMemory");
    }
    // Validate every package member's data pointer with VirtualQuery: the
    // page must be a committed, file-backed (MEM_MAPPED) region and the tensor
    // must fit within that one region. (Region-level guard: the mmap span list
    // lives in llama_model::impl, which this module does not touch. A pointer
    // into another file's mapping would still pass, but package members point
    // into the GGUF mapping by construction and a misdirected hint can only
    // warm the wrong pages, never feed wrong data.) Members failing the check
    // are skipped with a recorded count (hint semantics: degrade, don't abort).
    pf_ranges_ptr.assign(n_layer, {});
    pf_ranges_len.assign(n_layer, {});
    size_t n_ranges  = 0;
    size_t n_skipped = 0;
    for (int il = 0; il < n_layer; ++il) {
        const llama_ls_package & pkg = pkg_cache.at(il);
        for (const auto & m : pkg.members) {
            const ggml_tensor * t = m.src;
            if (t == nullptr || t->data == nullptr || ggml_nbytes(t) == 0) {
                continue;
            }
            char *       begin = (char *) t->data;
            const size_t len   = ggml_nbytes(t);
            MEMORY_BASIC_INFORMATION mbi {};
            const bool ok = VirtualQuery(begin, &mbi, sizeof(mbi)) != 0 &&
                            mbi.State == MEM_COMMIT && mbi.Type == MEM_MAPPED &&
                            begin >= (char *) mbi.BaseAddress &&
                            begin + len <= (char *) mbi.BaseAddress + mbi.RegionSize;
            if (ok) {
                pf_ranges_ptr[il].push_back(begin);
                pf_ranges_len[il].push_back(len);
                n_ranges++;
            } else {
                n_skipped++;
            }
        }
    }
    pf.reset(new pf_worker());
    pf->owner = this;
    pf->pPrefetchVirtualMemory = pPrefetchVirtualMemoryFn;
    try {
        pf->thread = std::thread([w = pf.get()] { w->run(); });
    } catch (const std::system_error & e) {
        pf.reset();
        error_msg = std::string("set_pp_prefetch_ahead: failed to start worker thread: ") + e.what();
        return false;
    }
    pp_prefetch_ahead = n;
    tel_prefetch_init(n, n_layer, n_ranges, n_skipped, pPrefetchVirtualMemoryFn != nullptr);
    return true;
#else
    (void) n;
    error_msg = "set_pp_prefetch_ahead: PrefetchVirtualMemory unavailable on this platform/SDK";
    return false;
#endif
}

void llama_layer_streaming::pf_enqueue_from(int il) {
    if (pf == nullptr || pp_prefetch_ahead <= 0) {
        return;
    }
    const int next = il + pp_prefetch_ahead;
    if (next < 0 || next >= n_layer || pf_ranges_ptr[next].empty()) {
        return;
    }
    size_t bytes_total = 0;
    for (const size_t l : pf_ranges_len[next]) {
        bytes_total += l;
    }
    {
        std::lock_guard<std::mutex> lk(pf->mtx);
        if (pf->queue.size() >= (size_t) pp_prefetch_ahead) {
            pf->queue.pop_front(); // N-deep window; the newest range wins
        }
        pf->queue.emplace_back(next, phase);
        pf->cv.notify_one();
    }
    tel_prefetch_submit(next, phase, (int) pf_ranges_ptr[next].size(), bytes_total);
}

void llama_layer_streaming::stop_prefetch_worker() {
    if (pf == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(pf->mtx);
        pf->stop = true;
    }
    pf->cv.notify_all();
    if (pf->thread.joinable()) {
        pf->thread.join();
    }
    pf.reset();
}

void llama_layer_streaming::pf_stats(uint64_t & n_submit, uint64_t & n_complete, uint64_t & n_err,
                                     uint64_t & bytes, uint64_t & us) {
    n_submit = n_complete = n_err = bytes = us = 0;
    if (pf != nullptr) {
        std::lock_guard<std::mutex> lk(pf->mtx);
        n_submit   = pf->n_submit;
        n_complete = pf->n_complete;
        n_err      = pf->n_err;
        bytes      = pf->bytes;
        us         = pf->us;
    }
}

void llama_layer_streaming::tel_prefetch_init(int ahead, int layers, size_t ranges, size_t skipped, bool api_ok) {
    tel_write(tel_out,
        "{\"event\":\"prefetch_init\",\"run_id\":%llu,\"ahead\":%d,\"layers\":%d,\"ranges\":%zu,"
        "\"skipped_members\":%zu,\"api_ok\":%s}",
        (unsigned long long) run_id, ahead, layers, ranges, skipped, api_ok ? "true" : "false");
}

void llama_layer_streaming::tel_prefetch_submit(int il, int phase, int ranges, size_t bytes) {
    tel_write(tel_out,
        "{\"event\":\"prefetch_submit\",\"run_id\":%llu,\"layer\":%d,\"phase\":%d,\"ranges\":%d,\"bytes\":%zu,\"t_us\":%lld}",
        (unsigned long long) run_id, il, phase, ranges, bytes, (long long) ggml_time_us());
}

void llama_layer_streaming::tel_prefetch_complete(int il, int phase, int ranges, size_t bytes, double dur_us, uint32_t err) {
    tel_write(tel_out,
        "{\"event\":\"prefetch_complete\",\"run_id\":%llu,\"layer\":%d,\"phase\":%d,\"ranges\":%d,\"bytes\":%zu,"
        "\"duration_us\":%.1f,\"error\":%u}",
        (unsigned long long) run_id, il, phase, ranges, bytes, dur_us, err);
}

llama_layer_streaming::~llama_layer_streaming() {
    stop_pp_async_feed();   // T-GPLR-25: join before the telemetry sink closes
    stop_prefetch_worker(); // T-GPLR-08: join before the telemetry sink closes
    // Any marker taken after this sink's opening flush is written here, before the stream
    // closes - that is what keeps a post-sink marker from being lost silently.
    llama_ls_tel_flush_early(tel_out);
    if (tel_out != nullptr) {
        fclose(tel_out);
        tel_out = nullptr;
    }
    for (auto & set : sets) {
        if (set.res != nullptr) {
            delete set.res;
            set.res = nullptr;
        }
        if (set.ctx != nullptr) {
            ggml_free(set.ctx);
            set.ctx = nullptr;
        }
    }
    cur = nullptr;
    for (auto & slot : slots) {
        if (slot.ctx_meta != nullptr) {
            ggml_free(slot.ctx_meta);
            slot.ctx_meta = nullptr;
        }
        if (slot.ev_copy_ready != nullptr) {
            ggml_backend_event_free((ggml_backend_event_t) slot.ev_copy_ready);
            slot.ev_copy_ready = nullptr;
        }
        if (slot.ev_consumer_done != nullptr) {
            ggml_backend_event_free((ggml_backend_event_t) slot.ev_consumer_done);
            slot.ev_consumer_done = nullptr;
        }
        if (slot.ev_copy_start != nullptr) {
            ggml_backend_event_free((ggml_backend_event_t) slot.ev_copy_start);
            slot.ev_copy_start = nullptr;
        }
        if (slot.ev_copy_end != nullptr) {
            ggml_backend_event_free((ggml_backend_event_t) slot.ev_copy_end);
            slot.ev_copy_end = nullptr;
        }
        if (slot.ev_af_gate != nullptr) { // T-GPLR-32
            ggml_backend_event_free((ggml_backend_event_t) slot.ev_af_gate);
            slot.ev_af_gate = nullptr;
        }
    }
    if (ctx_handoff != nullptr) {
        ggml_free(ctx_handoff);
        ctx_handoff = nullptr;
    }
    if (backend_copy != nullptr) {
        ggml_backend_free(backend_copy);
        backend_copy = nullptr;
    }
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

bool llama_layer_streaming::init(const llama_model & model, int n_slots_, int device_, const char * tel_path_, int max_ubatch, int max_n_kv,
                                 llama_pp_schedule schedule_, int64_t pp_tokens_, bool use_pinned_staging_, int telemetry_chunk_stride_,
                                 bool use_implicit_causal_mask_, bool pw_probe_,
                                 ggml_type act_slab_dtype_, int tg_max_width_, bool copy_src_registered_,
                                 bool copy_src_reg_churn_) {
    n_slots = n_slots_;
    device  = device_;
    use_pinned_staging = use_pinned_staging_;
    copy_src_registered = copy_src_registered_;
    copy_src_reg_churn = copy_src_reg_churn_;
    telemetry_chunk_stride = telemetry_chunk_stride_;
    use_implicit_causal_mask = use_implicit_causal_mask_;
    pw_probe = pw_probe_;
    act_slab_dtype = act_slab_dtype_;
    tg_max_width = tg_max_width_ > 0 ? tg_max_width_ : 1;

    if (n_slots <= 0 || n_slots > 16) {
        error_msg = "invalid slot count: must be in 1..16";
        return false;
    }
    if (max_ubatch <= 0 || max_n_kv <= 0) {
        error_msg = "invalid max ubatch or mask width";
        return false;
    }
    if (telemetry_chunk_stride <= 0) {
        error_msg = "invalid telemetry chunk stride";
        return false;
    }
    if (schedule_ != LLAMA_PP_SCHEDULE_BATCH_MAJOR && schedule_ != LLAMA_PP_SCHEDULE_LAYER_MAJOR) {
        error_msg = "invalid pp schedule";
        return false;
    }
    schedule  = schedule_;
    pp_tokens = pp_tokens_;
    if (schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR) {
        // T-Q38CTX-01: any positive pp length is legal; the layout is full
        // ubatch-sized chunks plus one optional 1..ubatch-1 token tail.
        // Never pads fake tokens.
        chunks = llama_ls_chunk_layout(pp_tokens_, max_ubatch);
        if (chunks.empty()) {
            error_msg = "layer-major requires pp_tokens > 0";
            return false;
        }
        n_chunks = (int64_t) chunks.size();
    }

    const llama_hparams & hparams = model.hparams;
    n_layer = (int) hparams.n_layer();
    // T-GPLR-82 S3/S4 (2026-10-10): per-layer "the split FFN graph was BUILT".
    // The model builder sets it when ls_split_in_play says the mechanism can
    // admit a row, and the executor reads it instead of experts_host_n, so the
    // two halves cannot disagree. Sized HERE -- init() is the bring-up that
    // precedes every graph build. Sizing it in begin_row_set() instead was an
    // out-of-bounds write (access violation 0xC0000005 during "building
    // persistent graphs", measured 2026-10-10), because begin_row_set() runs
    // AFTER the persistent graphs are built. Nothing clears it afterwards: the
    // graphs persist for the run.
    row_set_split_built.assign(n_layer, 0);
    n_embd_io = hparams.n_embd_inp();

    recr.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        recr[il] = hparams.is_recr(il) ? 1 : 0;
    }

    host_layers.clear();
    for (int il = 0; il < n_layer; ++il) {
        host_layers.push_back(&model.layers[il]);
    }
    // T-Q38HG-06: the NextN/MTP layer (if present) joins the host source set
    // so begin_handoff(mtp) can migrate blk.64 and graph_mtp can bind its
    // device bundle. Tensors are null when the model was loaded without
    // load_mtp; the package builders skip null members.
    if ((int) model.layers.size() > n_layer) {
        host_layers.push_back(&model.layers[n_layer]);
    }

    // T-GPLR-01: select the architecture adapter and enumerate+validate every
    // trunk-layer package descriptor before any device allocation or copy.
    {
        std::string reason;
        adapter = llama_ls_select_adapter(model.arch, reason);
        model_ref = &model;
        if (adapter == nullptr || adapter->package == nullptr) {
            error_msg = "layer-streaming: " + (reason.empty() ? "no adapter selected" : reason);
            return false;
        }
        pkg_cache.clear();
        pkg_cache.resize(n_layer);
        for (int il = 0; il < n_layer; ++il) {
            std::string perr;
            if (!adapter->package(model_ref, il, *host_layers[il], pkg_cache[il], perr)) {
                error_msg = "package enumeration failed for layer " + std::to_string(il) + ": " + perr;
                return false;
            }
            if (!llama_ls_validate_package(pkg_cache[il], perr, /*require_src=*/false)) {
                error_msg = "invalid package descriptor for layer " + std::to_string(il) + ": " + perr;
                return false;
            }
        }
    }

    stats.resize(n_layer);

    // copy backend: a dedicated CUDA backend instance (its own stream)
    backend_copy = ggml_backend_cuda_init(device);
    if (backend_copy == nullptr) {
        error_msg = "failed to init CUDA copy backend on device " + std::to_string(device);
        return false;
    }

    ggml_backend_dev_t dev_copy = ggml_backend_get_device(backend_copy);
    ggml_backend_buffer_type_t buft_cuda = ggml_backend_cuda_buffer_type(device);
    ggml_backend_buffer_type_t buft_cpu  = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t buft_staging = use_pinned_staging ? ggml_backend_cuda_host_buffer_type() : nullptr;

    // slot bundles: tensor objects in a per-slot meta context, then one CUDA
    // buffer per slot, then allocation and per-slot events
    slots.resize(n_slots);

    for (int s = 0; s < n_slots; ++s) {
        auto & slot = slots[s];

        slot.ev_copy_ready    = ggml_backend_event_new(dev_copy);
        slot.ev_consumer_done = ggml_backend_event_new(dev_copy);
        slot.ev_copy_start    = ggml_backend_event_new(dev_copy);
        slot.ev_copy_end      = ggml_backend_event_new(dev_copy);
        slot.ev_af_gate       = ggml_backend_event_new(dev_copy); // T-GPLR-32
        if (!slot.ev_copy_ready || !slot.ev_consumer_done || !slot.ev_copy_start || !slot.ev_copy_end || !slot.ev_af_gate) {
            error_msg = "failed to create CUDA events for slot " + std::to_string(s);
            return false;
        }

        ggml_init_params meta_params = { 4u*1024u*1024u, nullptr, true };
        slot.ctx_meta = ggml_init(meta_params);
        if (slot.ctx_meta == nullptr) {
            error_msg = "failed to init slot tensor context";
            return false;
        }

        // per-logical-layer tensor bundles: every layer assigned to this slot
        // (il % n_slots == s) gets its own bundle with that layer's exact
        // shapes and quant types. All bundles alias the same device buffer —
        // only one layer is resident at a time, so VRAM is unchanged
        // (buffer sized to the largest package). This supports dynamic/imatrix
        // models where different layers carry different quant types.
        // per-logical-layer tensor bundles. Dense policy: every package member
        // is a fresh device tensor aliasing the slot buffer. MoE expert-bank
        // policy (adapter merge_resident): the bundle starts as a copy of the
        // resident model layer and ONLY the package members are replaced by
        // slot device tensors; the other members keep pointing at their
        // existing CUDA-resident buffers. collect/load iterate the package
        // descriptor, so non-package members are never (re)allocated or copied.
        auto create_bundle = [&](const llama_layer & src, const llama_ls_package & pkg,
                                 llama_layer & dst, size_t & bytes) -> bool {
            if (adapter->merge_resident) {
                dst = src;
            }
            bytes = 0;
            for (const auto & m : pkg.members) {
                const ggml_tensor * t = m.src;
                if (t == nullptr) {
                    continue;
                }
                ls_dst(dst, m) = make_dev_tensor(slot.ctx_meta, t);
                if (ls_dst(dst, m) == nullptr) {
                    return false;
                }
                ggml_set_name(ls_dst(dst, m), t->name);
                bytes += ggml_backend_buft_get_alloc_size(buft_cuda, ls_dst(dst, m));
            }
            return true;
        };

        size_t slot_bytes_max = 0;
        for (int il = s; il < n_layer; il += n_slots) {
            llama_layer & dst = slot.layers[il];
            size_t bytes = 0;
            if (!create_bundle(*host_layers[il], pkg_cache[il], dst, bytes)) {
                error_msg = "failed to create slot tensor bundle for layer " + std::to_string(il);
                return false;
            }
            slot.layer_bytes[il] = bytes;
            slot_bytes_max = std::max(slot_bytes_max, bytes);
            if ((int) bundle_members.size() <= il) {
                bundle_members.resize(il + 1, 0);
            }
            bundle_members[il] = pkg_cache[il].members.size();
        }

        const size_t slot_bytes = slot_bytes_max + 4096;
        slot.buf.reset(ggml_backend_buft_alloc_buffer(buft_cuda, slot_bytes));
        if (!slot.buf) {
            error_msg = "failed to allocate slot buffer (" + std::to_string(slot_bytes) + " bytes)";
            return false;
        }

        auto collect_bundle = [&](const llama_ls_package & pkg, llama_layer & bundle,
                                  std::vector<ggml_tensor *> & out) {
            for (const auto & m : pkg.members) {
                ggml_tensor * t = ls_dst(bundle, m);
                if (t != nullptr) {
                    out.push_back(t);
                }
            }
        };

        // every per-layer bundle aliases the same buffer from offset 0;
        // they overlap by design (one resident layer per slot at a time)
        for (int il = s; il < n_layer; il += n_slots) {
            std::vector<ggml_tensor *> bundle_tensors;
            collect_bundle(pkg_cache[il], slot.layers[il], bundle_tensors);
            size_t bundle_off = 0;
            if (!alloc_tensors(slot.buf.get(), buft_cuda, bundle_tensors, bundle_off)) {
                error_msg = "failed to allocate slot tensors for layer " + std::to_string(il);
                return false;
            }

            // T-GPLR-33 V1 contiguity check (recorded per layer, evaluated
            // per stage): the bundle's members must tile [base, base+total)
            // EXACTLY — dense tiling (no alignment gaps). Members already
            // sit at offsets 0, 144 MiB, 288 MiB on the MoE vehicle; the
            // buffer alignment (256 B) divides 144 MiB, so the tiling is
            // expected to hold there. If it holds, a 1-D I8 overlay tensor
            // spanning the bank aliases the member region and staging can
            // issue ONE H2D for the package (set_tensor_async on the span
            // = one cudaMemcpyAsync, byte-equivalent to the member loop).
            const llama_ls_package & pkg33 = pkg_cache[il];
            char * buf_base = (char *) ggml_backend_buffer_get_base(slot.buf.get());
            size_t want_off = 0;
            bool tiled = !pkg33.members.empty();
            size_t total33 = 0;
            for (const auto & m : pkg33.members) {
                const ggml_tensor * dt = ls_dst(slot.layers[il], m);
                if (dt == nullptr || dt->data == nullptr) {
                    tiled = false;
                    break;
                }
                const size_t nb = ggml_nbytes(dt);
                if ((char *) dt->data != buf_base + want_off) {
                    tiled = false;
                    break;
                }
                want_off += nb;
                total33 += nb;
            }
            if (tiled && total33 > 0) {
                ggml_tensor * span = ggml_new_tensor_1d(slot.ctx_meta, GGML_TYPE_I8, total33);
                if (span != nullptr) {
                    ggml_backend_tensor_alloc(slot.buf.get(), span, buf_base);
                    if (span->data != nullptr) {
                        slot.bank_span[il] = span;
                    }
                }
            }
        }

        if (use_pinned_staging) {
            slot.buf_staging.reset(ggml_backend_buft_alloc_buffer(buft_staging, slot_bytes));
            if (!slot.buf_staging || ggml_backend_buffer_get_type(slot.buf_staging.get()) != buft_staging) {
                error_msg = "failed to allocate pinned staging buffer for slot " + std::to_string(s);
                return false;
            }
        }
    }

    // persistent activation buffer: one act pair per coexisting graph set.
    // Sized for both PP variants (each 2 x [n_embd_inp, max_ubatch] F32) plus
    // the TG continuing variants: under MTP batch verify the prebuild binds
    // one act pair per width w in 1..tg_max_width (cumulative
    // 2 * sum_w w * n_embd_io), matching begin_set's allocation order. The
    // legacy width-1-only budget is the w=1 term of the same sum (bit-identical
    // when tg_max_width == 1), so this generalizes rather than special-cases.
    // T-GPLR-11: a third PP pair covers the batch-major tail ubatch (width
    // t = pp_tokens % ubatch when 0 < t < ubatch and there are > 2 ubatches;
    // upper-bounded here by max_ubatch). Each PP pair is
    // 2 x n_embd_io x width floats, so the PP term is 3 pairs = 6 halves.
    {
        const int64_t tg_sum_w = (int64_t) tg_max_width * (tg_max_width + 1) / 2;
        const size_t act_bytes = (6 * (size_t) n_embd_io * max_ubatch + 2 * (size_t) n_embd_io * tg_sum_w + 4096) * sizeof(float);
        buf_act.reset(ggml_backend_buft_alloc_buffer(buft_cuda, act_bytes));
        if (!buf_act) {
            error_msg = "failed to allocate activation buffer";
            return false;
        }
    }

    // persistent host buffer for graph input tensors. In per-width (pw_mode)
    // the buffer holds only the static per-set inputs (max 3 sets), not the
    // whole-prompt embedding input. The per-chunk input data is rewritten into
    // these same tensor objects before each chunk launch.
    {
        const int n_masks = use_implicit_causal_mask ? 0 :
            (schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR ? (int) n_chunks : 2);
        const size_t mask_bytes = (size_t) max_n_kv * max_ubatch * sizeof(ggml_fp16_t);
        // T-GPLR-05 long-prefill fix: the embd total is exact (sum of all
        // chunk embd slices), but each chunk set also binds small non-embd
        // graph inputs (tokens/positions/out-ids) and alignment padding that
        // the flat 64 MiB slack cannot cover at 512-chunk scale (262K-token
        // prefill exhausted the buffer at chunk 493/512, measured ~0.3 MiB of
        // extra per chunk). Budget 1 MiB per chunk of headroom; this is a
        // host-side buffer, so the extra ~0.5 GB at 262K costs no VRAM.
        const size_t embd_bytes = schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR
            ? (size_t) n_embd_io * (size_t) pp_tokens * sizeof(float)
              + (size_t) n_chunks * (size_t) 1024u * 1024u : 0;
        // T-GPLR-62 S2: the row-set TG companion graphs each carry TWO
        // [n_expert_used, n_tokens] I32 host inputs -- sel_in (the router's
        // absolute expert ids) and sel_pos_in (the same experts as window
        // POSITIONS, T-GPLR-62 S2). They are INPUT-flagged, so finish_build's
        // context walk allocates them from this buffer whatever the set filter
        // later decides about the snapshot list. Budgeting them EXPLICITLY, at
        // max_ubatch rather than the eventual token count, means they are backed
        // from the primary buffer by design instead of surviving on incidental
        // slack and spilling to an extra host allocation. Host memory only, so
        // this costs no VRAM. Sized for the full trunk because every layer gets
        // a companion graph.
        const size_t rowset_input_bytes = row_set_requested
            ? (size_t) 2u * (size_t) n_layer * (size_t) model_ref->hparams.n_expert_used
              * (size_t) max_ubatch * sizeof(int32_t)
            : 0;
        const size_t input_bytes = pw_probe
            ? (size_t) 64u * 1024u * 1024u  // per-width: fixed budget for 3 sets
            : ((size_t) n_masks * mask_bytes + embd_bytes + rowset_input_bytes + 64u * 1024u * 1024u);
        buf_inputs.reset(ggml_backend_buft_alloc_buffer(buft_cpu, input_bytes));
        if (!buf_inputs) {
            error_msg = "failed to allocate input buffer";
            return false;
        }
    }

    // layer-major activation slabs (SDD section 13): two stable full-prompt
    // F32 slabs in one CUDA buffer. Per-chunk graph sets bind their act
    // slices as tensors allocated from this buffer at fixed offsets, so a
    // persistent graph always executes against the stable address it was
    // built with. Batch-major never allocates them (its per-set act pairs
    // use buf_act).
    if (schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR) {
        const size_t slab_elt = ggml_type_size(act_slab_dtype);
        const size_t slab_half_bytes = (size_t) n_embd_io * (size_t) pp_tokens * slab_elt;
        buf_slab.reset(ggml_backend_buft_alloc_buffer(buft_cuda, 2 * slab_half_bytes));
        if (!buf_slab) {
            error_msg = "failed to allocate layer-major activation slabs (" +
                        std::to_string(2 * slab_half_bytes) + " bytes = 2 x " +
                        std::to_string(slab_half_bytes) + ")";
            return false;
        }
    }

    // telemetry
    if (tel_path_ && tel_path_[0] != '\0') {
        tel_path = tel_path_;
        tel_out = fopen(tel_path.c_str(), "w");
        if (tel_out == nullptr) {
            error_msg = "failed to open telemetry file: " + tel_path;
            return false;
        }
        // T-GPLR-56 A1: the sink exists now, so the markers buffered before it (process
        // start, the model load) become the trace's first events, ahead of run_start.
        llama_ls_tel_flush_early(tel_out);
    }

    t0_us = ggml_time_us();
    run_id = (uint64_t) t0_us;

    // T-GPLR-01: emit the adapter package catalog for host-side byte/member
    // parity against the reference implementation.
    tel_pkg_catalog();

    return true;
}

// ---------------------------------------------------------------------------
// graph build (called from the qwen35 graph builder)
// ---------------------------------------------------------------------------

llama_layer_streaming::graph_set * llama_layer_streaming::find_set(int mode, int64_t n_tokens, int32_t rs_z, int32_t n_outputs, int32_t chunk_id) {
    for (auto & set : sets) {
        const bool chunk_match = pw_probe ? true : (set.chunk_id == chunk_id);
        if (set.mode == mode && set.n_tokens == n_tokens && set.rs_z == rs_z && set.n_outputs == n_outputs && chunk_match) {
            return &set;
        }
    }
    return nullptr;
}

// T-GPLR-47: swap-in-slot rebuild. The per-pass prebuild (T-GPLR-43) used to
// re-enter begin_set on an ALREADY-BUILT set and reset it IN PLACE: res->reset()
// + graph lists cleared + the rebuild written into the SAME ggml ctx, whose
// pool accumulated every prior build's tensor objects (measured: patch_cache
// members 7 -> 14 -> 28, T-GPLR-44 dumps) and whose act bindings survived only
// by the address-stability accident that the rebuild never re-allocates. The
// replacement constructs the fresh state OUTSIDE the live set and move-assigns
// it into the SAME vector slot, so:
//   - no orphaned tensor accumulates across passes (bounded per-set pool),
//   - every stored graph_set* stays valid BY CONSTRUCTION (assignment never
//     moves vector elements), closing the T-GPLR-44 stale-pointer hazard,
//   - the patch caches start empty and are re-walked from the FRESH ctx on
//     the first prepare_chunk, so the doubled-cache path is structurally gone.
// Act bindings are REPLAYED, not re-derived: the fresh tensors are bound to
// the old set's exact device regions (invariant 2e/2f of the task contract --
// the cursors are frozen post-first-pass and a re-derivation would alias live
// regions; see begin_set's invariant block). KV view re-derivation (invariant
// 5, the T-GPLR-43 crash fix) is untouched: it happens in the model build that
// follows, against the post-clear memory.
bool llama_layer_streaming::replace_set_with_fresh_rebuild(graph_set & old, int64_t max_nodes) {
    if (old.ctx == nullptr || old.res == nullptr || !old.built) {
        error_msg = "rebuild replacement: set is not a finished build";
        return false;
    }
    // INV2e: capture the act REGIONS before anything is torn down. Replay, never
    // re-derive: the cursors are frozen at their post-first-pass values and the
    // descriptor recompute is not a binding source (see begin_set's block).
    void * act_region[2] = { old.act[0] ? old.act[0]->data : nullptr,
                             old.act[1] ? old.act[1]->data : nullptr };
    if (act_region[0] == nullptr || act_region[1] == nullptr) {
        error_msg = "rebuild replacement: set has unbound act tensors";
        return false;
    }
    if (act_region[0] == act_region[1]) {
        error_msg = "rebuild replacement: act regions alias (A == B) -- refusing";
        return false;
    }
    // the owning buffer: layer-major chunk sets bind slab slices, batch-major
    // (and TG) sets allocate from buf_act (both stable module-owned buffers)
    ggml_backend_buffer_t act_buf = (old.chunk_id >= 0) ? buf_slab.get() : buf_act.get();
    if (old.act[0]->buffer != act_buf || old.act[1]->buffer != act_buf) {
        error_msg = "rebuild replacement: act tensors are not bound to the expected buffer";
        return false;
    }

    graph_set fresh;
    fresh.mode      = old.mode;
    fresh.n_tokens  = old.n_tokens;
    fresh.rs_z      = old.rs_z;
    fresh.n_outputs = old.n_outputs;
    fresh.chunk_id  = old.chunk_id;

    // same no-alloc params as begin_set's first build (~96 MiB tensor pool)
    ggml_init_params io_params = { 96u * 1024u * 1024u, nullptr, true };
    fresh.ctx = ggml_init(io_params);
    if (fresh.ctx == nullptr) {
        error_msg = "rebuild replacement: failed to init io context";
        return false;
    }
    fresh.res = new llm_graph_result(max_nodes);
    if (fresh.res == nullptr) {
        ggml_free(fresh.ctx);
        error_msg = "rebuild replacement: failed to create result object";
        return false;
    }

    // replay the act bindings onto the captured regions (fresh tensor objects,
    // fresh ctx; ONLY the device addresses carry over -- invariant 2g)
    for (int i = 0; i < 2; ++i) {
        fresh.act[i] = ggml_new_tensor_2d(fresh.ctx,
                                          old.chunk_id >= 0 ? act_slab_dtype : GGML_TYPE_F32,
                                          n_embd_io, old.n_tokens);
        if (fresh.act[i] == nullptr) {
            ggml_free(fresh.ctx);
            delete fresh.res;
            error_msg = "rebuild replacement: failed to create act tensor";
            return false;
        }
        ggml_backend_tensor_alloc(act_buf, fresh.act[i], act_region[i]);
        if (fresh.act[i]->data != act_region[i]) {
            ggml_free(fresh.ctx);
            delete fresh.res;
            error_msg = "rebuild replacement: act replay missed its region (INV2e)";
            return false;
        }
        ggml_set_name(fresh.act[i], i == 0 ? "stream_act_0" : "stream_act_1");
    }

    // swap into the SAME slot. `old` is a reference into `sets`; assignment
    // keeps the element address, so every stored graph_set* (cur, controller
    // lookups) now addresses fresh state. The old ctx/res are released after
    // the move (they were captured in the moved-from object).
    ggml_context *     old_ctx = old.ctx;
    llm_graph_result * old_res = old.res;
    // T-GPLR-62 S2: carry the rebuild-pending flag across the move. The caller
    // still has to re-derive the graphs (that is what the move made necessary),
    // so the replacement must NOT clear it -- finish_build clears it on success
    // and a failed build leaves it set for a retry. Explicit so the semantics do
    // not depend on `fresh`'s defaults happening to match the old set's.
    const bool was_flagged = old.needs_rebuild;
    old = std::move(fresh);
    old.needs_rebuild = was_flagged;
    // INV4: bounded per-set pool -- the old state is fully released here. The
    // old ctx is a no-alloc pool (tensor objects only), so ggml_free touches
    // no device memory; the slab/act buffers are module-owned and outlive it.
    ggml_free(old_ctx);
    delete old_res;
    return true;
}

// T-GPLR-49: record a bound slice's interval inside one slab half, refusing
// loudly when it would overlap an interval another live set already holds. The
// pre-T-GPLR-49 code had no such check: two sets sharing a slab region would
// have run silently against each other's activations.
bool llama_layer_streaming::claim_slab_region(int which, size_t start, size_t bytes) {
    if (which < 0 || which > 1 || bytes == 0) {
        error_msg = "claim_slab_region: invalid claim";
        return false;
    }
    const size_t end = start + bytes;
    for (const auto & c : slab_claims[which]) {
        if (llama_ls_slab_overlaps(c.first, c.second, start, end)) {
            error_msg = "layer-major activation slab " + std::string(which ? "B" : "A") +
                        " would alias: new slice [" + std::to_string(start) + "," + std::to_string(end) +
                        ") overlaps a live slice [" + std::to_string(c.first) + "," +
                        std::to_string(c.second) + ")";
            return false;
        }
    }
    slab_claims[which].emplace_back(start, end);
    return true;
}

// allocate one non-aliasing full-chunk slice of slab half `which` (0 = A,
// 1 = B) as a [n_embd_io, n_tokens] tensor in the slab dtype from the slab
// buffer. The tensor object lives in ctx; its data is a fixed region of the
// stable slab buffer, so a persistent graph bound to it always reads/writes
// the same device addresses (SDD section 13). Fails loudly if the slab half is
// exhausted; the cursor only advances on success.
// T-GPLR-49: the slice and half SIZES come from llama_ls_slab_region_of -- the one
// owner of the element-size arithmetic -- while the cursor `off` only chooses
// where the slice lands (pw mode's per-width-class allocator). The resulting
// interval is CLAIMED, so overlapping sets refuse instead of aliasing.
static bool slab_slice(llama_layer_streaming & ls, ggml_context * ctx, int which, size_t chunk_id,
                       int64_t n_tokens, size_t & off, ggml_tensor * & out) {
    if (chunk_id >= ls.chunks.size() || ls.chunks[chunk_id].n_tokens != n_tokens) {
        ls.error_msg = "slab_slice: chunk " + std::to_string(chunk_id) + " width " +
                       std::to_string(n_tokens) + " does not match the chunk descriptor";
        return false;
    }
    llama_ls_slab_region region{};
    if (!llama_ls_slab_region_of(ls.chunks.data(), ls.chunks.size(), chunk_id, ls.n_embd_io,
                              ls.act_slab_dtype, &region)) {
        ls.error_msg = "layer-major activation slab " + std::string(which ? "B" : "A") +
                       " has no region for chunk " + std::to_string(chunk_id);
        return false;
    }
    const size_t slice_bytes = region.bytes;
    const size_t half_bytes  = region.half;
    if (off + slice_bytes > half_bytes) {
        ls.error_msg = "layer-major activation slab " + std::string(which ? "B" : "A") + " exhausted (" +
                       std::to_string(off + slice_bytes) + ">" + std::to_string(half_bytes) + ")";
        return false;
    }
    if (!ls.claim_slab_region(which, off, slice_bytes)) {
        return false;   // error_msg names both overlapping intervals
    }
    ggml_tensor * t = ggml_new_tensor_2d(ctx, ls.act_slab_dtype, ls.n_embd_io, n_tokens);
    if (t == nullptr) {
        ls.error_msg = "failed to create layer-major activation slab slice";
        return false;
    }
    const char * base = (const char *) ggml_backend_buffer_get_base(ls.buf_slab.get());
    ggml_backend_tensor_alloc(ls.buf_slab.get(), t, (char *) base + (which ? half_bytes : 0) + off);
    if (t->data == nullptr) {
        ls.error_msg = "failed to bind layer-major activation slab slice";
        return false;
    }
    off += slice_bytes;
    out = t;
    return true;
}

llama_layer_streaming::graph_set * llama_layer_streaming::begin_set(int mode, int64_t n_tokens, int32_t rs_z, int32_t n_outputs, int32_t chunk_id) {
    if (n_tokens <= 0) {
        error_msg = "begin_set: invalid n_tokens";
        return nullptr;
    }
    if (n_outputs < 0) {
        error_msg = "begin_set: invalid n_outputs";
        return nullptr;
    }
    if (chunk_id >= 0) {
        if (schedule != LLAMA_PP_SCHEDULE_LAYER_MAJOR) {
            error_msg = "begin_set: chunk_id requires the layer-major schedule";
            return nullptr;
        }
        // T-Q38CTX-01: validate against the immutable per-run descriptor
        if (chunk_id >= (int32_t) chunks.size() || n_tokens != chunks[chunk_id].n_tokens) {
            error_msg = "begin_set: layer-major chunk " + std::to_string(chunk_id) +
                        " (n_tokens " + std::to_string(n_tokens) + ") does not match the " +
                        std::to_string(chunks.size()) + "-chunk descriptor";
            return nullptr;
        }
        // T-Q38PW-00/01: in pw mode, slab slices are cursor-driven for each
        // width class (patching redirects per chunk). In per-chunk mode, each
        // chunk gets its own distinct slab slice, derived from the descriptor.
        // T-GPLR-49: the derivation is llama_ls_slab_region_of's -- it sizes with
        // the SLAB element type (the pre-GPLR-49 formula used sizeof(float),
        // exactly 2x too large for the bf16 default) and it runs for EVERY
        // chunk including chunk 0 (which used to skip the derivation and
        // inherit whatever the cursor held). In pw mode nothing moves: the
        // slices stay cursor-driven and the claim check keeps them
        // non-aliasing, so the pw path's bindings are unchanged.
        if (!pw_probe) {
            llama_ls_slab_region region{};
            if (!llama_ls_slab_region_of(chunks.data(), chunks.size(), (size_t) chunk_id, n_embd_io,
                                      act_slab_dtype, &region)) {
                error_msg = "begin_set: no slab region for chunk " + std::to_string(chunk_id) +
                            " (descriptor width " + std::to_string(chunks[chunk_id].n_tokens) +
                            ", n_embd_io " + std::to_string(n_embd_io) + ")";
                return nullptr;
            }
            slab_off_A = region.offset;
            slab_off_B = region.offset;
        }
        // In pw mode, slab_off_A/B stay at 0 (all width classes share pos 0)
    }

    const int64_t max_nodes = (int64_t) n_layer * 2048 + 4096;
    graph_set * set = find_set(mode, n_tokens, rs_z, n_outputs, chunk_id);
    // T-GPLR-47: the per-pass prebuild's re-entry into an ALREADY-BUILT set is
    // the rebuild. With the runner's explicit flag set (pass-2+ prebuild only),
    // the set is replaced by a fresh-context rebuild instead of being reset in
    // place. INV3: rebuilds happen between passes, so no build may be in flight
    // (an unfinished build under cur that is not this set is a routing error).
    if (set != nullptr && rebuild_replacement && set->built) {
        if (cur != nullptr && !cur->built && cur != set) {
            error_msg = "begin_set: refusing rebuild replacement -- a build is in flight";
            return nullptr;
        }
        if (!replace_set_with_fresh_rebuild(*set, max_nodes)) {
            return nullptr;   // error_msg set; the old set stays live
        }
        // act_bind telemetry stays silent on replacement (INV1): the binds
        // describe the slab layout, which is unchanged by the replay; the
        // checker's one-bind-per-chunk invariant must keep holding.
        cur = set;
        return set;
    }
    // T-GPLR-43 fix: act_bind telemetry describes the slab layout, which is
    // bound exactly once per set. A rebuild (per-pass prebuild) re-derives
    // the same offsets into the same ctx; re-emitting them tripped the
    // checker's one-bind-per-chunk/slab invariant (T-GPLR-43 probe artifact:
    // 564 violations). Emit only when the set is newly created.
    const bool set_created = (set == nullptr);
    if (set == nullptr) {
        // create the set: own io context + own result object + own act pair.
        // the caller must build with gparams.res == set->res
        set = &sets.emplace_back();
        set->mode      = mode;
        set->n_tokens  = n_tokens;
        set->rs_z      = rs_z;
        set->n_outputs = n_outputs;
        set->chunk_id  = chunk_id;

        // ~256 B per tensor object; worst case ~30k tensors across 66 graphs
        ggml_init_params io_params = { 96u * 1024u * 1024u, nullptr, true };
        set->ctx = ggml_init(io_params);
        if (set->ctx == nullptr) {
            error_msg = "begin_set: failed to init io context";
            return nullptr;
        }

        // per-variant result object (input holders + output references);
        // the constructor resets a fresh result object
        set->res = new llm_graph_result(max_nodes);

        if (chunk_id >= 0) {
            // layer-major: the act pair is a pair of non-aliasing slab slices
            // (SDD section 13), bound to fixed slab offsets so the persistent
            // graphs execute against stable device addresses. The offsets are
            // emitted as act_bind telemetry so a host validator can prove the
            // slices are distinct.
            const size_t off_A = slab_off_A;
            const size_t off_B = slab_off_B;
            if (!slab_slice(*this, set->ctx, 0, (size_t) chunk_id, n_tokens, slab_off_A, set->act[0])) {
                return nullptr;
            }
            if (!slab_slice(*this, set->ctx, 1, (size_t) chunk_id, n_tokens, slab_off_B, set->act[1])) {
                return nullptr;
            }
            // T-GPLR-43 unification: begin_set is the SINGLE owner of
            // act_bind telemetry (the second emitter in llama-context.cpp
            // is deleted). Emission happens only on set creation, so a
            // per-pass rebuild (the default for --pp-repeat > 1) stays
            // silent in both modes.
            //   non-pw: one set per chunk -> emit this set's two slab binds.
            //   pw: one set per width class whose static slices sit at
            //   offset 0 -> emit the per-chunk cumulative-offset walk
            //   instead (trace contract: every chunk bound once per slab
            //   at distinct non-aliasing offsets), on chunk 0's creation —
            //   always the first set the layer-major prebuild builds.
            if (set_created) {
                if (!pw_probe) {
                    tel_act_bind(chunk_id, 0, off_A, n_tokens);
                    tel_act_bind(chunk_id, 1, off_B, n_tokens);
                } else if (chunk_id == 0) {
                    const size_t slab_elt = ggml_type_size(act_slab_dtype);
                    size_t off_walk = 0;
                    for (size_t c = 0; c < chunks.size(); ++c) {
                        tel_act_bind((int32_t) c, 0, off_walk, chunks[c].n_tokens);
                        tel_act_bind((int32_t) c, 1, off_walk, chunks[c].n_tokens);
                        off_walk += (size_t) chunks[c].n_tokens * (size_t) n_embd_io * slab_elt;
                    }
                }
            }
        } else {
            set->act[0] = ggml_new_tensor_2d(set->ctx, GGML_TYPE_F32, n_embd_io, n_tokens);
            set->act[1] = ggml_new_tensor_2d(set->ctx, GGML_TYPE_F32, n_embd_io, n_tokens);
            if (set->act[0] == nullptr || set->act[1] == nullptr) {
                error_msg = "begin_set: failed to create activation tensors";
                return nullptr;
            }
            std::vector<ggml_tensor *> act_tensors = { set->act[0], set->act[1] };
            if (!alloc_tensors(buf_act.get(), ggml_backend_cuda_buffer_type(device), act_tensors, buf_act_off)) {
                error_msg = "begin_set: activation buffer exhausted";
                return nullptr;
            }
        }
        ggml_set_name(set->act[0], "stream_act_0");
        ggml_set_name(set->act[1], "stream_act_1");
    }

    // in-place reset path: FIRST builds and re-entries WITHOUT the replacement
    // flag (handoff-invalidated TG sets, a rebuild that refused). A pass-2+
    // prebuild re-entry with the T-GPLR-47 flag never reaches here (it swapped
    // above), so on that path the existing ctx no longer accumulates orphans.
    // the in-flight build writes into gparams.res, which must be set->res (see
    // begin_build)
    set->res->reset();
    set->graphs.clear();
    set->graph_layer.clear();
    set->graph_entries.clear();
    set->routes.clear(); // T-GPLR-02: re-captured during this build
    // T-GPLR-13: two-part row-set TG sets re-capture these on every build
    set->ff_graphs.clear();
    set->ff_layer.clear();
    set->ff_input.clear();
    set->ff_input_wpos.clear(); // T-GPLR-62 S2: keep the two vectors the same length
    set->ff_input_hsel.clear(); // T-GPLR-82 S2: same, one entry per FFN graph
    set->ff_host_experts.clear(); // T-GPLR-61 step C: host-leg row outputs
    set->built = false;

    // T-GPLR-44 (measured, now fixed at the root by T-GPLR-47): the warning
    // below described the in-place rebuild. The per-pass prebuild re-entered
    // begin_set on a ctx that ACCUMULATED (the comment above), so clearing the
    // patch caches here made the next prepare_chunk re-walk the accumulated ctx
    // and double the cache (7 -> 14 members in the dumps); the scoped A/B arms
    // diverged on generated-id identity and the change was reverted, not tuned.
    // Since T-GPLR-47 the rebuild no longer re-enters this path at all: a
    // re-entered BUILT set is replaced wholesale by replace_set_with_fresh_rebuild
    // (fresh ctx, empty caches, same slot), so the accumulated-ctx hazard this
    // warning guarded against is structurally gone and the cache is built
    // exactly once per set lifetime by the ensure_patch_cache owner.
    // T-GPLR-45 keeps the discipline this comment described: the patch cache
    // (the consumption-classified member list) is built exactly once per set by
    // one owner (the ensure_patch_cache lambda in layer_major_pp), and it is
    // built from a FINISHED graph list; nothing here or downstream ever re-walks
    // a live set. The zero-consumer filter therefore never sees an accumulated
    // ctx, and the excluded list stays stable across the per-pass rebuild (the
    // 200k dump's three firings named the same sole orphan each time).

    cur = set;
    return set;
}

ggml_context * llama_layer_streaming::begin_build(int64_t n_tokens, int32_t rs_z, int32_t n_outputs) {
    // chunk id comes from cur_chunk_id (set by the prebuild/executor before
    // model.build_graph), so the per-chunk layer-major set is selected.
    graph_set * set = begin_set(phase, n_tokens, rs_z, n_outputs, cur_chunk_id);
    if (set == nullptr) {
        return nullptr;
    }
    return set->ctx;
}

ggml_cgraph * llama_layer_streaming::new_graph(int layer) {
    if (cur == nullptr || cur->ctx == nullptr) {
        error_msg = "new_graph: begin_build not called";
        return nullptr;
    }
    ggml_cgraph * g = ggml_new_graph_custom(cur->ctx, 2048, false);
    if (g == nullptr) {
        error_msg = "new_graph: failed to create graph";
        return nullptr;
    }
    cur->graphs.push_back(g);
    cur->graph_layer.push_back(layer);

    // persistent graph-entry key: (execution_mode, logical_layer, slot)
    graph_entry e;
    e.mode          = cur->mode;
    e.logical_layer = layer;
    e.slot          = layer >= 0 ? layer % n_slots : -1;
    cur->graph_entries.push_back(e);

    return g;
}

// T-GPLR-71: see the header. Recorded unconditionally, then cleared, so an
// un-instrumented build records "unknown" (-1) instead of inheriting the
// previous build's width. view_matches_live reads -1 as unproven -> rebuild.
void llama_layer_streaming::record_built_view(graph_set & set) {
    set.n_kv = cur_build_n_kv;
    cur_build_n_kv = -1;
}

bool llama_layer_streaming::finish_build() {
    if (cur == nullptr || cur->ctx == nullptr) {
        error_msg = "finish_build: begin_build not called";
        return false;
    }

    graph_set & set = *cur;

    // allocate every not-yet-allocated INPUT-flagged tensor from the persistent
    // host buffer (at this set's cursor) so the scheduler copies them H2D per
    // split, exactly like the normal path.
    {
        std::vector<ggml_tensor *> input_tensors;
        for (ggml_tensor * t = ggml_get_first_tensor(set.ctx); t != nullptr; t = ggml_get_next_tensor(set.ctx, t)) {
            if ((t->flags & GGML_TENSOR_FLAG_INPUT) && t->buffer == nullptr && t->data == nullptr) {
                input_tensors.push_back(t);
            }
        }
        // budgeted persistent buffer first; anything that no longer fits
        // (long-context masks exceed the init() sizing model) spills to an
        // extra host buffer instead of failing the build. Host side only --
        // no VRAM, and a shape that fits never creates a spill.
        if (!alloc_tensors(buf_inputs.get(), ggml_backend_cpu_buffer_type(), input_tensors, buf_inputs_off)) {
            if (!alloc_spill(input_tensors)) {
                error_msg = "finish_build: input buffer exhausted and spill allocation failed";
                return false;
            }
        }
    }

    // sanity: graphs must be [embed, layer 0..n-1, output]
    if ((int) set.graphs.size() != n_layer + 2) {
        error_msg = "finish_build: expected " + std::to_string(n_layer + 2) +
                    " graphs, got " + std::to_string(set.graphs.size());
        return false;
    }
    if (set.graph_entries.size() != set.graphs.size()) {
        error_msg = "finish_build: graph entry count mismatch";
        return false;
    }

    // T-GPLR-71: record the attention view width these graphs were built
    // against, BEFORE clearing needs_rebuild -- it is the fact the executor's
    // coverage test compares the next live width against. A failed build never
    // reaches here, so the flag and the width are cleared together.
    //
    // Recorded UNCONDITIONALLY, then the pending value is cleared. Clearing is
    // load-bearing: only two build sites set cur_build_n_kv, so without the
    // reset an un-instrumented build (reserve / speculative opt / non-default
    // graph type) would inherit the PREVIOUS build's width and stamp a
    // plausible-looking but unverified number onto a live set -- strictly worse
    // than recording "unknown". With the reset such a site records -1, which
    // view_matches_live now reads as unproven -> rebuild.
    record_built_view(set);
    set.built = true;
    // T-GPLR-62 S2: the graphs now bind the CURRENT bundle (the compact one
    // when a narrowing happened), so the rebuild-pending flag is satisfied.
    // Cleared here, at successful build completion, and nowhere earlier, so a
    // build that fails leaves the set flagged and the next decode retries.
    set.needs_rebuild = false;
    build_generation++;
    return true;
}

// ---------------------------------------------------------------------------
// execution
// ---------------------------------------------------------------------------

bool llama_layer_streaming::load_layer(int il, bool & copied) {
    copied = false;

    if (il < 0 || il >= n_layer) {
        error_msg = "load_layer: logical layer " + std::to_string(il) + " out of range [0, " + std::to_string(n_layer) + ")";
        return false;
    }

    // T-GPLR-25: with the async-feed worker active, copies for LATER layers
    // are submitted on the worker; the main thread only stages a layer when
    // the worker has not (worker-off path) or when the layer is already
    // resident in its slot (hit path). Guard slot bookkeeping against the
    // worker with the shared mutex.
    std::unique_lock<std::mutex> af_lk;
    if (af) {
        af_lk = std::unique_lock<std::mutex>(af->mtx);
    }
    struct af_lk_release {
        std::unique_lock<std::mutex> & lk;
        llama_layer_streaming * owner;
        ~af_lk_release() { if (owner->af && lk.owns_lock()) { lk.unlock(); } }
    } af_guard{af_lk, this};

    llama_layer_streaming_slot & slot = slots[il % n_slots];

    // T-GPLR-16: partial handoff — identical no-copy handling to the full
    // handoff branch below. Migrating banks (il >= partial_from) are
    // resident in the handoff buffer; below-split layers execute from their
    // host-mmap CPU tensors (slot_layer routes them there). In both cases no
    // H2D copy happens and the released slot machinery must not be touched
    // (no consumer wait, no eviction — the T-GPLR-10 phantom-eviction rule).
    if (handoff_active && handoff_partial) {
        if (slot.logical_layer == il && slot.generation > 0) {
            return true;
        }
        slot.t_copy_submit_us = ggml_time_us();
        ls_stage_record(slot.ev_copy_ready);
        slot.t_copy_enqueue_done_us = ggml_time_us();
        slot.generation++;
        slot.logical_layer = il;
        copied = true;
        tel_copy_submit(il, 0, slot.generation);
        return true;
    }

    // Handoff path: if this layer is in the handoff buffer (all layers
    // for full handoff, only attention layers for constrained), no H2D
    // copy needed. Recurrent layers in constrained mode also skip copies
    // — slot_layer() routes them to CPU-resident host_layers.
    if (handoff_active && handoff_layers.count(il)) {
        // Handoff TG performs no H2D copy and overwrites no slot data, so the
        // streaming slot-reuse protocol (consumer wait + eviction) does not
        // apply here: nothing resident is being discarded. Counting an
        // eviction per layer per token emitted phantom slot_evict events and
        // inflated phase_summary evictions (640 = 40 layers x 16 tokens x 1
        // slot on the MoE, 1024 on dense) with zero actual data movement.
        // The executor still re-arms consumer_pending; it is simply never
        // consumed while handoff is active.
        // If the slot already holds this layer, nothing to do
        if (slot.logical_layer == il && slot.generation > 0) {
            return true;
        }
        // Handoff: record a zero-cost copy event for telemetry consistency
        slot.t_copy_submit_us = ggml_time_us();
        ls_stage_record(slot.ev_copy_ready);
        slot.t_copy_enqueue_done_us = ggml_time_us();
        slot.generation++;
        slot.logical_layer = il;
        copied = true;
        tel_copy_submit(il, 0, slot.generation);
        return true;
    }

    // T-Q38HG-04: attention-only handoff — recurrent layers route to
    // CPU-resident host_layers via slot_layer(), so no H2D copy needed.
    if (handoff_active && handoff_attention_only && !handoff_layers.count(il)) {
        // Same as above: handoff load_layer copies nothing, so there is no
        // slot reuse to protect and no eviction to count (the recurrent
        // layers' authoritative data is the CPU-resident host_layers copy).
        if (slot.logical_layer == il && slot.generation > 0) {
            return true;
        }
        slot.t_copy_submit_us = ggml_time_us();
        ls_stage_record(slot.ev_copy_ready);
        slot.t_copy_enqueue_done_us = ggml_time_us();
        slot.generation++;
        slot.logical_layer = il;
        copied = true;
        tel_copy_submit(il, 0, slot.generation);
        return true;
    }

    // slot reuse: never overwrite while a consumer is outstanding
    // T-GPLR-25: the consumer-done EVENT wait can span a full layer compute;
    // it must not hold af->mtx (the worker would starve). Release before the
    // wait and reacquire for the bookkeeping.
    if (slot.consumer_pending) {
        const ggml_backend_event_t ev_cd = (ggml_backend_event_t) slot.ev_consumer_done;
        if (af && af_lk.owns_lock()) {
            af_lk.unlock();
        }
        ls_stage_wait(ev_cd);
        if (af && !af_lk.owns_lock()) {
            af_lk.lock();
        }
        slot.consumer_pending = false;
        total_evictions++;
        // evict the PRIOR resident layer (SDD: prior logical layer, generation)
        tel_slot_evict(slot.logical_layer, il % n_slots, slot.generation);
    }

    // fast path: the slot already holds this logical layer (same generation)
    if (slot.logical_layer == il && slot.generation > 0) {
        return true;
    }

    llama_layer & dst = slot.layers.at(il);
    const llama_ls_package & pkg = pkg_cache.at(il);

    // T-GPLR-02: the descriptor must still match the slot bundle built at
    // init; a divergence means a stale/partial package and must fail before
    // any copy submission, never silently run another layer's data.
    if ((int) bundle_members.size() <= il || bundle_members[il] != pkg.members.size()) {
        error_msg = "load_layer: package descriptor/bundle member mismatch for logical layer " +
                    std::to_string(il);
        return false;
    }

    const size_t nbytes = slot.layer_bytes.at(il);

    if (nbytes == 0) {
        error_msg = "load_layer: empty package for logical layer " + std::to_string(il);
        return false;
    }

    // issue the async H2D copies on the copy stream (does not block)
    // T-GPLR-32: the main thread's copy is NOT gated (the gate serializes the
    // worker's DMAs only, per the task contract). It still records over the
    // gate event so the next worker copy is ordered after it — the main copy
    // and the next worker copy are concurrent by design (2 slots), but the
    // main copy is enqueued while the previous worker copy computes, and the
    // re-record keeps the worker's one-in-flight invariant well-defined.
    ls_stage_record(slot.ev_af_gate);
    ls_stage_record(slot.ev_copy_start);
    slot.t_copy_submit_us = ggml_time_us();

    // the copy events carry the generation that will be assigned below, so
    // telemetry consumers can match copy_complete/layer_compute to a copy
    const uint32_t gen = slot.generation + 1;

    // T-GPLR-30: persistent pool mirror for this bank (see the af-worker
    // stage site for the protocol). Valid mirror = members copy from the
    // pool; lru miss = the loop fills the entry while staging.
    // T-GPLR-33: the member loop + span-batched submission live in the
    // shared stage_members helper (byte-identical to this site's old loop).
    // T-GPLR-37: direct mmap-sourced staging — page-lock this package's mapped
    // source span so the member DMAs read the model's mapping directly, with no
    // CPU hop. Failure aborts this load loudly (error_msg carries the driver
    // error), never a silent fall back to the pinned path.
    if (copy_src_registered && !ls_register_pkg_source(il, slot, pkg)) {
        return false;
    }

    size_t pool_bytes = 0;
    bool pool_valid = false;
    char * pool_base = pp_pool_acquire(il, pool_bytes, pool_valid);

    char * staging = slot.buf_staging ? (char *) ggml_backend_buffer_get_base(slot.buf_staging.get()) : nullptr;
    size_t staging_off = 0;
    int sb_path_mt = 0; // T-GPLR-33: the taken submission path (0/1/2)
    const bool staged_ok = stage_members(backend_copy, pkg, dst, il, gen, slot,
                                         pool_base, pool_bytes, pool_valid, staging, &staging_off,
                                         &sb_path_mt);
    pp_pool_release(il, pool_base, staging_off, staged_ok);
    if (!staged_ok) {
        error_msg = "load_layer: package descriptor/bundle member mismatch for logical layer " +
                    std::to_string(il);
        return false;
    }

    ls_stage_record(slot.ev_copy_end);
    ls_stage_record(slot.ev_copy_ready);
    slot.t_copy_enqueue_done_us = ggml_time_us();

    // T-GPLR-08: while the GPU consumes layer il, the worker warms layer
    // il + ahead's mapped pages into the working set so this slot's next
    // eviction memcpy faults resident pages. Hint-only; see pf_enqueue_from.
    pf_enqueue_from(il);

    // the slot now belongs to this logical layer
    slot.generation++;
    slot.logical_layer = il;
    // T-GPLR-25 residue fix: the staging thread CLAIMS the slot for its
    // generation (free_slot=0) regardless of who staged — the old code only
    // cleared it on the worker path, so after a main-thread copy the stale
    // free_slot=1 let the worker re-stage over an UNCONSUMED resident
    // generation (the layer-major seed enqueue hits exactly this: preload
    // stages layers 0..n_slots-1, the seed targets slot 0 before layer 0's
    // graph has run). record_consumer_done re-opens the slot after the
    // consumer finishes; until then every worker request for it is refused.
    if (af) {
        af->free_slot[il % n_slots] = 0;
    }

    total_bytes_transferred += nbytes;
    copied = true;

    tel_copy_submit(il, nbytes, gen, 0, sb_path_mt);

    return true;
}

bool llama_layer_streaming::wait_slot_ready(int il, double & wait_us, double & host_submit_us, double & ready_latency_us) {
    wait_us           = 0.0;
    host_submit_us    = 0.0;
    ready_latency_us  = 0.0;

    // T-GPLR-25: when the async-feed worker is active it may hold af->mtx
    // while mutating slot bookkeeping (it stages other layers into other
    // slots, but the shared mutex serializes the writers). The copy EVENT
    // (ev_copy_ready) remains the data-ordering guarantee; this lock only
    // makes the bookkeeping read consistent.
    std::unique_lock<std::mutex> af_lk;
    if (af) {
        af_lk = std::unique_lock<std::mutex>(af->mtx);
    }

    llama_layer_streaming_slot & slot = slots[il % n_slots];

    if (slot.logical_layer != il) {
        error_msg = "wait_slot_ready: slot " + std::to_string(il % n_slots) +
                    " holds logical layer " + std::to_string(slot.logical_layer) +
                    ", expected " + std::to_string(il);
        return false;
    }

    // if no copy was issued for this layer, the slot was already resident
    if (slot.t_copy_submit_us == 0) {
        // async-feed bookkeeping: the copy was submitted by the worker, so
        // t_copy_submit_us was reset by an earlier wait; nothing to drain.
        // The af_submitted flag was consumed by that wait too.
        slot.af_submitted = false;
        if (af) {
            af_lk.unlock();
        }
        return true;
    }

    // host-block until the copy that made this slot current is complete.
    // measures the availability wait and guarantees ordering.
    // T-GPLR-25: the event wait must NOT hold af->mtx (the worker needs it to
    // finish/record its own copies); the values read below are stable once
    // the event has fired for this generation because the worker never
    // mutates a slot whose consumer has not yet run record_consumer_done.
    const int64_t t_wait0 = ggml_time_us();
    const ggml_backend_event_t ev = (ggml_backend_event_t) slot.ev_copy_ready;
    const uint32_t gen = slot.generation;
    const int64_t t_submit_us = slot.t_copy_submit_us;
    const int64_t t_enq_us    = slot.t_copy_enqueue_done_us;
    if (af) {
        af_lk.unlock();
    }
    ggml_backend_event_synchronize(ev);
    const int64_t t_wait1 = ggml_time_us();

    wait_us          = (double) (t_wait1 - t_wait0);
    host_submit_us   = (double) (t_enq_us - t_submit_us);
    ready_latency_us = (double) (t_wait1 - t_enq_us);
    const double transfer_us = host_submit_us + ready_latency_us;
    if (af) {
        af_lk.lock();
    }
    const bool af_fed = slot.af_submitted; // captured before the flags clear
    slot.t_copy_submit_us       = 0;
    slot.t_copy_enqueue_done_us = 0;
    slot.af_submitted           = false;
    // T-GPLR-32: one worker DMA just completed (its copy_complete fires
    // below). Only decrement when the gate is active so the telemetry field
    // stays 0 (gate-off) for every legacy trace.
    if (af_fed && pp_af_serialize) {
        af_dma_inflight.fetch_sub(1, std::memory_order_relaxed);
    }
    (void) af_dma_inflight;
    if (af) {
        // the consumed slot stays CLAIMED until record_consumer_done; nothing
        // to free here. Record the overlap ONLY for worker-fed copies: the
        // main-thread lookahead's copies are reported by copy_submit and
        // copy_complete, and a consumption of a main-thread copy must not
        // emit async_feed_ready (that false positive made the worker look
        // active while it had never submitted anything).
        if (af_fed) {
            tel_async_feed_ready(il, il % n_slots, gen, wait_us < 100.0 ? ready_latency_us : 0.0);
        }
    }
    if (af) {
        af_lk.unlock();
    }

    tel_copy_complete(il, transfer_us, host_submit_us, ready_latency_us, gen);
    tel_layer_ready_wait(il, wait_us, wait_us < 100.0, gen);

    return true;
}

void llama_layer_streaming::record_consumer_done(int il, ggml_backend_t backend_compute) {
    // T-GPLR-25: releasing a slot under the shared mutex both serializes the
    // bookkeeping and gives the async-feed worker its fill window: the slot
    // becomes free here, and the controller enqueues the next layer right
    // after this call.
    std::unique_lock<std::mutex> af_lk;
    if (af) {
        af_lk = std::unique_lock<std::mutex>(af->mtx);
    }
    llama_layer_streaming_slot & slot = slots[il % n_slots];
    ggml_backend_event_record((ggml_backend_event_t) slot.ev_consumer_done, backend_compute);
    slot.consumer_pending = true;
    if (af) {
        af->free_slot[il % n_slots] = 1;
    }
}

// ---------------------------------------------------------------------------
// executor (llama_context member, defined here to keep llama-context.cpp
// default-path code untouched when LLAMA_LAYER_STREAMING is off)
// ---------------------------------------------------------------------------

static bool backend_is_cuda(ggml_backend_t backend) {
    return ggml_backend_is_cuda(backend);
}

// T-GPLR-75: which backend actually executed a graph's nodes. The row-set
// executor dispatches EVERY graph through ggml_backend_sched like the normal
// path, so a single per-layer `dev` cannot describe a two-part layer: part A
// (attention + residual + router) and part B (FFN + shared experts on the
// staged rows) can land on different devices. Classify each graph on its own:
//   0 = every node on a HOST buffer, 1 = every node on a device buffer,
//   2 = MIXED (the honest answer when a graph straddles), -1 = not classifiable.
// This is the measurement the K=2 arms lacked: they reported dev = -1 for all
// 5,120 row-set TG events, so "does --residency-tg use CPU inference" had no
// answer from the trace.
// T-GPLR-78: ops that allocate a new tensor descriptor but run NO kernel.
// Verified in this tree, ggml-cpu/ggml-cpu.c ggml_compute_forward(): RESHAPE,
// PERMUTE, VIEW and TRANSPOSE are each an explicit `// nop` case. They still
// appear in g->nodes with a real buffer, so before this filter a graph whose
// only host-side node was a VIEW classified as 2 (MIXED) -- which is how
// T-GPLR-75 reported 75% of attention "mixed" when ~100% of it was on the
// GPU (T-GPLR-76 census: the ONLY host-side op was VIEW, 60 of 2,420).
//
// GGML_OP_CONT is deliberately NOT here: it dispatches ggml_compute_forward_cont
// with n_tasks = n_threads, i.e. it is a real copy, and a host-side CONT means
// an actual device->host transfer. Dropping it would hide real host work.
static bool ls_op_is_metadata_only(ggml_op op) {
    switch (op) {
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NONE:
            return true;
        default:
            return false;
    }
}

static int ls_classify_graph_backend(const ggml_cgraph * g) {
    if (g == nullptr || g->n_nodes == 0) {
        return -1;
    }
    int n_host = 0;
    int n_dev  = 0;
    for (int k = 0; k < g->n_nodes; ++k) {
        const ggml_tensor * t = g->nodes[k];
        if (t == nullptr || t->buffer == nullptr) {
            continue;
        }
        if (ls_op_is_metadata_only(t->op)) {
            continue;
        }
        if (ggml_backend_buft_is_host(ggml_backend_buffer_get_type(t->buffer))) {
            ++n_host;
        } else {
            ++n_dev;
        }
    }
    if (n_host == 0 && n_dev == 0) {
        return -1;
    }
    if (n_host == 0) {
        return 1;
    }
    if (n_dev == 0) {
        return 0;
    }
    return 2;
}

// T-GPLR-76: WHICH ops sat on which side. The classifier above answers "did
// this graph straddle"; this answers "which ops", which is the part an owner
// can act on ("attention is on the CPU because MUL/SOFT_MAX have no CUDA
// kernel in this build" is a different conversation from "it is mixed").
// Tensors with no buffer are SKIPPED, exactly as in the classifier, so an
// unallocated node is never silently reported as device-side.
//
// T-GPLR-78: this census is deliberately NOT filtered by ls_op_is_metadata_only.
// The classifier answers "where did the work run"; the census answers "which
// ops are in the graph and where does each node's buffer live", and that
// question is only useful if the no-kernel VIEW nodes stay visible -- they are
// exactly what exposed the T-GPLR-75 misreading. A graph filtered here would
// have hidden the evidence that motivated the fix.
static void ls_ops_census(const ggml_cgraph * g,
                          std::vector<std::pair<std::string, int> > & host,
                          std::vector<std::pair<std::string, int> > & dev,
                          std::vector<std::pair<std::string, int> > & host_names) {
    if (g == nullptr) {
        return;
    }
    for (int k = 0; k < g->n_nodes; ++k) {
        const ggml_tensor * t = g->nodes[k];
        if (t == nullptr || t->buffer == nullptr) {
            continue;
        }
        const bool is_host = ggml_backend_buft_is_host(ggml_backend_buffer_get_type(t->buffer));
        // T-GPLR-79: name the host-side nodes. Done BEFORE the op lookup so a
        // host node is always attributable even if ggml_op_name is null, and
        // names are truncated so a long CB name cannot bloat the event.
        if (is_host) {
            // T-GPLR-79b: ggml names a view "%s (view)" from its source, so a
            // view of an unnamed source prints as "(view)" and the op census
            // cannot say WHICH memory it touches. Walk view_src to the memory
            // root and key on that instead: op, shape, type and hop count are
            // what actually identify a state tensor or a KV view.
            const ggml_tensor * root = t;
            int hops = 0;
            while (root->view_src != nullptr && hops < 16) {
                root = root->view_src;
                ++hops;
            }
            const char * rnm = ggml_get_name(root);
            std::ostringstream ks;
            ks << ggml_op_name(root->op) << " ne=[";
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                ks << (i ? "," : "") << root->ne[i];
            }
            ks << "] type=" << ggml_type_name(root->type)
               << " hops=" << hops
               << " rootbuf=" << (root->buffer == nullptr ? "none" :
                     (ggml_backend_buft_is_host(ggml_backend_buffer_get_type(root->buffer)) ? "host" : "dev"))
               << " rname='" << (rnm ? rnm : "") << "'"
               << " dst_off=" << ggml_nbytes(t);
            std::string tn = ks.str();
            if (tn.size() > 120) {
                tn = tn.substr(0, 120);
            }
            bool hf = false;
            for (size_t i = 0; i < host_names.size(); ++i) {
                if (host_names[i].first == tn) {
                    ++host_names[i].second;
                    hf = true;
                    break;
                }
            }
            if (!hf) {
                host_names.push_back(std::make_pair(tn, 1));
            }
        }
        std::vector<std::pair<std::string, int> > & into = is_host ? host : dev;
        const char * nm = ggml_op_name(t->op);
        const std::string name = nm ? nm : "?";
        bool found = false;
        for (size_t i = 0; i < into.size(); ++i) {
            if (into[i].first == name) {
                ++into[i].second;
                found = true;
                break;
            }
        }
        if (!found) {
            into.push_back(std::make_pair(name, 1));
        }
    }
}

void llama_layer_streaming::accum_graph_ops(const ggml_cgraph * g, bool part_a) {
    if (rowset_ops_emitted) {
        return;   // the first round is representative; do not re-census
    }
    ls_ops_census(g, part_a ? rowset_host_ops_a : rowset_host_ops_b,
                      part_a ? rowset_dev_ops_a  : rowset_dev_ops_b,
                      part_a ? rowset_host_names_a : rowset_host_names_b);
}

static std::string ls_ops_json(const std::vector<std::pair<std::string, int> > & v) {
    std::string s = "{";
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i ? "," : "");
        s += "\"" + v[i].first + "\":" + std::to_string(v[i].second);
    }
    return s + "}";
}

void llama_layer_streaming::tel_graph_backend_ops() {
    if (rowset_ops_emitted) {
        return;
    }
    rowset_ops_emitted = true;
    tel_write(tel_out,
        "{\"event\":\"graph_backend_ops\",\"run_id\":%llu,"
        "\"part_a_host\":%s,\"part_a_device\":%s,\"part_b_host\":%s,\"part_b_device\":%s,"
        "\"part_a_host_names\":%s,\"part_b_host_names\":%s,"
        "\"t_us\":%lld}",
        (unsigned long long) run_id,
        ls_ops_json(rowset_host_ops_a).c_str(), ls_ops_json(rowset_dev_ops_a).c_str(),
        ls_ops_json(rowset_host_ops_b).c_str(), ls_ops_json(rowset_dev_ops_b).c_str(),
        ls_ops_json(rowset_host_names_a).c_str(), ls_ops_json(rowset_host_names_b).c_str(),
        (long long) ggml_time_us());
}

bool llama_layer_streaming_prebuild(llama_context * ctx, const llama_batch & batch, int64_t tg_tokens, ggml_status & ret) {
    return ctx->layer_streaming_prebuild(batch, tg_tokens, ret);
}

int llama_layer_major_pp(llama_context * ctx, const llama_batch & batch, ggml_status & ret) {
    return ctx->layer_major_pp(batch, ret);
}

int64_t llama_layer_major_pp_first_token(llama_context * ctx) {
    return ctx->layer_major_pp_first_token();
}

// T-GPLR-03: argmax of the last output row of the PP logits. The layer-major
// controller exposes the last chunk's result tensors on gf_res_prev but never
// fills the context's host logits buffer (it bypasses decode()), so read the
// device/host result tensor directly. This is the quantity a normal-path
// greedy sampler generates first: P(.|prompt) at the last prompt position.
int64_t llama_context::layer_major_pp_first_token() {
    if (gf_res_prev == nullptr) {
        return -1;
    }
    ggml_tensor * t = gf_res_prev->get_logits();
    if (t == nullptr || t->data == nullptr || t->ne[0] <= 0 || t->ne[1] <= 0) {
        return -1;
    }
    ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t);
    if (backend_res == nullptr) {
        return -1;
    }
    const int64_t n_vocab = t->ne[0];
    const int64_t n_out   = t->ne[1];
    std::vector<float> row((size_t) n_vocab);
    ggml_backend_tensor_get(t, row.data(), (size_t) (n_out - 1) * t->nb[1], (size_t) n_vocab * sizeof(float));
    int64_t best = 0;
    for (int64_t v = 1; v < n_vocab; ++v) {
        if (row[(size_t) v] > row[(size_t) best]) {
            best = v;
        }
    }
    return best;
}

// One owner for the structured failure shape (error_msg -> failed -> error
// event -> GGML_STATUS_FAILED); byte-identical to the inline copies it
// replaces -- traces and checkers parse these exact strings.
static ggml_status ls_shape_failure(llama_layer_streaming * ls, std::string msg) {
    ls->error_msg = std::move(msg);
    ls->failed = true;
    ls->tel_error(ls->error_msg.c_str());
    return GGML_STATUS_FAILED;
}

// T-GPLR-61 step C: the host-leg non-finite check, extracted from
// layer_streaming_compute unchanged. The host leg's PRE-AGGREGATION
// per-row outputs were stashed at graph-build time (ff_host_experts); the
// aggregate cannot attribute a bad value to a row, so the scan is per row
// and the expert ID is read back from the hsel input that selected it.
// Flag absent -> no stash, no read, no sync: AC6 inertness.
// A persistent graph keeps the node addresses of its first allocation (ggml-alloc treats data != NULL as external),
// but the scheduler plans its CPU<->device copy tensors fresh on every run and can place them over those frozen nodes.
// Clearing the computed nodes lets every run plan the whole graph again
static void ls_reset_graph_allocs(ggml_cgraph * g) {
    for (int i = 0; i < g->n_nodes; ++i) {
        ggml_tensor * t = g->nodes[i];
        if (t->view_src == nullptr && t->op != GGML_OP_NONE && !(t->flags & GGML_TENSOR_FLAG_INPUT)) {
            t->data   = nullptr;
            t->buffer = nullptr;
        }
    }
    for (int i = 0; i < g->n_nodes; ++i) {
        ggml_tensor * t = g->nodes[i];
        if (t->view_src != nullptr && t->view_src->data == nullptr) {
            t->data   = nullptr;
            t->buffer = nullptr;
        }
    }
}

// first non-finite element of a named F32 tensor in the layer's FFN graph; empty when all finite or the tensor is absent
static std::string ls_first_nonfinite(ggml_cgraph * g, const char * name) {
    ggml_tensor * t = g != nullptr ? ggml_graph_get_tensor(g, name) : nullptr;
    if (t == nullptr || t->buffer == nullptr || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        return std::string();
    }
    std::vector<float> buf((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    for (size_t i = 0; i < buf.size(); ++i) {
        if (!std::isfinite(buf[i])) {
            return std::string(name) + " col " + std::to_string(i % (size_t) t->ne[0]) + " token " + std::to_string(i / (size_t) t->ne[0]);
        }
    }
    return std::string();
}

// diagnostic (LLAMA_LS_HOST_SHADOW): compare the CPU leg with a device recompute of the same rows
static void ls_report_host_shadow(llama_layer_streaming * ls, ggml_cgraph * g, int il) {
    ggml_tensor * h = ggml_graph_get_tensor(g, "split_host_out");
    ggml_tensor * s = ggml_graph_get_tensor(g, "split_shadow_out");
    if (h == nullptr || s == nullptr || h->type != GGML_TYPE_F32 || s->type != GGML_TYPE_F32 ||
        ggml_nelements(h) != ggml_nelements(s) || !ggml_is_contiguous(h) || !ggml_is_contiguous(s)) {
        return;
    }
    std::vector<float> a((size_t) ggml_nelements(h)), b((size_t) ggml_nelements(s));
    ggml_backend_tensor_get(h, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_get(s, b.data(), 0, b.size() * sizeof(float));
    double dmax = 0.0, rmax = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dmax = std::max(dmax, (double) std::fabs(a[i] - b[i]));
        rmax = std::max(rmax, (double) std::fabs(b[i]));
        dot += (double) a[i] * b[i];
        na  += (double) a[i] * a[i];
        nb  += (double) b[i] * b[i];
    }
    const double rel  = rmax > 0.0 ? dmax / rmax : dmax;
    const double cosv = (na > 0.0 && nb > 0.0) ? dot / std::sqrt(na * nb) : 1.0;
    static int64_t cur_ub = -1;
    static double  ub_rel = 0.0, ub_cos = 1.0;
    static int     ub_layer = -1, n_detail = 0;
    const int64_t ub = (int64_t) ls->n_ubatches;
    if (ub != cur_ub) {
        cur_ub = ub; ub_rel = 0.0; ub_cos = 1.0; ub_layer = -1;
    }
    if (rel > ub_rel) {
        ub_rel = rel; ub_layer = il;
    }
    ub_cos = std::min(ub_cos, cosv);
    if ((rel > 0.05 || ub <= 4) && n_detail < 320 && (il % 8 == 0)) {
        n_detail++;
        LLAMA_LOG_WARN("host-shadow detail ubatch %lld layer %d rel %.3e cos %.6f |host| %.3e |dev| %.3e\n",
                       (long long) ub, il, rel, cosv, std::sqrt(na), std::sqrt(nb));
    }
    if (il % 8 == 0 && ub <= 5) {
        auto rd = [&](const char * nm, std::vector<float> & v) {
            ggml_tensor * t = ggml_graph_get_tensor(g, nm);
            if (t == nullptr || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) { return false; }
            v.resize((size_t) ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
            return true;
        };
        std::vector<float> rf, dv;
        if (rd("split_ref_out", rf) && rd("split_dev_out", dv) && rf.size() == dv.size() && rf.size() == a.size()) {
            auto cosf = [&](const std::vector<float> & x, const std::vector<float> & y) {
                double d = 0, nx = 0, ny = 0;
                for (size_t i = 0; i < x.size(); ++i) { d += (double) x[i] * y[i]; nx += (double) x[i] * x[i]; ny += (double) y[i] * y[i]; }
                return (nx > 0 && ny > 0) ? d / std::sqrt(nx * ny) : 0.0;
            };
            std::vector<float> sh(a.size()), sd(a.size());
            double nref = 0;
            for (size_t i = 0; i < a.size(); ++i) { sh[i] = dv[i] + a[i]; sd[i] = dv[i] + b[i]; nref += (double) rf[i] * rf[i]; }
            LLAMA_LOG_WARN("ref-check ubatch %lld layer %d |ref| %.4e cos(ref,dev+host) %.6f cos(ref,dev+shadow) %.6f cos(ref,dev) %.6f\n", (long long) ub, il, std::sqrt(nref), cosf(rf, sh), cosf(rf, sd), cosf(rf, dv));
        }
    }
    if (il == ls->n_layer - 1) {
        LLAMA_LOG_WARN("host-shadow ubatch %lld max_rel %.3e at layer %d min_cos %.6f\n", (long long) ub, ub_rel, ub_layer, ub_cos);
    }
}

static ggml_status scan_host_expert_outputs(llama_layer_streaming * ls, llama_layer_streaming::graph_set * set,
                                            int il, int64_t n_tok, int & host_finite) {
    ggml_tensor * hex = nullptr;
    if ((size_t) il < set->ff_host_experts.size()) {
        hex = set->ff_host_experts[(size_t) il];
    }
    if (hex == nullptr || hex->data == nullptr || hex->buffer == nullptr ||
        !ggml_is_contiguous(hex)) {
        return ls_shape_failure(ls, "layer_streaming_compute: host-leg per-row outputs were not stashed for layer " +
                                    std::to_string(il) + " (build_moe_host_leg did not capture experts_out)");
    }
    if (hex->ne[1] != ls->experts_host_n || hex->ne[2] != n_tok) {
        return ls_shape_failure(ls, "layer_streaming_compute: host-leg per-row outputs have shape [" +
                                    std::to_string(hex->ne[0]) + ", " + std::to_string(hex->ne[1]) + ", " +
                                    std::to_string(hex->ne[2]) + "] for layer " + std::to_string(il) +
                                    ", expected [*, " + std::to_string(ls->experts_host_n) + ", " +
                                    std::to_string(n_tok) + "]");
    }
    ggml_tensor * hsel = set->ff_input_hsel[(size_t) il];
    if (hsel == nullptr || hsel->data == nullptr) {
        return ls_shape_failure(ls, "layer_streaming_compute: host-leg id input missing at scan for layer " +
                                    std::to_string(il));
    }
    // the CPU quantized matmul turns a NaN activation block into zeros, so the output rows alone cannot see a bad input
    static const char * const kGuardedTensors[2] = { "split_ffn_in", "split_dev_out" };
    for (const char * nm : kGuardedTensors) {
        const std::string where = ls_first_nonfinite(set->ff_graphs[(size_t) il], nm);
        if (!where.empty()) {
            ls->error_msg = "layer_streaming_compute: non-finite split FFN tensor " + where + " at layer " + std::to_string(il);
            host_finite = 1;
            return GGML_STATUS_SUCCESS;
        }
    }
    ls_report_host_shadow(ls, set->ff_graphs[(size_t) il], il);
    const int64_t ne0 = hex->ne[0], ne1 = hex->ne[1];
    std::vector<float> rowbuf;
    const float * rows = nullptr;
    if (ggml_backend_buffer_is_host(hex->buffer)) {
        rows = (const float *) hex->data;
    } else {
        rowbuf.resize((size_t) (ne0 * ne1 * hex->ne[2]));
        ggml_backend_tensor_get(hex, rowbuf.data(), 0, rowbuf.size() * sizeof(float));
        rows = rowbuf.data();
    }
    std::string bad;
    // T-GPLR-61 step-C evidence (fault injection, test-only): when
    // LLAMA_LS_TEST_NONFINITE is set, the FIRST scanned element is
    // forced to read as non-finite, so this abort path runs for real
    // end-to-end (scan loop, expert readback from hsel, message with
    // layer/token/expert, host_finite=1 record, FAILED return --
    // measured 2026-10-08, gplr61-nonfinite). Unset (the default)
    // the branch is never taken: zero behavior change. Nothing
    // downstream ever consumes the poisoned value because the run
    // aborts below on the same pass.
    static const bool test_nonfinite = getenv("LLAMA_LS_TEST_NONFINITE") != nullptr;
    bool injected = false;
    for (int64_t tt = 0; tt < hex->ne[2] && bad.empty(); ++tt) {
        for (int64_t r = 0; r < ne1 && bad.empty(); ++r) {
            const float * rp = rows + (tt * ne1 + r) * ne0;
            for (int64_t cc = 0; cc < ne0; ++cc) {
                float v = rp[cc];
                if (test_nonfinite && !injected) {
                    v = std::nanf("");
                    injected = true;
                }
                if (!std::isfinite(v)) {
                    const int32_t expert =
                        ((const int32_t *) hsel->data)[tt * hsel->ne[0] + r];
                    bad = "layer_streaming_compute: non-finite host-expert output at layer " +
                          std::to_string(il) + " token " + std::to_string(tt) +
                          " expert " + std::to_string(expert) +
                          " (row " + std::to_string(r) + ", col " + std::to_string(cc) + ")";
                    break;
                }
            }
        }
    }
    host_finite = bad.empty() ? 0 : 1;
    if (!bad.empty()) {
        ls->error_msg = bad;
    }
    return GGML_STATUS_SUCCESS;
}

ggml_status llama_context::layer_streaming_compute(llm_graph_result * res, const llm_graph_params & gparams) {
    llama_layer_streaming * ls = layer_streaming;
    if (ls == nullptr) {
        return GGML_STATUS_FAILED;
    }
    if (ls->failed) {
        return GGML_STATUS_FAILED;
    }

    ls->n_ubatches++;

    const llama_ubatch & ubatch = gparams.ubatch;

    // recurrent state-zero variant of this ubatch (fresh state -> 0, continuing -> -1)
    const auto * mctx_hybrid = static_cast<const llama_memory_hybrid_context *>(gparams.mctx);
    const int32_t rs_z = mctx_hybrid ? mctx_hybrid->get_recr()->get_rs_z() : -1;
    const int32_t n_outputs = (int32_t) gparams.n_outputs;

    // layer-major PP (SDD section 13, T-Q38LM-02) is driven by the dedicated
    // layer-major controller (llama_context::layer_major_pp), which runs every
    // chunk through each layer's prebuilt per-chunk set. This per-ubatch path
    // therefore handles only: (a) all batch-major execution, and (b) the
    // layer-major TG (phase 1), which is a standard one-token layer order and
    // reuses the TG-u1 set exactly like batch-major. Reaching this path in the
    // layer-major PP phase is a routing error (the controller must own it).
    const bool layer_major = (ls->schedule == LLAMA_PP_SCHEDULE_LAYER_MAJOR);
    if (layer_major && ls->phase == 0) {
        ls->error_msg = "layer-major PP must be driven by the layer-major controller, not the per-ubatch path";
        ls->failed = true;
        ls->tel_error(ls->error_msg.c_str());
        return GGML_STATUS_FAILED;
    }

    // select the prebuilt per-variant graph set (mode, n_tokens, rs_z, n_outputs, chunk).
    // The per-ubatch path never uses a per-chunk set (chunk_key = -1); per-chunk
    // sets are selected only by the layer-major controller.
    const int32_t chunk_key = -1;
    auto * set = ls->find_set(ls->phase, ubatch.n_tokens, rs_z, n_outputs, chunk_key);

    // T-GPLR-71: the width of the attention view this execution needs. This is
    // the SAME value the graphs will be built against when a build happens
    // below -- llama_kv_cache_context caches it at apply_ubatch() time -- so
    // comparing it with the set's recorded build width is comparing like with
    // like, not a prediction against a build.
    const llama_kv_cache_context * attn_mctx = mctx_hybrid ? mctx_hybrid->get_attn() : nullptr;
    const int32_t n_kv_req = attn_mctx ? (int32_t) attn_mctx->get_n_kv() : -1;
    ls->last_live_n_kv = n_kv_req;   // T-GPLR-71 step 5: read by phase_summary (AC1)

    // T-GPLR-71 step 3: a set built against a narrower view than is now live
    // must be re-derived, never reused -- under-reading means silently missing
    // context. Gated on phase 1 on purpose: the phase-0 sets are driven by the
    // layer-major controller (which refuses this path outright above), and
    // their per-chunk causal narrowing is a separate mechanism that must not
    // gain a rebuild trigger here. When the view granularity is never armed
    // (the default), the recorded and required widths are the same constant on
    // both sides and this never fires, which is what keeps --kv-graph-pad 0
    // bit-for-bit identical to a pre-T-GPLR-71 run.
    if (llama_layer_streaming::should_arm_view_rebuild(ls->phase, set, n_kv_req)) {
        ls->arm_tg_view_rebuild(*set, set->n_kv, n_kv_req);
    }
    // T-GPLR-62 S2: a set flagged by mark_row_set_sets_stale() is `built` (its
    // ctx, result object and act regions are all still valid and get REPLAYED)
    // but its graphs were baked against the pre-narrowing bundle, so it must be
    // re-derived before this execution reads it. Routing it through the same
    // build-on-miss path is what makes begin_set take its T-GPLR-47 replacement
    // branch; without the flag the executor would silently reuse dangling
    // pointers, which is the illegal access this whole mechanism exists to stop.
    if (set != nullptr && set->needs_rebuild) {
        if (!ls->rebuild_replacement) {
            ls->error_msg = "layer_streaming_compute: set is flagged for rebuild but replacement is not armed";
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return GGML_STATUS_FAILED;
        }
    }
    if (set == nullptr || !set->built || (set != nullptr && set->needs_rebuild)) {
        // not prebuilt (or flagged stale by the S2 narrowing): build this variant
        // now, counted in graph_builds telemetry.
        set = ls->begin_set(ls->phase, ubatch.n_tokens, rs_z, n_outputs, chunk_key);
        if (set == nullptr) {
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return GGML_STATUS_FAILED;
        }
        llm_graph_params bp = gparams;
        bp.res = set->res;
        ls->cur_chunk_id = chunk_key; // so build_graph's begin_build re-selects this set
        ls->cur_build_n_kv = n_kv_req; // so finish_build records what this build sees
        if (model.build_graph(bp) == nullptr) {
            ls->error_msg = "layer_streaming_compute: failed to build graph set (mode " +
                            std::to_string(ls->phase) + ", n_tokens " + std::to_string(ubatch.n_tokens) +
                            ", rs_z " + std::to_string(rs_z) + ", n_outputs " + std::to_string(n_outputs) + ")";
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return GGML_STATUS_FAILED;
        }
    }

    // shape guard: every prebuilt input object must match the current ubatch
    // shapes exactly (the input objects also re-point their mctx here). NOTE:
    // llm_graph_result::can_reuse is intentionally NOT used: its params-based
    // half compares the stored ubatch by pointer fields, which dangle across
    // the prebuild -> decode boundary. The tensor-shape checks below are the
    // actual guard (the set key already pins mode/n_tokens/rs_z/n_outputs).
    bool shapes_ok = true;
    for (size_t iinput = 0; iinput < set->res->inputs.size(); ++iinput) {
        auto & input = set->res->inputs[iinput];
        const bool ok = input->can_reuse(gparams);
        if (!ok) {
            shapes_ok = false;
            break;
        }
    }
    if (!shapes_ok) {
        ls->error_msg = "layer_streaming_compute: graph set (mode " + std::to_string(ls->phase) +
                ", n_tokens " + std::to_string(ubatch.n_tokens) + ", rs_z " + std::to_string(rs_z) +
                ", n_outputs " + std::to_string(n_outputs) + ") does not match current ubatch shapes";
        ls->failed = true;
        ls->tel_error(ls->error_msg.c_str());
        return GGML_STATUS_FAILED;
    }

    // write the ubatch input values into this set's input tensors
    set->res->set_inputs(&ubatch);

    const auto & graphs = set->graphs;
    const auto & glayer = set->graph_layer;

    if (graphs.empty()) {
        ls->error_msg = "layer_streaming_compute: no graphs";
        return GGML_STATUS_FAILED;
    }

    // locate the CUDA compute backend (for event ordering)
    ggml_backend_t backend_cuda = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched.get()); ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched.get(), i);
        if (backend_is_cuda(b)) {
            backend_cuda = b;
            break;
        }
    }
    if (backend_cuda == nullptr) {
        ls->error_msg = "layer_streaming_compute: no CUDA backend in scheduler";
        return GGML_STATUS_FAILED;
    }
    // T-GPLR-19: publish the compute backend for the row-cache refresh, which
    // needs it as the cross-backend copy destination (the cache->slot device-
    // side refresh). Resolved here because the scheduler owns it, not the
    // module.
    ls->backend_compute = backend_cuda;
    // T-GPLR-35: the staging aux stream lives in this context; a run that asked
    // for copy/compute overlap but cannot get it must fail loudly rather than
    // silently fall back to the serialized copy-stream path.
    if (ls->pp_copy_overlap && !ggml_backend_cuda_aux_stream_ok(backend_cuda)) {
        ls->error_msg = "copy-overlap requested but the compute backend's CUDA context has no auxiliary stream";
        return GGML_STATUS_FAILED;
    }

    // whether a layer is selected for execution
    auto should_exec = [&](int il) -> bool {
        if (ls->exec_layers.empty()) {
            return true;
        }
        for (int e : ls->exec_layers) {
            if (e == il) {
                return true;
            }
        }
        return false;
    };

    // -----------------------------------------------------------------------
    // T-GPLR-13: reactive expert row-set TG (moe-row-set policy).
    // Two-part graphs per layer: run part A (attn + router), read the router's
    // exact top-k IDs host-side, stage the missing expert rows into the layer's
    // slot under the copy-event/generation lifecycle, then run part B (FFN +
    // shared experts bound to the host-supplied top-k input). Routing changes
    // nothing: part A's probabilities ARE the router's output; part B recomputes
    // the identical norm/router and consumes the same staged weights. No
    // per-token route events here: rowset_fetch is the single route record.
    // (SDD amendment sections 4/6: staging happens BEFORE graph execution;
    // scheduler cross-backend staging is prohibited on this path.)
    // The presence of ff_graphs in a TG set IS the row-set layout: any TG
    // decode against a two-part set without active staging is a wiring error
    // and must fail loudly rather than silently run half a layer.
    if (ls->phase == 1 && !set->ff_graphs.empty()) {
        if (!ls->row_set_active) {
            ls->error_msg = "layer_streaming_compute: two-part row-set TG set executed without active row-set staging "
                            "(begin_row_set must run after PP before TG decode)";
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return GGML_STATUS_FAILED;
        }
        if (set->ff_graphs.size() != graphs.size() - 2 || set->ff_input.size() != set->ff_graphs.size()) {
            ls->error_msg = "layer_streaming_compute: row-set set has " +
                            std::to_string(set->ff_graphs.size()) + " FFN graphs for " +
                            std::to_string(graphs.size() - 2) + " layers";
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return GGML_STATUS_FAILED;
        }
        // T-GPLR-62 S2: the window-position vector must track ff_input exactly.
        // A short vector here would index out of bounds when the executor fills
        // it, so it is checked rather than assumed.
        if (set->ff_input_wpos.size() != set->ff_input.size()) {
            ls->error_msg = "layer_streaming_compute: row-set set has " +
                            std::to_string(set->ff_input_wpos.size()) +
                            " window-position inputs for " + std::to_string(set->ff_input.size()) + " FFN inputs";
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return GGML_STATUS_FAILED;
        }
        // T-GPLR-82 S2: the host-leg id vector must track ff_input exactly,
        // because the executor indexes it BY LAYER when it fills the host
        // membership. A short vector there is an out-of-bounds read of a
        // routing table, so it is checked rather than assumed.
        if (set->ff_input_hsel.size() != set->ff_input.size()) {
            ls->error_msg = "layer_streaming_compute: row-set set has " +
                            std::to_string(set->ff_input_hsel.size()) +
                            " host-leg inputs for " + std::to_string(set->ff_input.size()) + " FFN inputs";
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return GGML_STATUS_FAILED;
        }
        if (ls->row_set_spec_arm == LLAMA_LS_ARM_D1) {
            ls->d1_last_routes.assign((size_t) ubatch.n_tokens, {});
        }

        // F3/H8: time each part around the staging so the TG transfer term can
        // be attributed against real compute. setup_us is reset+alloc, compute_us
        // is launch+execute+sync -- the same split the other executors record.
        auto run_graph_plain = [&](ggml_cgraph * g, double & compute_us, double & setup_us) -> ggml_status {
            struct src_snapshot { ggml_tensor * src[GGML_MAX_SRC]; };
            std::vector<src_snapshot> src_saved;
            src_saved.reserve(g->n_nodes);
            for (int k = 0; k < g->n_nodes; ++k) {
                src_snapshot s;
                for (int j = 0; j < GGML_MAX_SRC; ++j) s.src[j] = g->nodes[k]->src[j];
                src_saved.push_back(s);
            }
            compute_us = 0.0;
            setup_us   = 0.0;
            const int64_t t0 = ggml_time_us();
            ggml_backend_sched_reset(sched.get());
            if (!ggml_backend_sched_alloc_graph(sched.get(), g)) {
                ls->error_msg = "layer_streaming_compute: row-set failed to allocate graph";
                ls->failed = true;
                return GGML_STATUS_ALLOC_FAILED;
            }
            const int64_t t1 = ggml_time_us();
            const auto status = ggml_backend_sched_graph_compute_async(sched.get(), g);
            if (status != GGML_STATUS_SUCCESS) {
                ls->failed = true;
                return status;
            }
            ggml_backend_sched_synchronize(sched.get());
            const int64_t t2 = ggml_time_us();
            for (int k = 0; k < g->n_nodes; ++k) {
                for (int j = 0; j < GGML_MAX_SRC; ++j) g->nodes[k]->src[j] = src_saved[k].src[j];
            }
            compute_us = (double) (t2 - t1);
            setup_us   = (double) (t1 - t0);
            return GGML_STATUS_SUCCESS;
        };

        // T-GPLR-82 S1: stamp the round start; rowset_round_end stamps the
        // end. Telemetry only -- no execution path reads this.
        ls->rowset_round_t0_us = ggml_time_us();

        // 1. embed graph (graphs[0])
        {
            double cu = 0.0, su = 0.0;
            const auto st = run_graph_plain(graphs[0], cu, su);
            if (st != GGML_STATUS_SUCCESS) {
                ls->tel_error(ls->error_msg.c_str());
                return st;
            }
        }

        // 2. per layer: part A -> read router IDs -> stage rows -> part B
        for (int il = 0; il < ls->n_layer; ++il) {
            if (!should_exec(il)) {
                continue;
            }
            // diagnostic (LLAMA_LS_HOST_SHADOW): norm of the activation slots entering this layer
            static const bool act_diag = getenv("LLAMA_LS_HOST_SHADOW") != nullptr;
            if (act_diag && (il == 0 || il == 1) && ls->n_ubatches <= 5) {
                for (int ai = 0; ai < 2; ++ai) {
                    ggml_tensor * at = set->act[ai];
                    double nrm = -1.0;
                    if (at != nullptr && at->type == GGML_TYPE_F32) {
                        std::vector<float> v((size_t) ggml_nelements(at));
                        ggml_backend_tensor_get(at, v.data(), 0, v.size() * sizeof(float));
                        nrm = 0.0;
                        for (float x : v) { nrm += (double) x * x; }
                        nrm = std::sqrt(nrm);
                    }
                    LLAMA_LOG_WARN("act-check ubatch %llu before layer %d act[%d] type %d |x| %.4e\n", (unsigned long long) ls->n_ubatches, il, ai, at ? (int) at->type : -1, nrm);
                }
            }
            // part A: attention + residual + post-norm + router (graphs[1+il])
            double attn_compute_us = 0.0, attn_setup_us = 0.0;
            const auto st_a = run_graph_plain(graphs[1 + il], attn_compute_us, attn_setup_us);
            if (st_a != GGML_STATUS_SUCCESS) {
                ls->tel_error(ls->error_msg.c_str());
                return st_a;
            }
            // T-GPLR-75: classified AFTER the run, because alloc_graph is what
            // assigns each node its buffer.
            const int attn_dev = ls_classify_graph_backend(graphs[1 + il]);
            ls->accum_graph_ops(graphs[1 + il], true);

            // read the router's exact top-k IDs from part A's probabilities
            std::vector<std::vector<int32_t>> ids_per_token;
            std::vector<float> probs_buf;
            if (!ls->read_router_ids(il, graphs[1 + il], ids_per_token, probs_buf)) {
                ls->failed = true;
                ls->tel_error(ls->error_msg.c_str());
                return GGML_STATUS_FAILED;
            }
            if (ls->row_set_spec_arm == LLAMA_LS_ARM_D1) {
                if (ids_per_token.size() != (size_t) ubatch.n_tokens) {
                    ls->error_msg = "layer_streaming_compute: D1 route row count " +
                                    std::to_string(ids_per_token.size()) + " != ubatch " +
                                    std::to_string(ubatch.n_tokens);
                    ls->failed = true;
                    ls->tel_error(ls->error_msg.c_str());
                    return GGML_STATUS_FAILED;
                }
                for (size_t t = 0; t < ids_per_token.size(); ++t) {
                    ls->d1_last_routes[t].resize((size_t) ls->n_layer);
                    ls->d1_last_routes[t][(size_t) il] = ids_per_token[t];
                }
            }

            // T-GPLR-82 S1: host-side span of this layer's row-set consume.
            // A real measurement, not an estimate. Today it covers the union
            // build plus load_row_set, because no host expert sub-op exists
            // yet; from S2 the same timer also covers that work. This is the
            // 36.5 vs 59.4 us/expert gap made observable rather than assumed.
            const int64_t t_host0 = ggml_time_us();

            // the token-level union of routed experts (the declared MTP width
            // is the number of target rows in this ubatch);
            // ids_per_token has exactly one row here)
            std::vector<int32_t> union_ids;
            for (const auto & ids : ids_per_token) {
                for (int32_t e : ids) {
                    union_ids.push_back(e);
                }
            }

            // stage missing rows for this token BEFORE part B executes
            if (!ls->load_row_set(il, union_ids, &ids_per_token)) {
                ls->failed = true;
                ls->tel_error(ls->error_msg.c_str());
                return GGML_STATUS_FAILED;
            }

            // bind the top-k input: sel_in is [n_expert_used, n_tokens] I32,
            // column-major token slices (matches ggml_argsort_top_k's layout:
            // element (k, t) at data[k + t*n_expert_used])
            ggml_tensor * sel_in = set->ff_input[(size_t) il];
            if (sel_in == nullptr || sel_in->data == nullptr) {
                ls->error_msg = "layer_streaming_compute: row-set FFN input missing for layer " + std::to_string(il);
                ls->failed = true;
                ls->tel_error(ls->error_msg.c_str());
                return GGML_STATUS_FAILED;
            }
            const int64_t n_used = sel_in->ne[0];
            const int64_t n_tok  = sel_in->ne[1];
            GGML_ASSERT((int64_t) ids_per_token.size() == n_tok);

            // T-GPLR-61 S3/S4: the per-token plan, the fixed-shape reorder,
            // and the step-C guards. The graph split is fixed at build (device
            // leg = first n_used - experts_host_n positions of sel_in with the
            // matching w_full view; host leg = the separate hsel input paired
            // positionally with the w_full suffix), so residency membership is
            // expressed by REORDERING each token's row: plan-host rows take the
            // suffix. With no host-bound row the row stays in router order --
            // byte-identical to the S2 positional fill. ls_plan_split carries
            // the step-C refusals: an ABSENT demanded row (staging defect) or
            // an unevaluable gate refuses loudly instead of guessing.
            std::vector<std::vector<int32_t>> plan_host_ids((size_t) n_tok);
            int experts_host_plan = 0;
            int rows_gate_cpu_sum = 0, rows_gate_gpu_sum = 0;
            uint64_t staged_avoided_sum = 0;
            llama_ls_policy::ls_gate_verdict gate_v;
            gate_v.evaluable = true;
            gate_v.open       = ls->row_set_gate_open[(size_t) il] != 0;
            gate_v.host_us    = ls->row_set_gate_host_us[(size_t) il];
            gate_v.wait_us    = ls->row_set_gate_wait_us[(size_t) il];
            // T-GPLR-82 S3/S4 (2026-10-10): the split graph is built only when
            // the mechanism can admit a row (decided in the builder). Read THAT,
            // not the flag, so a graph with no host leg is never asked for its id
            // input -- measured: demanding it aborts loudly ("host-leg id input
            // missing or misshaped for layer 0 (expected [2, 1])").
            const bool host_leg_built = ls->row_set_split_built.size() > (size_t) il &&
                                        ls->row_set_split_built[(size_t) il] != 0;
            if (host_leg_built) {
                const uint64_t kRowBytes = 1769472u;
                for (int64_t t = 0; t < n_tok; ++t) {
                    const std::vector<int32_t> & row = ids_per_token[(size_t) t];
                    std::vector<llama_ls_policy::ls_row_state> dem(row.size());
                    for (size_t k = 0; k < row.size(); ++k) {
                        dem[k] = (llama_ls_policy::ls_row_state)
                                 ls->row_set_state[(size_t) il][(size_t) row[k]];
                    }
                    const llama_ls_policy::ls_split_plan plan =
                        llama_ls_policy::ls_plan_split(dem, gate_v, (uint32_t) row.size(), kRowBytes);
                    if (!plan.ok) {
                        ls->error_msg = "layer_streaming_compute: split plan refused for layer " +
                                        std::to_string(il) + " token " + std::to_string(t) + ": " +
                                        (plan.refuse_reason ? plan.refuse_reason : "unknown");
                        ls->failed = true;
                        ls->tel_error(ls->error_msg.c_str());
                        return GGML_STATUS_FAILED;
                    }
                    rows_gate_cpu_sum += (int) plan.rows_gate_cpu;
                    rows_gate_gpu_sum += (int) plan.rows_gate_gpu;
                    staged_avoided_sum += plan.staged_bytes_avoided;

                    std::vector<size_t> host_pos, dev_pos;
                    for (uint32_t k = 0; k < plan.host.size(); ++k) {
                        if (plan.host[k].on_host) {
                            host_pos.push_back(k);
                        } else {
                            dev_pos.push_back(k);
                        }
                    }
                    experts_host_plan += (int) host_pos.size();
                    for (size_t k : host_pos) {
                        plan_host_ids[(size_t) t].push_back(row[k]);
                    }
                    if (host_pos.empty()) {
                        continue; // router order == S2 positional fill
                    }
                    if ((int) host_pos.size() > ls->experts_host_n) {
                        ls->error_msg = "layer_streaming_compute: plan wants " +
                                        std::to_string(host_pos.size()) + " host rows for layer " +
                                        std::to_string(il) + " token " + std::to_string(t) +
                                        " but the host leg width is " + std::to_string(ls->experts_host_n);
                        ls->failed = true;
                        ls->tel_error(ls->error_msg.c_str());
                        return GGML_STATUS_FAILED;
                    }
                    // Fixed-width partition: suffix = host rows ++ the LAST
                    // (host_w - n_host) device rows as fillers; the prefix
                    // keeps exactly n_used - host_w rows. Every demanded row
                    // appears exactly once, so the two partials still sum to
                    // the full weighted top-k.
                    const size_t fill_n = (size_t) ls->experts_host_n - host_pos.size();
                    const size_t keep   = dev_pos.size() - fill_n; // == n_used - host_w
                    // Step-C safety: the device prefix may contain ONLY READY
                    // rows -- a non-READY row there is the one shape that
                    // could consume an incomplete slab.
                    for (size_t i = 0; i < keep; ++i) {
                        if (ls->row_set_state[(size_t) il][(size_t) row[dev_pos[i]]] != 0) {
                            ls->error_msg = "layer_streaming_compute: expert " +
                                            std::to_string(row[dev_pos[i]]) + " of layer " +
                                            std::to_string(il) + " is not READY but would land on the device prefix";
                            ls->failed = true;
                            ls->tel_error(ls->error_msg.c_str());
                            return GGML_STATUS_FAILED;
                        }
                    }
                    std::vector<int32_t> reordered;
                    reordered.reserve(row.size());
                    for (size_t i = 0; i < keep; ++i) {
                        reordered.push_back(row[dev_pos[i]]);
                    }
                    for (size_t k : host_pos) {
                        reordered.push_back(row[k]);
                    }
                    for (size_t i = keep; i < dev_pos.size(); ++i) {
                        reordered.push_back(row[dev_pos[i]]);
                    }
                    ids_per_token[(size_t) t] = std::move(reordered);
                }
            }
            for (int64_t t = 0; t < n_tok; ++t) {
                GGML_ASSERT((int64_t) ids_per_token[(size_t) t].size() == n_used);
                memcpy((int32_t *) sel_in->data + t * n_used,
                       ids_per_token[(size_t) t].data(), (size_t) n_used * sizeof(int32_t));
            }

            // the device leg reads the first n_used - experts_host_n positions: each must hold a slab this round wrote
            if (host_leg_built) {
                const int64_t n_dev = n_used - ls->experts_host_n;
                const std::vector<uint8_t> & fresh = ls->row_set_fresh[(size_t) il];
                for (int64_t t = 0; t < n_tok; ++t) {
                    for (int64_t k = 0; k < n_dev; ++k) {
                        const int32_t e = ids_per_token[(size_t) t][(size_t) k];
                        if (e < 0 || (size_t) e >= fresh.size() || !fresh[(size_t) e]) {
                            ls->error_msg = "layer_streaming_compute: expert " + std::to_string(e) + " of layer " +
                                            std::to_string(il) + " token " + std::to_string(t) + " position " +
                                            std::to_string(k) + " would be read by the device leg but its slot slab was not written this round";
                            ls->failed = true;
                            ls->tel_error(ls->error_msg.c_str());
                            return GGML_STATUS_FAILED;
                        }
                    }
                }
            }

            // T-GPLR-62 S2: the SAME experts, re-expressed as positions in the
            // window that load_row_set just staged. The table is host-side and
            // was reset to -1 in full for this window, so a -1 here means the
            // expert the router selected is not in the staged union -- which is
            // the A10 subset condition, not a bookkeeping detail. It refuses
            // loudly and NEVER falls back to a stale position from a previous
            // window, never clamps, and never wraps.
            //
            // While the slot is still full-bank the graph ignores this tensor, so
            // this is currently an audit of the mapping rather than the mechanism
            // itself; that is deliberate, so the table can be validated before
            // anything depends on it.
            ggml_tensor * sel_pos = set->ff_input_wpos[(size_t) il];
            if (sel_pos != nullptr) {
                if (!ls->row_set_union_valid()) {
                    ls->error_msg = "layer_streaming_compute: no union position table for layer " +
                                    std::to_string(il) + " (load_row_set did not run)";
                    ls->failed = true;
                    ls->tel_error(ls->error_msg.c_str());
                    return GGML_STATUS_FAILED;
                }
                if (sel_pos->data == nullptr || sel_pos->ne[0] != n_used || sel_pos->ne[1] != n_tok) {
                    ls->error_msg = "layer_streaming_compute: row-set window-position input has the wrong "
                                    "shape for layer " + std::to_string(il);
                    ls->failed = true;
                    ls->tel_error(ls->error_msg.c_str());
                    return GGML_STATUS_FAILED;
                }
                const std::vector<int32_t> & upos = ls->row_set_union_pos();
                for (int64_t t = 0; t < n_tok; ++t) {
                    for (int64_t k = 0; k < n_used; ++k) {
                        const int32_t e = ids_per_token[(size_t) t][(size_t) k];
                        if (e < 0 || (size_t) e >= upos.size()) {
                            ls->error_msg = "layer_streaming_compute: expert " + std::to_string(e) +
                                            " outside the union table for layer " + std::to_string(il);
                            ls->failed = true;
                            ls->tel_error(ls->error_msg.c_str());
                            return GGML_STATUS_FAILED;
                        }
                        const int32_t p = upos[(size_t) e];
                        if (p < 0) {
                            ls->error_msg = "layer_streaming_compute: expert " + std::to_string(e) +
                                            " was selected for token " + std::to_string(t) +
                                            " but is not in the staged union for layer " + std::to_string(il) +
                                            " (union width " + std::to_string(upos.size()) + " table)";
                            ls->failed = true;
                            ls->tel_error(ls->error_msg.c_str());
                            return GGML_STATUS_FAILED;
                        }
                        ((int32_t *) sel_pos->data)[t * n_used + k] = p;
                    }
                }
            }


            // T-GPLR-82 S2: the HOST leg's routed ids.
            //
            // S2 is COUPLED and the membership rule is POSITIONAL: the last
            // `experts_host_n` positions of the router's top_k go to the host
            // leg and the device leg keeps the first n_expert_used -
            // experts_host_n. The graph shape is therefore fixed and the rule
            // costs nothing to evaluate. It is deliberately NOT cache-aware
            // yet -- S3 is where a MISSED row goes host instead of waiting for
            // its copy, turning this from a fixed suffix into a per-row choice.
            //
            // These are a COPY of the same top_k the device leg consumes, not a
            // view of it: this INPUT lives on the CPU side of the backend split,
            // and the DMA in flight is reading model weights, never these bytes.
            if (host_leg_built) {
                ggml_tensor * hsel = set->ff_input_hsel[(size_t) il];
                if (hsel == nullptr || hsel->data == nullptr ||
                    hsel->ne[0] != ls->experts_host_n || hsel->ne[1] != n_tok) {
                    ls->error_msg = "layer_streaming_compute: host-leg id input missing or misshaped "
                                    "for layer " + std::to_string(il) + " (expected [" +
                                    std::to_string(ls->experts_host_n) + ", " + std::to_string(n_tok) + "])";
                    ls->failed = true;
                    ls->tel_error(ls->error_msg.c_str());
                    return GGML_STATUS_FAILED;
                }
                const int64_t hn = ls->experts_host_n;
                for (int64_t t = 0; t < n_tok; ++t) {
                    const auto & row = ids_per_token[(size_t) t];
                    if ((int64_t) row.size() != n_used) {
                        ls->error_msg = "layer_streaming_compute: routed row width " +
                                        std::to_string(row.size()) + " != top_k " +
                                        std::to_string(n_used) + " for layer " + std::to_string(il);
                        ls->failed = true;
                        ls->tel_error(ls->error_msg.c_str());
                        return GGML_STATUS_FAILED;
                    }
                    memcpy((int32_t *) hsel->data + t * hn,
                           row.data() + (n_used - hn), (size_t) hn * sizeof(int32_t));
                }
            }

            // T-GPLR-61 step C: a row the gate approved for the host and
            // then did not execute is a loud abort. The fill above copies the
            // row suffix into hsel, so every plan-host id must sit in it.
            if (host_leg_built) {
                for (int64_t t = 0; t < n_tok; ++t) {
                    const std::vector<int32_t> & row = ids_per_token[(size_t) t];
                    for (int32_t e : plan_host_ids[(size_t) t]) {
                        bool seen = false;
                        for (int64_t k = n_used - ls->experts_host_n; k < n_used && !seen; ++k) {
                            seen = row[(size_t) k] == e;
                        }
                        if (!seen) {
                            ls->error_msg = "layer_streaming_compute: expert " + std::to_string(e) +
                                            " of layer " + std::to_string(il) + " token " +
                                            std::to_string(t) +
                                            " was approved for the host leg but is not in the executed suffix";
                            ls->failed = true;
                            ls->tel_error(ls->error_msg.c_str());
                            return GGML_STATUS_FAILED;
                        }
                    }
                }
            }
 // T-GPLR-61 S4: the ledger is the PLAN's disposition -- what was
            // INTENDED to run on the host -- not the graph's physical width;
            // ffn_dev below stays the independent record of what actually
            // executed (S1's intended-vs-happened pair). The clamp keeps AC4's
            // experts_device + experts_host == experts_total exact even when
            // the union is narrower than n_tok * top_k.
            const int experts_total = (int) ls->row_set_ids[il].size();
            int experts_host = experts_host_plan;
            if (experts_host > experts_total) {
                experts_host = experts_total;
            }

            if (act_diag && (il == 0 || il == 8) && ls->n_ubatches <= 5) {
                ggml_tensor * rt = set->act[(il + 1) & 1];
                double nrm = -1.0;
                if (rt != nullptr && rt->type == GGML_TYPE_F32) {
                    std::vector<float> v((size_t) ggml_nelements(rt));
                    ggml_backend_tensor_get(rt, v.data(), 0, v.size() * sizeof(float));
                    nrm = 0.0;
                    for (float x : v) { nrm += (double) x * x; }
                    nrm = std::sqrt(nrm);
                }
                LLAMA_LOG_WARN("resid-check ubatch %llu layer %d |residual before part B| %.4e\n", (unsigned long long) ls->n_ubatches, il, nrm);
                if (nrm > 0.0) {
                    std::vector<float> xv((size_t) ggml_nelements(rt));
                    ggml_backend_tensor_get(rt, xv.data(), 0, xv.size() * sizeof(float));
                    const ggml_tensor * wt = ls->model_ref->layers[il].attn_post_norm;
                    if (wt != nullptr && wt->type == GGML_TYPE_F32 && (size_t) ggml_nelements(wt) == xv.size()) {
                        std::vector<float> wv(xv.size());
                        ggml_backend_tensor_get(wt, wv.data(), 0, wv.size() * sizeof(float));
                        double ms = 0.0;
                        for (float x : xv) { ms += (double) x * x; }
                        ms /= (double) xv.size();
                        for (int e = 0; e < 3; ++e) {
                            const double eps = e == 0 ? 1e-6 : (e == 1 ? 1e-2 : 1.0);
                            const double sc = 1.0 / std::sqrt(ms + eps);
                            double acc = 0.0;
                            for (size_t k = 0; k < xv.size(); ++k) { const double y = (double) xv[k] * sc * wv[k]; acc += y * y; }
                            LLAMA_LOG_WARN("norm-predict ubatch %llu layer %d eps %.0e -> |cur| %.4e\n", (unsigned long long) ls->n_ubatches, il, eps, std::sqrt(acc));
                        }
                    }
                }
            }
            // part B: FFN + shared experts on the staged rows (ff_graphs[il])
            double ffn_compute_us = 0.0, ffn_setup_us = 0.0;
            static const bool no_reset = getenv("LLAMA_LS_TEST_NO_ALLOC_RESET") != nullptr;
            if (host_leg_built && !no_reset) {
                ls_reset_graph_allocs(set->ff_graphs[(size_t) il]);
            }
            const auto st_b = run_graph_plain(set->ff_graphs[(size_t) il], ffn_compute_us, ffn_setup_us);
            if (st_b != GGML_STATUS_SUCCESS) {
                ls->tel_error(ls->error_msg.c_str());
                return st_b;
            }
            const int ffn_dev = ls_classify_graph_backend(set->ff_graphs[(size_t) il]);
            ls->accum_graph_ops(set->ff_graphs[(size_t) il], false);
            // diagnostic (LLAMA_LS_HOST_SHADOW): the ids the graph SAW in hsel after compute versus the ids written before it
            static const bool host_ids_check = getenv("LLAMA_LS_HOST_SHADOW") != nullptr;
            if (host_ids_check) {
                const int64_t hn = host_leg_built ? ls->experts_host_n : 0;
                ggml_tensor * hs = host_leg_built ? set->ff_input_hsel[(size_t) il] : nullptr;
                std::vector<int32_t> got((size_t) (hn * n_tok));
                if (host_leg_built) {
                    ggml_backend_tensor_get(hs, got.data(), 0, got.size() * sizeof(int32_t));
                }
                for (int64_t t = 0; t < n_tok; ++t) {
                    for (int64_t k = 0; k < hn; ++k) {
                        const int32_t want = ids_per_token[(size_t) t][(size_t) (n_used - hn + k)];
                        if (got[(size_t) (t * hn + k)] != want) {
                            LLAMA_LOG_WARN("host-ids MISMATCH ubatch %llu layer %d token %lld k %lld want %d got %d hsel->data=%p buf=%p\n",
                                           (unsigned long long) ls->n_ubatches, il, (long long) t, (long long) k, want,
                                           got[(size_t) (t * hn + k)], (void *) hs->data, (void *) hs->buffer);
                        }
                    }
                }
                if (il % 8 == 0 && ls->n_ubatches <= 5) {
                    char wn[64];
                    snprintf(wn, sizeof(wn), "ffn_moe_weights_split-%d", il);
                    ggml_cgraph * fg = set->ff_graphs[(size_t) il];
                    ggml_tensor * tin = ggml_graph_get_tensor(fg, "split_ffn_in");
                    ggml_tensor * tw  = ggml_graph_get_tensor(fg, wn);
                    ggml_tensor * tdo = ggml_graph_get_tensor(fg, "split_dev_out");
                    double nin = -1.0, ndo = -1.0;
                    float ws[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
                    if (tin != nullptr && tin->type == GGML_TYPE_F32) {
                        std::vector<float> v((size_t) ggml_nelements(tin));
                        ggml_backend_tensor_get(tin, v.data(), 0, v.size() * sizeof(float));
                        nin = 0.0;
                        for (float x : v) { nin += (double) x * x; }
                        nin = std::sqrt(nin);
                    }
                    if (tdo != nullptr && tdo->type == GGML_TYPE_F32) {
                        std::vector<float> v((size_t) ggml_nelements(tdo));
                        ggml_backend_tensor_get(tdo, v.data(), 0, v.size() * sizeof(float));
                        ndo = 0.0;
                        for (float x : v) { ndo += (double) x * x; }
                        ndo = std::sqrt(ndo);
                    }
                    if (tw != nullptr && tw->type == GGML_TYPE_F32 && ggml_nelements(tw) >= 8) {
                        ggml_backend_tensor_get(tw, ws, 0, sizeof(ws));
                    }
                    LLAMA_LOG_WARN("flow-check ubatch %llu layer %d |cur| %.4e |dev_out| %.4e w %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                                   (unsigned long long) ls->n_ubatches, il, nin, ndo, ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7]);
                }
                // compare the CPU leg's bank bytes with the slot bytes for the suffix experts of token 0
                if (host_leg_built && il % 8 == 0 && ls->n_ubatches <= 5) {
                    const llama_layer & sl = ls->slot_layer(il);
                    const llama_layer & hl = ls->model_ref->layers[il];
                    const ggml_tensor * hm[3] = { hl.ffn_gate_exps, hl.ffn_up_exps, hl.ffn_down_exps };
                    const ggml_tensor * sm[3] = { sl.ffn_gate_exps, sl.ffn_up_exps, sl.ffn_down_exps };
                    for (int64_t k = 0; k < hn; ++k) {
                        const int32_t e = ids_per_token[0][(size_t) (n_used - hn + k)];
                        for (int mi = 0; mi < 3; ++mi) {
                            if (hm[mi] == nullptr || sm[mi] == nullptr) {
                                continue;
                            }
                            const size_t slab = ggml_row_size(hm[mi]->type, hm[mi]->ne[0]) * (size_t) hm[mi]->ne[1];
                            std::vector<uint8_t> hb(256), sb(256);
                            const bool host_ok = hm[mi]->buffer != nullptr && ggml_backend_buffer_is_host(hm[mi]->buffer);
                            if (host_ok) {
                                memcpy(hb.data(), (const uint8_t *) hm[mi]->data + (size_t) e * slab, 256);
                            }
                            ggml_backend_tensor_get(sm[mi], sb.data(), (size_t) e * slab, 256);
                            LLAMA_LOG_WARN("bank-check ubatch %llu layer %d expert %d member %d host_ptr=%p host_ok=%d slab_match=%d\n",
                                           (unsigned long long) ls->n_ubatches, il, e, mi, (void *) hm[mi]->data, (int) host_ok,
                                           (int) (host_ok && memcmp(hb.data(), sb.data(), 256) == 0));
                        }
                    }
                }
            }

            // slot lifecycle: part B consumed this slot generation; part A's
            // attention weights are model-resident (no slot ownership)
            ls->record_consumer_done(il, backend_cuda);
            // T-GPLR-61 step C: buffer read, scan and error shaping live in
            // scan_host_expert_outputs(); the abort runs here, after the
            // layer_compute record below carries host_finite.
            int host_finite = -1;
            if (host_leg_built) {
                if (scan_host_expert_outputs(ls, set, il, n_tok, host_finite) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
            }
            // T-GPLR-82 S2: the demand ledger now SPLITS.
            // S2 membership is POSITIONAL -- the last
            // experts_host_n positions of every top-k go to
            // the host leg -- so each token of this layer
            // routes exactly experts_host_n rows host and the
            // remainder device. Clamped to the union so a
            // narrower union can never make the device term
            // negative: experts_device + experts_host ==
            // experts_total is the AC2 invariant and it has
            // to hold on every emitted record.
            llama_layer_streaming::tel_layer_compute_params tel_rowset;
            tel_rowset.attn_compute_us = attn_compute_us;
            tel_rowset.attn_setup_us   = attn_setup_us;
            tel_rowset.ffn_compute_us  = ffn_compute_us;
            tel_rowset.ffn_setup_us    = ffn_setup_us;
            tel_rowset.attn_dev        = attn_dev;
            tel_rowset.ffn_dev         = ffn_dev;
            tel_rowset.experts_total   = experts_total;
            tel_rowset.experts_host    = experts_host;
            tel_rowset.experts_device  = experts_total - experts_host;
            tel_rowset.host_us         = (double) (ggml_time_us() - t_host0);
            // T-GPLR-61 S4/S1: the gate's own record,
            // emitted even when the split never fires.
            // -1 (not reported) when no host leg exists
            // at all; a 0 host share with live estimates
            // is a recorded did-not-fire, not a pass.
            tel_rowset.rows_gate_cpu   = host_leg_built ? rows_gate_cpu_sum : -1;
            tel_rowset.rows_gate_gpu   = host_leg_built ? rows_gate_gpu_sum : -1;
            tel_rowset.gate_host_us    = host_leg_built ? gate_v.host_us : -1.0;
            tel_rowset.gate_wait_us    = host_leg_built ? gate_v.wait_us : -1.0;
            tel_rowset.staged_bytes_avoided =
                host_leg_built ? (double) staged_avoided_sum : -1.0;
            tel_rowset.host_finite     = host_finite;
            ls->tel_layer_compute(il, attn_compute_us + ffn_compute_us, attn_setup_us + ffn_setup_us,
                                  ls->slots[il % ls->n_slots].generation, (int64_t) (ls->n_ubatches - 1),
                                  tel_rowset);
            if (host_finite == 1) {
                return ls_shape_failure(ls, ls->error_msg);
            }
            // T-GPLR-24 (H-A): the TG phase_summary must carry the row-set
            // terms in row-set modes. transfer = submit + ready-wait of this
            // layer's staging; wait = the ready-latency the executor blocked
            // on; compute/setup = the A+B sums. Same accounting the other
            // executors feed via add_layer_stats.
            ls->add_layer_stats(il, ls->rowset_last_submit_us + ls->rowset_last_ready_us,
                                ls->rowset_last_ready_us,
                                attn_compute_us + ffn_compute_us, attn_setup_us + ffn_setup_us,
                                attn_compute_us + ffn_compute_us + attn_setup_us + ffn_setup_us,
                                ls->rowset_last_ready_us < 100.0);
        }

        ls->tel_graph_backend_ops();

        // 3. output graph (graphs[1+n_layer])
        {
            double cu = 0.0, su = 0.0;
            const auto st = run_graph_plain(graphs[1 + ls->n_layer], cu, su);
            if (st != GGML_STATUS_SUCCESS) {
                ls->tel_error(ls->error_msg.c_str());
                return st;
            }
        }

        ls->rowset_round_end();

        // drain the copy stream so no copies are pending when the ubatch ends
        ls->ls_stage_drain();

        // expose this set's output references on the context result object
        res->t_logits      = set->res->t_logits;
        res->t_embd        = set->res->t_embd;
        res->t_embd_pooled = set->res->t_embd_pooled;
        res->t_h_nextn     = set->res->t_h_nextn;
        res->t_inp_tokens  = set->res->t_inp_tokens;
        return GGML_STATUS_SUCCESS;
    }

    // phase A: preload the first n_slots selected layers (overlap window)
    {
        int preloaded = 0;
        for (int il = 0; il < ls->n_layer && preloaded < ls->n_slots; ++il) {
            if (!should_exec(il)) {
                continue;
            }
            bool copied = false;
            if (!ls->load_layer(il, copied)) {
                ls->failed = true;
                ls->tel_error(ls->error_msg.c_str());
                return GGML_STATUS_FAILED;
            }
            preloaded++;
        }
    }

    // debug: dump cpy nodes (env LLAMA_LAYER_STREAMING_DEBUG)
    static const bool debug_cpy = getenv("LLAMA_LAYER_STREAMING_DEBUG") != nullptr;
    if (debug_cpy) {
        fprintf(stderr, "[ls-debug] ubatch %llu build_generation %llu graphs %zu\n",
            (unsigned long long) ls->n_ubatches, (unsigned long long) ls->build_generation, graphs.size());
        for (size_t i = 0; i < graphs.size(); ++i) {
            ggml_cgraph * g = graphs[i];
            fprintf(stderr, "[ls-debug] graph %zu layer %d nodes %d leafs %d\n", i, glayer[i], g->n_nodes, g->n_leafs);
            for (int j = 0; j < g->n_leafs; ++j) {
                ggml_tensor * leaf = g->leafs[j];
                fprintf(stderr, "  leaf: %s op=%s ne %lld %lld %lld %lld elems %lld buf=%s\n",
                    leaf->name, ggml_op_name(leaf->op),
                    (long long) leaf->ne[0], (long long) leaf->ne[1], (long long) leaf->ne[2], (long long) leaf->ne[3],
                    (long long) ggml_nelements(leaf), leaf->buffer ? ggml_backend_buffer_name(leaf->buffer) : "none");
            }
            for (int j = 0; j < g->n_nodes; ++j) {
                ggml_tensor * node = g->nodes[j];
                if (node->op == GGML_OP_CPY) {
                    fprintf(stderr, "  cpy: src=%s op=%s (%lld elems, ne %lld %lld %lld %lld) dst=%s (%lld elems, ne %lld %lld %lld %lld)\n",
                        node->src[0]->name, ggml_op_name(node->src[0]->op), (long long) ggml_nelements(node->src[0]),
                        (long long) node->src[0]->ne[0], (long long) node->src[0]->ne[1], (long long) node->src[0]->ne[2], (long long) node->src[0]->ne[3],
                        node->name, (long long) ggml_nelements(node),
                        (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3]);
                    for (int k = 0; k < GGML_MAX_SRC; ++k) {
                        if (node->src[0]->src[k]) {
                            fprintf(stderr, "      src0.src[%d]: %s op=%s ne %lld %lld %lld %lld\n", k,
                                node->src[0]->src[k]->name, ggml_op_name(node->src[0]->src[k]->op),
                                (long long) node->src[0]->src[k]->ne[0], (long long) node->src[0]->src[k]->ne[1], (long long) node->src[0]->src[k]->ne[2], (long long) node->src[0]->src[k]->ne[3]);
                        }
                    }
                }
            }
        }
    }

    // phase B: execute graphs in order
    for (size_t i = 0; i < graphs.size(); ++i) {
        ggml_cgraph * g = graphs[i];
        const int layer = glayer[i];

        // persistent graph-entry key check: the graph may only run against the
        // slot assignment it was built for (SDD: stale use must fail loudly)
        if (i >= set->graph_entries.size()) {
            ls->error_msg = "layer_streaming_compute: missing graph entry " + std::to_string(i);
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return GGML_STATUS_FAILED;
        }
        const auto & entry = set->graph_entries[i];
        if (entry.mode != ls->phase) {
            ls->error_msg = "layer_streaming_compute: graph " + std::to_string(i) +
                            " keyed for mode " + std::to_string(entry.mode) +
                            " but phase is " + std::to_string(ls->phase);
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return GGML_STATUS_FAILED;
        }
        if (entry.logical_layer != layer) {
            ls->error_msg = "layer_streaming_compute: graph " + std::to_string(i) +
                            " keyed for layer " + std::to_string(entry.logical_layer) +
                            " but graph_layer is " + std::to_string(layer);
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return GGML_STATUS_FAILED;
        }
        if (layer >= 0 && entry.slot != layer % ls->n_slots) {
            ls->error_msg = "layer_streaming_compute: graph " + std::to_string(i) +
                            " keyed for slot " + std::to_string(entry.slot) +
                            " but layer " + std::to_string(layer) +
                            " maps to slot " + std::to_string(layer % ls->n_slots);
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return GGML_STATUS_FAILED;
        }

        const bool is_layer = layer >= 0;
        if (is_layer && !should_exec(layer)) {
            continue;
        }

        double layer_transfer_us = 0.0;
        double layer_wait_us     = 0.0;
        bool   ready_at_consumption = false;

        if (is_layer) {
            const size_t req_bytes = ls->package_bytes(layer);
            ls->tel_slot_request(layer, ls->phase, req_bytes);

            // ensure the layer's package is resident and ready
            bool copied = false;
            if (!ls->load_layer(layer, copied)) {
                ls->failed = true;
                ls->tel_error(ls->error_msg.c_str());
                return GGML_STATUS_FAILED;
            }

            double wait_us = 0.0;
            double host_submit_us = 0.0;
            double ready_latency_us = 0.0;
            if (!ls->wait_slot_ready(layer, wait_us, host_submit_us, ready_latency_us)) {
                ls->failed = true;
                ls->tel_error(ls->error_msg.c_str());
                return GGML_STATUS_FAILED;
            }
            layer_transfer_us = host_submit_us + ready_latency_us;
            layer_wait_us     = wait_us;
            ready_at_consumption = (wait_us < 100.0);

            // strict stream ordering: compute waits for the copy event
            ggml_backend_event_wait(backend_cuda, (ggml_backend_event_t) ls->slots[layer % ls->n_slots].ev_copy_ready);
        }

        // scheduler setup (reset + allocate) for this graph
        //
        // NOTE: ggml_backend_sched_alloc_graph() rewires graph node src pointers
        // in place when it introduces cross-backend copies. Snapshot them first
        // and restore them after compute so the persistent graphs stay pristine
        // and can be re-split by the next ubatch (sched->ctx is freed on each
        // split, so a stale rewired pointer would alias unrelated tensors).
        struct src_snapshot {
            ggml_tensor * src[GGML_MAX_SRC];
        };
        std::vector<src_snapshot> src_saved;
        src_saved.reserve(g->n_nodes);
        for (int k = 0; k < g->n_nodes; ++k) {
            src_snapshot s;
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                s.src[j] = g->nodes[k]->src[j];
            }
            src_saved.push_back(s);
        }

        const int64_t t_setup0 = ggml_time_us();
        // T-GPLR-16 probe: unbuffered per-graph execution markers (remove after diagnosis).
        if (getenv("GGML_LS_GRAPH_EXEC_MARKERS") != nullptr) {
            fprintf(stderr, "[ls-exec] g=%d enter\n", i);
            fflush(stderr);
        }
        ggml_backend_sched_reset(sched.get());
        if (!ggml_backend_sched_alloc_graph(sched.get(), g)) {
            ls->error_msg = "layer_streaming_compute: failed to allocate graph " + std::to_string(i) +
                            " (layer " + std::to_string(layer) + ")";
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return GGML_STATUS_ALLOC_FAILED;
        }
        const int64_t t_setup1 = ggml_time_us();
        // T-GPLR-72: CPU clock starts with the compute region, so cpu_us spans
        // exactly the region compute_us measures.
        const int64_t c_setup1 = ls_thread_cpu_us();
        // T-GPLR-16 probe: fine-grained exec markers (remove after diagnosis).
        if (getenv("GGML_LS_GRAPH_EXEC_MARKERS") != nullptr) {
            fprintf(stderr, "[ls-exec] g=%d alloc_done\n", i);
            fflush(stderr);
        }

        // compute (async launch + host block until complete)
        // T-GPLR-16 probe: per-node execution-time hash of the TG router chain
        // (env LLAMA_LS_TG_ROUTER_PROBE). Uses the scheduler eval callback so
        // each tensor is read immediately after its own op completes, before
        // any later op can reuse its allocator block. Bounded to the first 2
        // TG ubatches on layers 0 (host-resident in partial mode) and 20
        // (GPU-resident control in both modes).
        static const bool probe_rt = getenv("LLAMA_LS_TG_ROUTER_PROBE") != nullptr;
        static int64_t rt_token = 0;
        static int64_t rt_cur_token = 0;
        static int rt_cur_layer = -1;
        static std::vector<uint8_t> rt_tmp;
        static auto rt_fnv = [](const uint8_t * p, size_t n) {
            uint64_t h = 1469598103934665603ULL;
            for (size_t b = 0; b < n; ++b) {
                h ^= p[b];
                h *= 1099511628211ULL;
            }
            return h;
        };
        static auto rt_match = [](const char * name) {
            return strncmp(name, "ffn_moe_logits-", 15) == 0 ||
                   strncmp(name, "ffn_moe_probs-", 14) == 0 ||
                   strncmp(name, "ffn_moe_argsort-", 16) == 0 ||
                   strncmp(name, "ffn_moe_topk-", 13) == 0 ||
                   strncmp(name, "ffn_moe_gate-", 13) == 0 ||
                   strncmp(name, "ffn_moe_up-", 11) == 0 ||
                   strncmp(name, "ffn_moe_swiglu-", 15) == 0 ||
                   strncmp(name, "ffn_moe_down-", 13) == 0 ||
                   strncmp(name, "ffn_moe_out-", 12) == 0 ||
                   strncmp(name, "attn_post_norm-", 15) == 0;
        };
        if (probe_rt && is_layer && ls->phase != 0 && rt_token < 3) {
            if (layer == 0) {
                rt_token++;
            }
            rt_cur_token = rt_token;
            rt_cur_layer = layer;
            auto rt_cb = [](struct ggml_tensor * t, bool ask, void * /*user_data*/) -> bool {
                if (ask) {
                    return rt_match(t->name);
                }
                if (!rt_match(t->name)) {
                    return true;
                }
                const size_t nbytes = (size_t) ggml_nbytes(t);
                uint64_t h = 0;
                const uint8_t * p = nullptr;
                if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
                    p = (const uint8_t *) t->data;
                    h = rt_fnv(p, nbytes);
                } else if (nbytes <= (64u << 20)) {
                    if (rt_tmp.capacity() < (64u << 20)) {
                        rt_tmp.reserve(64u << 20);
                    }
                    ggml_backend_tensor_get(t, rt_tmp.data(), 0, nbytes);
                    p = rt_tmp.data();
                    h = rt_fnv(p, nbytes);
                }
                // value sample: first 3 elements + sampled min/max as f32
                double vmin = 0, vmax = 0;
                float v0 = 0, v1 = 0, v2 = 0;
                if (p && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16)) {
                    const int64_t n = ggml_nelements(t);
                    const int64_t step = n > 4096 ? n / 4096 : 1;
                    bool first = true;
                    int got = 0;
                    for (int64_t e = 0; e < n; e += step) {
                        const uint8_t * q = p + e * (t->type == GGML_TYPE_F32 ? 4 : 2);
                        const float val = t->type == GGML_TYPE_F32 ? *(const float *) q
                            : ggml_fp16_to_fp32(*(const ggml_fp16_t *) q);
                        if (first) { vmin = val; vmax = val; first = false; }
                        if (val < vmin) { vmin = val; }
                        if (val > vmax) { vmax = val; }
                        if (got == 0) { v0 = val; } else if (got == 1) { v1 = val; } else if (got == 2) { v2 = val; }
                        got++;
                    }
                }
                fprintf(stderr, "[rt-probe] tok %lld L%02d %-32s op=%-10s buf=%-9s hash=%016llx type=%s v=[%.6g %.6g %.6g] min=%.6g max=%.6g\n",
                    (long long) rt_cur_token, rt_cur_layer, t->name, ggml_op_name(t->op),
                    t->buffer ? ggml_backend_buffer_name(t->buffer) : "none", (unsigned long long) h,
                    ggml_type_name(t->type), v0, v1, v2, vmin, vmax);
                return true;
            };
            ggml_backend_sched_set_eval_callback(sched.get(), rt_cb, nullptr);
        } else if (probe_rt) {
            ggml_backend_sched_set_eval_callback(sched.get(), nullptr, nullptr);
        }
        // T-GPLR-16 probe: unbuffered per-graph execution markers (remove after diagnosis).
        if (getenv("GGML_LS_GRAPH_EXEC_MARKERS") != nullptr) {
            fprintf(stderr, "[ls-exec] g=%d compute_enter\n", i);
            fflush(stderr);
        }
        const auto status = ggml_backend_sched_graph_compute_async(sched.get(), g);
        // T-GPLR-16 probe: unbuffered per-graph execution markers (remove after diagnosis).
        if (getenv("GGML_LS_GRAPH_EXEC_MARKERS") != nullptr) {
            fprintf(stderr, "[ls-exec] g=%d compute_done status=%d\n", i, (int) status);
            fflush(stderr);
        }
        if (status != GGML_STATUS_SUCCESS) {
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            return status;
        }
        ggml_backend_sched_synchronize(sched.get());
        const int64_t t_compute1 = ggml_time_us();
        const int64_t c_compute1 = ls_thread_cpu_us();
        const double layer_cpu_us = (c_setup1 < 0 || c_compute1 < 0)
            ? -1.0 : (double) (c_compute1 - c_setup1);
        // dev=1 iff this layer's weights are in a handoff bundle. Under
        // --handoff-partial N that is exactly the layers at/above the split
        // that are in handoff_layers (0/1 for 38,39 at the 8 GB point).
        const int layer_dev = (ls->handoff_active && ls->handoff_layers.count(layer)) ? 1 : 0;

        // restore the original node src pointers mutated by the split
        for (int k = 0; k < g->n_nodes; ++k) {
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                g->nodes[k]->src[j] = src_saved[k].src[j];
            }
        }

        // T-GPLR-16 probe: value-level hash of the TG mixed-split sandwich
        // (env LLAMA_LS_TG_SANDWICH_PROBE). CPU-side tensors are hashed directly
        // from host memory; CUDA-side inputs are pulled via a blocking read.
        // Bounded to the first probe_tokens TG ubatches to keep runs short.
        static const bool probe_sandwich = getenv("LLAMA_LS_TG_SANDWICH_PROBE") != nullptr;
        static int64_t probe_token_count = 0;
        if (probe_sandwich && is_layer && ls->phase != 0 && probe_token_count < 2) {
            if (layer == 0) {
                probe_token_count++;
                fprintf(stderr, "[ls-probe] TG token %lld (ubatch %llu)\n",
                    (long long) probe_token_count, (unsigned long long) ls->n_ubatches);
            }
            auto fnv = [](const uint8_t * p, size_t n) {
                uint64_t h = 1469598103934665603ULL;
                for (size_t b = 0; b < n; ++b) {
                    h ^= p[b];
                    h *= 1099511628211ULL;
                }
                return h;
            };
            static std::vector<uint8_t> tmp;
            const size_t tmp_cap = 64u * 1024u * 1024u;
            if (tmp.capacity() < tmp_cap) {
                tmp.reserve(tmp_cap);
            }
            auto hash_tensor = [&](ggml_tensor * t) -> uint64_t {
                const size_t nbytes = (size_t) ggml_nbytes(t);
                if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
                    return fnv((const uint8_t *) t->data, nbytes);
                }
                if (nbytes > tmp_cap) {
                    return 0;
                }
                ggml_backend_tensor_get(t, tmp.data(), 0, nbytes);
                return fnv(tmp.data(), nbytes);
            };
            for (int j = 0; j < g->n_leafs; ++j) {
                ggml_tensor * leaf = g->leafs[j];
                if (strstr(leaf->name, "ffn_gate_exps") || strstr(leaf->name, "ffn_up_exps") ||
                    strstr(leaf->name, "ffn_down_exps") || strstr(leaf->name, "ffn_gate_inp.w")) {
                    fprintf(stderr, "[ls-probe] L%02d leaf %-28s buf=%-9s hash=%016llx\n",
                        layer, leaf->name, leaf->buffer ? ggml_backend_buffer_name(leaf->buffer) : "none",
                        (unsigned long long) hash_tensor(leaf));
                }
            }
            for (int k = 0; k < g->n_nodes; ++k) {
                ggml_tensor * node = g->nodes[k];
                const bool match =
                    strncmp(node->name, "ffn_moe_gate-", 13) == 0 ||
                    strncmp(node->name, "ffn_moe_up-", 11) == 0 ||
                    strncmp(node->name, "ffn_moe_swiglu-", 15) == 0 ||
                    strncmp(node->name, "ffn_moe_down-", 13) == 0 ||
                    strncmp(node->name, "attn_post_norm-", 15) == 0 ||
                    strncmp(node->name, "ffn_moe_topk-", 13) == 0 ||
                    strncmp(node->name, "ffn_moe_logits-", 15) == 0 ||
                    strncmp(node->name, "ffn_moe_probs-", 14) == 0 ||
                    strncmp(node->name, "ffn_moe_argsort-", 16) == 0 ||
                    strncmp(node->name, "ffn_moe_weights_norm", 20) == 0 ||
                    strncmp(node->name, "ffn_moe_out-", 12) == 0;
                if (!match) {
                    continue;
                }
                fprintf(stderr, "[ls-probe] L%02d node %-28s op=%-11s buf=%-9s hash=%016llx\n",
                    layer, node->name, ggml_op_name(node->op),
                    node->buffer ? ggml_backend_buffer_name(node->buffer) : "none",
                    (unsigned long long) hash_tensor(node));
            }
        }

        if (is_layer) {
            const double compute_us = (double) (t_compute1 - t_setup1);
            const double setup_us   = (double) (t_setup1 - t_setup0);
            const double wall_us    = (double) (t_compute1 - t_setup0);

            ls->record_consumer_done(layer, backend_cuda);
            ls->add_layer_stats(layer, layer_transfer_us, layer_wait_us, compute_us, setup_us, wall_us, ready_at_consumption);
            // chunk: for batch-major the ubatch index; for layer-major the
            // per-chunk identity (SDD section 13) so the host validator can
            // enforce ascending chunk order per logical layer
            // the per-ubatch path (batch-major PP + layer-major TG) reports the
            // ubatch index as the chunk identity; per-chunk identities (0..C-1)
            // are emitted only by the layer-major controller
            const int64_t chunk = (int64_t) (ls->n_ubatches - 1);
            llama_layer_streaming::tel_layer_compute_params tel_plain;
            tel_plain.cpu_us = layer_cpu_us;
            tel_plain.dev    = layer_dev;
            ls->tel_layer_compute(layer, compute_us, setup_us, ls->slots[layer % ls->n_slots].generation,
                                  chunk, tel_plain);

            // lookahead: after this layer's compute is done, request the next
            // layer that maps to this slot (waits on ev_consumer_done, so the
            // copy overlaps the compute of layers layer+1 .. layer+n_slots-1).
            // In layer-major mode the next layer must NOT be requested until
            // this layer's final chunk completes (the layer-major controller
            // owns slot reuse across chunks); T-Q38LM-01 is bounded to PP and
            // a single selected layer, so no lookahead here.
            // During handoff all weights are resident — no lookahead needed.
            if (!layer_major && !ls->handoff_active) {
                const int next = layer + ls->n_slots;
                if (next < ls->n_layer && should_exec(next)) {
                    if (ls->af) {
                        // T-GPLR-25 residue fix: with the async-feed worker on,
                        // the just-released slot's refill is ENQUEUED on the
                        // worker instead of staged inline on the main thread
                        // (single writer per copy). The worker waits
                        // ev_consumer_done before touching the slot (the
                        // slot-reuse invariant) and reports through
                        // async_feed_submit / async_feed_ready. If the worker
                        // cannot stage it, the main thread's load_layer at
                        // consumption time remains the on-demand fallback.
                        std::lock_guard<std::mutex> afd_lk(ls->af->mtx);
                        if (ls->af->queue.size() < (size_t) ls->af_depth) {
                            ls->af->queue.push_back(next);
                            ls->af->cv.notify_one();
                        }
                    } else {
                        bool c2 = false;
                        if (!ls->load_layer(next, c2)) {
                            ls->failed = true;
                            ls->tel_error(ls->error_msg.c_str());
                            return GGML_STATUS_FAILED;
                        }
                    }
                }
            }
        }
    }

    // drain the copy stream so no copies are pending when the ubatch ends
    ls->ls_stage_drain();

    // expose this set's output references on the context result object so the
    // caller's logit/embedding capture reads the tensors this run produced
    res->t_logits      = set->res->t_logits;
    res->t_embd        = set->res->t_embd;
    res->t_embd_pooled = set->res->t_embd_pooled;
    res->t_h_nextn     = set->res->t_h_nextn;
    res->t_inp_tokens  = set->res->t_inp_tokens;

    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Layer-major PP controller (SDD section 13, T-Q38LM-02)
//
// Drives the full prompt in layer-major order: embed every chunk, then for each
// executed logical layer run every chunk (ascending token-position order) through
// that layer's prebuilt per-chunk sub-graph, then run every chunk's output graph.
// Each chunk's KV/recurrent inputs are rebound from its own memory context (all
// kept alive for the pass), so the per-layer causal state advances chunk 0 then
// chunk 1 exactly as the batch-major schedule does. The four-slot lookahead is
// kept in logical-layer order: before layer L the resident targets are L..L+3,
// the L package is reused across every chunk, and L+4 is requested only after
// L's final consumer completes.
// ---------------------------------------------------------------------------
int llama_context::layer_major_pp(const llama_batch & batch_inp, ggml_status & ret) {
    llama_layer_streaming * ls = layer_streaming;
    if (ls == nullptr) {
        ret = GGML_STATUS_FAILED;
        return -2;
    }
    if (ls->schedule != LLAMA_PP_SCHEDULE_LAYER_MAJOR || ls->phase != 0) {
        ret = GGML_STATUS_FAILED;
        return -2;
    }

    // T-GPLR-36 host ledger (diagnostics only, emitted under --pp-recon): the
    // phase summary's residual (wall - compute - pure DMA) is unattributed. These
    // buckets split the in-sweep wall by NAMED host section so the next design
    // decision is measured, not inferred. Host ggml_time_us() reads only; no
    // execution order, state, routing or arithmetic is affected.
    // T-GPLR-36 follow-up: open the host-memory pair for this pass (recon only;
    // the emitter self-gates). Paired with the pass_end sample below the host
    // ledger.
    ls->tel_host_mem_begin();

    // T-GPLR-45: ONE owner of the per-set patch-cache build (FA list +
    // consumption-classified input list + composition split + recon dump).
    // Called from the chunk-snapshot site BEFORE a set's first snapshot and
    // guarded again in prepare_chunk, so every consumer iterates the same
    // classified member list no matter which site got there first. This also
    // moves the build cost out of total_patch_us (it used to be charged to
    // the first launch's patch bucket); pass-2 patch_us comparisons are
    // unaffected either way, and the dump comment below records why.
    auto ensure_patch_cache = [&](llama_layer_streaming::graph_set * set, size_t set_idx) -> bool {
        if (set == nullptr || set->patch_cache_built) {
            return true;
        }
        auto & cache_fa = set->fa_nodes;
        for (size_t gi = 1; gi + 1 < set->graphs.size(); ++gi) {
            ggml_cgraph * g = set->graphs[gi];
            for (int k = 0; k < g->n_nodes; ++k) {
                if (g->nodes[k]->op == GGML_OP_FLASH_ATTN_EXT) {
                    cache_fa.push_back(g->nodes[k]);
                }
            }
        }
        // build_patch_split classifies the INPUT candidates (kept vs
        // zero-consumer, T-GPLR-45) and builds the per-graph split the dump
        // reports; one walk owns both so they cannot drift.
        // T-GPLR-62 S2: build_patch_split is the owner of the INPUT
        // classification AND of the guard that a row-set companion input is not
        // in the snapshot list. Its refusal must propagate: an unbacked or
        // double-owned input buffer is exactly the case where continuing would
        // produce a plausible wrong answer.
        if (!ls->build_patch_split(*set)) {
            return false;
        }
        set->patch_cache_built = true;
        // T-GPLR-44: report the cache composition once per set (recon only).
        // The per-graph split this dump describes was implemented and A/B
        // tested (commit f53c1bbf0): generated ids diverged from the control
        // at both shapes, because begin_set's cache reset made the launches
        // re-walk an ACCUMULATED ctx after the T-GPLR-43 pass-2 rebuild
        // (7 -> 14 doubled members in the dump). Reverted per the task's
        // falsifier; the dump stays so the finding stays measurable. T-GPLR-45
        // reuses the walk for the zero-consumer SET filter (not per-graph
        // scoping), which is the narrowed-restore lever that survived.
        ls->tel_patch_cache(*set, set_idx);
        return true;
    };

    const int64_t t_led_fn0 = ggml_time_us();
    double led_embed = 0.0, led_preload = 0.0, led_sweep = 0.0, led_output = 0.0;
    double led_patch = 0.0, led_setup = 0.0, led_compute = 0.0, led_graph_call = 0.0, led_slot = 0.0;
    int64_t led_launches = 0;
    // T-GPLR-39: sub-buckets inside what T-GPLR-36 called `other` (the sweep
    // wall no named bucket claims). At fixed launch count the T-GPLR-38 sweep
    // measured this bucket growing with UBATCH SIZE (237.9 -> 962.1 ms from
    // ubatch 512 -> 2048), so the residue must contain a per-sample cost that
    // scales with per-chunk tensor sizes. Named candidates, each timed where
    // it happens:
    //   led_route  - tel_route: device->host probs readback + host sort/hash
    //                (fires for chunk 0 and the last chunk of every layer;
    //                the readback and sort both scale with n_tok = ubatch)
    //   led_tel    - the remaining sampled telemetry writes in the chunk loop
    //                (tel_layer_compute etc.: per-launch, NOT ubatch-scaled)
    //   led_loop   - everything else the per-(layer,chunk) and per-layer loop
    //                bodies spend that no other bucket claims (per-launch)
    double led_route = 0.0, led_tel = 0.0;

    const auto & hparams = model.hparams;
    const int64_t n_embd = hparams.n_embd_inp();
    const bool    output_all = cparams.embeddings;
    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;
    const int64_t n_chunks = ls->n_chunks;
    // T-GPLR-41: identify which --pp-repeat pass this controller invocation is
    // (1-based). Monotonic across the process: the runner invokes layer_major_pp
    // exactly once per prefill pass, so the counter equals the pass number.
    const int pass_idx41 = ++ls->pp_mctx_pass_ctr;
    const llm_graph_type gtype = (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP)
        ? LLM_GRAPH_TYPE_DECODER_MTP : LLM_GRAPH_TYPE_DEFAULT;

    if (n_chunks <= 0 || (int64_t) ls->chunks.size() != n_chunks ||
        ls->chunks.back().token_start + ls->chunks.back().n_tokens != batch_inp.n_tokens) {
        ret = GGML_STATUS_FAILED;
        return -2;
    }

    // Create + apply each chunk's memory context and keep every one alive for the
    // whole pass. The per-chunk KV cells and recurrent s_copy/head bindings are
    // established by these applies; the per-chunk prebuilt sets were built against
    // the identical applies (SDD: deterministic cell mapping), so rebinding the
    // inputs from these live contexts reproduces the prebuild shapes exactly.
    std::vector<llama_memory_context_ptr> chunk_mctx(n_chunks);
    std::vector<const llama_ubatch *>     chunk_ubatch(n_chunks);
    std::vector<llama_layer_streaming::graph_set *> sets(n_chunks);
    std::vector<std::vector<uint8_t>> chunk_input_snap(n_chunks); // T-Q38PW-00 probe snapshots
    for (int64_t c = 0; c < n_chunks; ++c) {
        // T-Q38CTX-01: build the chunk from its immutable descriptor (real
        // width + token_start) so the final tail chunk is a real 1..ubatch
        // token chunk, never a padded one.
        const int n_c = (int) ls->chunks[c].n_tokens;
        const int64_t t0_c = ls->chunks[c].token_start;
        llama_batch batch = llama_batch_init(n_c, 0, 1);
        for (int i = 0; i < n_c; ++i) {
            const int idx = (int) (t0_c + i);
            batch.token[i]    = batch_inp.token[idx];
            batch.pos[i]      = t0_c + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = batch_inp.logits ? batch_inp.logits[idx] : false;
        }
        batch.n_tokens = n_c;

        llama_batch_allocr vballoc(hparams.n_pos_per_embd());
        if (!vballoc.init(batch, model.vocab, memory.get(), n_embd, n_seq_max, output_all)) {
            llama_batch_free(batch);
            ret = GGML_STATUS_FAILED;
            return -2;
        }
        auto vmctx = memory->init_batch(vballoc, n_c, output_all);
        if (!vmctx || vmctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            llama_batch_free(batch);
            ret = GGML_STATUS_FAILED;
            return -2;
        }
        if (!vmctx->apply()) {
            llama_batch_free(batch);
            ret = GGML_STATUS_FAILED;
            return -2;
        }

        // The rec/KV head, rs_z, and s_copy are read from the SHARED memory
        // state (get_head/get_rs_z return the current mem value, not a
        // per-mctx snapshot). So the per-chunk inputs MUST be rebound + written
        // immediately after this chunk's apply, while this chunk's state is
        // current. Re-reading them later (after a later chunk's apply) would
        // observe the wrong rs_z/head.
        const auto * mh = static_cast<const llama_memory_hybrid_context *>(vmctx.get());
        const int32_t rs_z = mh ? mh->get_recr()->get_rs_z() : -1;
        int32_t n_out = 0;
        const llama_ubatch & vub = vmctx->get_ubatch();
        for (uint32_t i = 0; i < (uint32_t) vub.n_tokens; ++i) {
            n_out += (int32_t) (vub.output[i] != 0);
        }
        if (ls->pw_probe && c > 0) {
            // T-Q38PW-00: later chunks reuse the per-width continuing set
            // (chunk_id=1, rs_z=-1). Chunk 0 uses the fresh set (chunk_id=0).
            // Both were prebuilt; the first type of each width class is
            // reused for all same-width same-rs_z chunks.
            auto * cont_set = ls->find_set(0, n_c, rs_z, n_out, 1);
            if (cont_set && cont_set->built) {
                sets[c] = cont_set;
            } else {
                // no explicit continuing set (shouldn't happen for >1 chunk)
                sets[c] = sets[c - 1];
            }
        } else {
            sets[c] = ls->find_set(0, n_c, rs_z, n_out, (int32_t) c);
        }
        if (sets[c] == nullptr || !sets[c]->built) {
            ls->error_msg = "layer_major_pp: missing/unbuilt per-chunk set " + std::to_string(c) +
                            " (token_start " + std::to_string(t0_c) + ", n_tokens " + std::to_string(n_c) +
                            ", rs_z " + std::to_string(rs_z) + ", n_out " + std::to_string(n_out) + ")";
            ls->failed = true;
            ls->tel_error(ls->error_msg.c_str());
            llama_batch_free(batch);
            ret = GGML_STATUS_FAILED;
            return -2;
        }
        // rebind the inputs to this chunk's live context (shape guard) and write
        // the ubatch values now, while this chunk's shared state is current
        {
            n_outputs = n_out; // graph_params reads the context n_outputs member
            // T-Q38PW-00: in probe mode, rebind chunk 0's set inputs to
            // every chunk's ubatch; the tensor data pointers are the same
            // but the ubatch values differ per chunk.
            auto * bind_set = sets[c];
            const auto gp = graph_params(bind_set->res, vub, vmctx.get(), gtype);
            for (size_t ii = 0; ii < bind_set->res->inputs.size(); ++ii) {
                if (!bind_set->res->inputs[ii]->can_reuse(gp)) {
                    ls->error_msg = "layer_major_pp: set " + std::to_string(c) +
                                    " input[" + std::to_string(ii) + "] shape mismatch at bind" +
                                    " (n_tokens " + std::to_string(n_c) + ", n_out " + std::to_string(n_out) + ")";
                    ls->failed = true;
                    ls->tel_error(ls->error_msg.c_str());
                    llama_batch_free(batch);
                    ret = GGML_STATUS_FAILED;
                    return -2;
                }
            }
            bind_set->res->set_inputs(&vub);
            // T-Q38PW-00: probe-mode snapshot: the shared set's input tensors
            // will be overwritten by the next chunk's apply, so copy this
            // chunk's input bytes aside now.
            if (ls->pw_probe) {
                // T-GPLR-45: the patch cache - and with it the zero-consumer
                // classification of the INPUT candidates - must exist BEFORE
                // the snapshot: snapshot capture, launch restore and the
                // composition split all iterate the same kept-member list, so
                // the classification cannot be allowed to arrive late.
                if (!ensure_patch_cache(bind_set, (size_t) c)) {
                    ls->failed = true;
                    return GGML_STATUS_FAILED;
                }
                std::vector<uint8_t> snap;
                for (ggml_tensor * t : bind_set->input_tensors) {
                    const size_t nb = ggml_nbytes(t);
                    snap.insert(snap.end(), (const uint8_t *) t->data, (const uint8_t *) t->data + nb);
                }
                chunk_input_snap[c] = std::move(snap);
            }
        }
        chunk_mctx[c] = std::move(vmctx);
        chunk_ubatch[c] = &chunk_mctx[c]->get_ubatch();
        llama_batch_free(batch);

        // T-GPLR-41 (diagnostics only, gated by --pp-mctx-recon): emit the
        // chunk's post-apply lifecycle facts BEFORE the sweep runs, so a
        // crashing pass still leaves its complete picture in the trace. The
        // events are read-only host work (no execution-order change) and are
        // unknown to both checkers, which ignore unrecognized event types.
        if (ls->pp_mctx_recon) {
            const llama_memory_hybrid_context * mh41 =
                static_cast<const llama_memory_hybrid_context *>(chunk_mctx[c].get());
            const llama_kv_cache_context * attn41 = mh41 ? mh41->get_attn() : nullptr;
            const uint32_t live_n_kv41 = attn41 ? attn41->get_n_kv() : 0;
            ls->tel_mctx_lifecycle(pass_idx41, c, n_chunks, n_c, live_n_kv41, rs_z,
                                   sets[c]->chunk_id, sets[c]->rs_z, sets[c]->n_outputs);
            // per-exec-layer: the set's baked KV-view shapes and the KV
            // storage addresses the views were built against
            for (size_t gi41 = 0; gi41 < sets[c]->graph_entries.size(); ++gi41) {
                const auto & ge41 = sets[c]->graph_entries[gi41];
                const int il41 = ge41.logical_layer;
                if (il41 < 0) {
                    continue;
                }
                // mirror should_exec (defined below the mctx loop)
                bool exec_ok41 = ls->exec_layers.empty();
                if (!exec_ok41) {
                    for (int e41 : ls->exec_layers) {
                        if (e41 == il41) { exec_ok41 = true; break; }
                    }
                }
                if (!exec_ok41) {
                    continue;
                }
                ggml_cgraph * lg41 = (ge41.slot >= 0 && ge41.slot < (int) sets[c]->graphs.size())
                    ? sets[c]->graphs[ge41.slot] : nullptr;
                if (lg41 == nullptr) {
                    continue;
                }
                long long k_ne2_41 = -1, v_ne2_41 = -1;
                const void * kst41 = nullptr, * vst41 = nullptr;
                long long ksrc41 = 0, vsrc41 = 0;
                for (int ni41 = 0; ni41 < lg41->n_nodes; ++ni41) {
                    ggml_tensor * n41 = lg41->nodes[ni41];
                    if (n41->op != GGML_OP_VIEW || n41->src[0] == nullptr) {
                        continue;
                    }
                    const char * nm41 = ggml_get_name(n41->src[0]);
                    if (nm41 == nullptr) {
                        continue;
                    }
                    const void * n_data41 = (const void *) n41->data;
                    if (strncmp(nm41, "cache_k_l", 9) == 0 && k_ne2_41 < 0) {
                        k_ne2_41 = (long long) n41->ne[2];
                        kst41 = (const void *) n41->src[0]->data;
                        ksrc41 = (long long)(uintptr_t) n_data41;
                    } else if (strncmp(nm41, "cache_v_l", 9) == 0 && v_ne2_41 < 0) {
                        v_ne2_41 = (long long) n41->ne[2];
                        vst41 = (const void *) n41->src[0]->data;
                        vsrc41 = (long long)(uintptr_t) n_data41;
                    }
                }
                ls->tel_mctx_kvview(pass_idx41, c, il41, k_ne2_41, v_ne2_41,
                                    kst41, vst41, ksrc41, vsrc41);
            }
        }
    }

    // locate the CUDA compute backend (for copy-event ordering)
    ggml_backend_t backend_cuda = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched.get()); ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched.get(), i);
        if (backend_is_cuda(b)) {
            backend_cuda = b;
            break;
        }
    }
    if (backend_cuda == nullptr) {
        ret = GGML_STATUS_FAILED;
        return -2;
    }
    // T-GPLR-35 recon finding: layer-major PP executes graphs directly against
    // the scheduler and never enters layer_streaming_compute, which was the
    // only place the CUDA compute backend was published. Publish it here too,
    // so that (a) the recon can measure copy-vs-compute timing and (b) the
    // aux-stream staging path can engage at all in the primary workload. A run
    // that asked for overlap but has no aux stream still fails loudly here.
    ls->backend_compute = backend_cuda;
    if (ls->pp_copy_overlap && !ggml_backend_cuda_aux_stream_ok(backend_cuda)) {
        ls->error_msg = "layer-major PP: copy-overlap requested but the compute backend's CUDA context has no auxiliary stream";
        ret = GGML_STATUS_FAILED;
        return -2;
    }

    auto should_exec = [&](int il) -> bool {
        if (ls->exec_layers.empty()) return true;
        for (int e : ls->exec_layers) if (e == il) return true;
        return false;
    };

    // execute one prebuilt sub-graph of a set against a chunk's live mctx/ubatch:
    // rebind + shape-guard the inputs, write the ubatch values, then sched
    // reset/alloc/compute/sync. The node src-pointer snapshot mirrors the
    // batch-major executor (alloc_graph may rewire cross-backend copies).
    // execute one prebuilt sub-graph of a set. The inputs for this set's chunk
    // were already rebound + written in the mctx loop above (while that
    // chunk's shared rec/KV state was current), so here we only reset/alloc/
    // compute/sync the scheduler for this graph. The node src-pointer snapshot
    // mirrors the batch-major executor (alloc_graph may rewire cross-backend
    // copies and we must leave the persistent graph pristine).
    auto run_graph = [&](llama_layer_streaming::graph_set & set, size_t gi, int c,
                         double & compute_us, double & setup_us) -> ggml_status {
        (void) c;
        ggml_cgraph * g = set.graphs[gi];

        struct src_snapshot { ggml_tensor * src[GGML_MAX_SRC]; };
        std::vector<src_snapshot> src_saved;
        src_saved.reserve(g->n_nodes);
        for (int k = 0; k < g->n_nodes; ++k) {
            src_snapshot s;
            for (int j = 0; j < GGML_MAX_SRC; ++j) s.src[j] = g->nodes[k]->src[j];
            src_saved.push_back(s);
        }

        const int64_t t0 = ggml_time_us();
        // T-GPLR-16 probe: unbuffered per-graph execution markers (remove after diagnosis).
        static const bool graph_exec_markers = getenv("GGML_LS_GRAPH_EXEC_MARKERS") != nullptr;
        if (graph_exec_markers) {
            fprintf(stderr, "[ls-exec] g=%zu enter\n", gi);
            fflush(stderr);
        }
        ggml_backend_sched_reset(sched.get());
        if (!ggml_backend_sched_alloc_graph(sched.get(), g)) {
            ls->error_msg = "layer_major_pp: failed to allocate graph " + std::to_string(gi) + " (chunk " + std::to_string(c) + ")";
            ls->failed = true;
            return GGML_STATUS_ALLOC_FAILED;
        }
        const int64_t t1 = ggml_time_us();
        const auto status = ggml_backend_sched_graph_compute_async(sched.get(), g);
        if (graph_exec_markers) {
            fprintf(stderr, "[ls-exec] g=%zu compute_done status=%d\n", gi, (int) status);
            fflush(stderr);
        }
        if (status != GGML_STATUS_SUCCESS) {
            ls->failed = true;
            return status;
        }
        ggml_backend_sched_synchronize(sched.get());
        const int64_t t2 = ggml_time_us();
        for (int k = 0; k < g->n_nodes; ++k) {
            for (int j = 0; j < GGML_MAX_SRC; ++j) g->nodes[k]->src[j] = src_saved[k].src[j];
        }
        compute_us = (double) (t2 - t1);
        setup_us   = (double) (t1 - t0);
        return GGML_STATUS_SUCCESS;
    };

    sched_reserve();

    // T-Q38PW-00: cumulative slab offset per chunk (bytes into each half)
    std::vector<size_t> slab_off_c(n_chunks, 0);
    void * slab_base = nullptr;
    size_t slab_half = 0;
    if (ls->pw_probe) {
        const size_t slab_elt = ggml_type_size(ls->act_slab_dtype);
        slab_base = ggml_backend_buffer_get_base(ls->buf_slab.get());
        slab_half = (size_t) ls->n_embd_io * (size_t) ls->pp_tokens * slab_elt;
        size_t off = 0;
        for (int c = 0; c < n_chunks; ++c) {
            slab_off_c[c] = off;
            off += (size_t) ls->chunks[c].n_tokens * (size_t) ls->n_embd_io * slab_elt;
        }
    }

    // T-Q38PW-00: per-chunk prepare. In probe mode the per-width sets were
    // prebuilt with slab slices at the widths' first-chunk positions (chunk 0
    // for fresh, chunk 1 for continuing). When a set is reused for a different
    // chunk, this patches act data pointers, restores input snapshots, and
    // rewrites the causal window. ggml's CUDA graph exec update diffs node
    // properties against the captured copy and re-instantiates on update failure.
    auto prepare_chunk = [&](int c) -> bool {
        // pw_probe off: there is nothing to patch, which is a success, not a
        // refusal. (This is the documented no-pw-probe behaviour: no act
        // repointing, no snapshot restore, and no patch_cache dump.)
        if (!ls->pw_probe) return true;
        const int64_t t0_patch = ggml_time_us();

        // 1. act slab slices: each per-width set has act tensors pointing
        // into the slab at fixed offsets (fresh at 0, continuing at chunk-1
        // offset). When a set serves a chunk with a different slab position,
        // repoint its act data to that chunk's slab slice.
        ggml_tensor * act0 = sets[c]->act[0];
        ggml_tensor * act1 = sets[c]->act[1];
        const char * slab_bytes = (const char *) slab_base;
        if ((const char *) act0->data != slab_bytes + slab_off_c[c]) {
            act0->data = (char *) slab_base + slab_off_c[c];
        }
        if ((const char *) act1->data != slab_bytes + slab_half + slab_off_c[c]) {
            act1->data = (char *) slab_base + slab_half + slab_off_c[c];
        }

        // T-Q38FA-04 + T-GPLR-45: the per-set patch cache (FA list,
        // consumption-classified input list, split, dump) is built by the ONE
        // owner lambda; the snapshot site classifies each set before its first
        // snapshot, so this is a guard for any set that arrives here
        // unclassified. Cached pointers point into this set's graphs, so they
        // are only valid while those graphs are: begin_set() clears them
        // together, which is what makes the guard safe. Measured on a
        // --pp-repeat 2 arm, the PP phase does NOT rebuild a built set
        // (graph_builds == 0 in both passes), so in practice the build
        // happens once per run per set - the T-GPLR-43 per-pass prebuild does
        // not rebuild these pw sets.
        if (!ensure_patch_cache(sets[c], (size_t) c)) {
            ls->failed = true;
            return false;
        }

        // 2. restore this chunk's input values from the snapshot taken during
        // the mctx loop (where the shared rec/KV state was current)
        if (!chunk_input_snap[c].empty()) {
            size_t snap_off = 0;
            for (ggml_tensor * t : sets[c]->input_tensors) {
                const size_t nb = ggml_nbytes(t);
                memcpy(t->data, chunk_input_snap[c].data() + snap_off, nb);
                snap_off += nb;
            }
        }

        // 3. causal window + KV bound: patch the cached FA nodes directly.
        // T-Q38FA-01: shrink K/V ne[1] to the valid window before set_causal.
        const int64_t q_pos_base = ls->chunks[c].token_start;
        const int64_t n_kv_valid = q_pos_base + ls->chunks[c].n_tokens;
        for (ggml_tensor * t : sets[c]->fa_nodes) {
            if (t->src[1]) { t->src[1]->ne[1] = n_kv_valid; }
            if (t->src[2]) { t->src[2]->ne[1] = n_kv_valid; }
            ggml_flash_attn_ext_set_causal(t, (int32_t) q_pos_base, (int32_t) n_kv_valid);
        }

        ls->total_patch_us += (double)(ggml_time_us() - t0_patch);
        return true;
    };

    // 1. embed every chunk (writes slab A[c]); no slot work
    const int64_t t_led_embed0 = ggml_time_us();
    for (int c = 0; c < n_chunks; ++c) {
        const int64_t t_p0 = ggml_time_us();
        if (!prepare_chunk(c)) {
            ret = GGML_STATUS_FAILED;
            return -2;
        }
        const int64_t t_p1 = ggml_time_us();
        double cu = 0.0, su = 0.0;
        const int64_t t_g0 = ggml_time_us();
        const auto st = run_graph(*sets[c], 0, c, cu, su);
        const int64_t t_g1 = ggml_time_us();
        led_patch      += (double) (t_p1 - t_p0);
        led_setup      += su;
        led_compute    += cu;
        led_graph_call += (double) (t_g1 - t_g0) - su - cu;
        led_launches++;
        ls->tel_embed_chunk(c, ls->chunks[c].n_tokens, su, cu, (double) (t_g1 - t_p0));
        if (st != GGML_STATUS_SUCCESS) {
            ret = st;
            return -2;
        }
        ls->n_ubatches++;
    }
    led_embed = (double) (ggml_time_us() - t_led_embed0);

    // 2. preload the first n_slots selected layers (overlap window)
    const int64_t t_led_pre0 = ggml_time_us();
    {
        int preloaded = 0;
        int last_pre = -1;
        for (int il = 0; il < ls->n_layer && preloaded < ls->n_slots; ++il) {
            if (!should_exec(il)) continue;
            bool copied = false;
            if (!ls->load_layer(il, copied)) {
                ls->tel_error(ls->error_msg.c_str());
                ret = GGML_STATUS_FAILED;
                return -2;
            }
            last_pre = il;
            preloaded++;
        }
        // T-GPLR-25: seed the async-feed queue with the layer after the last
        // preload so the worker starts staging as soon as a slot frees. With
        // the claim-on-stage protocol the first seed request is REFUSED by
        // the worker (its slot is still claimed by an unconsumed preload
        // layer — the safe outcome); worker staging effectively begins at the
        // first post-consumer_done enqueue.
        if (ls->af && last_pre >= 0) {
            const int nxt = ls->af_next_layer(last_pre);
            if (nxt >= 0) {
                std::lock_guard<std::mutex> lk(ls->af->mtx);
                ls->af->queue.push_back(nxt);
                ls->af->cv.notify_one();
            }
        }
    }
    led_preload = (double) (ggml_time_us() - t_led_pre0);

    // 3. layers on the outside, chunks on the inside (ascending chunk order)
    const int64_t t_led_sweep0 = ggml_time_us();
    for (int il = 0; il < ls->n_layer; ++il) {
        if (!should_exec(il)) continue;

        const int64_t t_slot0 = ggml_time_us();
        const size_t req_bytes = ls->package_bytes(il);
        const int64_t t_slotreq0 = ggml_time_us();
        ls->tel_slot_request(il, ls->phase, req_bytes);
        const double slotreq_span = (double) (ggml_time_us() - t_slotreq0);

        bool copied = false;
        if (!ls->load_layer(il, copied)) {
            ls->tel_error(ls->error_msg.c_str());
            ret = GGML_STATUS_FAILED;
            return -2;
        }
        double wait_us = 0.0, host_submit_us = 0.0, ready_latency_us = 0.0;
        if (!ls->wait_slot_ready(il, wait_us, host_submit_us, ready_latency_us)) {
            ls->tel_error(ls->error_msg.c_str());
            ret = GGML_STATUS_FAILED;
            return -2;
        }
        // T-GPLR-25: with the async-feed worker on, the NEXT executed layer's
        // copy is enqueued as soon as this layer's copy has been consumed —
        // the worker submits it into the just-released next slot while every
        // chunk of this layer computes. enqueue is a no-op when off.
        {
            const int nxt = ls->af_next_layer(il);
            if (nxt >= 0 && ls->af) {
                std::lock_guard<std::mutex> lk(ls->af->mtx);
                if (ls->af->queue.size() < (size_t) ls->af_depth) {
                    ls->af->queue.push_back(nxt);
                    ls->af->cv.notify_one();
                }
            }
        }
        const bool ready_at_consumption = (wait_us < 100.0);
        ggml_backend_event_wait(backend_cuda, (ggml_backend_event_t) ls->slots[il % ls->n_slots].ev_copy_ready);
        const int64_t t_slot1 = ggml_time_us();

        // run every chunk through this layer before the next layer starts
        double layer_compute_us = 0.0, layer_setup_us = 0.0;
        for (int c = 0; c < n_chunks; ++c) {
            const int64_t t_p0 = ggml_time_us();
            if (!prepare_chunk(c)) {
                ret = GGML_STATUS_FAILED;
                return -2;
            }
            const int64_t t_p1 = ggml_time_us();
            double cu = 0.0, su = 0.0;
            const int64_t t_g0 = ggml_time_us();
            const auto st = run_graph(*sets[c], 1 + il, c, cu, su);
            const int64_t t_g1 = ggml_time_us();
            led_patch      += (double) (t_p1 - t_p0);
            led_setup      += su;
            led_compute    += cu;
            led_graph_call += (double) (t_g1 - t_g0) - su - cu;
            led_launches++;
            if (st != GGML_STATUS_SUCCESS) {
                ret = st;
                return -2;
            }
            layer_compute_us += cu;
            layer_setup_us   += su;
            if (c == 0 || c + 1 == n_chunks || c % ls->telemetry_chunk_stride == 0) {
                const int64_t t_tel0 = ggml_time_us();
                ls->tel_layer_compute(il, cu, su, ls->slots[il % ls->n_slots].generation, c);
                // T-GPLR-02: observational route event for MoE layers. The
                // captured route marks which layers have an expert router;
                // the values are read from the graph's ARGSORT node dst.
                double route_span = 0.0;
                // T-GPLR-39: the route event is a DIAGNOSTIC (device readback +
                // host sort/hash per sampled layer). When the runner turned it
                // off, skip the whole readback; the row-set ROUTING path
                // (read_router_ids) is functional and unaffected.
                if (ls->route_telemetry && sets[c]->routes.count(il)) {
                    const int64_t t_route0 = ggml_time_us();
                    ggml_cgraph * lg = (1 + il < (int) sets[c]->graphs.size()) ? sets[c]->graphs[1 + il] : nullptr;
                    ls->tel_route(il, c, lg);
                    route_span = (double) (ggml_time_us() - t_route0);
                    led_route += route_span;
                }
                // T-GPLR-39: led_tel = the sampled-telemetry span minus the
                // route span (route is its own bucket) so they never
                // double-count inside `other`.
                led_tel += (double) (ggml_time_us() - t_tel0) - route_span;
            }
        }

        // this layer's final consumer (every chunk) is done; record it so the
        // slot may be reused, then request the lookahead layer for this slot
        const int64_t t_slot2 = ggml_time_us();
        led_slot += (double) (t_slot1 - t_slot0);
        // T-GPLR-39: the slot telemetry write inside the slot span, named so
        // `other` does not absorb it (per-layer, not ubatch-scaled).
        led_tel += slotreq_span;
        ls->record_consumer_done(il, backend_cuda);
        ls->add_layer_stats(il, host_submit_us + ready_latency_us, wait_us, layer_compute_us, layer_setup_us,
                            layer_compute_us + layer_setup_us, ready_at_consumption);
        const int next = il + ls->n_slots;
        if (next < ls->n_layer && should_exec(next)) {
            if (ls->af) {
                // T-GPLR-25 residue fix: single writer per copy. With the
                // worker on, the just-freed slot's refill is enqueued (the
                // worker refuses until record_consumer_done has freed the
                // slot, which happened just above); the consumption-time
                // load_layer remains the on-demand fallback when the worker
                // refuses or the depth-1 queue was busy.
                std::lock_guard<std::mutex> afd_lk(ls->af->mtx);
                if (ls->af->queue.size() < (size_t) ls->af_depth) {
                    ls->af->queue.push_back(next);
                    ls->af->cv.notify_one();
                }
            } else {
                bool c2 = false;
                if (!ls->load_layer(next, c2)) {
                    ls->tel_error(ls->error_msg.c_str());
                    ret = GGML_STATUS_FAILED;
                    return -2;
                }
            }
        }
        led_slot += (double) (ggml_time_us() - t_slot2);
    }
    led_sweep = (double) (ggml_time_us() - t_led_sweep0);

    // 4. output graph for every chunk (final norm + LM head; no slot work)
    const int64_t t_led_out0 = ggml_time_us();
    for (int c = 0; c < n_chunks; ++c) {
        const int64_t t_p0 = ggml_time_us();
        prepare_chunk(c);
        const int64_t t_p1 = ggml_time_us();
        double cu = 0.0, su = 0.0;
        const int64_t t_g0 = ggml_time_us();
        const auto st = run_graph(*sets[c], 1 + ls->n_layer, c, cu, su);
        const int64_t t_g1 = ggml_time_us();
        led_patch      += (double) (t_p1 - t_p0);
        led_setup      += su;
        led_compute    += cu;
        led_graph_call += (double) (t_g1 - t_g0) - su - cu;
        led_launches++;
        if (st != GGML_STATUS_SUCCESS) {
            ret = st;
            return -2;
        }
    }
    led_output = (double) (ggml_time_us() - t_led_out0);

    // drain the copy stream so no copies are pending when the pass ends
    ls->ls_stage_drain();

    // expose the last chunk's output references on the context result so a
    // subsequent TG step (T-Q38LM-04) reads consistent tensors
    {
        auto * last = sets[n_chunks - 1];
        gf_res_prev->t_logits      = last->res->t_logits;
        gf_res_prev->t_embd        = last->res->t_embd;
        gf_res_prev->t_embd_pooled = last->res->t_embd_pooled;    gf_res_prev->t_h_nextn     = last->res->t_h_nextn;
    gf_res_prev->t_inp_tokens  = last->res->t_inp_tokens;
    }

    // T-GPLR-36: emit the host ledger once per pass (recon only). `other_us` is
    // the sweep wall that no named bucket claims: loop overhead, telemetry
    // writes, and anything the buckets do not cover. All buckets are host wall
    // time, so compute + setup + graph_call + patch + slot + other close on sweep.
    if (ls->pp_recon && ls->tel_out) {
        const int64_t t_led_fn1 = ggml_time_us();
        const double led_other = led_sweep - (led_patch + led_setup + led_compute + led_graph_call + led_slot);
        // T-GPLR-39: split `other` into its named candidates. route+tel are
        // timed in place (subtracted from nothing else); loop = the residue.
        // Closure: other == route + tel + loop, by construction of this split.
        const double led_loop = led_other - (led_route + led_tel);
        tel_write(ls->tel_out, "{\"event\":\"host_ledger\",\"run_id\":%llu,"
                  "\"fn_us\":%.1f,\"embed_us\":%.1f,\"preload_us\":%.1f,\"sweep_us\":%.1f,\"output_us\":%.1f,"
                  "\"patch_us\":%.1f,\"setup_us\":%.1f,\"compute_us\":%.1f,\"graph_call_us\":%.1f,"
                  "\"slot_us\":%.1f,\"other_us\":%.1f,\"route_us\":%.1f,\"tel_us\":%.1f,\"loop_us\":%.1f,"
                  "\"launches\":%lld,\"t_us\":%lld}",
                  (unsigned long long) ls->run_id,
                  (double) (t_led_fn1 - t_led_fn0), led_embed, led_preload, led_sweep, led_output,
                  led_patch, led_setup, led_compute, led_graph_call, led_slot, led_other,
                  led_route, led_tel, led_loop,
                  (long long) led_launches, (long long) ggml_time_us());
    }

    // T-GPLR-36 follow-up: close the pass's host-memory pair. Outside the ledger
    // gate by design -- the emitter owns its own --pp-recon check, so the two
    // gates cannot drift apart.
    ls->tel_host_mem_end();

    ret = GGML_STATUS_SUCCESS;
    return 0;
}

// ---------------------------------------------------------------------------
// stats / telemetry
// ---------------------------------------------------------------------------

void llama_layer_streaming::add_layer_stats(int il, double transfer_us, double wait_us, double compute_us, double setup_us, double wall_us, bool ready_at_consumption) {
    if (il < 0 || il >= n_layer) {
        return;
    }
    layer_stats & st = stats[il];
    st.n++;
    st.transfer_us += transfer_us;
    st.wait_us     += wait_us;
    st.compute_us  += compute_us;
    st.setup_us    += setup_us;
    st.wall_us     += wall_us;
    if (transfer_us > 0) {
        st.transfer_min = std::min(st.transfer_min, transfer_us);
        st.transfer_max = std::max(st.transfer_max, transfer_us);
    }
    st.wait_min = std::min(st.wait_min, wait_us);
    st.wait_max = std::max(st.wait_max, wait_us);
    if (compute_us > 0) {
        st.compute_min = std::min(st.compute_min, compute_us);
        st.compute_max = std::max(st.compute_max, compute_us);
    }
    if (ready_at_consumption) {
        st.n_ready_at_consumption++;
    } else {
        st.n_wait_at_consumption++;
    }
}

// T-GPLR-05: single owner for the run_start identity head (event/run_id/t0_us)
// shared by the streaming scheduler emitter and the off-run audit emitter, so
// the two cannot drift apart on identity fields. Callers must emit the head
// and their payload in ONE tel_write call (tel_write terminates the line).
static std::string tel_run_start_head(uint64_t run_id, uint64_t t0_us) {
    char buf[96];
    snprintf(buf, sizeof(buf), "{\"event\":\"run_start\",\"run_id\":%llu,\"t0_us\":%lld,",
             (unsigned long long) run_id, (long long) t0_us);
    return std::string(buf);
}    void llama_layer_streaming::tel_run_start(const llama_model & model, size_t static_vram, size_t slot_vram, size_t free_vram, size_t total_vram,
                                          const char * schedule, const char * copy_source, const char * kv_cache, size_t act_slab_bytes,
                                          int64_t pp_tokens_, const std::vector<llama_ls_chunk> & chunks_,                                          ggml_type act_slab_dtype_, int64_t ctx_size_,
                                          const char * handoff_mode_, bool rowset_on_,
                                          int pp_prebuild_per_pass_, const char * pp_prebuild_per_pass_source_,
                                          const char * copy_source_source_,
                                          int route_telemetry_, const char * route_telemetry_source_,
                                          int tg_logit_margin_, int tg_tokens_) {
    // T-GPLR-54 A2 (wiring fix, A3 session 2026-09-22): the emitter gates on
    // the MEMBER, not on this setter's parameter - recording the depth in the
    // run_start string alone left the member at 0 and the trace silent while
    // run_start claimed the readout was on. Store the resolved depth; -1 (a
    // legacy caller that never resolves the flag) keeps the member off.
    tg_logit_margin = tg_logit_margin_ > 0 ? tg_logit_margin_ : 0;
    // T-Q38CTX-01 chunk manifest: the immutable per-run chunk descriptors the
    // host checker uses to cross-check act_bind chunk ids and widths. Batch-
    // major (or empty) emits 0 / an empty list.
    std::string chunks_str;
    if (!chunks_.empty()) {
        chunks_str = ",\"pp_tokens\":" + std::to_string(pp_tokens_) +
                     ",\"pp_n_chunks\":" + std::to_string(chunks_.size()) +
                     ",\"pp_chunks\":[";
        for (size_t c = 0; c < chunks_.size(); ++c) {
            if (c > 0) {
                chunks_str += ",";
            }
            chunks_str += "\"" + std::to_string(chunks_[c].token_start) + ":" + std::to_string(chunks_[c].n_tokens) + "\"";
        }
        chunks_str += "]";
    }
    // T-GPLR-04: explicit context capacity, separate from the PP token count.
    const std::string ctx_str =
        ctx_size_ > 0 ? ",\"ctx_size\":" + std::to_string(ctx_size_) : "";
    // T-GPLR-12: the auditable handoff decision (auto-on | auto-off | on | off).
    const std::string ho_mode_str = std::string(",\"handoff_mode\":\"") +
        (handoff_mode_ && handoff_mode_[0] ? handoff_mode_ : "off") + "\"";
    // T-GPLR-13: moe-row-set opt-in marker. Absent entirely when off so the
    // default-off immunity check stays a negative assertion (schema section 2).
    const std::string rowset_str =
        rowset_on_ ? ",\"rowset\":\"on\"" : "";
    // T-GPLR-48: the RESOLVED per-pass-rebuild decision and whether it came from
    // the explicit flag or from the KV-conditional default. This is the only
    // place a run records which of the two it used, so an explicit arm stays
    // distinguishable from a defaulted one in the trace alone. Absent (-1) only
    // for callers that never resolve the flag.
    std::string pbp_str;
    if (pp_prebuild_per_pass_ >= 0) {
        pbp_str = ",\"pp_prebuild_per_pass\":" + std::to_string(pp_prebuild_per_pass_) +
                  ",\"pp_prebuild_per_pass_source\":\"" +
                  (pp_prebuild_per_pass_source_ && pp_prebuild_per_pass_source_[0]
                       ? pp_prebuild_per_pass_source_ : "default") + "\"";
    }
    // T-GPLR-52 A3/AC4: the ENGINE-side staging source, derived from the state
    // flags ls.init set - not an echo of the declared copy_source. This is the
    // in-trace proof that a run's staging path is what its label says (the F5
    // lesson: run_start used to echo argv, so a pinned run carried only
    // copy_source=pinned and no proof the pinned path was active).
    // staging_source_match cross-checks the two; the strict checker treats a
    // mismatch as fatal, and legacy traces without the keys stay admissible.
    const char * stage_src_str =
        copy_src_reg_churn   ? "mmap-registered-churn" :
        copy_src_registered  ? "mmap-registered" :
        use_pinned_staging   ? "pinned" : "direct";
    const std::string stage_str = std::string(",\"staging_source\":\"") + stage_src_str +
        "\",\"staging_source_match\":" +
        (copy_source && stage_src_str == std::string(copy_source) ? "1" : "0");
    // T-GPLR-37: how the copy source was resolved ("explicit" | "default").
    // Emitted always: run_start already carried the RESOLVED copy_source; the
    // provenance key is what makes a defaulted run distinguishable from an
    // explicit one in the trace alone (the T-GPLR-48 lesson).
    const std::string cs_src_str = std::string(",\"copy_source_source\":\"") +
        (copy_source_source_ && copy_source_source_[0] ? copy_source_source_ : "default") + "\"";
    // T-GPLR-39: the resolved route-DIAGNOSTIC switch and its provenance, in
    // the pbp pattern (absent when the caller never resolves it).
    std::string rt_str;
    if (route_telemetry_ >= 0) {
        rt_str = std::string(",\"route_telemetry\":") + (route_telemetry_ ? "1" : "0") +
                 ",\"route_telemetry_source\":\"" +
                 (route_telemetry_source_ && route_telemetry_source_[0] ? route_telemetry_source_ : "default") + "\"";
    }
    // T-GPLR-54 A2: the resolved per-step logit readout depth and its
    // provenance, in the route_telemetry pattern (absent when the caller
    // never resolves it, so legacy recordings are unchanged).
    std::string tlm_str;
    if (tg_logit_margin_ >= 0) {
        tlm_str = std::string(",\"tg_logit_margin\":") + std::to_string(tg_logit_margin_) +
                 ",\"tg_logit_margin_source\":\"" + "explicit" + "\"";
    }
    // T-GPLR-63 B: the resolved TG budget. 0 is a real, common value (every
    // prefill-only run, including every CTest arm) and is exactly the fact
    // that run_start previously did not record, so a reader asking "did this
    // run have a TG phase?" had only run_end.n_tokens_generated -- which is 0
    // for a run that failed before generating anything, so it cannot
    // distinguish "prefill-only by schedule" from "prefill-only by failure".
    // The T-GPLR-12 handoff promise is a promise ABOUT the TG phase, so the
    // schedule witness belongs in the same record as the promise. No
    // _source key: --tg-tokens has no default/explicit distinction to record
    // (an explicit 0 and the implicit default 0 mean the same thing).
    std::string tgt_str;
    if (tg_tokens_ >= 0) {
        tgt_str = ",\"tg_tokens\":" + std::to_string(tg_tokens_);
    }
    // T-GPLR-58 A2: the RESOLVED speculative-placement mode, budget and reserve,
    // read from the module's own members -- set_row_set_spec() has already run at
    // this point, so these are the values the run will actually use, not an echo of
    // argv. Absent entirely in mode 0, so legacy traces are unchanged; a checker
    // still requires the keys whenever a leg claims prediction.
    std::string spec_str;
    if (row_set_spec_on) {
        spec_str = std::string(",\"row_spec_mode\":") + std::to_string(row_set_spec_mode) +
                   ",\"row_spec_budget\":" + std::to_string(row_set_spec_budget) +
                   ",\"row_spec_reserve\":" + std::to_string(row_set_spec_reserve) +
                   ",\"row_spec_pool\":" + std::to_string(row_set_spec_pool) +
                   ",\"row_spec_pool_bytes\":" + std::to_string(row_set_spec_pool_alloc_bytes);
        if (row_set_spec_mode == 2) {
            // MODE 2: the ARM is part of the RESOLVED configuration, not an argv echo,
            // and the null source is named separately -- a null leg must be identifiable
            // from its own trace, because a filename in this tree (arm-A1.jsonl /
            // arm-B1.jsonl) names an A/B session, not a predictor arm.
            spec_str += std::string(",\"row_spec_arm\":\"") + ls_pred_arm_name(row_set_spec_arm) +
                        "\",\"row_spec_null\":" + (row_set_spec_null ? "true" : "false");
        }
    }

    const std::string start_head = tel_run_start_head(run_id, t0_us);
    tel_write(tel_out,
        "%s\"model\":\"%s\",\"arch\":\"%s\",\"n_layer\":%d,\"n_slots\":%d,\"device\":%d,"
        "\"schedule\":\"%s\",\"copy_source\":\"%s\",\"kv_cache\":\"%s\",\"implicit_causal_mask\":%s,\"telemetry_chunk_stride\":%d,\"act_slab_bytes\":%zu,\"act_slab_dtype\":\"%s\"%s%s%s%s%s%s%s%s%s%s%s,"
        "\"static_vram_bytes\":%zu,\"slot_vram_bytes\":%zu,\"free_vram_bytes\":%zu,\"total_vram_bytes\":%zu}",
        start_head.c_str(),
        model.name.c_str(), llm_arch_name(model.arch), n_layer, n_slots, device,
        schedule ? schedule : "batch-major", copy_source ? copy_source : "direct", kv_cache ? kv_cache : "f16",
        use_implicit_causal_mask ? "true" : "false", telemetry_chunk_stride, act_slab_bytes,
        act_slab_dtype_ == GGML_TYPE_BF16 ? "bf16" : "f32", stage_str.c_str(), chunks_str.c_str(), ctx_str.c_str(),
        ho_mode_str.c_str(), tgt_str.c_str(), rowset_str.c_str(), pbp_str.c_str(), cs_src_str.c_str(), rt_str.c_str(), tlm_str.c_str(),
        spec_str.c_str(),
        static_vram, slot_vram, free_vram, total_vram);
}

void llama_layer_streaming::tel_run_end(int exit_status, size_t peak_vram, int64_t n_tokens_generated, const char * err) {
    tel_write(tel_out,
        "{\"event\":\"run_end\",\"run_id\":%llu,\"exit_status\":%d,\"peak_vram_bytes\":%zu,"
        "\"n_tokens_generated\":%lld,\"failed\":%s,\"error\":\"%s\",\"total_patch_us\":%.1f}",
        (unsigned long long) run_id, exit_status, peak_vram,
        (long long) n_tokens_generated, failed ? "true" : "false",
        err && err[0] ? err : (error_msg.empty() ? "none" : error_msg.c_str()),
        total_patch_us);
    llama_ls_tel_flush_early(tel_out); // late markers land in the trace, not in a dead buffer
    if (tel_out != nullptr) {
        fclose(tel_out);
        tel_out = nullptr;
    }
}

// T-GPLR-22: the refusal detail is free text and the event is one JSON line, so
// quotes, backslashes and control characters are folded to spaces rather than
// emitted raw (a raw '"' would make the line unparseable, which is exactly the
// failure this event exists to prevent).
static void tel_json_safe_into(std::string & out, const char * s) {
    if (s == nullptr) {
        return;
    }
    for (const char * p = s; *p != '\0'; ++p) {
        const unsigned char c = (unsigned char) *p;
        out += (c == '"' || c == '\\' || c < 0x20) ? ' ' : (char) c;
    }
}

// THE one writer of the run_refused record shape. Both emitters below go
// through it, so instance and one-shot traces cannot drift.
static void tel_write_run_refused(FILE * f, unsigned long long run_id, int phase,
                                  const char * reason, const char * detail, int exit_status,
                                  uint64_t bytes_needed, uint64_t bytes_allowed,
                                  const char * cap_source) {
    std::string safe_detail;
    tel_json_safe_into(safe_detail, detail);
    const long long now = (long long) ggml_time_us();
    tel_write(f,
        "{\"event\":\"run_refused\",\"run_id\":%llu,\"reason\":\"%s\",\"detail\":\"%s\","
        "\"exit_status\":%d,\"bytes_needed\":%llu,\"bytes_allowed\":%llu,\"cap_source\":\"%s\","
        "\"phase\":%d,\"t_us\":%lld}",
        run_id, reason ? reason : "unspecified", safe_detail.c_str(), exit_status,
        (unsigned long long) bytes_needed, (unsigned long long) bytes_allowed,
        cap_source ? cap_source : "none", phase, now);
}

void llama_layer_streaming::tel_run_refused(const char * reason, const char * detail, int exit_status,
                                            uint64_t bytes_needed, uint64_t bytes_allowed,
                                            const char * cap_source) {
    tel_write_run_refused(tel_out, (unsigned long long) run_id, phase, reason, detail, exit_status,
                          bytes_needed, bytes_allowed, cap_source);
    llama_ls_tel_flush_early(tel_out); // late markers land in the trace, not in a dead buffer
    if (tel_out != nullptr) {
        fclose(tel_out);
        tel_out = nullptr;
    }
}

// T-GPLR-22 (PRE-SINK): see the header. The capacity/handoff preflights run
// before init(), so there is no instance to emit from.
bool llama_layer_streaming::tel_run_refused_oneshot(const char * tel_path_, const char * reason,
                                                   const char * detail, int exit_status,
                                                   uint64_t bytes_needed, uint64_t bytes_allowed,
                                                   const char * cap_source) {
    if (tel_path_ == nullptr || tel_path_[0] == '\0') {
        return false;
    }
    FILE * f = fopen(tel_path_, "w");
    if (f == nullptr) {
        return false;
    }
    tel_write_run_refused(f, (unsigned long long) ggml_time_us(), 0, reason, detail, exit_status,
                          bytes_needed, bytes_allowed, cap_source);
    fclose(f);
    return true;
}

// T-GPLR-05: minimal one-shot telemetry for a forced `off` run (normal
// llama.cpp placement, no streaming init). `run_start` carries policy "off",
// arch, and the stable capability reason so unsupported models still produce
// an auditable trace instead of a silent failure.
bool llama_layer_streaming::tel_off_run(const char * tel_path_, const char * arch, const char * reason,
                                        int exit_status_, int64_t n_generated, int64_t t0_us_,
                                        int64_t pp_tokens_, int64_t tg_tokens_, int64_t ctx_size_) {
    if (tel_path_ == nullptr || tel_path_[0] == '\0') {
        return false;
    }
    tel_path = tel_path_;
    tel_out = fopen(tel_path.c_str(), "w");
    if (tel_out == nullptr) {
        error_msg = "failed to open telemetry file: " + tel_path;
        return false;
    }
    t0_us  = (uint64_t) t0_us_;
    run_id = (uint64_t) t0_us_;
    const std::string start_head = tel_run_start_head(run_id, t0_us);
    tel_write(tel_out,
        "%s\"arch\":\"%s\",\"policy\":\"off\",\"residency\":\"off\",\"capability_reason\":\"%s\","
        "\"pp_tokens\":%lld,\"tg_tokens\":%lld,\"ctx_size\":%lld}",
        start_head.c_str(),
        arch ? arch : "unknown", reason ? reason : "none",
        (long long) pp_tokens_, (long long) tg_tokens_, (long long) ctx_size_);
    // `err == nullptr` with no recorded error emits "none": a healthy off run
    // must not place a non-error in the error field (consumers key on non-empty
    // error text).
    tel_run_end(exit_status_, 0, n_generated, nullptr);
    return true;
}

// T-GPLR-44 step 1: patch-cache composition dump.
//
// The scoping lever rests on a claim that reading the call structure can only
// bound, never measure: that prepare_chunk rebuilds its two caches per SET
// while each launch consumes exactly one GRAPH of that set. This emits the
// measurement, once per set, at build time, so the lever's own first step can
// falsify it before any patch is written.
//
// Fields:
//   set_input_tensors / set_input_bytes / set_fa_nodes
//                        what one launch walks TODAY: the whole set, restored
//                        and re-patched on every (layer, chunk) launch.
//   graph_members[gi]    what a scoped launch for graph gi would walk instead:
//                        input tensors and bytes referenced by that graph's
//                        nodes, plus its FLASH_ATTN_EXT node count. Index
//                        matches graphs[gi] (0 = embed, 1+il = layer il, last
//                        = output).
//   union_input_tensors  closure check: how many of the set-level input
//                        tensors are referenced by at least one graph.
//                        union_matches == 1 means every cached tensor is
//                        attributed, so the per-graph numbers are complete. If
//                        it is 0, the dump cannot size the lever and the task
//                        must close rather than patch -- and `set_members`
//                        names the tensor responsible instead of leaving it to
//                        be guessed.
//   set_members          every cached input tensor: name, bytes, and how many
//                        graphs consume it (transitively -- an input consumed
//                        through a view is one hop past the node's direct src).
//   ctx_tensors          tensors in the set ctx, cached or not. Far more than
//                        the cache means the build leaves tensors behind, which
//                        matters because per-pass rebuilds are the default.
//   redundancy_input     set_input_bytes / mean(layer graph input bytes), and
//   redundancy_fa        set_fa_nodes / mean(layer graph FA nodes): how much
//                        more a launch walks than the graph it is about to run
//                        needs. A CEILING on the lever, not a wall-clock
//                        prediction -- the per-graph attribution says nothing
//                        about memcpy cost or the FA patch loop's share.
//
// Cost note: the build below is charged to the first launch's patch bucket in
// each set, so a baseline patch_us must be read from a later pass (the arms use
// pass 2 of --pp-repeat 2 for exactly this reason).
//
// T-GPLR-44 step 2: build the per-GRAPH split of the two patch caches.
// T-GPLR-45: the same walk also CLASSIFIES the INPUT candidates (kept vs
// zero-consumer) and populates the member list with the kept ones only, so the
// chunk-snapshot capture, the launch restore and the composition dump all
// iterate one list. A candidate no graph consumes (transitively, through
// views) is parked in excluded_inputs with its byte sum and is never restored
// again -- the T-GPLR-44 dump measured exactly one such orphan per set
// (inp_embd, 4 MiB, 99.6% of the restore bytes).
//
// ONE walk serves both the dump and the launches: tel_patch_cache reports the
// closure of exactly these lists, so the dump cannot certify a different split
// than prepare_chunk consumes. Two separate walks could drift, and a drifted
// dump would report a redundancy the patch does not deliver.
//
// Attribution is TRANSITIVE over src edges, not merely the nodes' direct sources:
// measured on the first arm (2026-09-18), a direct-src walk left 1 of 7 cached
// tensors attributed to NO graph. That is the shape an input takes when it is
// consumed through a view - the node's src is the view, and the INPUT-flagged
// tensor sits one hop further down. Scoping to direct sources alone would have
// dropped that restore silently.
bool llama_layer_streaming::build_patch_split(graph_set & gs) {
    const size_t ng = gs.graphs.size();

    // ---- T-GPLR-45: classify the INPUT candidates BEFORE building the split.
    // The candidate list is the set ctx's INPUT-flagged, allocated tensors in
    // ctx order (the same predicate the old cache capture and the chunk
    // snapshot capture walked). A candidate is KEPT iff at least one graph of
    // the set reaches it transitively (through views); otherwise it is parked
    // in excluded_inputs and never restored or snapshotted again -- the
    // T-GPLR-44 dump measured exactly one such orphan per set (inp_embd,
    // 4 MiB, zero consumers, 99.6% of the restore bytes). Nothing name-based:
    // a shape that consumes the embeddings branch keeps restoring it.
    {
        std::vector<ggml_tensor *> candidates;
        for (ggml_tensor * t = ggml_get_first_tensor(gs.ctx); t != nullptr; t = ggml_get_next_tensor(gs.ctx, t)) {
            if ((t->flags & GGML_TENSOR_FLAG_INPUT) && t->data != nullptr) {
                candidates.push_back(t);
            }
        }

        // Transitive consumer set over ALL graphs (embed, layers, output):
        // an input consumed through a view sits one hop past the node's
        // direct src, so the walk must recurse over src edges (the exact bug
        // T-GPLR-44's closure guard caught for the direct-src variant).
        std::unordered_set<const ggml_tensor *> reached;
        {
            std::vector<const ggml_tensor *> stack;
            for (size_t gi = 0; gi < ng; ++gi) {
                ggml_cgraph * g = gs.graphs[gi];
                if (g == nullptr) {
                    continue;
                }
                for (int k = 0; k < g->n_nodes; ++k) {
                    ggml_tensor * node = g->nodes[k];
                    if (node == nullptr) {
                        continue;
                    }
                    for (int s = 0; s < GGML_MAX_SRC; ++s) {
                        if (node->src[s] != nullptr) {
                            stack.push_back(node->src[s]);
                        }
                    }
                }
            }
            while (!stack.empty()) {
                const ggml_tensor * t = stack.back();
                stack.pop_back();
                if (!reached.insert(t).second) {
                    continue;
                }
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (t->src[s] != nullptr) {
                        stack.push_back(t->src[s]);
                    }
                }
            }
        }

        std::vector<ggml_tensor *> kept;
        kept.reserve(candidates.size());
        gs.excluded_inputs.clear();
        gs.excluded_bytes = 0;
        for (ggml_tensor * t : candidates) {
            if (reached.count(t) != 0) {
                kept.push_back(t);
            } else {
                gs.excluded_inputs.push_back(t);
                gs.excluded_bytes += ggml_nbytes(t);
            }
        }
        gs.input_tensors = std::move(kept);
    }

    // ---- T-GPLR-62 S2: the row-set COMPANION inputs are outside this list by
    // design, and that must stay true for a reason rather than by accident.
    //
    // The list above is the PROMPT-PROCESSING chunk snapshot / launch restore
    // set. It exists because a kernel can clobber a shared input buffer between
    // the chunk bind point (where shared rec/KV state is current) and the launch,
    // so the values are saved and put back. The row-set token-generation
    // companion graphs (`ff_graphs`, the part-B FFN) are not in `graphs`, so the
    // transitive walk above never reaches them, and their inputs -- sel_in and
    // sel_pos_in -- are classified as excluded. That is CORRECT and is what the
    // existing sel_in has always relied on: the executor writes each companion
    // input immediately before running its own graph, kernels treat it as
    // read-only, and the scheduler's alloc_graph performs the host-to-device
    // copy. Neither tensor needs, or has, snapshot protection.
    //
    // What must never happen is a companion input that is read by a graph IN
    // this set, because then it would be snapshotted and restored as a
    // prompt-processing input while a companion graph also writes it directly --
    // two owners of one buffer with one running offset between them. That is the
    // inconsistency this guard refuses. It cannot happen today; it could start
    // happening if a future change made a companion input reachable from a PP
    // graph (e.g. sharing it with the MTP head), which is why it is checked
    // rather than assumed.
    {
        std::unordered_set<const ggml_tensor *> kept_set;
        kept_set.reserve(gs.input_tensors.size() * 2 + 1);
        for (ggml_tensor * t : gs.input_tensors) {
            kept_set.insert(t);
        }
        for (const std::vector<ggml_tensor *> * lst : { &gs.ff_input, &gs.ff_input_wpos, &gs.ff_input_hsel }) {
            for (ggml_tensor * t : *lst) {
                if (t == nullptr) {
                    continue;
                }
                if (kept_set.count(t) != 0) {
                    error_msg = "build_patch_split: a row-set companion input is in the "
                                "prompt-processing snapshot list; the snapshot and the "
                                "companion executor would both own its bytes";
                    failed = true;
                    return false;
                }
                // Backed-by-contract: finish_build's context walk allocates every
                // INPUT-flagged tensor, so a companion input with no buffer would
                // mean the sizing model and the allocation walk disagree, and the
                // executor's null check would fire as a false A10 refusal.
                if (t->data == nullptr) {
                    error_msg = "build_patch_split: a row-set companion input is unbacked "
                                "(no host buffer); the input sizing model is short of it";
                    failed = true;
                    return false;
                }
            }
        }
    }

    gs.graph_inputs.assign(ng, {});
    gs.graph_fa_nodes.assign(ng, {});

    // Offsets are computed over the KEPT member list, in capture order. The
    // chunk snapshot and the launch restore both iterate this list with a
    // running offset, so capture order must match consume order; excluded
    // members have no snapshot bytes at all (T-GPLR-45 contract: one list,
    // one pass, no per-graph slices).
    gs.input_offsets.assign(gs.input_tensors.size(), 0);
    std::unordered_map<const ggml_tensor *, size_t> members;
    members.reserve(gs.input_tensors.size() * 2 + 1);
    size_t off = 0;
    for (size_t i = 0; i < gs.input_tensors.size(); ++i) {
        members.emplace(gs.input_tensors[i], i);
        gs.input_offsets[i] = off;
        off += ggml_nbytes(gs.input_tensors[i]);
    }
    gs.input_snap_bytes = off;

    for (size_t gi = 0; gi < ng; ++gi) {
        ggml_cgraph * g = gs.graphs[gi];
        if (g == nullptr) {
            continue;
        }
        std::unordered_set<const ggml_tensor *> visited;
        std::vector<const ggml_tensor *> stack;
        for (int k = 0; k < g->n_nodes; ++k) {
            ggml_tensor * node = g->nodes[k];
            if (node == nullptr) {
                continue;
            }
            if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                gs.graph_fa_nodes[gi].push_back(node);
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (node->src[s] != nullptr) {
                    stack.push_back(node->src[s]);
                }
            }
        }
        std::vector<char> in_this_graph(gs.input_tensors.size(), 0);
        while (!stack.empty()) {
            const ggml_tensor * t = stack.back();
            stack.pop_back();
            if (!visited.insert(t).second) {
                continue;
            }
            auto it = members.find(t);
            if (it != members.end() && !in_this_graph[it->second]) {
                // Reached twice through one graph is still one tensor's worth of
                // memcpy, so it counts once per graph.
                in_this_graph[it->second] = 1;
                gs.graph_inputs[gi].push_back(it->second);
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (t->src[s] != nullptr) {
                    stack.push_back(t->src[s]);
                }
            }
        }
    }

    gs.patch_split_ok = true;
    return true;
}

void llama_layer_streaming::tel_patch_cache(const graph_set & gs, size_t set_idx) {
    if (!pp_recon || tel_out == nullptr) {
        return;
    }

    const size_t ng = gs.graphs.size();
    std::vector<int64_t> g_in_count(ng, 0);
    std::vector<int64_t> g_in_bytes(ng, 0);
    std::vector<int64_t> g_fa_count(ng, 0);
    std::vector<int64_t> member_refs(gs.input_tensors.size(), 0);

    // Reported FROM the split, not from a second walk. The launches no longer
    // consume it: the scoping A/B failed generated-id identity and was reverted
    // (see begin_set); this reports the composition as evidence.
    for (size_t gi = 0; gi < ng; ++gi) {
        if (gi < gs.graph_inputs.size()) {
            g_in_count[gi] = (int64_t) gs.graph_inputs[gi].size();
            for (size_t m : gs.graph_inputs[gi]) {
                g_in_bytes[gi] += (int64_t) ggml_nbytes(gs.input_tensors[m]);
                member_refs[m]++;
            }
        }
        if (gi < gs.graph_fa_nodes.size()) {
            g_fa_count[gi] = (int64_t) gs.graph_fa_nodes[gi].size();
        }
    }

    int64_t set_in_bytes = 0;
    for (const ggml_tensor * t : gs.input_tensors) {
        set_in_bytes += (int64_t) ggml_nbytes(t);
    }
    // T-GPLR-45: the zero-consumer members the filter dropped, named per member
    // with graphs=0 (they are excluded BY the zero-consumer test, so the value
    // is structural, not measured). walked_bytes = kept + excluded is the byte
    // identity the analyzer asserts: the filter can neither create nor lose
    // bytes versus the unfiltered candidate walk.
    int64_t excluded_count = (int64_t) gs.excluded_inputs.size();
    int64_t walked_bytes = set_in_bytes + (int64_t) gs.excluded_bytes;
    std::string excluded_members;
    for (size_t i = 0; i < gs.excluded_inputs.size(); ++i) {
        const ggml_tensor * t = gs.excluded_inputs[i];
        char safe[GGML_MAX_NAME + 1];
        size_t n = 0;
        for (const char * p = t->name; *p != '\0' && n < GGML_MAX_NAME; ++p) {
            safe[n++] = (*p == '"' || *p == '\\') ? '_' : *p;
        }
        safe[n] = '\0';
        char mbuf[192];
        snprintf(mbuf, sizeof(mbuf), "%s{\"name\":\"%s\",\"bytes\":%lld,\"graphs\":0}",
                 i ? "," : "", safe, (long long) ggml_nbytes(t));
        excluded_members += mbuf;
    }
    int64_t union_count = 0;
    for (int64_t r : member_refs) {
        union_count += r > 0 ? 1 : 0;
    }
    const int64_t set_fa_count = (int64_t) gs.fa_nodes.size();

    // How many tensors the set ctx holds in total, versus how many of them the
    // cache selected. A ctx holding far more than the cache means the build left
    // tensors behind (relevant because T-GPLR-43 made per-pass rebuilds the
    // default); a cache holding tensors no graph consumes is the closure failure
    // below, and NAMING them is the diagnosis rather than a count.
    int64_t ctx_tensors = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(gs.ctx); t != nullptr;
         t = ggml_get_next_tensor(gs.ctx, t)) {
        ctx_tensors++;
    }

    std::string set_members;
    for (size_t i = 0; i < gs.input_tensors.size(); ++i) {
        const ggml_tensor * t = gs.input_tensors[i];
        // Names are compiler-generated identifiers here, but sanitise anyway: an
        // unescaped quote would make the whole telemetry line unparseable.
        char safe[GGML_MAX_NAME + 1];
        size_t n = 0;
        for (const char * p = t->name; *p != '\0' && n < GGML_MAX_NAME; ++p) {
            safe[n++] = (*p == '"' || *p == '\\') ? '_' : *p;
        }
        safe[n] = '\0';
        char mbuf[192];
        snprintf(mbuf, sizeof(mbuf), "%s{\"name\":\"%s\",\"bytes\":%lld,\"graphs\":%lld}",
                 i ? "," : "", safe, (long long) ggml_nbytes(t), (long long) member_refs[i]);
        set_members += mbuf;
    }

    // Layer graphs are graphs[1 .. ng-2] -- the same range the cache build
    // walks (0 = embed, last = output).
    int64_t layers = 0, layer_bytes = 0, layer_fa = 0;
    int64_t lmin_b = 0, lmax_b = 0, lmin_f = 0, lmax_f = 0;
    for (size_t gi = 1; gi + 1 < ng; ++gi) {
        if (layers == 0) {
            lmin_b = lmax_b = g_in_bytes[gi];
            lmin_f = lmax_f = g_fa_count[gi];
        } else {
            if (g_in_bytes[gi] < lmin_b) { lmin_b = g_in_bytes[gi]; }
            if (g_in_bytes[gi] > lmax_b) { lmax_b = g_in_bytes[gi]; }
            if (g_fa_count[gi] < lmin_f) { lmin_f = g_fa_count[gi]; }
            if (g_fa_count[gi] > lmax_f) { lmax_f = g_fa_count[gi]; }
        }
        layers++;
        layer_bytes += g_in_bytes[gi];
        layer_fa    += g_fa_count[gi];
    }
    const double mean_b = layers ? (double) layer_bytes / (double) layers : 0.0;
    const double mean_f = layers ? (double) layer_fa / (double) layers : 0.0;

    std::string members;
    char buf[128];
    for (size_t gi = 0; gi < ng; ++gi) {
        snprintf(buf, sizeof(buf), "%s{\"gi\":%lld,\"layers_in\":%lld,\"bytes_in\":%lld,\"fa_nodes\":%lld}",
                 gi ? "," : "", (long long) gi, (long long) g_in_count[gi],
                 (long long) g_in_bytes[gi], (long long) g_fa_count[gi]);
        members += buf;
    }

    tel_write(tel_out,
        "{\"event\":\"patch_cache\",\"run_id\":%llu,\"set\":%llu,\"n_graphs\":%llu,"
        "\"set_input_tensors\":%llu,\"set_input_bytes\":%lld,\"set_fa_nodes\":%lld,"
        "\"union_input_tensors\":%lld,\"union_matches\":%d,\"ctx_tensors\":%lld,"
        "\"layer_graphs\":%lld,\"layer_input_bytes_min\":%lld,\"layer_input_bytes_max\":%lld,"
        "\"layer_input_bytes_mean\":%.1f,\"layer_fa_min\":%lld,\"layer_fa_max\":%lld,"
        "\"layer_fa_mean\":%.1f,\"redundancy_input\":%.3f,\"redundancy_fa\":%.3f,"
        "\"excluded_count\":%lld,\"excluded_bytes\":%lld,\"walked_bytes\":%lld,"
        "\"excluded_members\":[%s],"
        "\"set_members\":[%s],\"graph_members\":[%s],\"t_us\":%lld}",
        (unsigned long long) run_id, (unsigned long long) set_idx, (unsigned long long) ng,
        (unsigned long long) gs.input_tensors.size(), (long long) set_in_bytes, (long long) set_fa_count,
        (long long) union_count, union_count == (int64_t) gs.input_tensors.size() ? 1 : 0,
        (long long) ctx_tensors,
        (long long) layers, (long long) lmin_b, (long long) lmax_b, mean_b,
        (long long) lmin_f, (long long) lmax_f, mean_f,
        mean_b > 0.0 ? (double) set_in_bytes / mean_b : 0.0,
        mean_f > 0.0 ? (double) set_fa_count / mean_f : 0.0,
        excluded_count, (long long) gs.excluded_bytes, (long long) walked_bytes,
        excluded_members.c_str(),
        set_members.c_str(), members.c_str(), (long long) ggml_time_us());
}

// T-GPLR-04: structured byte-class accounting for the explicit out-of-core
// capacity proof. Emitted after the preflight checks pass, before the timed
// PP. The host checker validates the additive sums.
void llama_layer_streaming::tel_byte_account(uint64_t full_model_plus_runtime, uint64_t bounded_window,
                                             uint64_t static_bytes, uint64_t per_layer_bytes, uint64_t slot_window_bytes,
                                             uint64_t kv_bytes, uint64_t state_bytes, uint64_t act_slab_bytes,
                                             uint64_t reserve_bytes, uint64_t free_vram_bytes, uint64_t total_vram_bytes,
                                             int64_t residency_budget_mib,
                                             uint64_t row_cache_bytes, uint64_t residency_cap_bytes,
                                             uint64_t prediction_pool_bytes) {
    const std::string pool_str = prediction_pool_bytes > 0
        ? ",\"prediction_pool\":" + std::to_string(prediction_pool_bytes) : "";
    tel_write(tel_out,
        "{\"event\":\"byte_account\",\"run_id\":%llu,"
        "\"full_model_plus_runtime\":%llu,\"bounded_window\":%llu,"
        "\"static\":%llu,\"per_layer\":%llu,\"slot_window\":%llu,\"kv\":%llu,\"state\":%llu,"
        "\"act_slab\":%llu,\"reserve\":%llu,"
        "\"free_vram\":%llu,\"total_vram\":%llu,\"residency_budget_mib\":%lld,"
        "\"row_cache\":%llu,\"residency_cap\":%llu%s,"
        "\"full_exceeds_free\":%s,\"bounded_within_free\":%s}",
        (unsigned long long) run_id,
        (unsigned long long) full_model_plus_runtime, (unsigned long long) bounded_window,
        (unsigned long long) static_bytes, (unsigned long long) per_layer_bytes,
        (unsigned long long) slot_window_bytes, (unsigned long long) kv_bytes,
        (unsigned long long) state_bytes, (unsigned long long) act_slab_bytes,
        (unsigned long long) reserve_bytes,
        (unsigned long long) free_vram_bytes, (unsigned long long) total_vram_bytes,
        (long long) residency_budget_mib,
        (unsigned long long) row_cache_bytes, (unsigned long long) residency_cap_bytes,
        pool_str.c_str(),
        full_model_plus_runtime > free_vram_bytes ? "true" : "false",
        bounded_window + row_cache_bytes + prediction_pool_bytes + act_slab_bytes + reserve_bytes <= free_vram_bytes ? "true" : "false");
}

// T-GPLR-36 follow-up: the engine's OWN host-side byte holdings, read from the
// live allocations (never re-derived), so a pass-boundary resident-set delta can
// be compared against what the residency machinery itself is holding: the
// per-slot pinned staging buffers (pinned copy source), the persistent pinned
// pool (T-GPLR-30), and the registered mapped source spans (T-GPLR-37).
// Device-side residency (slot windows, row cache, act slabs) is deliberately NOT
// included: byte_account owns that, and mixing the two would make this figure
// useless as the host-side term.
//
// Absence is meaningful: a pinned-source run holds staging bytes here, a
// mmap-registered run holds registered mapping bytes instead and no staging at
// all, so the two modes are distinguishable from the trace alone.
uint64_t llama_layer_streaming::host_bytes_accounted() const {
    uint64_t total = 0;
    for (const auto & s : slots) {
        if (s.buf_staging) {
            total += (uint64_t) ggml_backend_buffer_get_size(s.buf_staging.get());
        }
    }
    for (const auto & e : pp_pool) {
        total += (uint64_t) e.bytes;
    }
    for (size_t i = 0; i < all_reg_bytes.size(); ++i) {
        total += (uint64_t) all_reg_bytes[i];
    }
    return total;
}

void llama_layer_streaming::tel_host_mem_begin() {
    ++mem_pass_ctr;
    tel_host_mem_emit("pass_begin");
}

void llama_layer_streaming::tel_host_mem_end() {
    tel_host_mem_emit("pass_end");
}

// The one writer of the host_mem record shape. `d_*` is the move since the
// previous sample (the two samples of one pass are its own pair), `g_*` the move
// since the first sample of the run; both are clamped at 0 because a shrinking
// resident set is a real outcome (the OS trims, the mapping evicts) and encoding
// it as a wrapped unsigned would read as a 16 EiB jump. `eng_*` carries the
// engine's own accounted host bytes at the same instant, so a reader can say
// whether the delta is the residency machinery or something else.
void llama_layer_streaming::tel_host_mem_emit(const char * label) {
    if (!pp_recon || tel_out == nullptr) {
        return;
    }
    const llama_ls_host_mem m = llama_ls_host_mem_sample();
    const uint64_t eng = host_bytes_accounted();
    if (!mem_first_seen) {
        mem_first_seen   = true;
        mem_first_ws     = m.ws_bytes;
        mem_first_commit = m.commit_bytes;
        mem_first_eng    = eng;
        mem_prev_ws      = m.ws_bytes;
        mem_prev_commit  = m.commit_bytes;
        mem_prev_eng     = eng;
    }
    if (m.ws_bytes > mem_peak_ws) {
        mem_peak_ws = m.ws_bytes;
    }
    const uint64_t d_ws     = m.ws_bytes     >= mem_prev_ws     ? m.ws_bytes     - mem_prev_ws     : 0;
    const uint64_t d_commit = m.commit_bytes >= mem_prev_commit ? m.commit_bytes - mem_prev_commit : 0;
    const uint64_t d_eng    = eng            >= mem_prev_eng    ? eng            - mem_prev_eng    : 0;
    const uint64_t g_ws     = m.ws_bytes     >= mem_first_ws    ? m.ws_bytes     - mem_first_ws    : 0;
    const uint64_t g_commit = m.commit_bytes >= mem_first_commit? m.commit_bytes - mem_first_commit: 0;
    const uint64_t g_eng    = eng            >= mem_first_eng   ? eng            - mem_first_eng   : 0;
    const uint64_t file_bytes = m.ws_bytes  >= m.ws_private_bytes ? m.ws_bytes - m.ws_private_bytes : 0;
    tel_write(tel_out,
        "{\"event\":\"host_mem\",\"run_id\":%llu,\"label\":\"%s\",\"pass\":%d,"
        "\"ws_bytes\":%llu,\"ws_private_bytes\":%llu,\"ws_file_bytes\":%llu,\"commit_bytes\":%llu,"
        "\"ws_private_known\":%s,"
        "\"ws_peak_os_bytes\":%llu,\"ws_peak_seen_bytes\":%llu,"
        "\"d_ws_bytes\":%llu,\"d_commit_bytes\":%llu,\"g_ws_bytes\":%llu,\"g_commit_bytes\":%llu,"
        "\"eng_bytes\":%llu,\"d_eng_bytes\":%llu,\"g_eng_bytes\":%llu,"
        "\"sys_free_bytes\":%llu,\"sys_total_bytes\":%llu,"
        "\"sys_commit_free_bytes\":%llu,\"sys_commit_total_bytes\":%llu,"
        "\"reader_ok\":%s,\"t_us\":%lld}",
        (unsigned long long) run_id, label ? label : "unspecified", mem_pass_ctr,
        (unsigned long long) m.ws_bytes, (unsigned long long) m.ws_private_bytes,
        (unsigned long long) file_bytes, (unsigned long long) m.commit_bytes,
        m.ws_private_known ? "true" : "false",
        (unsigned long long) m.ws_peak_bytes, (unsigned long long) mem_peak_ws,
        (unsigned long long) d_ws, (unsigned long long) d_commit,
        (unsigned long long) g_ws, (unsigned long long) g_commit,
        (unsigned long long) eng, (unsigned long long) d_eng, (unsigned long long) g_eng,
        (unsigned long long) m.sys_free_bytes, (unsigned long long) m.sys_total_bytes,
        (unsigned long long) m.sys_commit_free_bytes, (unsigned long long) m.sys_commit_total_bytes,
        m.ok ? "true" : "false", (long long) ggml_time_us());
    mem_prev_ws     = m.ws_bytes;
    mem_prev_commit = m.commit_bytes;
    mem_prev_eng    = eng;
}

void llama_layer_streaming::tel_slot_request(int il, int phase, size_t bytes) {
    tel_write(tel_out, "{\"event\":\"slot_request\",\"run_id\":%llu,\"ubatch\":%llu,\"layer\":%d,\"slot\":%d,\"phase\":%d,\"bytes\":%zu,\"t_us\":%lld}",
        (unsigned long long) run_id, (unsigned long long) n_ubatches, il, il % n_slots, phase, bytes, (long long) ggml_time_us());
}

void llama_layer_streaming::tel_copy_submit(int il, size_t bytes, uint32_t generation, int af_inflight, int submit_batch) {
    // T-GPLR-33: submit_batch reports the ACTUALLY-TAKEN submission path for
    // this bank, threaded from stage_members (0 = per-member, 1 = single
    // span copy, 2 = mixed span+per-member) — not the flag state, so traces
    // are self-verifying.
    tel_write(tel_out, "{\"event\":\"copy_submit\",\"run_id\":%llu,\"ubatch\":%llu,\"layer\":%d,\"slot\":%d,\"generation\":%u,\"bytes\":%zu,\"af_inflight\":%d,\"submit_batch\":%d,\"phase\":%d,\"t_us\":%lld}",
        (unsigned long long) run_id, (unsigned long long) n_ubatches, il, il % n_slots, generation, bytes, af_inflight, submit_batch, phase, (long long) ggml_time_us());
}

void llama_layer_streaming::tel_copy_complete(int il, double transfer_us, double host_submit_us, double ready_latency_us, uint32_t generation) {
    tel_write(tel_out, "{\"event\":\"copy_complete\",\"run_id\":%llu,\"ubatch\":%llu,\"layer\":%d,\"slot\":%d,\"generation\":%u,\"transfer_us\":%.1f,\"host_submit_us\":%.1f,\"ready_latency_us\":%.1f,\"phase\":%d,\"t_us\":%lld}",
        (unsigned long long) run_id, (unsigned long long) n_ubatches, il, il % n_slots, generation, transfer_us, host_submit_us, ready_latency_us, phase, (long long) ggml_time_us());
}

void llama_layer_streaming::tel_layer_ready_wait(int il, double wait_us, bool ready_at_consumption, uint32_t generation) {
    tel_write(tel_out, "{\"event\":\"layer_ready_wait\",\"run_id\":%llu,\"ubatch\":%llu,\"layer\":%d,\"slot\":%d,\"generation\":%u,\"wait_us\":%.1f,\"ready_at_consumption\":%s,\"phase\":%d,\"t_us\":%lld}",
        (unsigned long long) run_id, (unsigned long long) n_ubatches, il, il % n_slots, generation, wait_us, ready_at_consumption ? "true" : "false", phase, (long long) ggml_time_us());
}

// The layer_compute record, written here once so the shape is visible in one
// place. F3/H8's optional part split (row-set TG: part A attn+router, part B
// FFN) is built into a small stack buffer only when a caller asks for it, so the
// non-row-set record stays byte-identical.
void llama_layer_streaming::tel_layer_compute(int il, double compute_us, double setup_us, uint32_t generation,
                                              int64_t chunk, const tel_layer_compute_params & params) {
    // T-GPLR-82 S1: 160 -> 320 bytes; T-GPLR-61 S4: 320 -> 512 for the gate
    // record. `parts` holds ONLY the row-set optional block (gated on
    // attn_compute_us >= 0.0), so growth cannot alter any non-row-set
    // record: the two prefill call sites pass attn_* = -1.0 and keep the
    // byte-identical short form.
    char parts[640]; // T-GPLR-61 step C: 512 -> 640 for the host_finite record
    parts[0] = '\0';
    if (params.attn_compute_us >= 0.0) {
        // experts_* is the demand ledger: what the router asked for, split
        // by where it was computed. experts_device + experts_host must equal
        // experts_total at every (layer, token) -- T-GPLR-82 AC2. The
        // rows_gate_* / gate_* / staged_bytes_avoided tail is T-GPLR-61's
        // S4 gate record: emitted even when the split never fires (-1 when
        // no host leg exists at all).
        snprintf(parts, sizeof(parts),
                 "\"attn_compute_us\":%.1f,\"attn_setup_us\":%.1f,\"ffn_compute_us\":%.1f,\"ffn_setup_us\":%.1f,"
                 "\"attn_dev\":%d,\"ffn_dev\":%d,"
                 "\"experts_total\":%d,\"experts_host\":%d,\"experts_device\":%d,\"host_us\":%.1f,"
                 "\"rows_gate_cpu\":%d,\"rows_gate_gpu\":%d,"
                 "\"gate_host_us\":%.1f,\"gate_wait_us\":%.1f,\"staged_bytes_avoided\":%.0f,"
                 "\"host_finite\":%d,",
                 params.attn_compute_us, params.attn_setup_us, params.ffn_compute_us, params.ffn_setup_us, params.attn_dev, params.ffn_dev,
                 params.experts_total, params.experts_host, params.experts_device, params.host_us,
                 params.rows_gate_cpu, params.rows_gate_gpu, params.gate_host_us, params.gate_wait_us, params.staged_bytes_avoided,
                 params.host_finite);
    }
    tel_write(tel_out,
        "{\"event\":\"layer_compute\",\"run_id\":%llu,\"ubatch\":%llu,\"layer\":%d,\"slot\":%d,"
        "\"generation\":%u,\"chunk\":%lld,\"compute_us\":%.1f,\"setup_us\":%.1f,"
        "\"cpu_us\":%.1f,\"dev\":%d,"
        "%s\"phase\":%d,\"t_us\":%lld}",
        (unsigned long long) run_id, (unsigned long long) n_ubatches, il, il % n_slots, generation,
        (long long) chunk, compute_us, setup_us, params.cpu_us, params.dev, parts, phase, (long long) ggml_time_us());
}

// T-GPLR-53 A2: the per-chunk embed-step timeline (see the header note).
// span_us is the chunk's own wall (prepare + graph call), so the per-pass sum of
// span_us reproduces host_ledger.embed_us instead of a derived estimate.
void llama_layer_streaming::tel_embed_chunk(int64_t chunk, int64_t n_tokens, double setup_us,
                                            double compute_us, double span_us) {
    if (!pp_recon) {
        return;
    }
    tel_write(tel_out,
        "{\"event\":\"embed_chunk\",\"run_id\":%llu,\"chunk\":%lld,\"n_tokens\":%lld,"
        "\"setup_us\":%.1f,\"compute_us\":%.1f,\"span_us\":%.1f,\"t_us\":%lld}",
        (unsigned long long) run_id, (long long) chunk, (long long) n_tokens,
        setup_us, compute_us, span_us, (long long) ggml_time_us());
}

// T-GPLR-02: emit one observational route event per MoE layer/chunk.
//
// The router's selected_experts (argsort_top_k) output buffer is not reliably
// readable after CUDA-graph replay (the allocator reuses the io region), but
// the argsort's source selection_probs ([n_expert, n_tokens] F32) is live.
// We read those probabilities and derive the exact top-k expert IDs on the
// host with llama_ls_route_topk_row -- equivalent to a stable descending sort
// with the index tie-break, which matches the CUDA stable radix sort the
// router uses (T-GPLR-39 replaced the per-token sort of all n_expert experts
// with that bounded selection: same ids in the same order, so the emitted
// hash and union are unchanged). Observational only: never feeds routing or
// graph construction.
void llama_layer_streaming::tel_route(int il, int64_t chunk, ggml_cgraph * g) {
    if (tel_out == nullptr || g == nullptr) {
        return;
    }
    ggml_tensor * probs = nullptr;
    for (int k = 0; k < ggml_graph_n_nodes(g); ++k) {
        ggml_tensor * node = ggml_graph_node(g, k);
        // T-GPLR-13: identify the router selection probs by their cb-hook
        // NAME ("ffn_moe_probs-<il>") rather than by shape or op: part A
        // also contains attention softmax nodes whose ne[0] can alias
        // n_expert at some KV lengths. In single-part graphs the ARGSORT's
        // src[0] carries the same named tensor; accept both shapes.
        if (node->op == GGML_OP_SOFT_MAX && node->name != nullptr &&
            strncmp(node->name, "ffn_moe_probs-", 14) == 0) {
            probs = node;
            break;
        }
        if (node->op == GGML_OP_ARGSORT && node->src[0] != nullptr &&
            node->src[0]->name != nullptr &&
            strncmp(node->src[0]->name, "ffn_moe_probs-", 14) == 0) {
            probs = node->src[0];
            break;
        }
    }
    if (probs == nullptr || probs->data == nullptr || probs->type != GGML_TYPE_F32) {
        tel_write(tel_out,
            "{\"event\":\"route\",\"run_id\":%llu,\"layer\":%d,\"chunk\":%lld,\"error\":\"no selection probs\"}",
            (unsigned long long) run_id, il, (long long) chunk);
        return;
    }
    const int64_t n_expert = probs->ne[0];
    const int64_t n_tok    = probs->ne[1];
    const int64_t n_used   = (model_ref != nullptr) ? (int64_t) model_ref->hparams.n_expert_used : 0;
    if (n_expert <= 0 || n_tok <= 0 || n_used <= 0 || n_used > n_expert) {
        tel_write(tel_out,
            "{\"event\":\"route\",\"run_id\":%llu,\"layer\":%d,\"chunk\":%lld,\"error\":\"bad route shape\"}",
            (unsigned long long) run_id, il, (long long) chunk);
        return;
    }
    std::vector<float> p((size_t) n_expert * n_tok);
    ggml_backend_tensor_get(probs, p.data(), 0, p.size() * sizeof(float));

    uint64_t hash = 1469598103934665603ull;
    // T-GPLR-39: the union is read only as a cardinality, so a byte map indexed
    // by expert id replaces the hash set (same value, no per-insert allocation).
    std::vector<uint8_t> seen((size_t) n_expert, 0);
    size_t n_union = 0;
    std::vector<int32_t> ids((size_t) n_used);
    std::string sampled;
    for (int64_t i = 0; i < n_tok; ++i) {
        const float * row = &p[(size_t) i * n_expert];
        llama_ls_route_topk_row(row, n_expert, n_used, ids.data());
        for (int64_t j = 0; j < n_used; ++j) {
            const int32_t v = ids[(size_t) j];
            hash ^= (uint64_t) (uint32_t) v;
            hash *= 1099511628211ull;
            if (!seen[(size_t) v]) {
                seen[(size_t) v] = 1;
                ++n_union;
            }
            if (i == 0 && j < 16) {
                sampled += (j ? "," : "") + std::to_string(v);
            }
        }
    }
    tel_write(tel_out,
        "{\"event\":\"route\",\"run_id\":%llu,\"layer\":%d,\"chunk\":%lld,\"top_k\":%lld,"
        "\"n_tokens\":%lld,\"route_union\":%zu,\"hash\":%llu,\"sampled_ids\":\"%s\",\"t_us\":%lld}",
        (unsigned long long) run_id, il, (long long) chunk, (long long) n_used, (long long) n_tok,
        n_union, (unsigned long long) hash, sampled.c_str(), (long long) ggml_time_us());
}

void llama_layer_streaming::tel_act_bind(int32_t chunk_id, int slab, size_t off_bytes, int64_t n_tokens) {
    tel_write(tel_out, "{\"event\":\"act_bind\",\"run_id\":%llu,\"chunk\":%d,\"slab\":%d,\"off_bytes\":%zu,\"n_bytes\":%zu,\"t_us\":%lld}",
        (unsigned long long) run_id, chunk_id, slab, off_bytes,
        (size_t) n_tokens * n_embd_io * ggml_type_size(act_slab_dtype),
        (long long) ggml_time_us());
}

void llama_layer_streaming::tel_graph_footprint(int64_t n_graph_sets, int64_t total_graphs, size_t io_reserved_bytes,
                                                size_t io_used_bytes, size_t host_input_bytes, size_t host_input_used_bytes,
                                                size_t act_slab_bytes, size_t slot_vram_bytes, int64_t graph_budget_mib) {
    tel_write(tel_out,
        "{\"event\":\"graph_footprint\",\"run_id\":%llu,\"n_graph_sets\":%lld,\"total_graphs\":%lld,"
        "\"io_ctx_reserved_bytes\":%zu,\"io_ctx_used_bytes\":%zu,"
        "\"host_input_bytes\":%zu,\"host_input_used_bytes\":%zu,"
        "\"act_slab_bytes\":%zu,\"slot_vram_bytes\":%zu,\"graph_budget_mib\":%lld,\"t_us\":%lld}",
        (unsigned long long) run_id, (long long) n_graph_sets, (long long) total_graphs,
        io_reserved_bytes, io_used_bytes, host_input_bytes, host_input_used_bytes,
        act_slab_bytes, slot_vram_bytes, (long long) graph_budget_mib, (long long) ggml_time_us());
}

void llama_layer_streaming::tel_slot_evict(int prior_layer, int slot, uint32_t generation) {
    tel_write(tel_out, "{\"event\":\"slot_evict\",\"run_id\":%llu,\"layer\":%d,\"slot\":%d,\"generation\":%u,\"t_us\":%lld}",
        (unsigned long long) run_id, prior_layer, slot, generation, (long long) ggml_time_us());
}

// T-GPLR-28: per-pass marker for --pp-repeat (in-process PP repeat). Emitted
// at the start of passes 2..N, after the KV clear and stats reset, so the
// trace's subsequent copy/compute/phase_summary events attribute cleanly to
// this pass. Pass 1 is the ordinary run start and needs no marker.
void llama_layer_streaming::tel_pp_pass(int pass_index, int pp_tokens) {
    if (tel_out == nullptr) {
        return;
    }
    tel_write(tel_out, "{\"event\":\"pp_pass\",\"run_id\":%llu,\"pass\":%d,\"pp_tokens\":%d,\"t_us\":%lld}",
        (unsigned long long) run_id, pass_index, pp_tokens, (long long) ggml_time_us());
}

// T-GPLR-41: one event per chunk in the layer-major mctx loop (after apply,
// before the sweep). Emitted for BOTH passes under --pp-mctx-recon so pass-1
// and pass-2 lifecycle facts are directly comparable; the crashing run still
// writes every pass-2 event because the sweep has not started yet.
void llama_layer_streaming::tel_mctx_lifecycle(int pass_index, int64_t chunk, int64_t n_chunks,
                                               int64_t n_tokens, uint32_t live_n_kv, int32_t rs_z,
                                               int32_t set_chunk_id, int32_t set_rs_z, int32_t set_n_out) {
    if (tel_out == nullptr) {
        return;
    }
    tel_write(tel_out,
        "{\"event\":\"mctx\",\"run_id\":%llu,\"pass\":%d,\"chunk\":%lld,\"n_chunks\":%lld,"
        "\"n_tokens\":%lld,\"live_n_kv\":%u,\"rs_z\":%d,\"set_chunk_id\":%d,"
        "\"set_rs_z\":%d,\"set_n_out\":%d,\"t_us\":%lld}",
        (unsigned long long) run_id, pass_index, (long long) chunk, (long long) n_chunks,
        (long long) n_tokens, live_n_kv, rs_z, set_chunk_id, set_rs_z, set_n_out,
        (long long) ggml_time_us());
}

// T-GPLR-41: one event per exec layer per chunk: the prebuilt graph's baked
// KV-view shapes (ne[2] == the n_kv the view was built with) plus the KV
// storage device addresses the views point at. A pass-2 change in either is
// the direct fingerprint of a stale-shape or stale-pointer mechanism.
void llama_layer_streaming::tel_mctx_kvview(int pass_index, int64_t chunk, int il,
                                            long long k_ne2, long long v_ne2,
                                            const void * k_storage, const void * v_storage,
                                            long long k_view_src, long long v_view_src) {
    if (tel_out == nullptr) {
        return;
    }
    tel_write(tel_out,
        "{\"event\":\"mctx_kvview\",\"run_id\":%llu,\"pass\":%d,\"chunk\":%lld,\"layer\":%d,"
        "\"k_ne2\":%lld,\"v_ne2\":%lld,\"k_storage\":\"0x%llx\",\"v_storage\":\"0x%llx\","
        "\"k_view_src\":\"0x%llx\",\"v_view_src\":\"0x%llx\",\"t_us\":%lld}",
        (unsigned long long) run_id, pass_index, (long long) chunk, il,
        k_ne2, v_ne2, (unsigned long long)(uintptr_t) k_storage, (unsigned long long)(uintptr_t) v_storage,
        (unsigned long long)(uintptr_t) k_view_src, (unsigned long long)(uintptr_t) v_view_src,
        (long long) ggml_time_us());
}

// T-GPLR-01: per-layer adapter/policy/member/byte facts for host-side parity.
// members includes null-skipped entries (catalog parity); bytes counts only
// the non-null members actually copied.
void llama_layer_streaming::tel_pkg_catalog() {
    if (tel_out == nullptr || adapter == nullptr) {
        return;
    }
    for (int il = 0; il < n_layer; ++il) {
        const llama_ls_package & pkg = pkg_cache[il];
        tel_write(tel_out,
            "{\"event\":\"pkg_member\",\"run_id\":%llu,\"adapter\":\"%s\",\"policy\":\"%s\","
            "\"layer\":%d,\"cls\":\"%s\",\"members\":%zu,\"bytes\":%zu,\"t_us\":%lld}",
            (unsigned long long) run_id, adapter->name, pkg.policy, il, pkg.cls,
            pkg.members.size(), package_bytes(il), (long long) ggml_time_us());
    }
}

// T-GPLR-54 A2: the per-step top-1/top-2 readout. One line per greedy step for
// the first tg_logit_margin steps of a run, gated at the emit site on N > 0 and
// step < N; off by default, so a default run's trace is byte-identical. The
// values are read-only annotations of the pick (the same penalized top-1 and
// top-2 the pick consumed); nothing here touches sampler state or arithmetic.
void llama_layer_streaming::tel_tg_logits(int step, int64_t top1_id, float top1_logit,
                                          int64_t top2_id, float top2_logit) {
    if (tel_out == nullptr || tg_logit_margin <= 0 || step >= tg_logit_margin) {
        return;
    }
    tel_write(tel_out,
        "{\"event\":\"tg_logits\",\"run_id\":%llu,\"step\":%d,\"top1_id\":%lld,\"top1_logit\":%.6f,"
        "\"top2_id\":%lld,\"top2_logit\":%.6f,\"margin\":%.6f,\"t_us\":%lld}",
        (unsigned long long) run_id, step, (long long) top1_id, (double) top1_logit,
        (long long) top2_id, (double) top2_logit, (double) (top1_logit - top2_logit),
        (long long) ggml_time_us());
}

void llama_layer_streaming::tel_tg_tokens(const std::vector<int64_t> & ids) {
    if (tel_out == nullptr) {
        return;
    }
    std::string out = "{\"event\":\"tg_tokens\",\"run_id\":" +
                      std::to_string(run_id) + ",\"n_tokens_generated\":" +
                      std::to_string(ids.size()) + ",\"ids\":[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            out += ",";
        }
        out += std::to_string(ids[i]);
    }
    out += "]}";
    tel_write(tel_out, out.c_str());
}

void llama_layer_streaming::tel_error(const char * msg) {
    tel_write(tel_out, "{\"event\":\"error\",\"run_id\":%llu,\"message\":\"%s\",\"t_us\":%lld}",
        (unsigned long long) run_id, msg, (long long) ggml_time_us());
}

void llama_layer_streaming::tel_mtp(int64_t n_tokens, int64_t drafted, int64_t accepted) {
    if (tel_out == nullptr) {
        return;
    }
    const double accept_rate = drafted > 0 ? (double) accepted / (double) drafted : 0.0;
    tel_write(tel_out,
        "{\"event\":\"mtp\",\"run_id\":%llu,\"tokens\":%lld,\"mtp_drafted\":%lld,"
        "\"mtp_accepted\":%lld,\"accept_rate\":%.4f}",
        (unsigned long long) run_id, (long long) n_tokens, (long long) drafted, (long long) accepted, accept_rate);
}

void llama_layer_streaming::tel_phase_summary(const char * phase, int64_t tokens, int64_t wall_us, uint64_t builds_at_phase_start, int64_t graph_prep_us) {
    double total_transfer = 0.0;
    double total_wait     = 0.0;
    double total_compute  = 0.0;
    uint64_t n_layers_run = 0;
    for (int il = 0; il < n_layer; ++il) {
        const layer_stats & st = stats[il];
        if (st.n == 0) {
            continue;
        }
        total_transfer += st.transfer_us;
        total_wait     += st.wait_us;
        total_compute  += st.compute_us;
        n_layers_run   += 1;
    }
    const double tokens_per_s = wall_us > 0 ? (double) tokens * 1e6 / (double) wall_us : 0.0;
    const uint64_t graph_builds_in_phase = build_generation - builds_at_phase_start;
    const uint64_t ubatches_in_phase = n_ubatches - stats_ubatch_base;

    // T-GPLR-71 step 5: the three fields that make AC1 and AC6 scorable from a
    // DEFAULT (non-probe) trace. They are emitted here, unconditionally, rather
    // than in tel_kv_view / tel_mctx_lifecycle -- those are pw-probe-gated, so a
    // default trace could not answer AC1 at all. No behaviour reads any of this.
    //
    //   kv_view_built : the attention view width, in KV columns, that the
    //                   phase-1 DECODE set (mode 1, n_tokens 1, the standard
    //                   greedy decode graph) was built against. -1 when no such
    //                   set exists -- which is the case on every PP line under
    //                   layer-major, and that absence is itself the evidence
    //                   that no phase-1 set is built during PP.
    //   kv_view_live  : the live get_n_kv() seen at the most recent execution,
    //                   i.e. the width the coverage rule actually judged
    //                   against. AC1 is a sweep: kv_view_built / kv_view_live
    //                   ~= 1 means the view tracked the live token count;
    //                   ~= ctx/live means it still tracks the allocation.
    //   view_rebuilds_total : cumulative arm_tg_view_rebuild() firings (AC6).
    int32_t kv_view_built = -1;
    for (const auto & s : sets) {
        if (s.mode == 1 && s.chunk_id == -1 && s.n_tokens == 1 && s.built) {
            kv_view_built = s.n_kv;
            break;
        }
    }
    tel_write(tel_out,
        "{\"event\":\"phase_summary\",\"run_id\":%llu,\"phase\":\"%s\",\"tokens\":%lld,"
        "\"ubatches\":%llu,\"wall_us\":%lld,\"tokens_per_s\":%.3f,\"bytes_transferred\":%llu,\"evictions\":%llu,"
        "\"total_transfer_us\":%.1f,\"total_wait_us\":%.1f,\"total_compute_us\":%.1f,\"layers_run\":%llu,"
        "\"graph_prep_us\":%lld,\"graph_builds\":%llu,"
        "\"kv_view_built\":%d,\"kv_view_live\":%d,\"view_rebuilds_total\":%llu}",
        (unsigned long long) run_id, phase, (long long) tokens, (unsigned long long) ubatches_in_phase, (long long) wall_us,
        tokens_per_s, (unsigned long long) total_bytes_transferred, (unsigned long long) total_evictions,
        total_transfer, total_wait, total_compute, (unsigned long long) n_layers_run,
        (long long) graph_prep_us, (unsigned long long) graph_builds_in_phase,
        kv_view_built, last_live_n_kv, (unsigned long long) view_rebuild_ctr);
}

void llama_layer_streaming::reset_stats() {
    for (auto & st : stats) {
        st.n = 0;
        st.transfer_us = st.wait_us = st.compute_us = st.setup_us = st.wall_us = 0.0;
        st.transfer_min = st.wait_min = st.compute_min = 1e30;
        st.transfer_max = st.wait_max = st.compute_max = 0.0;
        st.n_ready_at_consumption = 0;
        st.n_wait_at_consumption = 0;
    }
    total_bytes_transferred = 0;
    total_evictions = 0;
    stats_ubatch_base = n_ubatches;
    rowset_last_submit_us = rowset_last_transfer_us = rowset_last_ready_us = 0.0;
    // T-GPLR-41 note: pp_mctx_pass_ctr is deliberately NOT reset here — the
    // runner calls reset_stats() at every pass boundary INCLUDING before pass
    // 2, so a monotonic process-wide counter is the only correct pass number.
}

// ---------------------------------------------------------------------------
// handoff (T-Q38HG-02 / T-Q38HG-04 / T-Q38HG-06)
// ---------------------------------------------------------------------------

// T-GPLR-16: partial handoff (NCMOE-style mixed placement) for a
// merge-resident (moe-expert-bank) policy. Same migration machinery as
// begin_handoff's merge-resident branch (non-package members alias the
// resident model layer; package members become dev tensors aliasing one
// handoff buffer), but the migration list is the bank packages with
// il >= partial_from ONLY. Lower layers keep their host-mmap CPU banks;
// during TG (phase 1) slot_layer() routes them there so the scheduler runs
// their FFN on CPU (the T-Q38HG-04 mixed-execution precedent, inside one
// layer). During PP the slot window banks every layer unchanged — the CPU
// routing is phase-gated in slot_layer()/load_layer(), never a PP behavior.
bool llama_layer_streaming::begin_handoff_partial(int partial_from_) {
    if (handoff_active) {
        error_msg = "begin_handoff_partial: handoff already active";
        return false;
    }
    if (partial_from_ < 0 || partial_from_ > n_layer) {
        error_msg = "begin_handoff_partial: partial_from " + std::to_string(partial_from_) +
                    " out of range [0, " + std::to_string(n_layer) + "]";
        return false;
    }
    if (partial_from_ == n_layer) {
        // Every bank migrates: this is exactly the full handoff.
        return begin_handoff(false, false);
    }
    if (partial_from_ == 0) {
        error_msg = "begin_handoff_partial: partial_from 0 leaves no bank resident; "
                    "the slot window or full handoff covers that shape";
        return false;
    }

    handoff_partial = true;
    handoff_partial_from = partial_from_;

    ggml_backend_buffer_type_t buft_cuda = ggml_backend_cuda_buffer_type(device);

    // Byte accounting: only the migrating banks (il >= partial_from).
    size_t total_bytes = 0;
    for (int il = partial_from_; il < n_layer; ++il) {
        handoff_bytes[il] = package_bytes(il);
        total_bytes = (total_bytes + 255) / 256 * 256 + handoff_bytes[il];
    }
    total_bytes = (total_bytes + 4095) / 4096 * 4096;

    // T-GPLR-10 peak-order rule (inherited): release the slot buffers and
    // activation slabs BEFORE the handoff allocation. Post-PP the slot
    // machinery is idle (TG graphs rebuild against handoff bundles or host
    // tensors, never slots/slabs), so the post-handoff peak is resident
    // classes + KV/state + the partial handoff buffer.
    size_t released_slot_bytes = 0;
    size_t released_slab_bytes = 0;
    ls_stage_drain();
    for (auto & slot : slots) {
        if (slot.buf) {
            released_slot_bytes += ggml_backend_buffer_get_size(slot.buf.get());
            slot.buf.reset();
        }
        slot.buf_staging.reset();
        ls_unregister_slot_source(slot);
    }
    unregister_all_pkg_sources();
    if (buf_slab) {
        released_slab_bytes = ggml_backend_buffer_get_size(buf_slab.get());
        buf_slab.reset();
    }

    tel_write(tel_out,
        "{\"event\":\"handoff\",\"run_id\":%llu,\"action\":\"release\","
        "\"slot_bytes\":%zu,\"slab_bytes\":%zu,\"t_us\":%lld}",
        (unsigned long long) run_id, released_slot_bytes, released_slab_bytes,
        (long long) ggml_time_us());

    buf_handoff.reset(ggml_backend_buft_alloc_buffer(buft_cuda, total_bytes));
    if (!buf_handoff) {
        error_msg = "begin_handoff_partial: failed to allocate partial handoff buffer (" +
                    std::to_string(total_bytes) + " bytes)";
        abort_handoff_partial();
        return false;
    }
    size_t free_vram_after_handoff = 0, total_vram_after_handoff = 0;
    ggml_backend_cuda_get_device_memory(device, &free_vram_after_handoff, &total_vram_after_handoff);

    ggml_init_params meta_params = { 4u*1024u*1024u, nullptr, true };
    ctx_handoff = ggml_init(meta_params);
    if (ctx_handoff == nullptr) {
        error_msg = "begin_handoff_partial: failed to create handoff meta context";
        abort_handoff_partial();
        return false;
    }

    char * base = (char *) ggml_backend_buffer_get_base(buf_handoff.get());
    size_t layer_off = 0;

    const int64_t t_copy0 = ggml_time_us();
    int n_handoff_layers = 0;

    for (int il = partial_from_; il < n_layer; ++il) {
        llama_layer & dst = handoff_layers[il];
        const llama_ls_package & pkg = pkg_cache.at(il);

        // Merge-resident bundle rule (inherited from begin_handoff): the
        // bundle starts as a copy of the resident model layer so non-package
        // members (attention/router/shared-expert/norms) keep their CUDA
        // tensor pointers; only the package members are replaced by tensors
        // aliasing the handoff buffer.
        dst = *host_layers[il];

        for (const auto & m : pkg.members) {
            const ggml_tensor * src_t = m.src;
            if (src_t == nullptr) {
                continue;
            }
            ls_dst(dst, m) = make_dev_tensor(ctx_handoff, src_t);
            if (ls_dst(dst, m) == nullptr) {
                error_msg = "begin_handoff_partial: failed to create handoff tensor for layer " +
                            std::to_string(il);
                abort_handoff_partial();
                return false;
            }
        }

        layer_off = (layer_off + 255) / 256 * 256;
        size_t cursor = layer_off;
        for (const auto & m : pkg.members) {
            ggml_tensor * t = ls_dst(dst, m);
            if (t == nullptr) {
                continue;
            }
            ggml_backend_tensor_alloc(buf_handoff.get(), t, base + cursor);
            cursor += ggml_backend_buft_get_alloc_size(buft_cuda, t);
        }

        for (const auto & m : pkg.members) {
            const ggml_tensor * src_t = m.src;
            ggml_tensor       * dst_t = ls_dst(dst, m);
            if (src_t == nullptr) {
                continue;
            }
            ggml_backend_tensor_set_async(backend_copy, dst_t, src_t->data, 0, ggml_nbytes(src_t));
        }

        layer_off = cursor;
        n_handoff_layers++;
    }

    // Synchronize the copy stream
    ls_stage_drain();

    const int64_t t_copy1 = ggml_time_us();
    const double copy_s = (double)(t_copy1 - t_copy0) / 1e6;

    handoff_active = true;

    // Invalidate all TG graph sets (mode == 1) so they rebuild with the
    // partial handoff bundles (same rule as begin_handoff; PP sets are
    // untouched — the PP phase already completed).
    for (auto & set : sets) {
        if (set.mode == 1) {
            set.built = false;
        }
    }

    tel_write(tel_out,
        "{\"event\":\"handoff\",\"run_id\":%llu,\"action\":\"migrate\","
        "\"total_bytes\":%zu,\"copy_s\":%.6f,\"layers\":%d,\"attention_only\":false,\"mtp\":false,"
        "\"partial\":true,\"partial_from\":%d,\"free_vram\":%zu}",
        (unsigned long long) run_id, total_bytes, copy_s, n_handoff_layers,
        handoff_partial_from, free_vram_after_handoff);

    return true;
}

// T-GPLR-16: loud abort for a failed partial handoff. Clears every state bit
// begin_handoff_partial set. The slot buffers and activation slab released
// by the peak-order rule are NOT restorable without a full re-init; the
// runner treats a begin_handoff_partial failure as terminal (exit 1) rather
// than continuing with a half-released slot window.
void llama_layer_streaming::abort_handoff_partial() {
    handoff_bytes.clear();
    handoff_active = false;
    buf_handoff.reset();
    if (ctx_handoff != nullptr) {
        ggml_free(ctx_handoff);
        ctx_handoff = nullptr;
    }
    handoff_layers.clear();
    handoff_partial = false;
    handoff_partial_from = 0;
}

bool llama_layer_streaming::begin_handoff(bool attention_only, bool mtp) {
    if (handoff_active) {
        error_msg = "begin_handoff: handoff already active";
        return false;
    }
    // T-GPLR-10: a merge-resident policy (moe-expert-bank) pages exactly one
    // package class -- the complete routed-expert bank per logical layer -- so
    // a FULL handoff migrates the whole pkg_cache list (all 40 bank packages)
    // with the same migrate_layer lambda and byte accounting as dense. Partial
    // (attention-only) handoff has no meaning for it: the attention/recurrent
    // split does not exist inside the bank policy, so it must fail loudly
    // instead of migrating a silent subset.
    if (adapter != nullptr && adapter->merge_resident) {
        if (attention_only) {
            error_msg = "begin_handoff: attention-only handoff is not supported for the " +
                        std::string(adapter->policy) + " residency policy";
            return false;
        }
        // full handoff under merge_resident: migrate the complete package
        // list (the expert banks) verbatim, exactly like dense migrates all
        // 64 dense packages.
    }

    handoff_attention_only = attention_only;

    ggml_backend_buffer_type_t buft_cuda = ggml_backend_cuda_buffer_type(device);

    // Calculate total buffer size needed (all 64 for full, 16 attn for
    // constrained, +MTP layer when mtp).
    size_t total_bytes = 0;
    for (int il = 0; il < n_layer; ++il) {
        if (attention_only && recr[il]) {
            continue; // recurrent layers stay in slot bundles, never in handoff
        }
        handoff_bytes[il] = package_bytes(il);
        total_bytes = (total_bytes + 255) / 256 * 256 + handoff_bytes[il];
    }
    // Clear any partial handoff state so an early return cannot leave a stale
    // MTP byte/handler behind for a later begin_handoff(mtp=false) call.
    auto abort_handoff = [&]() {
        handoff_bytes.clear();
        handoff_layers.clear();
        handoff_pkg_mtp = llama_ls_package{};
        handoff_mtp = false;
        mtp_bytes = 0;
        buf_handoff.reset();
        if (ctx_handoff != nullptr) {
            ggml_free(ctx_handoff);
            ctx_handoff = nullptr;
        }
    };

    if (mtp) {
        if (adapter == nullptr || adapter->mtp_package == nullptr) {
            error_msg = "begin_handoff: adapter provides no MTP package";
            abort_handoff();
            return false;
        }
        if ((int) host_layers.size() <= n_layer) {
            error_msg = "begin_handoff: MTP layer requested but not present in the model";
            abort_handoff();
            return false;
        }
        std::string perr;
        llama_ls_package mtp_pkg;
        if (!adapter->mtp_package(model_ref, *host_layers[n_layer], mtp_pkg, perr) ||
            !llama_ls_validate_package(mtp_pkg, perr, /*require_src=*/false)) {
            error_msg = "begin_handoff: invalid MTP package: " + perr;
            abort_handoff();
            return false;
        }
        handoff_mtp = true;
        handoff_pkg_mtp = std::move(mtp_pkg);
        size_t bytes = 0;
        // Probe allocation sizes using a throwaway meta tensor of matching
        // shape/type (the real tensors are created inside ctx_handoff below).
        ggml_context * probe_ctx = ggml_init({ 6u*1024u*1024u, nullptr, true });
        if (probe_ctx == nullptr) {
            error_msg = "begin_handoff: failed to create probe context";
            abort_handoff();
            return false;
        }
        for (const auto & m : handoff_pkg_mtp.members) {
            const ggml_tensor * t = m.src;
            if (t == nullptr) {
                continue;
            }
            ggml_tensor * probe = make_dev_tensor(probe_ctx, t);
            if (probe == nullptr) {
                error_msg = "begin_handoff: failed to probe MTP tensor";
                ggml_free(probe_ctx);
                abort_handoff();
                return false;
            }
            bytes += ggml_backend_buft_get_alloc_size(buft_cuda, probe);
        }
        ggml_free(probe_ctx);
        handoff_bytes[n_layer] = bytes;
        mtp_bytes = bytes;
        total_bytes = (total_bytes + 255) / 256 * 256 + bytes;
    }
    total_bytes = (total_bytes + 4095) / 4096 * 4096;

    // T-GPLR-10 peak-order rule: the handoff buffer coexists with the
    // resident classes + KV/state only, never with the slot window. After the
    // PP phase every slot consumer is done and the layer-major slabs are idle
    // (the post-handoff TG graphs rebuild against the handoff bundles, never
    // against slots/slabs), so the slot buffers and activation slabs are
    // released BEFORE the handoff allocation: peak = resident classes +
    // KV/state + handoff buffer, never their sum with the slot window. The
    // release is skipped for attention-only handoff, where the constrained
    // dense path keeps its slot machinery live (T-Q38HG-04); under a
    // merge-resident policy attention-only is refused above, so the release
    // always fires on the MoE bank handoff. The event lets the host checker
    // verify the release order in telemetry.
    size_t released_slot_bytes = 0;
    size_t released_slab_bytes = 0;
    if (!attention_only) {
        ls_stage_drain();
        for (auto & slot : slots) {
            if (slot.buf) {
                released_slot_bytes += ggml_backend_buffer_get_size(slot.buf.get());
                slot.buf.reset();
            }
            slot.buf_staging.reset();
            ls_unregister_slot_source(slot);
        }
        unregister_all_pkg_sources();
        if (buf_slab) {
            released_slab_bytes = ggml_backend_buffer_get_size(buf_slab.get());
            buf_slab.reset();
        }
        tel_write(tel_out,
            "{\"event\":\"handoff\",\"run_id\":%llu,\"action\":\"release\","
            "\"slot_bytes\":%zu,\"slab_bytes\":%zu,\"t_us\":%lld}",
            (unsigned long long) run_id, released_slot_bytes, released_slab_bytes,
            (long long) ggml_time_us());
    }

    // Allocate the device buffer
    buf_handoff.reset(ggml_backend_buft_alloc_buffer(buft_cuda, total_bytes));
    if (!buf_handoff) {
        error_msg = "begin_handoff: failed to allocate handoff buffer (" +
                    std::to_string(total_bytes) + " bytes)";
        abort_handoff();
        return false;
    }
    // T-GPLR-10: free VRAM right after the handoff allocation -- the true
    // post-handoff peak (resident classes + KV/state + handoff buffer) is
    // total_vram - this reading; the run_end peak-vram metric samples after
    // end_handoff and therefore cannot see the handoff buffer.
    size_t free_vram_after_handoff = 0, total_vram_after_handoff = 0;
    ggml_backend_cuda_get_device_memory(device, &free_vram_after_handoff, &total_vram_after_handoff);

    // Create meta context for handoff tensor objects
    ggml_init_params meta_params = { 4u*1024u*1024u, nullptr, true };
    ctx_handoff = ggml_init(meta_params);
    if (ctx_handoff == nullptr) {
        error_msg = "begin_handoff: failed to create handoff meta context";
        abort_handoff();
        return false;
    }

    // Create per-layer tensor bundles and copy data
    char * base = (char *) ggml_backend_buffer_get_base(buf_handoff.get());
    size_t layer_off = 0;

    const int64_t t_copy0 = ggml_time_us();
    int n_handoff_layers = 0;

    auto migrate_layer = [&](int il, bool is_mtp) -> bool {
        llama_layer & dst = handoff_layers[il];
        const llama_ls_package & pkg = is_mtp ? handoff_pkg_mtp : pkg_cache.at(il);

        // T-GPLR-10: under a merge-resident policy (moe-expert-bank) the
        // package covers only the routed-expert bank, so the handoff bundle
        // starts as a copy of the resident model layer -- the non-package
        // members (attention/router/shared-expert/norms) keep their existing
        // CUDA-resident tensor pointers, exactly like the slot bundles do in
        // create_bundle -- and only the package members are replaced by
        // tensors aliasing the handoff buffer. Without this the post-handoff
        // TG graph rebuild would bind null non-bank members (segfault).
        if (adapter != nullptr && adapter->merge_resident) {
            dst = *host_layers[il];
        }

        // Create tensor objects in ctx_handoff
        for (const auto & m : pkg.members) {
            const ggml_tensor * src_t = m.src;
            if (src_t == nullptr) {
                // a pre-copied merge-resident bundle legitimately owns the
                // member tensor from the resident model layer
                if (adapter == nullptr || !adapter->merge_resident) {
                    GGML_ASSERT(ls_dst(dst, m) == nullptr && "handoff bundle null mismatch");
                }
                continue;
            }
            ls_dst(dst, m) = make_dev_tensor(ctx_handoff, src_t);
            if (ls_dst(dst, m) == nullptr) {
                error_msg = "begin_handoff: failed to create handoff tensor for layer " + std::to_string(il);
                return false;
            }
        }

        // Collect and allocate (aligned per layer)
        layer_off = (layer_off + 255) / 256 * 256;
        size_t cursor = layer_off;
        for (const auto & m : pkg.members) {
            ggml_tensor * t = ls_dst(dst, m);
            if (t == nullptr) {
                continue;
            }
            ggml_backend_tensor_alloc(buf_handoff.get(), t, base + cursor);
            cursor += ggml_backend_buft_get_alloc_size(buft_cuda, t);
        }

        // Async copy from host to device
        for (const auto & m : pkg.members) {
            const ggml_tensor * src_t = m.src;
            ggml_tensor       * dst_t = ls_dst(dst, m);
            if (src_t == nullptr) {
                continue;
            }
            ggml_backend_tensor_set_async(backend_copy, dst_t, src_t->data, 0, ggml_nbytes(src_t));
        }

        layer_off = cursor;
        n_handoff_layers++;
        return true;
    };

    for (int il = 0; il < n_layer; ++il) {
        if (attention_only && recr[il]) {
            continue; // recurrent layers stay in slot bundles
        }
        if (!migrate_layer(il, false)) {
            abort_handoff();
            return false;
        }
    }
    if (mtp) {
        if (!migrate_layer(n_layer, true)) {
            abort_handoff();
            return false;
        }
    }

    // Synchronize the copy stream
    ls_stage_drain();

    const int64_t t_copy1 = ggml_time_us();
    const double copy_s = (double)(t_copy1 - t_copy0) / 1e6;

    handoff_active = true;

    // Invalidate all TG graph sets (mode == 1) so they rebuild with handoff
    // tensors via slot_layer() on the first post-handoff decode.
    // begin_set() handles the reset in place (built = false takes the non-
    // replacement path even when the T-GPLR-47 flag is set, so a handoff
    // rebuild keeps today's behavior).
    for (auto & set : sets) {
        if (set.mode == 1) {
            set.built = false;
        }
    }

    // Emit telemetry
    tel_write(tel_out,
        "{\"event\":\"handoff\",\"run_id\":%llu,\"action\":\"migrate\","
        "\"total_bytes\":%zu,\"copy_s\":%.6f,\"layers\":%d,\"attention_only\":%s,\"mtp\":%s,"
        "\"free_vram\":%zu}",
        (unsigned long long) run_id, total_bytes, copy_s, n_handoff_layers,
        attention_only ? "true" : "false", mtp ? "true" : "false",
        free_vram_after_handoff);

    return true;
}

void llama_layer_streaming::end_handoff() {
    if (!handoff_active) {
        return;
    }

    buf_handoff.reset();
    if (ctx_handoff != nullptr) {
        ggml_free(ctx_handoff);
        ctx_handoff = nullptr;
    }
    handoff_layers.clear();
    handoff_bytes.clear();
    handoff_active = false;
    handoff_mtp = false;
    mtp_bytes = 0;
    // T-GPLR-16: clear the partial handoff state with everything else.
    handoff_partial = false;
    handoff_partial_from = 0;

    tel_write(tel_out,
        "{\"event\":\"handoff\",\"run_id\":%llu,\"action\":\"teardown\"}",
        (unsigned long long) run_id);
}

// ---------------------------------------------------------------------------
// T-GPLR-13: reactive expert row-set residency (moe-row-set TG policy)
//
// Rows stage into the EXISTING full-bank-shaped slot buffers (the bundles
// alias them), so no new VRAM class appears and the standard copy-event /
// generation / consumer-done lifecycle governs reuse exactly as the SDD
// requires. Slot reuse during TG follows the executor's eviction rule: a
// slot being overwritten is first fenced with its consumer-done event, then
// its generation bumps and its row-set coverage resets -- a fresh generation
// never starts with foreign rows.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// T-GPLR-19 (F2-M1): per-layer row cache -- SDD section 4 execution shape (i),
// "cache to slot-bundle device-side refresh before the row-consuming phase".
// One PRIVATE CUDA buffer per logical layer holds cap = min(n_expert, K*W*top_k)
// expert slabs for every package member, so a layer's retained rows survive
// every other layer's turn. The shared per-slot buffers cannot do this: all
// bundles of the layers that share a slot alias one device buffer from offset
// 0, which is exactly why reactive reuse measures zero. Allocation is
// all-or-nothing and loud at bring-up -- a cache that cannot be allocated must
// never degrade into re-fetching every row while still reporting cache
// telemetry.
// ---------------------------------------------------------------------------
bool llama_layer_streaming::row_cache_layer_cost(int il, int cap, ggml_context * meta,
                                                 std::vector<ggml_tensor *> * members, size_t * out_cost) const {
    const ggml_backend_buffer_type_t buft_cuda = ggml_backend_cuda_buffer_type(device);
    const llama_ls_package & pkg = pkg_cache.at((size_t) il);

    size_t need = 4096;
    for (const auto & m : pkg.members) {
        const ggml_tensor * t = m.src;
        if (t == nullptr) {
            if (members != nullptr) {
                members->push_back(nullptr);
            }
            continue;
        }
        // [ne0, ne1, cap]: position p occupies p * nb[2], so each retained
        // row is addressable at its own offset and the slot refresh can
        // copy it slab-by-slab without any host round trip.
        ggml_tensor * ct = ggml_new_tensor_4d(meta, t->type, t->ne[0], t->ne[1], cap, 1);
        if (ct == nullptr) {
            return false;
        }
        need += ggml_backend_buft_get_alloc_size(buft_cuda, ct) + 4096;
        if (members != nullptr) {
            ggml_set_name(ct, t->name);
            members->push_back(ct);
        }
    }

    *out_cost = need;
    return true;
}

size_t llama_layer_streaming::row_cache_bytes_for(int k) const {
    if (adapter == nullptr || !adapter->merge_resident) {
        return 0; // unsupported policy: begin_row_cache() reports that itself
    }

    // Same capacity rule the allocation uses: cap = min(n_expert, K*W*top_k)
    // with W == 1 (the telemetry validator rejects window_tokens != 1).
    const int n_exp = (int) model_ref->hparams.n_expert;
    const int top_k = (int) model_ref->hparams.n_expert_used;
    const int cap   = std::min(n_exp, (k < 1 ? 1 : k) * top_k);

    // one throwaway metadata context: the tensors never receive a buffer, so
    // nothing is allocated on the device and no cache state is touched
    ggml_init_params meta_params = { (size_t) 4 * 1024 * 1024, nullptr, true };
    ggml_context * meta = ggml_init(meta_params);
    if (meta == nullptr) {
        return 0;
    }

    size_t total = 0;
    for (int il = 0; il < n_layer; ++il) {
        size_t cost = 0;
        if (!row_cache_layer_cost(il, cap, meta, nullptr, &cost)) {
            ggml_free(meta);
            return 0;
        }
        total += cost;
    }
    ggml_free(meta);
    return total;
}

size_t llama_layer_streaming::row_set_spec_pool_bytes_for(int cap) const {
    if (adapter == nullptr || !adapter->merge_resident || cap <= 0) {
        return 0;
    }
    const int n_exp = (int) model_ref->hparams.n_expert;
    if (cap > n_exp) {
        return 0;
    }
    ggml_init_params meta_params = { (size_t) 4 * 1024 * 1024, nullptr, true };
    ggml_context * meta = ggml_init(meta_params);
    if (meta == nullptr) {
        return 0;
    }
    size_t total = 0;
    for (int il = 0; il < n_layer; ++il) {
        size_t cost = 0;
        if (!row_cache_layer_cost(il, cap, meta, nullptr, &cost)) {
            ggml_free(meta);
            return 0;
        }
        total += cost;
    }
    ggml_free(meta);
    return total;
}

bool llama_layer_streaming::begin_row_pool() {
    if (row_set_spec_pool <= 0) {
        return true;
    }
    if (row_set_spec_pool > (int) model_ref->hparams.n_expert) {
        error_msg = "begin_row_pool: capacity exceeds the model's expert count";
        return false;
    }
    if (row_set_spec_mode != 2 || !row_set_spec_on) {
        error_msg = "begin_row_pool: the prediction pool requires configured mode 2";
        return false;
    }
    if (row_set_spec_budget > row_set_spec_pool) {
        error_msg = "begin_row_pool: proposal budget exceeds prediction-pool capacity";
        return false;
    }
    if (pool_meta != nullptr) {
        error_msg = "begin_row_pool: pool already active";
        return false;
    }
    const ggml_backend_buffer_type_t buft_cuda = ggml_backend_cuda_buffer_type(device);
    ggml_init_params params = { (size_t) 4 * 1024 * 1024, nullptr, true };
    pool_meta = ggml_init(params);
    if (pool_meta == nullptr) {
        error_msg = "begin_row_pool: failed to create pool tensor context";
        return false;
    }
    pool_bufs.resize((size_t) n_layer);
    pool_tens.assign((size_t) n_layer, {});
    pool_expert.assign((size_t) n_layer, std::vector<int32_t>((size_t) row_set_spec_pool, -1));
    pool_state.assign((size_t) n_layer, std::vector<uint8_t>((size_t) row_set_spec_pool, LLAMA_LS_SPEC_FREE));
    pool_recency.assign((size_t) n_layer, std::vector<int64_t>((size_t) row_set_spec_pool, -1));
    pool_gen.assign((size_t) n_layer, std::vector<uint32_t>((size_t) row_set_spec_pool, 0));
    pool_issue_ord.assign((size_t) n_layer, std::vector<int32_t>((size_t) row_set_spec_pool, -1));
    pool_calls.assign((size_t) n_layer, 0);
    size_t total = 0;
    for (int il = 0; il < n_layer; ++il) {
        std::vector<ggml_tensor *> members;
        size_t need = 0;
        if (!row_cache_layer_cost(il, row_set_spec_pool, pool_meta, &members, &need)) {
            error_msg = "begin_row_pool: failed to create pool tensor for layer " + std::to_string(il);
            end_row_pool();
            return false;
        }
        pool_tens[(size_t) il] = members;
        pool_bufs[(size_t) il].reset(ggml_backend_buft_alloc_buffer(buft_cuda, need));
        if (!pool_bufs[(size_t) il]) {
            error_msg = "begin_row_pool: failed to allocate pool buffer for layer " + std::to_string(il);
            end_row_pool();
            return false;
        }
        std::vector<ggml_tensor *> real;
        for (ggml_tensor * t : members) {
            if (t != nullptr) {
                real.push_back(t);
            }
        }
        size_t off = 0;
        if (!alloc_tensors(pool_bufs[(size_t) il].get(), buft_cuda, real, off)) {
            error_msg = "begin_row_pool: failed to place pool tensors for layer " + std::to_string(il);
            end_row_pool();
            return false;
        }
        total += ggml_backend_buffer_get_size(pool_bufs[(size_t) il].get());
    }
    row_set_spec_pool_alloc_bytes = total;
    tel_write(tel_out,
        "{\"event\":\"spec_pool_init\",\"run_id\":%llu,\"capacity\":%d,\"layers\":%d,"
        "\"bytes\":%zu,\"proposal_budget\":%d,\"t_us\":%lld}",
        (unsigned long long) run_id, row_set_spec_pool, n_layer, total,
        row_set_spec_budget, (long long) ggml_time_us());
    return true;
}

void llama_layer_streaming::end_row_pool() {
    for (auto & evs : pool_ev) {
        for (void * ev : evs) {
            if (ev != nullptr) {
                ggml_backend_event_synchronize((ggml_backend_event_t) ev);
            }
        }
    }
    for (auto & evs : pool_ev) {
        for (void * ev : evs) {
            if (ev != nullptr) {
                ggml_backend_event_free((ggml_backend_event_t) ev);
            }
        }
    }
    pool_ev.clear();
    pool_ev_pos.clear();
    pool_events_ready = false;
    pool_bufs.clear();
    pool_tens.clear();
    pool_expert.clear();
    pool_state.clear();
    pool_recency.clear();
    pool_gen.clear();
    pool_issue_ord.clear();
    pool_calls.clear();
    if (pool_meta != nullptr) {
        ggml_free(pool_meta);
        pool_meta = nullptr;
    }
    row_set_spec_pool_alloc_bytes = 0;
}

int llama_layer_streaming::pool_pending_of(int il) const {
    if (il < 0 || (size_t) il >= pool_state.size()) {
        return 0;
    }
    int n = 0;
    for (uint8_t st : pool_state[(size_t) il]) {
        n += st == LLAMA_LS_SPEC_PENDING;
    }
    return n;
}

int llama_layer_streaming::pool_ready_of(int il) const {
    if (il < 0 || (size_t) il >= pool_state.size()) {
        return 0;
    }
    int n = 0;
    for (uint8_t st : pool_state[(size_t) il]) {
        n += st == LLAMA_LS_SPEC_READY;
    }
    return n;
}

bool llama_layer_streaming::pool_cancel_demand(int il, int32_t expert, uint32_t generation) {
    if (il < 0 || (size_t) il >= pool_state.size()) {
        return false;
    }
    const auto & st = pool_state[(size_t) il];
    const auto & ex = pool_expert[(size_t) il];
    const auto & gn = pool_gen[(size_t) il];
    for (int p = 0; p < row_set_spec_pool; ++p) {
        if (ex[(size_t) p] == expert && st[(size_t) p] == LLAMA_LS_SPEC_PENDING &&
            gn[(size_t) p] == generation) {
            pool_state[(size_t) il][(size_t) p] = LLAMA_LS_SPEC_CANCELLED;
            ++pool_cancelled;
            ++pool_cancelled_demand;
            ++pool_pending_fallbacks_last;
            ++pool_pending_fallbacks;
            return true;
        }
    }
    return false;
}

bool llama_layer_streaming::pool_issue_one(int il, int32_t expert, int pos, uint32_t generation, int64_t tick) {
    if (il < 0 || pos < 0 || expert < 0 || (size_t) il >= pool_state.size() ||
        (size_t) pos >= pool_state[(size_t) il].size()) {
        error_msg = "pool_issue_one: invalid layer, position, or expert id";
        return false;
    }
    auto & st = pool_state[(size_t) il];
    auto & ex = pool_expert[(size_t) il];
    if (st[(size_t) pos] != LLAMA_LS_SPEC_FREE || ex[(size_t) pos] >= 0) {
        error_msg = "pool_issue_one: target position is not FREE";
        return false;
    }
    for (size_t p = 0; p < st.size(); ++p) {
        if (p != (size_t) pos && st[p] != LLAMA_LS_SPEC_FREE && ex[p] == expert) {
            error_msg = "pool_issue_one: duplicate live pool expert";
            return false;
        }
    }
    if (!pool_events_ready) {
        const ggml_backend_dev_t dev = ggml_backend_get_device(backend_copy);
        pool_ev.assign(pool_state.size(), {});
        pool_ev_pos.assign(pool_state.size(), {});
        for (size_t l = 0; l < pool_state.size(); ++l) {
            pool_ev[l].assign((size_t) row_set_spec_budget, nullptr);
            pool_ev_pos[l].assign((size_t) row_set_spec_budget, -1);
            for (int s = 0; s < row_set_spec_budget; ++s) {
                pool_ev[l][(size_t) s] = ggml_backend_event_new(dev);
                if (pool_ev[l][(size_t) s] == nullptr) {
                    error_msg = "pool_issue_one: failed to create a completion event";
                    return false;
                }
            }
        }
        pool_events_ready = true;
    }
    int slot = -1;
    for (size_t s = 0; s < pool_ev_pos[(size_t) il].size(); ++s) {
        if (pool_ev_pos[(size_t) il][s] < 0) { slot = (int) s; break; }
    }
    if (slot < 0) {
        error_msg = "pool_issue_one: no free completion-event slot";
        return false;
    }
    const llama_ls_package & pkg = pkg_cache.at((size_t) il);
    const auto & mem = pool_tens[(size_t) il];
    if (mem.size() != pkg.members.size()) {
        error_msg = "pool_issue_one: pool tensor descriptor does not match the package";
        return false;
    }
    size_t bytes = 0;
    int copied_members = 0;
    for (size_t mi = 0; mi < pkg.members.size(); ++mi) {
        const ggml_tensor * src = pkg.members[mi].src;
        const ggml_tensor * dst = mem[mi];
        if (src == nullptr) {
            if (dst != nullptr) {
                error_msg = "pool_issue_one: non-null pool tensor has no package source";
                return false;
            }
            continue;
        }
        if (dst == nullptr || src->data == nullptr || dst->data == nullptr) {
            error_msg = "pool_issue_one: package source or pool tensor has no storage";
            return false;
        }
        const size_t slab = llama_ls_expert_slab_bytes(src->type, src->ne[0], src->ne[1]);
        const size_t src_bytes = ggml_nbytes(src);
        const size_t dst_bytes = ggml_nbytes(dst);
        if (slab == 0 || expert >= src->ne[2] || slab > src_bytes || slab > dst_bytes ||
            (size_t) expert > (src_bytes - slab) / slab ||
            (size_t) pos > (dst_bytes - slab) / slab) {
            error_msg = "pool_issue_one: expert slab is outside the source or destination tensor";
            return false;
        }
        ggml_backend_tensor_set_async(backend_copy, (ggml_tensor *) dst,
                                      (const char *) src->data + (size_t) expert * slab,
                                      (size_t) pos * slab, slab);
        bytes += slab;
        ++copied_members;
    }
    if (copied_members == 0) {
        error_msg = "pool_issue_one: package has no copyable expert members";
        return false;
    }
    ex[(size_t) pos] = expert;
    st[(size_t) pos] = LLAMA_LS_SPEC_PENDING;
    pool_recency[(size_t) il][(size_t) pos] = tick;
    pool_gen[(size_t) il][(size_t) pos] = generation;
    pool_issue_ord[(size_t) il][(size_t) pos] = ++pool_calls[(size_t) il];
    pool_ev_pos[(size_t) il][(size_t) slot] = pos;
    ggml_backend_event_record((ggml_backend_event_t) pool_ev[(size_t) il][(size_t) slot], backend_copy);
    ++pool_submitted_last;
    ++pool_submitted;
    pool_h2d_bytes_last += (int64_t) bytes;
    pool_h2d_bytes_total += (int64_t) bytes;
    return true;
}

void llama_layer_streaming::pool_reap(int il) {
    if (il < 0 || (size_t) il >= pool_ev.size()) {
        return;
    }
    auto & st = pool_state[(size_t) il];
    auto & evs = pool_ev[(size_t) il];
    auto & evp = pool_ev_pos[(size_t) il];
    for (size_t s = 0; s < evs.size(); ++s) {
        const int p = evp[s];
        if (p < 0) {
            continue;
        }
        if (st[(size_t) p] != LLAMA_LS_SPEC_PENDING && st[(size_t) p] != LLAMA_LS_SPEC_CANCELLED) {
            evp[s] = -1;
            continue;
        }
        const int64_t t0 = ggml_time_us();
        ggml_backend_event_synchronize((ggml_backend_event_t) evs[s]);
        const int64_t waited = ggml_time_us() - t0;
        pool_reap_wait_us_last += waited;
        pool_reap_wait_us += waited;
        const uint8_t before = st[(size_t) p];
        const int completed = llama_ls_pool_complete(
            pool_expert[(size_t) il].data(), st.data(), pool_recency[(size_t) il].data(),
            pool_gen[(size_t) il].data(), row_set_spec_pool, p);
        if (completed == 1) {
            ++pool_ready;
        } else if (completed == 2) {
            pool_issue_ord[(size_t) il][(size_t) p] = -1;
        } else if (before == LLAMA_LS_SPEC_PENDING || before == LLAMA_LS_SPEC_CANCELLED) {
            error_msg = "pool_reap: completion event owned an incompatible position";
            failed = true;
            tel_error(error_msg.c_str());
        }
        evp[s] = -1;
    }
}

void llama_layer_streaming::pool_begin_generation(int il, uint32_t generation) {
    if (il < 0 || (size_t) il >= pool_state.size()) {
        return;
    }
    int stale_pending = 0;
    int stale_ready = 0;
    for (int p = 0; p < row_set_spec_pool; ++p) {
        if (pool_gen[(size_t) il][(size_t) p] == generation) {
            continue;
        }
        stale_pending += pool_state[(size_t) il][(size_t) p] == LLAMA_LS_SPEC_PENDING;
        stale_ready += pool_state[(size_t) il][(size_t) p] == LLAMA_LS_SPEC_READY;
    }
    llama_ls_pool_begin_generation(
        pool_expert[(size_t) il].data(), pool_state[(size_t) il].data(),
        pool_recency[(size_t) il].data(), pool_gen[(size_t) il].data(),
        row_set_spec_pool, generation);
    pool_cancelled += stale_pending;
    pool_cancelled_generation += stale_pending;
    pool_stale_retired += stale_ready;
}

void llama_layer_streaming::pool_cancel_generation() {
    for (int il = 0; il < (int) pool_state.size(); ++il) {
        const uint32_t generation = slots.empty() ? 0 : slots[(size_t) (il % (int) slots.size())].generation;
        pool_begin_generation(il, generation);
        pool_reap(il);
    }
}

void llama_layer_streaming::pool_reset() {
    end_row_pool();
    pool_submitted_last = pool_submitted = pool_ready = pool_cancelled = 0;
    pool_cancelled_demand = pool_cancelled_generation = pool_cancelled_capacity = pool_stale_retired = 0;
    pool_evicted_last = pool_evicted = pool_consumed_last = pool_consumed = 0;
    pool_pending_fallbacks_last = pool_pending_fallbacks = 0;
    pool_d2d_bytes_last = pool_d2d_bytes_total = 0;
    pool_h2d_bytes_last = pool_h2d_bytes_total = 0;
    pool_reap_wait_us_last = pool_reap_wait_us = 0;
    pool_candidates_last.clear();
    pool_fills_last.clear();
    pool_hit_ids_last.clear();
    pool_skipped_real_last = pool_skipped_pool_last = pool_skipped_dup_last = 0;
    pool_skipped_invalid_last = pool_refused_budget_last = 0;
    pool_refused_capacity_last = pool_evicted_last = pool_stop_last = 0;
}

void llama_layer_streaming::tel_spec_pool(int il, uint32_t generation, int pending_now, int ready_now) {
    auto ids = [](const std::vector<int32_t> & v) {
        std::string s;
        for (size_t i = 0; i < v.size(); ++i) { s += (i ? "," : ""); s += std::to_string(v[i]); }
        return s;
    };
    tel_write(tel_out,
        "{\"event\":\"spec_pool\",\"run_id\":%llu,\"ubatch\":%llu,\"layer\":%d,\"generation\":%u,"
        "\"capacity\":%d,\"proposal_budget\":%d,\"candidates\":[%s],\"fill_ids\":[%s],\"hit_ids\":[%s],"
        "\"skipped_real\":%d,\"skipped_pool\":%d,\"skipped_dup\":%d,\"skipped_invalid\":%d,"
        "\"refused_budget\":%d,\"refused_capacity\":%d,\"evicted\":%d,\"stop\":%d,"
        "\"pending\":%d,\"ready\":%d,\"submitted\":%lld,\"submitted_total\":%lld,"
        "\"ready_total\":%lld,\"cancelled_total\":%lld,\"evicted_total\":%lld,"
        "\"consumed\":%lld,\"consumed_total\":%lld,\"pending_fallbacks\":%lld,"
        "\"pending_fallbacks_total\":%lld,\"reap_wait_us\":%lld,\"reap_wait_us_total\":%lld,"
        "\"h2d_bytes\":%lld,\"h2d_bytes_total\":%lld,\"d2d_bytes\":%lld,"
        "\"d2d_bytes_total\":%lld,\"cancelled_demand_total\":%lld,"
        "\"cancelled_generation_total\":%lld,\"cancelled_capacity_total\":%lld,"
        "\"stale_retired_total\":%lld,\"allocation_bytes\":%zu,\"t_us\":%lld}",
        (unsigned long long) run_id, (unsigned long long) n_ubatches, il, generation,
        row_set_spec_pool, row_set_spec_budget,
        ids(pool_candidates_last).c_str(), ids(pool_fills_last).c_str(), ids(pool_hit_ids_last).c_str(),
        pool_skipped_real_last, pool_skipped_pool_last, pool_skipped_dup_last, pool_skipped_invalid_last,
        pool_refused_budget_last, pool_refused_capacity_last, pool_evicted_last, pool_stop_last,
        pending_now, ready_now, (long long) pool_submitted_last, (long long) pool_submitted,
        (long long) pool_ready, (long long) pool_cancelled, (long long) pool_evicted,
        (long long) pool_consumed_last, (long long) pool_consumed,
        (long long) pool_pending_fallbacks_last, (long long) pool_pending_fallbacks,
        (long long) pool_reap_wait_us_last, (long long) pool_reap_wait_us,
        (long long) pool_h2d_bytes_last, (long long) pool_h2d_bytes_total,
        (long long) pool_d2d_bytes_last, (long long) pool_d2d_bytes_total,
        (long long) pool_cancelled_demand, (long long) pool_cancelled_generation,
        (long long) pool_cancelled_capacity, (long long) pool_stale_retired,
        row_set_spec_pool_alloc_bytes, (long long) ggml_time_us());
}

//
// Default-off: with row_set_spec_mode == 0 nothing below runs, no array is
// allocated and no event is created (P6). The decisions themselves live in the
// pure llama_ls_spec_* functions; this section owns WHEN they are asked, how a
// fill becomes a real copy, and the counters the checker reads.
// ---------------------------------------------------------------------------

int llama_layer_streaming::spec_pending_of(int il) const {
    if (il < 0 || (size_t) il >= spec_state.size()) {
        return 0;
    }
    int n = 0;
    for (uint8_t st : spec_state[(size_t) il]) {
        n += (st == LLAMA_LS_SPEC_PENDING);
    }
    return n;
}

int llama_layer_streaming::spec_ready_of(int il) const {
    if (il < 0 || (size_t) il >= spec_state.size()) {
        return 0;
    }
    int n = 0;
    const auto & st = spec_state[(size_t) il];
    const auto & fl = spec_flag[(size_t) il];
    for (size_t p = 0; p < st.size(); ++p) {
        n += (st[p] == LLAMA_LS_SPEC_READY && fl[p] != 0);
    }
    return n;
}


bool llama_layer_streaming::set_row_set_window_tokens(int tokens) {
    if (tokens < 1 || tokens > 256) {
        error_msg = "set_row_set_window_tokens: width must be in [1,256]";
        return false;
    }
    if (row_set_spec_mode != 0) {
        error_msg = "set_row_set_window_tokens: must be called before set_row_set_spec()";
        return false;
    }
    row_set_window_tokens = tokens;
    return true;
}

bool llama_layer_streaming::set_row_set_spec(int mode, int budget, const std::string & script_path) {
    if (mode == 0) {
        row_set_spec_mode = 0;
        row_set_spec_on   = false;
        row_set_spec_script.clear();
        spec_script.clear();
        // mode 2's own configuration goes with it: a mode-0 run holds no proposer
        // state, so a later arm() call cannot resurrect a half-configured predictor
        row_set_spec_arm  = -1;
        row_set_spec_null = false;
        spec_pred.reset();
        d1_state.reset();
        d1_last_routes.clear();
        d1_last_candidates.clear();
        d1_shape_emitted = false;
        row_set_window_tokens = 1;
        return true;
    }
    if (mode != 1 && mode != 2) {
        error_msg = "set_row_set_spec: mode " + std::to_string(mode) +
                    " is not a speculative-placement mode (0 = off, 1 = scripted, 2 = the arm's proposer)";
        return false;
    }
    if (model_ref == nullptr) {
        error_msg = "set_row_set_spec: called before the model is available";
        return false;
    }
    if (!row_set_retain) {
        error_msg = "set_row_set_spec: speculation needs the per-layer row cache (retention K >= 1); "
                    "the reactive slot-keyed path has no per-layer residency to speculate into";
        return false;
    }
    const int n_exp = (int) model_ref->hparams.n_expert;
    const int top_k = (int) model_ref->hparams.n_expert_used;
    const int K     = row_set_retain_k < 1 ? 1 : row_set_retain_k;
    const int cap   = std::min(n_exp, K * top_k);
    const int b     = budget > 0 ? budget : top_k;
    const int bmax  = row_set_spec_pool > 0 ? row_set_spec_pool : cap;
    if (b < 1 || b > bmax) {
        error_msg = "set_row_set_spec: budget " + std::to_string(b) + " outside [1," + std::to_string(bmax) +
                    "] for K=" + std::to_string(K) + ", top_k=" + std::to_string(top_k);
        return false;
    }

    if (mode == 2) {
        // MODE 2: the proposal source is the ARM, configured by
        // set_row_set_spec_arm() -- which needs the model's per-layer family tables,
        // so it is a separate call. A script here would be a SECOND and conflicting
        // proposal source, so it is refused rather than ignored.
        if (!script_path.empty()) {
            error_msg = "set_row_set_spec: mode 2 takes its proposals from the arm, not from a script (" +
                        script_path + "); pass --residency-tg-spec-arm instead";
            return false;
        }
        row_set_spec_mode    = 2;
        row_set_spec_budget  = b;
        row_set_spec_reserve = row_set_spec_pool > 0 ? 0 : row_set_window_tokens * top_k;
        if (row_set_spec_reserve > cap) {
            error_msg = "set_row_set_spec: MTP row-set width " + std::to_string(row_set_window_tokens) +
                        " needs reserve " + std::to_string(row_set_spec_reserve) +
                        " but K=" + std::to_string(K) + " provides cap " + std::to_string(cap);
            return false;
        }
        row_set_spec_on      = true;
        row_set_spec_script.clear();
        spec_script.clear();
        row_set_spec_arm  = -1;   // until set_row_set_spec_arm() names one (begin_row_cache refuses otherwise)
        row_set_spec_null = false;
        fprintf(stderr, "tg-spec (T-GPLR-58 A1): mode 2 (proposer arm) budget %d, reserve %d of cap %d; "
                        "the arm itself is configured next, and begin_row_cache refuses a mode-2 run without one\n",
                b, row_set_spec_reserve, cap);
        return true;
    }

    if (script_path.empty()) {
        error_msg = "set_row_set_spec: mode 1 needs a proposal script (the A0 driver; mode 2 is the arm)";
        return false;
    }

    // The script is the A0 proposal source: one entry per line,
    //   <layer> <ordinal> <expert> [<expert> ...]
    // meaning "at the end of this layer's (ordinal-1)-th demand stage, speculate
    // these ids for its next demand". Ordinals are 1-based and dense by design:
    // a gap is not an error, it is simply no speculation at that stage.
    FILE * f = fopen(script_path.c_str(), "rb");
    if (f == nullptr) {
        error_msg = "set_row_set_spec: cannot open the proposal script: " + script_path;
        return false;
    }
    std::unordered_map<uint64_t, std::vector<int32_t>> parsed;
    char line[8192];
    long long lineno = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        lineno++;
        std::string s(line);
        const size_t hash = s.find('#');
        if (hash != std::string::npos) {
            s = s.substr(0, hash);
        }
        std::istringstream is(s);
        long long il = -1;
        long long ord = -1;
        if (!(is >> il >> ord)) {
            continue; // blank / comment-only line
        }
        if (il < 0 || il >= n_layer) {
            fclose(f);
            error_msg = "set_row_set_spec: " + script_path + " line " + std::to_string(lineno) +
                        " names layer " + std::to_string(il) + " outside [0," + std::to_string(n_layer) + ")";
            return false;
        }
        if (ord < 1) {
            fclose(f);
            error_msg = "set_row_set_spec: " + script_path + " line " + std::to_string(lineno) +
                        " has ordinal " + std::to_string(ord) + " (ordinals are 1-based)";
            return false;
        }
        std::vector<int32_t> ids;
        long long e = 0;
        while (is >> e) {
            if (e < 0 || e >= n_exp) {
                fclose(f);
                error_msg = "set_row_set_spec: " + script_path + " line " + std::to_string(lineno) +
                            " names expert " + std::to_string(e) + " outside [0," + std::to_string(n_exp) + ")";
                return false;
            }
            ids.push_back((int32_t) e);
        }
        if ((int) ids.size() > b) {
            // a script may not exceed the declared budget: silently accepting it
            // would make the budget claim false
            fclose(f);
            error_msg = "set_row_set_spec: " + script_path + " line " + std::to_string(lineno) + " proposes " +
                        std::to_string(ids.size()) + " ids, above the budget " + std::to_string(b);
            return false;
        }
        if (!ids.empty()) {
            parsed[((uint64_t) il << 32) | (uint64_t) ord] = ids;
        }
    }
    fclose(f);

    spec_script       = std::move(parsed);
    row_set_spec_mode = 1;
    row_set_spec_budget = b;
    // P3's reserve: one full window's worth of rows (top_k), the largest union a
    // single demand stage can need under the W=1 shape this path enforces.
    row_set_spec_reserve = top_k;
    row_set_spec_on      = true;
    row_set_spec_script  = script_path;
    fprintf(stderr, "tg-spec (T-GPLR-58 A0): mode 1 (scripted) budget %d, reserve %d of cap %d, script %s (%zu entries)\n",
            b, row_set_spec_reserve, cap, script_path.c_str(), spec_script.size());
    return true;
}


bool llama_layer_streaming::set_row_set_spec_arm(int arm, bool null_source) {
    if (row_set_spec_mode != 2) {
        error_msg = "set_row_set_spec_arm: only mode 2 has a proposer arm (mode is " +
                    std::to_string(row_set_spec_mode) + ")";
        return false;
    }
    if (model_ref == nullptr) {
        error_msg = "set_row_set_spec_arm: called before the model is available";
        return false;
    }
    if (arm < 0 || arm > LLAMA_LS_ARM_D1) {
        error_msg = "set_row_set_spec_arm: arm " + std::to_string(arm) +
                    " is not one of coact, bar09, fusion or d1";
        return false;
    }
    if (arm == LLAMA_LS_ARM_D1) {
        if (row_set_spec_pool > 0) {
            error_msg = "set_row_set_spec_arm: D1 currently uses the real per-layer cache, not the separate prediction pool";
            return false;
        }
        const int n_exp = (int) model_ref->hparams.n_expert;
        const int top_k = (int) model_ref->hparams.n_expert_used;
        const int k = row_set_retain_k < 1 ? 1 : row_set_retain_k;
        const int cap = std::min(n_exp, k * top_k);
        if (row_set_spec_reserve >= cap) {
            error_msg = "set_row_set_spec_arm: D1 needs at least one free cache position beyond the declared target union (increase K)";
            return false;
        }
        spec_pred.reset();
        d1_state = std::make_unique<llama_ls_d1_state>();
        llama_ls_d1_init(d1_state.get(), n_layer, n_exp);
        row_set_spec_arm  = LLAMA_LS_ARM_D1;
        row_set_spec_null = null_source;
        spec_null_rng     = llama_ls_pred_null_seed();
        spec_prop_last      = 0;
        spec_arm_count_last = 0;
        fprintf(stderr, "tg-spec (T-GPLR-59 D1): proposed-token source%s, %d layers x %d experts, MTP window %d\n",
                null_source ? " with the COUNT-MATCHED NULL source" : "",
                n_layer, n_exp, row_set_window_tokens);
        return true;
    }
    // Only the families the arm READS are allocated -- except under the null, which
    // needs all of them: its count IS the arm's count, so the arm has to be computed
    // for there to be a count to match.
    uint32_t fam = LLAMA_LS_PRED_BIT_ALL;
    if (!null_source) {
        switch (arm) {
            case LLAMA_LS_ARM_COACT:  fam = LLAMA_LS_PRED_BIT_COACT;   break;
            case LLAMA_LS_ARM_BAR09:  fam = LLAMA_LS_PRED_BIT_MARKOV1; break;
            default:                  fam = LLAMA_LS_PRED_BIT_ALL;     break;
        }
    }
    const int n_exp = (int) model_ref->hparams.n_expert;
    spec_pred = std::make_unique<llama_ls_pred_state>();
    llama_ls_pred_init(spec_pred.get(), n_layer, n_exp, fam);
    d1_state.reset();
    row_set_spec_arm    = arm;
    row_set_spec_null   = null_source;
    spec_null_rng       = llama_ls_pred_null_seed();
    spec_prop_last      = 0;
    spec_arm_count_last = 0;
    fprintf(stderr, "tg-spec (T-GPLR-58 A1): proposer arm %s%s, families 0x%x, %d layers x %d experts%s\n",
            ls_pred_arm_name(arm), null_source ? " with the COUNT-MATCHED NULL source" : "",
            fam, n_layer, n_exp,
            null_source ? " (the arm is still computed: it supplies the count)" : "");
    return true;
}

bool llama_layer_streaming::d1_issue_preverify(const std::vector<int32_t> & draft_tokens) {
    if (row_set_spec_mode != 2 || row_set_spec_arm != LLAMA_LS_ARM_D1 || !d1_state) {
        error_msg = "d1_issue_preverify: D1 source is not configured";
        return false;
    }
    if (!row_set_active || !cache_active) {
        error_msg = "d1_issue_preverify: row-set staging is not active";
        return false;
    }
    if (draft_tokens.empty()) {
        error_msg = "d1_issue_preverify: no MTP draft tokens";
        return false;
    }
    if (backend_copy == nullptr || !spec_ensure_events()) {
        if (error_msg.empty()) {
            error_msg = "d1_issue_preverify: speculative copy backend is unavailable";
        }
        failed = true;
        return false;
    }

    d1_last_routes.clear();
    d1_last_candidates.assign(draft_tokens.size(), std::vector<std::vector<int32_t>>((size_t) n_layer));
    ++d1_round;
    const int n_exp = (int) model_ref->hparams.n_expert;
    for (int il = 0; il < n_layer; ++il) {
        spec_stage_submitted = 0;
        spec_fills_last.clear();
        spec_candidates_last.clear();
        spec_skipped_present_last = 0;
        spec_skipped_dup_last = 0;
        spec_plan_refused_capacity_last = 0;
        spec_plan_stop_last = LLAMA_LS_STOP_PROPOSALS;
        spec_prop_last = 0;
        spec_arm_count_last = 0;

        // D1 is settled at this pre-verify boundary, so a completed round never
        // leaves an event slot or a PENDING row for the coming demand.
        spec_reap(il);
        ++spec_calls[(size_t) il];
        const int cap = (int) spec_state[(size_t) il].size();
        for (size_t di = 0; di < draft_tokens.size(); ++di) {
            const int32_t token = draft_tokens[di];
            const llama_ls_propose_result pr = llama_ls_spec_propose_d1(
                d1_state.get(), (int) token, il, row_set_spec_budget, row_set_spec_null,
                cache_expert[(size_t) il].data(), spec_state[(size_t) il].data(), cap,
                row_set_spec_reserve, row_set_spec_budget, spec_pending_of(il),
                &spec_null_rng, n_exp);
            spec_prop_last += pr.n_proposals;
            spec_arm_count_last += pr.n_arm_count;
            // The generic spec event is one aggregate list per layer/round.
            // D1 has one proposal list per draft token, so the same expert can
            // legitimately appear in more than one draft.  Emit the aggregate
            // candidate set once; the per-draft exact proposals remain in
            // d1_pair.predicted_ids and arm_count still records the total count
            // used by the count-matched null.
            for (int i = 0; i < pr.n_proposals; ++i) {
                if (std::find(spec_candidates_last.begin(), spec_candidates_last.end(), pr.published[i]) == spec_candidates_last.end()) {
                    spec_candidates_last.push_back(pr.published[i]);
                }
            }
            d1_last_candidates[di][(size_t) il].assign(pr.published, pr.published + pr.n_proposals);
            spec_skipped_present_last += pr.plan.skipped_present;
            spec_skipped_dup_last += pr.plan.skipped_dup;
            spec_plan_refused_capacity_last += pr.plan.refused_capacity;
            spec_plan_stop_last = pr.plan.stop;
            const uint32_t gen = slots.empty() ? 0 : slots[(size_t) (il % (int) slots.size())].generation;
            for (int i = 0; i < pr.plan.n_issue; ++i) {
                if (spec_issue_one(il, pr.plan.issue[i], gen, ++cache_tick)) {
                    spec_fills_last.push_back(pr.plan.issue[i]);
                }
            }
        }
        // candidate_ids is the aggregate event's exact published set; D1's
        // arm_count remains the pre-dedup count used for the matched null.
        spec_prop_last = (int) spec_candidates_last.size();
    }

    // The D1 lead is a ROUND, not a layer stage. Complete the speculative copies
    // here, before target verification starts, so no demand wait can include a
    // speculative completion (P1 remains exactly zero for this source).
    ls_stage_drain();
    for (int il = 0; il < n_layer; ++il) {
        spec_reap(il);
        tel_spec(il, spec_pending_of(il), spec_ready_of(il));
    }
    if (failed) {
        error_msg = "d1_issue_preverify: speculative issue or settle failed";
        return false;
    }
    return true;
}

bool llama_layer_streaming::d1_observe_round(const std::vector<int32_t> & draft_tokens,
                                             const std::vector<uint8_t> & accepted) {
    if (!d1_state || draft_tokens.size() != accepted.size()) {
        error_msg = "d1_observe_round: draft/acceptance shape mismatch";
        return false;
    }
    if (d1_last_routes.size() != draft_tokens.size() + 1) {
        error_msg = "d1_observe_round: captured target width " + std::to_string(d1_last_routes.size()) +
                    " does not match frontier + drafts " + std::to_string(draft_tokens.size() + 1);
        return false;
    }
    if (d1_last_candidates.size() != draft_tokens.size()) {
        error_msg = "d1_observe_round: candidate capture width does not match drafts";
        return false;
    }
    const int top_k = (int) model_ref->hparams.n_expert_used;
    for (size_t j = 0; j < draft_tokens.size(); ++j) {
        if (accepted[j] > 1) {
            error_msg = "d1_observe_round: accepted flag is not boolean";
            return false;
        }
        if (d1_last_routes[j + 1].size() != (size_t) n_layer ||
            d1_last_candidates[j].size() != (size_t) n_layer) {
            error_msg = "d1_observe_round: captured layer vectors do not match the layer count";
            return false;
        }
        for (int il = 0; il < n_layer; ++il) {
            const auto & ids = d1_last_routes[j + 1][(size_t) il];
            if ((int) ids.size() != top_k) {
                error_msg = "d1_observe_round: target route width " + std::to_string(ids.size()) +
                            " != top_k " + std::to_string(top_k) + " at layer " + std::to_string(il);
                return false;
            }
        }
    }
    for (size_t j = 0; j < draft_tokens.size(); ++j) {
        const bool is_accepted = accepted[j] != 0;
        for (int il = 0; il < n_layer; ++il) {
            const auto & ids = d1_last_routes[j + 1][(size_t) il];
            const auto & predicted = d1_last_candidates[j][(size_t) il];
            tel_d1_pair((int) d1_round, (int) j, draft_tokens[j], il, is_accepted,
                        ids.data(), (int) ids.size(), predicted.data(), (int) predicted.size());
            llama_ls_d1_observe(d1_state.get(), (int) draft_tokens[j], il, is_accepted, ids.data(), (int) ids.size());
        }
    }
    d1_last_routes.clear();
    d1_last_candidates.clear();
    return true;
}

void llama_layer_streaming::d1_emit_summary() {
    if (!d1_state) {
        return;
    }
    tel_write(tel_out,
        "{\"event\":\"d1_summary\",\"run_id\":%llu,\"rounds\":%u,\"rows_seen\":%llu,"
        "\"accepted_rows\":%llu,\"acceptance_rate\":%.6f,\"pairing_valid\":%s,\"t_us\":%lld}",
        (unsigned long long) run_id, d1_round,
        (unsigned long long) d1_state->rows_seen,
        (unsigned long long) d1_state->accepted_rows,
        d1_state->rows_seen > 0 ? (double) d1_state->accepted_rows / (double) d1_state->rows_seen : 0.0,
        llama_ls_d1_pairing_valid(d1_state.get()) ? "true" : "false",
        (long long) ggml_time_us());
}

void llama_layer_streaming::spec_reset() {
    // DRAIN FIRST: a fill whose H2D is still in flight writes into a per-layer
    // cache buffer that end_row_cache() frees right after this call, so ending a
    // run with a fill outstanding would be a device-side use-after-free. Every
    // demand stage synchronizes its own copies before returning; a speculative
    // fill has no such owner, so the teardown owns it. This is the one place
    // where waiting on a speculative copy is correct, and it is not a demand.
    for (auto & evs : spec_ev) {
        for (void * ev : evs) {
            if (ev != nullptr) {
                ggml_backend_event_synchronize((ggml_backend_event_t) ev);
            }
        }
    }
    for (auto & evs : spec_ev) {
        for (void * ev : evs) {
            if (ev != nullptr) {
                ggml_backend_event_free((ggml_backend_event_t) ev);
            }
        }
    }
    spec_ev.clear();
    spec_ev_pos.clear();
    spec_state.clear();
    spec_flag.clear();
    spec_gen.clear();
    spec_tick.clear();
    spec_calls.clear();
    spec_issue_ord.clear();
    // the proposer's own state is large (n_layer x n_expert^2 per family table):
    // the run is over, so it is released rather than kept for a later run
    spec_pred.reset();
    d1_state.reset();
    d1_last_routes.clear();
    d1_last_candidates.clear();
    d1_shape_emitted = false;
    d1_round = 0;
    spec_null_rng         = 0;
    spec_stage_submitted  = 0;
    spec_fills_last.clear();
    spec_candidates_last.clear();
    spec_skipped_present_last = 0;
    spec_skipped_dup_last = 0;
    spec_plan_refused_capacity_last = 0;
    spec_plan_stop_last = 0;
    row_set_spec_arm      = -1;
    row_set_spec_null     = false;
    spec_prop_last        = 0;
    spec_arm_count_last   = 0;
    row_set_spec_on       = false;
    spec_in_demand_window = false;
}

void llama_layer_streaming::spec_cancel(int il, int need_free, const char * reason) {
    if (!row_set_spec_on || need_free <= 0 || il < 0 || (size_t) il >= spec_state.size()) {
        return;
    }
    auto & st = spec_state[(size_t) il];
    auto & rc = cache_recency[(size_t) il];
    const int cap = (int) st.size();
    std::vector<int> pos((size_t) cap, -1);
    const int n = llama_ls_spec_cancel_plan(rc.data(), st.data(), cap, need_free, pos.data(), cap);
    for (int i = 0; i < n; ++i) {
        const int p = pos[(size_t) i];
        if (p < 0 || st[(size_t) p] != LLAMA_LS_SPEC_PENDING) {
            continue;
        }
        // MARK ONLY: the position is released by the reap, after its copy
        // completed, so a write in flight is never reused and no demand stage
        // ever blocks here (P1).
        st[(size_t) p] = LLAMA_LS_SPEC_CANCELLED;
        spec_cancelled++;
        if (reason != nullptr && std::string(reason) == "generation_boundary") {
            spec_cancelled_generation++;
        } else if (reason != nullptr && std::string(reason) == "demand_duplicate") {
            spec_cancelled_demand++;
        } else {
            spec_cancelled_capacity++;
        }
    }
}

void llama_layer_streaming::spec_reap(int il) {
    if (!row_set_spec_on || il < 0 || (size_t) il >= spec_state.size()) {
        return;
    }
    if ((size_t) il >= spec_ev.size()) {
        return; // events are created lazily, on the first issue
    }
    auto & st = spec_state[(size_t) il];
    auto & ex = cache_expert[(size_t) il];
    auto & rc = cache_recency[(size_t) il];
    auto & fl = spec_flag[(size_t) il];
    auto & tk = spec_tick[(size_t) il];
    auto & evs = spec_ev[(size_t) il];
    auto & evp = spec_ev_pos[(size_t) il];
    for (size_t s = 0; s < evs.size(); ++s) {
        const int p = evp[s];
        if (p < 0) {
            continue;
        }
        if (st[(size_t) p] != LLAMA_LS_SPEC_PENDING && st[(size_t) p] != LLAMA_LS_SPEC_CANCELLED) {
            evp[s] = -1; // resolved elsewhere: the event slot is free
            continue;
        }
        const int64_t t0 = ggml_time_us();
        ggml_backend_event_synchronize((ggml_backend_event_t) evs[s]);
        const int64_t waited = ggml_time_us() - t0;
        spec_reap_wait_us += waited;
        if (spec_in_demand_window) {
            // P1's falsifier: with reaps on the speculation path only this stays
            // 0, and it turns non-zero the moment a reap moves inside a demand's
            // own wait.
            spec_waited_by_demand++;
        }
        if (st[(size_t) p] == LLAMA_LS_SPEC_PENDING) {
            // its OWN copy completed -> only NOW is it a residency claim (P2),
            // stamped with the recency it was issued under so a speculative row
            // loses every tie against a demanded one.
            rc[(size_t) p] = tk[(size_t) p] >= 0 ? tk[(size_t) p] : ++cache_tick;
            fl[(size_t) p] = 1;
            st[(size_t) p] = LLAMA_LS_SPEC_READY;
            spec_ready++;
        } else {
            // cancelled before completion: the position is released WITHOUT a
            // residency claim (the row is superseded by the demand's own copy)
            ex[(size_t) p] = -1;
            rc[(size_t) p] = -1;
            fl[(size_t) p] = 0;
            tk[(size_t) p] = -1;
            st[(size_t) p] = LLAMA_LS_SPEC_FREE;
            if ((size_t) il < spec_issue_ord.size()) {
                // T-GPLR-58 A2: a fill cancelled before landing was never
                // consumed, so its ordinal dies with it -- a stale ordinal on a
                // reused position would credit a later hit to a fill that never
                // reached the cache.
                spec_issue_ord[(size_t) il][(size_t) p] = -1;
            }
        }
        evp[s] = -1;
    }
}

bool llama_layer_streaming::spec_issue_one(int il, int32_t expert, uint32_t generation, int64_t tick) {
    auto & st = spec_state[(size_t) il];
    auto & ex = cache_expert[(size_t) il];
    auto & rc = cache_recency[(size_t) il];
    auto & fl = spec_flag[(size_t) il];
    auto & gn = spec_gen[(size_t) il];
    auto & tk = spec_tick[(size_t) il];
    const int cap = (int) st.size();

    const llama_ls_spec_admission ad = llama_ls_spec_admit(ex.data(), st.data(), cap,
                                                           row_set_spec_reserve, row_set_spec_budget,
                                                           spec_pending_of(il), expert);
    if (!ad.ok) {
        if (std::string(ad.reason) == "already_resident" || std::string(ad.reason) == "already_pending") {
            spec_dup_refused++;
        } else {
            // the P3 barrier (reserve / budget / no free position) -- structured,
            // never a silent degradation
            spec_refused_capacity++;
        }
        return false;
    }
    const int p = ad.pos;

    // event slot for this fill's OWN completion (lazily created: the copy
    // backend only exists by the time a fill is issued)
    auto & evs = spec_ev[(size_t) il];
    auto & evp = spec_ev_pos[(size_t) il];
    int slot = -1;
    for (size_t s = 0; s < evp.size(); ++s) {
        if (evp[s] < 0) {
            slot = (int) s;
            break;
        }
    }
    if (slot < 0) {
        spec_refused_capacity++; // no completion slot free: refuse rather than issue untracked
        return false;
    }

    const llama_ls_package & pkg = pkg_cache.at(il);
    const std::vector<ggml_tensor *> & cache_mem = cache_tens[(size_t) il];
    size_t bytes = 0;
    for (size_t mi = 0; mi < pkg.members.size(); ++mi) {
        const ggml_tensor * src_t   = pkg.members[mi].src;
        const ggml_tensor * cache_t = mi < cache_mem.size() ? cache_mem[mi] : nullptr;
        if (src_t == nullptr || cache_t == nullptr) {
            continue;
        }
        const size_t slab_bytes = ggml_row_size(src_t->type, src_t->ne[0]) * (size_t) src_t->ne[1];
        const size_t src_off    = (size_t) expert * slab_bytes;
        const size_t dst_off    = (size_t) p * slab_bytes;
        ggml_backend_tensor_set_async(backend_copy, (ggml_tensor *) cache_t,
                                      (const char *) src_t->data + src_off, dst_off, slab_bytes);
        bytes += slab_bytes;
    }

    // The fill occupies this cache position as soon as its copy is issued. Keeping
    // the expert id in the cache owner's array makes duplicate admission and the
    // demand reserve see the physical row while its copy is in flight.
    ex[(size_t) p] = expert;
    st[(size_t) p] = LLAMA_LS_SPEC_PENDING;
    fl[(size_t) p] = 0;
    if ((size_t) il < spec_issue_ord.size()) {
        // T-GPLR-58 A2: stamp the ISSUING stage so M7 can measure the head start
        // this fill actually got before a demand was served from it.
        spec_issue_ord[(size_t) il][(size_t) p] = spec_calls[(size_t) il];
    }
    gn[(size_t) p] = generation;
    tk[(size_t) p] = tick;
    evp[(size_t) slot] = p;
    ggml_backend_event_record((ggml_backend_event_t) evs[(size_t) slot], backend_copy);
    spec_submitted++;
    spec_stage_submitted++;
    spec_issued_bytes += bytes;
    return true;
}

// T-GPLR-58 A2 (M7): the lead of one consumed fill, in demand stages of its own
// layer. MEASURED ORDERING (A0 shape, read this before interpreting M7): the only
// hot-path reap runs at the END of a demand stage, so a fill issued after stage t
// first counts as READY at the end of stage t+1 -- the earliest possible
// consumption is stage t+2, i.e. lead 2. A lead of 1 is unreachable on that path,
// and a fill that meets its own demand one stage later is the FALLBACK
// (spec_matched_by_demand), not a lead-0 consumption.
bool llama_layer_streaming::spec_note_lead(int il, int pos) {
    if (il < 0 || (size_t) il >= spec_issue_ord.size()) {
        return false;
    }
    auto & ord = spec_issue_ord[(size_t) il];
    if (pos < 0 || (size_t) pos >= ord.size()) {
        return false;
    }
    const int32_t issued_at = ord[(size_t) pos];
    ord[(size_t) pos] = -1; // one consumption per fill
    if (issued_at < 0) {
        return false;       // an ordinary resident: no fill ever owned this position
    }
    // ONE OWNER PER RULE: the arithmetic and the bucket bounds live in the pure
    // API (llama_ls_spec_lead / _lead_bucket), so the host test proves exactly
    // what the trace checker reads. `consume_ord` is THIS stage's ordinal, one
    // past the count of stages this layer had already completed.
    const int lead   = llama_ls_spec_lead((int32_t) spec_calls[(size_t) il] + 1, issued_at);
    const int bucket = llama_ls_spec_lead_bucket(lead);
    if (bucket < 0) {
        return false; // no fill, or consumed in its own issuing stage
    }
    spec_consumed++;
    spec_lead_sum += lead;
    if (spec_lead_min == 0 || lead < spec_lead_min) {
        spec_lead_min = lead;
    }
    if (lead > spec_lead_max) {
        spec_lead_max = lead;
    }
    if (spec_lead_hist.size() != 8) {
        spec_lead_hist.assign(8, 0);
    }
    spec_lead_hist[(size_t) bucket]++;
    return true;
}


bool llama_layer_streaming::spec_ensure_events() {
    if (!row_set_spec_on || spec_ev.size() >= spec_state.size()) {
        return true;
    }
    const ggml_backend_dev_t dev_copy = ggml_backend_get_device(backend_copy);
    spec_ev.assign(spec_state.size(), {});
    spec_ev_pos.assign(spec_state.size(), {});
    for (size_t L = 0; L < spec_state.size(); ++L) {
        spec_ev[L].assign((size_t) row_set_spec_budget, nullptr);
        spec_ev_pos[L].assign((size_t) row_set_spec_budget, -1);
        for (int s = 0; s < row_set_spec_budget; ++s) {
            spec_ev[L][(size_t) s] = ggml_backend_event_new(dev_copy);
            if (spec_ev[L][(size_t) s] == nullptr) {
                error_msg = "spec_ensure_events: failed to create completion event for layer " + std::to_string(L);
                return false;
            }
        }
    }
    return true;
}

void llama_layer_streaming::spec_issue_next(int il, const int32_t * demand_ids, int n_demand) {
    if (!row_set_spec_on || il < 0 || (size_t) il >= spec_state.size()) {
        return;
    }
    if (backend_copy == nullptr) {
        return;
    }
    // T-GPLR-58 A2: the stage-scoped telemetry starts here. `submitted` on this
    // stage's event counts the fills THIS stage issues, and `fill_ids` names them.
    spec_stage_submitted = 0;
    spec_fills_last.clear();
    spec_candidates_last.clear();
    spec_skipped_present_last = 0;
    spec_skipped_dup_last = 0;
    spec_plan_refused_capacity_last = 0;
    spec_plan_stop_last = 0;

    // The REAP lives here, on the speculation path: it is the only place a
    // PENDING position resolves, and it runs after the demand's own wait has
    // closed, so a fill's completion can never extend a demand's stage wait.
    spec_reap(il);

    if (row_set_spec_mode == 2 && row_set_spec_arm == LLAMA_LS_ARM_D1) {
        // D1 publishes at the MTP pre-verify boundary and is settled there. The
        // post-demand path must not silently fall back to a past-only proposer.
        return;
    }

    if (row_set_spec_pool > 0 && row_set_spec_mode == 2) {
        pool_submitted_last = 0;
        pool_evicted_last = 0;
        pool_h2d_bytes_last = 0;
        pool_reap_wait_us_last = 0;
        pool_candidates_last.clear();
        pool_fills_last.clear();
        pool_skipped_real_last = pool_skipped_pool_last = pool_skipped_dup_last = 0;
        pool_skipped_invalid_last = pool_refused_budget_last = pool_refused_capacity_last = 0;
        pool_stop_last = 0;
        pool_reap(il);
        if (failed) {
            return;
        }
        if ((size_t) il >= pool_expert.size() || (size_t) il >= pool_state.size()) {
            error_msg = "spec_issue_next: prediction pool is not allocated";
            failed = true;
            return;
        }
        const llama_ls_propose_result pr = llama_ls_spec_propose(
            spec_pred.get(), row_set_spec_arm, row_set_spec_null, il, row_set_spec_budget,
            demand_ids, n_demand, pool_expert[(size_t) il].data(), pool_state[(size_t) il].data(),
            row_set_spec_pool, 0, row_set_spec_budget, pool_pending_of(il),
            &spec_null_rng, (int) model_ref->hparams.n_expert);
        pool_candidates_last.assign(pr.published, pr.published + pr.n_proposals);
        const llama_ls_pool_plan pp = llama_ls_pool_plan_make(
            cache_expert[(size_t) il].data(), spec_state[(size_t) il].data(), row_set_cap,
            pool_expert[(size_t) il].data(), pool_state[(size_t) il].data(),
            pool_recency[(size_t) il].data(), row_set_spec_pool,
            (int) model_ref->hparams.n_expert, row_set_spec_budget,
            pr.published, pr.n_proposals);
        pool_skipped_real_last = pp.skipped_real;
        pool_skipped_pool_last = pp.skipped_pool;
        pool_skipped_dup_last = pp.skipped_dup;
        pool_skipped_invalid_last = pp.skipped_invalid;
        pool_refused_budget_last = pp.refused_budget;
        pool_refused_capacity_last = pp.refused_capacity;
        pool_stop_last = pp.stop;
        const uint32_t gen = slots.empty() ? 0 : slots[(size_t) (il % (int) slots.size())].generation;
        for (int i = 0; i < pp.n_issue; ++i) {
            const int32_t e = pp.issue[i];
            const int pos = pp.issue_pos[i];
            if (pos < 0 || pos >= row_set_spec_pool) {
                error_msg = "pool planner returned an invalid position";
                failed = true;
                break;
            }
            if (pool_state[(size_t) il][(size_t) pos] == LLAMA_LS_SPEC_READY) {
                if (!llama_ls_pool_retire_ready(
                        pool_expert[(size_t) il].data(), pool_state[(size_t) il].data(),
                        pool_recency[(size_t) il].data(), pool_gen[(size_t) il].data(),
                        row_set_spec_pool, gen, pos)) {
                    error_msg = "pool planner selected a stale READY row";
                    failed = true;
                    break;
                }
                pool_issue_ord[(size_t) il][(size_t) pos] = -1;
                ++pool_evicted_last;
                ++pool_evicted;
            } else if (pool_state[(size_t) il][(size_t) pos] != LLAMA_LS_SPEC_FREE) {
                error_msg = "pool planner selected an occupied in-flight row";
                failed = true;
                break;
            }
            if (!pool_issue_one(il, e, pos, gen, ++cache_tick)) {
                failed = true;
                break;
            }
            pool_fills_last.push_back(e);
        }
        if (failed) {
            const int slot = slots.empty() ? -1 : il % (int) slots.size();
            tel_rowset_refused(il, slot, gen, "pool_issue", error_msg.c_str());
            return;
        }
        tel_spec_pool(il, gen, pool_pending_of(il), pool_ready_of(il));
        return;
    }

    // lazy allocation of the per-layer event slots (one per budgeted fill)
    if (!spec_ensure_events()) {
        failed = true;
        return;
    }

    const int ord = ++spec_calls[(size_t) il];

    if (row_set_spec_mode == 2) {
        // MODE 2: ask the arm -- observe the row this stage just served, propose for
        // this layer's NEXT demand row, plan against the live placement state, then
        // issue exactly what the plan admits. This is the whole of the engine's
        // proposer wiring; the policy itself is llama_ls_spec_propose().
        spec_prop_last      = 0;
        spec_arm_count_last = 0;
        const int cap = (int) spec_state[(size_t) il].size();
        const llama_ls_propose_result pr = llama_ls_spec_propose(
            spec_pred.get(), row_set_spec_arm, row_set_spec_null, il, row_set_spec_budget,
            demand_ids, n_demand,
            cache_expert[(size_t) il].data(), spec_state[(size_t) il].data(), cap,
            row_set_spec_reserve, row_set_spec_budget, spec_pending_of(il),
            &spec_null_rng, (int) model_ref->hparams.n_expert);
        spec_prop_last      = pr.n_proposals;
        spec_arm_count_last = pr.n_arm_count;
        spec_skipped_present_last = pr.plan.skipped_present;
        spec_skipped_dup_last = pr.plan.skipped_dup;
        spec_plan_refused_capacity_last = pr.plan.refused_capacity;
        spec_plan_stop_last = pr.plan.stop;
        for (int i = 0; i < pr.n_proposals && i < LLAMA_LS_PRED_MAX_OUT; ++i) {
            spec_candidates_last.push_back(pr.published[i]);
        }
        const uint32_t gen = slots.empty() ? 0 : slots[(size_t) (il % (int) slots.size())].generation;
        for (int i = 0; i < pr.plan.n_issue; ++i) {
            // stamp at ISSUE time so a speculative row is older than every row a
            // demand placed afterwards
            if (spec_issue_one(il, pr.plan.issue[i], gen, ++cache_tick)) {
                spec_fills_last.push_back(pr.plan.issue[i]);
            }
        }
        tel_spec(il, spec_pending_of(il), spec_ready_of(il));
        return;
    }

    const auto it = spec_script.find(((uint64_t) il << 32) | (uint64_t) (ord + 1));
    if (it != spec_script.end()) {
        const uint32_t gen = slots.empty() ? 0 : slots[(size_t) (il % (int) slots.size())].generation;
        for (int32_t e : it->second) {
            // stamp at ISSUE time so a speculative row is older than every row a
            // demand placed afterwards
            if (spec_issue_one(il, e, gen, ++cache_tick)) {
                spec_fills_last.push_back(e);
            }
        }
    }
    tel_spec(il, spec_pending_of(il), spec_ready_of(il));
}


void llama_layer_streaming::tel_spec(int il, int pending_now, int ready_spec_now) {
    // fixed shape so the emitted array is always the documented 8 buckets
    std::vector<int64_t> lh = spec_lead_hist;
    lh.resize(8, 0);
    // MODE 2 only, so a mode-0/1 event stays byte-identical to every recording
    // made before the proposer was wired: what the SOURCE published this stage, and
    // the ARM's count a null source was matched to (equal by construction, emitted so
    // the count match is checkable from the trace alone).
    std::string prop_str;
    if (row_set_spec_mode == 2) {
        prop_str = std::string("\"prop\":") + std::to_string(spec_prop_last) +
                   ",\"arm_count\":" + std::to_string(spec_arm_count_last) +
                   ",\"skipped_present\":" + std::to_string(spec_skipped_present_last) +
                   ",\"skipped_dup\":" + std::to_string(spec_skipped_dup_last) +
                   ",\"plan_refused_capacity\":" + std::to_string(spec_plan_refused_capacity_last) +
                   ",\"stop\":" + std::to_string(spec_plan_stop_last) +
                   ",\"stop_reason\":\"" + ls_spec_plan_stop_name(spec_plan_stop_last) + "\",";
        std::string candidates;
        for (size_t i = 0; i < spec_candidates_last.size(); ++i) {
            candidates += (i ? "," : "");
            candidates += std::to_string(spec_candidates_last[i]);
        }
        prop_str += "\"candidate_ids\":[" + candidates + "],";
    }
    // T-GPLR-58 A2: the ids THIS stage issued, in issue order, as `fill_ids`. The
    // runner's retention cache model replays them (ls_retention_errors), so a `spec`
    // event is a complete record of the layer's occupancy change; without the ids the
    // model can only see demand events and every fetch after a fill-issuing stage looks
    // like a retention-accounting violation. `submitted` is the per-stage count and never
    // exceeds the budget; `submitted_total` is the run-wide cumulative the P2 accounting
    // closes against (T-GPLR-58 A2, measured on the 2026-09-23 K-probe: the cumulative
    // value read as a +306 per-stage increment against a budget of 8).
    std::string fills;
    for (size_t i = 0; i < spec_fills_last.size(); ++i) {
        fills += (i ? "," : "");
        fills += std::to_string(spec_fills_last[i]);
    }
    tel_write(tel_out,
        "{\"event\":\"spec\",\"run_id\":%llu,\"ubatch\":%llu,\"layer\":%d,\"mode\":%d,\"budget\":%d,"
        "\"reserve\":%d,\"cap\":%d,\"pending\":%d,\"ready_spec\":%d,"
        "\"submitted\":%lld,\"submitted_total\":%lld,\"fill_ids\":[%s],"
        "\"ready\":%lld,\"cancelled\":%lld,\"cancelled_capacity\":%lld,"
        "\"cancelled_demand\":%lld,\"cancelled_generation\":%lld,"
        "\"refused_capacity\":%lld,\"dup_refused\":%lld,\"matched_by_demand\":%lld,"
        "\"waited_by_demand\":%lld,\"reap_wait_us\":%lld,\"issued_bytes\":%lld,"
        // T-GPLR-58 A2 (M7): the consumed-fill lead distribution. `consumed` counts
        // only fills a demand was actually served from; `matched_by_demand` above
        // remains the fallback count, never folded in here.
        "\"consumed\":%lld,\"lead_sum\":%lld,\"lead_min\":%d,\"lead_max\":%d,"
        "\"lead_hist\":[%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld],%s\"t_us\":%lld}",
        (unsigned long long) run_id, (unsigned long long) n_ubatches, il, row_set_spec_mode,
        row_set_spec_budget, row_set_spec_reserve, row_set_cap, pending_now, ready_spec_now,
        (long long) spec_stage_submitted, (long long) spec_submitted, fills.c_str(),
        (long long) spec_ready, (long long) spec_cancelled,
        (long long) spec_cancelled_capacity, (long long) spec_cancelled_demand,
        (long long) spec_cancelled_generation, (long long) spec_refused_capacity,
        (long long) spec_dup_refused, (long long) spec_matched_by_demand,
        (long long) spec_waited_by_demand, (long long) spec_reap_wait_us,
        (long long) spec_issued_bytes,
        (long long) spec_consumed, (long long) spec_lead_sum,
        (int) spec_lead_min, (int) spec_lead_max,
        (long long) lh[0], (long long) lh[1], (long long) lh[2], (long long) lh[3],
        (long long) lh[4], (long long) lh[5], (long long) lh[6], (long long) lh[7],
        prop_str.c_str(),
        (long long) ggml_time_us());
}

void llama_layer_streaming::tel_d1_shape() {
    if (d1_shape_emitted || !d1_state) {
        return;
    }
    d1_shape_emitted = true;
    tel_write(tel_out,
        "{\"event\":\"d1_shape\",\"run_id\":%llu,\"source\":\"d1\",\"n_expert\":%d,\"top_k\":%d,"
        "\"n_layer\":%d,\"window_tokens\":%d,\"draft_depth\":%d,\"budget\":%d,\"null\":%s,\"predicted_ids\":true,\"t_us\":%lld}",
        (unsigned long long) run_id, d1_state->n_expert, (int) model_ref->hparams.n_layer(),
        (int) model_ref->hparams.n_expert_used, row_set_window_tokens,
        row_set_window_tokens > 0 ? row_set_window_tokens - 1 : 0,
        row_set_spec_budget, row_set_spec_null ? "true" : "false",
        (long long) ggml_time_us());
}

void llama_layer_streaming::tel_d1_pair(int round, int draft_index, int32_t draft_token,
                                        int layer, bool accepted, const int32_t * ids, int n_ids,
                                        const int32_t * predicted, int n_predicted) {
    std::string id_list;
    for (int i = 0; i < n_ids; ++i) {
        id_list += (i ? "," : "");
        id_list += std::to_string(ids[i]);
    }
    std::string predicted_list;
    for (int i = 0; i < n_predicted; ++i) {
        predicted_list += (i ? "," : "");
        predicted_list += std::to_string(predicted[i]);
    }
    tel_write(tel_out,
        "{\"event\":\"d1_pair\",\"run_id\":%llu,\"round\":%d,\"draft_index\":%d,"
        "\"draft_token\":%d,\"layer\":%d,\"accepted\":%s,\"demand_ids\":[%s],"
        "\"predicted_ids\":[%s],\"t_us\":%lld}",
        (unsigned long long) run_id, round, draft_index, (int) draft_token, layer,
        accepted ? "true" : "false", id_list.c_str(), predicted_list.c_str(), (long long) ggml_time_us());
}

bool llama_layer_streaming::begin_row_cache() {
    if (!row_set_retain) {
        return true; // default-off: nothing allocated, nothing reported
    }
    if (cache_active) {
        error_msg = "begin_row_cache: per-layer cache already active";
        return false;
    }
    if (adapter == nullptr || !adapter->merge_resident) {
        error_msg = "begin_row_cache: requires a merge-resident (MoE expert-bank) policy";
        return false;
    }
    if (row_set_spec_mode == 2 && row_set_spec_arm < 0) {
        // loud, before any context or graph exists: a mode-2 run whose proposer was
        // never configured would otherwise look like an arm that simply proposes
        // nothing, and that is not a result
        error_msg = "begin_row_cache: mode 2 has no proposer arm; call set_row_set_spec_arm() "
                    "(--residency-tg-spec-arm) before allocating the cache";
        return false;
    }
    if (row_set_spec_arm == LLAMA_LS_ARM_D1 && !d1_state) {
        error_msg = "begin_row_cache: D1 source is selected but its table was not configured";
        return false;
    }

    const ggml_backend_buffer_type_t buft_cuda = ggml_backend_cuda_buffer_type(device);

    const int n_exp = (int) model_ref->hparams.n_expert;
    const int top_k = (int) model_ref->hparams.n_expert_used;
    const int K     = row_set_retain_k < 1 ? 1 : row_set_retain_k;
    // capacity = min(n_expert, K*W*top_k) rows per layer; W == 1 is enforced by
    // the telemetry validator (rowset_init window_tokens != 1 is an error), so
    // the window width is 1 by construction here.
    row_set_cap = std::min(n_exp, K * top_k);
    if (row_set_window_tokens * top_k > row_set_cap) {
        error_msg = "begin_row_cache: target row-set width " + std::to_string(row_set_window_tokens) +
                    " needs capacity " + std::to_string(row_set_window_tokens * top_k) +
                    " but K=" + std::to_string(K) + " provides " + std::to_string(row_set_cap);
        return false;
    }

    ggml_init_params meta_params = { (size_t) 4 * 1024 * 1024, nullptr, true };
    cache_meta = ggml_init(meta_params);
    if (cache_meta == nullptr) {
        error_msg = "begin_row_cache: failed to create the cache tensor context";
        return false;
    }

    cache_bufs.resize(n_layer); // ggml_backend_buffer_ptr is move-only: no assign()
    cache_tens.assign(n_layer, {});
    cache_expert.assign(n_layer, std::vector<int32_t>((size_t) row_set_cap, -1));
    cache_recency.assign(n_layer, std::vector<int64_t>((size_t) row_set_cap, -1));
    cache_tick = 0;
    // T-GPLR-58 A0/A1: speculative placement state. The per-layer arrays are
    // (re)assigned just below, so a CONFIGURED run needs no clearing here -- and
    // must not get one: spec_reset() is the TEARDOWN (it clears the mode, the arm
    // and the null source), while this call site runs AFTER set_row_set_spec() and
    // set_row_set_spec_arm() configured the run and BEFORE the allocation gated on
    // row_set_spec_on. Measured 2026-09-23 (T-GPLR-58 A3 session, the FIRST real arm
    // run): the mode was configured at bring-up -- the log reads "mode 2 (proposer
    // arm) budget 8, reserve 8 of cap 16" and "proposer arm coact" -- yet the trace
    // carried NO row_spec_mode in run_start and ZERO `spec` events, because this
    // reset had already turned a mode-2 run back into a plain retention run. The
    // host fixtures could not catch it: they drive the pure functions and never call
    // begin_row_cache() at all. The DEFAULT path (mode 0) still resets here, so P6
    // holds: nothing speculative survives an off leg.
    if (row_set_spec_mode == 0) {
        spec_reset();
    }
    if (row_set_spec_on) {
        spec_state.assign(n_layer, std::vector<uint8_t>((size_t) row_set_cap, LLAMA_LS_SPEC_FREE));
        spec_flag.assign(n_layer, std::vector<uint8_t>((size_t) row_set_cap, 0));
        spec_gen.assign(n_layer, std::vector<uint32_t>((size_t) row_set_cap, 0));
        spec_tick.assign(n_layer, std::vector<int64_t>((size_t) row_set_cap, -1));
        spec_calls.assign(n_layer, 0);
        spec_issue_ord.assign(n_layer, std::vector<int32_t>((size_t) row_set_cap, -1));
        spec_lead_hist.assign(8, 0);
        // the completion-event pool is created lazily at the first issue, when
        // the copy backend is guaranteed to exist
    }

    size_t allocated = 0;
    for (int il = 0; il < n_layer; ++il) {
        std::vector<ggml_tensor *> members;
        size_t need = 0;
        // shared with row_cache_bytes_for(), so the depth a caller selected from
        // the prediction is the depth that actually gets allocated
        if (!row_cache_layer_cost(il, row_set_cap, cache_meta, &members, &need)) {
            error_msg = "begin_row_cache: failed to create the cache tensor for layer " + std::to_string(il);
            return false;
        }
        cache_tens[(size_t) il] = members;

        // one private buffer per layer: a single shared buffer would
        // reintroduce exactly the aliasing this class exists to remove
        cache_bufs[(size_t) il].reset(ggml_backend_buft_alloc_buffer(buft_cuda, need));
        if (!cache_bufs[(size_t) il]) {
            error_msg = "begin_row_cache: failed to allocate the per-layer cache for layer " + std::to_string(il);
            return false;
        }
        std::vector<ggml_tensor *> real;
        for (ggml_tensor * ct : members) {
            if (ct != nullptr) {
                real.push_back(ct);
            }
        }
        size_t off = 0;
        if (!alloc_tensors(cache_bufs[(size_t) il].get(), buft_cuda, real, off)) {
            error_msg = "begin_row_cache: failed to place the cache tensors for layer " + std::to_string(il);
            return false;
        }
        allocated += need;
    }

    if (row_set_spec_pool > 0 && !begin_row_pool()) {
        end_row_cache();
        return false;
    }

    cache_active = true;
    row_cache_alloc_bytes = allocated;
    fprintf(stderr, "row-cache (T-GPLR-19, SDD section 4 shape (i)): %d layers x %d rows/layer = %.1f MiB private VRAM, K=%d, query-time retention\n",
            n_layer, row_set_cap, (double) allocated / (1024.0 * 1024.0), K);
    return true;
}

void llama_layer_streaming::end_row_cache() {
    if (cache_view_ctx != nullptr) {
        ggml_free(cache_view_ctx);
        cache_view_ctx = nullptr;
    }
    cache_view_used = 0;
    cache_view_cap  = 0;
    if (cache_meta != nullptr) {
        ggml_free(cache_meta);
        cache_meta = nullptr;
    }
    // T-GPLR-58 A0: release the speculative state (events included) with the
    // cache it belongs to. Counters are NOT reset here: they are the run's
    // evidence and are reported per stage by tel_spec().
    spec_reset();
    cache_bufs.clear();
    cache_tens.clear();
    cache_expert.clear();
    cache_recency.clear();
    end_row_pool();
    cache_active = false;
    row_cache_alloc_bytes = 0;
    backend_compute = nullptr;
}

// T-GPLR-63 G: the worst-slot and all-slots cost of one per-slot byte map.
//
// ONE owner for the reduction. begin_row_set() needs it twice -- once for the
// full-bank map (layer_bytes) and once for the compact row-set map
// (layer_bytes_rowset) -- to print "was -> now" in the window-narrowed console
// line. Those were two hand-copied loops that differed only in which map they
// read, which is the same two-owner hazard the host checkers had (T-GPLR-63
// checker partial merge): a later edit to one copy leaves the console reporting
// a before/after pair computed by two different reductions. The field is a
// pointer-to-member so the two call sites differ in the ARGUMENT, not in the
// code. Behaviour is unchanged -- the identical reduction, run twice.
static void ls_slot_byte_totals(const std::vector<llama_layer_streaming_slot> & slots,
                                const std::unordered_map<int, size_t> llama_layer_streaming_slot::* field,
                                size_t & worst, size_t & total) {
    worst = 0;
    total = 0;
    for (const auto & slot : slots) {
        size_t mx = 0;
        for (const auto & kv : slot.*field) {
            mx = std::max(mx, kv.second);
        }
        worst = std::max(worst, mx);
        total += mx;
    }
}

bool llama_layer_streaming::begin_row_set() {
    if (row_set_active) {
        error_msg = "begin_row_set: row-set staging already active";
        return false;
    }
    if (adapter == nullptr || !adapter->merge_resident) {
        error_msg = "begin_row_set: requires a merge-resident (MoE expert-bank) policy";
        return false;
    }
    if (handoff_active) {
        error_msg = "begin_row_set: conflicts with an active handoff";
        return false;
    }

    // drain pending copy work, then re-arm every slot's generation under the
    // standard protocol: a consumer of the PP-phase resident generation is
    // still outstanding (the last PP layer graph); fence it with the
    // consumer-done event before claiming the slot for TG row-set use.
    ls_stage_drain();
    for (auto & slot : slots) {
        if (slot.consumer_pending && slot.ev_consumer_done != nullptr) {
            ggml_backend_event_synchronize((ggml_backend_event_t) slot.ev_consumer_done);
            slot.consumer_pending = false;
        }
        slot.generation++;
        slot.logical_layer = -1;
        slot.t_copy_submit_us = 0;
        slot.t_copy_enqueue_done_us = 0;
    }

    const int n_exp = (int) model_ref->hparams.n_expert;
    row_set_have.assign(slots.size(), std::vector<uint8_t>(n_exp, 0));
    row_set_owner.assign(slots.size(), -1);
    row_set_ids.assign(n_layer, {});
    // T-GPLR-61 S3/S4: same bring-up allocation as its siblings -- a failed
    // allocation is loud here, mid-run would be worse.
    row_set_outstanding.assign(slots.size(), std::vector<uint8_t>(n_exp, 0));
    row_set_state.assign(n_layer, std::vector<uint8_t>((size_t) n_exp, 2)); // ABSENT until a load writes it
    row_set_fresh.assign(n_layer, std::vector<uint8_t>((size_t) n_exp, 0));
    row_set_gate_open.assign(n_layer, 0);
    row_set_gate_host_us.assign(n_layer, 0.0);
    row_set_gate_wait_us.assign(n_layer, 0.0);
    // T-GPLR-82 S3/S4: row_set_split_built is NOT (re)sized here. begin_row_set()
    // runs AFTER the persistent graphs are built, so the builder's write would be
    // out of bounds -- measured 2026-10-10 as an access violation (0xC0000005)
    // during "building persistent graphs". init() sizes it before any build, and
    // nothing clears it afterwards, because the graphs persist for the run.
    row_set_gen.assign(slots.size(), 0);
    for (auto & s : slots) {
        row_set_gen[&s - &slots[0]] = s.generation;
    }

    // T-GPLR-62 S2: narrow the slot bundles to the configured window BEFORE the
    // row-set phase goes live. This is the only place the shape changes, and it
    // is deliberately after the generation re-arm above so every copy event from
    // the prompt-processing phase has already been fenced by the consumer-done
    // wait. On failure the full-bank bundle and buffer are left untouched, so
    // the run continues to refuse rather than to half-narrow.
    if (row_set_window_width > 0 && !build_row_set_window(row_set_window_width)) {
        return false;
    }

    // T-GPLR-62 S2 (lifetime): the swap above just replaced every slot bundle
    // tensor and released the full-bank buffer, so every phase-1 set built at
    // bring-up now holds DANGLING MoE weight pointers. Flag them; the executor
    // re-derives each one through the T-GPLR-47 fresh-context replacement before
    // it is next executed, which re-binds every pointer against the compact
    // bundle without touching recurrent state (unlike re-running the prebuild,
    // which advances it -- measured: that produced rs_z=0 at the timed decode
    // and a wrong answer).
    if (row_set_window_width > 0 && !mark_row_set_sets_stale()) {
        return false;
    }

    // T-GPLR-62 S2 / A9: declare the mode BEFORE any window is staged, so a
    // trace that declares it and then produces no recon evidence is detectable
    // as "recon was requested but never ran" rather than passing as a clean run.
    if (row_set_recon) {
        tel_write(tel_out,
            "{\"event\":\"rowset_recon_begin\",\"run_id\":%llu,\"window_width\":%d,"
            "\"n_expert\":%d,\"policy\":\"byte-compare-slot-vs-checkpoint\",\"t_us\":%lld}",
            (unsigned long long) run_id, row_set_window_width,
            (int) model_ref->hparams.n_expert, (long long) ggml_time_us());
    }

    // T-GPLR-58 A0, P4's generation boundary: arming the row-set phase (or
    // re-arming it after PP) retires every queued fill. Marked, then reaped, so
    // a fill whose copy is still in flight keeps its position until the copy
    // completes -- its bytes are never overwritten mid-flight.
    if (row_set_spec_on) {
        for (int il = 0; il < n_layer; ++il) {
            spec_cancel(il, spec_pending_of(il), "generation_boundary");
            spec_reap(il);
        }
    }
    if (row_set_spec_pool > 0) {
        pool_cancel_generation();
        if (failed) {
            error_msg = "begin_row_set: prediction-pool generation boundary failed";
            return false;
        }
    }
    row_set_active = true;

    // T-GPLR-62 S2: one row-set slot == one existing full-bank-shaped slot
    // buffer, but the slot BUDGET must cover the WORST slot, not slot 0. The
    // layer->slot assignment is round-robin and the largest bank in the model
    // (the q6_K layers) does not land in slot 0, so reading slots[0] alone
    // under-reports the per-slot cost. Measured 2026-09-30: slot 0 reported a
    // 452,984,832 B bank while the true worst slot holds 522,190,848 B, which
    // is exactly the margin by which the W=16 console figure appeared to miss
    // the preflight's 124.50 MiB. The preflight budgets worst-case-per-slot, so
    // worst-case-per-slot is what is reported here.
    // T-GPLR-62 S2: rowset_init.slot_bytes KEEPS its original meaning -- slot 0's
    // layer-0 bank bytes. It is a recorded telemetry field with existing
    // consumers, so its value must not move as a side effect of fixing a console
    // line. (An earlier pass of this work silently widened it to the worst slot;
    // that was a compatibility break and is reverted here.) The worst-slot and
    // all-slots figures the console needs are computed separately below and are
    // deliberately NOT the ones recorded in the event.
    size_t slot_bytes = 0;
    if (!slots.empty() && slots[0].layer_bytes.count(0)) {
        slot_bytes = slots[0].layer_bytes.at(0);
    }
    // T-GPLR-19: retain_k tells the validators which residency rule applies --
    // 0 is the reactive slot-keyed policy (unchanged), >0 switches them to the
    // per-layer conservation rule (residency is the layer's private cache).
    tel_rowset_init((int) slots.size(), slot_bytes, row_set_window_tokens, n_exp,
                    (int) model_ref->hparams.n_expert_used,
                    row_set_retain ? row_set_retain_k : 0);

    // T-GPLR-62 S2: with a compact window the slot is no longer bank-shaped, so
    // state the width and the before/after per-slot cost on the console. The
    // auditable number for A1 remains graph_footprint.slot_vram_bytes, NOT
    // rowset_init.slot_bytes -- see the contract's A1-WRONG-FIELD note. Both
    // the worst slot and the slot total are printed so a reader can compare
    // like-for-like against the preflight's n_slots * ceil(largest * W / n_exp).
    // Worst-SLOT, not slot 0: layer->slot assignment is round-robin and the
    // largest bank in the model does not land in slot 0, so reading slot 0 alone
    // under-reports the per-slot cost (measured 2026-09-30: it made the W=16
    // figure appear to miss the preflight by exactly that margin).
    if (row_set_window_width > 0) {
        size_t full_worst = 0;
        size_t full_total = 0;
        ls_slot_byte_totals(slots, &llama_layer_streaming_slot::layer_bytes,
                           full_worst, full_total);
        size_t compact_worst = 0;
        size_t compact_total = 0;
        ls_slot_byte_totals(slots, &llama_layer_streaming_slot::layer_bytes_rowset,
                           compact_worst, compact_total);
        fprintf(stderr,
                "layer-streaming: row-set slot window narrowed to %d experts (was %d); "
                "worst-slot bytes %zu -> %zu (%.1f MiB freed/slot); "
                "all-slots %zu -> %zu (%.1f MiB freed total)\n",
                row_set_window_width, n_exp, full_worst, compact_worst,
                (double) (full_worst - compact_worst) / (1024.0 * 1024.0),
                full_total, compact_total,
                (double) (full_total - compact_total) / (1024.0 * 1024.0));
    }

    if (row_set_spec_arm == LLAMA_LS_ARM_D1 && !d1_shape_emitted) {
        tel_d1_shape();
    }
    return true;
}

// T-GPLR-62 S2 / gate A9: prove, BY BYTES, that every position of the staged
// window holds the expert that position is specified to hold.
//
// WHY THIS IS NOT A RESTATEMENT OF THE WRITE PATH. The obvious implementation
// bug is to compare the destination slab against the checkpoint using the SAME
// expression the writer used to choose it, which would make the check agree
// with the bug by construction -- the same tautology that made --why-victim
// worthless. So the expected bytes are re-derived here from the SPEC:
//
//   compact window (ne[2] == W):  slot position i      must hold expert uniq[i]
//   full bank    (ne[2] == n_exp): slot position uniq[i] must hold expert uniq[i]
//
// Each is the mode's own contract, stated independently of how the staging loop
// computed its offset. A wrong source slab, a stale row that was never
// overwritten, a mis-sized view, or an off-by-one in the offset arithmetic all
// show up here as a byte difference.
//
// RESIDUAL, stated honestly: this verifies the BYTES are where the mode says
// they are. It does not independently verify that the MODE is the one the
// operator intended -- that is what the width flag being off by default is for.
bool llama_layer_streaming::recon_row_set_window(int il, const llama_ls_package & pkg,
                                                llama_layer & dst,
                                                const std::vector<int32_t> & uniq,
                                                uint32_t gen) {
    const bool compact = row_set_window_width > 0;
    ggml_init_params vp = { /*mem_size=*/ 64u * 1024u, /*mem_buffer=*/ nullptr, /*no_alloc=*/ true };
    ggml_context * vctx = ggml_init(vp);
    if (vctx == nullptr) {
        error_msg = "recon_row_set_window: failed to init the view context for layer " + std::to_string(il);
        return false;
    }

    std::vector<uint8_t> want, got;
    bool   ok = true;
    size_t bad_offset = 0;
    size_t bad_slab_bytes = 0;
    int32_t bad_expert = -1;
    int     bad_member = -1;

    for (size_t mi = 0; mi < pkg.members.size() && ok; ++mi) {
        const llama_ls_member & m = pkg.members[mi];
        const ggml_tensor * src = m.src;
        ggml_tensor * dt = ls_dst(dst, m);
        if (src == nullptr || dt == nullptr) {
            continue;
        }
        // A compact bundle must never be asked for a position it does not have.
        const int64_t pos_extent = compact ? (int64_t) row_set_window_width : dt->ne[2];
        for (size_t i = 0; i < uniq.size() && ok; ++i) {
            // the SPEC's position for union member i (see the note above)
            const int64_t pos = compact ? (int64_t) i : (int64_t) uniq[i];
            if (pos < 0 || pos >= pos_extent) {
                tel_rowset_refused(il, (int) (il % (int) std::max<size_t>(n_slots, 1)), gen,
                                   "recon_mismatch",
                                   ("recon: union member " + std::to_string(i) + " maps to slot position " +
                                    std::to_string(pos) + " outside [0," + std::to_string(pos_extent) + ")")
                                       .c_str());
                error_msg = "recon_row_set_window: union position out of range for layer " + std::to_string(il);
                ok = false;
                break;
            }
            ggml_tensor * vs = ggml_view_2d(vctx, (ggml_tensor *) src,
                                             src->ne[0], src->ne[1], src->nb[1],
                                             (size_t) uniq[i] * src->nb[2]);
            ggml_tensor * vd = ggml_view_2d(vctx, dt,
                                             dt->ne[0], dt->ne[1], dt->nb[1],
                                             (size_t) pos * dt->nb[2]);
            if (vs == nullptr || vd == nullptr) {
                error_msg = "recon_row_set_window: failed to build a slab view for layer " + std::to_string(il);
                ok = false;
                break;
            }
            const size_t n = ggml_nbytes(vs);
            if (got.size() < n) { want.resize(n); got.resize(n); }
            bad_slab_bytes = n;
            // The copy-ready event has already been synchronized by the caller,
            // so these read the bytes the FFN is about to consume.
            ggml_backend_tensor_get(vs, want.data(), 0, n);
            ggml_backend_tensor_get(vd, got.data(),  0, n);
            ++rowset_recon_members;
            rowset_recon_compared += n;
            if (std::memcmp(want.data(), got.data(), n) != 0) {
                size_t at = 0;
                while (at < n && want[at] == got[at]) { ++at; }
                ++rowset_recon_bad;
                bad_offset = at; bad_expert = uniq[i]; bad_member = (int) mi;
                ok = false;
            }
        }
    }
    ++rowset_recon_windows;
    ggml_free(vctx);

    if (!ok && bad_expert >= 0) {
        const std::string why =
            "recon: slot bytes do NOT match the checkpoint at layer " + std::to_string(il) +
            " member " + std::to_string(bad_member) + " expert " + std::to_string(bad_expert) +
            " (first differing byte at offset " +            std::to_string(bad_offset) + " of " +
            std::to_string(bad_slab_bytes) + ")";
        tel_rowset_refused(il, (int) (il % (int) std::max<size_t>(n_slots, 1)), gen,
                           "recon_mismatch", why.c_str());
        error_msg = "load_row_set: " + why;
        return false;
    }
    return ok;
}

bool llama_layer_streaming::build_row_set_window(int width) {
    const int n_exp = (int) model_ref->hparams.n_expert;
    if (width < 1 || width > n_exp) {
        error_msg = "build_row_set_window: width " + std::to_string(width) +
                    " outside [1," + std::to_string(n_exp) + "]";
        return false;
    }
    if (width == n_exp) {
        // Requested width equals the bank: nothing to narrow, and leaving the
        // full-bank path alone is strictly safer than rebuilding it.
        return true;
    }

    const ggml_backend_buffer_type_t buft_cuda = ggml_backend_cuda_buffer_type(device);

    // PHASE 1 -- build every narrow bundle and size the compact buffer, WITHOUT
    // touching the live bundles or the full-bank buffer. If anything fails here
    // the slot is exactly as it was, so a refused narrowing is a clean refusal
    // and never a half-narrowed slot that would index out of bounds.
    struct staged_slot {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_ptr buf;
        std::unordered_map<int, size_t> bytes;
        std::vector<std::vector<ggml_tensor *>> tensors; // parallel to the layer list
        std::vector<int> layers;
        size_t need = 0;
    };
    std::vector<staged_slot> staged(slots.size());

    for (size_t si = 0; si < slots.size(); ++si) {
        staged_slot & st = staged[si];
        ggml_init_params mp = { 4u*1024u*1024u, nullptr, true };
        st.ctx = ggml_init(mp);
        if (st.ctx == nullptr) {
            error_msg = "build_row_set_window: failed to init the compact tensor context for slot " +
                        std::to_string(si);
            for (auto & done : staged) { if (done.ctx) { ggml_free(done.ctx); done.ctx = nullptr; } }
            return false;
        }
        for (int il = (int) si; il < n_layer; il += n_slots) {
            const llama_ls_package & pkg = pkg_cache[il];
            std::vector<ggml_tensor *> ts;
            size_t bytes = 0;
            for (const auto & m : pkg.members) {
                const ggml_tensor * t = m.src;
                if (t == nullptr) {
                    continue;
                }
                ggml_tensor * nt = make_dev_tensor(st.ctx, t, width);
                if (nt == nullptr) {
                    error_msg = "build_row_set_window: failed to create the compact tensor for "
                                "layer " + std::to_string(il);
                    for (auto & done : staged) { if (done.ctx) { ggml_free(done.ctx); done.ctx = nullptr; } }
                    return false;
                }
                ggml_set_name(nt, t->name);
                bytes += ggml_backend_buft_get_alloc_size(buft_cuda, nt);
                ts.push_back(nt);
            }
            st.bytes[il] = bytes;
            st.tensors.push_back(std::move(ts));
            st.layers.push_back(il);
            st.need = std::max(st.need, bytes);
        }
    }

    // PHASE 2 -- allocate the compact buffers. Still nothing destructive.
    for (size_t si = 0; si < slots.size(); ++si) {
        staged[si].buf.reset(ggml_backend_buft_alloc_buffer(buft_cuda, staged[si].need + 4096));
        if (!staged[si].buf) {
            error_msg = "build_row_set_window: failed to allocate the compact slot buffer " +
                        std::to_string(staged[si].need + 4096) + " bytes for slot " + std::to_string(si);
            for (auto & done : staged) { if (done.ctx) { ggml_free(done.ctx); done.ctx = nullptr; } }
            return false;
        }
        for (size_t li = 0; li < staged[si].tensors.size(); ++li) {
            size_t off = 0;
            if (!alloc_tensors(staged[si].buf.get(), buft_cuda, staged[si].tensors[li], off)) {
                error_msg = "build_row_set_window: failed to place the compact tensors for layer " +
                            std::to_string(staged[si].layers[li]);
                for (auto & done : staged) { if (done.ctx) { ggml_free(done.ctx); done.ctx = nullptr; } }
                return false;
            }
        }
    }

    // PHASE 3 -- commit. Every fallible step is done; from here the swap cannot
    // fail. The narrow bundles replace the full-bank ones in place, so every
    // consumer of slot.layers[il] sees the new shape, and THEN the full-bank
    // buffer is released -- which is where the reservation comes from.
    for (size_t si = 0; si < slots.size(); ++si) {
        llama_layer_streaming_slot & slot = slots[si];
        for (size_t li = 0; li < staged[si].layers.size(); ++li) {
            const int il = staged[si].layers[li];
            llama_layer & dst = slot.layers[il];
            if (adapter->merge_resident) {
                dst = *host_layers[il];
            }
            size_t k = 0;
            for (const auto & m : pkg_cache[il].members) {
                if (m.src == nullptr) {
                    continue;
                }
                ls_dst(dst, m) = staged[si].tensors[li][k++];
            }
            slot.layer_bytes_rowset[il] = staged[si].bytes[il];
        }
        // A bank_span is sized from the FULL-bank tiling and aliases the old
        // buffer's base, so it is stale the moment the bundles are replaced.
        // stage_members() re-validates it against the source sizes and would
        // fall back to per-member copies; dropping it here makes that certain
        // rather than merely likely.
        slot.bank_span.clear();
        slot.ctx_rowset = staged[si].ctx;
        slot.buf_rowset = std::move(staged[si].buf);
        slot.buf.reset();
    }

    row_set_window_width = width;
    return true;
}

// T-GPLR-62 S2 (lifetime): flag every phase-1 (TG) set for a fresh-context
// rebuild. The sets themselves are untouched here -- only the flag and the
// replacement arm -- so this cannot fail after the commit and cannot leave a
// half-flagged executor. It DOES refuse loudly if a phase-1 set exists that is
// not a finished build, because replace_set_with_fresh_rebuild() rejects those
// and the failure would otherwise surface as a confusing mid-TG error.
// T-GPLR-71: the view-width arm. This is deliberately NOT a new lifecycle
// path: it sets exactly the two flags T-GPLR-62 S2 sets (needs_rebuild and
// rebuild_replacement), so begin_set takes the identical T-GPLR-47 fresh-context
// replacement branch. Recurrent session state is untouched by that branch --
// it re-derives GRAPHS against the current memory context and re-applies
// nothing -- which is exactly why it is safe here and re-running
// layer_streaming_prebuild is not (that one calls vmctx->apply()).
void llama_layer_streaming::arm_tg_view_rebuild(graph_set & set, int32_t built, int32_t required) {
    set.needs_rebuild = true;
    rebuild_replacement = true;
    view_rebuild_ctr++;   // T-GPLR-71 step 5: read by phase_summary (AC6)

    // Same measured trap llama_ls_tel_mark() already fixes 400 lines above, and
    // for the same reason: on Windows ggml_time_us() divides by the static
    // timer_freq that ggml_time_init() sets, so a call before any llama init
    // dies with 0xC0000094. It cannot fire on the runner (llama_backend_init()
    // initialises the timer long before the first generated token), which is
    // exactly why it survived review -- this function had never been executed by
    // anything until the T-GPLR-71 guard test drove it, and that test is what
    // surfaced it. A telemetry line must never be able to kill the process, and
    // the state transition above must not depend on it. InitOnce-guarded, so this
    // is free and llama_backend_init()'s later call is a no-op.
    ggml_time_init();

    tel_write(tel_out,
        "{\"event\":\"tg_view_rebuild\",\"run_id\":%llu,\"set_mode\":%d,\"n_tokens\":%lld,"
        "\"rs_z\":%d,\"n_outputs\":%d,\"built_n_kv\":%d,\"required_n_kv\":%d,"
        "\"mechanism\":\"t-gplr-47-fresh-context-rebuild\",\"t_us\":%lld}",
        (unsigned long long) run_id, set.mode, (long long) set.n_tokens, set.rs_z,
        set.n_outputs, built, required, (long long) ggml_time_us());
    fprintf(stderr,
            "T-GPLR-71: phase-1 graph set (n_tokens %lld, rs_z %d) was built against a "
            "%d-column attention view and %d are now live; flagging it for a "
            "fresh-context rebuild.\n",
            (long long) set.n_tokens, set.rs_z, built, required);
    fflush(stderr);
}

bool llama_layer_streaming::mark_row_set_sets_stale() {
    int n_flagged = 0;
    for (auto & set : sets) {
        if (set.mode != 1) {
            continue;
        }
        if (!set.built || set.ctx == nullptr || set.res == nullptr) {
            error_msg = "mark_row_set_sets_stale: phase-1 set (n_tokens " +
                        std::to_string(set.n_tokens) + ", rs_z " + std::to_string(set.rs_z) +
                        ") is not a finished build; the narrowing cannot re-derive it";
            return false;
        }
        set.needs_rebuild = true;
        ++n_flagged;
    }
    if (n_flagged == 0) {
        error_msg = "mark_row_set_sets_stale: no phase-1 graph set exists to rebuild after the "
                    "narrowing; the two-part TG graphs would run against a replaced bundle";
        return false;
    }
    // Arm the replacement so the executor's build-on-miss takes begin_set's
    // T-GPLR-47 fresh-context branch. Stays armed: it is consumed per set, and
    // a rebuild triggered by anything else later is equally correct.
    rebuild_replacement = true;

    // Emitted so the trace records WHY a graph build is about to occur inside
    // the TG region. The validator treats this exactly like a handoff event
    // (T-Q38HG-02): a topology change legitimately invalidates prebuilt graphs,
    // and the resulting graph_builds > 0 is expected rather than a violation.
    tel_write(tel_out,
        "{\"event\":\"rowset_narrow\",\"run_id\":%llu,\"window_width\":%d,\"sets_flagged\":%d,"
        "\"mechanism\":\"t-gplr-47-fresh-context-rebuild\",\"t_us\":%lld}",
        (unsigned long long) run_id, row_set_window_width, n_flagged, (long long) ggml_time_us());
    fprintf(stderr,
            "T-GPLR-62 S2: flagged %d phase-1 graph set(s) for fresh-context rebuild "
            "(their weight pointers were baked against the replaced bundle)\n", n_flagged);
    return true;
}

void llama_layer_streaming::end_row_set() {
    if (!row_set_active && !cache_active) {
        return;
    }
    row_set_active = false;
    row_set_requested = false;
    row_set_have.clear();
    row_set_owner.clear();
    row_set_gen.clear();
    row_set_ids.clear();
    rowset_round_bytes_staged = rowset_round_bytes_bank = 0;
    rowset_round_rows_fetched = rowset_round_rows_reused = 0;
    rowset_round_rows_retained = rowset_round_rows_retired = 0;
    end_row_cache();
}

ggml_cgraph * llama_layer_streaming::new_ff_graph(int layer) {
    if (cur == nullptr || cur->ctx == nullptr) {
        error_msg = "new_ff_graph: begin_build not called";
        return nullptr;
    }
    ggml_cgraph * g = ggml_new_graph_custom(cur->ctx, 2048, false);
    if (g == nullptr) {
        error_msg = "new_ff_graph: failed to create graph";
        return nullptr;
    }
    cur->ff_graphs.push_back(g);
    cur->ff_layer.push_back(layer);
    cur->ff_input.push_back(nullptr); // the builder binds the INPUT right after
    cur->ff_input_wpos.push_back(nullptr); // T-GPLR-62 S2: same, for window positions
    cur->ff_input_hsel.push_back(nullptr); // T-GPLR-82 S2: same, for the host leg's ids
    cur->ff_host_experts.push_back(nullptr); // T-GPLR-61 step C: stashed by build_moe_host_leg
    return g;
}

bool llama_layer_streaming::load_row_set(int il, const std::vector<int32_t> & ids,
                                          const std::vector<std::vector<int32_t>> * per_token) {
    if (!row_set_active) {
        error_msg = "load_row_set: row-set staging is not active";
        return false;
    }
    if (il < 0 || il >= n_layer) {
        error_msg = "load_row_set: logical layer " + std::to_string(il) + " out of range";
        return false;
    }
    const int slot_idx = il % (int) slots.size();
    llama_layer_streaming_slot & slot = slots[slot_idx];

    llama_layer & dst = slot.layers.at(il);
    const llama_ls_package & pkg = pkg_cache.at(il);
    const int n_exp = (int) model_ref->hparams.n_expert;

    // ordered unique union of the router's executed IDs
    std::vector<int32_t> uniq(ids);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    for (int32_t e : uniq) {
        if (e < 0 || e >= n_exp) {
            tel_rowset_refused(il, slot_idx, slot.generation, "bad_shape",
                               ("expert id " + std::to_string(e) + " outside [0," + std::to_string(n_exp) + ")").c_str());
            error_msg = "load_row_set: expert id out of range for layer " + std::to_string(il);
            return false;
        }
    }
    if (uniq.empty()) {
        tel_rowset_refused(il, slot_idx, slot.generation, "coverage_gap", "empty row set");
        error_msg = "load_row_set: empty row set for layer " + std::to_string(il);
        return false;
    }

    // -----------------------------------------------------------------------
    // T-GPLR-62 S2: publish the union POSITION table for this window.
    //
    // The WHOLE table is reset to -1 first, then only the union members are
    // written. A partial write is the bug this ordering exists to prevent: an
    // expert absent from this window would otherwise keep the position it held
    // in a PREVIOUS window, and the device would compute with a different
    // expert's weights while producing a plausible answer. -1 is a refusal, not
    // a position.
    //
    // Window width: 0 means the slot is still full-bank (the pre-S2 shape), so
    // there is no width to violate and the check is skipped. Once a compact
    // width is configured, a union wider than the window is refused here rather
    // than truncated, because truncating would silently drop experts the router
    // executed.
    if (row_set_window_width > 0 && (int) uniq.size() > row_set_window_width) {
        tel_rowset_refused(il, slot_idx, slot.generation, "window_overflow",
            ("union of " + std::to_string(uniq.size()) + " experts exceeds the configured window width " +
             std::to_string(row_set_window_width) + " for layer " + std::to_string(il)).c_str());
        error_msg = "load_row_set: union wider than the slot window for layer " + std::to_string(il) +
                    " (" + std::to_string(uniq.size()) + " > " + std::to_string(row_set_window_width) + ")";
        return false;
    }
    row_set_union_pos_.assign((size_t) n_exp, -1);
    for (size_t i = 0; i < uniq.size(); ++i) {
        row_set_union_pos_[(size_t) uniq[i]] = (int32_t) i;
    }
    row_set_union_valid_ = true;

    // -----------------------------------------------------------------------
    // T-GPLR-61 S3/S4: the cost gate, the budget, and the copy-wait split.
    // The gate is evaluated ONCE per load from the measured constants
    // (tools/expert-cpu-bench, iteration-disjoint, 2026-09-30: 36.5 us per
    // host expert; 10.35 GB/s end-to-end feed; 1,769,472 B per Q4_K routed
    // row) and stashed per layer, so the executor's ls_plan_split and the
    // layer_compute record see the SAME verdict the wait decision used. An
    // unevaluable policy refuses the run (SDD moe-row-split failure contract);
    // with these constants it always evaluates, so a flag-absent run gains no
    // new refusal path.
    llama_ls_policy::ls_gate_constants gate_c;
    gate_c.host_us_per_expert = 36.5;
    gate_c.row_bytes           = 1769472u;
    gate_c.link_bytes_per_s    = 10.35e9;
    gate_c.policy              = llama_ls_policy::ls_gate_policy::HOST_ALWAYS;
    const llama_ls_policy::ls_gate_verdict gate_v = llama_ls_policy::ls_evaluate_gate(gate_c);
    if (!gate_v.evaluable) {
        tel_rowset_refused(il, slot_idx, slot.generation, "gate_unevaluable",
                           gate_v.refuse_reason ? gate_v.refuse_reason : "cost gate is not evaluable");
        error_msg = std::string("load_row_set: cost gate is not evaluable (") +
                    (gate_v.refuse_reason ? gate_v.refuse_reason : "unknown") + ")";
        return false;
    }
    row_set_gate_open[(size_t) il]    = gate_v.open ? 1 : 0;
    row_set_gate_host_us[(size_t) il] = gate_v.host_us;
    row_set_gate_wait_us[(size_t) il] = gate_v.wait_us;

    // S3 is one package: host leg + per-token demand + a non-zero budget, and
    // it stays OFF for the recon and predictor arms, whose own checks assume
    // every copy was awaited before consume. With the flag absent or the
    // budget at its default of 0, s3_cap stays 0 and both branches below take
    // exactly today's path: every staged row waits, one blocking sync.
    const bool s3 = experts_host_n > 0 && per_token != nullptr &&
                    // T-GPLR-82 FALSIFIER: the decoupling exists to let a MISSED row
                    // compute on the HOST leg while its copy runs. With the tail leg on the
                    // DEVICE there is no host leg to absorb a skipped wait, so a skipped
                    // wait means the device leg reads a row that was never staged --
                    // MEASURED 2026-10-10 (session gplr82-a2dev, exe a9ddf169, arm A2dev with
                    // an unbounded budget): 512 of 512 logit readouts non-finite while
                    // ffn_dev read 1 (all device) and the ledger reported 0 host rows. The
                    // conjunct is therefore part of the mode, not a tuning flag: this
                    // predicate and the graph builder must agree on whether a host leg
                    // exists or the run computes on unstaged bytes.
                    !experts_host_leg_device &&
                    experts_host_budget_us > 0.0 && !row_set_recon &&
                    !row_set_spec_on && row_set_spec_mode == 0 &&
                    row_set_spec_arm < 0 && row_set_spec_pool == 0;
    int s3_cap = 0;
    if (s3 && gate_v.open) {
        const double k = std::floor(experts_host_budget_us / gate_v.host_us);
        s3_cap = k >= (double) experts_host_n ? experts_host_n : (int) k;
        if (s3_cap < 0) {
            s3_cap = 0;
        }
    }

    // The per-token admission split. A row is PENDING pre-stage when it is in
    // this load's fetch (about to be issued) or outstanding from an earlier
    // skipped wait. Per token, the first min(pending, s3_cap) pending rows in
    // router order are admitted to the host leg; every remaining pending row
    // of ANY token joins the wait set. A row admitted by one token but waited
    // by another is waited: admission only removes a wait when no token needs
    // the row on the device. Called by both branches once fetch is known.
    std::vector<uint8_t> s3_host_row((size_t) n_exp, 0);
    std::vector<uint8_t> s3_wait_row((size_t) n_exp, 0);
    bool s3_split_run = false;
    bool s3_any_wait  = false;
    auto s3_compute_split = [&](const std::vector<int32_t> & fetch) {
        std::fill(s3_host_row.begin(), s3_host_row.end(), 0);
        std::fill(s3_wait_row.begin(), s3_wait_row.end(), 0);
        s3_split_run = false;
        s3_any_wait  = false;
        if (!(s3 && s3_cap > 0)) {
            return;
        }
        const std::vector<uint8_t> & outst = row_set_outstanding[(size_t) slot_idx];
        std::vector<uint8_t> in_fetch((size_t) n_exp, 0);
        for (int32_t e : fetch) {
            in_fetch[(size_t) e] = 1;
        }
        for (const auto & row : *per_token) {
            std::vector<int32_t> pend;
            pend.reserve(row.size());
            for (int32_t e : row) {
                if (e >= 0 && (size_t) e < (size_t) n_exp &&
                    (outst[(size_t) e] || in_fetch[(size_t) e])) {
                    pend.push_back(e);
                }
            }
            const int admit = (int) pend.size() < s3_cap ? (int) pend.size() : s3_cap;
            for (int i = 0; i < admit; ++i) {
                s3_host_row[(size_t) pend[(size_t) i]] = 1;
            }
            for (size_t i = (size_t) admit; i < pend.size(); ++i) {
                s3_wait_row[(size_t) pend[i]] = 1;
            }
        }
        s3_split_run = true;
        for (int32_t e : uniq) {
            if (s3_wait_row[(size_t) e]) {
                s3_any_wait = true;
                break;
            }
        }
    };

    // -----------------------------------------------------------------------
    // T-GPLR-19 (F2-M1) retention path -- SDD section 4 execution shape (i).
    // The slot-keyed bookkeeping below is bypassed on purpose: a slot's
    // bundles alias ONE device buffer across every layer that shares it, so
    // slot coverage can never survive an intervening owner. Residency is keyed
    // by LOGICAL LAYER over the layer's private cache, the complement is H2D'd
    // into that cache, and the slot bundle is then refreshed device-side for
    // the whole union. The FFN still reads the slot with the router's own IDs,
    // so routing and every arithmetic result are untouched.
    // -----------------------------------------------------------------------
    if (row_set_retain) {
        if (!cache_active) {
            tel_rowset_refused(il, slot_idx, slot.generation, "bad_shape",
                               "retention requested but the per-layer cache is not active");
            error_msg = "load_row_set: retention requested but the per-layer cache is not active";
            return false;
        }
        if (backend_compute == nullptr) {
            tel_rowset_refused(il, slot_idx, slot.generation, "bad_shape",
                               "retention requested but the compute backend is not resolved");
            error_msg = "load_row_set: memory retention needs the compute backend (not resolved)";
            return false;
        }
        auto & exp_t = cache_expert[(size_t) il];
        auto & rec_t = cache_recency[(size_t) il];
        const int cap = row_set_cap;
        std::vector<int32_t> pool_hit_pos(uniq.size(), -1);
        if (row_set_spec_pool > 0) {
            pool_hit_ids_last.clear();
            pool_consumed_last = 0;
            pool_pending_fallbacks_last = 0;
            pool_d2d_bytes_last = 0;
            pool_begin_generation(il, slot.generation);
        }
        // -------------------------------------------------------------------
        // T-GPLR-58 A0: speculative placement state for THIS layer.
        // P4's generation boundary: a fill issued under a previous slot
        // generation is stale and is cancelled BEFORE this window is
        // classified, so it can never be consumed as this generation's row.
        // The mark is non-blocking -- the position is released by the reap,
        // which runs on the speculation path only (P1).
        // `spec_dup_pos` collects fills this window finds IN FLIGHT for a row it
        // needs; they are cancelled as duplicates once the window is classified.
        // -------------------------------------------------------------------
        std::vector<int32_t> spec_dup_pos;
        uint8_t * sp_st = nullptr;
        std::vector<uint8_t> * sp_fl = nullptr;
        if (row_set_spec_on && (size_t) il < spec_state.size()) {
            sp_st = spec_state[(size_t) il].data();
            sp_fl = &spec_flag[(size_t) il];
            // P1's window: from here until the demand's own ready wait closes,
            // any reap would be a demand waiting on a speculative copy, so
            // spec_reap() COUNTS it there instead of assuming it away.
            spec_in_demand_window = true;
            for (int p = 0; p < cap; ++p) {
                if (sp_st[(size_t) p] == LLAMA_LS_SPEC_PENDING &&
                    spec_gen[(size_t) il][(size_t) p] != slot.generation) {
                    sp_st[(size_t) p] = LLAMA_LS_SPEC_CANCELLED;
                    spec_cancelled++;
                    spec_cancelled_generation++;
                }
            }
        }

        // 1. classify this window's union against the cache, and RECENCY-evict
        //    BEFORE placing, so an eviction can never displace a row this
        //    window still needs. The policy is a row-capacity LRU over the
        //    layer's routed rows -- the policy SDD gate 9's statistic (b)
        //    measures (T-GPLR-19 policy resolution, 2026-09-13). Its footprint
        //    is cap = min(n_expert, K*top_k) rows per layer, identical to the
        //    window-count rule it replaces, and it weakly dominates that rule:
        //    a row routed in the last K windows has at most cap-1 rows routed
        //    after it, so it is always among the cap most recently routed and
        //    can never be evicted while the window-count rule would keep it.
        //    Rows in THIS window's union are protected from eviction entirely.
        std::vector<int32_t> pos_of(uniq.size(), -1);
        std::vector<int32_t> fetch, fetch_pos;
        size_t occupied = 0;
        // `occupied` counts PHYSICAL occupancy: a speculative fill holds its
        // position while in flight. Only a READY position is a HIT (P2), so a
        // demand for a row that a fill is still copying fetches its own bytes --
        // it never waits for the fill (P1). A fill the window turns out to want
        // is recorded as matched (the guess was right, and too late to be used)
        // and cancelled as a duplicate below.
        for (int p = 0; p < cap; ++p) {
            const int32_t x = exp_t[(size_t) p];
            if (x < 0) {
                continue;
            }
            occupied++;
            const uint8_t st = sp_st != nullptr ? sp_st[(size_t) p] : (uint8_t) LLAMA_LS_SPEC_READY;
            const auto it = std::lower_bound(uniq.begin(), uniq.end(), x);
            if (it != uniq.end() && *it == x) {
                if (llama_ls_spec_is_hit(st)) {
                    pos_of[(size_t) (it - uniq.begin())] = p; // hit: served from the cache
                } else if (st == LLAMA_LS_SPEC_PENDING) {
                    spec_matched_by_demand++;
                    spec_dup_pos.push_back(p);
                }
            }
        }
        size_t resident = 0;
        for (size_t i = 0; i < uniq.size(); ++i) {
            resident += pos_of[i] >= 0;
        }
        if (row_set_spec_pool > 0) {
            for (size_t i = 0; i < uniq.size(); ++i) {
                if (pos_of[i] >= 0) {
                    continue;
                }
                const llama_ls_pool_lookup hit = llama_ls_pool_lookup_expert(
                    pool_expert[(size_t) il].data(), pool_state[(size_t) il].data(),
                    pool_gen[(size_t) il].data(), row_set_spec_pool, slot.generation, uniq[i]);
                if (hit.ready_pos >= 0) {
                    pool_hit_pos[i] = hit.ready_pos;
                    pool_hit_ids_last.push_back(uniq[i]);
                } else if (hit.pending_pos >= 0) {
                    pool_cancel_demand(il, uniq[i], slot.generation);
                }
            }
        }
        const size_t pool_hits = (size_t) std::count_if(
            pool_hit_pos.begin(), pool_hit_pos.end(), [](int32_t p) { return p >= 0; });
        const size_t need = uniq.size() - resident - pool_hits;

        // 2. evict least-recently-routed positions until the fetched complement
        //    fits. A row needed by this window is never a victim, so a refusal
        //    here means the union itself outgrew the DECLARED capacity -- loud,
        //    never silent growth and never a claim on another layer's bytes.
        size_t retired = 0;
        while (occupied + need > (size_t) cap) {
            int victim = -1;
            for (int p = 0; p < cap; ++p) {
                const int32_t x = exp_t[(size_t) p];
                if (x < 0) {
                    continue;
                }
                const uint8_t st = sp_st != nullptr ? sp_st[(size_t) p] : (uint8_t) LLAMA_LS_SPEC_READY;
                if (!llama_ls_spec_may_evict(st)) {
                    continue; // bytes in flight: the reap releases it, an eviction never does
                }
                const auto it = std::lower_bound(uniq.begin(), uniq.end(), x);
                if (it != uniq.end() && *it == x) {
                    continue; // needed by this window: never evicted
                }
                const uint8_t sf = (sp_st != nullptr && (*sp_fl)[(size_t) p] != 0) ? 1 : 0;
                if (victim < 0) {
                    victim = p;
                    continue;
                }
                const uint8_t vf = (sp_st != nullptr && (*sp_fl)[(size_t) victim] != 0) ? 1 : 0;
                // P3: a speculative row is the FIRST victim; then the incumbent's
                // own least-recently-routed rule, byte-for-byte unchanged.
                if (llama_ls_spec_victim_better(rec_t[(size_t) p], sf, true,
                                                rec_t[(size_t) victim], vf, true)) {
                    victim = p;
                }
            }
            if (victim < 0) {
                tel_rowset_refused(il, slot_idx, slot.generation, "insufficient_vram",
                    ("per-layer cache holds " + std::to_string(cap) + " rows; layer " + std::to_string(il) +
                     " still needs " + std::to_string(uniq.size()) + " after eviction").c_str());
                error_msg = "load_row_set: per-layer cache capacity exceeded for layer " + std::to_string(il);
                return false;
            }
            exp_t[(size_t) victim] = -1;
            rec_t[(size_t) victim] = -1;
            if (sp_st != nullptr) {
                // T-GPLR-58 A0: the row is gone, so its placement state goes
                // with it. Leaving a stale READY here would make the position
                // look occupied to the free-position search and silently shrink
                // the capacity the demand side was promised.
                sp_st[(size_t) victim]  = LLAMA_LS_SPEC_FREE;
                (*sp_fl)[(size_t) victim] = 0;
                spec_tick[(size_t) il][(size_t) victim] = -1;
                if ((size_t) il < spec_issue_ord.size()) {
                    spec_issue_ord[(size_t) il][(size_t) victim] = -1;
                }
            }
            occupied--;
            retired++;
        }

        // 3. place the complement and stamp recency across the WHOLE union in
        //    ascending expert-id order, so both a hit and a miss refresh the
        //    row's recency deterministically and the in-window tie-break matches
        //    the offline analyzer's.
        for (size_t i = 0; i < uniq.size(); ++i) {
            if (pool_hit_pos[i] >= 0) {
                continue; // consumed below after the slot's ready event
            }
            if (pos_of[i] >= 0) {
                rec_t[(size_t) pos_of[i]] = ++cache_tick;
                if (sp_fl != nullptr && (*sp_fl)[(size_t) pos_of[i]] != 0) {
                    // the demand consumed the speculative row: it is an ordinary
                    // resident from now on (no longer a first victim)
                    (*sp_fl)[(size_t) pos_of[i]] = 0;
                    spec_note_lead(il, (int) pos_of[i]); // T-GPLR-58 A2: M7's datum
                }
                continue;
            }
            int free_p = llama_ls_spec_free_pos(exp_t.data(), sp_st, cap);
            if (free_p < 0) {
                tel_rowset_refused(il, slot_idx, slot.generation, "insufficient_vram",
                    ("per-layer cache holds " + std::to_string(cap) + " rows; layer " + std::to_string(il) +
                     " still needs " + std::to_string(uniq.size()) + " after eviction").c_str());
                error_msg = "load_row_set: per-layer cache capacity exceeded for layer " + std::to_string(il);
                return false;
            }
            exp_t[(size_t) free_p] = uniq[i];
            rec_t[(size_t) free_p] = ++cache_tick;
            if (sp_st != nullptr) {
                // a row the demand itself placed: READY the moment its own copy
                // completes, and never counted as speculative (P2/P3)
                sp_st[(size_t) free_p] = LLAMA_LS_SPEC_READY;
                spec_tick[(size_t) il][(size_t) free_p] = -1;
                if ((size_t) il < spec_issue_ord.size()) {
                    spec_issue_ord[(size_t) il][(size_t) free_p] = -1;
                }
                spec_gen[(size_t) il][(size_t) free_p] = slot.generation;
            }
            pos_of[i] = free_p;
            fetch.push_back(uniq[i]);
            fetch_pos.push_back(free_p);
        }

        // T-GPLR-62 S1: post-placement CONTAINMENT self-check. Every member of
        // the union must now name exactly one source -- a cache position, a
        // prediction-pool position, or nothing. A member with no source is the
        // one condition under which step 4b would read a position it never
        // wrote, i.e. compute with a DIFFERENT expert than the router selected.
        // It is refused here, loudly, instead of being discovered at 4b.
        for (size_t i = 0; i < uniq.size(); ++i) {
            if (pos_of[i] >= 0 || pool_hit_pos[i] >= 0) {
                continue;
            }
            tel_rowset_refused(il, slot_idx, slot.generation, "coverage_gap",
                ("union member " + std::to_string(uniq[i]) + " of layer " + std::to_string(il) +
                 " has no source after placement (cache miss, no pool hit)").c_str());
            error_msg = "load_row_set: union member without a source for layer " + std::to_string(il) +
                        " (expert " + std::to_string(uniq[i]) + ")";
            return false;
        }

        // P4 at a demand stage: a fill for a row THIS window fetches is now a
        // duplicate -- its bytes are superseded by the demand's own copy, so the
        // claim is cancelled (marked here, released by the next reap, which is
        // why this costs the demand no wait).
        for (int32_t p : spec_dup_pos) {
            if (sp_st != nullptr && sp_st[(size_t) p] == LLAMA_LS_SPEC_PENDING) {
                sp_st[(size_t) p] = LLAMA_LS_SPEC_CANCELLED;
                spec_cancelled++;
                spec_cancelled_demand++;
            }
        }
        const size_t retained = resident;

        // T-GPLR-61 S3: admission split (common block above). The retention
        // path stages into the per-layer CACHE, so the split names fetch
        // indices: k_wait must complete before consume; k_host rows may
        // still be in flight at consume (the host leg reads the mmap).
        s3_compute_split(fetch);
        std::vector<size_t> s3_k_wait, s3_k_host;
        if (s3_split_run) {
            for (size_t k = 0; k < fetch.size(); ++k) {
                const int32_t e = fetch[k];
                if (s3_host_row[(size_t) e] && !s3_wait_row[(size_t) e]) {
                    s3_k_host.push_back(k);
                } else {
                    s3_k_wait.push_back(k);
                }
            }
        } else {
            s3_k_wait.resize(fetch.size());
            for (size_t k = 0; k < fetch.size(); ++k) {
                s3_k_wait[k] = k;
            }
        }
        const bool s3_need_wait = s3_split_run ? s3_any_wait : true;
        int64_t s3_t_enq1 = -1, s3_t_sync_end = -1, s3_waited = 0;

        // scratch context for the slab views the refresh needs; re-created when
        // exhausted so one window never pays a per-view allocation
        const size_t n_views = 2 * uniq.size() * pkg.members.size() + 8;
        if (cache_view_ctx == nullptr || cache_view_used + n_views > cache_view_cap) {
            if (cache_view_ctx != nullptr) {
                ggml_free(cache_view_ctx);
            }
            ggml_init_params vparams = { 4u * 1024u * 1024u, nullptr, true };
            cache_view_ctx = ggml_init(vparams);
            if (cache_view_ctx == nullptr) {
                error_msg = "load_row_set: failed to create the cache view context";
                return false;
            }
            cache_view_used = 0;
            cache_view_cap  = 2048;
        }
        cache_view_used += n_views;

        std::fill(row_set_fresh[(size_t) il].begin(), row_set_fresh[(size_t) il].end(), 0);
        ggml_backend_event_record((ggml_backend_event_t) slot.ev_copy_start, backend_copy);
        const int64_t t_submit0 = ggml_time_us();
        size_t bytes_staged = 0;
        const std::vector<ggml_tensor *> & cache_mem = cache_tens[(size_t) il];
        std::vector<ggml_tensor *> pool_mem;
        if (row_set_spec_pool > 0) {
            if (pool_tens.size() <= (size_t) il) {
                tel_rowset_refused(il, slot_idx, slot.generation, "pool_source",
                                   "pool hit has no per-layer tensor storage");
                error_msg = "load_row_set: prediction-pool tensor storage is not allocated";
                failed = true;
                return false;
            }
            pool_mem = pool_tens[(size_t) il];
        }
        if (pool_hits > 0 && pool_mem.size() != pkg.members.size()) {
            tel_rowset_refused(il, slot_idx, slot.generation, "pool_source",
                               "pool hit has no matching per-member tensor storage");
            error_msg = "load_row_set: prediction-pool hit has no matching tensor storage";
            failed = true;
            return false;
        }
        if (pool_hits > 0) {
            for (size_t mi = 0; mi < pkg.members.size(); ++mi) {
                if (pkg.members[mi].src != nullptr && pool_mem[mi] == nullptr) {
                    tel_rowset_refused(il, slot_idx, slot.generation, "pool_source",
                                       "pool hit member has no prediction tensor");
                    error_msg = "load_row_set: prediction-pool member is missing";
                    failed = true;
                    return false;
                }
            }
        }

        // 4a. H2D the complement into the cache at its assigned position. The
        //     source offset is the expert's natural slab offset, so the bytes
        //     handed to the copy are the checkpoint's own quantized rows.
        auto s3_h2d = [&](const std::vector<size_t> & ks) {
            if (ks.empty()) {
                return;
            }
            for (size_t mi = 0; mi < pkg.members.size(); ++mi) {
                const ggml_tensor * src_t   = pkg.members[mi].src;
                ggml_tensor         * cache_t = mi < cache_mem.size() ? cache_mem[mi] : nullptr;
                if (src_t == nullptr || cache_t == nullptr) {
                    continue;
                }
                const size_t slab_bytes = ggml_row_size(src_t->type, src_t->ne[0]) * (size_t) src_t->ne[1];
                for (size_t k : ks) {
                    const size_t src_off = fetch[k] * slab_bytes;
                    const size_t dst_off = fetch_pos[k] * slab_bytes;
                    ggml_backend_tensor_set_async(backend_copy, (ggml_tensor *) cache_t,
                                                  (const char *) src_t->data + src_off, dst_off, slab_bytes);
                    bytes_staged += slab_bytes;
                }
            }
        };
        // 4a (batch 1): H2D the rows the device leg must see complete.
        s3_h2d(s3_k_wait);

        // 4b. device-side refresh of the slot bundle from the cache for the
        //     WHOLE union, so the slot the FFN is about to read is complete.
        //     ggml_backend_tensor_copy_async queues the D2D on the COPY
        //     backend's stream and makes the compute stream wait on it, so the
        //     single host-block below orders both halves without an extra
        //     device synchronize.
        for (size_t mi = 0; mi < pkg.members.size(); ++mi) {
            const llama_ls_member & m = pkg.members[mi];
            const ggml_tensor * cache_t = mi < cache_mem.size() ? cache_mem[mi] : nullptr;
            ggml_tensor * dst_t = ls_dst(dst, m);
            if (cache_t == nullptr || dst_t == nullptr) {
                continue;
            }
            for (size_t i = 0; i < uniq.size(); ++i) {
                // T-GPLR-61 S3: a host-admitted row's cache write is in
                // the unsynced batch, so its slot refresh is deferred to
                // the row's next demand. The cache still lands now --
                // warmth preserved; nothing this round reads that slab.
                if (s3_split_run && s3_host_row[(size_t) uniq[i]] &&
                    !s3_wait_row[(size_t) uniq[i]]) {
                    continue;
                }
                const int32_t p = pos_of[i];
                const int pp = pool_hit_pos[i];
                ggml_tensor * src_cache = p >= 0 ? (ggml_tensor *) cache_t
                                                  : (pp >= 0 ? (ggml_tensor *) pool_mem[mi] : nullptr);
                // T-GPLR-62 S1: the pre-4b containment check makes this
                // unreachable, but a GGML_ASSERT here would ABORT the process --
                // no telemetry, no stable reason, no clean exit. SDD section 4
                // requires a loud structured refusal, so refuse instead.
                if (src_cache == nullptr) {
                    tel_rowset_refused(il, slot_idx, slot.generation, "coverage_gap",
                        ("row-set refresh without a source: expert " + std::to_string(uniq[i]) +
                         " of layer " + std::to_string(il) + " (member " + std::to_string(mi) + ")").c_str());
                    error_msg = "load_row_set: row-set refresh without a source for layer " +
                                std::to_string(il) + " (expert " + std::to_string(uniq[i]) + ")";
                    return false;
                }
                ggml_tensor * v_src = ggml_view_2d(cache_view_ctx, src_cache,
                                                   src_cache->ne[0], src_cache->ne[1], src_cache->nb[1],
                                                   (size_t) (p >= 0 ? p : pp) * src_cache->nb[2]);
                // T-GPLR-62 S2: the DESTINATION slab is the union INDEX when the
                // slot is a compact window, and the ABSOLUTE expert id when it is
                // still bank-shaped. These two must never be mixed: addressing a
                // narrow bundle by absolute id would write outside it, and
                // addressing a full bank by union index would silently put expert
                // uniq[i] at another expert's position. The switch is
                // row_set_window_width, which is only non-zero once
                // build_row_set_window() has actually rebuilt the bundles, so the
                // two conditions cannot disagree.
                const size_t dst_slab = (row_set_window_width > 0)
                                            ? i
                                            : (size_t) uniq[i];
                ggml_tensor * v_dst = ggml_view_2d(cache_view_ctx, dst_t,
                                                   dst_t->ne[0], dst_t->ne[1], dst_t->nb[1],
                                                   dst_slab * dst_t->nb[2]);
                if (v_src == nullptr || v_dst == nullptr) {
                    error_msg = "load_row_set: failed to create a slab view for layer " + std::to_string(il);
                    return false;
                }
                ggml_backend_tensor_copy_async(backend_copy, backend_compute, v_src, v_dst);
                row_set_fresh[(size_t) il][(size_t) uniq[i]] = 1;
            }
        }

        ggml_backend_event_record((ggml_backend_event_t) slot.ev_copy_end,   backend_copy);
        ls_stage_record(slot.ev_copy_ready);
        slot.t_copy_enqueue_done_us = ggml_time_us();
        s3_t_enq1 = slot.t_copy_enqueue_done_us;

        // host-blocking wait: the FFN graph may only consume after the copy
        // event completes (SDD lifecycle). This covers the H2D and the D2D.
        // T-GPLR-61 S3: skipped only when every pending demanded row was
        // admitted to the host leg; any waited batch keeps the sync as-is.
        if (s3_need_wait) {
            const int64_t w0 = ggml_time_us();
            ggml_backend_event_synchronize((ggml_backend_event_t) slot.ev_copy_ready);
            s3_waited     = ggml_time_us() - w0;
            s3_t_sync_end = ggml_time_us();
            std::fill(row_set_outstanding[(size_t) slot_idx].begin(),
                      row_set_outstanding[(size_t) slot_idx].end(), 0);
        }

        // T-GPLR-61 S3: batch 2 -- cache writes for host-admitted rows.
        // Issued AFTER the wait so they never extend it; no slot refresh
        // is queued for them (4b above skipped them).
        if (!s3_k_host.empty()) {
            s3_h2d(s3_k_host);
            ggml_backend_event_record((ggml_backend_event_t) slot.ev_copy_end,   backend_copy);
            ls_stage_record(slot.ev_copy_ready);
            slot.t_copy_enqueue_done_us = ggml_time_us();
            std::vector<uint8_t> & outst = row_set_outstanding[(size_t) slot_idx];
            for (size_t k : s3_k_host) {
                outst[(size_t) fetch[k]] = 1;
            }
        }
        const int64_t t_demand_ready = ggml_time_us();

        // T-GPLR-61 S3: stash the post-staging classification for the
        // executor (retention: every non-fetch union member is cache- or
        // pool-resident by construction, so only outstanding is PENDING).
        {
            std::vector<uint8_t> & stt = row_set_state[(size_t) il];
            std::fill(stt.begin(), stt.end(), (uint8_t) 2);
            const std::vector<uint8_t> & outst = row_set_outstanding[(size_t) slot_idx];
            // a row admitted to the host leg had its slot refresh skipped, whatever a later wait did to outst
            // LLAMA_LS_TEST_LEGACY_S3_STATE restores the old outst-only rule so the freshness guard can be shown to fire
            static const bool legacy_state = getenv("LLAMA_LS_TEST_LEGACY_S3_STATE") != nullptr;
            for (int32_t e : uniq) {
                const bool skipped = !legacy_state && s3_split_run && s3_host_row[(size_t) e] && !s3_wait_row[(size_t) e];
                stt[(size_t) e] = (skipped || outst[(size_t) e]) ? (uint8_t) 1 : (uint8_t) 0;
            }
        }

        // T-GPLR-62 S2 / A9: the bytes are now landed and ordered against the
        // compute stream, so this reads EXACTLY what the FFN is about to
        // consume. A mismatch refuses the run: a wrong expert in the window
        // yields a plausible answer, and a run whose answer cannot be trusted
        // is worse than a run that stops.
        if (row_set_recon && !recon_row_set_window(il, pkg, dst, uniq, slot.generation)) {
            failed = true;
            return false;
        }
        if (row_set_spec_pool > 0) {
            for (size_t i = 0; i < uniq.size(); ++i) {
                if (pool_hit_pos[i] < 0) {
                    continue;
                }
                int64_t hit_bytes = 0;
                for (size_t mi = 0; mi < pkg.members.size(); ++mi) {
                    const ggml_tensor * src = pkg.members[mi].src;
                    if (src != nullptr) {
                        hit_bytes += (int64_t) llama_ls_expert_slab_bytes(src->type, src->ne[0], src->ne[1]);
                    }
                }
                if (!llama_ls_pool_retire_ready(
                        pool_expert[(size_t) il].data(), pool_state[(size_t) il].data(),
                        pool_recency[(size_t) il].data(), pool_gen[(size_t) il].data(),
                        row_set_spec_pool, slot.generation, pool_hit_pos[i])) {
                    tel_rowset_refused(il, slot_idx, slot.generation, "pool_retire",
                                       "READY pool hit could not be retired after the slot ready event");
                    error_msg = "load_row_set: prediction-pool hit could not be retired";
                    failed = true;
                    return false;
                }
                pool_issue_ord[(size_t) il][(size_t) pool_hit_pos[i]] = -1;
                pool_d2d_bytes_last += hit_bytes;
                pool_d2d_bytes_total += hit_bytes;
                ++pool_consumed_last;
                ++pool_consumed;
            }
        }
        // T-GPLR-58 A0: the demand's own wait is over, so P1's window closes
        // here. Everything speculative -- the reap that resolves finished and
        // cancelled fills and the issue of the next guesses -- happens AFTER
        // this point, which is what keeps `spec_waited_by_demand` at 0 by
        // construction rather than by hope.
        spec_in_demand_window = false;
        if (row_set_spec_on) {
            // the demand's ids, in the trace's order, are the arm's observation
            // for this stage (mode 2; mode 1 ignores them)
            spec_issue_next(il, ids.data(), (int) ids.size());
            if (failed) {
                return false;
            }
        }

        const int64_t t_enq = slot.t_copy_enqueue_done_us;
        double host_submit_us = 0.0;
        double ready_latency_us = 0.0;
        if (!s3_k_host.empty() && s3_t_enq1 >= 0) {
            // T-GPLR-61 S3: two batches -- submit is the pure pack/enqueue
            // wall across both; availability-wait is ONLY what the consumer
            // actually blocked on (0 when the wait was skipped).
            const int64_t t_split = s3_t_sync_end >= 0 ? s3_t_sync_end : s3_t_enq1;
            host_submit_us   = (double) ((s3_t_enq1 - t_submit0) + (t_enq - t_split));
            ready_latency_us = (double) s3_waited;
        } else {
            host_submit_us   = (double) (t_enq - t_submit0);
            ready_latency_us = (double) (t_demand_ready - t_enq);
        }
        const double transfer_us      = host_submit_us + ready_latency_us;
        slot.t_copy_submit_us       = 0;
        slot.t_copy_enqueue_done_us = 0;

        size_t bytes_full_bank = 0;
        if (slot.layer_bytes.count(il)) {
            bytes_full_bank = slot.layer_bytes.at(il);
        }

        rowset_round_bytes_staged  += bytes_staged;
        rowset_round_bytes_bank    += bytes_full_bank;
        rowset_round_rows_fetched  += fetch.size();
        rowset_round_rows_reused   += retained + pool_hits;
        rowset_round_rows_retained += retained;
        rowset_round_rows_retired  += retired;
        // T-GPLR-24 (H-A): feed the phase summary. total_bytes_transferred is
        // what tel_phase_summary reports as bytes_transferred; the timing
        // triple is stashed for the executor's add_layer_stats call so the
        // TG phase_summary stops reporting zeros in row-set modes.
        total_bytes_transferred += bytes_staged;
        rowset_last_submit_us   = host_submit_us;
        rowset_last_transfer_us = transfer_us;
        rowset_last_ready_us    = ready_latency_us;

        slot.logical_layer = il;
        row_set_ids[il] = uniq;

        tel_rowset_fetch(il, slot_idx, slot.generation, (int64_t) (n_ubatches - 1),
                         (int) model_ref->hparams.n_expert_used, uniq, fetch.size(),
                         bytes_staged, bytes_full_bank, host_submit_us, transfer_us, ready_latency_us,
                         retained, retained, retired, row_set_spec_pool > 0 ? &pool_hit_ids_last : nullptr);
        return true;
    }
    // the slot must still belong to the generation the row coverage was
    // tracked for; a mismatch means the staging bookkeeping is stale
    const size_t slot_i = (size_t) slot_idx;
    if (slot_i < row_set_gen.size() && row_set_gen[slot_i] != slot.generation) {
        tel_rowset_refused(il, slot_idx, slot.generation, "stale_generation", "slot generation moved without a coverage reset");
        error_msg = "load_row_set: stale slot generation for layer " + std::to_string(il);
        return false;
    }

    // slot sharing guard: a bounded window serves several logical layers per
    // token (il % n_slots). Coverage lives with the SLOT; if another layer now
    // owns it, the previous coverage is meaningless and must reset before any
    // row is claimed resident. Skipping this would let layer il + n_slots
    // execute layer il's rows.
    if (slot_i < row_set_owner.size() && row_set_owner[slot_i] != il) {
        if (row_set_owner[slot_i] != -1) {
            std::fill(row_set_have[slot_i].begin(), row_set_have[slot_i].end(), 0);
        }
        row_set_owner[slot_i] = il;
    }

    // copy only the union rows that are not resident for this generation;
    // per member (gate/up/down), rows are contiguous [n_ff, n_expert] slabs
    std::vector<int32_t> fetch;
    fetch.reserve(uniq.size());
    auto & have = row_set_have[slot_i];
    for (int32_t e : uniq) {
        if (!have[(size_t) e]) {
            fetch.push_back(e);
        }
    }

    const int64_t t_submit0 = ggml_time_us();
    size_t bytes_staged = 0;

    // T-GPLR-61 S3: partition the fetch into a WAIT batch (the device leg
    // must see these complete before consume) and a HOST batch (admitted
    // rows whose copy may still be in flight at consume -- the host leg
    // reads the mmap source, not the slot). Without admission the flow is
    // exactly today's: one batch, one record, one blocking sync.
    s3_compute_split(fetch);
    std::vector<int32_t> fetch_wait = fetch;
    std::vector<int32_t> fetch_host;
    if (s3_split_run) {
        fetch_wait.clear();
        for (int32_t e : fetch) {
            if (s3_host_row[(size_t) e] && !s3_wait_row[(size_t) e]) {
                fetch_host.push_back(e);
            } else {
                fetch_wait.push_back(e);
            }
        }
    }
    const bool s3_need_wait = s3_split_run ? s3_any_wait : true;
    int64_t   s3_t_enq1     = -1;
    int64_t   s3_t_sync_end = -1;
    int64_t   s3_waited     = 0;

    if (!fetch.empty()) {
        // T-GPLR-18: when --copy-source pinned is active the fetched slabs are
        // packed into this slot's pinned host buffer and copied from there
        // (same shape as load_layer's pack-and-enqueue). The memcpy carries
        // the bytes unchanged, so quantization and routing are untouched.
        char * staging = slot.buf_staging ? (char *) ggml_backend_buffer_get_base(slot.buf_staging.get()) : nullptr;
        if (staging != nullptr) {
            size_t need = 0;
            for (const auto & m : pkg.members) {
                if (m.src != nullptr) {
                    need += ggml_row_size(m.src->type, m.src->ne[0]) * (size_t) m.src->ne[1] * fetch.size();
                }
            }
            const size_t staging_size = ggml_backend_buffer_get_size(slot.buf_staging.get());
            if (need > staging_size) {
                // never a partial row set: a window that cannot be packed whole
                // is a loud structured refusal
                tel_rowset_refused(il, slot_idx, slot.generation, "insufficient_vram",
                    ("pinned staging window " + std::to_string(need) + " B exceeds the slot staging buffer " +
                     std::to_string(staging_size) + " B").c_str());
                error_msg = "load_row_set: pinned staging window does not fit the slot buffer for layer " + std::to_string(il);
                return false;
            }
        }
    }

    auto s3_enqueue = [&](const std::vector<int32_t> & rows) {
        if (rows.empty()) {
            return;
        }
        char * staging = slot.buf_staging ? (char *) ggml_backend_buffer_get_base(slot.buf_staging.get()) : nullptr;
        ggml_backend_event_record((ggml_backend_event_t) slot.ev_copy_start, backend_copy);
        size_t staging_off = 0;
        for (const auto & m : pkg.members) {
            const ggml_tensor * src_t = m.src;
            ggml_tensor       * dst_t = ls_dst(dst, m);
            if (src_t == nullptr) {
                continue;
            }
            GGML_ASSERT(dst_t != nullptr && "missing slot tensor for row-set member");
            // one expert occupies a contiguous SLAB of ne[1] rows (n_ff), not
            // a single row: rows are the ne[0] inner dim. The offset is
            // expert-slab granular and byte-exact (quantization untouched).
            const size_t slab_bytes = ggml_row_size(src_t->type, src_t->ne[0]) * (size_t) src_t->ne[1];
            for (int32_t e : rows) {
                const size_t off = (size_t) e * slab_bytes;
                GGML_ASSERT(off + slab_bytes <= ggml_nbytes(src_t));
                const void * copy_src = (const char *) src_t->data + off;
                if (staging != nullptr) {
                    memcpy(staging + staging_off, copy_src, slab_bytes);
                    copy_src = staging + staging_off;
                    staging_off += slab_bytes;
                }
                ggml_backend_tensor_set_async(backend_copy, dst_t, copy_src, off, slab_bytes);
            }
        }
        ggml_backend_event_record((ggml_backend_event_t) slot.ev_copy_end,   backend_copy);
        ls_stage_record(slot.ev_copy_ready);
        slot.t_copy_enqueue_done_us = ggml_time_us();
        for (int32_t e : rows) {
            have[(size_t) e] = 1;
        }
    };

    if (!fetch_wait.empty()) {
        s3_enqueue(fetch_wait);
        s3_t_enq1 = slot.t_copy_enqueue_done_us;
    }

    // host-blocking wait: the FFN graph may only consume after the copy event
    // completes (SDD lifecycle). S3 skips it only when every pending demanded
    // row was admitted to the host leg -- the device leg then reads nothing
    // that is in flight. Any waited batch (or no admission at all) keeps the
    // unconditional sync exactly as before.
    if (s3_need_wait) {
        const int64_t w0 = ggml_time_us();
        ggml_backend_event_synchronize((ggml_backend_event_t) slot.ev_copy_ready);
        s3_waited     = ggml_time_us() - w0;
        s3_t_sync_end = ggml_time_us();
        std::fill(row_set_outstanding[(size_t) slot_idx].begin(),
                  row_set_outstanding[(size_t) slot_idx].end(), 0);
    }

    if (!fetch_host.empty()) {
        s3_enqueue(fetch_host);
        std::vector<uint8_t> & outst = row_set_outstanding[(size_t) slot_idx];
        for (int32_t e : fetch_host) {
            outst[(size_t) e] = 1;
        }
    }

    // T-GPLR-61 S3: stash the post-staging classification for the executor.
    // outstanding (issued, never observed complete) is PENDING; a resident
    // row is READY; a demanded row that is neither is the staging defect the
    // SDD names, and ABSENT is what ls_plan_split refuses on.
    {
        std::vector<uint8_t> & stt = row_set_state[(size_t) il];
        std::fill(stt.begin(), stt.end(), (uint8_t) 2);
        const std::vector<uint8_t> & outst = row_set_outstanding[(size_t) slot_idx];
        std::vector<uint8_t> & fresh = row_set_fresh[(size_t) il];
        std::fill(fresh.begin(), fresh.end(), 0);
        for (int32_t e : uniq) {
            stt[(size_t) e] = outst[(size_t) e] ? 1 : (have[(size_t) e] ? 0 : 2);
            fresh[(size_t) e] = stt[(size_t) e] == 0 ? 1 : 0;
        }
    }

    double host_submit_us = 0.0, ready_latency_us = 0.0;
    if (!fetch.empty()) {
        const int64_t t_enq = slot.t_copy_enqueue_done_us;
        if (s3_t_enq1 >= 0 && !fetch_host.empty()) {
            // T-GPLR-61 S3: two batches. submit is the pure pack/enqueue
            // wall across both; availability-wait is ONLY what the consumer
            // actually blocked on (0 when the wait was skipped). The two
            // stay distinct per the evidence contract.
            const int64_t t_split = s3_t_sync_end >= 0 ? s3_t_sync_end : s3_t_enq1;
            host_submit_us   = (double) ((s3_t_enq1 - t_submit0) + (t_enq - t_split));
            ready_latency_us = (double) s3_waited;
        } else {
            host_submit_us   = (double) (t_enq - t_submit0);
            ready_latency_us = (double) (ggml_time_us() - t_enq);
        }
        slot.t_copy_submit_us = 0;
        slot.t_copy_enqueue_done_us = 0;
    }
    const double transfer_us = host_submit_us + ready_latency_us;

    size_t bytes_full_bank = 0;
    if (slot.layer_bytes.count(il)) {
        bytes_full_bank = slot.layer_bytes.at(il);
    }

    // byte account: staged bytes are the fetched expert SLABS across all
    // members (gate/up/down), matching the copy loop exactly
    for (const auto & m : pkg.members) {
        if (m.src != nullptr) {
            bytes_staged += ggml_row_size(m.src->type, m.src->ne[0]) * (size_t) m.src->ne[1] * fetch.size();
        }
    }

    rowset_round_bytes_staged += bytes_staged;
    rowset_round_bytes_bank   += bytes_full_bank;
    rowset_round_rows_fetched += fetch.size();
    rowset_round_rows_reused  += uniq.size() - fetch.size();
    // T-GPLR-24 (H-A): same phase-summary feed as the retention path above.
    total_bytes_transferred += bytes_staged;
    rowset_last_submit_us   = host_submit_us;
    rowset_last_transfer_us = transfer_us;
    rowset_last_ready_us    = ready_latency_us;

    // record which layer consumed this slot for the executor's stale guard
    slot.logical_layer = il;
    row_set_ids[il] = uniq;

    tel_rowset_fetch(il, slot_idx, slot.generation, (int64_t) (n_ubatches - 1),
                     (int) model_ref->hparams.n_expert_used, uniq, fetch.size(),
                     bytes_staged, bytes_full_bank, host_submit_us, transfer_us, ready_latency_us,
                     uniq.size() - fetch.size(), /*n_rows_retained=*/0, /*n_rows_retired=*/0);
    return true;
}

bool llama_layer_streaming::read_router_ids(int il, ggml_cgraph * g,
                                            std::vector<std::vector<int32_t>> & ids_per_token,
                                            std::vector<float> & probs_out) {
    ids_per_token.clear();
    probs_out.clear();
    if (g == nullptr) {
        error_msg = "read_router_ids: null graph";
        return false;
    }
    ggml_tensor * probs = nullptr;
    for (int k = 0; k < ggml_graph_n_nodes(g); ++k) {
        ggml_tensor * node = ggml_graph_node(g, k);
        // T-GPLR-13: identify the router selection probs by their cb-hook
        // NAME ("ffn_moe_probs-<il>") rather than by shape or op: part A
        // also contains attention softmax nodes whose ne[0] can alias
        // n_expert at some KV lengths. In single-part graphs the ARGSORT's
        // src[0] carries the same named tensor; accept both shapes.
        if (node->op == GGML_OP_SOFT_MAX && node->name != nullptr &&
            strncmp(node->name, "ffn_moe_probs-", 14) == 0) {
            probs = node;
            break;
        }
        if (node->op == GGML_OP_ARGSORT && node->src[0] != nullptr &&
            node->src[0]->name != nullptr &&
            strncmp(node->src[0]->name, "ffn_moe_probs-", 14) == 0) {
            probs = node->src[0];
            break;
        }
    }
    if (probs == nullptr || probs->data == nullptr || probs->type != GGML_TYPE_F32) {
        error_msg = "read_router_ids: router selection probs not found in the layer graph";
        return false;
    }
    const int64_t n_expert = probs->ne[0];
    const int64_t n_tok    = probs->ne[1];
    const int64_t n_used   = (model_ref != nullptr) ? (int64_t) model_ref->hparams.n_expert_used : 0;
    if (n_expert <= 0 || n_tok <= 0 || n_used <= 0 || n_used > n_expert) {
        error_msg = "read_router_ids: bad route shape";
        return false;
    }
    std::vector<float> p((size_t) n_expert * n_tok);
    ggml_backend_tensor_get(probs, p.data(), 0, p.size() * sizeof(float));

    ids_per_token.assign((size_t) n_tok, {});
    // T-GPLR-39: the same bounded exact top-k selection tel_route uses, in
    // place of rebuilding the expert order and stable-sorting all n_expert
    // experts per token. These ids ARE the routing decision -- the caller
    // memcpys them into sel_in for part B -- so the replacement has to be
    // exact, id ORDER included, with ties keeping ascending expert index (the
    // CUDA router's stable radix-sort tie-break). llama_ls_route_topk_row is
    // that same stable-descending comparator applied as a bounded insertion;
    // the n_used <= n_expert guard above makes its clamp a no-op here, so it
    // writes exactly n_used ids per row as before.
    std::vector<int32_t> ids((size_t) n_used);
    for (int64_t i = 0; i < n_tok; ++i) {
        const float * row = &p[(size_t) i * n_expert];
        llama_ls_route_topk_row(row, n_expert, n_used, ids.data());
        ids_per_token[(size_t) i].assign(ids.begin(), ids.end());
    }
    probs_out = std::move(p);
    return true;
}

void llama_layer_streaming::rowset_round_end() {
    if (!row_set_active) {
        return;
    }
    tel_rowset_round((int64_t) (n_ubatches - 1), rowset_round_bytes_staged,
                     rowset_round_bytes_bank, rowset_round_rows_fetched,
                     rowset_round_rows_reused);
    // T-GPLR-62 S2 / A9: per-round byte-recon evidence. A mismatch refuses the
    // run inside load_row_set, so a round_end that REACHED this point has
    // bad == 0 by construction -- the event is the proof, not a report. The
    // validator also requires the event to be present at all when the mode was
    // declared, so "recon was on but nothing was checked" cannot pass silently.
    if (row_set_recon) {
        tel_write(tel_out,
            "{\"event\":\"rowset_recon\",\"run_id\":%llu,\"chunk\":%lld,\"windows\":%llu,"
            "\"members\":%llu,\"compared_bytes\":%llu,\"mismatches\":%llu,"
            "\"window_width\":%d,\"t_us\":%lld}",
            (unsigned long long) run_id, (long long) (n_ubatches - 1),
            (unsigned long long) rowset_recon_windows,
            (unsigned long long) rowset_recon_members,
            (unsigned long long) rowset_recon_compared,
            (unsigned long long) rowset_recon_bad,
            row_set_window_width, (long long) ggml_time_us());
    }
    rowset_round_bytes_staged = 0;
    rowset_round_bytes_bank   = 0;
    rowset_round_rows_fetched = 0;
    rowset_round_rows_reused  = 0;
}

void llama_layer_streaming::tel_rowset_init(int n_slots_, size_t slot_bytes, int window_tokens, int n_expert, int top_k, int retain_k) {
    tel_write(tel_out,
        "{\"event\":\"rowset_init\",\"run_id\":%llu,\"policy\":\"moe-row-set\",\"n_slots\":%d,"
        "\"slot_bytes\":%zu,\"window_tokens\":%d,\"n_expert\":%d,\"top_k\":%d,\"retain_k\":%d,\"t_us\":%lld}",
        (unsigned long long) run_id, n_slots_, slot_bytes, window_tokens, n_expert, top_k, retain_k,
        (long long) ggml_time_us());
}

void llama_layer_streaming::tel_rowset_fetch(int il, int slot, uint32_t generation, int64_t chunk, int top_k,
                                             const std::vector<int32_t> & union_ids, size_t n_rows_fetched,
                                             size_t bytes_staged, size_t bytes_full_bank,
                                             double host_submit_us, double transfer_us, double ready_latency_us,
                                             size_t n_rows_resident, size_t n_rows_retained, size_t n_rows_retired,
                                             const std::vector<int32_t> * pool_hit_ids) {
    // canonical union_ids string: ascending, comma-joined, no spaces; the
    // hash is FNV-1a-64 over exactly those bytes (== _fnv1a64 in the
    // route-overlap tool)
    std::string ids_str;
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < union_ids.size(); ++i) {
        if (i) {
            ids_str += ',';
            hash ^= (uint64_t) (uint8_t) ',';
            hash *= 1099511628211ull;
        }
        ids_str += std::to_string(union_ids[i]);
        for (char c : std::to_string(union_ids[i])) {
            hash ^= (uint64_t) (uint8_t) c;
            hash *= 1099511628211ull;
        }
    }
    std::string pool_ids_json;
    if (pool_hit_ids != nullptr) {
        pool_ids_json = ",\"pool_hit_ids\":[";
        for (size_t i = 0; i < pool_hit_ids->size(); ++i) {
            pool_ids_json += (i ? "," : "");
            pool_ids_json += std::to_string((*pool_hit_ids)[i]);
        }
        pool_ids_json += "]";
    }
    tel_write(tel_out,
        "{\"event\":\"rowset_fetch\",\"run_id\":%llu,\"ubatch\":%llu,\"layer\":%d,\"slot\":%d,"
        "\"generation\":%u,\"chunk\":%lld,\"phase\":%d,\"n_tokens\":1,\"top_k\":%d,\"route_union\":%zu,"
        "\"union_ids\":\"%s\",\"hash\":%llu,\"n_rows_fetched\":%zu,\"n_rows_resident\":%zu,"
        "\"n_rows_retained\":%zu,\"n_rows_retired\":%zu%s,"
        "\"bytes_staged\":%zu,\"bytes_full_bank\":%zu,\"host_submit_us\":%.1f,\"transfer_us\":%.1f,"
        "\"ready_latency_us\":%.1f,\"ready_at_consumption\":%s,\"t_us\":%lld}",
        (unsigned long long) run_id, (unsigned long long) n_ubatches, il, slot, generation,
        (long long) chunk, phase, top_k, union_ids.size(), ids_str.c_str(),
        (unsigned long long) hash, n_rows_fetched, n_rows_resident,
        n_rows_retained, n_rows_retired, pool_ids_json.c_str(),
        bytes_staged, bytes_full_bank, host_submit_us, transfer_us, ready_latency_us,
        // Same <100.0 us convention the row-set executor already feeds to
        // add_layer_stats(..., rowset_last_ready_us < 100.0), so the trace
        // and the in-memory statistic agree by construction. It is a LATENCY
        // PROXY, not an observed consumption-time wait: the row-set path has
        // no explicit wait site to observe. Reading it as a measured
        // readiness gate would repeat the S1 defect it is meant to close.
        ready_latency_us < 100.0 ? "true" : "false",
        (long long) ggml_time_us());
}

void llama_layer_streaming::tel_rowset_refused(int il, int slot, uint32_t generation, const char * reason, const char * detail) {
    tel_write(tel_out,
        "{\"event\":\"rowset_refused\",\"run_id\":%llu,\"layer\":%d,\"slot\":%d,\"generation\":%u,"
        "\"reason\":\"%s\",\"detail\":\"%s\",\"phase\":%d,\"t_us\":%lld}",
        (unsigned long long) run_id, il, slot, generation, reason, detail, phase,
        (long long) ggml_time_us());
}

void llama_layer_streaming::tel_rowset_round(int64_t round, size_t bytes_staged, size_t bytes_full_bank,
                                             uint64_t n_fetched, uint64_t n_reused) {
    tel_write(tel_out,
        "{\"event\":\"rowset_round\",\"run_id\":%llu,\"round\":%lld,\"n_tokens\":1,\"bytes_staged\":%zu,"
        "\"bytes_full_bank\":%zu,\"n_rows_fetched\":%llu,\"n_rows_reused\":%llu,"
        "\"t_begin_us\":%lld,\"round_wall_us\":%lld,\"t_us\":%lld}",
        (unsigned long long) run_id, (long long) round, bytes_staged, bytes_full_bank,
        (unsigned long long) n_fetched, (unsigned long long) n_reused,
        (long long) rowset_round_t0_us,
        (long long) ((rowset_round_t0_us > 0) ? (ggml_time_us() - rowset_round_t0_us) : 0),
        (long long) ggml_time_us());
}

// ---------------------------------------------------------------------------
// T-GPLR-00: host-only GGUF inventory (no CUDA, no tensor allocation)
//
// Parses GGUF metadata with gguf_init_from_file(no_alloc=true, ctx=nullptr),
// which reads only the header and tensor-info section. Classifies every tensor
// into one byte class (static / per-layer / expert-bank / MTP), computes the
// Q8_0 (or requested) KV bytes and recurrent-state bytes from the actual
// architecture metadata, and compares against a host-recorded physical VRAM
// total. Missing required metadata or an unclassified tensor is an explicit
// error, never a guessed zero.
// ---------------------------------------------------------------------------

namespace {

enum llama_ls_class {
    LS_CLS_STATIC = 0,   // embeddings, output norm, output head
    LS_CLS_ATTENTION,    // per-trunk-layer attention weights
    LS_CLS_RECURRENT,    // per-trunk-layer SSM/delta-net weights
    LS_CLS_ROUTER,       // per-trunk-layer MoE router (gate_inp) weights
    LS_CLS_SHARED_EXPERT,// per-trunk-layer shared-expert weights
    LS_CLS_DENSE_FFN,    // per-trunk-layer dense MLP weights
    LS_CLS_LAYER_NORMS,  // per-trunk-layer attention/post norms
    LS_CLS_EXPERTS,      // per-trunk-layer routed-expert bank
    LS_CLS_MTP_NEXTN,    // nextn.* tensors of the MTP block
    LS_CLS_MTP_LAYER,    // non-nextn tensors of the MTP block
    LS_CLS_COUNT
};

struct llama_ls_meta {
    int      trunk_layers   = 0;
    int      total_blocks    = 0;
    int      nextn_layers   = 0;
    int      expert_count   = 0;
    int      expert_used    = 0;
    int      n_embd         = 0;
    int      n_head_kv      = 0;
    int      head_k         = 0;
    int      head_v         = 0;
    int      attn_interval  = 4;
    int      ssm_d_conv     = 0;
    int      ssm_d_state    = 0;
    int      ssm_d_inner    = 0;
    int      ssm_n_group    = 0;
    int64_t  context_length = 0;
    bool     has_context_length = false;
    // per-block recurrent flag (true = Gated DeltaNet layer), from the
    // explicit attention.recurrent_layers array when present, else the
    // full_attention_interval heuristic. Index is the block id.
    std::vector<uint8_t> is_recr;
};

struct llama_ls_acc {
    uint64_t bytes   = 0;
    uint64_t members = 0;
    // expert-bank member presence: bit0 gate, bit1 up, bit2 down, bit3 fused gate_up
    uint32_t flags   = 0;
};

// Read an integer-valued KV of any GGUF integer type. Returns false when the
// key is absent or is not a scalar integer (a wrong type is an explicit error
// at the call site, never an abort and never a guessed zero).
bool ls_key_int(const gguf_context * g, const char * key, uint64_t & out) {
    const int64_t kid = gguf_find_key(g, key);
    if (kid < 0) {
        return false;
    }
    switch (gguf_get_kv_type(g, kid)) {
        case GGUF_TYPE_UINT8:  out = gguf_get_val_u8(g, kid);   return true;
        case GGUF_TYPE_INT8:   out = (uint64_t) (int64_t) gguf_get_val_i8(g, kid);   return true;
        case GGUF_TYPE_UINT16: out = gguf_get_val_u16(g, kid);  return true;
        case GGUF_TYPE_INT16:  out = (uint64_t) (int64_t) gguf_get_val_i16(g, kid);  return true;
        case GGUF_TYPE_UINT32: out = gguf_get_val_u32(g, kid);  return true;
        case GGUF_TYPE_INT32:  out = (uint64_t) (int64_t) gguf_get_val_i32(g, kid);  return true;
        case GGUF_TYPE_UINT64: out = gguf_get_val_u64(g, kid);  return true;
        case GGUF_TYPE_INT64:  out = (uint64_t) gguf_get_val_i64(g, kid);            return true;
        case GGUF_TYPE_BOOL:   out = gguf_get_val_bool(g, kid) ? 1u : 0u; return true;
        default: return false;
    }
}

bool ls_key_u32(const gguf_context * g, const char * key, uint32_t & out) {
    uint64_t v = 0;
    if (!ls_key_int(g, key, v) || v > 0x7fffffffull) {
        return false;
    }
    out = (uint32_t) v;
    return true;
}

// File size without 32-bit ftell truncation (the acceptance artifacts are >2 GiB).
bool ls_file_size(const char * path, uint64_t & out) {
    FILE * fp = fopen(path, "rb");
    if (fp == nullptr) {
        return false;
    }
    bool ok = false;
#if defined(_WIN32)
    ok = _fseeki64(fp, 0, SEEK_END) == 0;
    if (ok) {
        const long long sz = _ftelli64(fp);
        ok = sz >= 0;
        if (ok) out = (uint64_t) sz;
    }
#else
    ok = fseeko(fp, 0, SEEK_END) == 0;
    if (ok) {
        const off_t sz = ftello(fp);
        ok = sz >= 0;
        if (ok) out = (uint64_t) sz;
    }
#endif
    fclose(fp);
    return ok;
}

std::string ls_json_str(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char) c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

} // anonymous namespace

std::string llama_ls_model_arch(const char * path) {
    if (path == nullptr || path[0] == '\0') {
        return "";
    }
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path, params);
    if (g == nullptr) {
        return "";
    }
    std::string arch;
    const int64_t kid = gguf_find_key(g, "general.architecture");
    if (kid >= 0 && gguf_get_kv_type(g, kid) == GGUF_TYPE_STRING) {
        arch = gguf_get_val_str(g, kid);
    }
    gguf_free(g);
    return arch;
}

llama_ls_inventory llama_ls_model_inventory(
        const char * path,
        int64_t      ctx_capacity,
        ggml_type    kv_type,
        uint64_t     physical_vram_bytes,
        const char * physical_vram_source) {

    llama_ls_inventory res;

    if (path == nullptr || path[0] == '\0') {
        res.error = "inspect path is empty";
        return res;
    }
    if (ctx_capacity <= 0) {
        res.error = "ctx_capacity must be > 0";
        return res;
    }

    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path, params);
    if (g == nullptr) {
        res.error = std::string("failed to parse GGUF metadata: ") + path;
        return res;
    }

    // RAII release
    struct gguf_guard {
        gguf_context * ctx;
        ~gguf_guard() { gguf_free(ctx); }
    } guard{ g };

    auto fail = [&](const std::string & msg) {
        res.error = msg;
        return res;
    };

    const int64_t arch_kid = gguf_find_key(g, "general.architecture");
    if (arch_kid < 0 || gguf_get_kv_type(g, arch_kid) != GGUF_TYPE_STRING) {
        return fail("missing required metadata: general.architecture");
    }
    const std::string arch = gguf_get_val_str(g, arch_kid);
    if (arch != "qwen35" && arch != "qwen35moe") {
        return fail("unsupported architecture for layer residency: " + arch);
    }
    const bool is_moe = (arch == "qwen35moe");

    llama_ls_meta m;
    {
        uint32_t v;
        if (!ls_key_u32(g, (arch + ".block_count").c_str(), v)) {
            return fail("missing required metadata: " + arch + ".block_count");
        }
        m.total_blocks = (int) v;
        if (!ls_key_u32(g, (arch + ".embedding_length").c_str(), v)) {
            return fail("missing required metadata: " + arch + ".embedding_length");
        }
        m.n_embd = (int) v;
        if (!ls_key_u32(g, (arch + ".attention.head_count_kv").c_str(), v)) {
            return fail("missing required metadata: " + arch + ".attention.head_count_kv");
        }
        m.n_head_kv = (int) v;
        if (!ls_key_u32(g, (arch + ".attention.key_length").c_str(), v)) {
            return fail("missing required metadata: " + arch + ".attention.key_length");
        }
        m.head_k = (int) v;
        if (!ls_key_u32(g, (arch + ".attention.value_length").c_str(), v)) {
            return fail("missing required metadata: " + arch + ".attention.value_length");
        }
        m.head_v = (int) v;
        if (!ls_key_u32(g, (arch + ".ssm.conv_kernel").c_str(), v)) {
            return fail("missing required metadata: " + arch + ".ssm.conv_kernel");
        }
        m.ssm_d_conv = (int) v;
        if (!ls_key_u32(g, (arch + ".ssm.state_size").c_str(), v)) {
            return fail("missing required metadata: " + arch + ".ssm.state_size");
        }
        m.ssm_d_state = (int) v;
        if (!ls_key_u32(g, (arch + ".ssm.inner_size").c_str(), v)) {
            return fail("missing required metadata: " + arch + ".ssm.inner_size");
        }
        m.ssm_d_inner = (int) v;
        if (!ls_key_u32(g, (arch + ".ssm.group_count").c_str(), v)) {
            return fail("missing required metadata: " + arch + ".ssm.group_count");
        }
        m.ssm_n_group = (int) v;
        ls_key_u32(g, (arch + ".full_attention_interval").c_str(), v);
        m.attn_interval = v ? (int) v : 4;
        {
            uint64_t cl = 0;
            if (ls_key_int(g, (arch + ".context_length").c_str(), cl)) {
                m.context_length = (int64_t) cl;
                m.has_context_length = true;
            }
        }
        if (is_moe) {
            if (!ls_key_u32(g, (arch + ".expert_count").c_str(), v)) {
                return fail("missing required metadata: " + arch + ".expert_count");
            }
            m.expert_count = (int) v;
            if (!ls_key_u32(g, (arch + ".expert_used_count").c_str(), v)) {
                return fail("missing required metadata: " + arch + ".expert_used_count");
            }
            m.expert_used = (int) v;
            if (m.expert_count <= 0 || m.expert_used <= 0 || m.expert_used > m.expert_count) {
                return fail("invalid expert metadata: expert_used_count must be in 1..expert_count");
            }
        }
        uint32_t nextn = 0;
        ls_key_u32(g, (arch + ".nextn_predict_layers").c_str(), nextn);
        m.nextn_layers = (int) nextn;
        if (m.nextn_layers < 0 || m.nextn_layers >= m.total_blocks) {
            return fail("invalid " + arch + ".nextn_predict_layers");
        }
    }
    m.trunk_layers = m.total_blocks - m.nextn_layers;
    if (m.trunk_layers <= 0 || m.n_embd <= 0 || m.n_head_kv <= 0) {
        return fail("invalid architecture metadata");
    }
    if (m.head_k <= 0 || m.head_v <= 0 || m.attn_interval <= 0) {
        return fail("invalid head/interval metadata");
    }

    // recurrent-layer mask: explicit bool array wins (matches the controlling
    // loader, which reads LLM_KV_ATTENTION_RECURRENT_LAYERS first); otherwise
    // the full_attention_interval heuristic. Index = block id.
    m.is_recr.assign(m.total_blocks, 0);
    {
        const std::string rkey = arch + ".attention.recurrent_layers";
        const int64_t kid = gguf_find_key(g, rkey.c_str());
        if (kid >= 0 && gguf_get_kv_type(g, kid) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(g, kid) == GGUF_TYPE_BOOL) {
            const size_t n = gguf_get_arr_n(g, kid);
            if (n < (size_t) m.total_blocks) {
                return fail(rkey + " shorter than block_count");
            }
            const uint8_t * data = (const uint8_t *) gguf_get_arr_data(g, kid);
            for (size_t i = 0; i < (size_t) m.total_blocks; ++i) {
                m.is_recr[i] = data[i] ? 1 : 0;
            }
        } else {
            for (int il = 0; il < m.total_blocks; ++il) {
                m.is_recr[il] = (il < m.trunk_layers && (il + 1) % m.attn_interval != 0) ? 1 : 0;
            }
        }
    }

    std::string general_name;
    {
        const int64_t kid = gguf_find_key(g, "general.name");
        if (kid >= 0 && gguf_get_kv_type(g, kid) == GGUF_TYPE_STRING) {
            general_name = gguf_get_val_str(g, kid);
        }
    }

    // --- classify every tensor ---------------------------------------------

    auto starts = [](const std::string & s, const char * p) {
        return s.compare(0, strlen(p), p) == 0;
    };

    std::vector<llama_ls_acc> experts(m.trunk_layers);
    std::vector<llama_ls_acc> layer(m.trunk_layers);
    llama_ls_acc cls[LS_CLS_COUNT];
    std::map<std::string, std::pair<uint64_t, uint64_t>> types; // type -> {count, bytes}
    uint64_t total_data = 0;

    const int64_t n_tensors = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char *   name = gguf_get_tensor_name(g, i);
        const size_t   size = gguf_get_tensor_size(g, i);
        const ggml_type typ = gguf_get_tensor_type(g, i);
        const std::string n(name);

        types[ggml_type_name(typ)].first  += 1;
        types[ggml_type_name(typ)].second += size;
        total_data += size;

        int il = -1;
        if (starts(n, "blk.")) {
            if (sscanf(n.c_str(), "blk.%d.", &il) != 1 || il < 0) {
                return fail(std::string("malformed block tensor name: ") + n);
            }
            const bool nextn = n.find(".nextn.") != std::string::npos;
            if (il >= m.trunk_layers) {
                if (il >= m.total_blocks) {
                    return fail(std::string("tensor layer out of range: ") + n);
                }
                cls[nextn ? LS_CLS_MTP_NEXTN : LS_CLS_MTP_LAYER].bytes   += size;
                cls[nextn ? LS_CLS_MTP_NEXTN : LS_CLS_MTP_LAYER].members += 1;
                continue;
            }
            if (nextn) {
                return fail(std::string("nextn tensor in a trunk layer: ") + n);
            }
            const std::string rest = n.substr(n.find('.', 4) + 1); // after "blk.N."
            llama_ls_acc * dst;
            llama_ls_class sub;
            uint32_t flag = 0;
            if (is_moe && rest == "ffn_gate_exps.weight") {
                dst = &experts[il]; sub = LS_CLS_EXPERTS; flag = 1u;
            } else if (is_moe && rest == "ffn_up_exps.weight") {
                dst = &experts[il]; sub = LS_CLS_EXPERTS; flag = 2u;
            } else if (is_moe && rest == "ffn_down_exps.weight") {
                dst = &experts[il]; sub = LS_CLS_EXPERTS; flag = 4u;
            } else if (is_moe && rest == "ffn_gate_up_exps.weight") {
                // fused gate+up expert tensor supported by this tree
                dst = &experts[il]; sub = LS_CLS_EXPERTS; flag = 8u;
            } else if (rest == "attn_norm.weight" || rest == "post_attention_norm.weight") {
                dst = &layer[il];
                sub = LS_CLS_LAYER_NORMS;
            } else if (rest == "ffn_gate_inp.weight") {
                dst = &layer[il];
                sub = LS_CLS_ROUTER;
            } else if (rest.find("shexp") != std::string::npos) {
                dst = &layer[il];
                sub = LS_CLS_SHARED_EXPERT;
            } else if (rest.find("ssm") == 0) {
                dst = &layer[il];
                sub = LS_CLS_RECURRENT;
            } else if (rest == "attn_qkv.weight" || rest == "attn_gate.weight") {
                // the fused qkv/gate tensors exist for recurrent AND (in this
                // tree) full-attention layers: classify by the layer's own type
                dst = &layer[il];
                sub = m.is_recr[il] ? LS_CLS_RECURRENT : LS_CLS_ATTENTION;
            } else if (rest.find("attn_") == 0) {
                dst = &layer[il];
                sub = LS_CLS_ATTENTION;
            } else if (rest.find("ffn_") == 0) {
                dst = &layer[il];
                sub = LS_CLS_DENSE_FFN;
            } else {
                return fail(std::string("unclassified tensor: ") + n);
            }
            cls[sub].bytes   += size;
            cls[sub].members += 1;
            dst->bytes   += size;
            dst->members += 1;
            dst->flags   |= flag;
            continue;
        }
        if (n == "token_embd.weight" || n == "output_norm.weight" || n == "output.weight") {
            cls[LS_CLS_STATIC].bytes   += size;
            cls[LS_CLS_STATIC].members += 1;
            continue;
        }
        return fail(std::string("unclassified tensor: ") + n);
    }

    // --- completeness checks -------------------------------------------------

    if (is_moe) {
        // every trunk layer must carry a complete routed-expert bank: either
        // separate gate+up+down (flags 1|2|4) or fused gate_up+down (8|4)
        for (int il = 0; il < m.trunk_layers; ++il) {
            const bool separate = (experts[il].flags & 7u) == 7u;
            const bool fused    = (experts[il].flags & 12u) == 12u && (experts[il].flags & 3u) == 0u;
            if ((!separate && !fused) || experts[il].bytes == 0) {
                return fail("incomplete routed-expert bank in layer " + std::to_string(il));
            }
        }
    }
    for (int il = 0; il < m.trunk_layers; ++il) {
        if (layer[il].members == 0) {
            return fail("empty per-layer package in layer " + std::to_string(il));
        }
    }

    // --- runtime byte classes ------------------------------------------------

    const int64_t kv_size    = GGML_PAD((int64_t) ctx_capacity, 256);
    int64_t n_attn_layers    = 0;
    int64_t n_recr_layers    = 0;
    for (int il = 0; il < m.trunk_layers; ++il) {
        n_attn_layers += !m.is_recr[il];
        n_recr_layers +=  m.is_recr[il];
    }
    const int64_t n_embd_k_gqa = (int64_t) m.n_head_kv * m.head_k;
    const int64_t n_embd_v_gqa = (int64_t) m.n_head_kv * m.head_v;
    const uint64_t kv_bytes = (uint64_t) n_attn_layers * (uint64_t) kv_size *
        ((uint64_t) ggml_row_size(kv_type, n_embd_k_gqa) +
         (uint64_t) ggml_row_size(kv_type, n_embd_v_gqa));

    // Gated DeltaNet state, matching llama_hparams::n_embd_r()/n_embd_s() and
    // llama_memory_recurrent: one F32 row per (recurrent layer, sequence cell).
    // mem_size = max(1, n_seq_max) = 1 row; MTP snapshots excluded (MTP off).
    const int64_t n_embd_r = (m.ssm_d_conv > 0 ? m.ssm_d_conv - 1 : 0) *
                             (m.ssm_d_inner + 2 * m.ssm_n_group * m.ssm_d_state);
    const int64_t n_embd_s = (int64_t) m.ssm_d_state * m.ssm_d_inner;
    const uint64_t state_bytes = (uint64_t) n_recr_layers *
        ((uint64_t) ggml_row_size(GGML_TYPE_F32, n_embd_r) +
         (uint64_t) ggml_row_size(GGML_TYPE_F32, n_embd_s));

    // --- package candidates ----------------------------------------------------

    uint64_t moe_largest = 0; int moe_largest_layer = -1;
    uint64_t dense_largest = 0; int dense_largest_layer = -1;
    for (int il = 0; il < m.trunk_layers; ++il) {
        const uint64_t dense = layer[il].bytes + experts[il].bytes;
        if (experts[il].bytes > moe_largest) { moe_largest = experts[il].bytes; moe_largest_layer = il; }
        if (dense > dense_largest)            { dense_largest = dense;          dense_largest_layer = il; }
    }

    const uint64_t full_plus_runtime = total_data + kv_bytes + state_bytes;
    // per-trunk-layer non-expert total = every per-layer sub-class summed
    uint64_t per_layer_total = 0;
    for (int c = LS_CLS_ATTENTION; c <= LS_CLS_LAYER_NORMS; ++c) {
        per_layer_total += cls[c].bytes;
    }

    const uint64_t bounded1 = cls[LS_CLS_STATIC].bytes + per_layer_total +
                              moe_largest + kv_bytes + state_bytes;
    const uint64_t bounded2 = bounded1 + moe_largest;

    // --- deterministic JSON ------------------------------------------------------

    std::string j;
    auto add = [&](const std::string & s) { j += s; };
    auto addf = [&](const char * fmt, ...) {
        char small[512];
        va_list args; va_start(args, fmt);
        const int n = vsnprintf(small, sizeof(small), fmt, args);
        va_end(args);
        if (n >= 0 && (size_t) n < sizeof(small)) {
            j += small;
            return;
        }
        // grow rather than truncate: truncated output would emit invalid JSON
        std::vector<char> big((size_t) (n < 0 ? 4096 : n) + 1);
        va_start(args, fmt);
        vsnprintf(big.data(), big.size(), fmt, args);
        va_end(args);
        j += big.data();
    };

    add("{");
    addf("\"mode\":\"inspect\",\"path\":\"%s\"", ls_json_str(path).c_str());
    {
        uint64_t fsize = 0;
        if (ls_file_size(path, fsize)) {
            addf(",\"file_bytes\":%llu", (unsigned long long) fsize);
        } else {
            add(",\"file_bytes\":null");
        }
    }
    addf(",\"gguf_version\":%u", gguf_get_version(g));
    addf(",\"alignment\":%zu", gguf_get_alignment(g));
    addf(",\"data_offset\":%zu", gguf_get_data_offset(g));
    addf(",\"architecture\":\"%s\"", ls_json_str(arch).c_str());
    addf(",\"model_name\":\"%s\"", ls_json_str(general_name).c_str());
    addf(",\"n_tensors\":%lld", (long long) n_tensors);
    addf(",\"trunk_layers\":%d,\"total_block_count\":%d,\"nextn_predict_layers\":%d",
         m.trunk_layers, m.total_blocks, m.nextn_layers);
    addf(",\"expert_count\":%d,\"expert_used_count\":%d", m.expert_count, m.expert_used);
    addf(",\"embedding_length\":%d,\"head_count_kv\":%d,\"key_length\":%d,\"value_length\":%d",
         m.n_embd, m.n_head_kv, m.head_k, m.head_v);
    addf(",\"full_attention_interval\":%d,\"attn_layers\":%lld,\"recurrent_layers\":%lld",
         m.attn_interval, (long long) n_attn_layers, (long long) n_recr_layers);
    if (m.has_context_length) {
        addf(",\"context_length\":%lld", (long long) m.context_length);
    } else {
        add(",\"context_length\":null");
    }
    addf(",\"ssm\":{\"conv_kernel\":%d,\"state_size\":%d,\"inner_size\":%d,\"group_count\":%d}",
         m.ssm_d_conv, m.ssm_d_state, m.ssm_d_inner, m.ssm_n_group);

    add(",\"tensor_types\":{");
    {
        bool first = true;
        for (const auto & t : types) {
            addf("%s\"%s\":[%llu,%llu]", first ? "" : ",", ls_json_str(t.first).c_str(),
                 (unsigned long long) t.second.first, (unsigned long long) t.second.second);
            first = false;
        }
    }
    add("}");

    addf(",\"total_data_bytes\":%llu", (unsigned long long) total_data);
    add(",\"byte_classes\":{");
    addf("\"static\":%llu",  (unsigned long long) cls[LS_CLS_STATIC].bytes);
    addf(",\"per_layer\":%llu", (unsigned long long) per_layer_total);
    addf(",\"attention\":%llu", (unsigned long long) cls[LS_CLS_ATTENTION].bytes);
    addf(",\"recurrent\":%llu", (unsigned long long) cls[LS_CLS_RECURRENT].bytes);
    addf(",\"router\":%llu", (unsigned long long) cls[LS_CLS_ROUTER].bytes);
    addf(",\"shared_expert\":%llu", (unsigned long long) cls[LS_CLS_SHARED_EXPERT].bytes);
    addf(",\"dense_ffn\":%llu", (unsigned long long) cls[LS_CLS_DENSE_FFN].bytes);
    addf(",\"layer_norms\":%llu", (unsigned long long) cls[LS_CLS_LAYER_NORMS].bytes);
    addf(",\"routed_experts\":%llu", (unsigned long long) cls[LS_CLS_EXPERTS].bytes);
    addf(",\"mtp_layer\":%llu", (unsigned long long) cls[LS_CLS_MTP_LAYER].bytes);
    addf(",\"mtp_nextn\":%llu", (unsigned long long) cls[LS_CLS_MTP_NEXTN].bytes);
    add("}");

    add(",\"packages\":{\"moe_expert_bank\":[");
    for (int il = 0; il < m.trunk_layers; ++il) {
        addf("%s{\"layer\":%d,\"bytes\":%llu,\"members\":%llu}", il ? "," : "", il,
             (unsigned long long) experts[il].bytes, (unsigned long long) experts[il].members);
    }
    add("]");
    addf(",\"moe_largest_bytes\":%llu,\"moe_largest_layer\":%d", moe_largest, moe_largest_layer);
    add(",\"dense_layer\":[");
    for (int il = 0; il < m.trunk_layers; ++il) {
        const uint64_t dense = layer[il].bytes + experts[il].bytes;
        addf("%s{\"layer\":%d,\"bytes\":%llu,\"members\":%llu}", il ? "," : "", il,
             (unsigned long long) dense, (unsigned long long) (layer[il].members + experts[il].members));
    }
    add("]");
    addf(",\"dense_largest_bytes\":%llu,\"dense_largest_layer\":%d", dense_largest, dense_largest_layer);
    add("}");

    add(",\"runtime\":{");
    addf("\"ctx_capacity\":%lld,\"kv_size_padded\":%lld,\"kv_type\":\"%s\"",
         (long long) ctx_capacity, (long long) kv_size, ggml_type_name(kv_type));
    addf(",\"kv_bytes\":%llu", (unsigned long long) kv_bytes);
    addf(",\"kv_layers\":%lld", (long long) n_attn_layers);
    addf(",\"n_embd_k_gqa\":%lld,\"n_embd_v_gqa\":%lld", (long long) n_embd_k_gqa, (long long) n_embd_v_gqa);
    addf(",\"recurrent_state_bytes\":%llu", (unsigned long long) state_bytes);
    addf(",\"state_rows\":%d", 1); // mem_size = max(1, n_seq_max); MTP snapshots excluded
    add(",\"note\":\"graph/input, activation, workspace and CUDA-context bytes are unknown until runtime preflight\"");
    add("}");

    add(",\"capacity\":{");
    if (physical_vram_bytes > 0) {
        addf("\"physical_vram_bytes\":%llu", (unsigned long long) physical_vram_bytes);
        addf(",\"physical_vram_source\":\"%s\"", ls_json_str(physical_vram_source ? physical_vram_source : "unknown").c_str());
    } else {
        add("\"physical_vram_bytes\":null,\"physical_vram_source\":\"unavailable\"");
    }
    addf(",\"full_model_plus_runtime\":%llu", (unsigned long long) full_plus_runtime);
    addf(",\"bounded_one_slot\":%llu", (unsigned long long) bounded1);
    addf(",\"bounded_two_slot\":%llu", (unsigned long long) bounded2);
    if (physical_vram_bytes > 0) {
        const bool full_fits = full_plus_runtime <= physical_vram_bytes;
        addf(",\"full_fits_physical\":%s", full_fits ? "true" : "false");
        addf(",\"require_sub_full_budget\":%s", full_fits ? "true" : "false");
        addf(",\"bounded_one_fits_physical\":%s", bounded1 <= physical_vram_bytes ? "true" : "false");
        addf(",\"bounded_two_fits_physical\":%s", bounded2 <= physical_vram_bytes ? "true" : "false");
    } else {
        add(",\"full_fits_physical\":null,\"require_sub_full_budget\":null");
        add(",\"bounded_one_fits_physical\":null,\"bounded_two_fits_physical\":null");
    }
    add("}");

    add("}\n");

    res.ok   = true;
    res.json = std::move(j);
    return res;
}

#endif // LLAMA_LAYER_STREAMING
