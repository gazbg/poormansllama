#pragma once

// T-GPLR-61 step A -- host-only POLICY AND SHAPE for the `moe-row-split`
// placement class (docs/layer-residency-sdd.md section 4).
//
// WHAT THIS FILE IS. Pure decision functions: no CUDA handles, no backend
// buffers, no model load, no I/O. It decides, at demand time and per
// (layer, token), which demanded expert rows execute on the device and which
// execute on the host, and it emits two CONSTANT-WIDTH sub-op descriptors.
//
// WHY IT IS SEPARATE FROM llama-layer-streaming.h. Step A is host-only and must
// be provable without a GPU. Keeping it in its own header also means this file
// adds no members to any class, so it cannot silently change the layout of
// anything the linker already laid out.
//
// THE RULE THIS ENCODES, and why it is not a tunable threshold. A demanded row
// that is not device-resident computes on the host. The measured constants are
// host ~36.5 us/expert against ~171 us to fetch the same 1,769,472 B row at the
// best measured end-to-end feed of 10.35 GB/s -- the host is 3.4-6.7x cheaper at
// every k, with a margin constant in k, so there is no break-even point at which
// fetching wins. The estimator stays explicit and host-side precisely so it can
// be re-derived when the host or the model changes; a named policy that cannot
// be evaluated REFUSES rather than defaulting to something permissive.
//
// WHAT IS NOT PROVEN HERE. Nothing in this file has executed on hardware. It
// proves the partition and the refusal contract. It does not prove that the
// mechanism reaches it, that a host sub-op is correct, or that it is faster.

#include <cstddef>
#include <cstdint>
#include <vector>

// T-GPLR-82: the namespace is llama_ls_policy, NOT llama_layer_streaming.
// The latter is the CLASS in llama-layer-streaming.h, and C++ forbids a
// namespace and a class of the same name in one scope -- which only
// surfaced when a translation unit started including BOTH headers (the
// host leg in qwen35moe.cpp does). Renaming the namespace is the fix; the
// ls_* names themselves are unchanged.
namespace llama_ls_policy {

// Residency classification at demand time (SDD: "Trigger").
//
// READY   -- device-resident at this instant. Always executes on the device:
//            there is no transfer left to save, so the gate is never consulted.
// PENDING -- issued and in flight, copy not yet complete. The only gate
//            candidate; whether it goes to the host is the gate's decision.
// ABSENT  -- never issued. Per the SDD this is a staging defect, not a slow
//            path, so it is a loud refusal and never a silent host fallback.
enum class ls_row_state : int {
    READY   = 0,
    PENDING = 1,
    ABSENT  = 2,
};

// Named gate policies. UNKNOWN is a refusal, not a default: a run that cannot
// name the policy it is evaluating must not proceed on a guessed one.
enum class ls_gate_policy : int {
    UNKNOWN    = 0,  // refuse
    HOST_ALWAYS = 1, // measured-constant rule: not resident -> host
};

// Measured constants the gate evaluates. All four are telemetry inputs so a
// gate decision can be re-derived after the fact from the run that made it.
struct ls_gate_constants {
    double host_us_per_expert = 0.0;   // measured marginal host cost, one expert
    uint64_t row_bytes        = 0;     // bytes in one expert row
    double link_bytes_per_s   = 0.0;   // measured END-TO-END feed, not wire time
    ls_gate_policy policy     = ls_gate_policy::UNKNOWN;
};

struct ls_gate_verdict {
    bool        evaluable     = false;
    bool        open          = false;
    const char * refuse_reason = nullptr;
    double      host_us       = 0.0;
    double      wait_us       = 0.0;
};

// One entry of a constant-width sub-op. A pad slot carries no expert id, runs on
// no backend, and contributes exactly zero to the reduction (its router weight
// is zero), which is what lets every (layer, token) produce the same graph
// shapes and keeps the persistent graph set from rebuilding on a subset.
struct ls_split_slot {
    int32_t     expert_id = -1;
    ls_row_state state    = ls_row_state::ABSENT;
    bool        is_pad    = false;
    bool        on_host   = false;
};

// Two constant-width descriptors plus the counters the SDD's telemetry set names.
// Invariant AC1: experts_device + experts_host == experts_total whenever ok.
struct ls_split_plan {
    bool        ok            = false;
    const char * refuse_reason = nullptr;

    uint32_t top_k = 0;
    std::vector<ls_split_slot> device; // always exactly top_k entries when ok
    std::vector<ls_split_slot> host;   // always exactly top_k entries when ok

    uint32_t  experts_total        = 0;
    uint32_t  experts_device       = 0;
    uint32_t  experts_host         = 0;
    // rows_gate_cpu + rows_gate_gpu == experts_total: the gate's disposition of
    // every demanded row, READY rows included. READY always goes to the device
    // without consulting the gate, but it is still a row the gate DISPOSED of,
    // so counting only the PENDING ones would make the pair not sum.
    uint32_t  rows_gate_cpu        = 0;
    uint32_t  rows_gate_gpu        = 0;
    uint64_t staged_bytes_avoided  = 0;
};

// Classify one demanded row from the two facts the executor knows. `issued` is
// whether a copy was ever submitted; `copy_complete` is whether the CUDA event
// has signalled. issued && !complete is PENDING; !issued is ABSENT regardless of
// copy_complete, because a row with no outstanding copy cannot become ready.
static inline ls_row_state ls_classify_row(bool issued, bool copy_complete) {
    if (!issued) {
        return ls_row_state::ABSENT;
    }
    return copy_complete ? ls_row_state::READY : ls_row_state::PENDING;
}

// Evaluate the named cost gate. Never throws and never defaults: an unevaluable
// policy, or one whose constants cannot produce a comparison, returns a verdict
// with evaluable == false and a reason, and the caller refuses the run.
static inline ls_gate_verdict ls_evaluate_gate(const ls_gate_constants & c) {
    ls_gate_verdict v;
    if (c.policy == ls_gate_policy::UNKNOWN) {
        v.refuse_reason = "unknown gate policy";
        return v;
    }
    if (!(c.host_us_per_expert > 0.0) || !(c.link_bytes_per_s > 0.0) || c.row_bytes == 0) {
        v.refuse_reason = "gate constants are not evaluable";
        return v;
    }
    v.evaluable = true;
    v.host_us    = c.host_us_per_expert;
    v.wait_us    = (double) c.row_bytes / c.link_bytes_per_s * 1e6;
    v.open       = v.host_us < v.wait_us;
    return v;
}

// Partition the demanded rows and emit both constant-width descriptors.
//
// demanded states the router's EXACT executed id set, in order. It is
// authoritative: this function never drops, substitutes, delays or reorders an
// expert. top_k is the padded width of both sub-ops.
//
// Refusal contract:
//   - an ABSENT demanded row refuses the plan (staging defect, not a slow path)
//   - a gate verdict that is not evaluable refuses the plan
// A CLOSED gate is NOT a refusal: PENDING rows then return to the ordinary wait
// path, which means they sit in the device sub-op exactly as they would with the
// split disabled. That is the property AC1 pins.
static inline ls_split_plan ls_plan_split(const std::vector<ls_row_state> & demanded,
                                          const ls_gate_verdict &        gate,
                                          uint32_t                       top_k,
                                          uint64_t                       row_bytes) {
    ls_split_plan p;
    p.top_k         = top_k;
    p.experts_total = (uint32_t) demanded.size();

    if (!gate.evaluable) {
        p.refuse_reason = gate.refuse_reason ? gate.refuse_reason : "gate not evaluable";
        return p;
    }
    if (top_k == 0) {
        p.refuse_reason = "top_k must be positive to pad a constant-width sub-op";
        return p;
    }

    for (ls_row_state st : demanded) {
        if (st == ls_row_state::ABSENT) {
            p.refuse_reason = "demanded row was never issued";
            return p;
        }
    }

    p.device.assign(top_k, ls_split_slot{});
    p.host.assign(top_k, ls_split_slot{});

    for (size_t i = 0; i < demanded.size(); ++i) {
        const ls_row_state st = demanded[i];
        // READY always runs on the device: nothing left to save. PENDING goes to
        // the host only when the gate is open; otherwise it waits, unchanged.
        const bool to_host = (st == ls_row_state::PENDING) && gate.open;

        // Both sub-ops are indexed by the SAME demanded position, and both
        // carry the expert id. A slot that is not host-bound is selected out by
        // a ZERO ROUTER WEIGHT rather than by having no id: the SDD's padded
        // shape requires every (token, layer) to emit identical widths, and the
        // two partials only sum correctly if each is a full-width term with its
        // own weight. An id of -1 here would make the two sub-ops
        // differently-shaped instead of differently-weighted.
        ls_split_slot d;
        d.expert_id = (int32_t) i;
        d.state     = st;
        d.on_host   = false;
        p.device[i] = d;

        ls_split_slot h;
        h.expert_id = (int32_t) i;
        h.state     = st;
        h.on_host   = to_host;
        p.host[i]   = h;

        if (to_host) {
            p.experts_host++;
            p.rows_gate_cpu++;
            p.staged_bytes_avoided += row_bytes;
        } else {
            p.experts_device++;
            p.rows_gate_gpu++;
        }
    }

    // Pad the tail of both sub-ops to exactly top_k with zero-weight slots.
    for (size_t i = demanded.size(); i < top_k; ++i) {
        ls_split_slot pad;
        pad.expert_id = -1;
        pad.state     = ls_row_state::READY;
        pad.is_pad    = true;
        pad.on_host   = false;
        p.device[i]   = pad;
        p.host[i]     = pad;
    }

    p.ok = true;
    return p;
}

// ---------------------------------------------------------------------------
// T-GPLR-82 S2: the rewire guard the SDD requires for the moe-row-split class.
//
// The SDD is explicit that this class is permitted to place an expert weight at
// its DECLARED host placement and is forbidden from rewiring any weight source,
// and that "the guard for this class must assert specifically that no *weight*
// source on this path is ever rewired by the scheduler". The device leg's
// weights being device-placed is the row-set mechanism working, not a rewire,
// so the predicate is scoped to the HOST leg only: a weight that the plan
// assigned to the host leg must still BE host-placed when the graph is built.
//
// Placement facts arrive as booleans from the caller, which owns the ggml
// backend types, so this header stays free of backend headers and the whole
// guard is host-testable without a device.
struct ls_rewire_observation {
    bool on_host_leg    = false; // this weight was assigned to the host sub-op
    bool weight_is_host = false; // its declared placement is still host
    // T-GPLR-82 S2, the two execution-shape facts the plan names. Both are
    // refusal conditions, not warnings, and both default false so a caller that
    // knows nothing cannot fail a run by omission.
    //
    // min_batch_override_live: the T-GPLR-16 MIN_BATCH=1 route was reverted out
    // of the tree on 2026-09-22, so no such override exists to be live. This
    // field is the tripwire that says so at the call site: the moment such a
    // route is reintroduced, whoever reintroduces it has to touch the host-leg
    // construction, and that is where the field goes true.
    //
    // weight_in_compute_staging_buffer: the weight's buffer carries
    // GGML_BACKEND_BUFFER_USAGE_COMPUTE. That usage is what the scheduler's
    // compute_splits path creates for staging, so a host-leg weight living in
    // one IS the forbidden "op-offload path that rewires a weight source to a
    // compute-usage staging buffer", independent of where it was declared.
    // Both usages are host-placed, so weight_is_host alone cannot tell them
    // apart -- this field is what makes the placement check sharp.
    bool min_batch_override_live          = false;
    bool weight_in_compute_staging_buffer = false;
};

struct ls_rewire_verdict {
    bool         ok            = true;
    const char  * refuse_reason = nullptr;
    uint32_t     host_leg_weights = 0;
    uint32_t     rewired          = 0;
    uint32_t     staging_weights   = 0;
    uint32_t     min_batch_live    = 0;
};

// A host-leg weight that is not host-placed is exactly the rewire the SDD
// forbids. Refuse; never repair, never clamp, never continue.
static inline ls_rewire_verdict ls_check_no_rewire(
        const std::vector<ls_rewire_observation> & obs) {
    ls_rewire_verdict v;
    for (size_t i = 0; i < obs.size(); ++i) {
        if (!obs[i].on_host_leg) {
            continue;
        }
        v.host_leg_weights++;
        if (!obs[i].weight_is_host) {
            v.rewired++;
        }
    }
    for (size_t i = 0; i < obs.size(); ++i) {
        if (!obs[i].on_host_leg) {
            continue;
        }
        if (obs[i].weight_in_compute_staging_buffer) {
            v.staging_weights++;
        }
        if (obs[i].min_batch_override_live) {
            v.min_batch_live++;
        }
    }
    // Order is the order the SDD names them in: a weight that is not where it
    // was declared is the rewire itself, so it is reported before the shapes
    // that would cause one.
    if (v.rewired > 0) {
        v.ok            = false;
        v.refuse_reason = "host-leg expert weight is not host-placed: a weight "
                          "source was rewired (SDD section 4 forbids this)";
    } else if (v.staging_weights > 0) {
        v.ok            = false;
        v.refuse_reason = "host-leg expert weight lives in a COMPUTE-usage staging "
                          "buffer: the scheduler row-copy would rewire it (SDD "
                          "section 4 forbids this)";
    } else if (v.min_batch_live > 0) {
        v.ok            = false;
        v.refuse_reason = "MIN_BATCH row-copy override is live on the host leg: that "
                          "is the T-GPLR-16 deadlock shape (SDD section 4)";
    }
    return v;
}


// ---------------------------------------------------------------------------
// T-GPLR-82 S3/S4 (2026-10-10): IS THE HOST LEG IN PLAY AT ALL?
//
// WHY THIS PREDICATE EXISTS, AND WHY IT IS NOT COSMETIC. The split graph fixes
// its widths AT BUILD: dev_rows = top_k - host_w plus a host leg exactly host_w
// wide (qwen35moe.cpp, the `host_rows > 0` branch). The host leg's id input is
// filled UNCONDITIONALLY from the tail of the row (the hsel memcpy in
// llama-layer-streaming.cpp), so once that graph exists, host_w rows per
// (layer, token) execute on the CPU whether or not the plan admitted any. The
// plan cannot admit a single row until `experts_host_budget_us > 0` (plus the
// arm gates below); at the budget's default of 0 no row is ever admitted.
//
// MEASURED CONSEQUENCE (2026-10-10, session gplr82-ac1, exe 66b4eb9e). Arm A2d
// ran `--experts-host 2` with the budget at its default: host share 0.000000 and
// sum(experts_device) == sum(experts_total) == 163,840 -- the ledger says every
// row ran on the device -- yet the FFN graph was MIXED (`ffn_dev = 2` on all
// 20,480 records, against 1 for the control) and the run diverged from the
// control at TG step 1 with all 512 logit readouts FINITE. The split was out of
// play and changed the arithmetic anyway. This predicate restores the
// closed-gate property the executor documents: when the mechanism cannot admit a
// row, the UNSPLIT graph must be the one built.
//
// It mirrors the executor's own admission predicate (`s3`, computed in
// load_row_set) minus `per_token != nullptr`: that argument is a property of the
// call site rather than of the run, so it cannot be known at graph build. Every
// other conjunct is a run-level input fixed before any build happens.
//
// FAILING-FIRST BASELINE, RECORDED. With the predicate transcribed as the
// shipped build condition (`host_w > 0` alone), test-layer-streaming-expert-split
// exited 1 with `AC1 FAIL (6 failures)`: the zero-budget case and all five arm
// gates were asserted and all six failed, while the two "in play" controls
// passed. The conjuncts below are the fix; the same test then exits 0.
struct ls_split_enable {
    int    host_w    = 0;      // --experts-host N
    double budget_us = 0.0;    // --experts-host-budget B (0 admits nothing)
    bool   recon     = false;  // row-set recon arm
    bool   spec_on   = false;  // predictor arm
    int    spec_mode = 0;
    int    spec_arm  = -1;
    int    spec_pool = 0;
};

static inline bool ls_split_in_play(const ls_split_enable & e) {
    // Every conjunct below is one the executor's own admission predicate
    // (`s3`, load_row_set) also applies. Omitting any of them builds a shape
    // that commits host_w rows per (layer, token) to the CPU leg while the plan
    // admits none -- the measured A2d corruption.
    return e.host_w > 0 && e.budget_us > 0.0 &&
           !e.recon && !e.spec_on && e.spec_mode == 0 &&
           e.spec_arm < 0 && e.spec_pool == 0;
}

} // namespace llama_ls_policy
