// Host-only regression test for T-GPLR-61 step A: the `moe-row-split` policy
// and shape (AC1).
//
// WHY THIS EXISTS. The `moe-row-split` class says a demanded expert row that is
// not device-resident should compute on the host instead of waiting for a link
// fetch that the measurement prices at ~171 us against ~36.5 us to compute the
// row where it already lies. Step A is the decision half of that: residency
// classification, the named cost gate, the fixed-width padding plan, and the
// two sub-op descriptors. It is pure host-side code, so it is provable with no
// GPU, no model load and no CUDA handle.
//
// NO GPU, NO MODEL. Everything asserted here runs through the shipped policy
// symbols -- ls_classify_row, ls_evaluate_gate and ls_plan_split. Nothing is
// re-spelled: a copy of the policy would prove nothing about the code the engine
// will call in step B.
//
// WHAT IS AND IS NOT PROVEN. This proves the PARTITION and the REFUSAL
// CONTRACT: AC1's seven fixtures, plus the constant-width property that keeps
// the persistent graph set from rebuilding on a subset. It does NOT prove that
// any runner reaches this code, that a host sub-op is numerically correct (that
// is AC2, the identity probe, and it is a separate GPU task), or that the split
// is faster. AC5 applies in advance: a run in which the mechanism never fires is
// a recorded outcome, never a pass with an implied benefit.

#include "../src/llama-expert-split-policy.h"

#include <cstdio>
#include <vector>

using llama_ls_policy::ls_gate_constants;
using llama_ls_policy::ls_gate_policy;
using llama_ls_policy::ls_gate_verdict;
using llama_ls_policy::ls_row_state;
using llama_ls_policy::ls_split_plan;
using llama_ls_policy::ls_split_enable;

static int failures = 0;

static void expect(bool cond, const char * what) {
    if (cond) {
        fprintf(stderr, "ok   %s\n", what);
    } else {
        fprintf(stderr, "FAIL %s\n", what);
        failures++;
    }
}

// The measured constants from the SDD's cost gate, so the fixtures exercise the
// numbers the rule was derived from rather than convenient round numbers.
static ls_gate_constants measured_constants() {
    ls_gate_constants c;
    c.policy            = ls_gate_policy::HOST_ALWAYS;
    c.host_us_per_expert = 36.5;       // us, marginal, tools/expert-cpu-bench
    c.row_bytes          = 1769472;    // one Q4_K expert row
    c.link_bytes_per_s   = 10.35e9;    // best measured END-TO-END feed
    return c;
}

static std::vector<ls_row_state> rows(size_t n, ls_row_state st) {
    return std::vector<ls_row_state>(n, st);
}

// ---------------------------------------------------------------------------
// 0. The cost gate reproduces the measured comparison it was derived from.
// ---------------------------------------------------------------------------
static void test_gate_arithmetic() {
    fprintf(stderr, "\n== gate: measured constants give a 3.4-6.7x host margin ==\n");

    const ls_gate_constants c = measured_constants();
    const ls_gate_verdict v  = llama_ls_policy::ls_evaluate_gate(c);

    expect(v.evaluable, "measured constants evaluate");
    expect(v.host_us > 0.0 && v.wait_us > 0.0, "both estimates are populated");
    // 1,769,472 B / 10.35 GB/s = 170.96 us to fetch; 36.5 us to compute.
    expect(v.wait_us > 170.0 && v.wait_us < 172.0, "wait estimate lands at the measured ~171 us");
    expect(v.host_us < v.wait_us, "host is cheaper than the link, so the gate opens");
    expect(v.wait_us / v.host_us > 3.4 && v.wait_us / v.host_us < 6.7,
           "the margin is inside the SDD's recorded 3.4-6.7x band");
}

// ---------------------------------------------------------------------------
// 1. Classification. ABSENT dominates copy_complete, because a row with no
//    outstanding copy can never become ready.
// ---------------------------------------------------------------------------
static void test_classification() {
    fprintf(stderr, "\n== ls_classify_row: issued x copy_complete ==\n");

    expect(llama_ls_policy::ls_classify_row(true,  true)  == ls_row_state::READY,
           "issued + complete -> READY");
    expect(llama_ls_policy::ls_classify_row(true,  false) == ls_row_state::PENDING,
           "issued + in flight -> PENDING");
    expect(llama_ls_policy::ls_classify_row(false, false) == ls_row_state::ABSENT,
           "not issued -> ABSENT");
    expect(llama_ls_policy::ls_classify_row(false, true) == ls_row_state::ABSENT,
           "not issued stays ABSENT even if a stale complete flag is set");
}

// ---------------------------------------------------------------------------
// 2. AC1: all-READY never splits. Nothing is in flight, so there is no transfer
//    to save and the gate is never reached.
// ---------------------------------------------------------------------------
static void test_all_ready_never_splits() {
    fprintf(stderr, "\n== AC1.1: all-READY must not fire ==\n");

    const ls_gate_verdict g = llama_ls_policy::ls_evaluate_gate(measured_constants());
    const ls_split_plan   p = llama_ls_policy::ls_plan_split(
        rows(8, ls_row_state::READY), g, 8, 1769472);

    expect(p.ok, "plan accepted");
    expect(p.experts_host == 0, "experts_host == 0");
    expect(p.experts_device == 8, "every expert stays on the device");
    expect(p.rows_gate_cpu == 0, "the gate sent nothing to the host");
    expect(p.staged_bytes_avoided == 0, "no staged bytes avoided");
}

// ---------------------------------------------------------------------------
// 3. AC1: all-PENDING with the gate open routes every row to the host.
// ---------------------------------------------------------------------------
static void test_all_pending_gate_open() {
    fprintf(stderr, "\n== AC1.2: all-PENDING, gate open ==\n");

    const ls_gate_verdict g = llama_ls_policy::ls_evaluate_gate(measured_constants());
    const ls_split_plan   p = llama_ls_policy::ls_plan_split(
        rows(8, ls_row_state::PENDING), g, 8, 1769472);

    expect(p.ok, "plan accepted");
    expect(p.experts_host == 8, "every non-resident row computes on the host");
    expect(p.experts_device == 0, "the device sub-op carries no demanded work");
    expect(p.rows_gate_cpu == 8, "rows_gate_cpu == 8");
    expect(p.rows_gate_gpu == 0, "rows_gate_gpu == 0");
    expect(p.staged_bytes_avoided == 8ull * 1769472ull, "staged bytes avoided is 8 rows");
}

// ---------------------------------------------------------------------------
// 4. AC1: a closed gate returns PENDING rows to the ordinary wait path,
//    UNCHANGED. This is the property that makes the flag safe to add: with the
//    gate shut, the plan must be indistinguishable from no split at all.
// ---------------------------------------------------------------------------
static void test_closed_gate_returns_to_wait() {
    fprintf(stderr, "\n== AC1.3: closed gate returns PENDING rows to the wait path ==\n");

    // Force the gate shut by making the host look more expensive than the link:
    // a host cost above the fetch estimate must not open.
    ls_gate_constants slow = measured_constants();
    slow.host_us_per_expert = 5000.0;
    const ls_gate_verdict closed = llama_ls_policy::ls_evaluate_gate(slow);
    expect(closed.evaluable && !closed.open, "the slow-host constants close the gate");

    const ls_split_plan p = llama_ls_policy::ls_plan_split(
        rows(8, ls_row_state::PENDING), closed, 8, 1769472);

    expect(p.ok, "a closed gate is NOT a refusal");
    expect(p.experts_host == 0, "nothing goes to the host");
    expect(p.experts_device == 8, "every row waits on the device, as with the split off");
    expect(p.rows_gate_gpu == 8, "the 8 PENDING rows are recorded as gate->gpu");
    expect(p.staged_bytes_avoided == 0, "no staged bytes avoided");

    // Inertness equivalence: with the gate shut the descriptors must match an
    // all-READY plan slot for slot, because both mean "device, wait, execute".
    const ls_split_plan wait_plan = llama_ls_policy::ls_plan_split(
        rows(8, ls_row_state::READY), closed, 8, 1769472);
    bool identical = wait_plan.ok && wait_plan.device.size() == p.device.size();
    for (size_t i = 0; identical && i < p.device.size(); ++i) {
        identical = (p.device[i].expert_id == wait_plan.device[i].expert_id) &&
                    (p.device[i].on_host   == wait_plan.device[i].on_host)   &&
                    (p.device[i].is_pad    == wait_plan.device[i].is_pad);
    }
    expect(identical, "closed-gate device sub-op is slot-identical to the no-split plan");
}

// ---------------------------------------------------------------------------
// 5. AC1: a mixed set splits per row, and READY never moves.
// ---------------------------------------------------------------------------
static void test_mixed_set() {
    fprintf(stderr, "\n== AC1.4: mixed READY/PENDING ==\n");

    std::vector<ls_row_state> d = {
        ls_row_state::READY, ls_row_state::PENDING, ls_row_state::READY, ls_row_state::PENDING,
        ls_row_state::READY, ls_row_state::PENDING, ls_row_state::READY, ls_row_state::PENDING,
    };
    const ls_gate_verdict g = llama_ls_policy::ls_evaluate_gate(measured_constants());
    const ls_split_plan   p = llama_ls_policy::ls_plan_split(d, g, 8, 1769472);

    expect(p.ok, "plan accepted");
    expect(p.experts_device == 4 && p.experts_host == 4, "4 READY on device, 4 PENDING on host");
    expect(p.rows_gate_cpu == 4 && p.rows_gate_gpu == 4, "the gate counters split 4/4");

    bool host_slots_match = true;
    for (size_t i = 0; i < 8; ++i) {
        host_slots_match = host_slots_match &&
                           (p.host[i].on_host == (d[i] == ls_row_state::PENDING)) &&
                           (p.device[i].on_host == false);
    }
    expect(host_slots_match, "each row landed on the backend its own state selected");

    // The demand set is authoritative: every demanded expert appears exactly
    // once, in order, on exactly one backend. Nothing dropped or reordered.
    bool ordered = true;
    for (size_t i = 0; i < 8; ++i) {
        ordered = ordered && (p.device[i].expert_id == (int32_t) i) &&
                  (p.host[i].expert_id   == (int32_t) i);
    }
    expect(ordered, "the router's executed id set is preserved in order on both sub-ops");
}

// ---------------------------------------------------------------------------
// 6. AC1: an absent demanded row REFUSES. This is a staging defect, and the
//    SDD forbids a silent host fallback for it.
// ---------------------------------------------------------------------------
static void test_absent_row_refuses() {
    fprintf(stderr, "\n== AC1.5: an absent demanded row refuses ==\n");

    const ls_gate_verdict g = llama_ls_policy::ls_evaluate_gate(measured_constants());
    std::vector<ls_row_state> d = {
        ls_row_state::READY, ls_row_state::PENDING, ls_row_state::ABSENT, ls_row_state::READY,
    };
    const ls_split_plan p = llama_ls_policy::ls_plan_split(d, g, 8, 1769472);

    expect(!p.ok, "the plan is refused");
    expect(p.refuse_reason != nullptr, "a reason is named");
    expect(p.experts_host == 0 && p.experts_device == 0,
           "a refused plan executes nothing rather than guessing");
    expect(p.device.empty() && p.host.empty(), "no sub-op is emitted on refusal");
}

// ---------------------------------------------------------------------------
// 7. AC1: an unevaluable policy refuses rather than defaulting.
// ---------------------------------------------------------------------------
static void test_unevaluable_policy_refuses() {
    fprintf(stderr, "\n== AC1.6: an unevaluable gate policy refuses ==\n");

    // (a) the policy itself is unknown
    ls_gate_constants unknown = measured_constants();
    unknown.policy = ls_gate_policy::UNKNOWN;
    const ls_gate_verdict vu = llama_ls_policy::ls_evaluate_gate(unknown);
    expect(!vu.evaluable, "an unknown policy is not evaluable");

    const ls_split_plan pu = llama_ls_policy::ls_plan_split(
        rows(8, ls_row_state::PENDING), vu, 8, 1769472);
    expect(!pu.ok, "the plan refuses on an unknown policy");
    expect(pu.refuse_reason != nullptr, "the unknown policy names a reason");

    // (b) the policy is named but its constants cannot produce a comparison
    ls_gate_constants broken = measured_constants();
    broken.link_bytes_per_s = 0.0;
    const ls_gate_verdict vb = llama_ls_policy::ls_evaluate_gate(broken);
    expect(!vb.evaluable, "a zero feed rate is not evaluable");
    expect(!llama_ls_policy::ls_plan_split(rows(8, ls_row_state::PENDING), vb, 8, 1769472).ok,
           "the plan refuses on unevaluable constants");

    // A negative feed rate must not sneak through as "infinitely fast"
    ls_gate_constants negative = measured_constants();
    negative.link_bytes_per_s = -1.0;
    expect(!llama_ls_policy::ls_evaluate_gate(negative).evaluable,
           "a negative feed rate is refused, not treated as free transfer");
}

// ---------------------------------------------------------------------------
// 8. AC1: padding contributes exactly zero and both sub-ops are constant width
//    for EVERY input, which is what keeps the graph set from rebuilding.
// ---------------------------------------------------------------------------
static void test_padding_and_constant_width() {
    fprintf(stderr, "\n== AC1.7: padding is inert, width is constant ==\n");

    const ls_gate_verdict g = llama_ls_policy::ls_evaluate_gate(measured_constants());

    // top_k = 8 is the shape; demand fewer than 8 experts (smaller k) and check
    // the tail is padded, inert, and identical in width.
    for (size_t k = 1; k <= 8; ++k) {
        const ls_split_plan p = llama_ls_policy::ls_plan_split(
            rows(k, ls_row_state::PENDING), g, 8, 1769472);

        char msg[128];
        snprintf(msg, sizeof(msg),
                 "k=%zu: device sub-op is exactly top_k wide", k);
        expect(p.ok && p.device.size() == 8, msg);
        snprintf(msg, sizeof(msg),
                 "k=%zu: host sub-op is exactly top_k wide", k);
        expect(p.ok && p.host.size() == 8, msg);

        size_t pads = 0;
        bool pad_inert = true;
        for (size_t i = k; i < 8; ++i) {
            pads++;
            pad_inert = pad_inert && p.device[i].is_pad && p.host[i].is_pad &&
                        !p.device[i].on_host && !p.host[i].on_host &&
                        p.device[i].expert_id < 0 && p.host[i].expert_id < 0;
        }
        snprintf(msg, sizeof(msg),
                 "k=%zu: %zu pad slots, all inert (no id, no backend)", k, 8 - k);
        expect(pads == 8 - k && pad_inert, msg);

        // Padding contributes exactly zero: it is neither demanded nor counted.
        snprintf(msg, sizeof(msg),
                 "k=%zu: experts_total counts demanded rows only", k);
        expect(p.experts_total == k, msg);
        snprintf(msg, sizeof(msg),
                 "k=%zu: device + host == total (pads excluded)", k);
        expect(p.experts_device + p.experts_host == p.experts_total, msg);
        snprintf(msg, sizeof(msg),
                 "k=%zu: avoided bytes == demanded host rows only", k);
        expect(p.staged_bytes_avoided == k * 1769472ull, msg);
    }

    // Constant width must hold across every residency mix too, not just k.
    bool width_holds = true;
    for (size_t k = 1; k <= 8; ++k) {
        for (int pattern = 0; pattern < 4 && width_holds; ++pattern) {
            std::vector<ls_row_state> d;
            for (size_t i = 0; i < k; ++i) {
                d.push_back(((int) i % 2) == (pattern % 2) ? ls_row_state::READY : ls_row_state::PENDING);
            }
            const ls_split_plan p = llama_ls_policy::ls_plan_split(d, g, 8, 1769472);
            width_holds = width_holds && p.ok && p.device.size() == 8 && p.host.size() == 8 &&
                          p.experts_device + p.experts_host == p.experts_total;
        }
    }
    expect(width_holds, "every residency mix and every k yields the same two widths");
}

// ---------------------------------------------------------------------------
// 9. The partition is re-derived per (layer, token), never cached. Two calls
//    with the same rows but different residency must differ.
// ---------------------------------------------------------------------------
static void test_not_fixed_at_run_start() {
    fprintf(stderr, "\n== the split is per (layer, token), not a fixed plan ==\n");

    const ls_gate_verdict g = llama_ls_policy::ls_evaluate_gate(measured_constants());
    const ls_split_plan a = llama_ls_policy::ls_plan_split(
        rows(8, ls_row_state::READY), g, 8, 1769472);
    const ls_split_plan b = llama_ls_policy::ls_plan_split(
        rows(8, ls_row_state::PENDING), g, 8, 1769472);

    expect(a.experts_host == 0 && b.experts_host == 8,
           "the same demanded set splits differently once residency changes");
}

// --- T-GPLR-82 S2: rewire guard fixtures -----------------------------------

static void test_rewire_guard_accepts_host_placed() {
    std::vector<llama_ls_policy::ls_rewire_observation> obs;
    for (int i = 0; i < 8; ++i) {
        llama_ls_policy::ls_rewire_observation o;
        o.on_host_leg    = true;
        o.weight_is_host = true; // gate/up/down, all three bound to the mmap rows
        obs.push_back(o);
    }
    const llama_ls_policy::ls_rewire_verdict v = llama_ls_policy::ls_check_no_rewire(obs);
    expect(v.ok && v.rewired == 0 && v.host_leg_weights == 8,
           "a host leg whose weights are all host-placed passes the guard");
}

static void test_rewire_guard_refuses_one_rewired_weight() {
    std::vector<llama_ls_policy::ls_rewire_observation> obs(8);
    for (size_t i = 0; i < obs.size(); ++i) {
        obs[i].on_host_leg    = true;
        obs[i].weight_is_host = true;
    }
    obs[5].weight_is_host = false; // down_exps moved to a device buffer
    const llama_ls_policy::ls_rewire_verdict v = llama_ls_policy::ls_check_no_rewire(obs);
    expect(!v.ok && v.rewired == 1 && v.refuse_reason != nullptr,
           "ONE rewired host-leg weight refuses the run rather than repairing it");
}

static void test_rewire_guard_ignores_the_device_leg() {
    std::vector<llama_ls_policy::ls_rewire_observation> obs(8);
    for (size_t i = 0; i < obs.size(); ++i) {
        obs[i].on_host_leg    = false; // device leg
        obs[i].weight_is_host = false; // device weights are SUPPOSED to be device
    }
    const llama_ls_policy::ls_rewire_verdict v = llama_ls_policy::ls_check_no_rewire(obs);
    expect(v.ok && v.host_leg_weights == 0,
           "device-leg weights being device-placed is the row-set mechanism, not a rewire");
}

static void test_rewire_guard_refuses_a_compute_staging_buffer() {
    // T-GPLR-82 S2. A weight can be host-placed AND still be the forbidden
    // shape: the scheduler's compute_splits path stages into a COMPUTE-usage
    // buffer, and a host-leg weight living in one is exactly the "op-offload
    // path that rewires a weight source to a compute-usage staging buffer" the
    // SDD forbids. Placement alone cannot see it, which is why the observation
    // carries the usage flag as a separate fact.
    std::vector<llama_ls_policy::ls_rewire_observation> obs(3);
    for (size_t i = 0; i < obs.size(); ++i) {
        obs[i].on_host_leg                       = true;
        obs[i].weight_is_host                    = true;
        obs[i].weight_in_compute_staging_buffer  = (i == 2);
    }
    const llama_ls_policy::ls_rewire_verdict v = llama_ls_policy::ls_check_no_rewire(obs);
    expect(!v.ok && v.rewired == 0 && v.staging_weights == 1 && v.refuse_reason != nullptr,
           "a host-leg weight in a COMPUTE-usage staging buffer refuses, and it is "
           "reported as the staging shape rather than as a placement rewire");
}

static void test_rewire_guard_refuses_a_live_min_batch_override() {
    // T-GPLR-82 S2. The MIN_BATCH=1 route was reverted out of the tree on
    // 2026-09-22, so today nothing sets this; the fixture exists so that
    // re-introducing the route fails HERE, at the guard, instead of producing a
    // run that deadlocks on the device the way T-GPLR-16 did.
    std::vector<llama_ls_policy::ls_rewire_observation> obs(3);
    for (size_t i = 0; i < obs.size(); ++i) {
        obs[i].on_host_leg              = true;
        obs[i].weight_is_host           = true;
        obs[i].min_batch_override_live  = (i == 0);
    }
    const llama_ls_policy::ls_rewire_verdict v = llama_ls_policy::ls_check_no_rewire(obs);
    expect(!v.ok && v.min_batch_live == 1 && v.refuse_reason != nullptr,
           "a live MIN_BATCH row-copy override refuses the host leg by name");
}

// ---------------------------------------------------------------------------
// T-GPLR-82 S3/S4 (AC1 for the split point): the host leg must not be BUILT
// when the executor cannot admit a row.
//
// THE PROPERTY. The split graph commits host_w rows per (layer, token) to the
// CPU leg AT BUILD TIME -- dev_rows = top_k - host_w, and the host leg's ids are
// the unconditional tail of the row (the hsel memcpy, which runs whenever the
// width is non-zero, independent of any plan). The plan cannot admit a single
// row until the budget is non-zero, so building that shape with the budget at
// its default changes the arithmetic while the ledger honestly reports zero host
// rows. MEASURED on exe 66b4eb9e, session gplr82-ac1, arm A2d, 2026-10-10:
// host share 0.000000, sum(experts_device) == sum(experts_total) == 163,840,
// ffn_dev = 2 (mixed) on all 20,480 records against 1 for the control, and
// divergence from the control at TG step 1 with all 512 logits finite.
//
// Therefore ls_split_in_play must be FALSE whenever the executor's own admission
// predicate (`s3` in load_row_set) would admit nothing.
// ---------------------------------------------------------------------------
static void test_split_in_play_requires_a_nonzero_budget() {
    fprintf(stderr, "\n== split point: a zero budget must NOT build the host leg ==\n");

    ls_split_enable e;
    e.host_w    = 2;
    e.budget_us = 0.0;                                  // the flag's DEFAULT
    expect(!llama_ls_policy::ls_split_in_play(e),
           "width 2 with the budget at its default 0 is NOT in play (the measured A2d shape)");

    e.budget_us = 2.0e6;                                 // ~unbounded: 2e6/36.5 us -> cap = width
    expect(llama_ls_policy::ls_split_in_play(e),
           "width 2 with a non-zero budget IS in play");
}

static void test_split_in_play_requires_a_width() {
    fprintf(stderr, "\n== split point: with the flag absent nothing is in play ==\n");

    ls_split_enable e;
    e.host_w    = 0;                                     // --experts-host absent
    e.budget_us = 2.0e6;
    expect(!llama_ls_policy::ls_split_in_play(e),
           "budget with no width is NOT in play (the flag's own default path)");
}

static void test_split_in_play_matches_the_executor_arm_gates() {
    fprintf(stderr, "\n== split point: every arm gate the executor applies also closes the build ==\n");

    // The executor's s3 predicate carries all of these; a build that ignored one
    // of them would build a shape whose admission is off for the whole run.
    ls_split_enable base;
    base.host_w    = 2;
    base.budget_us = 2.0e6;

    ls_split_enable recon     = base; recon.recon         = true;
    ls_split_enable spec_on   = base; spec_on.spec_on     = true;
    ls_split_enable spec_mode = base; spec_mode.spec_mode = 1;
    ls_split_enable spec_arm  = base; spec_arm.spec_arm   = 0;
    ls_split_enable spec_pool = base; spec_pool.spec_pool = 1;

    expect(!llama_ls_policy::ls_split_in_play(recon),
           "the row-set recon arm keeps the split out of play");
    expect(!llama_ls_policy::ls_split_in_play(spec_on),
           "the predictor arm keeps the split out of play");
    expect(!llama_ls_policy::ls_split_in_play(spec_mode),
           "a non-zero spec mode keeps the split out of play");
    expect(!llama_ls_policy::ls_split_in_play(spec_arm),
           "a selected spec arm keeps the split out of play");
    expect(!llama_ls_policy::ls_split_in_play(spec_pool),
           "a spec pool keeps the split out of play");

    expect(llama_ls_policy::ls_split_in_play(base),
           "the plain run (width, budget, no arm) is in play");
}

int main() {
    fprintf(stderr, "T-GPLR-61 step A -- moe-row-split policy and shape (AC1)\n");

    test_gate_arithmetic();
    test_classification();
    test_all_ready_never_splits();
    test_all_pending_gate_open();
    test_closed_gate_returns_to_wait();
    test_mixed_set();
    test_absent_row_refuses();
    test_unevaluable_policy_refuses();
    test_padding_and_constant_width();
    test_not_fixed_at_run_start();

    test_split_in_play_requires_a_nonzero_budget();
    test_split_in_play_requires_a_width();
    test_split_in_play_matches_the_executor_arm_gates();

    test_rewire_guard_accepts_host_placed();
    test_rewire_guard_refuses_one_rewired_weight();
    test_rewire_guard_ignores_the_device_leg();
    test_rewire_guard_refuses_a_compute_staging_buffer();
    test_rewire_guard_refuses_a_live_min_batch_override();

    fprintf(stderr, "\n%s (%d failure%s)\n",
            failures == 0 ? "AC1 PASS" : "AC1 FAIL",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}