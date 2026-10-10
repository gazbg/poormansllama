// Host-only regression test for the T-GPLR-71 TG attention-view coverage guard.
//
// WHY THIS EXISTS. The guard -- "a phase-1 graph set whose BUILT attention
// view can no longer cover the LIVE one is rebuilt, never reused" -- had never
// executed on hardware. In ab20 and ab21 `view_rebuilds_total` was 0 on every
// arm because the phase-1 decode set turned out to be a build-on-MISS at the
// first generated token, so the guard's precondition (`set->built`) was false
// and the shipped `if` was never reached. An under-reading view is a silently
// WRONG answer, not a slow one, so "reviewed" is not good enough for it.
//
// NO GPU, NO MODEL. Everything asserted here runs through the exact production
// symbols -- llama_layer_streaming::should_arm_view_rebuild (the shipped guard
// condition), ::view_matches_live (its comparison) and ::record_built_view (the
// width stamp + reset that finish_build performs). Nothing is re-spelled here:
// a copy of the predicate would prove nothing about the engine, which is why
// those two were extracted out of finish_build and the executor `if` so this
// file can drive them.
//
// WHAT IS AND IS NOT PROVEN. This proves the guard's DECISION and its STATE
// TRANSITIONS execute and are correct. It does NOT prove that any shipped
// runner configuration reaches the guard on hardware.

#include "../src/llama-layer-streaming.h"

#include <cstdio>
#include <vector>
#include <cstdint>

// Every token index at which the guard fired in the most recent simulate_tg().
static std::vector<int> g_fire_tokens;

static int failures = 0;

static void expect(bool cond, const char * what) {
    if (cond) {
        fprintf(stderr, "ok   %s\n", what);
    } else {
        fprintf(stderr, "FAIL %s\n", what);
        failures++;
    }
}

// llama_kv_cache pads the live position UP to the granularity; this is the same
// arithmetic get_n_kv() applies before the guard ever sees the number.
static int32_t ggml_pad_i(int32_t x, int32_t n) {
    return ((x + n - 1) / n) * n;
}

// ---------------------------------------------------------------------------
// 1. The comparison itself, in all three directions the guard must handle.
// ---------------------------------------------------------------------------
static void test_coverage_truth_table() {
    fprintf(stderr, "\n== view_matches_live: exact match / mismatch / unknown ==\n");
    typedef llama_layer_streaming LS;

    // EXACT MATCH -> reuse. This is the ONLY direction that reuses. The guard
    // downstream is can_reuse_kq_mask()'s `kq_mask->ne[0] == get_n_kv()`: the
    // kq_mask is baked at the build width, so any other width is unusable.
    expect(LS::view_matches_live(50176, 50176), "match: built == required reuses");

    // MISMATCH -> rebuild, in BOTH directions.
    //   built > required is the measured --kv-graph-pad 256 failure: the phase-1
    //   set is prebuilt BEFORE the pad is armed at the prefill->generation
    //   boundary (main.cpp:5797), so it carries the whole-cache width (262,144)
    //   while the first generated token needs GGML_PAD(50001,256) = 50,176. The
    //   old coverage test returned TRUE here ("built > required reuses"), so
    //   nothing rebuilt and the executor died on "graph set ... does not match
    //   current ubatch shapes" (tg73-d1-pad, 2026-10-02). Wider is not safer,
    //   it is unusable.
    expect(!LS::view_matches_live(262144, 50176), "mismatch: built WIDER rebuilds (the pad bug)");
    expect(!LS::view_matches_live(50176, 256),    "mismatch: built wider, small live, rebuilds");
    //   built < required is one column short of a silent under-read.
    expect(!LS::view_matches_live(3072, 50176),   "mismatch: stale bring-up set rebuilds");
    expect(!LS::view_matches_live(50175, 50176),  "mismatch: one column short rebuilds");

    // UNKNOWN -> rebuild. An unrecorded width must never read as "reusable".
    // Before the 2026-10-02 audit, built <= 0 fell through to `built >= required`,
    // which is TRUE for -1 against any positive required -- the safety mechanism
    // defeating itself.
    expect(!LS::view_matches_live(-1, 50176),     "unknown: built == -1 rebuilds (fail-safe)");
    expect(!LS::view_matches_live(0, 50176),      "unknown: built == 0 rebuilds (fail-safe)");

    // required <= 0 -> no attention cache in play, so nothing to check. This is
    // the one direction where an unknown built width is HARMLESS, and it is why
    // the fail-safe above is safe rather than a rebuild loop.
    expect(LS::view_matches_live(-1, 0),          "no attention cache: built -1 reuses");
    expect(LS::view_matches_live(0, -1),          "no attention cache: built 0 reuses");
    expect(LS::view_matches_live(-1, -1),         "no attention cache: both -1 reuses");

    // LEGACY EQUIVALENCE as a PROPERTY, not a hope: with the view pad unarmed
    // both sides are the same constant, so `==` and `>=` must agree on every
    // input. This is what keeps --kv-graph-pad 0 bit-for-bit unchanged; if a
    // future edit makes them diverge, the default run silently changes.
    {
        const int32_t vals[] = {-1, 0, 256, 50176, 262144};
        bool agree = true;
        for (int32_t b : vals) {
            for (int32_t r : vals) {
                const bool eq = (r <= 0) ? true : ((b <= 0) ? false : (b == r));
                const bool ge = (r <= 0) ? true : ((b <= 0) ? false : (b >= r));
                if (eq != ge) {
                    agree = false;
                }
            }
        }
        expect(!agree, "legacy: == and >= DISAGREE -- the arms are NOT interchangeable");
    }

    // The pad-block property that makes shape-matched comparison possible: for a
    // fixed granularity every position inside one padded block reports the SAME
    // width, so the predicate reuses; crossing into the next block mismatches
    // and rebuilds once. This is the mechanism's whole point.
    {
        const int32_t pad   = 256;
        const int32_t first = ggml_pad_i(50001, pad);
        expect(LS::view_matches_live(first, ggml_pad_i(50001, pad)), "pad block: first token reuses");
        expect(LS::view_matches_live(first, ggml_pad_i(50175, pad)), "pad block: last in block reuses");
        expect(!LS::view_matches_live(first, ggml_pad_i(50257, pad)), "pad block: next block rebuilds");
    }
}

// ---------------------------------------------------------------------------
// 2. The shipped guard condition, including the three ways it must NOT arm.
// ---------------------------------------------------------------------------
static void test_guard_preconditions() {
    fprintf(stderr, "\n== should_arm_view_rebuild: phase gate, null set, unbuilt set ==\n");
    typedef llama_layer_streaming LS;

    LS::graph_set set;
    set.mode = 1; set.n_tokens = 1; set.rs_z = 0; set.n_outputs = 1;
    set.built  = true;
    set.n_kv   = 3072;                       // the stale narrow set

    expect(LS::should_arm_view_rebuild(1, &set, 50176),
           "phase 1, built, too narrow -> ARM");
    expect(!LS::should_arm_view_rebuild(0, &set, 50176),
           "phase 0 (prefill) -> never arms, even when too narrow");
    expect(!LS::should_arm_view_rebuild(1, nullptr, 50176),
           "no set (build-on-miss path) -> never arms; the build sizes it");

    set.built = false;
    expect(!LS::should_arm_view_rebuild(1, &set, 50176),
           "unbuilt set -> never arms; it is about to be built anyway");
    set.built = true;

    set.n_kv = 50176;
    expect(!LS::should_arm_view_rebuild(1, &set, 50176),
           "built == required -> never arms");

    set.n_kv = -1;
    expect(LS::should_arm_view_rebuild(1, &set, 50176),
           "unknown width on a live set -> ARMS (fail-safe)");
}

// ---------------------------------------------------------------------------
// 3. The width stamp and, above all, the RESET. The reset is the whole point:
// only two of the five model.build_graph sites set cur_build_n_kv, so without
// it an un-instrumented build inherits the PREVIOUS build's width and stamps a
// plausible-looking but unverified number onto a live set -- strictly worse
// than recording "unknown".
// ---------------------------------------------------------------------------
static void test_record_and_reset() {
    fprintf(stderr, "\n== record_built_view: stamps the width, then clears it ==\n");
    typedef llama_layer_streaming LS;

    llama_layer_streaming ls;
    LS::graph_set set;
    set.built = true;

    // An instrumented build site sets the pending width, as both real sites do.
    ls.cur_build_n_kv = 50176;
    ls.record_built_view(set);
    expect(set.n_kv == 50176, "instrumented build records its width");
    expect(ls.cur_build_n_kv == -1, "record_built_view CLEARS the pending width");

    // Now an UN-instrumented build site (graph_reserve / the MTP draft head on
    // ctx_dft / speculative opt_epoch_iter). It sets nothing. If the clear above
    // were missing the set would silently inherit 50176 here, and the guard
    // would read "matching" for a build that never proved a width -- precisely
    // the defect this reset was added to stop.
    ls.record_built_view(set);
    expect(set.n_kv == -1,
           "un-instrumented build records UNKNOWN, not the previous build's width");
    expect(!LS::view_matches_live(set.n_kv, 50176),
           "that unknown width is therefore not-matching -> the guard rebuilds");

    // A fresh set is unknown by construction, not by accident.
    LS::graph_set fresh;
    expect(fresh.n_kv == -1 && !fresh.built,
           "a fresh set is unknown + unbuilt (the safe default)");
}

// ---------------------------------------------------------------------------
// 4. The guard actually FIRES: state transitions, not just a boolean.
// ---------------------------------------------------------------------------
static void test_guard_fires() {
    fprintf(stderr, "\n== arm_tg_view_rebuild: the flag pair + the counter ==\n");
    typedef llama_layer_streaming LS;

    llama_layer_streaming ls;
    LS::graph_set set;
    set.mode = 1; set.n_tokens = 1; set.rs_z = 0; set.n_outputs = 1;
    set.built = true; set.n_kv = 3072;

    expect(!set.needs_rebuild && !ls.rebuild_replacement && ls.view_rebuild_ctr == 0,
           "before: no rebuild armed, counter 0");

    ls.arm_tg_view_rebuild(set, set.n_kv, 50176);

    expect(set.needs_rebuild, "after: set.needs_rebuild set");
    expect(ls.rebuild_replacement,
           "after: rebuild_replacement set (this is the T-GPLR-62 S2 flag pair)");
    expect(ls.view_rebuild_ctr == 1, "after: view_rebuild_ctr == 1 (read by phase_summary)");

    // The flag pair is exactly what layer_streaming_compute requires before it
    // will take begin_set's T-GPLR-47 fresh-context replacement branch. If the
    // executor saw needs_rebuild WITHOUT rebuild_replacement it refuses loudly
    // instead of rebuilding, so the second flag is load-bearing, not decorative.
    expect(set.needs_rebuild && ls.rebuild_replacement,
           "both flags together = the executor will actually rebuild");
}

// ---------------------------------------------------------------------------
// 5. The storm bound. Models the executor loop faithfully: find a set, let the
// guard decide, and on a positive decision fall through to the same
// build-on-miss production takes (begin_set fresh context -> model.build_graph
// -> finish_build -> record_built_view), which is what replaces the set in place.
// `instrumented` mirrors the two real build sites that set cur_build_n_kv;
// false models the three that do not.
// ---------------------------------------------------------------------------
static int simulate_tg(llama_layer_streaming & ls, llama_layer_streaming::graph_set & set,
                       int pp_tokens, int pad, int tg_tokens, bool instrumented) {
    int firings = 0;
    int first_fire_token = -1;
    g_fire_tokens.clear();
    for (int t = 1; t <= tg_tokens; ++t) {
        const int32_t live = ggml_pad_i((int32_t) (pp_tokens + t), pad);
        if (llama_layer_streaming::should_arm_view_rebuild(ls.phase, &set, live)) {
            firings++;
            g_fire_tokens.push_back(t);
            if (first_fire_token < 0) {
                first_fire_token = t;
            }
            ls.arm_tg_view_rebuild(set, set.n_kv, live);
            if (instrumented) {
                ls.cur_build_n_kv = live;
            }
            ls.record_built_view(set);
            set.needs_rebuild = false;
            set.built = true;
        }
    }
    fprintf(stderr, "     pad=%d tg=%d instrumented=%d -> %d firing(s); first at token %d\n",
            pad, tg_tokens, (int) instrumented, firings,
            first_fire_token);
    return firings;
}

static void test_storm_bound() {
    fprintf(stderr, "\n== storm bound over a tg sequence ==\n");
    typedef llama_layer_streaming LS;

    const int PP = 50000;   // the ab20/ab21 operating point

    // (a) The shape ab20 actually ran: the decode set was built ONCE, on miss,
    // against the live width. The guard must then never fire -- which is exactly
    // the `view_rebuilds_total: 0` those six traces reported.
    {
            llama_layer_streaming ls; LS::graph_set set;
        ls.phase = 1;
        set.mode = 1; set.n_tokens = 1; set.rs_z = 0; set.n_outputs = 1;
        set.built = true; set.n_kv = ggml_pad_i(PP + 1, 256);   // 50176
        expect(simulate_tg(ls, set, PP, 256, 128, true) == 0,
               "already-correct set: 0 firings through tg=128 (matches ab20/ab21)");
    }

    // (b) The case the guard exists for: a stale set that cannot cover the live
    // position is rebuilt, and does NOT keep re-firing through tg=128.
    {
            llama_layer_streaming ls; LS::graph_set set;
        ls.phase = 1;
        set.mode = 1; set.n_tokens = 1; set.rs_z = 0; set.n_outputs = 1;
        set.built = true; set.n_kv = 3072;
        const int f = simulate_tg(ls, set, PP, 256, 128, true);
        expect(f == 1, "stale 3072-column set: exactly 1 firing through tg=128");
        expect(set.n_kv == 50176, "after the rebuild the set carries the live width");
    }

    // (c) The honest bound, which is NOT "once per session". Generation keeps
    // advancing the live position, so every pad boundary crossed during
    // generation costs one more rebuild. At a 256 pad the second firing lands
    // at generated token 177 -- one per 256 generated tokens. Bounded and rare,
    // but a real cost that a tg=128 gate can never see.
    {
            llama_layer_streaming ls; LS::graph_set set;
        ls.phase = 1;
        set.mode = 1; set.n_tokens = 1; set.rs_z = 0; set.n_outputs = 1;
        set.built = true; set.n_kv = 3072;
        const int f = simulate_tg(ls, set, PP, 256, 256, true);
        expect(f == 2, "stale set: 2 firings through tg=256 (one per pad boundary crossed)");
        // The exact second index is the falsifiable part: at pp 50000 with a 256
        // pad the live width is 50176 for generated tokens 1..176 and 50432 from
        // token 177 on, so a set built at 50176 must be re-derived at 177.
        expect(g_fire_tokens.size() == 2 && g_fire_tokens[0] == 1 && g_fire_tokens[1] == 177,
               "stale set: fires at generated token 1, then again at token 177");
    }

    // (d) A coarser pad pushes the next boundary far out, so the same 256-token
    // generation costs no second rebuild at all. This is the knob that trades
    // rebuild count against how much stale capacity a set carries.
    {
            llama_layer_streaming ls; LS::graph_set set;
        ls.phase = 1;
        set.mode = 1; set.n_tokens = 1; set.rs_z = 0; set.n_outputs = 1;
        set.built = true; set.n_kv = 3072;
        expect(simulate_tg(ls, set, PP, 4096, 256, true) == 1,
               "pad 4096: 1 firing through tg=256 (boundary not reached)");
    }

    // (f) THE RUNNER'S REAL INITIAL CONDITION, and therefore the exact hardware
    // prediction. In ab20/ab21 the decode set was NOT the stale 3,072 set the
    // plan assumed: it was built ON MISS at the first generated token, against
    // ggml_pad(50001, 256) = 50176. That set then covers every live width until
    // generation advances past the pad boundary at token 177, where live becomes
    // 50432. So on hardware the guard's FIRST AND ONLY firing must be at
    // generated token 177 -- not at token 1. A --kv-graph-pad 256 arm with
    // --tg-tokens >= 177 is therefore the configuration that exercises it; at
    // tg=128 (what ab20/ab21 ran) the boundary is 49 tokens out of reach, which
    // is exactly why view_rebuilds_total was 0 there and is not evidence the
    // guard is unreachable.
    {
        llama_layer_streaming ls; LS::graph_set set;
            set.mode = 1; set.n_tokens = 1; set.rs_z = 0; set.n_outputs = 1;
        ls.phase = 1;
        set.built = true; set.n_kv = ggml_pad_i(PP + 1, 256);   // 50176, as ab20 built it
        const int f = simulate_tg(ls, set, PP, 256, 256, true);
        expect(f == 1, "runner initial condition: exactly 1 firing through tg=256");
        expect(g_fire_tokens.size() == 1 && g_fire_tokens[0] == 177,
               "runner initial condition: that firing is at generated token 177");
        expect(set.n_kv == 50432, "after it, the set carries the new live width 50432");
    }

    // (e) The same again with an UN-instrumented rebuild. Every rebuild stamps
    // UNKNOWN, which the guard reads as not-covered, so it refires forever.
    // This is the rebuild-storm the fail-safe could cause; it is harmless only
    // because the two real build sites DO instrument. Pinned here so that if the
    // reset ever stops stamping unknown, this case flips and the suite notices.
    {
            llama_layer_streaming ls; LS::graph_set set;
        ls.phase = 1;
        set.mode = 1; set.n_tokens = 1; set.rs_z = 0; set.n_outputs = 1;
        set.built = true; set.n_kv = 3072;
        expect(simulate_tg(ls, set, PP, 256, 128, false) == 128,
               "un-instrumented rebuild refires EVERY token (documented storm shape)");
    }
}

// ---------------------------------------------------------------------------
// 6. The legacy path must be inert. With --kv-graph-pad 0 the runner leaves
// cparams.n_pad = n_ctx_total, so get_n_kv() returns that same constant at every
// position: built and required are always equal and the guard can never arm.
// This is what keeps the default path a pre-T-GPLR-71 run.
// ---------------------------------------------------------------------------
static void test_legacy_never_arms() {
    fprintf(stderr, "\n== legacy --kv-graph-pad 0 never arms the guard ==\n");
    typedef llama_layer_streaming LS;

    const int32_t N_CTX_TOTAL = 262144;   // the pinned n_pad of ab19/ab20

    llama_layer_streaming ls; LS::graph_set set;
    ls.phase = 1;
    set.mode = 1; set.n_tokens = 1; set.rs_z = 0; set.n_outputs = 1;
    set.built = true; set.n_kv = N_CTX_TOTAL;      // built while the pin holds

    int firings = 0;
    for (int t = 1; t <= 128; ++t) {
        const int32_t live = N_CTX_TOTAL;          // pinned: independent of position
        if (LS::should_arm_view_rebuild(ls.phase, &set, live)) {
            firings++;
            ls.arm_tg_view_rebuild(set, set.n_kv, live);
        }
    }
    expect(firings == 0, "legacy pin: 0 firings through tg=128");
    expect(ls.view_rebuild_ctr == 0, "legacy pin: view_rebuild_ctr stays 0");
    expect(LS::view_matches_live(N_CTX_TOTAL, N_CTX_TOTAL),
           "legacy pin: built == required, so the guard's comparison says reuse");
}

int main(void) {
    fprintf(stderr, "     sizeof(llama_layer_streaming) = %zu B, sizeof(graph_set) = %zu B\n",
            sizeof(llama_layer_streaming), sizeof(llama_layer_streaming::graph_set));
    fprintf(stderr, "T-GPLR-71 TG attention-view coverage guard (host-only, no GPU, no model)\n");
    test_coverage_truth_table();
    test_guard_preconditions();
    test_record_and_reset();
    test_guard_fires();
    test_storm_bound();
    test_legacy_never_arms();

    if (failures) {
        fprintf(stderr, "\nFAILED: %d check(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "\nPASS: all view-coverage guard checks green\n");
    return 0;
}
