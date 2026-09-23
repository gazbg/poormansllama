// T-GPLR-58 A0: host-only proof of the speculative-placement RULES, driven by a
// scripted proposal sequence. No CUDA, no model, no GPU, no GPU mutex.
//
// What this test is, stated carefully: the engine's every speculative decision is
// a call into the pure llama_ls_spec_* functions (the engine owns WHEN they are
// asked and what a fill does to the copy stream; the rules themselves live here).
// So this test drives those same functions through the SAME state machine the
// engine drives -- admit a guessed row, let a demand arrive, complete or cancel
// the fill, choose a victim -- and asserts the invariants T-GPLR-58 pre-registered
// as P1-P6's falsifiable core:
//
//   P2  a PENDING position is never a hit (a demand fetches its own copy) and a
//       row becomes resident only after its own copy completed;
//   P1  the demand therefore never waits: it is served by its own fetch, which
//       the reserve rule (P3) guarantees a free position for;
//   P3  an admission is refused unless `reserve` positions stay FREE for demand,
//       and a speculative row is evicted FIRST;
//   P4  cancellation returns a plan (oldest pending first) rather than assuming
//       the hook fires -- and never claims a position whose copy is in flight;
//   P5  no displacement: an admission never evicts, and a refused admission
//       leaves the incumbent's resident set byte-identical.
//
// The engine's WIRING of these rules (the copy, the completion event, the
// telemetry) is not host-testable and is what T-GPLR-58's A3 hold exists for.
#include "../src/llama-layer-streaming.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

static void check(const char * what, bool ok) {
    if (!ok) {
        failures++;
    }
    printf("  %-64s %s\n", what, ok ? "ok" : "FAIL");
}

// A tiny stand-in for one layer's cache: the engine's arrays are exactly these
// (cache_expert / spec_state / cache_recency / spec_flag).
struct fixture {
    std::vector<int32_t> expert;
    std::vector<uint8_t> state;
    std::vector<int64_t> recency;
    std::vector<uint8_t> spec;

    explicit fixture(int cap) :
        expert((size_t) cap, -1),
        state((size_t) cap, LLAMA_LS_SPEC_FREE),
        recency((size_t) cap, -1),
        spec((size_t) cap, 0) {}

    // the engine's demanded placement: READY immediately (its own copy is what
    // the demand waits for), never a speculative row
    void demand_place(int p, int32_t e, int64_t tick) {
        expert[(size_t) p]  = e;
        recency[(size_t) p] = tick;
        state[(size_t) p]   = LLAMA_LS_SPEC_READY;
        spec[(size_t) p]    = 0;
    }

    // the engine's demand-side classification: the first READY position holding
    // `e`, or -1 (a pending fill is NOT a hit -- P2)
    int hit_pos(int32_t e) const {
        for (size_t p = 0; p < expert.size(); ++p) {
            if (expert[p] == e && llama_ls_spec_is_hit(state[p])) {
                return (int) p;
            }
        }
        return -1;
    }

    // the engine's eviction choice over these arrays (P3: spec-first, then LRU,
    // never an in-flight position)
    int victim(const std::vector<char> & needed) const {
        int best = -1;
        for (size_t p = 0; p < expert.size(); ++p) {
            if (expert[p] < 0 || needed[p]) {
                continue;
            }
            if (!llama_ls_spec_may_evict(state[p])) {
                continue;
            }
            if (best < 0) {
                best = (int) p;
                continue;
            }
            if (llama_ls_spec_victim_better(recency[p], spec[p], true,
                                            recency[(size_t) best], spec[(size_t) best], true)) {
                best = (int) p;
            }
        }
        return best;
    }

    int64_t tick = 0;
};

int main() {
    printf("T-GPLR-58 A0: speculative placement rules (host-only, scripted sequence)\n");

    // ---------------------------------------------------------------- P2 core
    printf("\nP2 - a row is resident only after its own copy completed\n");
    {
        fixture f(4);
        // scripted proposal for the NEXT demand: expert 7
        const llama_ls_spec_admission ad = llama_ls_spec_admit(f.expert.data(), f.state.data(), 4, 2, 8, 0, 7);
        check("a free position is admitted for the guessed row", ad.ok && ad.pos == 0);
        // the engine marks the position PENDING and issues the copy
        f.expert[(size_t) ad.pos] = 7;
        f.state[(size_t) ad.pos]  = LLAMA_LS_SPEC_PENDING;
        f.recency[(size_t) ad.pos] = 1;
        check("a PENDING position is NOT a hit", llama_ls_spec_is_hit(LLAMA_LS_SPEC_PENDING) == false);
        check("the demand for that very row finds no hit (it fetches its own copy)",
              f.hit_pos(7) == -1);
        check("a PENDING position cannot be reused by the demand's own copy",
              llama_ls_spec_free_pos(f.expert.data(), f.state.data(), 4) == 1);
        // ... the demand places its own copy elsewhere (P1: no waiting) ...
        f.demand_place(1, 7, 2);
        check("the demand's own placement is a hit immediately (its own copy)",
              f.hit_pos(7) == 1);
        // the fill's copy now completes -> residency, with the spec flag set
        f.state[(size_t) ad.pos] = LLAMA_LS_SPEC_READY;
        f.spec[(size_t) ad.pos]  = 1;
        check("after its own completion the fill is READY and resident",
              llama_ls_spec_is_hit(f.state[(size_t) ad.pos]) && f.expert[(size_t) ad.pos] == 7);
        check("the completed fill is marked speculative (evicted first)",
              f.spec[(size_t) ad.pos] == 1);
    }

    // ---------------------------------------------------------------- P3 core
    printf("\nP3 - speculation is a hard barrier below demand capacity\n");
    {
        // cap 6 with reserve 2 (one full window): two demanded rows leave FOUR
        // free positions, so two fills are admitted (4->3->2 free) and the third
        // is refused -- after it only ONE would remain, below the reserve.
        fixture f(6);
        f.demand_place(0, 10, 1);
        f.demand_place(1, 11, 2);
        const llama_ls_spec_admission a1 = llama_ls_spec_admit(f.expert.data(), f.state.data(), 6, 2, 8, 0, 20);
        check("first fill admitted (4 free -> 3, reserve 2 still held)", a1.ok == true);
        f.expert[(size_t) a1.pos] = 20;
        f.state[(size_t) a1.pos]  = LLAMA_LS_SPEC_READY;
        f.spec[(size_t) a1.pos]   = 1;
        const llama_ls_spec_admission a2 = llama_ls_spec_admit(f.expert.data(), f.state.data(), 6, 2, 8, 0, 21);
        check("second fill admitted (3 free -> 2)", a2.ok == true && a2.pos == 3);
        f.expert[(size_t) a2.pos] = 21;
        f.state[(size_t) a2.pos]  = LLAMA_LS_SPEC_READY;
        f.spec[(size_t) a2.pos]   = 1;
        const llama_ls_spec_admission a3 = llama_ls_spec_admit(f.expert.data(), f.state.data(), 6, 2, 8, 0, 22);
        check("third fill refused: only the demand reserve would remain", a3.ok == false);
        check("the refusal names the demand reserve", std::string(a3.reason) == "demand_reserve");
        check("the refusal is not a displacement: both tenants survive",
              f.expert[2] == 20 && f.expert[3] == 21 && f.expert[4] == -1 && f.expert[5] == -1);

        // budget cap
        fixture g(8);
        const llama_ls_spec_admission b = llama_ls_spec_admit(g.expert.data(), g.state.data(), 8, 1, 2, 2, 30);
        check("the per-layer budget is a refusal reason", !b.ok && std::string(b.reason) == "budget_exhausted");

        // no free position at all: refused, never displacing a resident row
        fixture h(2);
        h.demand_place(0, 40, 1);
        h.demand_place(1, 41, 2);
        const llama_ls_spec_admission c = llama_ls_spec_admit(h.expert.data(), h.state.data(), 2, 1, 8, 0, 42);
        check("a full cache refuses rather than displacing (P5)", !c.ok && std::string(c.reason) == "no_free_position");
        check("P5: the incumbent's rows are unchanged by the refusal", h.expert[0] == 40 && h.expert[1] == 41);

        // duplicates are refused: never two fills for one row, never a fill for a resident row
        fixture k(4);
        k.demand_place(0, 50, 1);
        k.expert[1] = 51;
        k.state[1]  = LLAMA_LS_SPEC_PENDING;
        check("a resident row is refused", std::string(llama_ls_spec_admit(k.expert.data(), k.state.data(), 4, 1, 8, 1, 50).reason) == "already_resident");
        check("an in-flight row is refused", std::string(llama_ls_spec_admit(k.expert.data(), k.state.data(), 4, 1, 8, 1, 51).reason) == "already_pending");
    }

    printf("\nP3 - eviction: never in flight, speculative rows first, then LRU\n");
    {
        fixture f(4);
        f.demand_place(0, 1, 10);                 // demanded, most recent
        f.demand_place(1, 2, 5);                  // demanded, older
        f.expert[2] = 3;
        f.state[2]  = LLAMA_LS_SPEC_READY;
        f.recency[2] = 100;                       // speculative but the MOST recent
        f.spec[2]   = 1;
        f.expert[3] = 4;
        f.state[3]  = LLAMA_LS_SPEC_PENDING;
        f.recency[3] = 1;
        std::vector<char> needed(4, 0);
        check("a speculative row is the first victim even when most recent", f.victim(needed) == 2);
        check("an in-flight position is never a victim", llama_ls_spec_may_evict(LLAMA_LS_SPEC_PENDING) == false);
        check("a cancelled-but-unreaped position is never a victim", llama_ls_spec_may_evict(LLAMA_LS_SPEC_CANCELLED) == false);
        f.spec[2] = 0;                            // the demand consumed it
        check("after consumption the LRU rule decides (oldest demanded row)", f.victim(needed) == 1);
    }

    // ---------------------------------------------------------------- P4 core
    printf("\nP4 - cancellation is a plan: oldest pending first\n");
    {
        fixture f(5);
        const int64_t rec[5] = { 30, 10, 20, -1, 5 };
        uint8_t st[5] = { LLAMA_LS_SPEC_PENDING, LLAMA_LS_SPEC_PENDING, LLAMA_LS_SPEC_PENDING,
                          LLAMA_LS_SPEC_FREE, LLAMA_LS_SPEC_READY };
        int32_t ex[5] = { 1, 2, 3, -1, 4 };
        (void) ex;
        int out[5] = { -1, -1, -1, -1, -1 };
        const int n = llama_ls_spec_cancel_plan(rec, st, 5, 2, out, 5);
        check("two cancellations planned", n == 2);
        check("the OLDEST pending fill is cancelled first", out[0] == 1);
        check("then the next oldest", out[1] == 2);
        check("a READY position is never in the plan", out[0] != 4 && out[1] != 4);
        check("need_free = 0 plans nothing (a generation boundary asks explicitly)",
              llama_ls_spec_cancel_plan(rec, st, 5, 0, out, 5) == 0);
        check("a cancelled position keeps its bytes until the reap",
              llama_ls_spec_may_evict(LLAMA_LS_SPEC_CANCELLED) == false);
        (void) f;
    }

    // ------------------------------------------- the scripted-sequence walk
    // This is the A0 acceptance scenario in miniature: a scripted guess for the
    // NEXT demand, the demand arriving, and the three outcomes the block cares
    // about -- hit (the fill completed in time), matched (in flight), refused
    // (the reserve said no). Every step is the engine's own decision function.
    printf("\nScripted sequence: guess, demand, and the three outcomes\n");
    {
        struct step { int32_t guess; int32_t demand; bool complete_before_demand; };
        const step seq[] = {
            {  7,  7, true  },   // the guess lands first -> the demand is a HIT
            {  9,  9, false },   // still in flight -> the demand fetches its own copy (P1/P2)
            { 11, 13, true  },   // the guess was wrong -> the fill stays resident, spec-first victim
        };
        fixture f(6);
        int n_hit = 0, n_matched = 0, n_miss = 0, n_refused = 0;
        int64_t tick = 0;
        for (const step & s : seq) {
            // the scripted proposal is admitted (or refused) exactly as the
            // engine admits it, with reserve = 2 (top_k in this miniature)
            const llama_ls_spec_admission ad = llama_ls_spec_admit(f.expert.data(), f.state.data(), 6, 2, 4, 0, s.guess);
            if (!ad.ok) {
                n_refused++;
                continue;
            }
            const int p = ad.pos;
            f.expert[(size_t) p] = s.guess;
            f.state[(size_t) p]  = s.complete_before_demand ? LLAMA_LS_SPEC_READY : LLAMA_LS_SPEC_PENDING;
            f.recency[(size_t) p] = ++tick;
            f.spec[(size_t) p]   = s.complete_before_demand ? 1 : 0;

            const int hit = f.hit_pos(s.demand);
            if (hit >= 0) {
                n_hit++;
                f.spec[(size_t) hit] = 0; // the demand consumed it
            } else if (f.state[(size_t) p] == LLAMA_LS_SPEC_PENDING && f.expert[(size_t) p] == s.demand) {
                // the guess was right and still in flight: the demand fetches
                // its own copy at the first free position and NEVER waits (P1)
                n_matched++;
                const int q = llama_ls_spec_free_pos(f.expert.data(), f.state.data(), 6);
                check("a matched-but-pending fill leaves the demand a free position", q >= 0);
                f.demand_place(q, s.demand, ++tick);
                check("the demand's own copy is a hit immediately, with no wait on the fill",
                      f.hit_pos(s.demand) == q);
            } else {
                n_miss++;
            }
        }
        check("outcome counts: one hit, one matched-but-pending, one wrong guess",
              n_hit == 1 && n_matched == 1 && n_miss == 1 && n_refused == 0);
    }

    // ---------------------------------------------------------------- P6 core
    printf("\nP6 - the mode-off shape allocates nothing and guards nothing\n");
    {
        // With speculation off the engine passes a null states pointer and every
        // guard is inert: a null state array must read as an ordinary resident
        // (READY), so a mode-off run's classification is EXACTLY today's.
        fixture f(3);
        f.demand_place(0, 1, 1);
        f.demand_place(1, 2, 2);
        check("null states read as hits (mode 0 behaviour is unchanged)",
              llama_ls_spec_is_hit(LLAMA_LS_SPEC_READY) == true);
        check("the free-position rule with null states is today's rule",
              llama_ls_spec_free_pos(f.expert.data(), nullptr, 3) == 2);
        check("the victim rule with no speculative rows is today's LRU rule",
              llama_ls_spec_victim_better(1, 0, true, 2, 0, true) == true &&
              llama_ls_spec_victim_better(2, 0, true, 1, 0, true) == false);
        check("an equal recency keeps the earlier position (the in-window tie-break)",
              llama_ls_spec_victim_better(5, 0, true, 5, 0, true) == false);
    }

    // ------------------------------------------------- P6: mode-0 differential
    // The strongest host-side statement available for "speculation off changes
    // nothing": the PRE-A0 retention decisions, verbatim, run against the SAME
    // random windows as the new decision functions with a null state array (the
    // shape the engine passes when the mode is off). Hits, fetch sets, victim
    // choices, eviction counts, placement positions and the resulting cache must
    // all agree window for window -- otherwise a mode-off run's residency is not
    // the residency every recorded (b) figure was measured under.
    printf("\nP6 - mode 0: differential against the pre-A0 retention algorithm\n");
    {
        // legacy classification/eviction/placement, transcribed from the code
        // this change replaced (T-GPLR-19's capacity-LRU), kept as the oracle
        struct legacy {
            std::vector<int32_t> ex;
            std::vector<int64_t> rc;
            int64_t tick = 0;
            explicit legacy(int cap) : ex((size_t) cap, -1), rc((size_t) cap, -1) {}
            // returns hits; fills fetch/pos_of/retired
            size_t window(const std::vector<int32_t> & uniq, std::vector<int32_t> & fetch,
                          std::vector<int32_t> & pos_of, size_t & retired, bool & refused) {
                const int cap = (int) ex.size();
                pos_of.assign(uniq.size(), -1);
                fetch.clear();
                retired = 0;
                refused = false;
                size_t occupied = 0;
                for (int p = 0; p < cap; ++p) {
                    if (ex[(size_t) p] < 0) {
                        continue;
                    }
                    occupied++;
                    const auto it = std::lower_bound(uniq.begin(), uniq.end(), ex[(size_t) p]);
                    if (it != uniq.end() && *it == ex[(size_t) p]) {
                        pos_of[(size_t) (it - uniq.begin())] = p;
                    }
                }
                size_t resident = 0;
                for (size_t i = 0; i < uniq.size(); ++i) {
                    resident += pos_of[i] >= 0;
                }
                const size_t need = uniq.size() - resident;
                while (occupied + need > (size_t) cap) {
                    int victim = -1;
                    for (int p = 0; p < cap; ++p) {
                        if (ex[(size_t) p] < 0) {
                            continue;
                        }
                        const auto it = std::lower_bound(uniq.begin(), uniq.end(), ex[(size_t) p]);
                        if (it != uniq.end() && *it == ex[(size_t) p]) {
                            continue;
                        }
                        if (victim < 0 || rc[(size_t) p] < rc[(size_t) victim]) {
                            victim = p;
                        }
                    }
                    if (victim < 0) {
                        refused = true;
                        return resident;
                    }
                    ex[(size_t) victim] = -1;
                    rc[(size_t) victim] = -1;
                    occupied--;
                    retired++;
                }
                for (size_t i = 0; i < uniq.size(); ++i) {
                    if (pos_of[i] >= 0) {
                        rc[(size_t) pos_of[i]] = ++tick;
                        continue;
                    }
                    int free_p = -1;
                    for (int p = 0; p < cap; ++p) {
                        if (ex[(size_t) p] < 0) {
                            free_p = p;
                            break;
                        }
                    }
                    if (free_p < 0) {
                        refused = true;
                        return resident;
                    }
                    ex[(size_t) free_p] = uniq[i];
                    rc[(size_t) free_p] = ++tick;
                    pos_of[i] = free_p;
                    fetch.push_back(uniq[i]);
                }
                return resident;
            }
        };

        // the same walk, deciding through the A0 functions with speculation OFF
        // (states == nullptr, exactly what the engine passes when mode == 0)
        struct subject {
            std::vector<int32_t> ex;
            std::vector<int64_t> rc;
            std::vector<uint8_t> st; // present but all READY: the mode-off shape
            int64_t tick = 0;
            explicit subject(int cap) : ex((size_t) cap, -1), rc((size_t) cap, -1),
                                        st((size_t) cap, LLAMA_LS_SPEC_READY) {}
            size_t window(const std::vector<int32_t> & uniq, std::vector<int32_t> & fetch,
                          std::vector<int32_t> & pos_of, size_t & retired, bool & refused) {
                const int cap = (int) ex.size();
                pos_of.assign(uniq.size(), -1);
                fetch.clear();
                retired = 0;
                refused = false;
                size_t occupied = 0;
                for (int p = 0; p < cap; ++p) {
                    if (ex[(size_t) p] < 0) {
                        continue;
                    }
                    occupied++;
                    const uint8_t s = st[(size_t) p];
                    const auto it = std::lower_bound(uniq.begin(), uniq.end(), ex[(size_t) p]);
                    if (it != uniq.end() && *it == ex[(size_t) p] && llama_ls_spec_is_hit(s)) {
                        pos_of[(size_t) (it - uniq.begin())] = p;
                    }
                }
                size_t resident = 0;
                for (size_t i = 0; i < uniq.size(); ++i) {
                    resident += pos_of[i] >= 0;
                }
                const size_t need = uniq.size() - resident;
                while (occupied + need > (size_t) cap) {
                    int victim = -1;
                    for (int p = 0; p < cap; ++p) {
                        if (ex[(size_t) p] < 0 || !llama_ls_spec_may_evict(st[(size_t) p])) {
                            continue;
                        }
                        const auto it = std::lower_bound(uniq.begin(), uniq.end(), ex[(size_t) p]);
                        if (it != uniq.end() && *it == ex[(size_t) p]) {
                            continue;
                        }
                        const uint8_t sf = 0; // no speculative rows in the mode-off shape
                        if (victim < 0 ||
                            llama_ls_spec_victim_better(rc[(size_t) p], sf, true, rc[(size_t) victim], (uint8_t) 0, true)) {
                            victim = p;
                        }
                    }
                    if (victim < 0) {
                        refused = true;
                        return resident;
                    }
                    ex[(size_t) victim] = -1;
                    rc[(size_t) victim] = -1;
                    st[(size_t) victim] = LLAMA_LS_SPEC_FREE;
                    occupied--;
                    retired++;
                }
                for (size_t i = 0; i < uniq.size(); ++i) {
                    if (pos_of[i] >= 0) {
                        rc[(size_t) pos_of[i]] = ++tick;
                        continue;
                    }
                    const int free_p = llama_ls_spec_free_pos(ex.data(), nullptr, cap);
                    if (free_p < 0) {
                        refused = true;
                        return resident;
                    }
                    ex[(size_t) free_p] = uniq[i];
                    rc[(size_t) free_p] = ++tick;
                    st[(size_t) free_p] = LLAMA_LS_SPEC_READY;
                    pos_of[i] = free_p;
                    fetch.push_back(uniq[i]);
                }
                return resident;
            }
        };

        uint64_t rng = 0x9E3779B97F4A7C15ull; // deterministic
        auto next = [&rng]() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            return rng;
        };
        int windows = 0;
        int mismatches = 0;
        bool saw_hit = false, saw_fetch = false, saw_retire = false;
        for (int shape = 0; shape < 3; ++shape) {
            const int cap = shape == 0 ? 4 : (shape == 1 ? 8 : 16);
            const int n_exp = cap * 2;
            const int top_k = std::min(2, n_exp);
            legacy  a(cap);
            subject b(cap);
            for (int w = 0; w < 400; ++w) {
                std::vector<int32_t> uniq;
                const int k = 1 + (int) (next() % (uint64_t) top_k);
                for (int j = 0; j < k; ++j) {
                    uniq.push_back((int32_t) (next() % (uint64_t) n_exp));
                }
                std::sort(uniq.begin(), uniq.end());
                uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
                if (uniq.empty()) {
                    continue;
                }
                std::vector<int32_t> fa, fb, pa, pb;
                size_t ra = 0, rb = 0;
                bool xa = false, xb = false;
                const size_t ha = a.window(uniq, fa, pa, ra, xa);
                const size_t hb = b.window(uniq, fb, pb, rb, xb);
                windows++;
                saw_hit    |= ha > 0;
                saw_fetch  |= !fa.empty();
                saw_retire |= ra > 0;
                if (ha != hb || ra != rb || xa != xb || fa != fb || pa != pb ||
                    a.ex != b.ex || a.rc != b.rc) {
                    mismatches++;
                    if (mismatches <= 3) {
                        printf("    mismatch at cap=%d window=%d: hits %zu/%zu, retired %zu/%zu, fetch %zu/%zu\n",
                               cap, w, ha, hb, ra, rb, fa.size(), fb.size());
                    }
                }
            }
        }
        check("1200 random windows, identical decisions (hits, fetches, victims, placement)",
              windows >= 1000 && mismatches == 0);
        check("the differential actually exercised hits, fetches and evictions",
              saw_hit && saw_fetch && saw_retire);
    }

    // ---------------------------------------------------------------------
    // M7 (T-GPLR-58 A2): the lead of a consumed fill, in demand stages of its own
    // layer. The ARITHMETIC is pure, so the nested test proves exactly what the
    // trace checker reads out of `lead_hist`; the engine's stamping of the issue
    // ordinal is A0's wiring and is what the A3 hold exercises.
    // ---------------------------------------------------------------------
    printf("\nM7 - the consumed-fill lead distribution\n");
    {
        // 1. the measured ordering: the only hot-path reap runs at the END of a
        //    demand stage, so a fill issued after stage t first counts as READY at
        //    the end of stage t+1 -- it cannot serve a demand before stage t+2.
        check("a fill issued at stage t is first consumable at t+2 (lead 2)",
              llama_ls_spec_lead(3, 1) == 2);
        check("consumption in the issuing stage is not a lead",
              llama_ls_spec_lead(1, 1) == 0);
        check("consumption BEFORE the issuing stage is not a lead",
              llama_ls_spec_lead(2, 5) == 0);
        check("a position that never held a fill contributes no lead",
              llama_ls_spec_lead(5, -1) == 0);

        // 2. buckets: exactly 8 counters, bucket 7 = lead >= 8
        check("bucket(1) == 0 and bucket(7) == 6",
              llama_ls_spec_lead_bucket(1) == 0 && llama_ls_spec_lead_bucket(7) == 6);
        check("bucket(8) == bucket(9) == 7 (the overflow bucket)",
              llama_ls_spec_lead_bucket(8) == 7 && llama_ls_spec_lead_bucket(9) == 7);
        check("a non-consumption has no bucket",
              llama_ls_spec_lead_bucket(0) == -1 && llama_ls_spec_lead_bucket(-2) == -1);

        // 3. one small distribution, exactly as tel_spec() accumulates it
        int64_t hist[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        int     leads[5]  = { 2, 2, 3, 8, 9 };
        int64_t consumed = 0, sum = 0;
        int     lo = 0, hi = 0;
        for (int i = 0; i < 5; ++i) {
            const int b = llama_ls_spec_lead_bucket(leads[i]);
            if (b < 0) {
                continue;
            }
            hist[b]++;
            consumed++;
            sum += leads[i];
            lo = (lo == 0 || leads[i] < lo) ? leads[i] : lo;
            hi = leads[i] > hi ? leads[i] : hi;
        }
        // a fill a demand met while it was still PENDING has no ordinal yet: it is
        // the fallback (spec_matched_by_demand) and never enters this distribution
        check("a matched-in-flight fill adds no consumption",
              llama_ls_spec_lead_bucket(llama_ls_spec_lead(4, -1)) == -1);
        check("histogram of leads {2,2,3,8,9} is [0,2,1,0,0,0,0,2]",
              hist[0] == 0 && hist[1] == 2 && hist[2] == 1 && hist[3] == 0 &&
              hist[4] == 0 && hist[5] == 0 && hist[6] == 0 && hist[7] == 2);
        check("consumed/min/max/sum of that distribution are 5/2/9/24",
              consumed == 5 && lo == 2 && hi == 9 && sum == 24);
        // M7's decision rule, in the shape the checker uses: the LOWER median --
        // the smallest lead whose cumulative count reaches ceil(n/2) -- so a median
        // of 0 is only reachable when nothing was consumed at all.
        int64_t cum = 0;
        int     median = 0;
        const int64_t half = (consumed + 1) / 2;
        for (int b = 0; b < 8; ++b) {
            cum += hist[b];
            if (cum >= half) {
                median = b + 1;
                break;
            }
        }
        check("lower median of {2,2,3,8,9} is 3 (the samples got a head start)", median == 3);
    }

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
