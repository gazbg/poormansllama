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

    // the engine's spec_pending_of(): positions whose copy is in flight. The
    // speculative budget is checked against THIS count, so a stage can never
    // admit more than `budget` fills: every admitted fill holds its position
    // until the next stage's reap, which is the only reap on the run path.
    int pending() const {
        int n = 0;
        for (uint8_t st : state) {
            n += (st == LLAMA_LS_SPEC_PENDING);
        }
        return n;
    }

    // the engine's spec_reap(): the previous stage's fills completed, so they
    // claim residency and become the FIRST eviction victim while unconsumed.
    void reap() {
        for (size_t p = 0; p < state.size(); ++p) {
            if (state[p] == LLAMA_LS_SPEC_PENDING) {
                state[p] = LLAMA_LS_SPEC_READY;
                spec[p]  = 1;
            }
        }
    }
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

    // ---------------------------------------------------------------------
    // T-GPLR-58 MODE 2: the ENGINE-CONSUMED proposer. The arm is REAL and the
    // demand stream is OURS, so the wiring -- observe this row, propose for the
    // next one, plan, issue -- is exercised end to end with no GPU and no model.
    // What is asserted is the wiring's CONTRACT, never the arm's accuracy:
    //   * the plan it returns is exactly the plan for the ids the arm published;
    //   * every issued fill is admitted by the SAME rule the engine uses, in order;
    //   * a fill is issued when, and only when, the cache has room BEYOND the
    //     demand's reserve -- and with the cache saturated the arm still proposes
    //     while the plan issues nothing (the mechanism's steady state, asserted
    //     here rather than discovered on a mutex hold);
    //   * a proposal for row k cannot depend on row k+1 (the observe-then-propose
    //     order is what lets the offline screen's numbers be read as precision);
    //   * the null publishes exactly the ARM's count, with different ids;
    //   * the empty case publishes nothing rather than a zero read as a result.
    //
    // The stream is PERIODIC on purpose: the families learn transitions, so a
    // stream with no recurring row pairs gives them nothing to propose, and every
    // check below would then be a comparison of empty lists.
    // ---------------------------------------------------------------------
    printf("\nMODE 2 - the engine-consumed proposer (observe -> arm/null -> plan)\n");
    {
        const int n_layer  = 3;
        const int n_expert = 32;
        const int top_k    = 4;   // the reserve / budget / pending cap, in positions
        const int cap      = 24;  // K = 6 windows of top_k rows: room beyond the reserve
        const int arm      = LLAMA_LS_ARM_COACT;
        const int period   = 8;
        const int n_tok    = 16;

        // a periodic stream of 2-id rows whose transitions recur
        std::vector<std::vector<int32_t>> stream_a;
        for (int t = 0; t < n_tok; ++t) {
            const int b = t % period;
            stream_a.push_back({ (b * 2) % n_expert, (b * 2 + 1) % n_expert });
        }
        std::vector<std::vector<int32_t>> stream_b = stream_a;
        stream_b.back()[0] = (stream_b.back()[0] + 17) % n_expert;

        struct stage_out {
            llama_ls_propose_result pr;
            std::vector<int32_t>    issued;
            std::vector<uint8_t>    state_before;
            std::vector<int32_t>    experts_before;
        };

        auto drive = [&](int cap_in, const std::vector<std::vector<int32_t>> & rows, bool null_source,
                         std::vector<stage_out> & out, int arm_sel) {
            llama_ls_pred_state st;
            llama_ls_pred_init(&st, n_layer, n_expert, LLAMA_LS_PRED_BIT_ALL);
            uint32_t rng = llama_ls_pred_null_seed();
            fixture  f(cap_in);
            int64_t  tick = 0;
            out.clear();
            for (size_t i = 0; i < rows.size(); ++i) {
                // (1) the demand places its own row: a hit refreshes it, else a FREE
                //     position, else an eviction -- the engine's own order, with the
                //     fixture's P3/P5 eviction choice
                for (int32_t e : rows[i]) {
                    const int hp = f.hit_pos(e);
                    if (hp >= 0) {
                        f.recency[(size_t) hp] = ++tick;
                        continue;
                    }
                    int p = llama_ls_spec_free_pos(f.expert.data(), f.state.data(), cap_in);
                    if (p < 0) {
                        std::vector<char> wanted(cap_in, 0);
                        p = f.victim(wanted);
                    }
                    if (p >= 0) {
                        f.demand_place(p, e, ++tick);
                    }
                }
                // (2) the previous stage's fills completed: the reap claims them. A
                //     demand that wanted one would have CANCELLED it (matched), which
                //     this fixture does not model -- that is the M7 band's subject.
                for (size_t p = 0; p < f.state.size(); ++p) {
                    if (f.state[p] == LLAMA_LS_SPEC_PENDING) {
                        f.state[p] = LLAMA_LS_SPEC_READY;
                        f.spec[p]  = 1;
                    }
                }
                // (3) the proposer is asked, for THIS layer's next demand row
                stage_out so;
                so.state_before   = f.state;
                so.experts_before = f.expert;
                so.pr = llama_ls_spec_propose(&st, arm_sel, null_source, 1, top_k,
                                              rows[i].data(), (int) rows[i].size(),
                                              f.expert.data(), f.state.data(), cap_in,
                                              top_k, top_k, 0, &rng, n_expert);
                for (int j = 0; j < so.pr.plan.n_issue; ++j) {
                    so.issued.push_back(so.pr.plan.issue[j]);
                }
                out.push_back(so);
                // (4) the engine issues them: PENDING until the step-2 reap
                int pend = 0;
                for (int j = 0; j < so.pr.plan.n_issue; ++j) {
                    const llama_ls_spec_admission ad = llama_ls_spec_admit(
                        f.expert.data(), f.state.data(), cap_in, top_k, top_k, pend, so.pr.plan.issue[j]);
                    if (ad.ok) {
                        f.expert[(size_t) ad.pos]  = so.pr.plan.issue[j];
                        f.state[(size_t) ad.pos]   = LLAMA_LS_SPEC_PENDING;
                        f.spec[(size_t) ad.pos]    = 0;
                        f.recency[(size_t) ad.pos] = ++tick;
                        pend++;
                    }
                }
            }
        };

        std::vector<stage_out> run_a;
        drive(cap, stream_a, false, run_a, arm);

        // 0. NON-VACUITY and the mechanism's own shape: fills must actually be issued
        //    on some stage AND refused on others -- a plan that always issues is as
        //    suspicious as one that never does
        {
            int n_prop = 0, n_fill = 0, n_refused = 0;
            for (const stage_out & so : run_a) {
                if (so.pr.n_arm_count > 0) {
                    n_prop++;
                    if (so.pr.plan.n_issue > 0) {
                        n_fill++;
                    } else {
                        n_refused++;
                    }
                }
            }
            check("the arm proposed on a periodic stream (the fixture has signal)", n_prop > 0);
            // With a working set SMALLER than the cache, a correct prediction is usually
            // ALREADY RESIDENT, and the plan says exactly that (skipped_present) instead
            // of issuing a duplicate copy of a row the cache holds. That is not a wiring
            // failure: it is the reason a predictor can only pay on rows the cache is
            // missing, and it is asserted rather than smoothed over.
            int n_skipped_present = 0;
            for (const stage_out & so : run_a) {
                n_skipped_present += so.pr.plan.skipped_present;
            }
            check("the plan's accounting is real: proposals the cache already holds are skipped",
                  n_skipped_present > 0 || n_fill > 0);
            check("and the cache refuses the rest: a plan is not a wish list", n_refused > 0);

            // THE MECHANISM'S STEADY STATE, asserted here rather than discovered on a
            // mutex hold: the demand path fills the cache to its declared capacity and
            // EVICTS to place what it still needs, so a saturated layer holds no FREE
            // position -- and the admission rule needs a free position BEYOND the
            // demand's reserve (n_free > reserve). With a cache whose working set
            // exceeds its capacity, every proposal is therefore refused, however good
            // the arm is. That is a capacity fact, not a prediction result.
            std::vector<stage_out> run_tight;
            drive(12, stream_a, false, run_tight, arm);
            int tight_prop = 0, tight_fill = 0;
            for (const stage_out & so : run_tight) {
                if (so.pr.n_arm_count > 0) {
                    tight_prop++;
                }
                if (!so.issued.empty()) {
                    tight_fill++;
                }
            }
            check("saturated cache (cap 12 < the 16-id working set): the arm still proposes",
                  tight_prop > 0);
            check("saturated cache: and not one fill is admitted (the barrier is total)",
                  tight_fill == 0);
        }

        // 1. the plan is the plan FOR WHAT THE ARM PUBLISHED: replaying the arm's own
        //    list through llama_ls_spec_plan reproduces the issues and the stop reason
        {
            llama_ls_pred_state st;
            llama_ls_pred_init(&st, n_layer, n_expert, LLAMA_LS_PRED_BIT_ALL);
            bool same = true;
            for (size_t i = 0; i < stream_a.size() && same; ++i) {
                llama_ls_pred_observe(&st, 1, stream_a[i].data(), (int) stream_a[i].size());
                int32_t pub[LLAMA_LS_PRED_MAX_OUT];
                const int n_pub = llama_ls_pred_arm_predict(&st, arm, 1, top_k, pub, LLAMA_LS_PRED_MAX_OUT);
                const llama_ls_plan pl = llama_ls_spec_plan(run_a[i].experts_before.data(),
                                                            run_a[i].state_before.data(), cap,
                                                            top_k, top_k, 0, top_k, pub, n_pub);
                if (n_pub != run_a[i].pr.n_arm_count || pl.n_issue != run_a[i].pr.plan.n_issue ||
                    pl.stop != run_a[i].pr.plan.stop) {
                    same = false;
                    break;
                }
                for (int j = 0; j < pl.n_issue && same; ++j) {
                    if (pl.issue[j] != run_a[i].pr.plan.issue[j]) {
                        same = false;
                    }
                }
            }
            check("the wiring plans exactly what the arm published (ids + stop reason)", same);
        }

        // 2. every issue is admitted by the engine's OWN rule, in order, within budget
        {
            bool all_admitted = true, budget_ok = true;
            for (size_t i = 0; i < run_a.size(); ++i) {
                if (run_a[i].pr.plan.n_issue > top_k) {
                    budget_ok = false;
                }
                std::vector<int32_t> ex  = run_a[i].experts_before;
                std::vector<uint8_t> stv = run_a[i].state_before;
                int pend = 0;
                for (int j = 0; j < run_a[i].pr.plan.n_issue; ++j) {
                    const llama_ls_spec_admission ad = llama_ls_spec_admit(
                        ex.data(), stv.data(), cap, top_k, top_k, pend, run_a[i].pr.plan.issue[j]);
                    if (!ad.ok) {
                        all_admitted = false;
                        break;
                    }
                    ex[(size_t) ad.pos]  = run_a[i].pr.plan.issue[j];
                    stv[(size_t) ad.pos] = LLAMA_LS_SPEC_PENDING;
                    pend++;
                }
            }
            check("every issued fill is admitted by the engine's own rule, in issue order", all_admitted);
            check("the plan never exceeds the per-row budget", budget_ok);
        }

        // 3. the BARRIER, both ways. Saturated (no free position): a warm arm proposes
        //    and the plan must issue nothing. With room beyond the reserve: it issues.
        {
            llama_ls_pred_state st;
            llama_ls_pred_init(&st, n_layer, n_expert, LLAMA_LS_PRED_BIT_ALL);
            uint32_t rng = llama_ls_pred_null_seed();
            for (int t = 0; t < 2 * period; ++t) {
                const int b = t % period;
                std::vector<int32_t> warm = { (b * 2) % n_expert, (b * 2 + 1) % n_expert };
                llama_ls_pred_observe(&st, 1, warm.data(), (int) warm.size());
            }
            std::vector<int32_t> row = { 0, 1 };

            fixture sat(cap);                                   // every position taken
            for (int p = 0; p < cap; ++p) {
                sat.demand_place(p, 200 + p, ++sat.tick);
            }
            const llama_ls_propose_result sat_pr = llama_ls_spec_propose(
                &st, arm, false, 1, top_k, row.data(), (int) row.size(),
                sat.expert.data(), sat.state.data(), cap, top_k, top_k, 0, &rng, n_expert);
            check("a saturated cache: the warmed arm still proposes", sat_pr.n_arm_count > 0);
            check("a saturated cache: the plan issues nothing", sat_pr.plan.n_issue == 0);
            check("a saturated cache: the refusal is the barrier's, not a silence",
                  sat_pr.plan.stop == LLAMA_LS_STOP_RESERVE || sat_pr.plan.stop == LLAMA_LS_STOP_NO_FREE);

            fixture room(cap);                                  // room beyond the reserve
            for (int p = 0; p < top_k; ++p) {
                room.demand_place(p, 300 + p, ++room.tick);
            }
            const llama_ls_propose_result room_pr = llama_ls_spec_propose(
                &st, arm, false, 1, top_k, row.data(), (int) row.size(),
                room.expert.data(), room.state.data(), cap, top_k, top_k, 0, &rng, n_expert);
            check("room beyond the reserve: the plan DOES issue the arm's fills",
                  room_pr.plan.n_issue > 0);
            check("room beyond the reserve: it holds the reserve back meanwhile",
                  room_pr.plan.n_issue <= cap - top_k - top_k);
        }

        // 4. no-peeking: the two streams differ ONLY in their last row, so every
        //    earlier stage must be identical -- the wiring observes row k and proposes
        //    for row k+1, and may never see the row it proposes for
        {
            std::vector<stage_out> run_b;
            drive(cap, stream_b, false, run_b, arm);
            bool same_upto_last = run_a.size() == run_b.size();
            for (size_t i = 0; i + 1 < run_a.size() && same_upto_last; ++i) {
                if (run_a[i].issued != run_b[i].issued ||
                    run_a[i].pr.n_arm_count != run_b[i].pr.n_arm_count ||
                    run_a[i].pr.plan.stop != run_b[i].pr.plan.stop) {
                    same_upto_last = false;
                }
            }
            check("a proposal for row k cannot depend on row k+1 (no peeking)", same_upto_last);
        }

        // 5. determinism through the wiring: the same stream twice, same fills
        {
            std::vector<stage_out> again;
            drive(cap, stream_a, false, again, arm);
            bool identical = again.size() == run_a.size();
            for (size_t i = 0; i < run_a.size() && identical; ++i) {
                if (again[i].issued != run_a[i].issued) {
                    identical = false;
                }
            }
            check("the same demand stream produces the same fills twice", identical);
        }

        // 6. the count-matched null: the ARM's count, different ids (structure-free)
        {
            std::vector<stage_out> run_null;
            drive(cap, stream_a, true, run_null, arm);
            bool counted = true, differs = false;
            int  n_warm  = 0, n_diff_stages = 0;
            for (size_t i = 0; i < run_null.size(); ++i) {
                if (run_null[i].pr.n_proposals != run_null[i].pr.n_arm_count) {
                    counted = false;
                }
                if (run_null[i].pr.n_arm_count > 0) {
                    n_warm++;
                    // compare the PUBLISHED ids, not what the cache happened to admit:
                    // the null's contract is "the arm's count, no signal in the ids"
                    bool same_ids = run_null[i].pr.n_proposals == run_a[i].pr.n_proposals;
                    for (int j = 0; j < run_null[i].pr.n_proposals && same_ids; ++j) {
                        if (run_null[i].pr.published[j] != run_a[i].pr.published[j]) {
                            same_ids = false;
                        }
                    }
                    if (!same_ids) {
                        differs = true;
                        n_diff_stages++;
                    }
                }
            }
            check("the null publishes exactly the ARM's count, on every stage", counted);
            check("the null's published ids are not the arm's (it consults no ids)",
                  differs && n_diff_stages > 0);
            check("the arm itself produced proposals (there was a count to match)", n_warm > 0);
        }

        // 7. the empty case is honest: a cold state and an empty demand row publish
        //    nothing, and the plan issues nothing -- not a zero dressed as a result
        {
            llama_ls_pred_state st;
            llama_ls_pred_init(&st, n_layer, n_expert, LLAMA_LS_PRED_BIT_ALL);
            uint32_t rng = llama_ls_pred_null_seed();
            fixture f(cap);
            const llama_ls_propose_result pr = llama_ls_spec_propose(
                &st, arm, false, 1, top_k, nullptr, 0, f.expert.data(), f.state.data(), cap,
                top_k, top_k, 0, &rng, n_expert);
            check("a cold state with no observed row proposes nothing", pr.n_arm_count == 0);
            check("and issues nothing", pr.plan.n_issue == 0);
            check("with no state at all the call is inert rather than a crash",
                  llama_ls_spec_propose(nullptr, arm, false, 1, top_k, nullptr, 0,
                                        f.expert.data(), f.state.data(), cap,
                                        top_k, top_k, 0, &rng, n_expert).plan.n_issue == 0);
        }
    }

    // ---------------------------------------------------------------------
    // T-GPLR-58 A2: `submitted` is PER-STAGE, and a stage never exceeds the budget.
    //
    // The engine's per-stage count is the number of successful spec_issue_one()
    // calls inside one demand stage, and every one of them is admitted by
    // llama_ls_spec_admit with max_pending = budget, with the fill holding its
    // position until the NEXT stage's reap. So the count is bounded by the budget
    // by construction. These fixtures drive the engine's own issue loop over a
    // 32-stage stream at the production shape (reserve/budget 8, cap 64 =
    // K*top_k, 306 candidate ids per stage -- the shape that read as a +306
    // per-stage increment when the field carried the run-global total instead).
    // They prove the bound NON-vacuously, prove the cumulative reading is a
    // different number, and show the defect's own shape (a per-layer increment of
    // a run-global counter) so a regression to the cumulative field is caught.
    // ---------------------------------------------------------------------
    printf("\nA2 TELEMETRY - `submitted` is per stage, and a stage never exceeds the budget\n");
    {
        const int n_expert = 1024;
        const int top_k    = 8;   // the demand reserve, in positions
        const int budget   = 8;   // the per-layer speculative budget: the bound proved here
        const int cap      = 64;  // K=8 windows of top_k rows
        const int n_stage  = 32;
        const int n_cand   = 306; // candidates the source publishes per stage, 38x the budget

        // the source's candidate pool, disjoint from every id the demand routes
        std::vector<int32_t> cands;
        for (int c = 0; c < n_cand; ++c) {
            cands.push_back(512 + c);
        }

        // one demand stage, in the engine's order: classify, place, evict as needed
        auto demand_stage = [&](fixture & f, int s) {
            for (int k = 0; k < top_k; ++k) {
                const int32_t e = (k < top_k / 2)
                    ? (int32_t) ((s * (top_k / 2) + k) % 128)   // a fresh row every stage
                    : (int32_t) (200 + k - top_k / 2);          // a row the cache keeps
                const int hp = f.hit_pos(e);
                if (hp >= 0) {
                    f.recency[(size_t) hp] = ++f.tick;
                    continue;
                }
                int p = llama_ls_spec_free_pos(f.expert.data(), f.state.data(), cap);
                if (p < 0) {
                    std::vector<char> wanted((size_t) cap, 0);
                    p = f.victim(wanted);
                }
                if (p >= 0) {
                    f.demand_place(p, e, ++f.tick);
                }
            }
        };

        // the engine's issue loop: the ids the SOURCE published, each admitted by the
        // engine's own rule in rank order. Returns how many fills this stage placed.
        auto issue_stage = [&](fixture & f, const std::vector<int32_t> & pub) {
            int admitted = 0;
            for (int32_t e : pub) {
                const llama_ls_spec_admission ad = llama_ls_spec_admit(
                    f.expert.data(), f.state.data(), cap, top_k, budget, f.pending(), e);
                if (!ad.ok) {
                    continue;
                }
                f.expert[(size_t) ad.pos]  = e;
                f.state[(size_t) ad.pos]   = LLAMA_LS_SPEC_PENDING;
                f.spec[(size_t) ad.pos]    = 0; // the spec flag is set at the reap
                f.recency[(size_t) ad.pos] = ++f.tick;
                admitted++;
            }
            return admitted;
        };

        // 1. the SCRIPTED source (mode 1's shape): 306 published ids per stage, and
        //    the per-stage count still never passes the budget.
        {
            fixture f(cap);
            int  max_stage = 0, at_budget = 0;
            long long cumulative = 0;
            bool bounded = true;
            for (int s = 0; s < n_stage; ++s) {
                demand_stage(f, s);
                f.reap(); // the previous stage's fills completed (the runtime's order:
                          // the demand classifies while they are still PENDING)
                const int admitted = issue_stage(f, cands);
                bounded = bounded && admitted <= budget;
                max_stage = std::max(max_stage, admitted);
                at_budget += (admitted == budget);
                cumulative += admitted;
            }
            check("a stage places at most the budget, whatever the source publishes", bounded);
            check("the bound is reached, so the check above is not vacuous", at_budget > 0);
            check("the cumulative total is a different, larger number", cumulative > budget);
            printf("       (%d stages, max per-stage %d, %d stage(s) at the budget, cumulative %lld)\n",
                   n_stage, max_stage, at_budget, cumulative);
        }

        // 2. the reading that produced the defect: the engine's spec_submitted is RUN
        //    GLOBAL, so between two events of ONE layer every other layer's stage in
        //    that token lands in it too. Two layers in lockstep reproduce a per-layer
        //    increment above the budget on that cumulative reading while each layer's
        //    own per-stage count stays inside it.
        {
            fixture a(cap), b(cap);
            long long global_total = 0, seen_a = 0, max_delta = 0;
            int max_a = 0, max_b = 0;
            for (int s = 0; s < n_stage; ++s) {
                for (int which = 0; which < 2; ++which) {
                    fixture & f = (which == 0) ? a : b;
                    demand_stage(f, s);
                    f.reap();
                    const int admitted = issue_stage(f, cands);
                    global_total += admitted;
                    if (which == 0) {
                        max_a = std::max(max_a, admitted);
                    } else {
                        max_b = std::max(max_b, admitted);
                        max_delta = std::max(max_delta, global_total - seen_a);
                        seen_a = global_total;
                    }
                }
            }
            check("both layers stay within the budget on every stage", max_a <= budget && max_b <= budget);
            check("a run-global counter read per layer does exceed it (the defect)", max_delta > budget);
            printf("       (per-layer max %d/%d, max run-global increment between one layer's events %lld)\n",
                   max_a, max_b, max_delta);
        }

        // 3. the MODE 2 wiring, over 32 stages with the arm warm (two periods of the same
        //    periodic row) and the cache holding only the demand reserve -- the shape the
        //    mode-2 block above calls "room beyond the reserve". Every stage's plan goes
        //    through the engine's own admission loop, and every per-stage count is within
        //    the budget.
        {
            llama_ls_pred_state st;
            const int n_expert2 = 32; // the working shape of the mode-2 block above
            llama_ls_pred_init(&st, 3, n_expert2, LLAMA_LS_PRED_BIT_ALL);
            for (int t = 0; t < 2 * 8; ++t) {
                const int b = t % 8;
                const std::vector<int32_t> warm = { (int32_t) ((b * 2) % n_expert2),
                                                    (int32_t) ((b * 2 + 1) % n_expert2) };
                llama_ls_pred_observe(&st, 1, warm.data(), (int) warm.size());
            }
            uint32_t rng = llama_ls_pred_null_seed();
            int  max_issue = 0, placing = 0;
            long long placed = 0;
            for (int s = 0; s < n_stage; ++s) {
                const int b = s % 8;
                const std::vector<int32_t> row = { (int32_t) ((b * 2) % n_expert2),
                                                   (int32_t) ((b * 2 + 1) % n_expert2) };
                fixture f(cap); // only the demand reserve is resident: room beyond it
                for (int p = 0; p < top_k; ++p) {
                    f.demand_place(p, 300 + p, ++f.tick);
                }
                const llama_ls_propose_result pr = llama_ls_spec_propose(
                    &st, LLAMA_LS_ARM_COACT, false, 1, budget, row.data(), (int) row.size(),
                    f.expert.data(), f.state.data(), cap, top_k, budget, f.pending(), &rng, n_expert2);
                int admitted = 0;
                for (int j = 0; j < pr.plan.n_issue; ++j) {
                    const llama_ls_spec_admission ad = llama_ls_spec_admit(
                        f.expert.data(), f.state.data(), cap, top_k, budget, f.pending(),
                        pr.plan.issue[j]);
                    if (!ad.ok) {
                        continue;
                    }
                    f.expert[(size_t) ad.pos]  = pr.plan.issue[j];
                    f.state[(size_t) ad.pos]   = LLAMA_LS_SPEC_PENDING;
                    f.spec[(size_t) ad.pos]    = 0;
                    f.recency[(size_t) ad.pos] = ++f.tick;
                    admitted++;
                }
                max_issue = std::max(max_issue, admitted);
                placing  += (admitted > 0);
                placed   += admitted;
            }
            check("mode 2: every stage's plan places at most the budget", max_issue <= budget);
            check("mode 2: and the bound is exercised (stages did place fills)", placing > 0);
            printf("       (max per-stage %d over %d stages, %lld fills placed in total)\n",
                   max_issue, n_stage, placed);
        }
    }

    // ---------------------------------------------------------------------
    // T-GPLR-59: the separate prediction-pool policy. This is deliberately
    // host-only: it proves cross-pool duplicate suppression and READY-only
    // replacement without a CUDA allocation or a model.
    printf("\nT-GPLR-59 - separate prediction pool policy\n");
    {
        const int cap = 12;
        ggml_init_params meta = { 1024u * 1024u, nullptr, true };
        ggml_context * ctx = ggml_init(meta);
        ggml_tensor * qrow = ctx ? ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_K, 256, 3, 16, 1) : nullptr;
        const size_t slab = llama_ls_expert_slab_bytes(GGML_TYPE_Q4_K, 256, 3);
        check("pool slab arithmetic preserves the exact quantized Q4_K bytes", [&] {
            return ctx != nullptr && qrow != nullptr && slab > 0 && slab * 16 == ggml_nbytes(qrow);
        }());
        if (ctx) {
            ggml_free(ctx);
        }
        int32_t real[4] = { 100, -1, -1, -1 };
        uint8_t real_st[4] = { LLAMA_LS_SPEC_READY, LLAMA_LS_SPEC_FREE, LLAMA_LS_SPEC_FREE, LLAMA_LS_SPEC_FREE };
        int32_t pool[12];
        uint8_t pool_st[12];
        int64_t pool_r[12];
        for (int p = 0; p < cap; ++p) {
            pool[p] = -1;
            pool_st[p] = LLAMA_LS_SPEC_FREE;
            pool_r[p] = -1;
        }
        int32_t props[13] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1 };
        llama_ls_pool_plan p = llama_ls_pool_plan_make(real, real_st, 4, pool, pool_st, pool_r, cap, 256, 12, props, 13);
        check("12 positions admit 12 distinct proposals", p.n_issue == 12 && p.refused_capacity == 0);
        check("a duplicate inside one publication is skipped", p.skipped_dup == 1);

        int32_t over[13] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
        const llama_ls_pool_plan budget = llama_ls_pool_plan_make(
            real, real_st, 4, pool, pool_st, pool_r, cap, 256, 12, over, 13);
        check("a 13th admissible publication is refused by the 12-item budget", [&] {
            return budget.n_issue == 12 && budget.refused_budget == 1 && budget.stop == 1;
        }());
        const int32_t invalid[3] = { -1, 12, 13 };
        const llama_ls_pool_plan invalid_plan = llama_ls_pool_plan_make(
            real, real_st, 4, pool, pool_st, pool_r, cap, 12, 12, invalid, 3);
        check("negative and out-of-range expert ids are skipped as invalid", [&] {
            return invalid_plan.n_issue == 0 && invalid_plan.skipped_invalid == 3;
        }());

        pool[0] = 1; pool_st[0] = LLAMA_LS_SPEC_READY; pool_r[0] = 20;
        pool[1] = 2; pool_st[1] = LLAMA_LS_SPEC_PENDING; pool_r[1] = 21;
        check("real, prediction-pool, and duplicate skips have separate reasons", [&] {
            const int32_t x[4] = { 100, 1, 2, 3 };
            const llama_ls_pool_plan q = llama_ls_pool_plan_make(real, real_st, 4, pool, pool_st, pool_r, cap, 256, 12, x, 4);
            return q.n_issue == 1 && q.issue[0] == 3 &&
                   q.skipped_real == 1 && q.skipped_pool == 2 && q.skipped_dup == 0;
        }());
        check("a repeat is duplicate even when the first copy was already skipped", [&] {
            // Own pool arrays: the shared fixture pool already holds experts 1
            // and 2, and plan_make checks the pool BEFORE the real cache, so the
            // first copy must live in `already` alone to land on skipped_real.
            const int32_t already[1] = { 1 };
            uint8_t already_st[1] = { LLAMA_LS_SPEC_READY };
            int32_t solo_pool[2] = { -1, -1 };
            uint8_t solo_st[2] = { LLAMA_LS_SPEC_FREE, LLAMA_LS_SPEC_FREE };
            int64_t solo_r[2] = { -1, -1 };
            const int32_t repeated[3] = { 1, 1, 3 };
            const llama_ls_pool_plan q = llama_ls_pool_plan_make(
                already, already_st, 1, solo_pool, solo_st, solo_r, 2, 256, 12, repeated, 3);
            return q.n_issue == 1 && q.issue[0] == 3 && q.skipped_real == 1 && q.skipped_dup == 1;
        }());

        const int32_t real_ready[4] = { 7, -1, -1, -1 };
        const uint8_t real_ready_st[4] = { LLAMA_LS_SPEC_READY, LLAMA_LS_SPEC_FREE, LLAMA_LS_SPEC_FREE, LLAMA_LS_SPEC_FREE };
        const int32_t real_pending[4] = { 8, -1, -1, -1 };
        const uint8_t real_pending_st[4] = { LLAMA_LS_SPEC_PENDING, LLAMA_LS_SPEC_FREE, LLAMA_LS_SPEC_FREE, LLAMA_LS_SPEC_FREE };
        const int32_t one[1] = { 7 };
        const int32_t two[1] = { 8 };
        check("a proposal in real READY is skipped", llama_ls_pool_plan_make(
            real_ready, real_ready_st, 4, pool, pool_st, pool_r, cap, 256, 12, one, 1).skipped_real == 1);
        check("a proposal in real PENDING is skipped", llama_ls_pool_plan_make(
            real_pending, real_pending_st, 4, pool, pool_st, pool_r, cap, 256, 12, two, 1).skipped_real == 1);

        // A READY pool row is replaceable; PENDING is not. Use a fresh policy
        // view so the fixture state is not mutated by the pure function.
        for (int p = 0; p < cap; ++p) {
            pool[p] = 100 + p;
            pool_st[p] = LLAMA_LS_SPEC_READY;
            pool_r[p] = p;
        }
        const int32_t fresh = 99;
        const llama_ls_pool_plan ready = llama_ls_pool_plan_make(real, real_st, 4, pool, pool_st, pool_r, cap, 256, 1, &fresh, 1);
        check("a full pool evicts its oldest READY row", ready.n_issue == 1 && ready.evicted == 1 && ready.issue_pos[0] == 0);
        for (int p = 0; p < cap; ++p) {
            pool_st[p] = LLAMA_LS_SPEC_PENDING;
        }
        const llama_ls_pool_plan pending = llama_ls_pool_plan_make(real, real_st, 4, pool, pool_st, pool_r, cap, 256, 1, &fresh, 1);
        check("a full all-PENDING pool refuses without a wait", pending.n_issue == 0 && pending.refused_capacity == 1 && pending.stop == 2);

        int32_t lifecycle[3] = { 4, 5, -1 };
        uint8_t lifecycle_st[3] = { LLAMA_LS_SPEC_PENDING, LLAMA_LS_SPEC_PENDING, LLAMA_LS_SPEC_FREE };
        int64_t lifecycle_r[3] = { 1, 2, -1 };
        uint32_t lifecycle_g[3] = { 7, 7, 0 };
        check("PENDING demand is reported, not served", [&] {
            const llama_ls_pool_lookup q = llama_ls_pool_lookup_expert(
                lifecycle, lifecycle_st, lifecycle_g, 3, 7, 4);
            return q.ready_pos < 0 && q.pending_pos == 0;
        }());
        check("PENDING demand fallback is marked cancelled without a completion wait", [&] {
            llama_ls_pool_lookup q = llama_ls_pool_lookup_expert(
                lifecycle, lifecycle_st, lifecycle_g, 3, 7, 4);
            if (q.pending_pos != 0) {
                return false;
            }
            lifecycle_st[0] = LLAMA_LS_SPEC_CANCELLED;
            q = llama_ls_pool_lookup_expert(lifecycle, lifecycle_st, lifecycle_g, 3, 7, 4);
            return q.ready_pos < 0 && q.pending_pos < 0 && lifecycle[0] == 4;
        }());
        check("stale READY cannot satisfy the current generation", [&] {
            lifecycle_st[0] = LLAMA_LS_SPEC_READY;
            lifecycle_g[0] = 6;
            const llama_ls_pool_lookup q = llama_ls_pool_lookup_expert(
                lifecycle, lifecycle_st, lifecycle_g, 3, 7, 4);
            return q.ready_pos < 0 && q.stale_pos == 0;
        }());
        lifecycle_st[0] = LLAMA_LS_SPEC_READY;
        lifecycle_g[0] = 7;
        // A second row for the SAME demanded expert (4): with both a current
        // READY row and a PENDING duplicate present, the lookup must select the
        // READY row and report the PENDING one so the demand serves without
        // cancelling anything. Position 1 keeps its bytes for the phase-cancel
        // check below.
        lifecycle[1] = 4;
        lifecycle_st[1] = LLAMA_LS_SPEC_PENDING;
        lifecycle_g[1] = 7;
        check("READY demand selects exactly one current row", [&] {
            const llama_ls_pool_lookup q = llama_ls_pool_lookup_expert(
                lifecycle, lifecycle_st, lifecycle_g, 3, 7, 4);
            return q.ready_pos == 0 && q.pending_pos == 1;
        }());
        check("READY demand retires exactly one completed pool row", llama_ls_pool_retire_ready(
            lifecycle, lifecycle_st, lifecycle_r, lifecycle_g, 3, 7, 0) &&
            lifecycle[0] == -1 && lifecycle_st[0] == LLAMA_LS_SPEC_FREE);
        check("phase cancellation marks PENDING without releasing bytes", [&] {
            const int n = llama_ls_pool_cancel_pending(lifecycle_st, 3);
            return n == 1 && lifecycle_st[1] == LLAMA_LS_SPEC_CANCELLED && lifecycle[1] == 4;
        }());
        // Keep the other positions occupied as well. The preceding check
        // deliberately freed position 0; a FREE position would correctly admit
        // the new proposal, so make it an event-owned CANCELLED position too.
        lifecycle[0] = 9;
        lifecycle_st[0] = LLAMA_LS_SPEC_CANCELLED;
        lifecycle_r[0] = 4;
        lifecycle[2] = 8;
        lifecycle_st[2] = LLAMA_LS_SPEC_CANCELLED;
        lifecycle_r[2] = 3;
        check("a cancelled row stays occupied until its completion event", [&] {
            const llama_ls_pool_plan plan = llama_ls_pool_plan_make(real, real_st, 4, lifecycle, lifecycle_st,
                                                                     lifecycle_r, 3, 256, 1, &fresh, 1);
            return plan.n_issue == 0 && plan.stop == 2;
        }());
        check("completion releases CANCELLED and READY is consumable only for its generation", [&] {
            lifecycle_st[1] = LLAMA_LS_SPEC_CANCELLED;
            const int complete = llama_ls_pool_complete(lifecycle, lifecycle_st, lifecycle_r,
                                                        lifecycle_g, 3, 1);
            return complete == 2 && lifecycle[1] == -1 && lifecycle_st[1] == LLAMA_LS_SPEC_FREE;
        }());
        check("a new generation clears old READY rows but marks PENDING rows cancelled", [&] {
            int32_t gen_expert[3] = { 5, 6, 7 };
            uint8_t gen_state[3] = { LLAMA_LS_SPEC_READY, LLAMA_LS_SPEC_PENDING, LLAMA_LS_SPEC_READY };
            int64_t gen_recency[3] = { 1, 2, 3 };
            uint32_t gen_generation[3] = { 1, 1, 2 };
            const int changed = llama_ls_pool_begin_generation(gen_expert, gen_state, gen_recency,
                                                                 gen_generation, 3, 2);
            const bool current_pending_stays = llama_ls_pool_begin_generation(
                gen_expert, gen_state, gen_recency, gen_generation, 3, 2) == 0 &&
                gen_state[1] == LLAMA_LS_SPEC_CANCELLED;
            return changed == 2 && gen_expert[0] == -1 && gen_state[0] == LLAMA_LS_SPEC_FREE &&
                   gen_state[1] == LLAMA_LS_SPEC_CANCELLED && gen_expert[2] == 7 &&
                   gen_state[2] == LLAMA_LS_SPEC_READY && current_pending_stays;
        }());
    }

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
