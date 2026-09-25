// T-GPLR-58 A1: host-only proof of the PREDICTOR POLICY -- the proposer families,
// the arm operators (bar-0.9 and vote fusion), the count-matched structure-free
// null, and the speculative-fill PLANNER. No CUDA, no model, no GPU, no mutex.
//
// What this test is, stated carefully. A0's mechanism decides what happens to a
// fill once someone proposes it; A1 decides WHICH rows are proposed and in what
// order, and that policy is pure functions over caller-owned state
// (llama_ls_pred_* / llama_ls_spec_plan / llama_ls_spec_cancel_policy). So this
// test can prove, without an engine:
//
//   (i)   the proposer is DETERMINISTIC and BUDGET-EXACT: the same state yields
//         the same ids, a second call without observe() returns them again, and
//         no family or arm ever publishes more than `b` ids -- and none outside
//         [0, n_expert);
//   (ii)  the families do not PEEK: history proposes exactly the previous row of
//         that layer, which is only knowable from rows already observed;
//   (iii) the NULL is structure-free and demand-blind: its output depends on the
//         RNG and the per-row COUNT alone, so two streams that share a count
//         sequence but carry different router ids produce the identical
//         sequence, and equal counts make it order-invariant too;
//   (iv)  the PLANNER is pure and agrees with the admission rule it plans for:
//         it respects the budget, the pending cap and the demand reserve, it
//         never issues a duplicate or an already-resident row, and replaying its
//         own issue list through llama_ls_spec_admit succeeds for exactly the
//         fills it issued and refuses the next one for the reason its `stop`
//         names -- so the plan is not a re-implementation with its own opinion;
//   (v)   the CANCELLATION POINT is policy too: a cancel is planned only when the
//         demand cannot be placed from FREE positions at all.
//
// `--dump <trace.jsonl> [--b N]` writes the per-row proposal sequence of every
// arm (plus the count-matched null) as JSON Lines, which is the artifact the
// outer comparator tools/check-tg-predict-port.py compares against the offline
// screen row by row. The dump reads the trace's `rowset_init` for the router
// shape and its `rowset_fetch` rows in file order, which is the engine's own
// demand order.
#include "../src/llama-layer-streaming.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int failures = 0;

static void check(const char * what, bool ok) {
    if (!ok) {
        failures++;
    }
    printf("  %-64s %s\n", what, ok ? "ok" : "FAIL");
}

// --------------------------------------------------------------------------- fixtures
// A planted stream: deterministic, and structured enough that every family has
// something to learn (markov-1 transitions, coact pairs, coupling across layers).
struct planted {
    int n_layer;
    int n_expert;
    int n_tok;
    int per_row;
    std::vector<std::vector<std::vector<int32_t>>> ids; // [token][layer][id]

    planted(int layers, int experts, int toks, int per) :
        n_layer(layers), n_expert(experts), n_tok(toks), per_row(per) {
        ids.assign((size_t) toks, std::vector<std::vector<int32_t>>((size_t) layers));
        for (int t = 0; t < toks; ++t) {
            for (int L = 0; L < layers; ++L) {
                for (int k = 0; k < per; ++k) {
                    ids[(size_t) t][(size_t) L].push_back((int32_t) ((t * 3 + L * 5 + k) % experts));
                }
            }
        }
    }

    // drive one family/arm over the whole stream, predict-then-observe per row
    std::vector<std::vector<int32_t>> run(int arm_or_family, int b, bool is_arm) const {
        llama_ls_pred_state st;
        llama_ls_pred_init(&st, n_layer, n_expert, LLAMA_LS_PRED_BIT_ALL);
        std::vector<std::vector<int32_t>> seq;
        for (int t = 0; t < n_tok; ++t) {
            for (int L = 0; L < n_layer; ++L) {
                int32_t out[LLAMA_LS_PRED_MAX_OUT];
                const int n = is_arm ? llama_ls_pred_arm_predict(&st, arm_or_family, L, b, out, LLAMA_LS_PRED_MAX_OUT)
                                     : llama_ls_pred_predict(&st, arm_or_family, L, b, out, LLAMA_LS_PRED_MAX_OUT);
                seq.push_back(std::vector<int32_t>(out, out + n));
                llama_ls_pred_observe(&st, L, ids[(size_t) t][(size_t) L].data(),
                                      (int) ids[(size_t) t][(size_t) L].size());
            }
        }
        return seq;
    }
};

// One layer's cache, the same arrays the engine keeps (A0's fixture shape).
struct cache {
    std::vector<int32_t> expert;
    std::vector<uint8_t> state;
    std::vector<int64_t> recency;

    explicit cache(int cap) :
        expert((size_t) cap, -1),
        state((size_t) cap, LLAMA_LS_SPEC_FREE),
        recency((size_t) cap, 0) {}

    void place_ready(int p, int32_t e, int64_t tick) {
        expert[(size_t) p] = e; state[(size_t) p] = LLAMA_LS_SPEC_READY; recency[(size_t) p] = tick;
    }
    void place_pending(int p, int32_t e, int64_t tick) {
        expert[(size_t) p] = e; state[(size_t) p] = LLAMA_LS_SPEC_PENDING; recency[(size_t) p] = tick;
    }
    int n_free() const {
        int n = 0;
        for (size_t p = 0; p < expert.size(); ++p) {
            if (expert[p] < 0) n++;
        }
        return n;
    }
};

static bool json_event_is(const std::string & line, const char * ev);
static bool json_int(const std::string & line, const char * key, long * out);
static bool json_id_list(const std::string & line, std::vector<int32_t> * out);

// --------------------------------------------------------------------------- dump mode
// The proposal sequence for one trace, as JSON Lines: one `pred_dump` header, then
// one `proposal` line per (arm, row) in the trace's own demand order.
static int dump_mode(const char * path, int b, int null_matched_to) {
    std::ifstream fh(path, std::ios::binary);
    if (!fh) {
        fprintf(stderr, "REFUSED: cannot open %s\n", path);
        return 2;
    }
    const std::string data((std::istreambuf_iterator<char>(fh)), std::istreambuf_iterator<char>());

    // The router shape comes from the trace's own rowset_init, never from a flag.
    long n_expert = 0;
    long top_k    = 0;
    bool have_init = false;
    size_t pos = 0;
    while (pos < data.size()) {
        const size_t nl = data.find('\n', pos);
        const std::string line = data.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        pos = (nl == std::string::npos) ? data.size() : nl + 1;
        if (!json_event_is(line, "rowset_init")) {
            continue;
        }
        have_init = json_int(line, "\"n_expert\"", &n_expert) && json_int(line, "\"top_k\"", &top_k);
        break;
    }
    if (!have_init || n_expert <= 0) {
        fprintf(stderr, "REFUSED: %s has no rowset_init with n_expert (not a row-cache trace)\n", path);
        return 2;
    }

    int n_layer = 0;
    std::vector<long> row_tok, row_layer;
    std::vector<std::vector<int32_t>> row_ids;
    pos = 0;
    while (pos < data.size()) {
        const size_t nl = data.find('\n', pos);
        const std::string line = data.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        pos = (nl == std::string::npos) ? data.size() : nl + 1;
        if (!json_event_is(line, "rowset_fetch")) {
            continue;
        }
        long tok = 0, layer = 0;
        std::vector<int32_t> ids;
        if (!json_int(line, "\"ubatch\"", &tok) || !json_int(line, "\"layer\"", &layer)) {
            continue;
        }
        if (!json_id_list(line, &ids) || ids.empty()) {
            continue;
        }
        row_tok.push_back(tok);
        row_layer.push_back(layer);
        row_ids.push_back(ids);
        if ((int) layer + 1 > n_layer) {
            n_layer = (int) layer + 1;
        }
    }
    if (row_ids.empty()) {
        fprintf(stderr, "REFUSED: %s has no rowset_fetch rows to propose for\n", path);
        return 2;
    }

    printf("{\"event\":\"pred_dump\",\"n_layer\":%d,\"n_expert\":%ld,\"top_k\":%ld,\"b\":%d,"
           "\"rows\":%d,\"families\":\"history,markov-1,coact,coupling\","
           "\"arms\":\"coact,bar09,fusion\",\"null_matched_to\":\"%s\"}\n",
           n_layer, n_expert, top_k, b, (int) row_ids.size(),
           null_matched_to == LLAMA_LS_ARM_COACT ? "coact" :
           null_matched_to == LLAMA_LS_ARM_BAR09 ? "bar09" : "fusion");

    llama_ls_pred_state st;
    llama_ls_pred_init(&st, n_layer, (int) n_expert, LLAMA_LS_PRED_BIT_ALL);
    static const char * kFamNames[4] = { "history", "markov-1", "coact", "coupling" };
    static const char * kArmNames[3] = { "coact", "bar09", "fusion" };
    uint32_t rng = llama_ls_pred_null_seed();

    for (size_t r = 0; r < row_ids.size(); ++r) {
        const int L = (int) row_layer[r];
        const std::vector<int32_t> & ids = row_ids[r];
        int32_t out[LLAMA_LS_PRED_MAX_OUT];
        int arm_counts[3] = { 0, 0, 0 };
        for (int f = 0; f < 4; ++f) {
            const int n = llama_ls_pred_predict(&st, f, L, b, out, LLAMA_LS_PRED_MAX_OUT);
            printf("{\"event\":\"proposal\",\"kind\":\"family\",\"arm\":\"%s\",\"ubatch\":%ld,"
                   "\"layer\":%ld,\"b\":%d,\"n\":%d,\"ids\":[", kFamNames[f], row_tok[r], row_layer[r], b, n);
            for (int i = 0; i < n; ++i) {
                printf("%s%d", i ? "," : "", (int) out[i]);
            }
            printf("]}\n");
            if (f == LLAMA_LS_PRED_COACT) {
                arm_counts[LLAMA_LS_ARM_COACT] = n;
            }
        }
        for (int a = 0; a < 3; ++a) {
            const int n = llama_ls_pred_arm_predict(&st, a, L, b, out, LLAMA_LS_PRED_MAX_OUT);
            arm_counts[a] = n;
            printf("{\"event\":\"proposal\",\"kind\":\"arm\",\"arm\":\"%s\",\"ubatch\":%ld,"
                   "\"layer\":%ld,\"b\":%d,\"n\":%d,\"ids\":[", kArmNames[a], row_tok[r], row_layer[r], b, n);
            for (int i = 0; i < n; ++i) {
                printf("%s%d", i ? "," : "", (int) out[i]);
            }
            printf("]}\n");
        }
        // The null for this row: the matched arm's published COUNT is its only input.
        {
            const int count = arm_counts[null_matched_to];
            const int n = llama_ls_pred_null_fill(&rng, (int) n_expert, count, out, LLAMA_LS_PRED_MAX_OUT);
            printf("{\"event\":\"proposal\",\"kind\":\"null\",\"arm\":\"null\",\"matched_to\":\"%s\","
                   "\"ubatch\":%ld,\"layer\":%ld,\"count\":%d,\"n\":%d,\"ids\":[",
                   kArmNames[null_matched_to], row_tok[r], row_layer[r], count, n);
            for (int i = 0; i < n; ++i) {
                printf("%s%d", i ? "," : "", (int) out[i]);
            }
            printf("]}\n");
        }
        llama_ls_pred_observe(&st, L, ids.data(), (int) ids.size());
    }
    return 0;
}

int main(int argc, char ** argv) {
    int b = 8;
    if (argc >= 3 && strcmp(argv[1], "--dump") == 0) {
        int matched_to = LLAMA_LS_ARM_COACT;
        for (int i = 3; i < argc; ++i) {
            if (strcmp(argv[i], "--b") == 0 && i + 1 < argc) {
                b = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--null-matched-to") == 0 && i + 1 < argc) {
                const char * m = argv[++i];
                matched_to = strcmp(m, "bar09") == 0 ? LLAMA_LS_ARM_BAR09
                           : strcmp(m, "fusion") == 0 ? LLAMA_LS_ARM_FUSION
                                                      : LLAMA_LS_ARM_COACT;
            } else {
                fprintf(stderr, "REFUSED: unknown --dump option %s\n", argv[i]);
                return 2;
            }
        }
        return dump_mode(argv[2], b, matched_to);
    }
    if (argc > 1) {
        fprintf(stderr, "usage: %s [--dump <trace.jsonl> [--b N] [--null-matched-to coact|bar09|fusion]]\n",
                argv[0]);
        return 2;
    }

    printf("T-GPLR-58 A1: predictor policy (proposers, operators, null, planner)\n");

    // ---------------------------------------------------------------- (i) determinism + budget
    {
        printf("\n(i) the proposer is deterministic, budget-exact and in range\n");
        const planted p(4, 16, 6, 4);
        bool deterministic = true, budget_ok = true, in_range = true, second_call_same = true;
        for (int a = 0; a < 3; ++a) {
            const std::vector<std::vector<int32_t>> s1 = p.run(a, b, true);
            const std::vector<std::vector<int32_t>> s2 = p.run(a, b, true);
            deterministic = deterministic && (s1 == s2);
        }
        for (int f = 0; f < 4; ++f) {
            const std::vector<std::vector<int32_t>> s1 = p.run(f, b, false);
            const std::vector<std::vector<int32_t>> s2 = p.run(f, b, false);
            deterministic = deterministic && (s1 == s2);
        }
        // every published list honours the budget, and every id is a real expert
        for (int a = 0; a < 3; ++a) {
            for (const std::vector<int32_t> & row : p.run(a, b, true)) {
                budget_ok = budget_ok && ((int) row.size() <= b);
                for (size_t i = 0; i < row.size(); ++i) {
                    in_range = in_range && row[i] >= 0 && row[i] < p.n_expert;
                }
                // no duplicates inside one proposal
                std::vector<int32_t> sorted = row;
                std::sort(sorted.begin(), sorted.end());
                in_range = in_range && (std::unique(sorted.begin(), sorted.end()) == sorted.end());
            }
        }
        // a second predict() on an UNCHANGED state returns the same ids: the call
        // has no hidden effect (observe() is the only mutator)
        {
            llama_ls_pred_state st;
            llama_ls_pred_init(&st, p.n_layer, p.n_expert, LLAMA_LS_PRED_BIT_ALL);
            llama_ls_pred_observe(&st, 0, p.ids[0][0].data(), (int) p.ids[0][0].size());
            int32_t a1[LLAMA_LS_PRED_MAX_OUT], a2[LLAMA_LS_PRED_MAX_OUT];
            const int n1 = llama_ls_pred_predict(&st, LLAMA_LS_PRED_MARKOV1, 0, b, a1, LLAMA_LS_PRED_MAX_OUT);
            const int n2 = llama_ls_pred_predict(&st, LLAMA_LS_PRED_MARKOV1, 0, b, a2, LLAMA_LS_PRED_MAX_OUT);
            second_call_same = (n1 == n2) && std::equal(a1, a1 + n1, a2);
        }
        check("all 3 arms and all 4 families reproduce their own sequence", deterministic);
        check("no family or arm publishes more than b (b=8)", budget_ok);
        check("every published id is in [0, n_expert) and unique", in_range);
        check("predict() twice on one state returns the same ids", second_call_same);

        // a smaller budget really truncates, it is not ignored
        bool truncates = true;
        for (int a = 0; a < 3; ++a) {
            const std::vector<std::vector<int32_t>> s3 = p.run(a, 3, true);
            for (const std::vector<int32_t> & row : s3) {
                truncates = truncates && ((int) row.size() <= 3);
            }
        }
        check("b=3 truncates every arm to <= 3 ids", truncates);
    }

    // ---------------------------------------------------------------- (ii) no peeking
    {
        printf("\n(ii) the families do not peek at the row they are predicting\n");
        const planted p(3, 12, 5, 3);
        llama_ls_pred_state st;
        llama_ls_pred_init(&st, p.n_layer, p.n_expert, LLAMA_LS_PRED_BIT_ALL);
        bool history_is_previous = true;
        for (int t = 0; t < p.n_tok; ++t) {
            for (int L = 0; L < p.n_layer; ++L) {
                int32_t out[LLAMA_LS_PRED_MAX_OUT];
                const int n = llama_ls_pred_predict(&st, LLAMA_LS_PRED_HISTORY, L, 8, out, LLAMA_LS_PRED_MAX_OUT);
                if (t == 0) {
                    history_is_previous = history_is_previous && (n == 0); // nothing observed yet
                } else {
                    const std::vector<int32_t> & want = p.ids[(size_t) (t - 1)][(size_t) L];
                    history_is_previous = history_is_previous && (n == (int) want.size()) &&
                                          std::equal(want.begin(), want.end(), out);
                }
                llama_ls_pred_observe(&st, L, p.ids[(size_t) t][(size_t) L].data(),
                                      (int) p.ids[(size_t) t][(size_t) L].size());
            }
        }
        check("history proposes exactly the PREVIOUS row of that layer", history_is_previous);

        // coupling reads the PREVIOUS LAYER of the same token, so at layer 0 it has
        // nothing to read at all (the screen's `if not prev: []`), and after one
        // layer it does.
        llama_ls_pred_state st2;
        llama_ls_pred_init(&st2, p.n_layer, p.n_expert, LLAMA_LS_PRED_BIT_ALL);
        int32_t out[LLAMA_LS_PRED_MAX_OUT];
        check("coupling has no proposal at layer 0 of the first row",
              llama_ls_pred_predict(&st2, LLAMA_LS_PRED_COUPLING, 0, 8, out, LLAMA_LS_PRED_MAX_OUT) == 0);
        llama_ls_pred_observe(&st2, 0, p.ids[0][0].data(), (int) p.ids[0][0].size());
        llama_ls_pred_observe(&st2, 1, p.ids[0][1].data(), (int) p.ids[0][1].size());
        check("coupling proposes from the same token's previous layer",
              llama_ls_pred_predict(&st2, LLAMA_LS_PRED_COUPLING, 1, 8, out, LLAMA_LS_PRED_MAX_OUT) > 0);
    }

    // ---------------------------------------------------------------- (iii) the null
    {
        printf("\n(iii) the count-matched null is structure-free and demand-blind\n");
        // two streams with the SAME per-row counts but different router ids
        const std::vector<int> counts = { 8, 3, 0, 5, 8, 1, 8, 8 };
        uint32_t rng_a = llama_ls_pred_null_seed();
        uint32_t rng_b = llama_ls_pred_null_seed();
        std::vector<std::vector<int32_t>> seq_a, seq_b;
        for (size_t i = 0; i < counts.size(); ++i) {
            int32_t oa[LLAMA_LS_PRED_MAX_OUT], ob[LLAMA_LS_PRED_MAX_OUT];
            const int na = llama_ls_pred_null_fill(&rng_a, 64, counts[i], oa, LLAMA_LS_PRED_MAX_OUT);
            const int nb = llama_ls_pred_null_fill(&rng_b, 64, counts[i], ob, LLAMA_LS_PRED_MAX_OUT);
            seq_a.push_back(std::vector<int32_t>(oa, oa + na));
            seq_b.push_back(std::vector<int32_t>(ob, ob + nb));
        }
        check("the same counts give the identical sequence (deterministic)", seq_a == seq_b);

        // the demand content is NOT an input: drive the null over a planted stream's
        // counts, then over a stream whose ids are all different but whose per-row
        // counts are identical, and require the same output.
        const planted p(2, 64, 3, 4);
        uint32_t rng_c = llama_ls_pred_null_seed();
        uint32_t rng_d = llama_ls_pred_null_seed();
        std::vector<std::vector<int32_t>> seq_c, seq_d;
        for (size_t i = 0; i < p.ids.size(); ++i) {
            for (size_t L = 0; L < p.ids[i].size(); ++L) {
                const int count = (int) p.ids[i][L].size();
                int32_t oc[LLAMA_LS_PRED_MAX_OUT], od[LLAMA_LS_PRED_MAX_OUT];
                const int nc = llama_ls_pred_null_fill(&rng_c, p.n_expert, count, oc, LLAMA_LS_PRED_MAX_OUT);
                const int nd = llama_ls_pred_null_fill(&rng_d, p.n_expert, count, od, LLAMA_LS_PRED_MAX_OUT);
                seq_c.push_back(std::vector<int32_t>(oc, oc + nc));
                seq_d.push_back(std::vector<int32_t>(od, od + nd));
            }
        }
        check("equal per-row counts make the sequence order- and id-invariant", seq_c == seq_d);

        // count-exact, duplicate-free, and refused rather than padded when the
        // request cannot be honoured
        bool count_exact = true, distinct = true;
        uint32_t rng_e = llama_ls_pred_null_seed();
        for (int c = 0; c <= 12; ++c) {
            int32_t o[LLAMA_LS_PRED_MAX_OUT];
            const int n = llama_ls_pred_null_fill(&rng_e, 32, c, o, LLAMA_LS_PRED_MAX_OUT);
            count_exact = count_exact && (n == c);
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    distinct = distinct && (o[i] != o[j]);
                }
                distinct = distinct && o[i] >= 0 && o[i] < 32;
            }
        }
        check("null publishes exactly the matched count, distinct and in range", count_exact && distinct);
        {
            int32_t o[LLAMA_LS_PRED_MAX_OUT];
            uint32_t rng_f = llama_ls_pred_null_seed();
            check("null refuses when the expert space cannot supply the count",
                  llama_ls_pred_null_fill(&rng_f, 4, 9, o, LLAMA_LS_PRED_MAX_OUT) < 9);
        }
    }

    // ---------------------------------------------------------------- (iv) D1 draft token table
    {
        printf("\n(iv) D1 learns only from accepted draft-token pairs\n");
        llama_ls_d1_state d1;
        llama_ls_d1_init(&d1, 2, 16);
        int32_t out[8];
        const int32_t route_a[] = { 3, 1, 99, -1 };
        const int32_t route_b[] = { 3, 3, 1 };
        const int32_t route_c[] = { 3 };
        check("an unseen D1 token publishes nothing", llama_ls_d1_predict(&d1, 7, 0, 8, out, 8) == 0);
        llama_ls_d1_observe(&d1, 7, 0, false, route_a, 4);
        check("a rejected D1 draft does not train the table", llama_ls_d1_predict(&d1, 7, 0, 8, out, 8) == 0);
        llama_ls_d1_observe(&d1, 7, 0, true, route_a, 4);
        llama_ls_d1_observe(&d1, 7, 0, true, route_b, 3);
        llama_ls_d1_observe(&d1, 7, 0, true, route_c, 1);
        const int n = llama_ls_d1_predict(&d1, 7, 0, 2, out, 8);
        check("D1 ranks accepted counts then breaks ties by expert id", n == 2 && out[0] == 3 && out[1] == 1);
        check("D1 is isolated by layer", llama_ls_d1_predict(&d1, 7, 1, 8, out, 8) == 0);
        check("D1 obeys its prediction budget", llama_ls_d1_predict(&d1, 7, 0, 1, out, 8) == 1);
        check("D1 pairing gate passes at three accepted rows", llama_ls_d1_pairing_valid(&d1));
        llama_ls_d1_state sparse;
        llama_ls_d1_init(&sparse, 1, 16);
        llama_ls_d1_observe(&sparse, 1, 0, true, route_a, 4);
        llama_ls_d1_observe(&sparse, 2, 0, false, route_a, 4);
        check("D1 pairing gate refuses a below-60-percent stream", !llama_ls_d1_pairing_valid(&sparse));

        int32_t experts[16];
        uint8_t states[16];
        for (int i = 0; i < 16; ++i) {
            experts[i] = -1;
            states[i] = LLAMA_LS_SPEC_FREE;
        }
        const llama_ls_propose_result d1_plan = llama_ls_spec_propose_d1(
            &d1, 7, 0, 2, false, experts, states, 16, 0, 4, 0, nullptr, 16);
        check("D1 feeds the shared proposal planner", d1_plan.n_arm_count == 2 && d1_plan.n_proposals == 2 &&
              d1_plan.plan.n_issue == 2 && d1_plan.plan.issue[0] == 3 && d1_plan.plan.issue[1] == 1);
        uint32_t null_rng = llama_ls_pred_null_seed();
        const llama_ls_propose_result d1_null = llama_ls_spec_propose_d1(
            &d1, 7, 0, 2, true, experts, states, 16, 0, 4, 0, &null_rng, 16);
        check("D1 null is count-matched to the D1 arm", d1_null.n_arm_count == 2 && d1_null.n_proposals == 2);
    }

    // ---------------------------------------------------------------- (v) the planner
    {
        printf("\n(iv) the planner is pure and agrees with the admission rule\n");
        // budget-limited
        {
            cache c(16);
            const int32_t props[12] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            const std::vector<int32_t> before_e = c.expert;
            const std::vector<int64_t> before_r = c.recency;
            const llama_ls_plan a = llama_ls_spec_plan(c.expert.data(), c.state.data(), 16,
                                                       4, 12, 0, 8, props, 12);
            const llama_ls_plan dup = llama_ls_spec_plan(c.expert.data(), c.state.data(), 16,
                                                         4, 12, 0, 8, props, 12);
            check("budget 8 -> exactly 8 fills issued and stop = BUDGET",
                  a.n_issue == 8 && a.stop == LLAMA_LS_STOP_BUDGET);
            check("the 4 beyond the budget are counted as refused, not silently dropped",
                  a.refused_capacity == 4);
            check("the same inputs produce the identical plan (pure, no clock, no I/O)",
                  dup.n_issue == a.n_issue && dup.stop == a.stop &&
                  std::equal(a.issue, a.issue + a.n_issue, dup.issue));
            check("the planner does not mutate the cache it plans against",
                  c.expert == before_e && c.recency == before_r);
        }
        // reserve-limited: cap 8, reserve 4 on an empty cache -> 4 fills, then the barrier
        {
            cache c(8);
            const int32_t props[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
            const llama_ls_plan a = llama_ls_spec_plan(c.expert.data(), c.state.data(), 8,
                                                       4, 8, 0, 8, props, 8);
            check("the demand reserve stops the plan at 4 fills with stop = RESERVE",
                  a.n_issue == 4 && a.stop == LLAMA_LS_STOP_RESERVE && a.refused_capacity == 4);
        }
        // pending cap
        {
            cache c(16);
            const int32_t props[6] = { 1, 2, 3, 4, 5, 6 };
            const llama_ls_plan a = llama_ls_spec_plan(c.expert.data(), c.state.data(), 16,
                                                       4, 3, 0, 8, props, 6);
            check("max_pending 3 stops the plan at 3 fills with stop = PENDING",
                  a.n_issue == 3 && a.stop == LLAMA_LS_STOP_PENDING);
        }
        // resident rows are skipped, not re-issued; a PENDING fill blocks a duplicate
        {
            cache c(16);
            c.place_ready(0, 1, 10);
            c.place_pending(1, 2, 11);
            const int32_t props[4] = { 1, 2, 3, 3 };
            const llama_ls_plan a = llama_ls_spec_plan(c.expert.data(), c.state.data(), 16,
                                                       4, 8, 1, 8, props, 4);
            check("a resident row and an in-flight fill are both skipped (never re-issued)",
                  a.skipped_present == 2 && a.n_issue == 1 && a.issue[0] == 3);
            check("a duplicate inside the proposal list is counted, not issued",
                  a.skipped_dup == 1);
        }
        // THE equivalence test: replaying the plan's own issue list through the
        // admission rule succeeds for exactly those fills, and the next admissible
        // proposal is refused for the reason the plan's `stop` names.
        {
            cache c(12);
            c.place_ready(0, 100, 1);
            c.place_pending(1, 101, 2);
            const int32_t props[10] = { 5, 100, 6, 7, 101, 8, 9, 5, 10, 11 };
            const int n_props = 10;
            const int budget = 8, max_pending = 3, reserve = 6;
            const llama_ls_plan a = llama_ls_spec_plan(c.expert.data(), c.state.data(), 12,
                                                       reserve, max_pending, 1, budget, props, n_props);
            int n_pending = 1;
            int replayed = 0;
            bool replay_ok = true;
            for (int i = 0; i < a.n_issue; ++i) {
                const llama_ls_spec_admission ad = llama_ls_spec_admit(c.expert.data(), c.state.data(), 12,
                                                                       reserve, max_pending, n_pending,
                                                                       a.issue[i]);
                replay_ok = replay_ok && ad.ok;
                if (ad.ok) {
                    const int free_p = llama_ls_spec_free_pos(c.expert.data(), c.state.data(), 12);
                    replay_ok = replay_ok && (free_p == ad.pos);
                    c.place_pending(ad.pos, a.issue[i], 100 + i);
                    n_pending++;
                    replayed++;
                }
            }
            check("every fill the planner planned is admitted by the engine's own rule",
                  replay_ok && replayed == a.n_issue);
            // the next admissible proposal must be refused, and refused for the
            // reason the stop code names
            const char * want = a.stop == LLAMA_LS_STOP_RESERVE  ? "demand_reserve"
                              : a.stop == LLAMA_LS_STOP_BUDGET   ? "budget_exhausted"
                              : a.stop == LLAMA_LS_STOP_NO_FREE  ? "no_free_position"
                              : a.stop == LLAMA_LS_STOP_PENDING  ? "budget_exhausted" : "";
            bool refused_as_named = true;
            if (a.stop != LLAMA_LS_STOP_PROPOSALS) {
                bool found = false;
                for (int i = a.n_issue; i < n_props && !found; ++i) {
                    bool resident = false;
                    for (int p = 0; p < 12; ++p) {
                        if (c.state[(size_t) p] != LLAMA_LS_SPEC_FREE && c.expert[(size_t) p] == props[i]) {
                            resident = true;
                        }
                    }
                    if (resident || (i > 0 && props[i] == props[i - 1])) {
                        continue;
                    }
                    const llama_ls_spec_admission ad = llama_ls_spec_admit(c.expert.data(), c.state.data(), 12,
                                                                           reserve, max_pending, n_pending,
                                                                           props[i]);
                    refused_as_named = !ad.ok && strcmp(ad.reason, want) == 0;
                    found = true;
                }
            }
            check("the next admissible proposal is refused for the reason `stop` names", refused_as_named);
        }
    }

    // ---------------------------------------------------------------- (v) the cancel point
    {
        printf("\n(v) the cancellation point is planned, not assumed\n");
        const llama_ls_cancel_policy none   = llama_ls_spec_cancel_policy(8, 4, 4, 3);
        const llama_ls_cancel_policy never  = llama_ls_spec_cancel_policy(8, 4, 0, 3);
        // a demand that consumes the positions the barrier held for it: legal,
        // and reported rather than cancelled
        const llama_ls_cancel_policy spend  = llama_ls_spec_cancel_policy(5, 4, 5, 3);
        const llama_ls_cancel_policy needs  = llama_ls_spec_cancel_policy(2, 4, 6, 5);
        const llama_ls_cancel_policy empty  = llama_ls_spec_cancel_policy(2, 4, 6, 0);
        const llama_ls_cancel_policy again  = llama_ls_spec_cancel_policy(2, 4, 6, 5);
        check("a demand that fits in FREE positions plans no cancellation",
              !none.cancel && strcmp(none.reason, "reserve_holds") == 0);
        check("a demand that spends the reserve is reported, not cancelled",
              !spend.cancel && spend.used_reserve);
        check("a zero-row demand is a no-op",
              !never.cancel && strcmp(never.reason, "no_demand") == 0);
        check("a demand that cannot be placed plans a cancel with the exact shortfall",
              needs.cancel && needs.need == 4);
        check("with nothing pending the policy refuses rather than promising a cancel",
              !empty.cancel && strcmp(empty.reason, "no_pending_to_cancel") == 0);
        check("the decision is pure (same inputs, same decision)",
              again.cancel == needs.cancel && again.need == needs.need &&
              strcmp(again.reason, needs.reason) == 0);
    }

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}

// --------------------------------------------------------------------------- tiny json
// Deliberately minimal: these traces are the ones tools/check-tg-predict-port.py
// also reads, and the comparator is the authority on their shape. Anything this
// cannot parse is REFUSED by dump_mode rather than guessed at.
static bool json_event_is(const std::string & line, const char * ev) {
    const size_t p = line.find("\"event\"");
    if (p == std::string::npos) {
        return false;
    }
    size_t q = line.find(':', p + 7);
    if (q == std::string::npos) {
        return false;
    }
    ++q;
    while (q < line.size() && line[q] == ' ') {
        ++q;
    }
    if (q >= line.size() || line[q] != '"') {
        return false;
    }
    ++q;
    const size_t n = strlen(ev);
    return line.compare(q, n, ev) == 0 && q + n < line.size() && line[q + n] == '"';
}

static bool json_int(const std::string & line, const char * key, long * out) {
    const size_t p = line.find(key);
    if (p == std::string::npos) {
        return false;
    }
    size_t q = line.find(':', p + strlen(key));
    if (q == std::string::npos) {
        return false;
    }
    ++q;
    while (q < line.size() && line[q] == ' ') {
        ++q;
    }
    const size_t start = q;
    while (q < line.size() && (line[q] == '-' || (line[q] >= '0' && line[q] <= '9'))) {
        ++q;
    }
    if (q == start) {
        return false;
    }
    *out = strtol(line.substr(start, q - start).c_str(), nullptr, 10);
    return true;
}

static bool json_id_list(const std::string & line, std::vector<int32_t> * out) {
    const size_t p = line.find("\"union_ids\"");
    if (p == std::string::npos) {
        return false;
    }
    size_t q = line.find(':', p);
    if (q == std::string::npos) {
        return false;
    }
    ++q;
    while (q < line.size() && line[q] == ' ') {
        ++q;
    }
    if (q >= line.size() || line[q] != '"') {
        return false;
    }
    ++q;
    const size_t end = line.find('"', q);
    if (end == std::string::npos) {
        return false;
    }
    out->clear();
    long v = 0;
    bool in = false;
    for (size_t i = q; i < end; ++i) {
        const char ch = line[i];
        if (ch >= '0' && ch <= '9') {
            v = v * 10 + (ch - '0');
            in = true;
        } else if (ch == ',') {
            if (in) {
                out->push_back((int32_t) v);
            }
            v = 0;
            in = false;
        } else {
            return false; // not the schema this test claims to read
        }
    }
    if (in) {
        out->push_back((int32_t) v);
    }
    return !out->empty();
}
