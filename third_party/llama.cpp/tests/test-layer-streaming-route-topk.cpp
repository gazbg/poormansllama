// T-GPLR-39: host-only equivalence test for the exact top-k router row
// selection that replaced the per-token stable sort in BOTH places that
// derived top-k router ids on the host: tel_route (the route telemetry) and
// read_router_ids (the row-set routing decision memcpy'd into sel_in).
//
// The reference is the PRE-T-GPLR-39 derivation verbatim: reset the expert
// order ascending, std::stable_sort it descending by probability, keep the
// first k. The subject is llama_ls_route_topk_row. The route event's
// FNV-1a-64 hash covers the per-token selection ORDER (with repeats), so the
// test pins whole id sequences and the hash, not just the id set: a set-only
// check would miss an order deviation and every recorded route hash would
// silently change with it.
//
// No CUDA, no model, no GPU: pure float arithmetic on host buffers.
#include "../src/llama-layer-streaming.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

// pre-T-GPLR-39 tel_route derivation, kept verbatim as the oracle
static std::vector<int32_t> reference_ids(const std::vector<float> & p, int64_t n_expert, int64_t n_tok, int64_t k) {
    std::vector<int32_t> out;
    out.reserve((size_t) (n_tok * k));
    std::vector<int32_t> order((size_t) n_expert);
    for (int64_t i = 0; i < n_tok; ++i) {
        const float * row = &p[(size_t) i * (size_t) n_expert];
        for (int64_t e = 0; e < n_expert; ++e) {
            order[(size_t) e] = (int32_t) e;
        }
        std::stable_sort(order.begin(), order.end(),
                         [&](int32_t a, int32_t b) { return row[a] > row[b]; });
        for (int64_t j = 0; j < k; ++j) {
            out.push_back(order[(size_t) j]);
        }
    }
    return out;
}

// the route event's hash over the emitted order (tel_route's FNV-1a-64)
static uint64_t fnv1a64_order(const std::vector<int32_t> & ids) {
    uint64_t h = 1469598103934665603ull;
    for (int32_t v : ids) {
        h ^= (uint64_t) (uint32_t) v;
        h *= 1099511628211ull;
    }
    return h;
}

// deterministic, dependency-free PRNG
static uint32_t xs = 123456789u;
static float rnd01() {
    xs ^= xs << 13;
    xs ^= xs >> 17;
    xs ^= xs << 5;
    return (float) (xs & 0xFFFFFFu) / (float) 0x1000000u;
}

static void check_case(const char * name, const std::vector<float> & p, int64_t n_expert, int64_t n_tok, int64_t k) {
    const std::vector<int32_t> want = reference_ids(p, n_expert, n_tok, k);
    std::vector<int32_t> got((size_t) (n_tok * k), -1);
    std::vector<int32_t> row((size_t) k, -1);
    for (int64_t i = 0; i < n_tok; ++i) {
        llama_ls_route_topk_row(&p[(size_t) i * (size_t) n_expert], n_expert, k, row.data());
        for (int64_t j = 0; j < k; ++j) {
            got[(size_t) (i * k + j)] = row[(size_t) j];
        }
    }
    if (got != want) {
        failures++;
        fprintf(stderr, "FAIL %s: id order differs (n_expert=%lld n_tok=%lld k=%lld)\n",
                name, (long long) n_expert, (long long) n_tok, (long long) k);
        for (size_t n = 0; n < got.size(); ++n) {
            if (got[n] != want[n]) {
                fprintf(stderr, "     first divergence at flattened index %zu (token %zu, slot %zu): got %d want %d\n",
                        n, n / (size_t) k, n % (size_t) k, got[n], want[n]);
                break;
            }
        }
        return;
    }
    const uint64_t hw = fnv1a64_order(want);
    const uint64_t hg = fnv1a64_order(got);
    if (hw != hg) {
        failures++;
        fprintf(stderr, "FAIL %s: hash differs: got %016llx want %016llx\n",
                name, (unsigned long long) hg, (unsigned long long) hw);
        return;
    }
    printf("PASS %s (n_expert=%lld n_tok=%lld k=%lld, hash %016llx)\n",
           name, (long long) n_expert, (long long) n_tok, (long long) k, (unsigned long long) hg);
}

// The read_router_ids call pattern (T-GPLR-39): one reused ids buffer of
// n_used, assigned per token into ids_per_token[i], then packed by the CALLER
// into the FFN input as element (k, t) at data[k + t*n_used]. Size checks
// mirror the caller's assertions (ids_per_token.size() == n_tok, each row ==
// n_used); the packed compare covers the memcpy layout as well as the ids, and
// the oracle is the pre-change per-token derivation.
static void check_rowset_pattern(const char * name, const std::vector<float> & p,
                                 int64_t n_expert, int64_t n_tok, int64_t k) {
    const std::vector<int32_t> flat_want = reference_ids(p, n_expert, n_tok, k);

    std::vector<std::vector<int32_t>> ids_per_token((size_t) n_tok);
    std::vector<int32_t> ids((size_t) k, -1);
    for (int64_t i = 0; i < n_tok; ++i) {
        llama_ls_route_topk_row(&p[(size_t) i * (size_t) n_expert], n_expert, k, ids.data());
        ids_per_token[(size_t) i].assign(ids.begin(), ids.end());
    }

    for (int64_t i = 0; i < n_tok; ++i) {
        if ((int64_t) ids_per_token[(size_t) i].size() != k) {
            failures++;
            fprintf(stderr, "FAIL %s: token %lld row has %zu ids, expected %lld\n",
                    name, (long long) i, ids_per_token[(size_t) i].size(), (long long) k);
            return;
        }
        for (int64_t j = 0; j < k; ++j) {
            if (ids_per_token[(size_t) i][(size_t) j] != flat_want[(size_t) (i * k + j)]) {
                failures++;
                fprintf(stderr, "FAIL %s: token %lld slot %lld: got %d want %d\n",
                        name, (long long) i, (long long) j,
                        ids_per_token[(size_t) i][(size_t) j], flat_want[(size_t) (i * k + j)]);
                return;
            }
        }
    }

    // the caller's packing: element (k, t) lands at packed[k + t*n_used]
    std::vector<int32_t> packed((size_t) (n_tok * k), -1);
    for (int64_t t = 0; t < n_tok; ++t) {
        memcpy(packed.data() + (size_t) (t * k), ids_per_token[(size_t) t].data(),
               (size_t) k * sizeof(int32_t));
    }
    if (packed != flat_want) {
        failures++;
        fprintf(stderr, "FAIL %s: packed sel_in layout differs\n", name);
        return;
    }
    printf("PASS %s (n_expert=%lld n_tok=%lld k=%lld, packed hash %016llx)\n",
           name, (long long) n_expert, (long long) n_tok, (long long) k,
           (unsigned long long) fnv1a64_order(packed));
}

int main(void) {
    // 1. dense random probabilities, the MoE shape
    {
        const int64_t ne = 512, nt = 8, k = 8;
        std::vector<float> p((size_t) (ne * nt));
        for (float & v : p) {
            v = rnd01();
        }
        check_case("dense-random", p, ne, nt, k);
    }

    // 2. dense random at the k extremes (k=1, k=n_expert-1, k=n_expert)
    {
        const int64_t ne = 512, nt = 4;
        std::vector<float> p((size_t) (ne * nt));
        for (float & v : p) {
            v = rnd01();
        }
        check_case("k=1",               p, ne, nt, 1);
        check_case("k=32",              p, ne, nt, 32);
        check_case("k=n_expert-1",      p, ne, nt, ne - 1);
        check_case("k=n_expert",        p, ne, nt, ne);   // full sorted order
    }

    // 3. heavy ties: only 8 distinct probabilities over 512 experts
    {
        const int64_t ne = 512, nt = 4, k = 8;
        std::vector<float> p((size_t) (ne * nt));
        for (float & v : p) {
            v = (float) (xs % 8u) / 8.0f;
            rnd01();
        }
        check_case("heavy-ties", p, ne, nt, k);
    }

    // 4. all equal -> ties resolve to ascending expert index (0..k-1)
    {
        const int64_t ne = 512, nt = 3, k = 8;
        std::vector<float> p((size_t) (ne * nt), 0.5f);
        check_case("all-equal", p, ne, nt, k);
    }

    // 5. all zero: softmax underflow makes large tie blocks at exactly 0.0f
    {
        const int64_t ne = 256, nt = 4, k = 8;
        std::vector<float> p((size_t) (ne * nt), 0.0f);
        check_case("all-zero-underflow", p, ne, nt, k);
    }

    // 6. mixed signed zeros plus a few positives: -0.0 and +0.0 compare equal
    {
        const int64_t ne = 64, nt = 4, k = 8;
        std::vector<float> p((size_t) (ne * nt), -0.0f);
        for (size_t n = 0; n < p.size(); n += 7) {
            p[n] = rnd01();
        }
        for (size_t n = 0; n < p.size(); n += 5) {
            p[n] = 0.0f;
        }
        check_case("signed-zeros", p, ne, nt, k);
    }

    // 7. a tie block straddling the k boundary (exactly k equal maxima)
    {
        const int64_t ne = 32, nt = 3, k = 8;
        std::vector<float> p((size_t) (ne * nt), 0.25f);   // every row all-equal
        check_case("boundary-ties", p, ne, nt, k);
    }

    // 8. tiny shapes, every k
    for (int64_t ne = 1; ne <= 5; ++ne) {
        std::vector<float> p((size_t) (ne * 3));
        for (float & v : p) {
            v = (float) (xs % 3u);   // many ties at tiny widths
            rnd01();
        }
        for (int64_t k = 1; k <= ne; ++k) {
            check_case("tiny", p, ne, 3, k);
        }
    }

    // 9. clamp: k > n_expert must fill exactly n_expert entries and match the
    //    full sorted order (tel_route guards n_used <= n_expert; this is the
    //    defensive path).
    {
        const int64_t ne = 16, nt = 2;
        std::vector<float> p((size_t) (ne * nt));
        for (float & v : p) {
            v = rnd01();
        }
        const std::vector<int32_t> want = reference_ids(p, ne, nt, ne);
        std::vector<int32_t> got;
        std::vector<int32_t> row((size_t) ne, -1);
        for (int64_t i = 0; i < nt; ++i) {
            llama_ls_route_topk_row(&p[(size_t) i * (size_t) ne], ne, ne + 7, row.data());
            for (int64_t j = 0; j < ne; ++j) {
                got.push_back(row[(size_t) j]);
            }
        }
        if (got != want) {
            failures++;
            fprintf(stderr, "FAIL k>n_expert clamp: id order differs\n");
        } else {
            printf("PASS k>n_expert clamp (n_expert=%lld, k=%lld)\n", (long long) ne, (long long) (ne + 7));
        }
    }

    // 10. degenerate inputs must not crash and must not write
    {
        std::vector<float> p(8, 1.0f);
        std::vector<int32_t> buf(8, -12345);
        llama_ls_route_topk_row(nullptr, 8, 8, buf.data());
        llama_ls_route_topk_row(p.data(), 8, 8, nullptr);
        llama_ls_route_topk_row(p.data(), 0, 8, buf.data());
        llama_ls_route_topk_row(p.data(), 8, 0, buf.data());
        bool intact = true;
        for (int32_t v : buf) {
            if (v != -12345) {
                intact = false;
            }
        }
        if (!intact) {
            failures++;
            fprintf(stderr, "FAIL degenerate inputs: buffer was written\n");
        } else {
            printf("PASS degenerate inputs (no crash, no write)\n");
        }
    }

    // 11. the read_router_ids / row-set pattern: multi-row ids_per_token plus
    //     the caller's sel_in packing
    {
        const int64_t ne = 512, nt = 16, k = 8;
        std::vector<float> p((size_t) (ne * nt));
        for (float & v : p) {
            v = rnd01();
        }
        check_rowset_pattern("rowset-ubatch", p, ne, nt, k);
    }
    {
        // duplicate rows: consecutive tokens sharing routing (the union case)
        const int64_t ne = 256, nt = 8, k = 8;
        std::vector<float> p((size_t) (ne * nt));
        for (int64_t i = 0; i < nt / 2; ++i) {
            for (int64_t e = 0; e < ne; ++e) {
                const float v = rnd01();
                p[(size_t) (i * ne + e)] = v;
                p[(size_t) ((i + nt / 2) * ne + e)] = v;
            }
        }
        check_rowset_pattern("rowset-duplicate-rows", p, ne, nt, k);
    }
    {
        // ties across every row, including across the merge point of the rows
        const int64_t ne = 512, nt = 5, k = 8;
        std::vector<float> p((size_t) (ne * nt));
        for (float & v : p) {
            v = (float) (xs % 4u) / 4.0f;
            rnd01();
        }
        check_rowset_pattern("rowset-heavy-ties", p, ne, nt, k);
    }
    {
        // W = 1: the shape the row-set call site documents today
        const int64_t ne = 512, nt = 1, k = 8;
        std::vector<float> p((size_t) (ne * nt));
        for (float & v : p) {
            v = rnd01();
        }
        check_rowset_pattern("rowset-single-row", p, ne, nt, k);
    }
    {
        // k extremes on the multi-row path (k = n_expert is legal here: the
        // function rejects n_used > n_expert before this loop)
        const int64_t ne = 64, nt = 4;
        std::vector<float> p((size_t) (ne * nt));
        for (float & v : p) {
            v = rnd01();
        }
        check_rowset_pattern("rowset-k=1",      p, ne, nt, 1);
        check_rowset_pattern("rowset-k=n_expert", p, ne, nt, ne);
    }

    if (failures != 0) {
        fprintf(stderr, "\nFAILED: %d case(s)\n", failures);
        return 1;
    }
    printf("\ntest-layer-streaming-route-topk: all cases PASS\n");
    return 0;
}
