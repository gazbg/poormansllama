// Host-only unit test for the layer-major chunk layout (T-Q38CTX-01).
// No GPU, no model: verifies the pure {token_start, n_tokens} arithmetic
// that splits an arbitrary PP length into full ubatch-sized chunks plus one
// optional 1..ubatch-1 token tail. Runs on the CPU without CUDA.
#include "../src/llama-layer-streaming.h"

#include <cstdio>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

static int failures = 0;

static void check_layout(int64_t pp, int ubatch,
                         const std::vector<std::pair<int64_t, int64_t>> & expect) {
    const std::vector<llama_ls_chunk> got = llama_ls_chunk_layout(pp, ubatch);
    if ((int) got.size() != (int) expect.size()) {
        fprintf(stderr, "FAIL pp=%lld ubatch=%d: got %zu chunks, expected %zu\n",
                (long long) pp, ubatch, got.size(), expect.size());
        failures++;
        return;
    }
    bool ok = true;
    for (size_t i = 0; i < expect.size(); ++i) {
        if (got[i].token_start != expect[i].first || got[i].n_tokens != expect[i].second) {
            fprintf(stderr, "FAIL pp=%lld ubatch=%d chunk %zu: got {%lld,%lld}, expected {%lld,%lld}\n",
                    (long long) pp, ubatch, i,
                    (long long) got[i].token_start, (long long) got[i].n_tokens,
                    (long long) expect[i].first, (long long) expect[i].second);
            ok = false;
        }
    }
    // structural invariants: starts at 0, contiguous, covers exactly pp tokens,
    // every chunk width in 1..ubatch, only the last chunk may be sub-full
    int64_t cursor = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const bool last = (i + 1 == got.size());
        if (got[i].token_start != cursor || got[i].n_tokens < 1 ||
            got[i].n_tokens > ubatch || (!last && got[i].n_tokens != ubatch)) {
            fprintf(stderr, "FAIL pp=%lld ubatch=%d chunk %zu: layout invariant violated "
                    "(start %lld, n %lld)\n",
                    (long long) pp, ubatch, i,
                    (long long) got[i].token_start, (long long) got[i].n_tokens);
            ok = false;
        }
        cursor += got[i].n_tokens;
    }
    if (cursor != pp) {
        fprintf(stderr, "FAIL pp=%lld ubatch=%d: chunks cover %lld tokens, expected %lld\n",
                (long long) pp, ubatch, (long long) cursor, (long long) pp);
        ok = false;
    }
    if (ok) {
        fprintf(stderr, "ok   pp=%lld ubatch=%d -> %zu chunks\n", (long long) pp, ubatch, got.size());
    } else {
        failures++;
    }
}

// ---------------------------------------------------------------------------
// T-GPLR-49: the layer-major activation-slab region arithmetic and the
// non-aliasing claim rule. Before T-GPLR-49 begin_set derived a chunk's slab
// offset as sum_{j<chunk} n_j * n_embd_io * sizeof(float) -- exactly 2x the
// truth for a bf16 slab, which is the default since T-GPLR-12 -- and skipped
// chunk 0, which then inherited whatever the cursor happened to hold. Both are
// asserted below, host-only, with no model and no GPU.
// ---------------------------------------------------------------------------

// Per chunk: descriptor-derived offset, slab-dtype sizing (llama_ls_slab_region_of),
// fits one half, and
// the claim intervals of one half never overlap. Returns its failure count.
static int check_slab_regions(int64_t pp, int ubatch, ggml_type dtype, int64_t n_embd, bool verbose) {
    int bad = 0;
    const std::vector<llama_ls_chunk> ch = llama_ls_chunk_layout(pp, ubatch);
    if (ch.empty()) {
        fprintf(stderr, "FAIL slab: empty layout for pp=%lld ubatch=%d\n", (long long) pp, ubatch);
        return 1;
    }
    const size_t elt = ggml_type_size(dtype);
    const size_t row = (size_t) n_embd * elt;
    size_t expect = 0;
    std::vector<std::pair<size_t, size_t>> claims[2];
    for (size_t c = 0; c < ch.size(); ++c) {
        llama_ls_slab_region r{};
        if (!llama_ls_slab_region_of(ch.data(), ch.size(), c, n_embd, dtype, &r)) {
            fprintf(stderr, "FAIL slab pp=%lld chunk %zu: no region\n", (long long) pp, c);
            return bad + 1;
        }
        const size_t want_bytes = (size_t) ch[c].n_tokens * row;
        const size_t want_half  = (size_t) pp * row;
        if (r.offset != expect || r.bytes != want_bytes || r.half != want_half) {
            fprintf(stderr, "FAIL slab pp=%lld chunk %zu: got {off %zu, bytes %zu, half %zu}, "
                    "expected {off %zu, bytes %zu, half %zu}\n",
                    (long long) pp, c, r.offset, r.bytes, r.half, expect, want_bytes, want_half);
            bad++;
        }
        if (c == 0 && r.offset != 0) {
            fprintf(stderr, "FAIL slab pp=%lld: chunk 0 offset %zu, expected 0 "
                    "(it must not inherit a cursor)\n", (long long) pp, r.offset);
            bad++;
        }
        if (r.offset + r.bytes > r.half) {
            fprintf(stderr, "FAIL slab pp=%lld chunk %zu: region runs past the half\n", (long long) pp, c);
            bad++;
        }
        for (int which = 0; which < 2; ++which) {
            for (const auto & q : claims[which]) {
                if (llama_ls_slab_overlaps(q.first, q.second, r.offset, r.offset + r.bytes)) {
                    fprintf(stderr, "FAIL slab pp=%lld chunk %zu: claim overlaps a live one\n",
                            (long long) pp, c);
                    bad++;
                }
            }
            claims[which].emplace_back(r.offset, r.offset + r.bytes);
        }
        expect += want_bytes;
    }
    if (expect != (size_t) pp * row) {
        fprintf(stderr, "FAIL slab pp=%lld: chunks consume %zu B, one half is %zu B\n",
                (long long) pp, expect, (size_t) pp * row);
        bad++;
    }
    if (bad == 0 && verbose) {
        fprintf(stderr, "ok   slab pp=%lld ubatch=%d %s n_embd=%lld -> %zu chunks, half %zu B\n",
                (long long) pp, ubatch, dtype == GGML_TYPE_BF16 ? "bf16" : "f32",
                (long long) n_embd, ch.size(), (size_t) pp * row);
    }
    return bad;
}

// The PRE-fix formula, kept here only so the f32 case can prove the fix did not
// move the values that formula used to produce (f32 is the one dtype where
// sizeof(float) was right): sum_{j<chunk} n_j * n_embd_io * sizeof(float).
static size_t legacy_offset_f32(const std::vector<llama_ls_chunk> & ch, size_t chunk, int64_t n_embd) {
    size_t off = 0;
    for (size_t c = 0; c < chunk; ++c) {
        off += (size_t) ch[c].n_tokens * (size_t) n_embd * sizeof(float);
    }
    return off;
}

int main() {
    // invalid inputs
    if (!llama_ls_chunk_layout(0, 512).empty()) {
        fprintf(stderr, "FAIL: pp=0 must return an empty layout\n");
        failures++;
    }
    if (!llama_ls_chunk_layout(1, 0).empty()) {
        fprintf(stderr, "FAIL: ubatch=0 must return an empty layout\n");
        failures++;
    }

    // single-chunk layouts
    check_layout(1,     512, {{0, 1}});
    check_layout(511,   512, {{0, 511}});
    check_layout(512,   512, {{0, 512}});

    // full chunk + 1-token tail (the plan's 513 parity case)
    check_layout(513,   512, {{0, 512}, {512, 1}});

    // exact multiples keep the uniform whole-chunk layout unchanged
    check_layout(1024,  512, {{0, 512}, {512, 512}});
    check_layout(20480, 512, {{0, 512}, {512, 512}, {1024, 512}, {1536, 512}, {2048, 512},
                              {2560, 512}, {3072, 512}, {3584, 512}, {4096, 512}, {4608, 512},
                              {5120, 512}, {5632, 512}, {6144, 512}, {6656, 512}, {7168, 512},
                              {7680, 512}, {8192, 512}, {8704, 512}, {9216, 512}, {9728, 512},
                              {10240, 512}, {10752, 512}, {11264, 512}, {11776, 512}, {12288, 512},
                              {12800, 512}, {13312, 512}, {13824, 512}, {14336, 512}, {14848, 512},
                              {15360, 512}, {15872, 512}, {16384, 512}, {16896, 512}, {17408, 512},
                              {17920, 512}, {18432, 512}, {18944, 512}, {19456, 512}, {19968, 512}});

    // 100,000 tokens = 195 full chunks + 160-token tail
    {
        std::vector<std::pair<int64_t, int64_t>> expect;
        for (int c = 0; c < 195; ++c) {
            expect.push_back({(int64_t) c * 512, 512});
        }
        expect.push_back({195 * 512, 160});
        check_layout(100000, 512, expect);
    }

    // non-512 ubatch sizes (the layout is general)
    check_layout(10, 64, {{0, 10}});
    check_layout(70, 64, {{0, 64}, {64, 6}});

    // ------------------------------------------------------------------
    // T-GPLR-49: slab-region arithmetic. Offset = sum_{j<chunk} n_j * row,
    // row = n_embd_io * ggml_type_size(act_dtype); the last chunk ends exactly
    // at the half, and no two chunk claims of a half overlap.
    // ------------------------------------------------------------------
    failures += check_slab_regions(512,    512, GGML_TYPE_BF16, 2048, true);
    failures += check_slab_regions(513,    512, GGML_TYPE_BF16, 2048, true);
    failures += check_slab_regions(4096,   512, GGML_TYPE_BF16, 2048, true);
    failures += check_slab_regions(50000,  512, GGML_TYPE_BF16, 2048, true);
    failures += check_slab_regions(200000, 512, GGML_TYPE_BF16, 2048, true);
    failures += check_slab_regions(4096,   512, GGML_TYPE_F32,  2048, true);
    failures += check_slab_regions(513,    128, GGML_TYPE_BF16,  128, true);  // non-512 ubatch, tiny width

    // The fix must be a NO-OP for f32 (where sizeof(float) was the right
    // element size) and exactly halve every offset for bf16.
    {
        const std::vector<llama_ls_chunk> ch = llama_ls_chunk_layout(4096, 512);
        bool ok = true;
        for (size_t c = 0; c < ch.size(); ++c) {
            llama_ls_slab_region f32{};
            llama_ls_slab_region bf16{};
            if (!llama_ls_slab_region_of(ch.data(), ch.size(), c, 2048, GGML_TYPE_F32, &f32) ||
                !llama_ls_slab_region_of(ch.data(), ch.size(), c, 2048, GGML_TYPE_BF16, &bf16)) {
                fprintf(stderr, "FAIL slab 4k: region derivation failed for chunk %zu\n", c);
                failures++;
                ok = false;
                break;
            }
            if (f32.offset != legacy_offset_f32(ch, c, 2048)) {
                fprintf(stderr, "FAIL slab 4k chunk %zu: f32 offset %zu != legacy %zu "
                        "(the fix must not move f32)\n", c, f32.offset, legacy_offset_f32(ch, c, 2048));
                failures++;
                ok = false;
            }
            if (bf16.offset * 2 != f32.offset) {
                fprintf(stderr, "FAIL slab 4k chunk %zu: bf16 offset %zu is not half the f32 offset %zu\n",
                        c, bf16.offset, f32.offset);
                failures++;
                ok = false;
            }
        }
        if (ok) {
            fprintf(stderr, "ok   slab 4k: f32 unchanged from the legacy formula, bf16 exactly halved\n");
        }
    }

    // The pre-fix formula, evaluated for a bf16 slab, must be WRONG and wrong in
    // the measured direction: exactly 2x the descriptor truth for every chunk
    // >= 1 (sizeof(float) = 4 vs the 2-byte bf16 element). This is the falsifier
    // the fix had to satisfy; if these ever become equal, this guard stops
    // proving anything. Chunk 0 is 0 under both, which is why the cursor-inherit
    // defect was invisible until a set was created after a non-zero cursor.
    {
        const std::vector<llama_ls_chunk> ch = llama_ls_chunk_layout(4096, 512);
        bool ok = true;
        for (size_t c = 1; c < ch.size(); ++c) {
            llama_ls_slab_region real{};
            if (!llama_ls_slab_region_of(ch.data(), ch.size(), c, 2048, GGML_TYPE_BF16, &real)) {
                fprintf(stderr, "FAIL slab legacy: no region for chunk %zu\n", c);
                failures++;
                ok = false;
                break;
            }
            const size_t legacy = legacy_offset_f32(ch, c, 2048);   // the old formula on a bf16 slab
            if (legacy != real.offset * 2) {
                fprintf(stderr, "FAIL slab legacy chunk %zu: pre-fix formula %zu, expected 2x the real %zu\n",
                        c, legacy, real.offset);
                failures++;
                ok = false;
            }
        }
        if (ok) {
            fprintf(stderr, "ok   slab: the pre-fix formula is exactly 2x the descriptor truth at bf16\n");
        }
    }

    // Refusals: the pure function must never invent a region. (Note that a
    // valid descriptor can never overflow a half -- offset + bytes <= half
    // holds by construction -- so the only refusals are invalid inputs, and the
    // cursor-path exhaustion check inside slab_slice stays the runtime guard.)
    {
        const std::vector<llama_ls_chunk> ch = llama_ls_chunk_layout(4096, 512);
        llama_ls_slab_region r{};
        bool ok = true;
        if (llama_ls_slab_region_of(nullptr, ch.size(), 0, 2048, GGML_TYPE_BF16, &r)) {
            fprintf(stderr, "FAIL slab refusal: null descriptor accepted\n"); failures++; ok = false;
        }
        if (llama_ls_slab_region_of(ch.data(), ch.size(), 0, 2048, GGML_TYPE_BF16, nullptr)) {
            fprintf(stderr, "FAIL slab refusal: null out accepted\n"); failures++; ok = false;
        }
        if (llama_ls_slab_region_of(ch.data(), 0, 0, 2048, GGML_TYPE_BF16, &r)) {
            fprintf(stderr, "FAIL slab refusal: empty descriptor accepted\n"); failures++; ok = false;
        }
        if (llama_ls_slab_region_of(ch.data(), ch.size(), 0, 0, GGML_TYPE_BF16, &r)) {
            fprintf(stderr, "FAIL slab refusal: n_embd_io = 0 accepted\n"); failures++; ok = false;
        }
        if (llama_ls_slab_region_of(ch.data(), ch.size(), ch.size(), 2048, GGML_TYPE_BF16, &r)) {
            fprintf(stderr, "FAIL slab refusal: chunk_id == n accepted\n"); failures++; ok = false;
        }
        if (llama_ls_slab_region_of(ch.data(), ch.size(), ch.size() + 5, 2048, GGML_TYPE_BF16, &r)) {
            fprintf(stderr, "FAIL slab refusal: chunk_id > n accepted\n"); failures++; ok = false;
        }
        const std::vector<llama_ls_chunk> zero_width = { {0, 512}, {512, 0} };
        if (llama_ls_slab_region_of(zero_width.data(), zero_width.size(), 0, 2048, GGML_TYPE_BF16, &r)) {
            fprintf(stderr, "FAIL slab refusal: zero-width chunk accepted\n"); failures++; ok = false;
        }
        if (ok) {
            fprintf(stderr, "ok   slab: invalid inputs refused\n");
        }
    }

    // The non-aliasing rule: predicate semantics plus the seeded duplicate
    // claim that MUST be detected (the corrupting direction).
    {
        bool ok = true;
        struct { const char * what; size_t as, ae, bs, be; bool want; } cases[] = {
            { "touching halves", 0, 1024, 1024, 2048, false },
            { "identical",       0, 1024,    0, 1024, true  },
            { "one byte shared", 0, 1025, 1024, 2048, true  },
            { "contained",     100,  200,   50,  300, true  },
            { "disjoint",        0, 1024, 4096, 5120, false },
            { "degenerate A",  500,  500,    0, 1024, false },
            { "degenerate B",    0, 1024,  500,  500, false },
        };
        for (const auto & c : cases) {
            const bool got = llama_ls_slab_overlaps(c.as, c.ae, c.bs, c.be);
            if (got != c.want) {
                fprintf(stderr, "FAIL overlap '%s': got %d, expected %d\n", c.what, (int) got, (int) c.want);
                failures++;
                ok = false;
            }
        }
        const std::vector<llama_ls_chunk> ch = llama_ls_chunk_layout(4096, 512);
        llama_ls_slab_region c1{};
        if (!llama_ls_slab_region_of(ch.data(), ch.size(), 1, 2048, GGML_TYPE_BF16, &c1)) {
            fprintf(stderr, "FAIL overlap control: no region for chunk 1\n"); failures++; ok = false;
        } else if (!llama_ls_slab_overlaps(c1.offset, c1.offset + c1.bytes, c1.offset, c1.offset + c1.bytes)) {
            fprintf(stderr, "FAIL overlap control: a duplicated claim MUST be detected\n");
            failures++; ok = false;
        } else if (llama_ls_slab_overlaps(c1.offset, c1.offset + c1.bytes, c1.offset + c1.bytes, c1.offset + 2 * c1.bytes)) {
            fprintf(stderr, "FAIL overlap control: a touching (non-overlapping) claim must NOT be flagged\n");
            failures++; ok = false;
        }
        if (ok) {
            fprintf(stderr, "ok   slab: overlap predicate + duplicate-claim negative control\n");
        }
    }

    // The claim REGISTRY itself, not just the predicate it calls: this is the
    // mechanism contract 2 chose for non-aliasing, so the wiring has to be
    // exercised directly. The seeded duplicate claim is the specified negative
    // control and it MUST be refused loudly, naming both intervals -- a
    // silent acceptance here would be exactly the aliasing the task forbids.
    {
        llama_layer_streaming ls;
        bool ok = true;
        if (!ls.claim_slab_region(0, 0, 1024)) {
            fprintf(stderr, "FAIL claim: the first claim was refused: %s\n", ls.error_msg.c_str());
            failures++; ok = false;
        }
        if (!ls.claim_slab_region(0, 1024, 1024)) {   // touching: legal
            fprintf(stderr, "FAIL claim: a touching claim was refused: %s\n", ls.error_msg.c_str());
            failures++; ok = false;
        }
        ls.error_msg.clear();
        if (ls.claim_slab_region(0, 1024, 1024)) {   // the seeded duplicate
            fprintf(stderr, "FAIL claim: a repeated claim was ACCEPTED (silent aliasing)\n");
            failures++; ok = false;
        } else if (ls.error_msg.empty()) {
            fprintf(stderr, "FAIL claim: refused without naming the collision\n");
            failures++; ok = false;
        } else if (ls.error_msg.find("1024") == std::string::npos ||
                   ls.error_msg.find("2048") == std::string::npos) {
            fprintf(stderr, "FAIL claim: message names no interval pair: %s\n", ls.error_msg.c_str());
            failures++; ok = false;
        }
        if (ls.claim_slab_region(0, 2047, 1)) {      // one shared byte
            fprintf(stderr, "FAIL claim: a one-byte overlap was accepted\n");
            failures++; ok = false;
        }
        if (!ls.claim_slab_region(1, 1024, 1024)) {  // half B is independent
            fprintf(stderr, "FAIL claim: half B refused what half A holds: %s\n", ls.error_msg.c_str());
            failures++; ok = false;
        }
        if (ls.claim_slab_region(2, 0, 16) || ls.claim_slab_region(-1, 0, 16) ||
            ls.claim_slab_region(0, 0, 0)) {
            fprintf(stderr, "FAIL claim: an invalid or empty claim was accepted\n");
            failures++; ok = false;
        }
        if (ok) {
            fprintf(stderr, "ok   slab: the claim registry refuses duplicates loudly, halves independent\n");
        }
    }

    if (failures) {
        fprintf(stderr, "test-layer-streaming-layout: %d failure(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "test-layer-streaming-layout: all layout checks passed\n");
    return 0;
}
