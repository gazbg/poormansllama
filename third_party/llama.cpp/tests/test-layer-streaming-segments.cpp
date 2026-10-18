// Host-only test (no CUDA, no model, no ggml) of the segmented layer-major prefill arithmetic (src/llama-ls-segments.h, G28).
//
// WHY THIS EXISTS. A wrong slab offset in the segmented prefill would make a chunk read or overwrite another chunk's activations: a plausible-looking wrong answer that no speed number shows.
// The offsets are pure arithmetic, so they are proven here for the shapes that matter: the historical single sweep must be unchanged byte for byte; with segments the offsets restart at every
// boundary; the tail chunk may be narrower; no two chunks of the SAME segment overlap; and every segment fits inside one slab half of ls_slab_tokens() tokens.
//
// Build:  g++ -O2 -std=c++17 -I../src test-layer-streaming-segments.cpp   (or cl /EHsc /std:c++17 /I..\src ...)
#include "../src/llama-ls-segments.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

struct mock_chunk {
    int64_t token_start;
    int64_t n_tokens;
};

static int failures = 0;

static void expect(bool cond, const char * what) {
    if (cond) {
        fprintf(stderr, "ok   %s\n", what);
    } else {
        fprintf(stderr, "FAIL %s\n", what);
        failures++;
    }
}

static std::vector<mock_chunk> layout(int64_t pp_tokens, int64_t ubatch) {
    std::vector<mock_chunk> out;
    for (int64_t s = 0; s < pp_tokens; s += ubatch) {
        out.push_back({s, std::min<int64_t>(ubatch, pp_tokens - s)});
    }
    return out;
}

int main() {
    fprintf(stderr, "== ls_segment_chunks ==\n");
    expect(ls_segment_chunks(0, 2048, 100) == 0, "seg_tokens 0 = off");
    expect(ls_segment_chunks(-5, 2048, 100) == 0, "negative seg_tokens = off");
    expect(ls_segment_chunks(16384, 2048, 100) == 8, "16384 / 2048 = 8 chunks per segment");
    expect(ls_segment_chunks(16384, 2048, 5) == 5, "a segment larger than the prompt is one segment of n_chunks");
    expect(ls_segment_chunks(16384, 2048, 8) == 8, "exactly the prompt = one segment");
    expect(ls_segment_chunks(1000, 2048, 100) == -1, "smaller than a chunk is refused");
    expect(ls_segment_chunks(3000, 2048, 100) == -1, "not a multiple of the chunk width is refused");
    expect(ls_segment_chunks(2048, 0, 100) == -1, "chunk width 0 is refused");

    fprintf(stderr, "\n== single sweep (seg_chunks >= n_chunks or <= 0) = the historical cumulative offsets ==\n");
    {
        const auto ch = layout(5000, 2048);       // 2048, 2048, 904
        const size_t e = 4 * 2048;                 // n_embd_io 2048 (bytes per token per half in f32 = 8192)
        for (int64_t seg : {(int64_t) 0, (int64_t) 3, (int64_t) 100}) {
            const auto off = ls_segment_slab_offsets(ch.data(), ch.size(), seg, 2048, 4);
            expect(off.size() == 3 && off[0] == 0 && off[1] == 2048 * e && off[2] == 4096 * e, "cumulative offsets for a single sweep");
        }
    }

    fprintf(stderr, "\n== segments: offsets restart, tail narrower, no overlap inside a segment, every segment fits one half ==\n");
    for (int64_t pp : {(int64_t) 8192, (int64_t) 16385, (int64_t) 49997, (int64_t) 250000}) {
        for (int64_t ub : {(int64_t) 512, (int64_t) 2048}) {
            for (int64_t seg_tokens : {4 * ub, 8 * ub, 16384L / ub * ub}) {
                if (seg_tokens < 4 * ub) { continue; }
                const auto ch = layout(pp, ub);
                const int64_t sc = ls_segment_chunks(seg_tokens, ub, (int64_t) ch.size());
                const int64_t slab_tokens = ls_slab_tokens(pp, seg_tokens);
                const size_t elt = 4, e = (size_t) 2048 * elt;
                const auto off = ls_segment_slab_offsets(ch.data(), ch.size(), sc, 2048, elt);
                bool ok = sc >= 1 && off.size() == ch.size();
                size_t half = (size_t) slab_tokens * e;
                for (size_t c = 0; ok && c < ch.size(); ++c) {
                    const size_t bytes = (size_t) ch[c].n_tokens * e;
                    if ((int64_t) c % sc == 0 && off[c] != 0) { ok = false; }                         // restarts at 0
                    if (off[c] + bytes > half) { ok = false; }                                         // inside the half
                    if ((int64_t) c % sc != 0 && off[c] != off[c - 1] + (size_t) ch[c - 1].n_tokens * e) { ok = false; }   // contiguous inside the segment
                    if (ls_segment_of((int64_t) c, sc) != (int64_t) c / sc) { ok = false; }
                }
                char what[160];
                snprintf(what, sizeof(what), "pp %lld ubatch %lld segment %lld tokens (%lld chunks): offsets valid, slab half %.1f MiB", (long long) pp, (long long) ub, (long long) seg_tokens, (long long) sc,
                         (double) half / (1024.0 * 1024.0));
                expect(ok, what);
            }
        }
    }

    fprintf(stderr, "\n== slab size ==\n");
    expect(ls_slab_tokens(250000, 0) == 250000, "no segmentation: the whole prompt");
    expect(ls_slab_tokens(250000, 16384) == 16384, "segmentation: one segment");
    expect(ls_slab_tokens(10000, 16384) == 10000, "a segment larger than the prompt: the prompt");
    {
        // the f32 slab at 250k: 2 halves x 2048 x tokens x 4 B = 3,906 MiB unsegmented, 256 MiB with 16,384-token segments
        const double full = 2.0 * 2048 * 250000 * 4 / (1024.0 * 1024.0), seg = 2.0 * 2048 * 16384 * 4 / (1024.0 * 1024.0);
        char what[160];
        snprintf(what, sizeof(what), "f32 slab at 250k tokens: %.0f MiB whole, %.0f MiB with 16,384-token segments", full, seg);
        expect(full > 1900 && seg == 256.0, what);
    }

    fprintf(stderr, "\nsegments test: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
