// G28: host-pure arithmetic of the SEGMENTED layer-major prefill (no ggml, no CUDA, no model): which chunks form a segment, and where a chunk's activation slice sits inside its segment's slab half.
//
// The layer-major prefill keeps the inter-layer activations of the whole prompt in a GPU slab (2 halves x n_embd_io x prompt tokens x element size: 16 KiB per token in f32), which is
// what breaks the owner's "16,000 MiB stay free" rule at long prompts. With --pp-segment-tokens S the chunk list is processed in segments of S tokens: for each segment the embed,
// the sweep over all layers and the output run against a slab that holds ONE segment. Per layer the chunks are still visited in ascending order (segment k+1 after segment k), so the
// arithmetic is the unsegmented one; the price is one more pass of the layer banks over PCIe per segment. Header-only so that the production code and the host test share the symbols.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Chunks per segment for segments of `seg_tokens` tokens over chunks of `chunk_width` tokens (the width of a full chunk = the ubatch):
//   0   segmentation off (seg_tokens <= 0)
//   -1  refused: the chunk width is not positive, or seg_tokens is not a positive multiple of it (the caller must refuse, never round)
//   k   min(n_chunks, seg_tokens / chunk_width); k == n_chunks is a single segment, the historical sweep
static inline int64_t ls_segment_chunks(int64_t seg_tokens, int64_t chunk_width, int64_t n_chunks) {
    if (seg_tokens <= 0) {
        return 0;
    }
    if (chunk_width <= 0 || seg_tokens % chunk_width != 0) {
        return -1;
    }
    const int64_t k = seg_tokens / chunk_width;
    if (k < 1) {
        return -1;
    }
    return k < n_chunks ? k : n_chunks;
}

// the segment a chunk belongs to (0 without segmentation)
static inline int64_t ls_segment_of(int64_t chunk, int64_t seg_chunks) {
    return seg_chunks > 0 ? chunk / seg_chunks : 0;
}

// Byte offset of every chunk inside its segment's slab half: the cumulative size of the chunks before it IN THE SAME SEGMENT, restarting at 0 at every segment boundary.
// seg_chunks <= 0 or >= n_chunks is one segment and gives the historical whole-prompt cumulative offsets.
template <typename Chunk>
static inline std::vector<size_t> ls_segment_slab_offsets(const Chunk * chunks, size_t n_chunks, int64_t seg_chunks, int64_t n_embd_io, size_t slab_elt) {
    std::vector<size_t> off(n_chunks, 0);
    size_t acc = 0;
    for (size_t c = 0; c < n_chunks; ++c) {
        if (seg_chunks > 0 && seg_chunks < (int64_t) n_chunks && (int64_t) c % seg_chunks == 0) {
            acc = 0;
        }
        off[c] = acc;
        acc += (size_t) chunks[c].n_tokens * (size_t) n_embd_io * slab_elt;
    }
    return off;
}

// tokens one slab half must hold: the segment size when it is smaller than the prompt, else the prompt
static inline int64_t ls_slab_tokens(int64_t pp_tokens, int64_t seg_tokens) {
    return (seg_tokens > 0 && seg_tokens < pp_tokens) ? seg_tokens : pp_tokens;
}
