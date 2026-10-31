// G32: layer-major prefill for the server (llama-server --lm-prefill).
//
// A llama_decode() whose batch is one sequence's next >= min_tokens prompt tokens is not split into ubatches that each run all layers (every ubatch would copy all expert banks from the host).
// It runs layer-major instead: the batch is cut into chunks of `ubatch` tokens, and for every decoder layer in turn the layer's expert bank is copied ONCE into a VRAM slot window while
// all chunks go through that layer in ascending order. The chunks hand their hidden state to the next layer through an activation slab.
//
// Unlike the runner's layer_major_pp (llama-layer-streaming.cpp), which runs prebuilt persistent graph sets for one fixed layout, every piece (embed, one layer of one chunk, the head) is built
// by the ordinary graph machinery when it is launched (llm_graph_lm in llama-graph.h): the positions, the KV slots, the recurrent state and the attention window of a chunk therefore come from
// the chunk's own memory context, for any batch size and any starting position. The slot protocol (copy stream, events, async feed of the next bank) is the runner's (llama_layer_streaming).
//
// Two architectures: the Qwen3.5/3.6 MoE models (the streamed package is the routed-expert bank of a layer; everything else is device resident) and the dense Qwen3.5/3.8 models (the streamed package
// is the WHOLE decoder layer: attention or Gated DeltaNet block, FFN and norms; a layer is either entirely on the device or entirely in host memory). The KV cache and the recurrent state of a dense
// model must be on the device for every layer, whatever the placement of its weights (-ngl 99 with the host layers placed by tensor overrides).
//
// What it does NOT do (the batch then takes the ordinary decode path): several sequences in one batch, embeddings / backend samplers / MTP, logits for other than the last token, a KV cache whose
// cells are not laid out cell == position (the implicit causal mask of the flash attention kernel needs that). A context with rollback snapshots (n_rs_seq > 0, MTP) is run in the plain mode.

#include "llama-hybrid.h"

#include "llama-context.h"

#ifdef LLAMA_LAYER_STREAMING

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-layer-streaming.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "llama-lm-prefill.h"
#include "llama-dense-stream.h"
#include "llama-io.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// set by llama-hybrid-api.cpp when the VRAM expert cache exists: begin = make room for `need_bytes` (false: cannot, the pass is not run), end = give the room back
bool (*llama_lm_vram_hook)(llama_context * ctx, int64_t need_bytes, int begin) = nullptr;

// true while llama_context::decode() has taken the batch for the layer-major pass (the hybrid prefill window must not open for it)
bool llama_lm_prefill_pending = false;

llama_lm_state::~llama_lm_state() = default;

void llama_lm_state_deleter::operator()(llama_lm_state * s) const {
    delete s;
}

static bool lm_log_on() {
    static const bool on = getenv("LLAMA_LM_LOG") != nullptr;
    return on;
}

// G42: the serializer of a snapshot. llama_memory_recurrent::state_write writes the rows through write_tensor(); this writer serves them from the host copies taken inside the pass, so the blob has the format
// llama_state_seq_get_data_ext() gives (the serializer is the real one, nothing is duplicated)
static constexpr uint32_t k_lm_io_magic = 0xaf143cd8;     // io_magic of llama-context.cpp

class llama_io_write_lm_snap : public llama_io_write_i {
public:
    llama_io_write_lm_snap(std::vector<uint8_t> & out_, const std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> & rows_) : out(out_), rows(rows_) {}

    void write(const void * src, size_t size) override {
        const uint8_t * p = (const uint8_t *) src;
        out.insert(out.end(), p, p + size);
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        GGML_UNUSED(offset);
        for (const auto & r : rows) {
            if (r.first == tensor) {
                if (r.second.size() != size) {
                    throw std::runtime_error("lm snapshot: the row of a tensor has another size than the serializer asks for");
                }
                out.insert(out.end(), r.second.begin(), r.second.end());
                return;
            }
        }
        throw std::runtime_error("lm snapshot: no snapshot row for a tensor the serializer writes");
    }
    size_t n_bytes() override {
        return out.size();
    }

private:
    std::vector<uint8_t> & out;
    const std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> & rows;
};

static uint64_t lm_fnv64(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

//
// plan
//

llama_lm_plan llama_context::lm_make_plan(const llama_batch & b) const {
    llama_lm_plan pl;

    if (lm_state == nullptr) {
        return pl;
    }
    const auto & P = lm_state->p;

    if (b.n_tokens < P.min_tokens && !(lm_state->armed_batch && lm_state->armed_n == b.n_tokens)) {
        pl.why = "below --lm-prefill-min";
        return pl;
    }
    if (b.token == nullptr || b.embd != nullptr) {
        pl.why = "the batch carries embeddings";
        return pl;
    }
    bool layer_inp = false;
    for (const bool e : cparams.embeddings_layer_inp) {
        layer_inp = layer_inp || e;
    }
    if (cparams.embeddings || cparams.embeddings_nextn || layer_inp || !sampling.samplers.empty() || !cparams.causal_attn || !cparams.flash_attn) {
        pl.why = "the context asks for embeddings / backend sampling, or has no causal flash attention";
        return pl;
    }
    if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || layer_streaming != nullptr) {
        pl.why = "not a default context";
        return pl;
    }

    const int32_t n = b.n_tokens;

    llama_seq_id seq = 0;
    for (int32_t i = 0; i < n; ++i) {
        if (b.n_seq_id != nullptr && b.n_seq_id[i] != 1) {
            pl.why = "a token belongs to several sequences";
            return pl;
        }
        const llama_seq_id s = b.seq_id != nullptr ? b.seq_id[i][0] : 0;
        if (i == 0) {
            seq = s;
        } else if (s != seq) {
            pl.why = "the batch holds several sequences";
            return pl;
        }
    }
    if (seq < 0 || (uint32_t) seq >= cparams.n_seq_max) {
        pl.why = "invalid sequence id";
        return pl;
    }

    auto * mh = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (mh == nullptr) {
        pl.why = "no hybrid memory";
        return pl;
    }

    // rollback snapshots (MTP): the pass runs in the plain mode (see lm_run), so the state of the seq must stand in the plain plane
    if (cparams.n_rs_seq != 0 && mh->get_mem_recr()->rs_idx[seq] != 0) {
        pl.why = "a recurrent rollback is pending";
        return pl;
    }

    const llama_pos p0 = memory->seq_pos_max(seq) + 1;
    if (b.pos != nullptr) {
        for (int32_t i = 0; i < n; ++i) {
            if (b.pos[i] != p0 + i) {
                pl.why = "the positions do not continue the sequence";
                return pl;
            }
        }
    }

    // logits: none, or the last token's only
    if (b.logits != nullptr) {
        for (int32_t i = 0; i + 1 < n; ++i) {
            if (b.logits[i] != 0) {
                pl.why = "logits are wanted for more than the last token";
                return pl;
            }
        }
        pl.want_logits = b.logits[n - 1] != 0;
    } else {
        pl.want_logits = true;
    }

    // the chunk layout: chunk boundaries sit at multiples of `ubatch` in ABSOLUTE position (a pass that resumes at p0 starts with the short chunk that reaches the next multiple), so a pass
    // behind a cached prefix cuts its chunks where a fresh pass over the same prompt cuts them
    const int32_t W = P.ubatch;
    {
        // G42: a chunk boundary at every requested cut as well (the cuts are ascending); a cut that sits on a multiple of W costs nothing
        const std::vector<int32_t> & cuts = lm_state->snap_req;
        size_t ci = 0;
        for (int32_t t = 0; t < n;) {
            int32_t w = std::min(W - (int32_t) ((p0 + t) % W), n - t);
            while (ci < cuts.size() && cuts[ci] <= t) {
                ci++;
            }
            if (ci < cuts.size() && cuts[ci] < t + w) {
                w = cuts[ci] - t;
            }
            pl.widths.push_back(w);
            t += w;
            if (ci < cuts.size() && cuts[ci] == t && t < n) {
                pl.snap_cuts.push_back(t);
                pl.snap_chunk.push_back((int32_t) pl.widths.size() - 1);
                ci++;
            }
        }
    }

    // a pending K-shift would be applied by memory_update after the plan is made
    if (mh->get_mem_attn()->get_has_shift()) {
        pl.why = "a K-shift is pending";
        return pl;
    }

    // the sequence's cells must be one contiguous block (cell == c0 + position) with room behind it, and the recurrent state must stand exactly at p0
    const int64_t c0 = mh->get_mem_attn()->lm_block(seq, p0, (uint32_t) n, &pl.why_detail);
    if (c0 < 0) {
        pl.why = "the sequence's KV cells are not one contiguous block with room behind it (interleaved with another sequence's, a shift, a restored state)";
        return pl;
    }
    if (mh->get_mem_recr()->seq_pos_max(seq) != p0 - 1) {
        pl.why = "the recurrent state is not at the batch's first position";
        return pl;
    }

    pl.ok  = true;
    pl.seq = seq;
    pl.p0  = p0;
    pl.c0  = c0;
    pl.n   = n;
    return pl;
}

// LLAMA_LM_LOG=1: say why a batch that is big enough was not taken
void llama_context::lm_note(const llama_batch & b, const llama_lm_plan & pl) const {
    if (lm_state != nullptr && !pl.ok && lm_log_on() && b.n_tokens >= lm_state->p.min_tokens) {
        LLAMA_LOG_WARN("lm-prefill: a batch of %d tokens takes the ordinary path: %s%s%s%s\n", (int) b.n_tokens, pl.why != nullptr ? pl.why : "?",
                       pl.why_detail.empty() ? "" : " [", pl.why_detail.c_str(), pl.why_detail.empty() ? "" : "]");
    }
}

//
// resources of one pass
//

static int64_t lm_dev_free() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t f = 0, t = 0;
            ggml_backend_dev_memory(d, &f, &t);
            return (int64_t) f;
        }
    }
    return 0;
}

bool llama_context::lm_acquire(const llama_lm_plan & pl) {
    auto & S = *lm_state;
    int32_t wmax = 1;
    for (const int32_t w : pl.widths) {
        wmax = std::max(wmax, w);
    }
    if (S.armed) {
        // G42: the resources were prepared (in the idle time, or for this very batch): adopt them when they are big enough, else give them back and acquire again (the cache is shrunk already: no new shrink)
        if (S.slab_tokens >= std::min<int64_t>(pl.n, S.p.segment) && S.armed_w >= wmax) {
            S.armed        = false;
            S.armed_batch  = false;
            S.cur_w        = wmax;
            S.free_pre     = lm_dev_free() + S.slot_total + 2*(int64_t) S.slab_half;
            S.t_acquire_us = 0;
            S.n_arm_used++;
            return true;
        }
        S.free_pre = 0;     // no pass ran: nothing to measure
        lm_release();
    }
    return lm_acquire_n(pl.n, wmax);
}

bool llama_context::lm_acquire_n(int64_t n_tokens, int32_t wmax) {
    auto & S = *lm_state;
    const auto & P = S.p;

    // test hook: the resources are refused, the batch takes the ordinary path
    static const bool fail_acquire = getenv("LLAMA_LM_FAIL_ACQUIRE") != nullptr;
    if (fail_acquire) {
        LLAMA_LOG_WARN("%s: lm-prefill: LLAMA_LM_FAIL_ACQUIRE: refusing the resources\n", __func__);
        return false;
    }

    const int64_t t_a0 = ggml_time_us();
    synchronize();

    const int64_t seg_tokens = std::min<int64_t>(n_tokens, P.segment);
    const size_t  elt        = ggml_type_size(S.act_type);
    const size_t  half       = (size_t) model.hparams.n_embd_inp() * (size_t) seg_tokens * elt;

    size_t slot_bytes = 0;
    for (const auto & slot : S.ls->slots) {
        for (const auto & kv : slot.layer_bytes) {
            slot_bytes = std::max(slot_bytes, kv.second);
        }
    }
    const int64_t slot_total = (int64_t) (slot_bytes + 4096) * P.n_slots;
    int64_t sched_est = S.sched_est > 0 ? (int64_t) S.sched_est : (int64_t) ((double) P.ubatch/2048.0*560.0*1048576.0) + (int64_t) 96*1048576;
    const bool dense = model.arch == LLM_ARCH_QWEN35;
    if (dense) {
        // a dense layer's pieces hold the FFN activations (gate, up and their product: 3 x n_ff floats per token), the widest projection output pair and a few residual-stream copies.
        // The formula bounds the footprint from above; measured on Qwen3.8-27B (all samples of 12 logs): 63 MiB at 256 tokens, 125-126 at 512, 244-304 at 1024, 282-599 at 2048, 1108 at 4096, i.e.
        // at most 0.29 MiB per token against the formula's 0.37. The first pass (nothing measured yet) asks for 0.8 x the formula + 16 MiB, which is above every one of those samples (the
        // 160 MiB of margin sits on top); the next ones use what the pass really took. At 1.0 x + 64 MiB a card with ~640 MiB to spare never ran a single pass at chunk 1024.
        const int64_t per_tok = 4*(3*(int64_t) model.hparams.n_ff(0) + 2*(int64_t) std::max<int64_t>(2*model.hparams.n_head(0)*model.hparams.n_embd_head_k(0), (int64_t) model.hparams.ssm_d_inner + 2*model.hparams.ssm_n_group*model.hparams.ssm_d_state) + 4*(int64_t) model.hparams.n_embd);
        sched_est = (int64_t) (0.8*(double) per_tok*(double) wmax) + (int64_t) 16*1048576;
    }
    // G40: the scheduler's buffer is not all a pass costs (CUDA pools, graph temporaries): ask for what the last pass really took, scaled to this plan's widest chunk.
    // The footprint is ~0.25 MiB per token of the widest chunk plus a fixed ~8 MiB (measured: 4 tokens 8 MiB, 256 -> 63, 512 -> 125, 1024 -> 246, 2048 -> 525, 4096 -> 1108), so the scale goes both
    // ways (a narrower pass needs less) and carries its own fixed term; only passes of >= 256 tokens calibrate it (see lm_release)
    if (S.var_meas > 0 && S.var_w > 0) {
        const double scale = (double) wmax/(double) S.var_w;
        const int64_t meas = (int64_t) ((double) S.var_meas*scale) + (int64_t) 16*1048576 + (int64_t) 32*1048576;
        sched_est = dense ? meas : std::max(sched_est, meas);
    }
    const int64_t need = slot_total + 2*(int64_t) half + sched_est + (int64_t) 64*1048576;

    if (llama_lm_vram_hook != nullptr && !llama_lm_vram_hook(this, need, 1)) {
        LLAMA_LOG_WARN("%s: lm-prefill: could not make %.0f MiB of device memory available; this batch takes the ordinary path\n", __func__, (double) need/1048576.0);
        return false;
    }
    if (llama_lm_vram_hook == nullptr) {
        // no expert cache to shrink (dense models, MoE without the hybrid engine): the room has to exist already. A pass that runs out of device memory half way has changed the cache, and the
        // allocator of ggml does not survive a failed reservation, so the check is made BEFORE anything is touched
        const int64_t margin = (int64_t) 96*1048576;
        const int64_t held   = dstream ? (int64_t) dstream->held_bytes() : 0;     // the decode ring of slots gives its memory back for the pass
        const int64_t free_b = lm_dev_free() + held;
        if (free_b < need + margin) {
            LLAMA_LOG_WARN("%s: lm-prefill: %.0f MiB of device memory are free, the pass needs %.0f MiB (slot window %.0f + slab %.0f + compute buffer %.0f + margin %.0f); this batch takes the ordinary path\n",
                           __func__, (double) free_b/1048576.0, (double) (need + margin)/1048576.0, (double) slot_total/1048576.0, 2*(double) half/1048576.0, (double) sched_est/1048576.0, (double) (margin + 64*1048576)/1048576.0);
            return false;
        }
    }
    if (dstream && dstream->held_bytes() > 0) {
        dstream->suspend();
        reset_graph_reuse();     // a graph built while the ring is away has no markers and must not be reused after it is back
    }
    S.free_pre   = lm_dev_free();
    S.slot_total = slot_total;
    S.cur_w      = wmax;
    S.vram_hooked = llama_lm_vram_hook != nullptr;

    S.slots_live = true;    // also for a window that is only partly allocated: lm_release frees what exists
    if (!S.ls->lm_alloc_slots()) {
        LLAMA_LOG_WARN("%s: lm-prefill: the slot window could not be allocated (%s); this batch takes the ordinary path\n", __func__, S.ls->error_msg.c_str());
        lm_release();
        return false;
    }

    S.slab.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(P.device), 2*half));
    if (!S.slab) {
        LLAMA_LOG_WARN("%s: lm-prefill: the activation slab (%.1f MiB) could not be allocated; this batch takes the ordinary path\n", __func__, 2*(double) half/1048576.0);
        lm_release();
        return false;
    }
    S.slab_half   = half;
    S.slab_tokens = seg_tokens;
    S.t_acquire_us = ggml_time_us() - t_a0;

    // the pass runs on its own scheduler: its compute buffer is allocated by the pass and returned with it, and the context's persistent decode graphs keep the buffer they were planned on.
    // It is swapped in by lm_run (decode() reserves the context's own scheduler first).
    S.sched_pp.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), 8192, false, cparams.op_offload));
    if (!S.sched_pp) {
        LLAMA_LOG_WARN("%s: lm-prefill: the pass scheduler could not be created; this batch takes the ordinary path\n", __func__);
        lm_release();
        return false;
    }

    S.live = true;
    return true;
}

void llama_context::lm_release() {
    auto & S = *lm_state;
    const int64_t t_r0 = ggml_time_us();

    // G40: what this pass really took on the device besides the slot window and the slab (the next pass asks for it)
    if (S.live && S.free_pre > 0) {
        const int64_t var = S.free_pre - lm_dev_free() - S.slot_total - 2*(int64_t) S.slab_half;
        // a pass of a few tokens (the 4-token tail behind a checkpoint) measures only the fixed cost; scaled by the width ratio it asked for 2 GB and starved the next pass (found 2026-10-20)
        if (var > 0 && S.cur_w >= 256) {
            S.var_meas = var;
            S.var_w    = S.cur_w;
        }
        if (lm_log_on()) {
            LLAMA_LOG_WARN("lm-prefill: footprint beyond the slot window and the slab: %.0f MiB (widest chunk %d)\n", (double) var/1048576.0, (int) S.cur_w);
        }
    }
    S.free_pre = 0;

    if (S.sched_swapped) {
        ggml_backend_sched_synchronize(sched.get());
        S.sched_est = 0;
        for (auto * b : backend_ptrs) {
            if (ggml_backend_is_cuda(b)) {
                S.sched_est = std::max<size_t>(S.sched_est, ggml_backend_sched_get_buffer_size(sched.get(), b));
            }
        }
        std::swap(sched, S.sched_pp);
        S.sched_swapped = false;
        S.sched_pp.reset();     // frees the pass's compute buffer
    }
    S.sched_pp.reset();
    if (S.slots_live) {
        S.ls->lm_release_slots();
        S.slots_live = false;
    }
    S.slab.reset();
    S.live = false;
    S.armed = false;
    S.armed_batch = false;

    if (dstream && dstream->is_suspended()) {
        if (!dstream->resume()) {
            LLAMA_LOG_ERROR("%s: dense-stream: could not take the ring of slots back from the pass\n", __func__);
        }
        reset_graph_reuse();
    }
    const int64_t t_r1 = ggml_time_us();
    if (S.vram_hooked && llama_lm_vram_hook != nullptr) {
        llama_lm_vram_hook(this, 0, 0);
    }
    S.vram_hooked = false;
    if (lm_log_on()) {
        LLAMA_LOG_WARN("lm-prefill: resources: acquire %.0f ms, release %.0f ms (+ %.0f ms for the expert cache)\n", (double) S.t_acquire_us/1e3, (double) (t_r1 - t_r0)/1e3, (double) (ggml_time_us() - t_r1)/1e3);
    }
}

//
// the pass
//

// returns 0 when the batch is done, 1 when nothing was run (the caller takes the ordinary path), < 0 on a failure after the memory was changed
int llama_context::lm_run(const llama_lm_plan & pl, uint32_t n_outputs_all) {
    auto & S  = *lm_state;
    auto & ls = *S.ls;
    const auto & P = S.p;
    const auto & hparams = model.hparams;

    const int64_t t_start = ggml_time_us();

    const int      n_layer   = (int) hparams.n_layer();
    const int64_t  n_embd_io = hparams.n_embd_inp();
    const size_t   elt       = ggml_type_size(S.act_type);
    const size_t   row_bytes = (size_t) n_embd_io * elt;
    const bool     output_all = cparams.embeddings;
    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;
    const llama_batch & B = balloc->get_batch();

    ggml_backend_t backend_cuda = nullptr;
    for (auto * b : backend_ptrs) {
        if (ggml_backend_is_cuda(b)) {
            backend_cuda = b;
            break;
        }
    }
    if (backend_cuda == nullptr) {
        return 1;
    }
    ls.backend_compute = backend_cuda;
    ls.phase           = 0;

    // test hooks: LLAMA_LM_FAIL_RUN=1 declines before anything is changed (the ordinary path takes the batch), =2 fails after the first layer of the first chunk (the memory was changed)
    static const int fail_mode = getenv("LLAMA_LM_FAIL_RUN") != nullptr ? atoi(getenv("LLAMA_LM_FAIL_RUN")) : 0;
    static const int fail_at   = getenv("LLAMA_LM_FAIL_AT") != nullptr ? atoi(getenv("LLAMA_LM_FAIL_AT")) : 1;     // the n-th pass attempt (1-based) fails; 0 = every attempt
    S.n_attempt++;
    const int fail_run = (fail_at == 0 || S.n_attempt == fail_at) ? fail_mode : 0;
    if (fail_run == 1) {
        LLAMA_LOG_WARN("%s: lm-prefill: LLAMA_LM_FAIL_RUN=1: declining the batch\n", __func__);
        return 1;
    }

    // A context with rollback snapshots (MTP, n_rs_seq > 0) keeps the state in (1 + n_rs_seq) planes and builds the snapshot-writing graph. The pass writes plane 0 with the plain graph
    // and splits its chunks without the keep-tail rule, so the graphs and the memory see n_rs_seq == 0 until the pass is over (the plan checked that the seq stands in plane 0).
    struct rs_plain_t {
        uint32_t * cp;
        uint32_t * mem;
        uint32_t   cp0;
        uint32_t   mem0;
        rs_plain_t(uint32_t * cp, uint32_t * mem) : cp(cp), mem(mem), cp0(*cp), mem0(*mem) { *cp = 0; *mem = 0; }
        ~rs_plain_t() { *cp = cp0; *mem = mem0; }
    } rs_plain { &cparams.n_rs_seq, &static_cast<llama_memory_hybrid *>(memory.get())->get_mem_recr()->n_rs_seq };

    // the outputs the ordinary decode would reserve (and reset), before anything is changed
    if (output_reserve(n_outputs_all) < n_outputs_all || (n_outputs_all > 0 && logits.data == nullptr)) {
        return 1;
    }

    // from here on `sched` is the pass's scheduler (S.sched_pp holds the context's own until lm_release swaps back)
    std::swap(sched, S.sched_pp);
    S.sched_swapped = true;

    const int n_chunks = (int) pl.widths.size();
    std::vector<int64_t> chunk_t0(n_chunks);
    {
        int64_t t = 0;
        for (int c = 0; c < n_chunks; ++c) {
            chunk_t0[c] = t;
            t += pl.widths[c];
        }
    }
    // segments: as many whole chunks as fit in `segment` rows (the slab half has that many rows); the first chunk of a resumed pass can be short
    std::vector<int> seg_first;
    {
        int64_t rows = 0;
        for (int c = 0; c < n_chunks; ++c) {
            if (c == 0 || rows + pl.widths[c] > P.segment) {
                seg_first.push_back(c);
                rows = 0;
            }
            rows += pl.widths[c];
        }
        seg_first.push_back(n_chunks);
    }
    const int n_segs = (int) seg_first.size() - 1;

    // one result object for every piece of the pass
    llm_graph_result_ptr res(new llm_graph_result(4096));

    double t_embed = 0.0, t_build = 0.0, t_alloc = 0.0, t_inputs = 0.0, t_compute = 0.0, t_slotwait = 0.0, t_head = 0.0;
    int64_t n_launch = 0;
    int64_t n_recurrent_prep = 0;

    struct chunk_t {
        llama_memory_context_ptr mctx;
        int32_t                  n = 0;
        int64_t                  t0 = 0;       // first token of the chunk inside the batch
        int64_t                  slab_row = 0; // first row of the chunk inside the segment's slab half
    };
    std::vector<chunk_t> chunks((size_t) n_chunks);

    // G42: the recurrent state at the plan's cuts: rows copied to the host when the chunk that ends at a cut has gone through a recurrent layer
    struct snap_rows_t {
        std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> rows;
    };
    std::vector<snap_rows_t> snap_rows(pl.snap_chunk.size());
    std::vector<int>         snap_of_chunk((size_t) n_chunks, -1);
    for (size_t k = 0; k < pl.snap_chunk.size(); ++k) {
        snap_of_chunk[(size_t) pl.snap_chunk[k]] = (int) k;
    }
    double t_snap_copy = 0.0;

    const llama_memory_context_i * last_mctx = nullptr;
    const llama_ubatch *           last_ub   = nullptr;

    bool memory_changed = false;
    auto fail = [&](const char * what, int code) -> int {
        LLAMA_LOG_ERROR("%s: lm-prefill: %s\n", __func__, what);
        if (memory_changed) {
            // the cache is in an unknown state: drop the sequence, the caller reports the failure
            memory->seq_rm(pl.seq, -1, -1);
            n_outputs = 0;
            return code < 0 ? code : -3;
        }
        return code;
    };

    // build, allocate, set the inputs and run one piece; the piece is synchronised before the call returns (the next piece reuses the compute buffer)
    auto run_piece = [&](const llm_graph_lm & lmi, const llama_ubatch & ub, const llama_memory_context_i * mc, double & t_piece_compute) -> bool {
        const int64_t t0 = ggml_time_us();
        res->reset();
        ggml_backend_sched_reset(sched.get());
        n_outputs = lmi.kind == 2 ? 1 : 0;
        llm_graph_params gp = graph_params(res.get(), ub, mc, LLM_GRAPH_TYPE_DEFAULT);
        gp.layer_streaming = nullptr;
        gp.lm              = &lmi;
        ggml_cgraph * gf = model.build_graph(gp);
        if (gf == nullptr) {
            LLAMA_LOG_ERROR("%s: lm-prefill: graph build failed (kind %d, layer %d)\n", __func__, lmi.kind, lmi.il);
            return false;
        }
        const int64_t t1 = ggml_time_us();
        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: lm-prefill: graph allocation failed (kind %d, layer %d)\n", __func__, lmi.kind, lmi.il);
            return false;
        }
        const int64_t t2 = ggml_time_us();
        res->set_inputs(&ub);
        const int64_t t3 = ggml_time_us();
        const ggml_status st = graph_compute(gf, true);
        if (st != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: lm-prefill: graph compute failed (kind %d, layer %d, status %d)\n", __func__, lmi.kind, lmi.il, (int) st);
            return false;
        }
        ggml_backend_sched_synchronize(sched.get());
        const int64_t t4 = ggml_time_us();
        t_build        += (double) (t1 - t0);
        t_alloc        += (double) (t2 - t1);
        t_inputs       += (double) (t3 - t2);
        t_piece_compute = (double) (t4 - t3);
        t_compute      += t_piece_compute;
        n_launch++;
        return true;
    };

    char * slab_base = (char *) ggml_backend_buffer_get_base(S.slab.get());

    // find_slot places every chunk at exactly the cells the plan checked (c0 + position); cleared on every way out of the pass
    struct force_guard_t {
        llama_kv_cache * kv;
        ~force_guard_t() { kv->lm_force(-1); }
    } force_guard { static_cast<llama_memory_hybrid *>(memory.get())->get_mem_attn() };

    auto half_ptr = [&](int which, int64_t row) -> void * {
        return slab_base + (size_t) which*S.slab_half + (size_t) row*row_bytes;
    };

    // is layer il computed from the model's own (already device-resident) experts?
    auto is_native = [&](int il) { return S.native[(size_t) il] != 0; };
    auto layer_of  = [&](int il) -> const llama_layer & { return is_native(il) ? model.layers[il] : ls.slot_layer(il); };

    for (int seg = 0; seg < n_segs; ++seg) {
        const int c0 = seg_first[(size_t) seg];
        const int c1 = seg_first[(size_t) seg + 1];

        // the previous segment is complete for every layer: its memory contexts go, the next chunks are applied behind it (KV cells and recurrent state continue where it left them)
        for (int c = 0; c < c0; ++c) {
            chunks[c].mctx.reset();
        }

        // 1. apply the chunks' memory contexts: the KV cells, the recurrent cell and the inputs of every chunk are fixed here
        int64_t row = 0;
        for (int c = c0; c < c1; ++c) {
            const int32_t n_c = pl.widths[c];
            llama_batch cb = {
                /*.n_tokens =*/ n_c,
                /*.token    =*/ B.token + chunk_t0[c],
                /*.embd     =*/ nullptr,
                /*.pos      =*/ B.pos + chunk_t0[c],
                /*.n_seq_id =*/ B.n_seq_id + chunk_t0[c],
                /*.seq_id   =*/ B.seq_id + chunk_t0[c],
                /*.logits   =*/ nullptr,
            };
            llama_batch_allocr vballoc(hparams.n_pos_per_embd());
            llama_batch_compat vcompat(this, cb);   // llama_batch_ext since upstream #24669
            if (!vballoc.init(*vcompat.batch_ext, model.vocab, output_all)) {
                return fail("could not initialise a chunk batch", 1);
            }
            force_guard.kv->lm_force(pl.c0 + pl.p0 + chunk_t0[c]);
            auto vmctx = memory->init_batch(vballoc, n_c, output_all);
            if (!vmctx || vmctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
                return fail("the memory could not take a chunk", 1);
            }
            memory_changed = true;
            if (!vmctx->apply()) {
                return fail("the memory could not apply a chunk", -3);
            }

            if (c == 0) {
                // the first chunk of the pass: make the recurrent state cell what the in-graph "zero this cell, then gather from src" would have made it, and let every
                // chunk (this one included) run with the identity gather it will see once all chunks are applied
                const auto * mh = static_cast<const llama_memory_hybrid_context *>(vmctx.get());
                const auto * rc = mh->get_recr();
                const int32_t  rs_z = rc->get_rs_z();
                const uint32_t head = rc->get_head();
                const int32_t  src0 = rc->s_copy(0);
                if (rc->get_n_rs() != 1) {
                    return fail("unexpected recurrent cell range", -3);
                }
                if (rs_z >= 0 || src0 != (int32_t) head) {
                    n_recurrent_prep++;
                    std::vector<uint8_t> tmp;
                    for (int il = 0; il < n_layer; ++il) {
                        if (!hparams.is_recr(il)) {
                            continue;
                        }
                        for (ggml_tensor * t : { rc->get_r_l(il), rc->get_s_l(il) }) {
                            if (t == nullptr) {
                                continue;
                            }
                            const size_t rb = t->nb[1];
                            if (rs_z >= 0) {
                                ggml_backend_tensor_memset(t, 0, (size_t) rs_z*rb, rb);
                            }
                            if (src0 != (int32_t) head) {
                                tmp.resize(rb);
                                ggml_backend_tensor_get(t, tmp.data(), (size_t) src0*rb, rb);
                                ggml_backend_tensor_set(t, tmp.data(), (size_t) head*rb, rb);
                            }
                        }
                    }
                }
            }

            chunks[c].mctx     = std::move(vmctx);
            chunks[c].n        = n_c;
            chunks[c].t0       = chunk_t0[c];
            chunks[c].slab_row = row;
            row += n_c;
            last_mctx = chunks[c].mctx.get();
            last_ub   = &chunks[c].mctx->get_ubatch();
        }

        // 2. embed every chunk (slab half 0)
        {
            const int64_t t_e0 = ggml_time_us();
            for (int c = c0; c < c1; ++c) {
                llm_graph_lm lmi;
                lmi.kind     = 0;
                lmi.act_buf  = S.slab.get();
                lmi.act_type = S.act_type;
                lmi.act_out  = half_ptr(0, chunks[c].slab_row);
                double tc = 0.0;
                if (!run_piece(lmi, chunks[c].mctx->get_ubatch(), chunks[c].mctx.get(), tc)) {
                    return fail("embedding failed", -3);
                }
            }
            t_embed += (double) (ggml_time_us() - t_e0);
        }

        // 3. preload the first n_slots streamed layers, seed the async feed
        {
            std::vector<uint8_t> slot_taken((size_t) ls.n_slots, 0);
            int preloaded = 0;
            int last_pre  = -1;
            for (int il = 0; il < n_layer && preloaded < ls.n_slots; ++il) {
                if (is_native(il) || slot_taken[(size_t) (il % ls.n_slots)]) {
                    continue;
                }
                slot_taken[(size_t) (il % ls.n_slots)] = 1;
                bool copied = false;
                if (!ls.load_layer(il, copied)) {
                    return fail(ls.error_msg.c_str(), -3);
                }
                last_pre = il;
                preloaded++;
            }
            if (ls.af && last_pre >= 0) {
                const int nxt = ls.af_next_layer(last_pre);
                if (nxt >= 0) {
                    ls.af_push(nxt);
                }
            }
        }

        // 4. layers on the outside, chunks on the inside (ascending)
        for (int il = 0; il < n_layer; ++il) {
            const bool native = is_native(il);
            if (!native) {
                bool copied = false;
                if (!ls.load_layer(il, copied)) {
                    return fail(ls.error_msg.c_str(), -3);
                }
                double wait_us = 0.0, host_submit_us = 0.0, ready_latency_us = 0.0;
                if (!ls.wait_slot_ready(il, wait_us, host_submit_us, ready_latency_us)) {
                    return fail(ls.error_msg.c_str(), -3);
                }
                t_slotwait += wait_us;
                {
                    const int nxt = ls.af_next_layer(il);
                    if (nxt >= 0 && ls.af) {
                        ls.af_push(nxt);
                    }
                }
                ggml_backend_event_wait(backend_cuda, (ggml_backend_event_t) ls.slots[il % ls.n_slots].ev_copy_ready);
            }

            if (fail_run == 2 && il == 1) {
                return fail("LLAMA_LM_FAIL_RUN=2: injected failure", -3);
            }
            for (int c = c0; c < c1; ++c) {
                llm_graph_lm lmi;
                lmi.kind       = 1;
                lmi.il         = il;
                lmi.layer      = &layer_of(il);
                lmi.act_buf    = S.slab.get();
                lmi.act_type   = S.act_type;
                lmi.act_in     = half_ptr(il & 1, chunks[c].slab_row);
                lmi.act_out    = half_ptr((il + 1) & 1, chunks[c].slab_row);
                lmi.q_pos_base = (int32_t) (pl.p0 + chunks[c].t0);
                lmi.n_kv_valid = lmi.q_pos_base + chunks[c].n;
                lmi.kv_cell0   = (int32_t) pl.c0;
                double tc = 0.0;
                if (!run_piece(lmi, chunks[c].mctx->get_ubatch(), chunks[c].mctx.get(), tc)) {
                    return fail("layer failed", -3);
                }
                if (snap_of_chunk[(size_t) c] >= 0 && hparams.is_recr(il)) {
                    // G42: the layer's state now is the state at the end of this chunk (the piece is synchronised)
                    const int64_t t_s0 = ggml_time_us();
                    const auto * rcs = static_cast<const llama_memory_hybrid_context *>(chunks[c].mctx.get())->get_recr();
                    const uint32_t hd = rcs->get_head();
                    auto & sr = snap_rows[(size_t) snap_of_chunk[(size_t) c]];
                    for (ggml_tensor * t : { rcs->get_r_l(il), rcs->get_s_l(il) }) {
                        if (t == nullptr) {
                            continue;
                        }
                        const size_t rb = t->nb[1];
                        sr.rows.emplace_back(t, std::vector<uint8_t>(rb));
                        ggml_backend_tensor_get(t, sr.rows.back().second.data(), (size_t) hd*rb, rb);
                    }
                    t_snap_copy += (double) (ggml_time_us() - t_s0);
                }
            }

            if (!native) {
                ls.record_consumer_done(il, backend_cuda);
                int next = il + ls.n_slots;
                while (next < n_layer && is_native(next)) {
                    next += ls.n_slots;
                }
                if (next < n_layer) {
                    if (ls.af) {
                        ls.af_push(next);
                    } else {
                        bool c2 = false;
                        if (!ls.load_layer(next, c2)) {
                            return fail(ls.error_msg.c_str(), -3);
                        }
                    }
                }
            }
        }

        // 5. the head of the last token of the last segment
        if (seg == n_segs - 1 && pl.want_logits) {
            const int64_t t_h0 = ggml_time_us();
            const auto & lc = chunks[c1 - 1];
            llm_graph_lm lmi;
            lmi.kind     = 2;
            lmi.act_buf  = S.slab.get();
            lmi.act_type = S.act_type;
            lmi.act_in   = half_ptr(n_layer & 1, lc.slab_row + lc.n - 1);
            double tc = 0.0;
            if (!run_piece(lmi, lc.mctx->get_ubatch(), lc.mctx.get(), tc)) {
                return fail("head failed", -3);
            }
            ggml_tensor * t_logits = res->get_logits();
            GGML_ASSERT(t_logits != nullptr && t_logits->ne[0] == (int64_t) model.vocab.n_tokens() && t_logits->ne[1] == 1);
            ggml_backend_tensor_get(t_logits, logits.data, 0, ggml_nbytes(t_logits));
            t_head += (double) (ggml_time_us() - t_h0);
        }
    }

    // the feed worker and the copy stream are idle before the slot window goes
    ls.af_wait_idle();
    ls.ls_stage_drain();
    for (auto & c : chunks) {
        c.mctx.reset();
    }
    (void) last_mctx;
    (void) last_ub;

    // G42: the snapshots become the blobs the caller stores as checkpoints (the real serializer, the host rows and the cell position substituted)
    S.snaps.clear();
    if (!snap_rows.empty()) {
        const int64_t t_b0 = ggml_time_us();
        llama_memory_recurrent * mr = static_cast<llama_memory_hybrid *>(memory.get())->get_mem_recr();
        bool snap_ok = true;
        for (size_t k = 0; k < snap_rows.size() && snap_ok; ++k) {
            llama_lm_state::snap_t sn;
            sn.cut = pl.snap_cuts[k];
            {
                size_t est = 4096;      // the blob is the rows plus a few headers: one allocation, no regrowth
                for (const auto & r : snap_rows[k].rows) {
                    est += r.second.size();
                }
                sn.blob.reserve(est);
            }
            llama_io_write_lm_snap io(sn.blob, snap_rows[k].rows);
            io.write(&k_lm_io_magic, sizeof(k_lm_io_magic));
            io.write(&pl.seq, sizeof(pl.seq));
            mr->state_pos_override = pl.p0 + pl.snap_cuts[k] - 1;
            try {
                mr->state_write(io, pl.seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            } catch (const std::exception & e) {
                LLAMA_LOG_ERROR("%s: lm-prefill: snapshot %d at %d tokens could not be written: %s\n", __func__, (int) k, (int) pl.snap_cuts[k], e.what());
                snap_ok = false;
            }
            mr->state_pos_override = -1;
            if (snap_ok) {
                if (lm_log_on()) {
                    static const bool hash_on = getenv("LLAMA_LM_HASH") != nullptr;     // a byte loop over 63 MiB: ~100 ms, never in a timing run
                    LLAMA_LOG_WARN("lm-prefill: snapshot %d: %d tokens into the batch (position %d), %zu bytes, fnv %016llx\n", (int) k, (int) pl.snap_cuts[k], (int) (pl.p0 + pl.snap_cuts[k] - 1), sn.blob.size(),
                                   (unsigned long long) (hash_on ? lm_fnv64(sn.blob.data(), sn.blob.size()) : 0));
                }
                S.snaps.push_back(std::move(sn));
            }
        }
        if (!snap_ok) {
            S.snaps.clear();
        }
        S.n_snap += (int64_t) S.snaps.size();
        if (lm_log_on()) {
            LLAMA_LOG_WARN("lm-prefill: %d snapshots: copies %.0f ms, blobs %.0f ms\n", (int) S.snaps.size(), t_snap_copy/1e3, (double) (ggml_time_us() - t_b0)/1e3);
        }
    }

    // outputs as the ordinary decode would leave them
    n_outputs = n_outputs_all;
    if (n_outputs_all > 0) {
        GGML_ASSERT(n_outputs_all == 1 && pl.want_logits);
        output_ids[pl.n - 1] = 0;
    }

    const double wall_s = (double) (ggml_time_us() - t_start) / 1e6;
    S.n_pass++;
    S.n_tok   += pl.n;
    S.t_total += wall_s;
    if (lm_log_on()) {
        LLAMA_LOG_WARN("lm-prefill: pass %lld: %d tokens at position %d in %d chunks / %d segments: %.3f s (%.0f t/s); embed %.0f ms, slot wait %.0f ms, head %.0f ms, %lld launches: build %.0f ms, alloc %.0f ms, inputs %.0f ms, compute %.0f ms (sum), recurrent prep %lld\n",
                       (long long) S.n_pass, (int) pl.n, (int) pl.p0, n_chunks, n_segs, wall_s, (double) pl.n/wall_s, t_embed/1e3, t_slotwait/1e3, t_head/1e3, (long long) n_launch,
                       t_build/1e3, t_alloc/1e3, t_inputs/1e3, t_compute/1e3, (long long) n_recurrent_prep);
    }
    return 0;
}

//
// G42: the pass prepared in the idle time, and the snapshots
//

bool llama_context::lm_stage_first() {
    auto & S  = *lm_state;
    auto & ls = *S.ls;
    ggml_backend_t backend_cuda = nullptr;
    for (auto * b : backend_ptrs) {
        if (ggml_backend_is_cuda(b)) {
            backend_cuda = b;
            break;
        }
    }
    if (backend_cuda == nullptr) {
        return false;
    }
    ls.backend_compute = backend_cuda;
    ls.phase           = 0;
    // the same layers the pass preloads at the start of a segment: the first n_slots streamed ones, one per slot
    std::vector<uint8_t> slot_taken((size_t) ls.n_slots, 0);
    int preloaded = 0;
    for (int il = 0; il < S.n_layer && preloaded < ls.n_slots; ++il) {
        if (S.native[(size_t) il] != 0 || slot_taken[(size_t) (il % ls.n_slots)]) {
            continue;
        }
        slot_taken[(size_t) (il % ls.n_slots)] = 1;
        bool copied = false;
        if (!ls.load_layer(il, copied)) {
            LLAMA_LOG_WARN("%s: lm-prefill: the first banks could not be staged: %s\n", __func__, ls.error_msg.c_str());
            return false;
        }
        preloaded++;
    }
    return true;
}

bool llama_context::lm_prearm(llama_seq_id seq, int64_t n_tokens) {
    if (lm_state == nullptr || n_tokens < 1) {
        return false;
    }
    auto & S = *lm_state;
    const auto & P = S.p;
    if (S.live && !S.armed) {
        return false;     // a pass is running
    }
    if (cparams.embeddings || cparams.embeddings_nextn || !sampling.samplers.empty() || !cparams.causal_attn || !cparams.flash_attn || cparams.n_rs_seq != 0 ||
        cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || layer_streaming != nullptr) {
        return false;
    }
    auto * mh = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (mh == nullptr) {
        return false;
    }
    if (seq >= 0) {
        // the checks lm_make_plan makes for the batch itself
        std::string detail;
        const llama_pos p0 = memory->seq_pos_max(seq) + 1;
        const bool ok = (uint32_t) seq < cparams.n_seq_max && !mh->get_mem_attn()->get_has_shift() &&
                        mh->get_mem_attn()->lm_block(seq, p0, (uint32_t) n_tokens, &detail) >= 0 && mh->get_mem_recr()->seq_pos_max(seq) == p0 - 1;
        if (!ok) {
            if (lm_log_on()) {
                LLAMA_LOG_WARN("lm-prefill: not armed for %lld tokens of sequence %d at position %d: %s\n", (long long) n_tokens, (int) seq, (int) p0, detail.empty() ? "the recurrent state or the cache layout" : detail.c_str());
            }
            return false;
        }
    }
    if (seq >= 0 && (output_reserve(1) < 1 || logits.data == nullptr)) {
        // lm_run declines a batch whose outputs cannot be reserved: find out now, while the caller can still cut the batch at its checkpoints
        return false;
    }
    const int64_t rows = std::min<int64_t>(n_tokens, P.segment);
    const int32_t wmax = (int32_t) std::min<int64_t>(n_tokens, P.ubatch);
    if (S.armed) {
        if (S.slab_tokens >= rows && S.armed_w >= wmax) {
            S.armed_batch = S.armed_batch || seq >= 0;
            if (seq >= 0) {
                S.armed_n = n_tokens;
            }
            return true;
        }
        S.free_pre = 0;
        lm_release();      // too small for this batch: acquire again (the cache is shrunk already)
    }
    const int64_t t0 = ggml_time_us();
    if (!lm_acquire_n(n_tokens, wmax)) {
        return false;
    }
    if (!lm_stage_first()) {
        S.free_pre = 0;
        lm_release();
        return false;
    }
    // idle preparation: the copies are queued on the copy stream, but a WDDM driver keeps the command buffer until a synchronising call, so the banks would arrive only with the next request's first
    // synchronisation (73 ms for a checkpoint restore that takes 10 ms otherwise, k5453): wait for them here, in the idle time
    static const bool no_sync = getenv("LLAMA_LM_ARM_NOSYNC") != nullptr;
    if (seq < 0 && !no_sync) {
        const int64_t t_s0 = ggml_time_us();
        S.ls->ls_stage_drain();
        if (lm_log_on()) {
            LLAMA_LOG_WARN("lm-prefill: the staged banks of the arm are in place after %.1f ms more\n", (double) (ggml_time_us() - t_s0)/1e3);
        }
    }
    S.armed       = true;
    S.armed_batch = seq >= 0;
    S.armed_n     = seq >= 0 ? n_tokens : 0;
    S.armed_w     = wmax;
    S.n_arm++;
    S.t_arm_us    = ggml_time_us() - t0;
    if (lm_log_on()) {
        LLAMA_LOG_WARN("lm-prefill: armed for %lld tokens (%s): %.0f ms (room and allocation %.0f ms)\n", (long long) n_tokens, seq >= 0 ? "a batch" : "the idle time", (double) S.t_arm_us/1e3, (double) S.t_acquire_us/1e3);
    }
    return true;
}

void llama_context::lm_disarm() {
    if (lm_state != nullptr && lm_state->armed) {
        lm_state->n_arm_wasted++;
        if (lm_log_on()) {
            LLAMA_LOG_WARN("lm-prefill: disarmed: the prepared pass was not used\n");
        }
        lm_state->free_pre = 0;
        lm_release();
    }
}

bool llama_context::lm_decode_begin(const llama_batch & b, llama_lm_plan & plan) {
    auto & S = *lm_state;
    S.snaps.clear();                       // the snapshots of an earlier decode go
    plan = lm_make_plan(b);
    S.snap_req.clear();                    // a request holds for one decode
    if (!plan.ok) {
        S.armed_batch = false;             // the batch armed for was not this one
    }
    lm_note(b, plan);
    if (plan.ok) {
        const bool on = lm_acquire(plan);
        plan.ok = on;
        return on;
    }
    if (S.armed && (b.n_tokens <= 16 || b.n_tokens >= S.p.min_tokens)) {
        // a decode step (the cache must be whole again) or a big batch the pass cannot take (the prefill window may open): the prepared resources go back
        lm_disarm();
    }
    return false;
}

void llama_context::lm_release_if_live() {
    if (lm_state != nullptr && lm_state->live) {
        lm_state->free_pre = 0;
        lm_release();
    }
}

void llama_context::lm_snap_request(const int32_t * cuts, int32_t n_cuts) {
    if (lm_state == nullptr) {
        return;
    }
    auto & v = lm_state->snap_req;
    v.clear();
    for (int32_t i = 0; cuts != nullptr && i < n_cuts; ++i) {
        if (cuts[i] > 0) {
            v.push_back(cuts[i]);
        }
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

int32_t llama_context::lm_snap_count() const {
    return lm_state != nullptr ? (int32_t) lm_state->snaps.size() : 0;
}

int32_t llama_context::lm_snap_cut(int32_t i) const {
    return lm_state != nullptr && i >= 0 && (size_t) i < lm_state->snaps.size() ? lm_state->snaps[(size_t) i].cut : -1;
}

size_t llama_context::lm_snap_data(int32_t i, uint8_t * dst, size_t size) const {
    if (lm_state == nullptr || i < 0 || (size_t) i >= lm_state->snaps.size()) {
        return 0;
    }
    const auto & b = lm_state->snaps[(size_t) i].blob;
    if (dst == nullptr) {
        return b.size();
    }
    if (size < b.size()) {
        return 0;
    }
    memcpy(dst, b.data(), b.size());
    return b.size();
}

int64_t llama_context::lm_n_pass() const {
    return lm_state != nullptr ? lm_state->n_pass : 0;
}

bool llama_context::lm_is_armed() const {
    return lm_state != nullptr && lm_state->armed;
}

//
// enable / disable
//

bool llama_context::lm_prefill_enable(const llama_lm_prefill_params & p, std::string & err) {
    if (lm_state != nullptr) {
        err = "already enabled";
        return false;
    }
    const bool dense = model.arch == LLM_ARCH_QWEN35;
    if (model.arch != LLM_ARCH_QWEN35MOE && !dense) {
        err = "the layer-major prefill supports the Qwen3.5/3.6 MoE and the dense Qwen3.5/3.8 architectures only";
        return false;
    }
    auto * mh = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (mh == nullptr) {
        err = "the context has no hybrid (attention + recurrent) memory";
        return false;
    }
    if (!cparams.flash_attn || !cparams.causal_attn) {
        err = "flash attention is required (the prefill uses its implicit causal mask)";
        return false;
    }
    if (p.n_slots < 1 || p.n_slots > 16 || p.ubatch < 1 || p.segment < p.ubatch || p.segment % p.ubatch != 0 || p.min_tokens < 1) {
        err = "invalid parameters (slots 1..16, ubatch >= 1, segment a multiple of ubatch, min_tokens >= 1)";
        return false;
    }
    bool have_cuda = false;
    for (auto * b : backend_ptrs) {
        have_cuda = have_cuda || ggml_backend_is_cuda(b);
    }
    if (!have_cuda) {
        err = "no CUDA backend";
        return false;
    }

    auto S = std::unique_ptr<llama_lm_state, llama_lm_state_deleter>(new llama_lm_state());
    S->p        = p;
    S->act_type = p.act_bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    S->n_layer  = (int) model.hparams.n_layer();

    // which layers stream their routed experts from the host, which already have them on the device
    S->native.assign((size_t) S->n_layer, 0);
    std::vector<int> native_list, streamed_list;
    for (int il = 0; il < S->n_layer && !dense; ++il) {
        const llama_layer & ly = model.layers[il];
        if (ly.attn_norm == nullptr || ly.attn_norm->buffer == nullptr || ggml_backend_buffer_is_host(ly.attn_norm->buffer)) {
            err = "layer " + std::to_string(il) + " is not offloaded to the device (its attention weights are on the host): the layer-major pass computes everything but the routed experts on the device";
            return false;
        }
        const ggml_tensor * t = ly.ffn_gate_exps != nullptr ? ly.ffn_gate_exps : ly.ffn_gate_up_exps;
        if (t == nullptr || t->buffer == nullptr) {
            err = "layer " + std::to_string(il) + " has no routed-expert bank";
            return false;
        }
        if (ggml_backend_buffer_is_host(t->buffer)) {
            streamed_list.push_back(il);
        } else {
            S->native[(size_t) il] = 1;
            native_list.push_back(il);
        }
    }
    if (streamed_list.empty() && !dense) {
        err = "every routed-expert bank is already on the device: there is nothing to stream";
        return false;
    }

    if (dense) {
        // the KV cache and the recurrent state are the model context's and stay on the device for every layer: a layer whose weights are streamed computes its attention on the device
        // (a KV buffer in host memory would put the flash attention on the CPU and copy the whole cache every chunk)
        for (const auto & kv : memory->memory_breakdown()) {
            if (kv.second > 0 && ggml_backend_buft_is_host(kv.first)) {
                err = "the KV cache / recurrent state has " + std::to_string(kv.second/1048576) + " MiB in host memory (" + ggml_backend_buft_name(kv.first) +
                      "): the layer-major pass of a dense model needs them on the device for every layer (use -ngl 99 and place the host layers with tensor overrides, not with a smaller -ngl; no --no-kv-offload)";
                return false;
            }
        }
        if (model.output == nullptr || model.output->buffer == nullptr || ggml_backend_buffer_is_host(model.output->buffer)) {
            LLAMA_LOG_WARN("%s: lm-prefill: the output projection is in host memory: the head of every pass runs on the CPU\n", __func__);
        }
    }

    S->ls.reset(new llama_layer_streaming());
    llama_layer_streaming & ls = *S->ls;
    if (!ls.init(model, p.n_slots, p.device, nullptr, 64, 256, LLAMA_PP_SCHEDULE_BATCH_MAJOR, 0, false, 10, false, false, S->act_type, 1, true, false)) {
        err = "layer streaming init failed: " + ls.error_msg;
        return false;
    }

    if (dense) {
        // a dense layer is streamed as a whole: its package (attention or recurrent block, FFN, norms) is either entirely on the device (native: no slot, no copy) or entirely in host memory
        for (int il = 0; il < S->n_layer; ++il) {
            size_t n_dev = 0, n_host = 0;
            for (const auto & m : ls.pkg_cache[(size_t) il].members) {
                if (m.src == nullptr || m.src->buffer == nullptr) {
                    continue;
                }
                (ggml_backend_buffer_is_host(m.src->buffer) ? n_host : n_dev)++;
            }
            if (n_dev > 0 && n_host > 0) {
                err = "layer " + std::to_string(il) + " has " + std::to_string(n_dev) + " weight tensors on the device and " + std::to_string(n_host) +
                      " in host memory: the layer-major pass streams a dense layer as a whole (place all of its tensors with one tensor override)";
                return false;
            }
            if (n_dev + n_host == 0) {
                err = "layer " + std::to_string(il) + " has no weight tensors";
                return false;
            }
            if (n_host == 0) {
                S->native[(size_t) il] = 1;
                native_list.push_back(il);
            } else {
                streamed_list.push_back(il);
            }
        }
        if (streamed_list.empty()) {
            err = "every layer is already on the device: there is nothing to stream";
            return false;
        }
    }
    if (!native_list.empty() && !ls.set_pp_cpu_layers(native_list)) {
        err = "layer streaming: " + ls.error_msg;
        return false;
    }
    if (p.n_slots >= 2 && !ls.set_pp_async_feed(1)) {
        err = "layer streaming: " + ls.error_msg;
        return false;
    }
    // the banks are read straight from the host mapping: page-lock the streamed ones once (spans that the hybrid engine pinned already count as pinned)
    ls.exec_layers = streamed_list;
    int64_t reg_bytes = 0;
    double  reg_us    = 0.0;
    const bool reg_ok = ls.register_all_pkg_sources(reg_bytes, reg_us);
    ls.exec_layers.clear();
    if (!reg_ok) {
        err = "page-locking the expert banks failed: " + ls.error_msg;
        return false;
    }
    // the window is held only while a pass runs
    ls.lm_release_slots();

    LLAMA_LOG_INFO("%s: lm-prefill: %s: %zu streamed layers, %zu with their %s on the device, %d slots, chunks of %d tokens, segments of %d tokens, %s slab, from batches of %d tokens (%.1f MiB page-locked here, %.0f ms)\n",
                   __func__, dense ? "dense" : "MoE", streamed_list.size(), native_list.size(), dense ? "weights" : "experts", p.n_slots, p.ubatch, p.segment, p.act_bf16 ? "bf16" : "f32", p.min_tokens, (double) reg_bytes/1048576.0, reg_us/1e3);

    lm_state = std::move(S);
    if (auto * mh = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
        mh->get_mem_attn()->lm_set_affinity(true);
    }
    return true;
}

void llama_context::lm_prefill_disable() {
    if (lm_state == nullptr) {
        return;
    }
    if (lm_state->live) {
        lm_release();
    }
    synchronize();
    lm_state.reset();
    if (auto * mh = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
        mh->get_mem_attn()->lm_set_affinity(false);
    }
}

//
// C API
//

struct llama_lm_prefill_params llama_lm_prefill_default_params(void) {
    llama_lm_prefill_params p;
    p.n_slots    = 2;
    p.ubatch     = 2048;
    p.segment    = 16384;
    p.min_tokens = 1024;
    p.act_bf16   = 0;
    p.device     = 0;
    return p;
}

bool llama_lm_prefill_enable(struct llama_context * ctx, const struct llama_lm_prefill_params * p, char * err, size_t err_len) {
    std::string e;
    bool ok = false;
    try {
        ok = ctx != nullptr && p != nullptr && ctx->lm_prefill_enable(*p, e);
    } catch (const std::exception & ex) {
        e = ex.what();
        ok = false;
    }
    if (!ok && err != nullptr && err_len > 0) {
        snprintf(err, err_len, "%s", e.c_str());
    }
    return ok;
}

void llama_lm_prefill_disable(struct llama_context * ctx) {
    if (ctx != nullptr) {
        ctx->lm_prefill_disable();
    }
}

bool llama_lm_prefill_active(const struct llama_context * ctx) {
    return ctx != nullptr && ctx->lm_prefill_is_enabled();
}

void llama_lm_prefill_snap_request(struct llama_context * ctx, const int32_t * cuts, int32_t n_cuts) {
    if (ctx != nullptr) {
        ctx->lm_snap_request(cuts, n_cuts);
    }
}

int32_t llama_lm_prefill_snap_count(const struct llama_context * ctx) {
    return ctx != nullptr ? ctx->lm_snap_count() : 0;
}

int32_t llama_lm_prefill_snap_cut(const struct llama_context * ctx, int32_t i) {
    return ctx != nullptr ? ctx->lm_snap_cut(i) : -1;
}

size_t llama_lm_prefill_snap_data(struct llama_context * ctx, int32_t i, uint8_t * dst, size_t size) {
    return ctx != nullptr ? ctx->lm_snap_data(i, dst, size) : 0;
}

bool llama_lm_prefill_arm(struct llama_context * ctx, llama_seq_id seq_id, int32_t n_tokens) {
    bool ok = false;
    try {
        ok = ctx != nullptr && ctx->lm_prearm(seq_id, n_tokens);
    } catch (const std::exception & ex) {
        LLAMA_LOG_ERROR("%s: lm-prefill: %s\n", __func__, ex.what());
        ok = false;
    }
    return ok;
}

void llama_lm_prefill_disarm(struct llama_context * ctx) {
    if (ctx != nullptr) {
        ctx->lm_disarm();
    }
}

bool llama_lm_prefill_armed(const struct llama_context * ctx) {
    return ctx != nullptr && ctx->lm_is_armed();
}

int64_t llama_lm_prefill_n_pass(const struct llama_context * ctx) {
    return ctx != nullptr ? ctx->lm_n_pass() : 0;
}

#else // LLAMA_LAYER_STREAMING

struct llama_lm_prefill_params llama_lm_prefill_default_params(void) {
    llama_lm_prefill_params p = {};
    return p;
}

bool llama_lm_prefill_enable(struct llama_context *, const struct llama_lm_prefill_params *, char * err, size_t err_len) {
    if (err != nullptr && err_len > 0) {
        snprintf(err, err_len, "%s", "this build has no layer-streaming (CUDA) support");
    }
    return false;
}

void llama_lm_prefill_disable(struct llama_context *) {}

bool llama_lm_prefill_active(const struct llama_context *) {
    return false;
}

void llama_lm_prefill_snap_request(struct llama_context *, const int32_t *, int32_t) {}
int32_t llama_lm_prefill_snap_count(const struct llama_context *) { return 0; }
int32_t llama_lm_prefill_snap_cut(const struct llama_context *, int32_t) { return -1; }
size_t llama_lm_prefill_snap_data(struct llama_context *, int32_t, uint8_t *, size_t) { return 0; }
bool llama_lm_prefill_arm(struct llama_context *, llama_seq_id, int32_t) { return false; }
void llama_lm_prefill_disarm(struct llama_context *) {}
bool llama_lm_prefill_armed(const struct llama_context *) { return false; }
int64_t llama_lm_prefill_n_pass(const struct llama_context *) { return 0; }

#endif // LLAMA_LAYER_STREAMING
