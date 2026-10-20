// Dense decode streaming (private fork, CUDA): see llama-dense-stream.h.

#include "llama-hybrid.h"

#include "llama-context.h"

#ifdef LLAMA_LAYER_STREAMING

#include "llama-dense-stream.h"
#include "llama-impl.h"
#include "llama-layer-streaming.h"
#include "llama-memory-hybrid.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cstring>

static constexpr int32_t k_ds_magic = 0x44535452;     // 'DSTR': op_params[4] of a marker node

void llama_dense_stream_deleter::operator()(llama_dense_stream * s) const {
    delete s;
}

// the CUDA backend has ONE mark function: a second ring in the same process would take it over
static llama_dense_stream * g_ds_active = nullptr;

static ggml_tensor *& ds_dst(llama_layer & l, const llama_ls_member & m) {
    return m.dst_base != nullptr ? l.*(m.dst_base) : l.nextn.*(m.dst_nextn);
}

llama_dense_stream::~llama_dense_stream() {
    if (g_ds_active == this) {
        g_ds_active = nullptr;
    }
    if (n_wait > 0 && (n_recover > 0 || getenv("LLAMA_LM_LOG") != nullptr)) {
        LLAMA_LOG_WARN("dense-stream: %lld waits, %lld dones, %lld recoveries, %lld copies queued (%.0f MiB, %.1f ms of host time), suspended %lld times for the prefill pass\n",
                       (long long) n_wait, (long long) n_done, (long long) n_recover, (long long) n_copies, (double) bytes_copied/1048576.0, t_issue_us/1e3, (long long) n_suspend);
    }
    if (mark_installed) {
        ggml_backend_cuda_set_stream_mark(nullptr, nullptr);
    }
    if (backend_copy != nullptr) {
        ggml_backend_synchronize(backend_copy);
    }
    if (backend_compute != nullptr) {
        ggml_backend_synchronize(backend_compute);
    }
    for (auto & s : slots) {
        if (s.ev_ready != nullptr) {
            ggml_backend_event_free(s.ev_ready);
        }
        if (s.ev_done != nullptr) {
            ggml_backend_event_free(s.ev_done);
        }
    }
    for (const void * p : pinned) {
        ggml_backend_cuda_unregister_host_range(p);
    }
    slots.clear();
    if (backend_copy != nullptr) {
        ggml_backend_free(backend_copy);
    }
}

bool llama_dense_stream::init(const llama_model & model, const std::vector<int> & want, int n_slots_, int device, ggml_backend_t backend_compute_, std::string & err) {
    backend_compute = backend_compute_;
    if (g_ds_active != nullptr) {
        err = "another dense decode stream exists in this process (the CUDA backend has one mark function)";
        return false;
    }
    const int K = (int) want.size();
    if (K < 1) {
        err = "no streamed layers";
        return false;
    }
    if (n_slots_ < 1 || n_slots_ > 16) {
        err = "invalid slot count: must be in 1..16";
        return false;
    }
    if (K > n_slots_ && K % n_slots_ != 0) {
        err = "the number of streamed layers (" + std::to_string(K) + ") must be a multiple of the number of slots (" + std::to_string(n_slots_) + ") or not exceed it: the ring of slots has to close at the end of a token";
        return false;
    }
    const int n_layer = (int) model.hparams.n_layer();
    k_of_layer.assign((size_t) n_layer, -1);
    layers.resize((size_t) K);

    const llama_ls_adapter * adapter = llama_ls_qwen35_adapter();

    // 1. the packages: every member must be a host tensor
    std::vector<llama_ls_package> pkgs((size_t) K);
    for (int k = 0; k < K; ++k) {
        const int il = want[(size_t) k];
        if (il < 0 || il >= n_layer || (k > 0 && il <= want[(size_t) k - 1])) {
            err = "the streamed layers must be ascending layer numbers inside the model";
            return false;
        }
        std::string perr;
        if (!adapter->package(&model, il, model.layers[(size_t) il], pkgs[(size_t) k], perr)) {
            err = "layer " + std::to_string(il) + ": " + perr;
            return false;
        }
        for (const auto & m : pkgs[(size_t) k].members) {
            if (m.src == nullptr) {
                continue;
            }
            if (m.src->buffer == nullptr || m.src->data == nullptr || !ggml_backend_buffer_is_host(m.src->buffer)) {
                err = "layer " + std::to_string(il) + " is not in host memory (tensor " + m.src->name + "): a streamed layer has all of its weights on the host";
                return false;
            }
        }
        layers[(size_t) k].il   = il;
        layers[(size_t) k].slot = k % n_slots_;
        k_of_layer[(size_t) il] = k;
    }

    // 2. the copy backend (its own stream) and the slots
    backend_copy = ggml_backend_cuda_init(device);
    if (backend_copy == nullptr) {
        err = "could not create the CUDA copy backend";
        return false;
    }
    dev = ggml_backend_get_device(backend_copy);
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device);

    slots.resize((size_t) n_slots_);
    for (int s = 0; s < n_slots_; ++s) {
        slots[(size_t) s].ev_ready = ggml_backend_event_new(dev);
        slots[(size_t) s].ev_done  = ggml_backend_event_new(dev);
        if (slots[(size_t) s].ev_ready == nullptr || slots[(size_t) s].ev_done == nullptr) {
            err = "could not create CUDA events";
            return false;
        }
        size_t n_tensors = 0;
        for (int k = s; k < K; k += n_slots_) {
            n_tensors += pkgs[(size_t) k].members.size();
        }
        ggml_init_params ip = { (n_tensors + 16) * ggml_tensor_overhead() * 2, nullptr, true };
        slots[(size_t) s].ctx_meta.reset(ggml_init(ip));
        if (!slots[(size_t) s].ctx_meta) {
            err = "could not create the slot tensor context";
            return false;
        }

        // the bundle of every layer of this slot aliases the slot's buffer from offset 0 (one layer is resident at a time)
        size_t slot_bytes = 0;
        for (int k = s; k < K; k += n_slots_) {
            auto & L = layers[(size_t) k];
            size_t off = 0;
            const size_t align = ggml_backend_buft_get_alignment(buft);
            for (const auto & m : pkgs[(size_t) k].members) {
                if (m.src == nullptr) {
                    continue;
                }
                ggml_tensor * t = ggml_new_tensor_4d(slots[(size_t) s].ctx_meta.get(), m.src->type, m.src->ne[0], m.src->ne[1], m.src->ne[2], m.src->ne[3]);
                ggml_set_name(t, m.src->name);
                ds_dst(L.bundle, m) = t;
                member_t mem;
                mem.dst    = t;
                mem.src    = m.src->data;
                mem.nbytes = ggml_nbytes(m.src);
                L.members.push_back(mem);
                off = (off + align - 1) / align * align;
                off += ggml_backend_buft_get_alloc_size(buft, t);
            }
            slot_bytes = std::max(slot_bytes, off);
        }
        slots[(size_t) s].buf.reset(ggml_backend_buft_alloc_buffer(buft, slot_bytes + 4096));
        if (!slots[(size_t) s].buf) {
            err = "could not allocate slot " + std::to_string(s) + " (" + std::to_string((slot_bytes + 4096) >> 20) + " MiB of device memory)";
            return false;
        }
        char * base = (char *) ggml_backend_buffer_get_base(slots[(size_t) s].buf.get());
        const size_t align = ggml_backend_buffer_get_alignment(slots[(size_t) s].buf.get());
        slots[(size_t) s].bytes = slot_bytes + 4096;
        bytes_held += slots[(size_t) s].bytes;
        for (int k = s; k < K; k += n_slots_) {
            size_t off = 0;
            for (auto & mem : layers[(size_t) k].members) {
                off = (off + align - 1) / align * align;
                mem.off = off;
                ggml_backend_tensor_alloc(slots[(size_t) s].buf.get(), mem.dst, base + off);
                off += ggml_backend_buft_get_alloc_size(buft, mem.dst);
            }
        }
    }

    // 3. page-lock the host weights (an async copy from pageable memory is staged by the driver, slowly, and uses the CPU). Not only the streamed layers: a registered range that ends INSIDE a tensor makes the
    // ordinary path's copy of that tensor (cudaMemcpyAsync over a range that is partly page-locked) fail with 'invalid argument' (measured), so every layer that lives in host memory is registered, which
    // leaves a boundary only where a host layer meets a layer that is on the device (whose host copy nobody reads)
    {
        struct span_t { uintptr_t lo, hi; };
        std::vector<span_t> spans;
        const uintptr_t page = 4095;
        for (int il = 0; il < n_layer; ++il) {
            llama_ls_package pk;
            std::string perr;
            if (!adapter->package(&model, il, model.layers[(size_t) il], pk, perr)) {
                continue;
            }
            uintptr_t lo = (uintptr_t) -1, hi = 0;
            bool all_host = true;
            for (const auto & m : pk.members) {
                if (m.src == nullptr) {
                    continue;
                }
                if (m.src->buffer == nullptr || m.src->data == nullptr || !ggml_backend_buffer_is_host(m.src->buffer)) {
                    all_host = false;
                    break;
                }
                lo = std::min(lo, (uintptr_t) m.src->data);
                hi = std::max(hi, (uintptr_t) m.src->data + ggml_nbytes(m.src));
            }
            if (all_host && hi > lo) {
                spans.push_back({ lo & ~page, (hi + page) & ~page });
            }
        }
        std::sort(spans.begin(), spans.end(), [](const span_t & a, const span_t & b) { return a.lo < b.lo; });
        for (size_t i = 0; i < spans.size();) {
            uintptr_t lo = spans[i].lo, hi = spans[i].hi;
            size_t j = i + 1;
            while (j < spans.size() && spans[j].lo <= hi) {
                hi = std::max(hi, spans[j].hi);
                ++j;
            }
            char e[256] = { 0 };
            if (ggml_backend_cuda_register_host_range((const void *) lo, (size_t) (hi - lo), e, sizeof(e))) {
                pinned.push_back((const void *) lo);
                pinned_bytes.push_back((size_t) (hi - lo));
            } else if (strstr(e, "already mapped") == nullptr) {
                err = std::string("page-locking the streamed layers failed: ") + e;
                return false;
            }
            i = j;
        }
    }

    // 4. the first n_slots layers, in place before the first token
    for (int k = 0; k < std::min(K, n_slots_); ++k) {
        issue_copy(k);
    }
    ggml_backend_synchronize(backend_copy);

    ggml_backend_cuda_set_stream_mark(&llama_dense_stream::mark_cb, this);
    mark_installed = true;
    g_ds_active = this;
    return true;
}

void llama_dense_stream::issue_copy(int k) {
    const int64_t t0 = ggml_time_us();
    auto & L = layers[(size_t) k];
    auto & S = slots[(size_t) L.slot];
    ggml_backend_event_wait(backend_copy, S.ev_done);           // the copy stream waits for the slot's last consumer (a never recorded event is already complete)
    for (const auto & m : L.members) {
        ggml_backend_tensor_set_async(backend_copy, m.dst, m.src, 0, m.nbytes);
        bytes_copied += (int64_t) m.nbytes;
    }
    ggml_backend_event_record(S.ev_ready, backend_copy);
    S.holds = k;
    n_copies++;
    t_issue_us += (double) (ggml_time_us() - t0);
}

void llama_dense_stream::suspend() {
    if (suspended) {
        return;
    }
    ggml_backend_synchronize(backend_copy);
    ggml_backend_synchronize(backend_compute);
    for (auto & s : slots) {
        s.buf.reset();
        s.holds = -1;
    }
    for (auto & L : layers) {
        for (auto & m : L.members) {
            m.dst->buffer = nullptr;
            m.dst->data   = nullptr;
        }
    }
    suspended = true;
    n_suspend++;
}

bool llama_dense_stream::resume() {
    if (!suspended) {
        return true;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);     // the device of the copy backend: the one init() was given
    for (size_t s = 0; s < slots.size(); ++s) {
        slots[s].buf.reset(ggml_backend_buft_alloc_buffer(buft, slots[s].bytes));
        if (!slots[s].buf) {
            for (auto & t : slots) {
                t.buf.reset();
            }
            return false;
        }
    }
    const int K = n_streamed();
    for (int k = 0; k < K; ++k) {
        auto & S = slots[(size_t) layers[(size_t) k].slot];
        char * base = (char *) ggml_backend_buffer_get_base(S.buf.get());
        for (auto & m : layers[(size_t) k].members) {
            ggml_backend_tensor_alloc(S.buf.get(), m.dst, base + m.off);
        }
    }
    suspended = false;
    for (int k = 0; k < std::min(K, n_slots()); ++k) {
        issue_copy(k);
    }
    ggml_backend_synchronize(backend_copy);
    return true;
}

void llama_dense_stream::on_wait(int k) {
    if (suspended && !resume()) {
        GGML_ABORT("dense-stream: could not take the ring of slots back from the prefill pass");
    }
    n_wait++;
    auto & S = slots[(size_t) layers[(size_t) k].slot];
    if (S.holds != k) {
        // a graph that did not run to its end (or a ring that was never closed): the layer is not in its slot. Wait for everything that may still read the slot, then copy it.
        n_recover++;
        ggml_backend_synchronize(backend_compute);
        ggml_backend_synchronize(backend_copy);
        issue_copy(k);
    }
    ggml_backend_event_wait(backend_compute, S.ev_ready);
}

void llama_dense_stream::on_done(int k) {
    n_done++;
    auto & S = slots[(size_t) layers[(size_t) k].slot];
    ggml_backend_event_record(S.ev_done, backend_compute);
    const int K = n_streamed();
    const int n = n_slots();
    if (K > n) {
        issue_copy((k + n) % K);     // the next layer of this slot; past the end of the token: the first layers of the next one
    }
}

void llama_dense_stream::mark_cb(void * user, const ggml_tensor * node, int phase) {
    if (node->op_params[4] != k_ds_magic) {
        return;
    }
    auto * self = (llama_dense_stream *) user;
    const int kind = node->op_params[5];
    const int k    = node->op_params[6];
    if (kind == 0 && phase == 0) {
        self->on_wait(k);
    } else if (kind == 1 && phase == 1) {
        self->on_done(k);
    }
}

ggml_tensor * llama_dense_stream::mark(ggml_context * ctx, ggml_backend_sched_t sched, ggml_tensor * x, int kind, int k) const {
    ggml_tensor * t = ggml_scale(ctx, x, 1.0f);      // a multiplication by exactly 1.0: the values are untouched
    t->flags |= GGML_TENSOR_FLAG_STREAM_MARK;
    t->op_params[4] = k_ds_magic;
    t->op_params[5] = kind;
    t->op_params[6] = k;
    ggml_set_name(t, kind == 0 ? "dstream_wait" : "dstream_done");
    // the marker must run on the CUDA backend whatever backend produced x
    const int n = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n; ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_is_cuda(b)) {
            ggml_backend_sched_set_tensor_backend(sched, t, b);
            break;
        }
    }
    return t;
}

//
// enable / disable
//

bool llama_context::dense_stream_enable(const llama_dense_stream_params & p, std::string & err) {
    if (dstream != nullptr) {
        err = "already enabled";
        return false;
    }
    if (model.arch != LLM_ARCH_QWEN35) {
        err = "dense decode streaming supports the dense Qwen3.5/3.8 architecture only";
        return false;
    }
    if (dynamic_cast<llama_memory_hybrid *>(memory.get()) == nullptr) {
        err = "the context has no hybrid (attention + recurrent) memory";
        return false;
    }
    ggml_backend_t backend_cuda = nullptr;
    for (auto * b : backend_ptrs) {
        if (ggml_backend_is_cuda(b)) {
            backend_cuda = b;
            break;
        }
    }
    if (backend_cuda == nullptr) {
        err = "no CUDA backend";
        return false;
    }
    if (layer_streaming != nullptr) {
        err = "not a default context";
        return false;
    }
    std::vector<int> layers(p.layers, p.layers + p.n_layers);
    auto S = std::unique_ptr<llama_dense_stream, llama_dense_stream_deleter>(new llama_dense_stream());
    if (!S->init(model, layers, p.n_slots, p.device, backend_cuda, err)) {
        return false;
    }
    LLAMA_LOG_INFO("%s: dense-stream: %d streamed layers through %d slots\n", __func__, S->n_streamed(), S->n_slots());
    synchronize();
    dstream = std::move(S);
    gf_res_prev->reset();      // a decode graph built before the markers existed must not be reused
    return true;
}

void llama_context::dense_stream_disable() {
    if (dstream == nullptr) {
        return;
    }
    synchronize();
    dstream.reset();
    gf_res_prev->reset();
}

void llama_context::dense_stream_stats(int64_t * v) const {
    if (dstream == nullptr) {
        for (int i = 0; i < 6; ++i) {
            v[i] = 0;
        }
        return;
    }
    v[0] = dstream->n_wait;
    v[1] = dstream->n_done;
    v[2] = dstream->n_recover;
    v[3] = dstream->n_copies;
    v[4] = dstream->bytes_copied;
    v[5] = (int64_t) dstream->t_issue_us;
}

//
// C API
//

bool llama_dense_stream_enable(struct llama_context * ctx, const struct llama_dense_stream_params * p, char * err, size_t err_len) {
    std::string e;
    bool ok = false;
    try {
        ok = ctx != nullptr && p != nullptr && ctx->dense_stream_enable(*p, e);
    } catch (const std::exception & ex) {
        e = ex.what();
        ok = false;
    }
    if (!ok && err != nullptr && err_len > 0) {
        snprintf(err, err_len, "%s", e.c_str());
    }
    return ok;
}

void llama_dense_stream_disable(struct llama_context * ctx) {
    if (ctx != nullptr) {
        ctx->dense_stream_disable();
    }
}

bool llama_dense_stream_active(const struct llama_context * ctx) {
    return ctx != nullptr && ctx->dense_stream_is_enabled();
}

void llama_dense_stream_stats(const struct llama_context * ctx, int64_t * v6) {
    if (ctx != nullptr && v6 != nullptr) {
        ctx->dense_stream_stats(v6);
    }
}

#else // LLAMA_LAYER_STREAMING

bool llama_dense_stream_enable(struct llama_context *, const struct llama_dense_stream_params *, char * err, size_t err_len) {
    if (err != nullptr && err_len > 0) {
        snprintf(err, err_len, "%s", "this build has no layer-streaming (CUDA) support");
    }
    return false;
}

void llama_dense_stream_disable(struct llama_context *) {}

bool llama_dense_stream_active(const struct llama_context *) {
    return false;
}

void llama_dense_stream_stats(const struct llama_context *, int64_t *) {}

#endif // LLAMA_LAYER_STREAMING
