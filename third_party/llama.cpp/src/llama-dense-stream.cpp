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

void llama_fattn_head_tile(bool on) {
    ggml_backend_cuda_set_fattn_head_tile(on);
}

// the CUDA backend has ONE mark function: a second ring in the same process would take it over
static llama_dense_stream * g_ds_active = nullptr;

// busy wait that polls the driver less often than a bare loop (every query takes the driver lock the compute thread needs too)
static void spin_us(int us) {
    const int64_t t_end = ggml_time_us() + us;
    while (ggml_time_us() < t_end) {
        std::this_thread::yield();
    }
}

static double env_double(const char * name, double def) {
    const char * e = getenv(name);
    return e != nullptr && e[0] != '\0' ? atof(e) : def;
}

static ggml_tensor *& ds_dst(llama_layer & l, const llama_ls_member & m) {
    return m.dst_base != nullptr ? l.*(m.dst_base) : l.nextn.*(m.dst_nextn);
}

llama_dense_stream::~llama_dense_stream() {
    stop_worker();
    if (hop_kernel && getenv("LLAMA_CUDA_HOP_KERNEL") == nullptr) {
        ggml_backend_cuda_set_hop_kernel(false);
    }
    if (g_ds_active == this) {
        g_ds_active = nullptr;
    }
    if (n_wait > 0 && (n_recover > 0 || getenv("LLAMA_LM_LOG") != nullptr)) {
        LLAMA_LOG_WARN("dense-stream: %lld waits, %lld dones, %lld recoveries, %lld copies queued (%.0f MiB, %.1f ms of host time), suspended %lld times for the prefill pass, %lld passes, %lld plans adopted after the first\n",
                       (long long) n_wait, (long long) n_done, (long long) n_recover, (long long) n_copies, (double) bytes_copied/1048576.0, t_issue_us/1e3, (long long) n_suspend, (long long) n_passes, (long long) n_replans);
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
    for (auto e : piece_ev) {
        ggml_backend_event_free(e);
    }
    for (void * e : ev_t0) {
        ggml_backend_cuda_timing_event_free(e);
    }
    for (void * e : ev_t1) {
        ggml_backend_cuda_timing_event_free(e);
    }
    for (const void * p : pinned) {
        ggml_backend_cuda_unregister_host_range(p);
    }
    slots.clear();
    if (backend_copy != nullptr) {
        ggml_backend_free(backend_copy);
    }
}

bool llama_dense_stream::init(const llama_model & model, const std::vector<int> & host, const std::vector<int> & want, const std::vector<int> & dedicated, int n_slots_, int device, ggml_backend_t backend_compute_, bool dynamic_, int percentile, std::string & err) {
    backend_compute = backend_compute_;
    dynamic = dynamic_;
    if (g_ds_active != nullptr) {
        err = "another dense decode stream exists in this process (the CUDA backend has one mark function)";
        return false;
    }
    if (n_slots_ < 1 || n_slots_ > 16) {
        err = "invalid slot count: must be in 1..16";
        return false;
    }
    const int n_layer = (int) model.hparams.n_layer();
    n_layer_model = n_layer;
    const std::vector<int> & cl = dynamic ? host : want;      // the layers that can be ring layers
    if (cl.empty()) {
        err = dynamic ? "no host layers" : "no streamed layers";
        return false;
    }
    if (!dynamic) {
        const int K = (int) want.size();
        if (K > n_slots_ && K % n_slots_ != 0) {
            err = "the number of streamed layers (" + std::to_string(K) + ") must be a multiple of the number of slots (" + std::to_string(n_slots_) + ") or not exceed it: the ring of slots has to close at the end of a token";
            return false;
        }
    } else if ((int) cl.size() < n_slots_) {
        err = "a dynamic ring needs at least as many host layers as slots";
        return false;
    }
    cand_of_layer.assign((size_t) n_layer, -1);
    k_of_layer.assign((size_t) n_layer, -1);
    cand.resize(cl.size());

    const llama_ls_adapter * adapter = llama_ls_qwen35_adapter();

    // 1. the packages: every member must be a host tensor
    std::vector<llama_ls_package> pkgs(cl.size());
    for (size_t c = 0; c < cl.size(); ++c) {
        const int il = cl[c];
        if (il < 0 || il >= n_layer || (c > 0 && il <= cl[c - 1])) {
            err = "the streamed layers must be ascending layer numbers inside the model";
            return false;
        }
        std::string perr;
        if (!adapter->package(&model, il, model.layers[(size_t) il], pkgs[c], perr)) {
            err = "layer " + std::to_string(il) + ": " + perr;
            return false;
        }
        for (const auto & m : pkgs[c].members) {
            if (m.src == nullptr) {
                continue;
            }
            if (m.src->buffer == nullptr || m.src->data == nullptr || !ggml_backend_buffer_is_host(m.src->buffer)) {
                err = "layer " + std::to_string(il) + " is not in host memory (tensor " + m.src->name + "): a streamed layer has all of its weights on the host";
                return false;
            }
        }
        cand[c].il = il;
        cand_of_layer[(size_t) il] = (int) c;
    }

    // 2. the copy backend (its own stream) and the slots
    backend_copy = ggml_backend_cuda_init(device);
    if (backend_copy == nullptr) {
        err = "could not create the CUDA copy backend";
        return false;
    }
    dev = ggml_backend_get_device(backend_copy);
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device);
    const size_t align_t = ggml_backend_buft_get_alignment(buft);

    // the members of every candidate: the same offsets in every slot, one set of slot-bound tensors per slot
    size_t slot_bytes = 0;
    size_t n_tensors = 0;
    for (size_t c = 0; c < cl.size(); ++c) {
        auto & C = cand[c];
        size_t pkg_bytes = 0;
        for (const auto & m : pkgs[c].members) {
            if (m.src == nullptr) {
                continue;
            }
            member_t mem;
            mem.src    = m.src->data;
            mem.nbytes = ggml_nbytes(m.src);
            pkg_bytes += mem.nbytes;
            mem.dst.assign((size_t) n_slots_, nullptr);
            C.members.push_back(mem);
            n_tensors++;
        }
        C.bytes = pkg_bytes;
        C.bundle.resize((size_t) n_slots_);
    }

    slots.resize((size_t) n_slots_);
    for (int s = 0; s < n_slots_; ++s) {
        slots[(size_t) s].ev_ready = ggml_backend_event_new(dev);
        slots[(size_t) s].ev_done  = ggml_backend_event_new(dev);
        if (slots[(size_t) s].ev_ready == nullptr || slots[(size_t) s].ev_done == nullptr) {
            err = "could not create CUDA events";
            return false;
        }
        ggml_init_params ip = { (n_tensors + 16) * ggml_tensor_overhead() * 2, nullptr, true };
        slots[(size_t) s].ctx_meta.reset(ggml_init(ip));
        if (!slots[(size_t) s].ctx_meta) {
            err = "could not create the slot tensor context";
            return false;
        }
    }
    for (size_t c = 0; c < cl.size(); ++c) {
        auto & C = cand[c];
        size_t off = 0;     // after the last slot: the size of the package in the slot buffer (the offsets are the same in every slot)
        for (int s = 0; s < n_slots_; ++s) {
            size_t mi = 0;
            off = 0;
            for (const auto & m : pkgs[c].members) {
                if (m.src == nullptr) {
                    continue;
                }
                ggml_tensor * t = ggml_new_tensor_4d(slots[(size_t) s].ctx_meta.get(), m.src->type, m.src->ne[0], m.src->ne[1], m.src->ne[2], m.src->ne[3]);
                ggml_set_name(t, m.src->name);
                ds_dst(C.bundle[(size_t) s], m) = t;
                C.members[mi].dst[(size_t) s] = t;
                off = (off + align_t - 1) / align_t * align_t;
                C.members[mi].off = off;
                off += ggml_backend_buft_get_alloc_size(buft, t);
                ++mi;
            }
        }
        slot_bytes = std::max(slot_bytes, off);
    }
    for (int s = 0; s < n_slots_; ++s) {
        slots[(size_t) s].buf.reset(ggml_backend_buft_alloc_buffer(buft, slot_bytes + 4096));
        if (!slots[(size_t) s].buf) {
            err = "could not allocate slot " + std::to_string(s) + " (" + std::to_string((slot_bytes + 4096) >> 20) + " MiB of device memory)";
            return false;
        }
        slots[(size_t) s].bytes = slot_bytes + 4096;
        bytes_held += slots[(size_t) s].bytes;
    }
    bind_all();

    // the dedicated layers (G-EVICT): a buffer of their own each, the same slot-bound tensors as a ring layer's, filled once (and again after every layer-major pass)
    ded_of_layer.assign((size_t) n_layer, -1);
    ded.resize(dedicated.size());
    for (size_t d = 0; d < dedicated.size(); ++d) {
        const int il = dedicated[d];
        if (il < 0 || il >= n_layer || cand_of_layer[(size_t) il] >= 0 || ded_of_layer[(size_t) il] >= 0) {
            err = "dedicated layer " + std::to_string(il) + " is outside the model, a ring candidate or listed twice";
            return false;
        }
        llama_ls_package pk;
        std::string perr;
        if (!adapter->package(&model, il, model.layers[(size_t) il], pk, perr)) {
            err = "dedicated layer " + std::to_string(il) + ": " + perr;
            return false;
        }
        size_t n_members = 0;
        for (const auto & m : pk.members) {
            if (m.src == nullptr) {
                continue;
            }
            if (m.src->buffer == nullptr || m.src->data == nullptr || !ggml_backend_buffer_is_host(m.src->buffer)) {
                err = "dedicated layer " + std::to_string(il) + " is not in host memory (tensor " + m.src->name + "): it has to be placed on the host so that the prefill pass can stream it";
                return false;
            }
            n_members++;
        }
        auto & D = ded[d];
        D.il = il;
        ded_of_layer[(size_t) il] = (int) d;
        ggml_init_params ip = { (n_members + 16) * ggml_tensor_overhead() * 2, nullptr, true };
        D.ctx_meta.reset(ggml_init(ip));
        if (!D.ctx_meta) {
            err = "could not create the tensor context of a dedicated layer";
            return false;
        }
        size_t off = 0;
        for (const auto & m : pk.members) {
            if (m.src == nullptr) {
                continue;
            }
            ggml_tensor * t = ggml_new_tensor_4d(D.ctx_meta.get(), m.src->type, m.src->ne[0], m.src->ne[1], m.src->ne[2], m.src->ne[3]);
            ggml_set_name(t, m.src->name);
            ds_dst(D.bundle, m) = t;
            member_t mem;
            mem.src    = m.src->data;
            mem.nbytes = ggml_nbytes(m.src);
            off = (off + align_t - 1) / align_t * align_t;
            mem.off = off;
            mem.dst.assign(1, t);
            off += ggml_backend_buft_get_alloc_size(buft, t);
            D.members.push_back(mem);
        }
        D.bytes = off + 4096;
        D.buf.reset(ggml_backend_buft_alloc_buffer(buft, D.bytes));
        if (!D.buf) {
            err = "could not allocate the slot of dedicated layer " + std::to_string(il) + " (" + std::to_string(D.bytes >> 20) + " MiB of device memory)";
            return false;
        }
        bytes_held += D.bytes;
    }
    bind_ded();

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

    fill_ded();

    // 4. the dynamic plan: the knobs, the timing events, the first plan from priors
    pass_t_wait.assign(cand.size(), 0);
    pass_t_done.assign(cand.size(), 0);
    if (dynamic) {
        ctl.pct           = std::min(99.0, std::max(50.0, (double) percentile));
        ctl.margin_base   = std::max(0.0, env_double("LLAMA_DS_MARGIN", 0.25));
        ctl.overhead_ms   = env_double("LLAMA_DS_OVERHEAD_MS", 2.0);
        ctl.replan_every  = std::max(1, (int) env_double("LLAMA_DS_REPLAN", 2));
        ctl.window_passes = (size_t) std::max(4.0, env_double("LLAMA_DS_WINDOW_TOKENS", 16));
        ctl.c_prior_ms    = env_double("LLAMA_DS_PRIOR_C_MS", 7.5);
        ctl.d_prior_ms    = env_double("LLAMA_DS_PRIOR_COPY_MS", 33.0);
        ctl.gap_ms        = env_double("LLAMA_DS_GAP_MS", 3.0);
        ctl.n_slots       = n_slots_;
        ctl.n_layer       = n_layer;
        ctl.mib.assign((size_t) n_layer, 0.0);
        for (const auto & C : cand) {
            ctl.host.push_back(C.il);
            ctl.mib[(size_t) C.il] = (double) C.bytes / 1048576.0;
        }
        ctl.configure();
        for (size_t c = 0; c < cand.size(); ++c) {
            ev_t0.push_back(ggml_backend_cuda_timing_event_new(backend_compute, 0));
            ev_t1.push_back(ggml_backend_cuda_timing_event_new(backend_compute, 0));
        }
        const llama_dense_plan::outcome first = llama_dense_plan::plan(ctl.make_inputs(ctl.margin));
        if (first.ring.empty()) {
            err = "the dynamic plan found no ring layer that pays (CPU layer " + std::to_string(ctl.c_prior_ms) + " ms against the device layer plus the copy interference " + std::to_string(ctl.overhead_ms) + " ms)";
            return false;
        }
        set_plan(first.ring);
    } else {
        set_plan(want);
    }
    ggml_backend_synchronize(backend_copy);

    // Paced by default on Windows (WDDM): an unpaced layer copy blocks the activation hops of the CPU layers for its whole duration (3.64 against 2.49 tok/s, docs/plans/dense-decode-ring-windows.md)
#ifdef _WIN32
    const int def_kib = 1024, def_poll = 20;
#else
    const int def_kib = 0, def_poll = 0;
#endif
    const int kib = getenv("LLAMA_DS_PIECE_KIB") != nullptr ? atoi(getenv("LLAMA_DS_PIECE_KIB")) : def_kib;
    if (kib > 0) {
        piece_bytes = (size_t) kib * 1024;
        window = getenv("LLAMA_DS_WINDOW") != nullptr ? std::max(1, atoi(getenv("LLAMA_DS_WINDOW"))) : 2;
        poll_us = getenv("LLAMA_DS_POLL_US") != nullptr ? std::max(0, atoi(getenv("LLAMA_DS_POLL_US"))) : def_poll;
        for (int i = 0; i < window; ++i) {
            piece_ev.push_back(ggml_backend_event_new(dev));
        }
        worker = std::thread(&llama_dense_stream::worker_main, this);
        LLAMA_LOG_INFO("dense-stream: paced copy, pieces of %zu KiB, %d in flight\n", piece_bytes / 1024, window);
        // the activation hops of the CPU layers as kernels on pinned memory: they no longer wait behind a piece on the copy engine (+3% on top of the pacing)
        if (getenv("LLAMA_DS_HOP_KERNEL") == nullptr || atoi(getenv("LLAMA_DS_HOP_KERNEL")) != 0) {
            ggml_backend_cuda_set_hop_kernel(true);
            hop_kernel = true;
        }
    }

    ggml_backend_cuda_set_stream_mark(&llama_dense_stream::mark_cb, this);
    mark_installed = true;
    g_ds_active = this;
    return true;
}

void llama_dense_stream::bind_all() {
    for (auto & C : cand) {
        for (auto & m : C.members) {
            for (size_t s = 0; s < slots.size(); ++s) {
                char * base = (char *) ggml_backend_buffer_get_base(slots[s].buf.get());
                ggml_backend_tensor_alloc(slots[s].buf.get(), m.dst[s], base + m.off);
            }
        }
    }
}

void llama_dense_stream::bind_ded() {
    for (auto & D : ded) {
        char * base = (char *) ggml_backend_buffer_get_base(D.buf.get());
        for (auto & m : D.members) {
            ggml_backend_tensor_alloc(D.buf.get(), m.dst[0], base + m.off);
        }
    }
}

void llama_dense_stream::fill_ded() {
    if (ded.empty()) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    size_t bytes = 0;
    for (auto & D : ded) {
        for (const auto & m : D.members) {
            ggml_backend_tensor_set_async(backend_copy, m.dst[0], m.src, 0, m.nbytes);
            bytes += m.nbytes;
        }
    }
    ggml_backend_synchronize(backend_copy);
    if (getenv("LLAMA_LM_LOG") != nullptr) {
        LLAMA_LOG_WARN("dense-stream: %zu dedicated layers filled (%.0f MiB in %.0f ms)\n", ded.size(), (double) bytes/1048576.0, (double) (ggml_time_us() - t0)/1000.0);
    }
}

// the active plan; the first n_slots ring layers are the contents of the slots before the pass starts
void llama_dense_stream::set_plan(const std::vector<int> & r) {
    ring = r;
    k_of_layer.assign((size_t) n_layer_model, -1);
    for (size_t k = 0; k < ring.size(); ++k) {
        k_of_layer[(size_t) ring[k]] = (int) k;
    }
    epoch++;
    pass_ring = ring;
    pass_k = (int) ring.size();
    if (!suspended) {
        for (int s = 0; s < std::min((int) ring.size(), n_slots()); ++s) {
            if (slots[(size_t) s].holds != ring[(size_t) s]) {
                post_copy(ring[(size_t) s], s);
            }
        }
    }
}

void llama_dense_stream::issue_copy(int il, int slot) {
    const int64_t t0 = ggml_time_us();
    auto & L = cand[(size_t) cand_of_layer[(size_t) il]];
    auto & S = slots[(size_t) slot];
    ggml_backend_event_wait(backend_copy, S.ev_done);           // the copy stream waits for the slot's last consumer (a never recorded event is already complete)
    static const bool nocopy = getenv("LLAMA_DS_NOCOPY") != nullptr;     // diagnostic: the slots are never filled (timing only, the output is garbage)
    for (const auto & m : L.members) {
        if (!nocopy) {
            ggml_backend_tensor_set_async(backend_copy, m.dst[(size_t) slot], m.src, 0, m.nbytes);
        }
        bytes_copied += (int64_t) m.nbytes;
    }
    ggml_backend_event_record(S.ev_ready, backend_copy);
    S.holds = il;
    {
        std::lock_guard<std::mutex> lk(mu);
        S.rec = il;
    }
    n_copies++;
    t_issue_us += (double) (ggml_time_us() - t0);
}

void llama_dense_stream::post_copy(int il, int slot) {
    if (piece_bytes == 0) {
        issue_copy(il, slot);
        return;
    }
    auto & S = slots[(size_t) slot];
    S.holds = il;
    {
        std::lock_guard<std::mutex> lk(mu);
        S.rec = -1;
        reqs.push_back({ il, slot });
    }
    cv.notify_all();
}

void llama_dense_stream::copy_paced(int il, int slot) {
    auto & L = cand[(size_t) cand_of_layer[(size_t) il]];
    auto & S = slots[(size_t) slot];
    ggml_backend_event_wait(backend_copy, S.ev_done);
    if (dynamic) {
        // the copy starts when the slot's last consumer is done: wait for that on the host so that the time below is the DMA alone
        while (!ggml_backend_cuda_event_ready(S.ev_done)) {
            spin_us(poll_us > 0 ? poll_us : 20);
        }
    }
    const int64_t t0 = ggml_time_us();
    size_t j = 0;
    for (const auto & m : L.members) {
        for (size_t off = 0; off < m.nbytes; off += piece_bytes) {
            ggml_backend_event_t e = piece_ev[j % (size_t) window];
            if (j >= (size_t) window) {
                while (!ggml_backend_cuda_event_ready(e)) {
                    spin_us(poll_us);
                }
            }
            ggml_backend_tensor_set_async(backend_copy, m.dst[(size_t) slot], (const char *) m.src + off, off, std::min(piece_bytes, m.nbytes - off));
            ggml_backend_event_record(e, backend_copy);
            (void) ggml_backend_cuda_event_ready(e);     // hand the piece to the GPU now (WDDM batches submissions)
            j++;
        }
        bytes_copied += (int64_t) m.nbytes;
    }
    ggml_backend_event_record(S.ev_ready, backend_copy);
    (void) ggml_backend_cuda_event_ready(S.ev_ready);
    n_copies++;
    t_copy_start_us = t0;
}

void llama_dense_stream::worker_main() {
    std::unique_lock<std::mutex> lk(mu);
    for (;;) {
        cv.wait(lk, [&] { return stopping || !reqs.empty(); });
        if (reqs.empty()) {
            return;
        }
        const req_t r = reqs.front();
        reqs.pop_front();
        busy = true;
        lk.unlock();
        copy_paced(r.il, r.slot);
        lk.lock();
        slots[(size_t) r.slot].rec = r.il;      // the compute thread may wait for the layer from here on
        lk.unlock();
        if (dynamic) {
            // the duration of the copy: from the start of the DMA to its end (the pieces are small: the wait is the last window)
            auto & S = slots[(size_t) r.slot];
            while (!ggml_backend_cuda_event_ready(S.ev_ready)) {
                spin_us(poll_us > 0 ? poll_us : 20);
            }
            const double ms = (double) (ggml_time_us() - t_copy_start_us) / 1000.0;
            std::lock_guard<std::mutex> lg(mu_stat);
            copy_samples.push_back(ms);
        }
        lk.lock();
        busy = false;
        cv.notify_all();
    }
}

void llama_dense_stream::drain_worker() {
    if (!worker.joinable()) {
        return;
    }
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return reqs.empty() && !busy; });
}

void llama_dense_stream::stop_worker() {
    if (!worker.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mu);
        stopping = true;
    }
    cv.notify_all();
    worker.join();
}

void llama_dense_stream::suspend() {
    if (suspended) {
        return;
    }
    drain_worker();
    ggml_backend_synchronize(backend_copy);
    ggml_backend_synchronize(backend_compute);
    for (auto & s : slots) {
        s.buf.reset();
        s.holds = -1;
    }
    for (auto & C : cand) {
        for (auto & m : C.members) {
            for (auto * t : m.dst) {
                t->buffer = nullptr;
                t->data   = nullptr;
            }
        }
    }
    for (auto & D : ded) {
        D.buf.reset();
        for (auto & m : D.members) {
            m.dst[0]->buffer = nullptr;
            m.dst[0]->data   = nullptr;
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
    for (auto & D : ded) {
        D.buf.reset(ggml_backend_buft_alloc_buffer(buft, D.bytes));
        if (!D.buf) {
            for (auto & t : slots) {
                t.buf.reset();
            }
            for (auto & e : ded) {
                e.buf.reset();
            }
            return false;
        }
    }
    bind_all();
    bind_ded();
    suspended = false;
    for (int s = 0; s < std::min((int) ring.size(), n_slots()); ++s) {
        issue_copy(ring[(size_t) s], s);
    }
    fill_ded();
    ggml_backend_synchronize(backend_copy);
    return true;
}

void llama_dense_stream::on_wait(int k) {
    if (suspended && !resume()) {
        GGML_ABORT("dense-stream: could not take the ring of slots back from the prefill pass");
    }
    GGML_ASSERT(k >= 0 && (size_t) k < ring.size());
    n_wait++;
    pass_waits++;
    const int il = ring[(size_t) k];
    const int s  = k % n_slots();
    auto & S = slots[(size_t) s];
    if (S.holds != il) {
        // a graph that did not run to its end (or a ring that was never closed): the layer is not in its slot. Wait for everything that may still read the slot, then copy it.
        n_recover++;
        drain_worker();
        ggml_backend_synchronize(backend_compute);
        ggml_backend_synchronize(backend_copy);
        issue_copy(il, s);
    }
    if (worker.joinable()) {
        // the worker records the ready event after the last piece: waiting on the event before that would wait for the layer that was in the slot before
        const int64_t t0 = ggml_time_us();
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return S.rec == il; });
        t_stall_us += (double) (ggml_time_us() - t0);
    }
    if (!ggml_backend_cuda_event_ready(S.ev_ready)) {
        n_notready++;
        pass_late++;
    }
    if (n_wait % 160 == 0 && getenv("LLAMA_LM_LOG") != nullptr) {
        LLAMA_LOG_WARN("dense-stream: %lld waits, %lld not ready, %lld recoveries, %lld copies, host stalled %.1f ms for the worker; ring %zu layers, plan %d, %lld passes\n", (long long) n_wait, (long long) n_notready, (long long) n_recover, (long long) n_copies, t_stall_us / 1e3, ring.size(), epoch, (long long) n_passes);
    }
    ggml_backend_event_wait(backend_compute, S.ev_ready);
    if (dynamic) {
        ggml_backend_cuda_timing_event_record(ev_t0[(size_t) k]);      // after the wait: the device time of the layer alone
        pass_t_wait[(size_t) k] = ggml_time_us();
    }
}

void llama_dense_stream::on_done(int k) {
    GGML_ASSERT(k >= 0 && (size_t) k < ring.size());
    n_done++;
    pass_dones++;
    auto & S = slots[(size_t) (k % n_slots())];
    ggml_backend_event_record(S.ev_done, backend_compute);
    if (dynamic) {
        ggml_backend_cuda_timing_event_record(ev_t1[(size_t) k]);
        pass_t_done[(size_t) k] = ggml_time_us();
    }
    const int K = n_streamed();
    const int n = n_slots();
    if (K > n) {
        post_copy(ring[(size_t) ((k + n) % K)], k % n);      // the next layer of this slot; past the end of the token: the first layers of the next one
        // WDDM batches submissions: without a query the queued copy starts only at the next synchronizing call, i.e. when the GPU already waits for it
        static const bool flush = getenv("LLAMA_DS_FLUSH") == nullptr || atoi(getenv("LLAMA_DS_FLUSH")) != 0;
        if (flush) {
            (void) ggml_backend_cuda_event_ready(S.ev_done);
            if (piece_bytes == 0) {
                (void) ggml_backend_cuda_event_ready(S.ev_ready);
            }
        }
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
// the dynamic plan
//

// the statistics of the pass that just ended
void llama_dense_stream::close_pass() {
    {
        std::lock_guard<std::mutex> lg(mu_stat);
        for (double ms : copy_samples) {
            ctl.observe_copy(ms);
        }
        copy_samples.clear();
    }
    const int K = pass_k;
    if (!dynamic || K <= 0 || pass_waits != K || pass_dones != K || (int) pass_ring.size() != K) {
        return;       // a pass that did not run to its end says nothing about the layers
    }
    n_passes++;
    // the device time of every ring layer (the events of the pass are complete: its output was read back)
    double g_sum = 0.0;
    int    g_n = 0;
    for (int k = 0; k < K; ++k) {
        const double ms = ggml_backend_cuda_timing_event_elapsed_ms(ev_t0[(size_t) k], ev_t1[(size_t) k]);
        if (ms > 0.0 && ms < 50.0) {
            ctl.observe_gpu(ms);
            g_sum += ms;
            g_n++;
        }
    }
    const double g = g_n > 0 ? g_sum / g_n : ctl.g_prior_ms;
    // the CPU time between two ring layers: from the end of one to the start of the next, minus the device layers inside; per MiB of the CPU layers inside
    std::vector<char> in_ring((size_t) n_layer_model, 0);
    for (int il : pass_ring) {
        in_ring[(size_t) il] = 1;
    }
    double x_sum = 0.0, mib_sum = 0.0;
    for (int k = 0; k + 1 < K; ++k) {
        double mib = 0.0;
        int n_dev = 0;
        for (int il = pass_ring[(size_t) k] + 1; il < pass_ring[(size_t) (k + 1)]; ++il) {
            if (cand_of_layer[(size_t) il] >= 0 && !in_ring[(size_t) il]) {
                mib += ctl.mib[(size_t) il];
            } else {
                n_dev++;
            }
        }
        if (mib <= 0.0) {
            continue;
        }
        const double x = (double) (pass_t_wait[(size_t) (k + 1)] - pass_t_done[(size_t) k]) / 1000.0;
        x_sum += std::max(0.0, x - g * n_dev);
        mib_sum += mib;
    }
    ctl.observe_pass(mib_sum > 0.0 ? x_sum / mib_sum : 0.0, (double) pass_late / (double) K);
}

bool llama_dense_stream::replan() {
    std::vector<int> next;
    llama_dense_plan::outcome nw, cur;
    const double late = ctl.late_mean();
    if (!ctl.decide(ring, next, &nw, &cur)) {
        // what the controller thinks every 20 decisions, also when it keeps the plan (LLAMA_LM_LOG=1)
        static int calls = 0;
        if (nw.t_ms > 0.0 && ++calls % 20 == 1 && getenv("LLAMA_LM_LOG") != nullptr) {
            const llama_dense_plan::inputs in = ctl.make_inputs(ctl.margin);
            LLAMA_LOG_WARN("dense-stream: keeps plan %d (%zu ring layers, predicted %.0f ms, stall %.1f): CPU %.2f ms per %.0f MiB layer (p%.0f), device %.2f ms, copy %.1f ms (p%.0f), margin %.2f, late %.1f%%; the best plan now has %zu ring layers, predicted %.0f ms\n",
                           epoch, ring.size(), cur.t_ms, cur.stall_ms, in.cpu_ms[(size_t) ctl.host[0]], ctl.mean_mib(), 100.0 - ctl.pct, in.gpu_ms, in.copy_ms, ctl.pct, ctl.margin, late * 100.0, nw.ring.size(), nw.t_ms);
        }
        return false;
    }
    const size_t k_old = ring.size();
    set_plan(next);
    n_replans++;
    if (getenv("LLAMA_LM_LOG") != nullptr) {
        std::string list;
        for (int il : ring) {
            list += (list.empty() ? "" : ",") + std::to_string(il);
        }
        const llama_dense_plan::inputs in = ctl.make_inputs(ctl.margin);
        LLAMA_LOG_WARN("dense-stream: plan %d: %zu ring layers [%s] (was %zu): CPU %.2f ms per %.0f MiB layer (p%.0f), device layer %.2f ms, copy %.1f ms (p%.0f), margin %.2f, late %.1f%%; predicted pass %.0f ms (was %.0f, stall %.1f)\n",
                       epoch, ring.size(), list.c_str(), k_old, in.cpu_ms[(size_t) ctl.host[0]], ctl.mean_mib(), 100.0 - ctl.pct, in.gpu_ms, in.copy_ms, ctl.pct, ctl.margin, late * 100.0, nw.t_ms, cur.t_ms, cur.stall_ms);
    }
    return true;
}

bool llama_dense_stream::begin_pass() {
    bool resumed = false;
    if (suspended) {
        if (!resume()) {
            GGML_ABORT("dense-stream: could not take the slots back from the prefill pass");
        }
        resumed = true;
    }
    close_pass();
    pass_waits = pass_dones = pass_late = 0;
    bool changed = false;
    if (dynamic) {
        changed = replan();
    }
    pass_ring = ring;
    pass_k = (int) ring.size();
    return changed || resumed;
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
    std::vector<int> host;
    if (p.host_layers != nullptr && p.n_host > 0) {
        host.assign(p.host_layers, p.host_layers + p.n_host);
    } else {
        host = layers;
    }
    std::vector<int> dedicated;
    if (p.dedicated != nullptr && p.n_dedicated > 0) {
        dedicated.assign(p.dedicated, p.dedicated + p.n_dedicated);
    }
    auto S = std::unique_ptr<llama_dense_stream, llama_dense_stream_deleter>(new llama_dense_stream());
    if (!S->init(model, host, layers, dedicated, p.n_slots, p.device, backend_cuda, p.dynamic != 0, p.percentile > 0 ? p.percentile : 95, err)) {
        return false;
    }
    LLAMA_LOG_INFO("%s: dense-stream: %d %s layers through %d slots, %d dedicated layers\n", __func__, S->n_streamed(), S->is_dynamic() ? "dynamic ring" : "streamed", S->n_slots(), S->n_dedicated());
    synchronize();
    dstream = std::move(S);
    reset_graph_reuse();      // a decode graph built before the markers existed must not be reused
    return true;
}

void llama_context::dense_stream_disable() {
    if (dstream == nullptr) {
        return;
    }
    synchronize();
    dstream.reset();
    reset_graph_reuse();
}

bool llama_context::dense_stream_begin_pass() {
    return dstream != nullptr && dstream->begin_pass();
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
