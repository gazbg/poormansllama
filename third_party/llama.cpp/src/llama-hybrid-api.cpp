#include "llama-hybrid.h"

#include "llama-context.h"
#include "llama-impl.h"

#ifdef LLAMA_LAYER_STREAMING
#include "llama-moe-hybrid.h"
extern void (*llama_kv_lazy_pressure_cb)(llama_context * lctx, int64_t commit_bytes);   // llama-kv-cache.cpp
extern void (*llama_hybrid_decode_hook)(llama_context * ctx, int32_t n_tokens);          // llama-context.cpp
extern bool (*llama_lm_vram_hook)(llama_context * ctx, int64_t need_bytes, int begin);   // llama-lm-prefill.cpp
extern bool llama_lm_prefill_pending;                                                    // llama-lm-prefill.cpp
#endif

#include <atomic>
#include <chrono>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#endif

bool llama_hybrid_available(void) {
#ifdef LLAMA_LAYER_STREAMING
    return true;
#else
    return false;
#endif
}

#ifdef LLAMA_LAYER_STREAMING
static void hybrid_setenv_default(const char * key, const char * value) {
    if (getenv(key) != nullptr) {
        return;
    }
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 0);
#endif
}
#endif

void llama_hybrid_prepare_env(int32_t initial_slots, const char * freq_path) {
#ifdef LLAMA_LAYER_STREAMING
    char slots[16];
    snprintf(slots, sizeof(slots), "%d", initial_slots > 0 ? initial_slots : 14);
    hybrid_setenv_default("LLAMA_MOE_HYBRID", slots);
    hybrid_setenv_default("LLAMA_MOE_HYBRID_B", "16");          // promotions budget per step
    hybrid_setenv_default("LLAMA_MOE_HYBRID_SCAN", "2");
    hybrid_setenv_default("LLAMA_MOE_HYBRID_LEAN", "1");        // lean host leg
    hybrid_setenv_default("LLAMA_MOE_HYBRID_LEAN_PF", "2");     // lean prefetch K = 2
    hybrid_setenv_default("LLAMA_CUDA_INPUT_BATCH", "1");       // batched small input uploads
    hybrid_setenv_default("LLAMA_FATTN_GQA8", "1");             // packed GQA decode attention (1-4 rows)
    hybrid_setenv_default("LLAMA_KV_LAZY", "1");                // the KV cache commits its rows on demand (the expert cache gets the rest)
    hybrid_setenv_default("LLAMA_KV_LAZY_ONDEMAND", "1");       // ... and the cache itself does the committing (G31r; the runner commits its rows up front)
    if (freq_path != nullptr && *freq_path != '\0') {
        hybrid_setenv_default("LLAMA_MOE_HYBRID_FREQ", freq_path);
    }
#else
    GGML_UNUSED(initial_slots);
    GGML_UNUSED(freq_path);
#endif
}

#ifdef LLAMA_LAYER_STREAMING
static int64_t g_hybrid_keep_free = 0;
static int64_t g_hybrid_safety    = 0;
static int64_t g_hybrid_slack     = 0;
static int     g_hybrid_min_slots = 8;
static llama_context * g_hybrid_ctx = nullptr;

#ifdef _WIN32
// diagnostic only (LLAMA_HYBRID_CRASH_TRACE=1): symbolised stack of an access violation
static LONG WINAPI hybrid_crash_filter(EXCEPTION_POINTERS * ep) {
    if (ep == nullptr || ep->ExceptionRecord == nullptr || ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    static std::atomic<bool> once{false};
    if (once.exchange(true)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(proc, nullptr, TRUE);
    const EXCEPTION_RECORD * er = ep->ExceptionRecord;
    fprintf(stderr, "CRASH-TRACE: access violation: %s of address %p at %p, thread %lu\n", er->ExceptionInformation[0] == 0 ? "read" : (er->ExceptionInformation[0] == 1 ? "write" : "execute"),
            (void *) er->ExceptionInformation[1], er->ExceptionAddress, (unsigned long) GetCurrentThreadId());
    void * frames[64];
    const USHORT n = CaptureStackBackTrace(0, 64, frames, nullptr);
    alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 512];
    for (USHORT i = 0; i < n; ++i) {
        SYMBOL_INFO * sym = (SYMBOL_INFO *) buf;
        memset(sym, 0, sizeof(SYMBOL_INFO));
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen   = 511;
        DWORD64 disp = 0;
        IMAGEHLP_LINE64 line;
        memset(&line, 0, sizeof(line));
        line.SizeOfStruct = sizeof(line);
        DWORD ldisp = 0;
        IMAGEHLP_MODULE64 mod;
        memset(&mod, 0, sizeof(mod));
        mod.SizeOfStruct = sizeof(mod);
        const DWORD64 a = (DWORD64) frames[i];
        const bool has_mod = SymGetModuleInfo64(proc, a, &mod) != FALSE;
        if (SymFromAddr(proc, a, &disp, sym)) {
            if (SymGetLineFromAddr64(proc, a, &ldisp, &line)) {
                fprintf(stderr, "CRASH-TRACE: #%02u %s!%s+0x%llx (%s:%lu)\n", (unsigned) i, has_mod ? mod.ModuleName : "?", sym->Name, (unsigned long long) disp, line.FileName, (unsigned long) line.LineNumber);
            } else {
                fprintf(stderr, "CRASH-TRACE: #%02u %s!%s+0x%llx\n", (unsigned) i, has_mod ? mod.ModuleName : "?", sym->Name, (unsigned long long) disp);
            }
        } else {
            fprintf(stderr, "CRASH-TRACE: #%02u %s+0x%llx\n", (unsigned) i, has_mod ? mod.ModuleName : "?", has_mod ? (unsigned long long) (a - mod.BaseOfImage) : (unsigned long long) a);
        }
    }
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

// ---- G31o: the prefill window (see g31o_prefill_window_apply.py.txt)
struct hybrid_prefill_state {
    bool     on          = false;
    bool     active      = false;   // the big ubatch is in force
    bool     failed      = false;   // the big reserve failed once: the window is off for the rest of the run
    uint32_t big         = 0;
    uint32_t ub_base     = 0;
    int32_t  min_tokens  = 1024;
    int32_t  max_slots   = 256;
    double   mib_per_row = 0.0;     // measured growth of the compute buffers per extra ubatch row
    int64_t  n_begin     = 0;
};
static hybrid_prefill_state g_hp;
static bool g_lm_regrow_pending = false;   // G32: the expert cache shrank for a layer-major pass and has not grown back yet

static bool hybrid_diag() {
    static const bool d = getenv("LLAMA_HYBRID_FREE_WATCH_MS") != nullptr;
    return d;
}

static int64_t hybrid_free_bytes() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(ggml_backend_dev_get(i), &free_b, &total_b);
            return (int64_t) free_b;
        }
    }
    return 0;
}

static void hybrid_prefill_regrow(llama_context * ctx) {
    g_lm_regrow_pending = false;
    int before = 0, after = 0;
    int64_t free_before = 0;
    llama_moe_hybrid_auto_resize_impl(g_hybrid_keep_free, g_hybrid_safety + g_hybrid_slack, g_hybrid_min_slots, g_hp.max_slots, &before, &after, &free_before);
    ctx->drop_graph_reuse();
    if (hybrid_diag()) {
        LLAMA_LOG_WARN("hybrid-moe: prefill window: expert cache %d -> %d slots (free %lld MiB before)\n", before, after, (long long) (free_before/1048576));
    }
}

static void hybrid_prefill_begin(llama_context * ctx) {
    const int64_t t0 = ggml_time_us();
    g_hp.ub_base = ctx->get_cparams().n_ubatch;
    if (g_hp.big <= g_hp.ub_base) {
        g_hp.failed = true;
        return;
    }
    const int64_t rows  = (int64_t) g_hp.big - (int64_t) g_hp.ub_base;
    const double  per   = g_hp.mib_per_row > 0.0 ? g_hp.mib_per_row*1.08 : 0.80;   // MiB per extra row: measured after the first window, a safe guess before
    const int64_t need  = (int64_t) (rows*per*1048576.0);
    const int64_t margin_b = g_hybrid_keep_free + g_hybrid_safety;
    int64_t n_before = 0;
    int     slots_before = 0;
    llama_moe_hybrid_cache_info_impl(&n_before, &slots_before);
    bool shrunk = false;
    llama_moe_hybrid_make_room_impl(need + margin_b, 0, g_hybrid_min_slots, &shrunk);
    if (hybrid_free_bytes() < need + margin_b) {
        // even the smallest cache cannot make room: this prompt runs on the small ubatch
        LLAMA_LOG_WARN("hybrid-moe: prefill window skipped: %.0f MiB would be needed for -ub %u and the margin does not allow it\n", (double) need/1048576.0, g_hp.big);
        if (shrunk) {
            hybrid_prefill_regrow(ctx);
        }
        return;
    }
    int64_t freed = 0;
    const int64_t dev_free_before = hybrid_free_bytes();
    try {
        freed = ctx->reserve_ubatch(g_hp.big);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("hybrid-moe: prefill window: reserving -ub %u failed (%s); the window is off\n", g_hp.big, e.what());
        g_hp.failed = true;
        try {
            ctx->reserve_ubatch(g_hp.ub_base);
        } catch (const std::exception & e2) {
            LLAMA_LOG_ERROR("hybrid-moe: prefill window: re-reserving -ub %u failed (%s)\n", g_hp.ub_base, e2.what());
        }
        hybrid_prefill_regrow(ctx);
        return;
    }
    GGML_UNUSED(freed);   // the sum over all backends (the host's compute buffer too): not what the device pays
    const int64_t growth = dev_free_before - hybrid_free_bytes();
    if (growth > 0) {
        g_hp.mib_per_row = (double) growth/1048576.0/(double) rows;
    }
    // the estimate may have been short: keep the margin
    bool shrunk2 = false;
    llama_moe_hybrid_make_room_impl(margin_b, 0, g_hybrid_min_slots, &shrunk2);
    ctx->drop_graph_reuse();
    g_hp.active = true;
    g_hp.n_begin++;
    if (hybrid_diag()) {
        int64_t n_after = 0;
        int     slots_after = 0;
        llama_moe_hybrid_cache_info_impl(&n_after, &slots_after);
        LLAMA_LOG_WARN("hybrid-moe: prefill window #%lld begin: -ub %u -> %u, compute buffers %+.1f MiB (%.3f MiB per row), expert cache %d -> %d slots, free %lld MiB, %.1f ms\n",
                       (long long) g_hp.n_begin, g_hp.ub_base, g_hp.big, (double) growth/1048576.0, g_hp.mib_per_row, slots_before, slots_after, (long long) (hybrid_free_bytes()/1048576), (ggml_time_us() - t0)/1000.0);
    }
}

static void hybrid_prefill_end(llama_context * ctx, bool regrow = true) {
    const int64_t t0 = ggml_time_us();
    g_hp.active = false;
    try {
        ctx->reserve_ubatch(g_hp.ub_base);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("hybrid-moe: prefill window: re-reserving -ub %u failed (%s)\n", g_hp.ub_base, e.what());
        g_hp.failed = true;
    }
    if (regrow) {
        hybrid_prefill_regrow(ctx);
    } else {
        g_lm_regrow_pending = true;   // the cache grows back at the first small decode
    }
    if (hybrid_diag()) {
        LLAMA_LOG_WARN("hybrid-moe: prefill window end: -ub %u, %.1f ms\n", g_hp.ub_base, (ggml_time_us() - t0)/1000.0);
    }
}

// ---- G31s: the routing of the prompt's last tokens seeds the expert cache's popularity
struct hybrid_seed_state {
    bool                  on         = false;
    int                   K          = 32;
    int                   n_layer    = 0;
    int                   n_used     = 0;
    int                   filled     = 0;      // tokens captured by the last probed decode (<= K)
    bool                  valid      = false;  // a capture is waiting for the next small decode
    int                   n_layers_seen = 0;
    std::vector<int32_t>  ids;                 // [K][n_layer][n_used]
    std::vector<uint8_t>  tmp;
    ggml_backend_sched_eval_callback orig_cb = nullptr;
    void *                orig_ud    = nullptr;
};
static hybrid_seed_state g_seed;

static bool hybrid_probe_cb(ggml_tensor * t, bool ask, void * ud) {
    GGML_UNUSED(ud);
    if (strncmp(t->name, "ffn_moe_topk-", 13) != 0) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    const int il = atoi(t->name + 13);
    if (il < 0 || il >= g_seed.n_layer || t->type != GGML_TYPE_I32 || t->ne[0] != g_seed.n_used || t->ne[1] < 1) {
        return true;
    }
    const int64_t n_tok = t->ne[1];
    const int     Kc    = (int) std::min<int64_t>(g_seed.K, n_tok);
    const size_t  bytes = (size_t) ((n_tok - 1)*t->nb[1] + t->ne[0]*t->nb[0]);
    g_seed.tmp.resize(bytes);
    ggml_backend_tensor_get(t, g_seed.tmp.data(), 0, bytes);
    if (il == 0) {
        std::fill(g_seed.ids.begin(), g_seed.ids.end(), -1);
        g_seed.filled        = Kc;
        g_seed.n_layers_seen = 0;
    }
    if (Kc != g_seed.filled) {
        return true;
    }
    for (int j = 0; j < Kc; ++j) {
        const int64_t row = n_tok - Kc + j;
        for (int k = 0; k < g_seed.n_used; ++k) {
            g_seed.ids[((size_t) j*g_seed.n_layer + il)*g_seed.n_used + k] = *(const int32_t *) (g_seed.tmp.data() + row*t->nb[1] + k*t->nb[0]);
        }
    }
    if (++g_seed.n_layers_seen >= g_seed.n_layer) {
        g_seed.valid = true;
    }
    return true;
}

static void hybrid_decode_hook(llama_context * ctx, int32_t n_tokens) {
    if (ctx != g_hybrid_ctx) {
        return;
    }
    // a capture from the previous (prompt-like) decode folds into the statistics before the window closes and the cache is refilled, and before the first generated token
    if (g_seed.on && g_seed.valid) {
        if (n_tokens <= 16 && g_seed.filled > 0) {
            llama_moe_hybrid_seed_impl(g_seed.ids.data(), g_seed.filled, nullptr, nullptr);
        }
        g_seed.valid = false;
    }
    if (g_hp.on && !g_hp.failed) {
        if (!g_hp.active && n_tokens >= g_hp.min_tokens && !llama_lm_prefill_pending) {
            hybrid_prefill_begin(ctx);
        } else if (g_hp.active && n_tokens <= 16) {
            // the big ubatch stays for the chunks that follow the big batches (the 516-token checkpoint chunk); a decode step, a speculative step or the 4-token chunk closes the window
            hybrid_prefill_end(ctx);
        }
    }
    if (g_lm_regrow_pending && !llama_lm_prefill_pending && n_tokens <= 16) {
        hybrid_prefill_regrow(ctx);
    }
    if (g_seed.on) {
        // probe the prompt-like decodes below the window's size (the checkpoint chunk): the callback reads the router's top-k of the last K tokens; re-applied every call because a re-reserve replaces the scheduler
        const int32_t hi = g_hp.on ? g_hp.min_tokens : 1024;
        const bool    want = n_tokens >= 17 && n_tokens < hi;
        ctx->set_eval_cb(want ? hybrid_probe_cb : g_seed.orig_cb, want ? (void *) &g_seed : g_seed.orig_ud);
    }
}

static std::thread       g_watch;
static std::atomic<bool> g_watch_started{false};
static std::atomic<bool> g_watch_stop{false};

// measurement only: the runtime's free device memory while the engine exists (see g31e_free_watch_apply.py.txt)
static void hybrid_free_watch(int period_ms) {
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev = ggml_backend_dev_get(i);
            break;
        }
    }
    if (dev == nullptr) {
        LLAMA_LOG_WARN("hybrid-moe: free-watch: no GPU device found\n");
        return;
    }
    LLAMA_LOG_WARN("hybrid-moe: free-watch: started, period %d ms\n", period_ms);
    int64_t lowest = INT64_MAX, logged = INT64_MAX;
    int64_t cache_bytes = 0;
    int     n_slot = 0;
    while (!g_watch_stop.load()) {
        if (!llama_moe_hybrid_cache_info_impl(&cache_bytes, &n_slot)) {
            n_slot = 0;    // the cache is being rebuilt (or is gone): sample the memory anyway
        }
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        lowest = std::min<int64_t>(lowest, (int64_t) free_b);
        if (lowest < logged - 64*1048576) {
            logged = lowest;
            LLAMA_LOG_WARN("hybrid-moe: free-watch: new minimum of free device memory %lld MiB (expert cache %d slots)\n", (long long) (lowest/1048576), n_slot);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
    }
    LLAMA_LOG_WARN("hybrid-moe: free-watch: the engine is gone; lowest free device memory %lld MiB\n", (long long) (lowest/1048576));
}

// G32: the layer-major prefill needs the slot window, the slab and the pass's compute buffer: the expert cache gives the room (begin) and takes it back (end)
static bool hybrid_lm_vram(llama_context * ctx, int64_t need_bytes, int begin) {
    if (ctx != g_hybrid_ctx) {
        return true;
    }
    if (begin) {
        if (g_hp.active) {
            // a window opened for an earlier batch of this request (one the pass could not take) holds ~0.8 GiB of compute buffers: close it, the pass takes this batch
            hybrid_prefill_end(ctx, false);
        }
        const int64_t margin_b = g_hybrid_keep_free + g_hybrid_safety;
        const int64_t target_b = need_bytes + margin_b;
        // the pass blocks the server loop, so no decode runs while it holds the room: the cache may drop below the decode floor for its duration (it regrows at the first small decode)
        const int     pass_floor = std::max(2, g_hybrid_min_slots/4);
        bool shrunk = false;
        int64_t cache_b0 = 0, cache_b1 = 0;
        int     slots0 = 0, slots1 = 0;
        llama_moe_hybrid_cache_info_impl(&cache_b0, &slots0);
        const int64_t free0 = hybrid_free_bytes();
        bool room_ok = true;
        if (free0 < target_b) {
            // make_room estimates the freed bytes from the mean row size, so a shrink can land a MiB short of the target: aim above it, the check below stays at the target
            room_ok = llama_moe_hybrid_make_room_impl(target_b + (int64_t) 64*1048576, 0, pass_floor, &shrunk);
            if (hybrid_free_bytes() < target_b) {
                bool shrunk2 = false;
                room_ok = llama_moe_hybrid_make_room_impl(target_b + (int64_t) 256*1048576, 0, pass_floor, &shrunk2) && room_ok;
                shrunk = shrunk || shrunk2;
            }
        }
        if (shrunk) {
            ctx->drop_graph_reuse();
        }
        if (hybrid_free_bytes() < target_b) {
            llama_moe_hybrid_cache_info_impl(&cache_b1, &slots1);
            LLAMA_LOG_WARN("hybrid-moe: lm-prefill: no room for %.0f MiB + margin %.0f MiB: free %lld -> %lld MiB, expert cache %d -> %d slots (min %d), make_room %s, shrunk %d\n",
                           (double) need_bytes/1048576.0, (double) margin_b/1048576.0, (long long) (free0/1048576), (long long) (hybrid_free_bytes()/1048576), slots0, slots1, g_hybrid_min_slots,
                           room_ok ? "true" : "false", shrunk ? 1 : 0);
            if (shrunk) {
                hybrid_prefill_regrow(ctx);
            }
            return false;
        }
        if (hybrid_diag()) {
            LLAMA_LOG_WARN("hybrid-moe: lm-prefill: room for %.0f MiB made (%s), free %lld MiB\n", (double) need_bytes/1048576.0, shrunk ? "the expert cache shrank" : "no shrink needed", (long long) (hybrid_free_bytes()/1048576));
        }
        return true;
    }
    // the cache grows back at the first small decode (the checkpoint chunk of the server is not one; the first token of the answer is): consecutive passes of one request, and the chunks between them, share one shrink
    g_lm_regrow_pending = true;
    return true;
}

static void hybrid_kv_pressure(llama_context * lctx, int64_t commit_bytes) {
    GGML_UNUSED(lctx);   // not trusted: the registered context is the one whose graphs hold the old cache tensors
    bool shrunk = false;
    if (!llama_moe_hybrid_make_room_impl(commit_bytes + g_hybrid_keep_free + g_hybrid_safety, g_hybrid_slack, g_hybrid_min_slots, &shrunk)) {
        LLAMA_LOG_WARN("%s: hybrid-moe: could not make room for a KV commit of %.1f MiB\n", __func__, (double) commit_bytes/1048576.0);
        return;
    }
    if (shrunk && g_hybrid_ctx != nullptr) {
        g_hybrid_ctx->drop_graph_reuse();
    }
}
#endif

void llama_hybrid_set_margin(struct llama_context * ctx, int32_t keep_free_mib, int32_t safety_mib, int32_t slack_mib, int32_t min_slots) {
#ifdef LLAMA_LAYER_STREAMING
    const int64_t MiB = 1048576;
    g_hybrid_keep_free = (int64_t) keep_free_mib*MiB;
    g_hybrid_safety    = (int64_t) safety_mib*MiB;
    g_hybrid_slack     = (int64_t) slack_mib*MiB;
    g_hybrid_min_slots = min_slots > 0 ? min_slots : 8;
    g_hybrid_ctx       = ctx;
    llama_moe_hybrid_set_keep_stats(true);
    llama_kv_lazy_pressure_cb = hybrid_kv_pressure;
    llama_lm_vram_hook        = hybrid_lm_vram;
#ifdef _WIN32
    if (const char * c = getenv("LLAMA_HYBRID_CRASH_TRACE"); c != nullptr && atoi(c) > 0) {
        static std::atomic<bool> installed{false};
        if (!installed.exchange(true)) {
            AddVectoredExceptionHandler(1, hybrid_crash_filter);
        }
    }
#endif
    if (const char * w = getenv("LLAMA_HYBRID_FREE_WATCH_MS"); w != nullptr && atoi(w) > 0 && !g_watch_started.exchange(true)) {
        g_watch_stop = false;
        g_watch = std::thread(hybrid_free_watch, atoi(w));
    }
#else
    GGML_UNUSED(ctx); GGML_UNUSED(keep_free_mib); GGML_UNUSED(safety_mib); GGML_UNUSED(slack_mib); GGML_UNUSED(min_slots);
#endif
}

void llama_hybrid_set_prefill(struct llama_context * ctx, int32_t ubatch_big, int32_t min_tokens, int32_t max_slots) {
#ifdef LLAMA_LAYER_STREAMING
    g_hp = hybrid_prefill_state();
    if (ubatch_big <= 0 || ctx == nullptr || ctx != g_hybrid_ctx) {
        return;
    }
    g_hp.on         = true;
    g_hp.big        = (uint32_t) ubatch_big;
    g_hp.min_tokens = min_tokens > 0 ? min_tokens : 1024;
    g_hp.max_slots  = max_slots > 0 ? max_slots : 256;
    llama_hybrid_decode_hook = hybrid_decode_hook;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(ubatch_big); GGML_UNUSED(min_tokens); GGML_UNUSED(max_slots);
#endif
}

void llama_hybrid_set_seed(struct llama_context * ctx, int32_t n_tokens) {
#ifdef LLAMA_LAYER_STREAMING
    g_seed = hybrid_seed_state();
    if (n_tokens <= 0 || ctx == nullptr || ctx != g_hybrid_ctx) {
        return;
    }
    int n_layer = 0, n_used = 0;
    if (!llama_moe_hybrid_seed_impl(nullptr, 0, &n_layer, &n_used)) {
        return;
    }
    g_seed.on      = true;
    g_seed.K       = n_tokens;
    g_seed.n_layer = n_layer;
    g_seed.n_used  = n_used;
    g_seed.ids.assign((size_t) n_tokens*n_layer*n_used, -1);
    g_seed.orig_cb = ctx->get_cparams().cb_eval;
    g_seed.orig_ud = ctx->get_cparams().cb_eval_user_data;
    llama_hybrid_decode_hook = hybrid_decode_hook;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(n_tokens);
#endif
}

void llama_hybrid_shutdown(void) {
#ifdef LLAMA_LAYER_STREAMING
    llama_hybrid_decode_hook = nullptr;
    g_seed = hybrid_seed_state();
    llama_moe_hybrid_set_keep_stats(false);
    g_hp = hybrid_prefill_state();
    g_watch_stop = true;
    if (g_watch.joinable()) {
        g_watch.join();
    }
    g_watch_started = false;
    llama_kv_lazy_pressure_cb = nullptr;
    llama_lm_vram_hook = nullptr;
    g_hybrid_ctx = nullptr;
#endif
}

bool llama_hybrid_active(void) {
#ifdef LLAMA_LAYER_STREAMING
    int64_t cache_bytes = 0;
    int     n_slot      = 0;
    return llama_moe_hybrid_cache_info_impl(&cache_bytes, &n_slot);
#else
    return false;
#endif
}

bool llama_hybrid_auto_size(struct llama_context * ctx, int32_t keep_free_mib, int32_t safety_mib, int32_t min_slots, int32_t max_slots,
                            int32_t * slots_before, int32_t * slots_after, int32_t * free_before_mib) {
#ifdef LLAMA_LAYER_STREAMING
    const int64_t MiB = 1048576;
    int     before = 0, after = 0;
    int64_t free_before = 0;
    if (!llama_moe_hybrid_auto_resize_impl((int64_t) keep_free_mib*MiB, (int64_t) safety_mib*MiB, (int) min_slots, (int) max_slots, &before, &after, &free_before)) {
        return false;
    }
    // G32s: the count the cache was sized with is also the ceiling of every later regrow (after a layer-major pass, after a prefill window); a fixed --hybrid-moe-slots N must stay N
    g_hp.max_slots = max_slots > 0 ? max_slots : 256;
    if (slots_before != nullptr) {
        *slots_before = before;
    }
    if (slots_after != nullptr) {
        *slots_after = after;
    }
    if (free_before_mib != nullptr) {
        *free_before_mib = (int32_t) (free_before / MiB);
    }
    if (ctx != nullptr && after != before) {
        ctx->drop_graph_reuse();
    }
    return true;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(keep_free_mib); GGML_UNUSED(safety_mib); GGML_UNUSED(min_slots); GGML_UNUSED(max_slots);
    GGML_UNUSED(slots_before); GGML_UNUSED(slots_after); GGML_UNUSED(free_before_mib);
    return false;
#endif
}
