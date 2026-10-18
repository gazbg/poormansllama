#include "common.cuh"
#include "moe-hybrid.cuh"

// T-Q36-AER experimental hybrid MoE decode. publish maps the router's selected
// experts to VRAM cache slots and posts the misses to a host mailbox; wait_add
// spins until the host engine has written the missed experts' weighted sum and
// adds it. Both kernels use plain launches (no PDL): the host handshake needs
// full stream order. A lost or failed host reply traps instead of continuing
// with a partial sum.
//
// G12: 1 .. GGML_MOE_HYBRID_MAX_WIDTH tokens per launch (an MTP verify step decodes the frontier plus its drafts in one ubatch). Block t of the
// grid owns token t: its own selected / weights / cur / pub rows, its own mailbox (the host engine serves them one after the other) and its own
// device request counter state[t]. The blocks of one launch run concurrently, so all the requests of a layer are in flight together.

// default device-side wait limit: 5 s. A 500 ms limit turns a long host stall into a crash (g7-timeout: an injected 1.2 s stall); g7-timeout also proved a resident
// wait_add survives a 4 s host stall under WDDM (no TDR) and the run completes with identical text. The one 500 ms+ event seen in the field was NOT a stall: it was the
// first-KV-pad-boundary deadlock (engine thread blocked in a CUDA call while the launching thread held the driver lock), removed by the dedicated copy thread
// (docs/plans/tg-beat-stock-8gb-262k.md, G5). LLAMA_MOE_HYBRID_TIMEOUT_MS overrides.
static constexpr uint64_t MOE_HYBRID_TIMEOUT_NS = 5000ull * 1000ull * 1000ull;

// LLAMA_MOE_HYBRID_TIMEOUT_MS, set by the host engine at init (0 = MOE_HYBRID_TIMEOUT_NS); the launch passes it by value
static int g_moe_hybrid_timeout_ms = 0;

// exported without touching ggml-cuda.h (every CUDA translation unit includes it); the declaration lives in src/llama-moe-hybrid.cpp
extern "C" __declspec(dllexport) void ggml_backend_cuda_moe_hybrid_set_timeout_ms(int ms) {
    g_moe_hybrid_timeout_ms = ms > 0 ? ms : 0;
}

// LLAMA_MOE_HYBRID_XDUMP (study tool, G13): a device-mapped pinned ring [token][layer][n_embd] that every publish kernel of a one-token graph fills with the layer's FFN input
// (the vector the router sees), indexed by the token counter the host writes into state[12] before each token. Null = off (the kernels skip it).
static float * g_moe_hybrid_xd      = nullptr;
static int     g_moe_hybrid_xd_ntok = 0;

extern "C" __declspec(dllexport) void ggml_backend_cuda_moe_hybrid_set_xdump(void * dev_ptr, int n_tok) {
    g_moe_hybrid_xd      = (float *) dev_ptr;
    g_moe_hybrid_xd_ntok = n_tok > 0 ? n_tok : 0;
}

static __device__ __forceinline__ uint64_t moe_hybrid_globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __global__ void moe_hybrid_publish_f32(
        const int32_t * selected,
        const float   * weights,
        const float   * cur,
        const int32_t * map,
        int32_t       * state,
        int32_t       * dst,
        char          * mb_base,
        int32_t       * trace,
        const int il,
        const int n_used,
        const int n_expert,
        const int n_slot,
        const int n_embd,
        const int vec,
        const int sel_stride,     // int32 elements between two tokens' selected rows
        const int cur_stride,     // float elements between two tokens' cur rows
        const int mb_stride,      // bytes between two mailboxes
        const int trace_stride,   // int32 elements between two tokens' trace blocks
        float *   xd,             // optional capture ring (see g_moe_hybrid_xd), null = off
        const int xd_ntok,
        const int xd_layers) {
    __shared__ int s_nmiss;

    const int tid = threadIdx.x;

    // block t = token t
    const int t = blockIdx.x;
    ggml_moe_hybrid_mailbox * mb = (ggml_moe_hybrid_mailbox *) (mb_base + (size_t) t*mb_stride);
    selected += (size_t) t*sel_stride;
    weights  += (size_t) t*n_used;
    cur      += (size_t) t*cur_stride;
    state    += t;
    dst      += (size_t) t*16;
    trace    += (size_t) t*trace_stride;

    if (xd != nullptr && t == 0) {
        const int tokidx = state[12];
        if (tokidx >= 0 && tokidx < xd_ntok) {
            float * dst_x = xd + ((size_t) tokidx*xd_layers + il)*n_embd;
            for (int i = tid; i < n_embd; i += blockDim.x) {
                dst_x[i] = cur[i];
            }
        }
    }

    if (tid < WARP_SIZE) {
        int  e    = 0;
        bool miss = false;
        if (tid < n_used) {
            e = selected[tid];
            if (e < 0 || e >= n_expert) {
                printf("moe_hybrid_publish: layer %d token %d bad expert id %d\n", il, t, e);
                __trap();
            }
            int slot = map[il*n_expert + e];
            miss = slot < 0 || slot >= n_slot;
            if (miss) {
                slot = n_slot; // zero slot: the cached sum gets 0 for this expert
            }
            dst[tid] = slot;
            trace[il*GGML_MOE_HYBRID_MAX_USED + tid] = e;
        }
        const unsigned mask = __ballot_sync(0xffffffff, miss);
        if (miss) {
            const int pos = __popc(mask & ((1u << tid) - 1u));
            mb->ids[pos] = e;
            mb->w[pos]   = weights[tid];
        }
        if (tid == 0) {
            s_nmiss = __popc(mask);
        }
    }
    __syncthreads();

    const int nmiss = s_nmiss; // block-uniform
    int32_t   seq   = 0;
    if (nmiss > 0) {
        if (vec) {
            const float4 * src = (const float4 *) cur;
            float4 *       dx  = (float4 *) mb->x;
            for (int i = tid; i < n_embd/4; i += blockDim.x) {
                dx[i] = src[i];
            }
        } else {
            for (int i = tid; i < n_embd; i += blockDim.x) {
                mb->x[i] = cur[i];
            }
        }
        // every thread orders its own writes before the barrier; the request flag follows the barrier
        __threadfence_system();
        __syncthreads();
    }

    if (tid == 0) {
        seq = state[0];
        if (nmiss > 0) {
            seq++;
            state[0]  = seq;
            mb->layer = il;
            mb->nmiss = nmiss;
            __threadfence_system();
            mb->req_seq = seq;
        }
        dst[8] = nmiss;
        dst[9] = seq;
        mb->prof[il][0] = moe_hybrid_globaltimer();
    }
}

static __global__ void moe_hybrid_wait_add_f32(
        const float   * cur,
        const int32_t * pub,
        char          * mb_base,
        float         * dst,
        const int il,
        const int n_embd,
        const int vec,
        const uint64_t timeout_ns,
        const int cur_stride,     // float elements between two tokens' rows (cur and dst share the contiguous layout)
        const int mb_stride) {    // bytes between two mailboxes
    __shared__ int s_ok;

    const int tid   = threadIdx.x;

    // block t = token t
    const int t = blockIdx.x;
    ggml_moe_hybrid_mailbox * mb = (ggml_moe_hybrid_mailbox *) (mb_base + (size_t) t*mb_stride);
    pub += (size_t) t*16;
    cur += (size_t) t*cur_stride;
    dst += (size_t) t*cur_stride;

    const int nmiss = pub[8];

    if (nmiss == 0) {
        if (tid == 0) {
            const uint64_t tt = moe_hybrid_globaltimer();
            mb->prof[il][1] = tt;
            mb->prof[il][2] = tt;
            mb->prof[il][4] = tt;
        }
        for (int i = tid; i < n_embd; i += blockDim.x) {
            dst[i] = cur[i];
        }
        return;
    }

    if (tid == 0) {
        const int32_t  seq = pub[9];
        const uint64_t t0  = moe_hybrid_globaltimer();
        int ok = 1;
        while (mb->res_seq != seq) {
            if (moe_hybrid_globaltimer() - t0 > timeout_ns) {
                ok = 0;
                break;
            }
        }
        mb->prof[il][1] = t0;
        mb->prof[il][2] = moe_hybrid_globaltimer();
        __threadfence_system();
        if (ok && *((volatile int32_t *) &mb->err) != 0) {
            ok = -1;
        }
        if (ok != 1) {
            printf("moe_hybrid_wait_add: token %d seq %d %s (res_seq %d, err %d)\n", t, seq,
                ok == 0 ? "timed out" : "host error", mb->res_seq, *((volatile int32_t *) &mb->err));
        }
        s_ok = ok;
    }
    __syncthreads();

    if (s_ok != 1) {
        __trap();
    }

    // one PCIe round trip per thread: the host result is read as float4 when the pointers allow it
    if (vec) {
        const float4 * c4 = (const float4 *) cur;
        const float4 * y4 = (const float4 *) mb->y;
        float4 *       d4 = (float4 *) dst;
        for (int i = tid; i < n_embd/4; i += blockDim.x) {
            const float4 c = c4[i];
            const float4 y = __ldcv(&y4[i]);
            d4[i] = make_float4(c.x + y.x, c.y + y.y, c.z + y.z, c.w + y.w);
        }
    } else {
        for (int i = tid; i < n_embd; i += blockDim.x) {
            dst[i] = cur[i] + __ldcv(&mb->y[i]);
        }
    }
    __syncthreads();
    if (tid == 0) {
        mb->prof[il][4] = moe_hybrid_globaltimer();
    }
}

static char * moe_hybrid_mailbox(const ggml_tensor * pub) {
    uint64_t p;
    memcpy(&p, (const int32_t *) pub->op_params + 4, sizeof(p));
    return (char *) (uintptr_t) p;
}

// float4 access needs a multiple-of-4 row and 16-byte aligned device buffers (the mailbox is 256-aligned)
static int moe_hybrid_vec_ok(int64_t n_embd, const void * a, const void * b) {
    return n_embd % 4 == 0 && (((uintptr_t) a | (uintptr_t) b) & 15) == 0;
}

void ggml_cuda_op_moe_hybrid_publish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * selected = dst->src[0];
    const ggml_tensor * weights  = dst->src[1];
    const ggml_tensor * cur      = dst->src[2];
    const ggml_tensor * map      = dst->src[3];
    const ggml_tensor * state    = dst->src[4];

    GGML_ASSERT(selected->nb[0] == sizeof(int32_t) && selected->nb[1] % sizeof(int32_t) == 0);
    GGML_ASSERT(ggml_is_contiguous(weights) && ggml_is_contiguous(cur) && ggml_is_contiguous(map));
    GGML_ASSERT(selected->ne[0] <= GGML_MOE_HYBRID_MAX_USED && cur->ne[0] <= GGML_MOE_HYBRID_MAX_EMBD);

    const int n_tok = (int) selected->ne[1];
    GGML_ASSERT(n_tok >= 1 && n_tok <= GGML_MOE_HYBRID_MAX_WIDTH && cur->ne[1] == n_tok && dst->ne[1] == n_tok);

    const int32_t * params = (const int32_t *) dst->op_params;
    GGML_ASSERT(params[0] >= 0 && params[0] < GGML_MOE_HYBRID_MAX_LAYER);
    uint64_t trace;
    memcpy(&trace, params + 6, sizeof(trace));
    const int trace_stride = params[3];
    const int mb_stride    = params[8];
    GGML_ASSERT(trace_stride >= GGML_MOE_HYBRID_MAX_USED && mb_stride >= (int) sizeof(ggml_moe_hybrid_mailbox) && mb_stride % 256 == 0);

    char * mb = moe_hybrid_mailbox(dst);
    moe_hybrid_publish_f32<<<n_tok, 256, 0, ctx.stream()>>>(
        (const int32_t *) selected->data, (const float *) weights->data, (const float *) cur->data,
        (const int32_t *) map->data, (int32_t *) state->data, (int32_t *) dst->data,
        mb, (int32_t *) (uintptr_t) trace,
        params[0], (int) selected->ne[0], params[2], params[1], (int) cur->ne[0],
        moe_hybrid_vec_ok(cur->ne[0], cur->data, ((ggml_moe_hybrid_mailbox *) mb)->x),
        (int) (selected->nb[1]/sizeof(int32_t)), (int) (cur->nb[1]/sizeof(float)), mb_stride, trace_stride,
        n_tok == 1 ? g_moe_hybrid_xd : nullptr, g_moe_hybrid_xd_ntok, trace_stride/GGML_MOE_HYBRID_MAX_USED);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_moe_hybrid_wait_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur = dst->src[0];
    const ggml_tensor * pub = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(cur) && cur->ne[0] <= GGML_MOE_HYBRID_MAX_EMBD);

    const int n_tok = (int) cur->ne[1];
    GGML_ASSERT(n_tok >= 1 && n_tok <= GGML_MOE_HYBRID_MAX_WIDTH && pub->ne[1] == n_tok);

    const int il = ((const int32_t *) pub->op_params)[0];
    GGML_ASSERT(il >= 0 && il < GGML_MOE_HYBRID_MAX_LAYER);
    const int mb_stride = ((const int32_t *) pub->op_params)[8];

    char * mb = moe_hybrid_mailbox(pub);
    moe_hybrid_wait_add_f32<<<n_tok, 1024, 0, ctx.stream()>>>(
        (const float *) cur->data, (const int32_t *) pub->data, mb,
        (float *) dst->data, il, (int) cur->ne[0],
        moe_hybrid_vec_ok(cur->ne[0], cur->data, dst->data) && moe_hybrid_vec_ok(cur->ne[0], ((ggml_moe_hybrid_mailbox *) mb)->y, ((ggml_moe_hybrid_mailbox *) mb)->y),
        g_moe_hybrid_timeout_ms > 0 ? (uint64_t) g_moe_hybrid_timeout_ms * 1000000ull : MOE_HYBRID_TIMEOUT_NS,
        (int) (cur->nb[1]/sizeof(float)), mb_stride);
    CUDA_CHECK(cudaGetLastError());
}
