#include "common.cuh"
#include "moe-hybrid.cuh"

// T-Q36-AER experimental hybrid MoE decode. publish maps the router's selected
// experts to VRAM cache slots and posts the misses to a host mailbox; wait_add
// spins until the host engine has written the missed experts' weighted sum and
// adds it. Both kernels use plain launches (no PDL): the host handshake needs
// full stream order. A lost or failed host reply traps instead of continuing
// with a partial sum.

static constexpr uint64_t MOE_HYBRID_TIMEOUT_NS = 500ull * 1000ull * 1000ull;

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
        ggml_moe_hybrid_mailbox * mb,
        int32_t       * trace,
        const int il,
        const int n_used,
        const int n_expert,
        const int n_slot,
        const int n_embd,
        const int vec) {
    __shared__ int s_nmiss;

    const int tid = threadIdx.x;

    if (tid < WARP_SIZE) {
        int  e    = 0;
        bool miss = false;
        if (tid < n_used) {
            e = selected[tid];
            if (e < 0 || e >= n_expert) {
                printf("moe_hybrid_publish: layer %d bad expert id %d\n", il, e);
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
        ggml_moe_hybrid_mailbox * mb,
        float         * dst,
        const int il,
        const int n_embd,
        const int vec) {
    __shared__ int s_ok;

    const int tid   = threadIdx.x;
    const int nmiss = pub[8];

    if (nmiss == 0) {
        if (tid == 0) {
            const uint64_t t = moe_hybrid_globaltimer();
            mb->prof[il][1] = t;
            mb->prof[il][2] = t;
            mb->prof[il][4] = t;
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
            if (moe_hybrid_globaltimer() - t0 > MOE_HYBRID_TIMEOUT_NS) {
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
            printf("moe_hybrid_wait_add: seq %d %s (res_seq %d, err %d)\n", seq,
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

static ggml_moe_hybrid_mailbox * moe_hybrid_mailbox(const ggml_tensor * pub) {
    uint64_t p;
    memcpy(&p, (const int32_t *) pub->op_params + 4, sizeof(p));
    return (ggml_moe_hybrid_mailbox *) (uintptr_t) p;
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

    GGML_ASSERT(selected->nb[0] == sizeof(int32_t));
    GGML_ASSERT(ggml_is_contiguous(weights) && ggml_is_contiguous(cur) && ggml_is_contiguous(map));
    GGML_ASSERT(selected->ne[0] <= GGML_MOE_HYBRID_MAX_USED && cur->ne[0] <= GGML_MOE_HYBRID_MAX_EMBD);

    const int32_t * params = (const int32_t *) dst->op_params;
    GGML_ASSERT(params[0] >= 0 && params[0] < GGML_MOE_HYBRID_MAX_LAYER);
    uint64_t trace;
    memcpy(&trace, params + 6, sizeof(trace));

    ggml_moe_hybrid_mailbox * mb = moe_hybrid_mailbox(dst);
    moe_hybrid_publish_f32<<<1, 256, 0, ctx.stream()>>>(
        (const int32_t *) selected->data, (const float *) weights->data, (const float *) cur->data,
        (const int32_t *) map->data, (int32_t *) state->data, (int32_t *) dst->data,
        mb, (int32_t *) (uintptr_t) trace,
        params[0], (int) selected->ne[0], params[2], params[1], (int) cur->ne[0],
        moe_hybrid_vec_ok(cur->ne[0], cur->data, mb->x));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_moe_hybrid_wait_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur = dst->src[0];
    const ggml_tensor * pub = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(cur) && cur->ne[0] <= GGML_MOE_HYBRID_MAX_EMBD);

    const int il = ((const int32_t *) pub->op_params)[0];
    GGML_ASSERT(il >= 0 && il < GGML_MOE_HYBRID_MAX_LAYER);

    ggml_moe_hybrid_mailbox * mb = moe_hybrid_mailbox(pub);
    moe_hybrid_wait_add_f32<<<1, 1024, 0, ctx.stream()>>>(
        (const float *) cur->data, (const int32_t *) pub->data, mb,
        (float *) dst->data, il, (int) cur->ne[0],
        moe_hybrid_vec_ok(cur->ne[0], cur->data, dst->data) && moe_hybrid_vec_ok(cur->ne[0], mb->y, mb->y));
    CUDA_CHECK(cudaGetLastError());
}
